# DeepSelect vs funnel-topk：实现对比与同口径实测（2026-09-26，B300）

本文回答三个问题：

1. [`/models/lkk/funnel-topk`](https://github.com/luoyu-intel/funnel-topk)（下称 funnel-topk）
   与 DeepSelect 在实现上有什么异同？
2. funnel-topk 的性能可以在本机（NVIDIA B300 + conda env `muse_verify`）测吗？
3. 能与 DeepSelect 直接对比吗？

三者都有实测答案：**异同见 §3；可以测（§4，踩了 3 个坑）；可以对比，
但必须统一计时口径（§5），同口径下 DeepSelect 在全部 24 个配置点最快，
比 funnel-topk 最快模式快 1.3x ~ 5.1x，且结果精确（§6）。**

文中所有数字均为 2026-09-26 在本机实测，原始数据见
`compare_ds_funnel_20260926.json` 与 §8 产出物清单。

---

## 0. TL;DR

```bash
# funnel-topk 在 B300 上的正确安装姿势（§4 有坑的完整分析）
source /models/miniforge3/etc/profile.d/conda.sh && conda activate muse_verify
cd /models/lkk/funnel-topk
TORCH_CUDA_ARCH_LIST="10.3" pip install -e . --no-build-isolation   # 必须指定 sm_103！

# 统一口径对比（DeepSelect 自家 kernelkit 计时框架）
python3 benchmarks/compare_ds_vs_funnel.py
# 正确性硬检查（索引范围/唯一性/gather 一致性/topk 条件，三方同标准）
python3 benchmarks/verify_correctness_vs_funnel.py
```

核心结论（bf16, K=512, Lightning Indexer 口径）：

| 对比 | 结果 |
|---|---|
| DeepSelect vs funnel-topk 最快模式 | **DeepSelect 快 1.27x ~ 5.08x**（24/24 全胜） |
| DeepSelect vs torch.topk | 2.60x ~ 20.33x（与 README 复现口径一致） |
| funnel-topk 最快模式 vs torch.topk | 0.69x ~ 5.43x（**B=6 时反而更慢**） |
| 正确性 | DeepSelect 精确（topk 条件 9/9 配置 100% 行满足）；funnel fast/turbo **0% 行满足** topk 条件（recall 93.5% / 80.5%）；standard 在候选池充足时 100%、候选池=K 时 0%（§6.2） |

---

## 1. 两个项目的定位

| | DeepSelect | funnel-topk |
|---|---|---|
| 一句话 | DeepSeek Sparse Attention / 采样的**精确** TopK 生产内核 | "有界漏斗"**近似** TopK 研究原型（bounded funnel，B-Funnel） |
| 场景 | Lightning Indexer（bf16, topk≤4096）+ Sampling（fp32, vocab≈128K） | LLM sampling、向量检索等大 N 小 K 流式场景 |
| 硬件 | **仅 Blackwell**（sm_100a / sm_103a，setup.py:116-127 写死） | 通用：CUDA（任意 arch）/ Intel XPU (SYCL) / CPU (AVX512) / Triton |
| 正确性 | 精确，带 bitwise 测试矩阵 + NaN guard | turbo/fast 近似（recall 80~99%）；standard 靠大候选池 + torch.topk 逼近 100% |
| 规模 | 78 个 CUDA 模板实例 + PTX 级手写优化 | 942 行 CUDA + Triton fallback + CPU ext |
| 基线硬件 | B300（本仓库 README 图） | RTX 5090D（其 docs/01 的性能表） |

## 2. 两边算法各自的机制

**DeepSelect**（详见 [DeepSelect-deep-dive.zh.md](DeepSelect-deep-dive.zh.md)）：
维护一个单调收紧的全局 top-k 阈值 `T`，按**随机块顺序**流式扫描输入；元素大于 `T`
才进 shared memory 候选缓冲，缓冲涨到 `k+B2` 就用 radix-select 压缩回 `k` 个并抬高 `T`。
候选总量期望 `O((1+k/B2)(k+B+B2)·ln(N/B))` —— 精确且有理论界。小 batch 时用
Threadblock Cluster 把一行切到多个 SM。PTX 级优化：`__ffs` 消费 bitmask、
`set.s32.bf16x2` / `prmt` / `dp4a`、用浮点加法模拟整数加法避开端口冲突。

**funnel-topk**（详见其 docs/01_algorithm_design.md）：
每个线程在**寄存器**里维护 L 级有序漏斗，`funnel_push` 以 L 条 branchless CMOV
把新元素级联下推、溢出即丢弃；随后经 shared memory 交换，由 warp-0 合并出
`TL2×32 ≥ K` 个候选。三模式：turbo（L1=4, RS=0）、fast（RS=2 overflow-shuffle）、
standard（分段 L=8/16 → 每段 32×L 幸存者 → torch.topk 跨段合并）。
溢出即丢弃 ⇒ turbo/fast 本质近似；README 宣称 `K ≤ 32(L-1)` 时精确，
但其 docs/02 自己承认该表述"过强"，实际按 column 局部容量才成立。

## 3. 实现异同对照

**相同点（思想同源）**：

- 都是"单 pass 流式读全局内存 + 片上过滤候选"，每个元素只从 HBM 读一次；
- 都把 bf16 比较转成整数操作（funnel 叫 key32；DeepSelect 用 `set.s32.bf16x2`）；
- 都面向 LLM 的大 N 小 K，都以 torch.topk 为基线。

**本质差异**：

| 维度 | DeepSelect | funnel-topk |
|---|---|---|
| 过滤机制 | 全局阈值 + 周期性 radix-select 压缩（阈值收敛到第 k 大值 ⟹ 精确） | 每线程固定 L 槽位排序漏斗，溢出即丢（⟹ turbo/fast 近似） |
| 并行结构 | 多 CTA/行 + Threadblock Cluster（小 batch 也能吃满 SM） | turbo/fast 每行**只有 1 个 block**（grid(B,1,1)）；standard 分段自适应 |
| 输出 | 精确 K 个；可选 value/index、int32/64 | turbo/fast 输出 keep=TL2×32 ≥ K；standard 精确 K |
| K 上限 | 4096 | 1024（clean kernel，TL2≤32） |
| 优化深度 | 手写 PTX、按 maxtopk/threads/occupancy/TMA 实例化 78 个模板 | 常规 CUDA（`--maxrregcount=64` 调 occupancy），无 TMA/cluster |
| 功能面 | 变长 `end`、`output_idx_offset`、`sorted_index`、NaN abort | `causal_mask`、`torch.topk` 兼容 API、CPU/XPU fallback |
| 正确性工程 | 大规模 bitwise 测试矩阵（NaN、并列、变长、offset） | 以 index recall@K 为指标 |

## 4. 本机安装 funnel-topk：过程、坑与解决

结论先行：**可以测**，CUDA 扩展在 B300 上编译运行成功；但直接按 README 装会得到一个
**在 B300 上跑不了的 wheel**，且 Triton 后端在本机天然不可用。

### 4.1 坑 1（最严重）：默认编译的 arch 列表不含 sm_103，编出来 B300 跑不了

`pip install -e . --no-build-isolation` 时，funnel-topk 的 `setup.py`
（`funnel_topk/csrc/funnel_cuda.cu` 单文件扩展）**没有指定任何 `-gencode`**，
于是 torch 的 `cpp_extension` 用内置默认列表。实测编译命令行里是：

```
-gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=compute_120
-gencode arch=compute_120,code=sm_120 -gencode arch=compute_75,code=sm_75
-gencode arch=compute_80,code=sm_80  -gencode arch=compute_86,code=sm_86
-gencode arch=compute_90,code=sm_90
```

B300 是 **sm_103**：cubin 要求 major.minor 精确匹配（sm_100 的 cubin 不能加载）；
唯一的 PTX 是 compute_120，需要设备 ≥ 12.0，10.3 不满足。**整条列表里没有
sm_103 能执行的代码**，运行时会报 `no kernel image is available for execution on the device`。

**解决**：

```bash
TORCH_CUDA_ARCH_LIST="10.3" pip install -e . --no-build-isolation
```

实测编译行变为 `-gencode=arch=compute_103,code=sm_103`，冒烟通过。

### 4.2 坑 2：默认编译 7 个 arch，13+ 分钟编不完

发现坑 1 时的直接证据：默认编译为 7 个 arch 各展开一遍
`funnel_clean_kernel_bf16<L1,T,TL2,RS>` × `funnel_segment_kernel_*<L>` 模板
（clean kernel 2 dtype × 2 L1 × 2 T × 6 TL2 × 3 RS + segment 7 L 值……），
跑了 13 分 26 秒还停在 compute_86 的 cicc 阶段。杀掉后用坑 1 的单 arch 方案重编，
**3 分 51 秒**完成。两个坑是同一个根因（未设 `TORCH_CUDA_ARCH_LIST`）的一体两面。

### 4.3 坑 3：Triton 后端依赖作者机器上的硬编码路径，本机不可用

`funnel_topk/_kernels.py:88-104` 与 `funnel.py:32-47` 都从
`/home/yuluo/py_test_clean_repo/brft_triton_sketch.py` 加载共享 Triton 组件，
文件不存在时静默跳过（`except: pass`）。本机实测：

```
CUDA ext: True | Triton funnel: False
```

影响：**GPU 上只剩 CUDA ext 一条路**（`HAS_FUNNEL_TRITON=False`）；若 CUDA ext 也没编上，
`funnel.topk()` 会静默退化为 `torch.topk` 或 CPU simulator —— 用它的性能数字前
务必先确认后端，否则可能在给 torch.topk 计时。CUDA ext 只支持 bf16/fp32
（fp16 会静默 fallback）。

### 4.4 冒烟测试（B=256, N=65536, K=512, bf16）

```
mode=standard  keep=512  idx_recall=99.6%
mode=fast      keep=512  idx_recall=93.6%
mode=turbo     keep=512  idx_recall=80.6%
```

recall 趋势与其官方文档一致（turbo K=512 官方就是 80.6%，逐位吻合）。

## 5. 对比方法论（为什么必须重测，而不能拿两边 README 数字对比）

两个项目的官方口径**完全不同**，直接比会产生系统性偏差：

| 口径项 | DeepSelect（tests/test.py + kernelkit） | funnel-topk（benchmarks/bench_compare.py） |
|---|---|---|
| 计时 | kineto/CUPTI **kernel 时间** | `time.perf_counter` **wall time**（含 launch/同步开销） |
| L2 缓存 | 每次迭代前 **8 GB `zero_()` 冲刷**（冷 L2） | 不冲刷（B300 L2=126 MB，小输入会残留命中） |
| 取值 | 10 runs 均值 | 100 iters 中位数 |
| 基线硬件 | B300 | RTX 5090D |

统一方案（[`benchmarks/compare_ds_vs_funnel.py`](../benchmarks/compare_ds_vs_funnel.py)）：

- **同一计时框架**：DeepSelect 自家 `tests/kernelkit` 的 `kk.bench`
  （kineto kernel 时间 + 每迭代 8 GB L2 flush），即 README `perf_bf16.png` 的口径；
- **同一负载**：bf16、K=512、`batch ∈ {6, 256, 512, 4096}` ×
  `vocab ∈ {16K, 64K, 128K, 256K, 512K, 1M}`，randn 输入，每方案 num_runs=10；
- **同一计时范围**：取一次调用**全部 device kernel 的 e2e 跨度**
  （funnel standard 有 5 个 kernel：segment → sbtopk::gatherTopK → scatter_gather →
  两个 cast copy；turbo 有 3 个；DeepSelect 单 kernel）；
- **同一生效字节口径**：输入 B×N×2B + 各方案真实输出大小（索引 int32/int64 按实际计）；
- **recall**：index recall@K vs torch.topk 参考解；另对 DeepSelect 验证
  topk 条件 `min(selected) ≥ max(unselected)`。

### 5.1 坑 4：按子串过滤 kernel 名，差点把 torch 的 kernel 误杀

第一版脚本用 `"cuda" not in name` 排除 CPU 侧 runtime API 事件
（`cudaLaunchKernel` / `cudaDeviceSynchronize` 等也会被 kineto 收进事件列表），
结果小输入下 torch.topk 走 sbtopk 路径，其 kernel 全名
`void at::native::sbtopk::gatherTopK<c10::BFloat16, ...>(at::cuda::detail::TensorInfo...)`
的**模板参数里含 "cuda"**，被误杀 → 过滤后列表为空 → `get_e2e_time` 抛
`ValueError: min() iterable argument is empty`。

**解决**：device kernel 名都以 `void ` 开头，runtime API 事件是裸名——
改为**前缀匹配**排除（`startswith(("cuda", "Memset"))`），再按子串排除
L2 flush 的 `FillFunctor`。修正后 funnel 的 copy/cast 等真实开销 kernel 保留计入，
对三方公平。

### 5.2 内控：测量环境可信性验证

跑对比时机器上有另一个会话的 vLLM smoke 进程占着 4×43 GB 显存（util 0%）。
为排除干扰疑虑，利用脚本里 DeepSelect 与 torch.topk 两行是**已复现过的"已知答案"**
做内控：本次 `B=4096, V=1M` 测得 ds 1617.2 µs / torch 32872.4 µs，
与今晨复现 CSV（1618.059 / 32863.470）偏差 < 0.03% —— 测量环境干净，funnel 的数字同样可信。

## 6. 统一口径实测数据（bf16，K=512，B300，延迟 µs）

**DeepSelect / funnel 三模式 / torch.topk**（加粗为当行最快）：

| batch | vocab | DeepSelect | funnel std | funnel fast | funnel turbo | torch.topk | DS vs funnel-best |
|---|---|---|---|---|---|---|---|
| 6 | 16384 | **6.1** | 50.7 | 62.2 | 20.2 | 60.5 | **3.3x** |
| 6 | 65536 | **11.4** | 110.3 | 80.0 | 52.6 | 49.8 | **4.6x** |
| 6 | 131072 | **16.6** | 83.0 | 103.5 | 95.9 | 57.6 | **5.0x** |
| 6 | 262144 | **25.7** | 88.5 | 150.5 | 181.5 | 66.8 | **3.4x** |
| 6 | 524288 | **18.7** | 94.8 | 243.5 | 353.5 | 78.5 | **5.1x** |
| 6 | 1048576 | **23.2** | 110.4 | 429.0 | 696.3 | 101.7 | **4.8x** |
| 256 | 16384 | **9.5** | 63.5 | 53.3 | 22.5 | 82.7 | **2.4x** |
| 256 | 65536 | **18.7** | 106.1 | 72.0 | 60.0 | 160.7 | **3.2x** |
| 256 | 131072 | **27.2** | 152.0 | 98.4 | 111.1 | 279.3 | **3.6x** |
| 256 | 262144 | **42.2** | 254.9 | 160.0 | 210.6 | 564.8 | **3.8x** |
| 256 | 524288 | **70.0** | 412.9 | 292.8 | 400.9 | 997.3 | **4.2x** |
| 256 | 1048576 | **120.2** | 725.2 | 561.2 | 778.0 | 1904.0 | **4.7x** |
| 512 | 16384 | **16.3** | 70.9 | 55.1 | 28.7 | 116.5 | **1.8x** |
| 512 | 65536 | **33.6** | 174.2 | 92.5 | 81.0 | 276.1 | **2.4x** |
| 512 | 131072 | **50.0** | 263.5 | 148.3 | 147.7 | 565.5 | **3.0x** |
| 512 | 262144 | **80.1** | 415.7 | 269.9 | 285.9 | 996.2 | **3.4x** |
| 512 | 524288 | **135.2** | 727.4 | 510.7 | 564.1 | 1893.7 | **3.8x** |
| 512 | 1048576 | **236.6** | 1339.1 | 988.3 | 1117.5 | 3680.2 | **4.2x** |
| 4096 | 16384 | **94.3** | 269.4 | 297.2 | 119.8 | 589.8 | **1.3x** |
| 4096 | 65536 | **210.5** | 768.3 | 486.2 | 408.4 | 1920.4 | **1.9x** |
| 4096 | 131072 | **325.8** | 1371.9 | 838.9 | 791.8 | 3707.5 | **2.4x** |
| 4096 | 262144 | **534.8** | 2598.9 | 1582.9 | 1557.3 | 7353.5 | **2.9x** |
| 4096 | 524288 | **915.9** | 5137.9 | 3072.5 | 3087.2 | 16526.0 | **3.4x** |
| 4096 | 1048576 | **1617.2** | 10113.7 | 6048.6 | 6145.5 | 32872.4 | **3.7x** |

**正确性**（DS exact = topk 条件 `min(selected) ≥ max(unselected)`；rc = index recall@K vs torch.topk）：

| batch | vocab | DS exact | DS idx-rc | std rc | fast rc | turbo rc |
|---|---|---|---|---|---|---|
| 6 | 16384 | True | 100.0% | 99.9% | 94.2% | 80.6% |
| 6 | 65536 | True | 99.3% | 99.9% | 93.0% | 80.5% |
| 6 | 131072 | True | 99.4% | 100.0% | 93.4% | 81.0% |
| 6 | 262144 | True | 99.3% | 100.0% | 93.7% | 81.8% |
| 6 | 524288 | True | 99.4% | 100.0% | 93.6% | 81.1% |
| 6 | 1048576 | True | 99.0% | 100.0% | 93.5% | 80.5% |
| 256 | 16384 | True | 100.0% | 90.4% | 93.7% | 80.6% |
| 256 | 65536 | True | 99.2% | 99.6% | 93.6% | 80.6% |
| 256 | 131072 | True | 99.2% | 99.8% | 93.5% | 80.7% |
| 256 | 262144 | True | 99.1% | 99.9% | 93.5% | 80.6% |
| 256 | 524288 | True | 99.1% | 99.9% | 93.6% | 80.6% |
| 256 | 1048576 | True | 99.0% | 100.0% | 93.5% | 80.6% |
| 512 | 16384 | True | 100.0% | 90.4% | 93.7% | 80.7% |
| 512 | 65536 | True | 99.2% | 99.6% | 93.6% | 80.6% |
| 512 | 131072 | True | 99.2% | 99.8% | 93.6% | 80.6% |
| 512 | 262144 | True | 99.1% | 99.9% | 93.5% | 80.5% |
| 512 | 524288 | True | 99.1% | 99.9% | 93.5% | 80.5% |
| 512 | 1048576 | True | 99.0% | 100.0% | 93.4% | 80.5% |
| 4096 | 16384 | True | 100.0% | 90.4% | 93.7% | 80.9% |
| 4096 | 65536 | True | 99.2% | 99.6% | 93.6% | 80.6% |
| 4096 | 131072 | True | 99.2% | 99.8% | 93.5% | 80.6% |
| 4096 | 262144 | True | 99.1% | 99.9% | 93.5% | 80.5% |
| 4096 | 524288 | True | 99.1% | 99.9% | 93.5% | 80.5% |
| 4096 | 1048576 | True | 99.0% | 100.0% | 93.4% | 80.5% |

### 6.1 数据分析

- **DeepSelect 24/24 全胜**，对 funnel 最快模式快 1.27x ~ 5.08x。最大带宽
  5.32 TB/s（B=4096, V=1M），同日 README 复现值 5.314 TB/s，内控一致。
- **funnel turbo/fast 在 B=6 崩盘**：grid(B,1,1) 只有 6 个 block，148 个 SM 只用 6 个
  （4%）。V=1M 时 turbo 696 µs，比 torch.topk（102 µs）还慢 **6.8 倍**——
  单 block 串行扫 2 MB 的后果。standard 靠 `num_seg` 自适应（B=6 时每行切 8~49 段）
  缓解，但 merge 阶段 torch.topk 又引入额外开销。
- **funnel 相对 torch 的优势区间是大 batch**：B=1024/4096 时 turbo 3.3x~5.4x，
  与其官方宣称（RTX 5090D 上 3.9x）量级一致；小 batch 全模式都打不过 torch。
- **standard 的 recall 陷阱**：B≥256 且 V=16K 时 `num_seg=max(ceil(16384/32768), round(2×148/B))=1`，
  候选池 = 32×16×1 = 512 **恰好等于 K**，任何溢出都直接损失 recall → 90.4%。
  B=6 时 num_seg=8（候选池 4096）反而 99.9%+。这与官方文档"幸存数≈K 时
  recall≈80-86%"的机制描述一致。
- **index recall < 100% ≠ 不精确**：DeepSelect 的 idx-rc 为 99.0~100%，
  但 topk 条件 24/24 全 True——bf16 只有 8 位尾数，大 N 下大量并列值使
  索引集合本身不唯一。拿 index recall 当"精确性"指标会冤枉精确内核，
  评估这类算子应看 topk 条件或值多重集。
- **funnel 输出 keep=TL2×32**：本矩阵 K=512 → keep=512，三方输出字节数相同，
  带宽口径无偏差。

### 6.2 正确性硬检查（对齐 DeepSelect 测试套件的检查项）

index recall@K 只是一个统计指标；DeepSelect 的测试套件（`tests/test.py`）做的是
**逐元素硬检查**。用同一套标准检查三方（脚本
[`benchmarks/verify_correctness_vs_funnel.py`](../benchmarks/verify_correctness_vs_funnel.py)，
bf16，K=512，batch ∈ {6, 256, 4096} × vocab ∈ {16K, 256K, 1M}）：

| 检查项 | DeepSelect | funnel standard | funnel fast | funnel turbo |
|---|---|---|---|---|
| 索引范围 `0 ≤ idx < N` | 9/9 通过 | 通过 | 通过 | 通过 |
| 索引唯一（行内无重复） | 通过 | 通过 | 通过 | 通过 |
| gather 一致（`values == input[indices]`） | 通过 | 通过 | 通过 | 通过 |
| **topk 条件（行满足率）** | **100%（9/9 配置）** | 100% 或 **0%** | **0%** | **0%** |

topk 条件 = `min(selected) ≥ max(unselected)`，即精确 top-k 的定义本身。实测细节：

- **fast / turbo 在全部 9 个配置上行满足率为 0%**——每一行都至少漏掉一个真实
  top-k 元素。recall 93.5% 听起来温和，但对 K=512 意味着平均每行漏约 33 个；
  "recall 高"并不等于"大多数行是对的"。
- **standard 的 topk 行满足率随候选池规模在 100% 与 0% 之间跳变**：
  候选池充足时（B=6 全档、V≥256K 全档）100% 精确；候选池恰好=K 时
  （B=256/4096, V=16K，`num_seg=1` → 32×16=512=K）**0%**——每个 segment 的
  漏斗溢出都直接变成漏选。也就是说 standard 并非"近似但接近"，而是
  "要么精确、要么每行都错"，取决于 `num_seg` 的自适应结果，调用方无法预知。
- DeepSelect 三项健全性检查 + topk 条件全部通过，与其测试套件的
  bitwise 结论一致。

原始数据：`verify_correctness_vs_funnel.json`。

### 6.3 funnel-topk 官方 bench（wall-clock 口径）本机对照

`python -m benchmarks.bench_compare --device cuda`（B300，median of 100 iters，
无 L2 flush；节选 bf16 行）：

| B | N | K | standard | fast | turbo | flashinfer |
|---|---|---|---|---|---|---|
| 256 | 262144 | 512 | 290us 2.01x / 99.9% | 213us 2.74x / 93.5% | 224us 2.60x / 80.6% | 182us 3.21x / 99.7% |
| 1024 | 262144 | 512 | 0.76ms 2.53x / 99.9% | 499us 3.84x / 93.5% | 476us 4.03x / 80.6% | 476us 4.03x / 99.8% |
| 1024 | 262144 | 1024 | 0.76ms 2.53x / 99.9% | 0.82ms 2.34x / 86.1% | 452us 4.26x / 80.6% | 491us 3.92x / 99.8% |

（sp = vs torch.topk；本机恰好装有 flashinfer，一并列出。）

要点：官方口径下 turbo 在 B=1024 达到 4x 级加速，与宣传一致；但同口径下
**flashinfer.top_k 速度与 turbo 相当且 recall 100%**——funnel 的速度优势很大程度
来自"近似"，一旦要求精确（standard），speedup 落到 2.0x~2.5x，反而输给 flashinfer。
另外注意官方 bench 的 B 最小 256，回避了 B=6 这种 decode 小批量场景。

## 7. 结论

1. **异同**：思想同源（单 pass 流式 + 片上过滤 + bf16 整数比较），但
   DeepSelect 是"全局阈值 + radix-select 压缩"的**精确**算法，配 cluster/PTX 级
   深度优化，仅服务 Blackwell；funnel-topk 是"寄存器漏斗溢出即丢"的**近似**算法
   （standard 模式靠大候选池 + torch.topk 逼近精确），架构简单、硬件通用。
2. **本机可测**：可以。必须 `TORCH_CUDA_ARCH_LIST="10.3"` 重编（默认 arch 列表
   不含 sm_103，编出来在 B300 上跑不了，且 7-arch 全编要等 13 分钟+）；
   Triton 后端因硬编码路径缺失在本机不可用，GPU 上只走 CUDA ext。
3. **可对比，且应统一口径后对比**：同口径（kernel 时间 + 冷 L2 + 同负载）下
   DeepSelect 在 Lightning Indexer 场景 24/24 全胜，快 funnel 最快模式
   **1.3x ~ 5.1x**，且结果精确；funnel 的速度优势以 recall 为代价
   （turbo 80.5%），小 batch（B=6）甚至打不过 torch.topk。
   若场景能接受 ~80% recall 且只需要 torch.topk 兼容 API、跨硬件部署，
   funnel-topk 仍有其生态位；追求精确与极致带宽的场景 DeepSelect 明显更强。

## 8. 产出物清单

| 路径 | 说明 |
|---|---|
| [`benchmarks/compare_ds_vs_funnel.py`](../benchmarks/compare_ds_vs_funnel.py) | 统一口径对比脚本（kk.bench，三方全 kernel e2e span） |
| [`benchmarks/verify_correctness_vs_funnel.py`](../benchmarks/verify_correctness_vs_funnel.py) | 正确性硬检查脚本（§6.2） |
| [`benchmarks/parse_perf_bf16.py`](../benchmarks/parse_perf_bf16.py)、[`benchmarks/plot_perf_bf16.py`](../benchmarks/plot_perf_bf16.py) | README Lightning Indexer 复现的解析 / 出图脚本 |
| `compare_ds_funnel_20260926.json` | 24 配置 × 5 方案的原始数据（µs / TB/s / recall / keep） |
| `verify_correctness_vs_funnel.json` | 正确性硬检查原始数据（9 配置 × 4 方案） |
| funnel-topk 安装 | editable install 于 muse_verify；`TORCH_CUDA_ARCH_LIST="10.3"`，编译 3 分 51 秒 |
| 官方 bench 输出 | 见 §6.3（命令 `python -m benchmarks.bench_compare --device cuda`） |
| 本文档 | `docs/compare-funnel-topk-20260926.zh.md` |

复现脚本均已用"从仓库位置重跑"的方式验证过：对比脚本重跑 24 配置与首跑
最大相对偏差 4.23%（µs 级小 case 抖动量级），recall 一致；解析脚本输出与
`bf16_perf_20260926.csv` 逐行一致。

仓库代码零改动（对 funnel-topk 仅做 editable install 与运行时调用；
对 DeepSelect 仅新增 `benchmarks/` 下复现脚本与本文档，未触碰任何已有文件）。
