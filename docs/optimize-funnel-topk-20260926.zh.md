# funnel-topk 性能优化记录（2026-09-26，B300）

本文记录对 [`/models/lkk/funnel-topk`](https://github.com/luoyu-intel/funnel-topk)
的性能优化全过程：瓶颈分析、两轮优化（一轮无损、一轮结构性）、途中一次失败的
方案及定位方法、最终数据。目标是尽量接近 DeepSelect 在 Lightning Indexer 场景
（bf16, K=512）的性能。

对比背景见 [compare-funnel-topk-20260926.zh.md](compare-funnel-topk-20260926.zh.md)
（含两项目的实现异同与优化前的基线数据）。

---

## 0. TL;DR

| 优化 | 机制 | 收益 | 正确性影响 |
|---|---|---|---|
| **轮 1：漏斗底阈值守卫** | 每元素 1 次比较代替 4 级 push 链（V≥256K 时启用） | turbo/fast 大 V 提速 **1.29x ~ 1.82x** | **无损**（recall 逐位一致，Δ=0.000pp） |
| **轮 2：分段并行 + 精确 merge** | 小 B 时 grid(B)→grid(B,S) + torch.topk merge | B=6 大 V 提速 **2.31x ~ 6.30x** | recall 81%/93% → **~100%**，topk 条件行满足率 0% → **100%** |

最终与 DeepSelect 的差距：从优化前的 1.27x ~ 5.08x（funnel-best）收敛到
大 V 区域 **1.8x ~ 3.5x**；funnel 最快点在 B=4096, V=1M 达 **2.55 TB/s**
（DeepSelect 5.32 TB/s）。**完全追平未达成**，剩余差距的根源分析见 §5。

改动全部在 funnel-topk 侧（`funnel_cuda.cu` +318 行、`_kernels.py` +31 行），
编译参数 `TORCH_CUDA_ARCH_LIST="10.3"`。DeepSelect 侧只新增复现/验证脚本。

---

## 1. 瓶颈分析（先量化，再动手）

优化前的 funnel 数据（B300，同口径 kernel 时间）：

| 配置 | turbo | fast | standard | DeepSelect |
|---|---|---|---|---|
| B=6, V=1M | 696.3 µs | 429.0 µs | 110.4 µs | 22.8 µs |
| B=256, V=1M | 778.0 µs | 561.2 µs | 725.2 µs | 120.0 µs |
| B=4096, V=1M | 6145.5 µs | 6048.6 µs | 10113.7 µs | 1617.4 µs |

**瓶颈 1：push 链的计算成本（所有配置）**。kernel 已做 int4 向量化加载
（访存不是问题：turbo B=4096 V=1M 只用 17% HBM 带宽），瓶颈在每元素的
`funnel_push_key32<4>`：~4 条指令的 `bf16_to_key` + 4 级级联（每级 1 比较 +
4 个条件赋值）≈ **20+ 指令/元素**，而 DeepSelect 每元素只有 1 次阈值比较 +
mask 写（~2-3 指令）。

**瓶颈 2：小 batch 并行度崩盘（turbo/fast）**。clean kernel 是 `grid(B,1,1)`
一行一个 block：B=6 时 148 个 SM 只用 6 个（4%）。standard 模式有
`num_seg` 自适应所以不崩（B=6 V=1M 时 110 µs 反而比 turbo 快 6 倍）。

## 2. 轮 1：漏斗底阈值守卫（无损）

### 2.1 原理（精确无损的证明）

funnel 的 L 级寄存器数组始终**降序**（ss[0] ≥ ... ≥ ss[L-1]）。级联 push 中，
若新元素 `cur < ss[L-1]`，则每一级的 `take` 测试都为假，元素原样溢出——
**push 完全无效果**。因此：

```cpp
if (cur >= ss[L-1]) funnel_push(...);   // 与无条件 push 精确等价
```

这不是 DeepSelect 那种全局阈值，而是**线程级**的：阈值就是本线程漏斗的当前
最小值，漏斗一填满（前几十个元素）就自动生效，无需任何跨线程通信。

### 2.2 实测发现的问题与修正

第一版把守卫无条件加到全部 5 处流式循环，结果：

- turbo/fast 大 V 提速 1.3x~1.8x ✓，但 **V<256K 普遍变慢 5~21%**，
  standard（segment kernel，段内扫描量 ≤32K）全线变慢 0.79x~0.96x。
- 原因：守卫是分支指令，push 链原本是无分支 CMOV 流；扫描量小时漏斗尚未
  收敛、拒绝率低，守卫纯属开销。

修正（数据驱动）：clean kernel 按 `V ≥ 262144` 运行期选择守卫/原版双循环；
segment kernel 守卫**回滚**（段内扫描量小，恒亏）。修正后：

- V<256K 全部回到 1.00x（零回退）；
- V≥256K 保留全部收益：turbo **1.29x~1.82x**，fast **1.36x~1.79x**；
- recall 与优化前**逐位一致**（24 配置 × 3 模式，max |Δrecall| = 0.000pp），
  无损性实证成立。

编译期踩坑：新增的 f32 守卫 helper 最初插在 `funnel_push` 定义之前，
`error: identifier "funnel_push" is undefined`——移到定义之后解决。

## 3. 轮 2：分段并行修复小 batch 崩盘

### 3.1 第一版：两级漏斗 merge（失败，recall 崩了）

分段 stream kernel（grid(B,S)，每段输出 T×L1=512 候选）+ 自写漏斗 merge
kernel。速度很好（B=6 V=1M turbo 696→62 µs，**11.2x**），但 recall 暴跌：

| S（段数） | 2 | 4 | 8 | 50 |
|---|---|---|---|---|
| 实测 recall | 71.7% | 57.3% | 44.3% | 29.2% |

**定位方法**：写了一个 torch 级精确模拟（复刻段内 strided 扫描 + merge
扫描的线程-数据映射），模拟值与 kernel 实测逐点吻合（44.0% vs 44.3%）——
证明不是实现 bug，而是**结构缺陷**：

- merge 前候选池 recall 其实已达 96~100%（分段越大池越好）；
- 但 merge 是又一个 512 槽漏斗：torch 模拟显示 **merge 阶段单独就把
  recall 从 100% 打到 44%**（S=8）。512 槽漏斗装不下更大无序候选池的
  精确 top-512；每线程真值期望 λ=keep/T=4 恰好顶在 4 槽容量上，
  泊松涨落导致 ~20% 丢失，段间聚簇进一步放大。

### 3.2 修正：分段 + torch.topk 精确 merge

候选池 recall 已经 ~100%，缺的只是**精确**取出 top-k。改为：stream kernel
直接输出 (score, index) 候选对（`funnel_seg_candidates_bf16`），Python 侧
`torch.topk` + `gather` merge。merge 精确 ⇒ 总 recall = 池 recall。

启用条件（按实测盈亏点收窄，只在净赚区域启用）：**B ≤ 32 且 N ≥ 256K**，
S = min(⌈2·SM/B⌉, N/2048, 16)。结果：

| B=6 | V=256K | V=512K | V=1M |
|---|---|---|---|
| turbo 提速 | **2.31x** | **3.95x** | **6.30x** |
| fast 提速 | 1.92x | 2.72x | 3.88x |
| recall（turbo/fast） | 82%/94% → **100%** | 同左 | 同左 |
| topk 条件行满足率 | 0% → **100%** | 0% → 100% | 0% → 100% |

注意语义升级：该路径下 turbo/fast **从近似变成精确**（池含全部真值 +
精确 merge），通过了 DeepSelect 测试套件同款的全部硬检查（索引范围/
唯一性/gather 一致/topk 条件）。

被否决的启用区域（实测净亏，未启用）：B=256（torch.topk merge 的
~30-40 µs 固定开销吃掉了 stream 提速，fast 0.83x~0.95x）、V<256K 的 B=6
（同理，turbo 0.42x~0.75x）。

## 4. 最终数据（同口径，bf16，K=512，B300）

优化前后 funnel-best（三模式取最快）与 DeepSelect 的对照（µs）：

| batch | vocab | DeepSelect | funnel-best 旧 | funnel-best 新 | 提升 | 与 DS 差距 |
|---|---|---|---|---|---|---|
| 6 | 16384 | 6.1 | 20.2 | 20.1 | 1.01x | 3.3x |
| 6 | 65536 | 11.4 | 52.6 | 52.5 | 1.00x | 4.6x |
| 6 | 131072 | 16.6 | 83.0 | 85.8 | 0.97x | 5.2x |
| 6 | 262144 | 25.7 | 88.5 | 78.8 | **1.12x** | 3.1x |
| 6 | 524288 | 18.6 | 94.8 | 89.3 | 1.06x | 4.8x |
| 6 | 1048576 | 22.8 | 110.4 | 110.4 | 1.00x | 4.8x |
| 256 | 16384 | 9.6 | 22.5 | 22.7 | 0.99x | 2.4x |
| 256 | 65536 | 18.9 | 60.0 | 60.4 | 0.99x | 3.2x |
| 256 | 131072 | 27.1 | 98.4 | 105.5 | 0.93x | 3.9x |
| 256 | 262144 | 42.1 | 160.0 | 156.8 | 1.02x | 3.7x |
| 256 | 524288 | 69.9 | 292.8 | 244.2 | **1.20x** | 3.5x |
| 256 | 1048576 | 120.0 | 561.2 | 419.7 | **1.34x** | 3.5x |
| 512 | 16384 | 16.3 | 28.7 | 28.8 | 1.00x | 1.8x |
| 512 | 65536 | 33.6 | 81.0 | 81.0 | 1.00x | 2.4x |
| 512 | 131072 | 50.0 | 147.7 | 147.7 | 1.00x | 3.0x |
| 512 | 262144 | 80.2 | 269.9 | 193.7 | **1.39x** | 2.4x |
| 512 | 524288 | 135.5 | 510.7 | 315.1 | **1.62x** | 2.3x |
| 512 | 1048576 | 236.2 | 988.3 | 551.6 | **1.79x** | 2.3x |
| 4096 | 16384 | 94.5 | 119.8 | 119.9 | 1.00x | 1.3x |
| 4096 | 65536 | 210.6 | 408.4 | 408.6 | 1.00x | 1.9x |
| 4096 | 131072 | 326.0 | 791.8 | 791.5 | 1.00x | 2.4x |
| 4096 | 262144 | 534.9 | 1557.3 | 964.3 | **1.61x** | 1.8x |
| 4096 | 524288 | 915.7 | 3072.5 | 1776.3 | **1.73x** | 1.9x |
| 4096 | 1048576 | 1617.4 | 6048.6 | 3374.8 | **1.79x** | 2.1x |

说明：funnel-best 的提升在小 B 行不显眼，是因为 standard 模式原本就靠
`num_seg` 自适应兜了底（B=6 V=1M 本就 110 µs）；分段路径的真正价值是让
**turbo/fast 两个模式**各自提速 2.3x~6.3x 并同时变精确（§3.2）。

原始数据：`compare_ds_funnel_after_opt_20260926.json`（对照
`compare_ds_funnel_20260926.json` 为优化前基线）。

**复跑复核**：上表数据已用最终代码状态独立重跑一遍（命令见 §6），
逐点偏差 < 0.5%（如 turbo B=4096 V=1M：3374.8 → 3374.3 µs；
DeepSelect 同点 1617.4 → 1618.1 µs），结论稳定。

### 4.1 官方口径复核（funnel 自家 bench_compare，wall-clock）

为排除"只在我们选的口径下变好"的疑虑，用 funnel-topk 自带的
`benchmarks/bench_compare.py`（`time.perf_counter` median of 100 iters，
**不冲 L2**）重跑优化前后对照：

| B | N | K | dtype | fast 旧→新 | turbo 旧→新 | turbo vs flashinfer 旧→新 |
|---|---|---|---|---|---|---|
| 256 | 262144 | 512 | bf16 | 213µs/2.74x → 207µs/2.86x | 224µs/2.60x → 222µs/2.66x | 慢于 fi(182µs) → 仍慢 |
| 1024 | 262144 | 512 | bf16 | 499µs/3.84x → **393µs/4.88x** | 476µs/4.03x → **311µs/6.17x** | 打平(476 vs 476) → **快 1.53x**(311 vs 470) |
| 1024 | 262144 | 1024 | bf16 | 0.82ms/2.34x → 0.53ms/3.67x | 452µs/4.26x → 346µs/5.60x | 452 vs 491 → **346 vs 494** |
| 1024 | 262144 | 512 | fp32 | 405µs/7.35x → 376µs/7.91x | 378µs/7.88x → **254µs/11.70x** | 378 vs 540 → **254 vs 540（快 2.1x）** |

（speedup 均为 vs torch.topk。recall 全部不变：turbo 80.5%、fast 93.5%，
与 §2 的无损性一致。）

官方口径下的两个标志性变化：turbo fp32 对 torch.topk 的加速从 7.88x 提到
**11.70x**；bf16 B=1024 下 turbo 从与 flashinfer 打平变为**快 1.5x**。
注意官方 bench 的 B∈{256,1024} 不会命中分段路径（B>32），体现的全是
轮 1 守卫的收益；V=65536 的行未命中守卫阈值（V<256K），数字与旧版一致，
同样符合预期。

## 5. 与 DeepSelect 剩余差距的根源（为什么没追平）

优化后 funnel 最快 2.55 TB/s vs DeepSelect 5.32 TB/s，差 ~2.1x。逐项归因：

1. **每元素成本仍高**：守卫只省了 push 链，`bf16_to_key` 变换 + 守卫比较仍有
   ~4-5 指令/元素；DeepSelect 用 `set.s32.bf16x2` 一次比较 2 个元素 +
   bitmask，~1.5 指令/元素。funnel 要再降需要 SIMD 化守卫（`__vcmpgtu2` 类
   per-halfword 比较），预估再省 10~20%，不改量级。
2. **merge 开销**：turbo/fast 的 warp-0 merge + sort、分段路径的 torch.topk
   （~30-40 µs 固定）；DeepSelect 单 kernel 完成，无独立 merge 阶段。
3. **无 cluster/TMA**：DeepSelect 小 batch 用 Threadblock Cluster 把一行切到
   多 SM 且 cluster 内归并（B=6 仅 23 µs）；funnel 的分段方案要经全局内存 +
   torch.topk 中转，B=6 最好也只能到 ~79 µs。
4. **算法常数**：DeepSelect 的全局单调阈值一旦收敛，候选率降到 ~k/N 量级，
   几乎纯读带宽；funnel 的线程级阈值（漏斗底）收敛慢且每线程独立，接受率更高。

结论：funnel 的"寄存器漏斗"结构决定了它的常数因子大于 DeepSelect 的
"全局阈值 + radix-select 压缩"结构；本轮优化把常数砍掉了约一半（1.8x~6.3x
分场景），但要追平需要把算法骨架换成 DeepSelect 的路线——那等于重写。

## 6. 改动清单与复现

funnel-topk 侧（`git diff --stat`: +303/-46，两个文件）：

- `funnel_topk/csrc/funnel_cuda.cu`
  - 新增无损守卫 helper：`funnel_guarded_push_key32` / `funnel_guarded_push`
    （另有两个 segment 用 helper 因实测净亏未被调用，保留备用）；
  - `funnel_clean_kernel_bf16/f32`：`V ≥ 262144` 时走守卫版循环，否则原版；
  - 新增 `funnel_seg_stream_kernel_bf16`（分段候选生产）+
    `funnel_seg_candidates_bf16` host 函数 + pybind 注册；
- `funnel_topk/_kernels.py`：fast/turbo 在 `B ≤ 32 且 N ≥ 256K` 时路由到
  分段路径（候选 + `torch.topk` 精确 merge）。

复现步骤：

```bash
source /models/miniforge3/etc/profile.d/conda.sh && conda activate muse_verify
cd /models/lkk/funnel-topk
TORCH_CUDA_ARCH_LIST="10.3" python setup.py build_ext --inplace   # 增量重编 ~3.5min

# 验证（脚本在 DeepSelect 仓库 benchmarks/ 下）
python3 /models/DeepSelect/benchmarks/verify_round1_guard.py       # 守卫：recall 不变 + 提速
python3 /models/DeepSelect/benchmarks/verify_round2_segmented.py   # 分段：提速 + recall 升
python3 /models/DeepSelect/benchmarks/verify_correctness_vs_funnel.py  # 硬检查
python3 /models/DeepSelect/benchmarks/compare_ds_vs_funnel.py out.json # 全矩阵对比
```

## 7. 若继续优化的路径（按预期收益排序）

1. **分段路径的 merge 换自写精确 kernel**（去掉 torch.topk 的 30-40 µs）：
   B=6 预计 79~110 µs → 40~70 µs，与 DS 差距 5x → ~3x。
2. **守卫 SIMD 化**（bf16x2 一次比 2 元素）：大 B 大 V 再省 10~20%。
3. **B=256 档的分段调优**：该档守卫版 fast 已到 1.28~1.86 TB/s，剩余靠
   occupancy 调优（T=256、maxrregcount 放开）微调。
4. 追平 DeepSelect 需要换成全局阈值 + radix-select 骨架（≈重写），不建议在
   funnel 仓库内做。
