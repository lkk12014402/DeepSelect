# Lightning Indexer Scenario 复现指南（环境 → 安装 → 跑分 → 出图）

本文记录在 **NVIDIA B300** 上从零开始安装 DeepSelect 并复现 README 中
[Lightning Indexer Scenario](../README.md#lightning-indexer-scenario) 性能数据（即
`assets/perf_bf16.png`）的完整流程，包含实测过程中踩到的坑。

文中所有版本号、耗时、数据均为本次实测结果，非估算。

---

## 0. TL;DR

```bash
# 1) 代理 + 环境（本次复用已有环境 muse_verify）
export https_proxy=http://proxy.ims.intel.com:911
export http_proxy=http://proxy.ims.intel.com:911
source /models/miniforge3/etc/profile.d/conda.sh
conda activate muse_verify

# 2) 子模块
cd /models/DeepSelect && git submodule update --init --recursive

# 3) 编译安装（不要用 `pip install .`，见 §4.2 的坑）
MAX_JOBS=18 NVCC_THREADS=4 python setup.py bdist_wheel
pip install --no-deps --force-reinstall dist/deep_select-*.whl

# 4) 跑 Lightning Indexer（bf16）性能用例
CUDA_VISIBLE_DEVICES=0 python -u tests/test.py --perf-only --dtype bf16 -rf \
    | tee /tmp/perf_bf16.log

# 5) 解析 + 出图：把 §7 / §8 里的两段 python 存成文件后执行
python3 parse_perf_bf16.py /tmp/perf_bf16.log ./bf16_perf.csv
python3 plot_perf_bf16.py  ./bf16_perf.csv      ./perf_bf16_repro.png
```

本次实测结论：

- 90 个 bf16 性能用例**全部通过正确性检查**（`All 90 cases passed!`）。
- 有效带宽范围 **0.006 ~ 5.313 TB/s**，相对 `torch.topk` 加速 **2.36x ~ 20.31x**，
  与 README 宣称的 "2 ~ 20x speedup" 一致。
- 编译耗时 **约 2.5 分钟**（70 核机器，`MAX_JOBS=18`）；整套 bf16 跑分耗时 **44 秒**。

---

## 1. 硬件与软件基线

本次实测环境（`nvidia-smi -i 0` / `python -c "import torch; ..."`）：

| 项目 | 值 |
|---|---|
| GPU | NVIDIA B300 SXM6 AC（`major=10, minor=3` → **sm_103**），274 GB 显存，148 SM，L2 = 126 MB |
| Driver | 580.159.04 |
| CUDA Toolkit (`nvcc`) | **13.3** (V13.3.33)，`CUDA_HOME=/usr/local/cuda` |
| Python | 3.12.14（conda env `muse_verify`） |
| PyTorch | 2.13.0+cu130 |
| Triton | 3.7.1（`tests/kernelkit/bench.py` 依赖 `import triton`） |
| setuptools / wheel / ninja | 80.10.2 / 有 / 1.13.0 |
| DeepSelect | `1.0.0+0f03b68.20260924.151248`（git rev `0f03b68` "Simplify & optimize NaN checking"） |
| cutlass 子模块 | `v4.4.0-35-gae6bccf3` |
| matplotlib | 3.11.2（仅出图需要，本次现装） |

### 1.1 硬性前提

`setup.py:116-127` 写死了两件必须满足的事：

1. **NVCC ≥ 12.9**，否则直接 `raise RuntimeError`。
2. **只为 `sm_100a` / `sm_103a` 生成代码**（`-gencode arch=compute_100a,code=sm_100a` 等；
   sm_80 / sm_90a 的 gencode 被注释掉了）。也就是说 **H100 / A100 / L20 上编译能过但跑不起来**，
   必须是 B200/B300 这一代。

开工前先自检：

```bash
nvcc --version | grep release                                   # 需要 >= 12.9
python -c "import torch; print(torch.cuda.get_device_capability(0))"  # 需要 (10,0) 或 (10,3)
python -c "import triton; print(triton.__version__)"            # 跑测试脚本需要
```

> 注：`torch.cuda.get_arch_list()` 里看不到 `sm_103a` 是正常的 —— DeepSelect 的扩展由本机
> `nvcc` 单独编译，与 PyTorch 预编译 wheel 的 arch 列表无关。实测 torch 2.13 的
> `_get_cuda_arch_flags()` 检测到 `extra_compile_args` 里已有 `arch=` 后会跳过追加自己的
> `-gencode`，所以 `setup.py` 的 flag 是最终生效的 flag。

---

## 2. 准备 conda 环境

### 2.1 复用已有环境（本次做法）

```bash
export https_proxy=http://proxy.ims.intel.com:911
export http_proxy=http://proxy.ims.intel.com:911
source /models/miniforge3/etc/profile.d/conda.sh
conda activate muse_verify
```

只要该环境里已有符合 §1.1 的 torch + CUDA，就可以直接跳到 §3。

### 2.2 新建干净环境（参考做法，本次未实测）

```bash
conda create -y -n deepselect python=3.12
conda activate deepselect
# 按本机 CUDA 大版本选 wheel 源，cu130 对应 torch 2.13.x
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install ninja triton matplotlib
```

代理只在需要联网时设置即可；`bdist_wheel` 这一步是纯本地编译，不联网。

---

## 3. 拉取子模块

```bash
cd /models/DeepSelect
git submodule update --init --recursive
git submodule status          # 期望看到 csrc/3rdparty/cutlass (v4.4.0-xx)
```

`.gitmodules` 里**只有 cutlass 一个子模块**（约 149 MB）。
`csrc/3rdparty/kerutils` 是直接随仓库带进来的头文件，不需要单独拉。

---

## 4. 编译并安装

### 4.1 推荐命令

```bash
export MAX_JOBS=18 NVCC_THREADS=4      # 18 × 4 ≈ 70，正好喂满 CPU
python setup.py bdist_wheel            # 产物在 dist/
pip install --no-deps --force-reinstall dist/deep_select-*.whl
```

- 需要 `--no-deps`：wheel 元数据里没有声明 torch 依赖，加 `--no-deps` 可避免 pip 顺手重装 torch。
- 需要 `--no-build-isolation`（如果用 `pip install .`）：`setup.py:8` 在构建期 `import tests.kernelkit`，
  而 kernelkit 又 `import torch`；构建隔离环境里没有 torch，会直接失败。
- 编译规模：`csrc/api.cpp` + 78 个 CUDA 实例化文件（bf16 40 / fp32 30 / cluster 8），
  每个都要编 `sm_100a` 和 `sm_103a` 两份。实测 **2 分 32 秒** 编完。
- 增量重编只需约 8 秒（`ninja: no work to do.`）。

### 4.2 坑：`pip install -v .` 会在最后一步失败

README 里的官方命令是 `pip install -v .`。本次实测：**编译全部成功、wheel 也生成了**，
但 pip 在收尾时拒绝安装：

```
Created wheel for deep_select: filename=deep_select-1.0.0+0f03b68.20260924.150904-...
WARNING: Built wheel for deep_select is invalid: Wheel has unexpected file name:
         expected '1.0.0+0f03b68.20260924.150856', got '1.0.0+0f03b68.20260924.150904'
Failed to build deep_select
error: failed-wheel-build-for-install
```

原因在 `setup.py:204`：版本号里拼了构建时刻的时间戳
（`version=f"{__version__}+{git_rev}.{datetime_rev}"`）。PEP 517 流程下 pip 会先查询一次元数据
（得到 `...150856`），再执行一次实际构建（得到 `...150904`），两次版本号不一致 → pip 认为 wheel 文件名不合法。
**这跟环境、CUDA、代码都没关系**，纯粹是版本号策略与 pip 的校验冲突。

规避方式二选一：

1. **本文采用的**：走 `python setup.py bdist_wheel` + `pip install dist/*.whl`，跳过 pip 的双次元数据查询。
2. 临时把 `setup.py:204` 的 `.{datetime_rev}` 去掉再 `pip install .`（会改动仓库文件，本文未采用）。

### 4.3 编译期自带的注册溢出检查

`setup.py:167-182` 用 `SpillCheckBuildExtension` 在链接完成后扫描产物，
发现寄存器溢出就 `raise RuntimeError("Register spilling detected. Build failed!")`。
本次实测输出：

```
Checking register spills in: build/lib.../deep_select/deep_select_cuda.cpython-312-x86_64-linux-gnu.so
No register spills detected.
```

如果想跳过这项检查（例如换 GPU 架构调试），设 `DEEP_SELECT_DISABLE_REG_SPILL_CHECK=1`。

### 4.4 安装校验

```bash
cd /tmp && python -c "
import torch, deep_select
print('module:', deep_select.__file__)
print('stride requirement (bytes):', deep_select.get_stride_requirement())
x = torch.randn(8, 204800, dtype=torch.bfloat16, device='cuda')
v, i = deep_select.topk(x, 1024, sorted_index=True, indices_type=torch.int32)
print('values', tuple(v.shape), v.dtype, '| indices', tuple(i.shape), i.dtype)
"
```

期望：`module` 指向 `.../site-packages/deep_select/__init__.py`，
`stride requirement (bytes): (1024, 32)`，shape/dtype 正确。

> **务必在 `/tmp` 之类的目录里跑**，别在仓库根目录跑。仓库根下有同名源码包
> `deep_select/`，`sys.path[0]` 会让它遮蔽已安装的扩展包（那里没有 `.so`），
> 报 `cannot import name 'deep_select_cuda'`。

**关于 stride requirement 的实际含义**：`(1024, 32)` 表示输入按行排布时，行首地址（严格说是
`stride(0) * itemsize`）必须是 **1024 字节**的整数倍，输出张量的行 stride 必须是 **32 字节**的整数倍。
bf16 是 2 字节，所以 `vocab_size` 需为 **512 的倍数**，fp32 则为 **256 的倍数**；不满足就要自行 padding
（`tests/lib.py:216-227` 的 `generate_testcase()` 就是这么对齐输入的）。`deep_select.topk()` 内部的
输出张量分配（`deep_select/interface.py:60-68`）会把输出的行 stride 向上取整到 32 字节的倍数，
所以当 `topk * itemsize` 不是 32 的倍数时（例如 `topk=2132` + int32）输出是**非连续**的，
`.view()` 会报错，用 `.reshape()` 或 `.contiguous()` 兜住；`topk=512/1024` 这类 2 的幂则恰好连续。

### 4.5 正确性自测脚本（以及为什么不能直接逐元素比 `torch.topk`）

```python
import torch, deep_select
torch.manual_seed(0)
b, V, k = 8, 204800, 1024
x = torch.randn(b, V, dtype=torch.bfloat16, device="cuda")
v, i = deep_select.topk(x, k, sorted=False, indices_type=torch.int32, return_value=True)
ref = torch.topk(x, k, dim=1)

print("sorted-value max abs diff:",
      (torch.sort(v.float(), descending=True, dim=1).values - ref.values.float()).abs().max().item())  # 0.0
print("gather consistency:", torch.equal(x.gather(1, i.long()), v))                                     # True
mask = torch.zeros_like(x, dtype=torch.bool); mask.scatter_(1, i.long(), True)
print("topk condition ok:", bool((v.float().amin(1) >=
      x.float().masked_fill(mask, float('-inf')).amax(1)).all()))                                        # True
```

初学时常犯的错：直接 `(v - ref.values).abs().max()`，本次实测这样得到 **2.73** 的"误差"，
看着像内核算错了。实际上：

- `sorted=False` 时 DeepSelect 的输出**不按 value 排序**，而 `torch.topk` 默认降序输出，逐元素比毫无意义；
- bf16 只有 8 位尾数，20 万量级的 `randn` 里存在大量**并列值**，索引集合本身就不唯一。

正确的比较方式是上面三行：值的多重集、`input[indices] == values` 的一致性、以及 topk 判定条件
（已选集合的最小值 ≥ 未选集合的最大值）。这三项本次实测全部为 `0.0 / True / True`。

---

## 5. 跑 Lightning Indexer 基准

### 5.1 命令

```bash
cd /models/DeepSelect
CUDA_VISIBLE_DEVICES=0 python -u tests/test.py --perf-only --dtype bf16 -rf 2>&1 | tee /tmp/perf_bf16.log
```

参数含义（`tests/test.py:164-250`）：

| 参数 | 作用 |
|---|---|
| `--perf-only` | 只跑 `num_runs > 0` 的性能用例，跳过那批数量庞大的正确性用例 |
| `--dtype bf16` | 只保留 bf16 用例，即 Lightning Indexer；否则还会带上 5 个 fp32 Sampler 用例 |
| `-rf` / `--run-to-finish` | 某条失败时不 `sys.exit(1)`，跑完全部再汇总 |
| `-nc` / `--no-cooldown` | 关掉每条性能用例之间的 `time.sleep(0.2)`；**复现数据时不要加**，让 GPU 保持时钟/温度稳定 |
| `-u`（python 层） | 不缓冲 stdout，方便 `tee` 后实时看进度 |

用例矩阵（`tests/test.py:225-242`）：

```
topk        ∈ {512, 1024}
batch_size  ∈ {6, 256, 512, 768, 4096}          # 6=RL rollout, 256/512/768=decode, 4096=prefill
vocab_size  ∈ {256, 1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576}
配置         sorted_value=False, sorted_index=False, return_value=False, out_idx=int32
输入分布     NormalFloatDistribution()（即 randn），num_runs=10
```

合计 2 × 5 × 9 = **90 条**。README 的 `assets/perf_bf16.png` 只画了其中一部分
（`topk=512`、`vocab ∈ [16K, 1M]`、`batch ∈ {6, 512, 4096}`；这个子集是按参考图的像素反读推断的，
官方并未在 README 正文里列出取样点）。

实测耗时：**44 秒**（15:13:41 → 15:14:25）。结尾应看到

```
All 90 cases passed!
```

—— 这句话很重要：`--perf-only` 并没有关掉正确性检查（`check_correctness` 默认 `True`），
所以这 90 条是**边测性能边验结果**的，跑通即说明这份数据是"算对了"的数据。

### 5.2 度量口径（务必理解，否则会误读数据）

`tests/test.py:137-160` + `tests/kernelkit/bench.py:110-164`：

- 计时用 **kineto/CUPTI 的 kernel 时间**，不是 wall time。DeepSelect 命中单个 kernel 时取该
  kernel 时间；`torch.topk` 是多 kernel 算子，取这些 kernel 的 **e2e 时间跨度**。
- 每次迭代前用一次 **8 GB 的 `zero_()`** 冲刷 L2（`bench.py:117-118`），所以是"冷 L2"带宽。
  B300 的 L2 是 126 MB，而这 8 GB 写入足以把它彻底挤空，因此**即使输入只有十几 MB 也不会命中
  L2**，测出来的是真实的 HBM 带宽而不是 L2 带宽。
- **TB/s = 有效字节数 / kernel 时间**，其中字节数 = 输入实际读取量 + 输出张量大小
  （`test.py:138`）。因为 `return_value=False`，这里不含 value 输出。
- `speedup = torch.topk 时间 / DeepSelect 时间`，直接打印在同一行末尾。
- 输入规模 < 带宽上限时，**耗时由 kernel launch 开销主导**：`batch=6` 各点都在 2~25 µs 量级，
  带宽只有 0.01 ~ 0.55 TB/s。判断是否进入带宽受限区，看 `topk` 行的绝对时间而不是 TB/s 更直观。

### 5.3 复现注意事项

- **独占一张卡**。本次 4 张 B300 里有 3 张被其他进程占着（1/2/3 号卡显存 18 GB~220 GB 不等），
  因此固定用 `CUDA_VISIBLE_DEVICES=0`。共享 GPU 会让 kernel 时间被抢占，带宽数字直接失真。
- 不要改 `num_runs`（默认 10）。样本少时大 case 波动尚可，小 case 的抖动会明显影响 speedup。
- 输入分布固定 `randn`。换成带大量并列值的分布（`UniformUIntDistribution(0x0, 0x20)` 等）会走
  不同的 pivot 路径，带宽会变，但这不是 README 图表的口径。

---

## 6. 实测数据

下表由 `parse_perf_bf16.py`（§7）从原始日志解析而来，并已与 `/tmp/perf_bf16.log` **逐条对拍，
90 行零差异**。"—" 表示 `vocab_size < topk` 时 `tests/test.py:147` 主动跳过 `torch.topk` 对比。

### 6.1 topk = 512（README 图表口径）

**DeepSelect 有效带宽 (TB/s)**

| batch \ vocab | 256 | 1K | 4K | 16K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| **6** | 0.01 | 0.01 | 0.01 | 0.03 | 0.07 | 0.10 | 0.12 | 0.34 | 0.55 |
| **256** | 0.35 | 0.25 | 0.51 | 0.93 | 1.80 | 2.49 | 3.18 | 3.85 | 4.46 |
| **512** | 0.65 | 0.31 | 0.59 | 1.07 | 2.02 | 2.70 | 3.36 | 3.97 | 4.55 |
| **768** | 0.80 | 0.33 | 0.63 | 1.16 | 2.12 | 2.79 | 3.44 | 4.04 | 4.59 |
| **4096** | 1.45 | 0.42 | 0.82 | 1.51 | 2.60 | 3.32 | 4.03 | 4.70 | **5.31** |

**torch.topk 有效带宽 (TB/s)**

| batch \ vocab | 256 | 1K | 4K | 16K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| **6** | — | 0.002 | 0.003 | 0.004 | 0.016 | 0.027 | 0.047 | 0.079 | 0.123 |
| **256** | — | 0.053 | 0.046 | 0.106 | 0.212 | 0.249 | 0.236 | 0.270 | 0.283 |
| **512** | — | 0.058 | 0.078 | 0.152 | 0.248 | 0.238 | 0.270 | 0.284 | 0.292 |
| **768** | — | 0.061 | 0.098 | 0.159 | 0.234 | 0.251 | 0.276 | 0.289 | 0.296 |
| **4096** | — | 0.171 | 0.231 | 0.242 | 0.284 | 0.292 | 0.293 | 0.261 | 0.262 |

**speedup**

| batch \ vocab | 256 | 1K | 4K | 16K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| **6** | — | 2.6x | 4.5x | 9.1x | 4.5x | 3.5x | 2.6x | 4.3x | 4.5x |
| **256** | — | 4.7x | 10.9x | 8.7x | 8.5x | 10.0x | 13.4x | 14.3x | 15.8x |
| **512** | — | 5.3x | 7.5x | 7.1x | 8.1x | 11.3x | 12.4x | 14.0x | 15.6x |
| **768** | — | 5.4x | 6.4x | 7.3x | 9.1x | 11.1x | 12.4x | 14.0x | 15.5x |
| **4096** | — | 2.5x | 3.5x | 6.2x | 9.1x | 11.4x | 13.7x | 18.1x | **20.3x** |

### 6.2 topk = 1024（同一命令附带产出）

**DeepSelect / speedup**

| batch \ vocab | 256 | 1K | 4K | 16K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| **6** | 0.01 / — | 0.02 / 5.3x | 0.02 / 4.5x | 0.04 / 9.5x | 0.07 / 4.0x | 0.09 / 3.1x | 0.11 / 2.4x | 0.33 / 4.2x | 0.50 / 4.0x |
| **256** | 0.65 / — | 0.86 / 10.5x | 0.58 / 10.4x | 0.95 / 8.5x | 1.71 / 8.0x | 2.24 / 9.1x | 2.93 / 12.3x | 3.54 / 13.2x | 4.11 / 14.5x |
| **512** | 1.14 / — | 1.51 / 16.7x | 0.67 / 7.1x | 1.12 / 7.1x | 1.89 / 7.6x | 2.37 / 9.9x | 3.06 / 11.3x | 3.67 / 12.9x | 4.19 / 14.4x |
| **768** | 1.38 / — | 1.89 / 19.8x | 0.72 / 6.1x | 1.21 / 7.2x | 1.96 / 8.4x | 2.48 / 9.9x | 3.11 / 11.3x | 3.72 / 12.8x | 4.22 / 14.3x |
| **4096** | 2.48 / — | 3.31 / 13.0x | 0.94 / 3.4x | 1.58 / 6.2x | 2.36 / 8.3x | 2.94 / 10.1x | 3.65 / 12.4x | 4.31 / 16.6x | 4.85 / 18.5x |

（`torch.topk` 基线与 topk=512 基本重合：除 `batch=6` 外，`vocab ≥ 64K` 时稳定落在
0.21 ~ 0.30 TB/s。完整数值见 `bf16_perf.csv`，其中 `t_tbps` 列。）

### 6.3 与 README 的对照

`assets/perf_bf16.png` 是**分组柱状图**，3 个子图（`batch=6 / 512 / 4096`），x 轴为
`vocab size`（16K、64K、128K、256K、512K、1M，类目轴），y 轴
`Effective bandwidth (TB/s)`，共享 0~7 轴，配色 `DeepSelect=#66CCFE`、`torch.topk=#ED0000`，
总标题 `bf16, topk=512: DeepSelect vs torch.topk`。

把该 PNG 按像素反读（y 轴约 59.6 px / TB/s，即 1 px ≈ 0.017 TB/s）后与本次实测逐点比较，
**最大偏差约 0.04 TB/s**（`batch=512, vocab=1M` 一处，约 2 px），其余各点在 1~2 px 之内 ——
也就是在读数分辨率之内吻合。例如 `batch=4096, vocab=1M`：本次 5.313 TB/s；
`batch=512, vocab=1M`：本次 4.546 TB/s。注意这个对比是基于对参考图的像素测量，
不是官方原始数据。

结论：README 的 2 ~ 20x 加速区间、以及"batch 越大 / vocab 越长，带宽越接近 HBM 上限"的趋势
在 B300 上完全复现。`batch=6` 那几列的绝对带宽偏低不是异常 —— 那里单条 kernel 只有
2 ~ 25 µs，固定开销占主导，README 的原图也刻意没有把 `vocab ≤ 4K` 的点画进去。

---

## 7. 把日志解析成表格 / CSV

`parse_perf_bf16.py`（保存为文件后 `python3 parse_perf_bf16.py <log> <csv>`）：

```python
"""Turn the stdout of `tests/test.py --perf-only --dtype bf16` into a tidy CSV."""
import re, csv, sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/perf_bf16.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/models/DeepSelect/bf16_perf.csv"

CASE  = re.compile(r"Running on TestParam\(batch_size=(\d+), vocab_size=(\d+), topk=(\d+),")
DS    = re.compile(r"^topk\s*:\s*([\d.]+) us,\s*([\d.]+) TB/s")
TORCH = re.compile(r"^torch\.topk\s*:\s*([\d.]+) us,\s*([\d.]+) TB/s.*speedup ([\d.]+)x")

rows, cur = [], None
with open(LOG) as f:
    for ln in f:
        if (m := CASE.search(ln)):
            cur = {"topk": int(m.group(3)), "batch_size": int(m.group(1)),
                   "vocab_size": int(m.group(2)),
                   "ds_us": None, "ds_tbps": None, "t_us": None, "t_tbps": None, "sp": None}
            rows.append(cur); continue
        if cur is None: continue
        if (m := DS.match(ln)):
            cur["ds_us"], cur["ds_tbps"] = float(m.group(1)), float(m.group(2))
        elif (m := TORCH.match(ln)):
            cur["t_us"], cur["t_tbps"], cur["sp"] = map(float, m.groups())

cols = ["topk","batch_size","vocab_size","ds_us","ds_tbps","t_us","t_tbps","sp"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
print(f"{len(rows)} cases -> {OUT}")
```

输出形如：

```
topk,batch_size,vocab_size,ds_us,ds_tbps,t_us,t_tbps,sp
512,6,256,1.92,0.008,,,
512,6,1024,4.154,0.006,10.684,0.002,2.57
512,4096,1048576,1618.455,5.313,32863.152,0.262,20.31
```

**建议做一次对拍**：解析脚本很容易因日志里混入的 `USDT:... profiler_start/stop` 之类的行而漏读。
本次实测的校验方法是重新解析日志并与 CSV 逐字段比较，结果应为
`csv rows: 90 / log cases: 90 / mismatches: 0`。至少要抽查 2~3 条，用
`grep -A6 "batch_size=6, vocab_size=65536, topk=512" /tmp/perf_bf16.log | grep "us,"`
直接看原始行。

> 教训：不要凭肉眼在混杂的日志里挑数字。本次第一版表格就是从未经对拍的日志摘录里出的，
> 把 4.47x 写成了 49.45x。所有进入文档的数字都应来自对拍后的 CSV。

---

## 8. 出图（对齐 README 版式）

`plot_perf_bf16.py`（`python3 plot_perf_bf16.py <csv> <png>`）。版式按 §6.3 描述复刻：
3 子图分组柱状图、类目 x 轴、共享 0~7 TB/s 轴、README 同款配色。

```python
import sys, csv, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "/models/DeepSelect/bf16_perf.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "/models/DeepSelect/perf_bf16_repro.png"

TOPK, BATCHES = 512, [6, 512, 4096]
VOCABS   = [16384, 65536, 131072, 262144, 524288, 1048576]
XLABELS  = ["16K", "64K", "128K", "256K", "512K", "1M"]

data = collections.defaultdict(dict)          # batch -> vocab -> (deepselect, torch.topk)
with open(CSV_PATH) as f:
    for r in csv.DictReader(f):
        if int(r["topk"]) != TOPK: continue
        data[int(r["batch_size"])][int(r["vocab_size"])] = (
            float(r["ds_tbps"]),
            float(r["t_tbps"]) if r["t_tbps"] not in ("", "None", None) else None,
        )

fig, axes = plt.subplots(1, len(BATCHES), figsize=(11.25, 3.3), sharey=True)
width = 0.38
for ax, b in zip(axes, BATCHES):
    x = range(len(VOCABS))
    ds = [data[b][v][0] for v in VOCABS]
    tt = [data[b][v][1] or 0.0 for v in VOCABS]   # vocab<topk 时无 torch 基线，画 0
    ax.bar([i - width/2 for i in x], ds, width, color="#66CCFE", label="DeepSelect")
    ax.bar([i + width/2 for i in x], tt, width, color="#ED0000", label="torch.topk")
    ax.set_xticks(list(x)); ax.set_xticklabels(XLABELS, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"batch={b}", fontsize=9)
    ax.set_xlabel("vocab size", fontsize=8)
    ax.set_ylim(0, 7); ax.set_yticks(range(8))
    ax.grid(True, axis="y", color="#DDDDDD", lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("Effective bandwidth (TB/s)", fontsize=8)
axes[0].legend(fontsize=8, loc="upper left", framealpha=1.0)
fig.suptitle(f"bf16, topk={TOPK}: DeepSelect vs torch.topk  (reproduced on NVIDIA B300)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT_PATH, dpi=200)
print("wrote", OUT_PATH)
```

> 代码里 `data[b][v][1] or 0.0` 只是防御性写法：`vocab < topk` 时 `tests/test.py:147` 会跳过
> `torch.topk` 对比，CSV 里对应 `t_tbps` 为空。本图取的是 `vocab ≥ 16K`，全部都有基线，
> 该分支不会触发；若你把 `VOCABS` 扩到 256，就要改成跳过画点或画 NaN，否则会看到一根 0 高的红柱。
> 想看全量 5 个 batch × 9 个 vocab 的折线趋势，把 `BATCHES` 换成 `[6,256,512,768,4096]`、
> `VOCABS` 换成 9 个点并改用 `ax.plot(...)` + `ax.set_xscale("log", base=2)`。

---

## 9. 本次产出物

| 路径 | 说明 | 是否被 git 跟踪 |
|---|---|---|
| `/tmp/deepselect_build.log` | 首次 `pip install -v .` 的完整日志（含失败原因） | — |
| `/tmp/deepselect_bdist.log` | `bdist_wheel` 日志（含 spill 检查结果） | — |
| `/tmp/perf_bf16.log` | 90 条用例的原始跑分日志，**唯一数据源** | — |
| `bf16_perf.csv` | 解析后的结构化数据（90 行，已与日志对拍） | 否，`.gitignore` 的 `*perf.csv` 已覆盖 |
| `perf_bf16_repro.png` | 复刻 `assets/perf_bf16.png` 的图表 | 未跟踪（`.gitignore` 不含 png） |
| `dist/deep_select-1.0.0+0f03b68.*.whl` | 构建产物 | 否，`.gitignore` 的 `dist/` |

`git status --porcelain` 本次只剩 `?? perf_bf16_repro.png` 一项（外加本文件），
仓库原有代码**未做任何修改**。

---

## 10. 故障排查速查

| 现象 | 原因 / 处理 |
|---|---|
| `sm100 compilation requires NVCC 12.9 or higher.` | `nvcc` 版本过低或 `CUDA_HOME` 指错 |
| `Unsupported platform: ...` | 非 CUDA 平台探测失败；用 `DEEP_SELECT_BUILD_TARGET_PLATFORM=CUDA` 强制（`setup.py:186-193`） |
| `Wheel has unexpected file name` | §4.2 的时间戳版本号问题，改用 `bdist_wheel` + 安装 whl |
| `Register spilling detected. Build failed!` | 编译器优化退化，换 `nvcc` 版本；调试时可 `DEEP_SELECT_DISABLE_REG_SPILL_CHECK=1` |
| `cannot import name 'deep_select_cuda'` | 在仓库根目录跑了 python，源码包遮蔽了安装包（见 §4.4） |
| 运行期无可用 kernel / 崩溃 | GPU 不是 sm_100a / sm_103a，见 §1.1 |
| 带宽明显低于本文 | 与其他进程共享 GPU；或加了 `-nc` 导致热降频；或 `vocab_size` 不是 512 的倍数导致输入未对齐 |
| 小 batch 带宽只有 0.0x TB/s | 正常，launch 开销主导，工作集本身只有 MB 级，看 µs 而非 TB/s |

---

## 11. 参考

- 官方 README：[README.md](../README.md)（Supported Cases / Performance / Installation / Usage）
- 算法与实现分析：[DeepSelect-deep-dive.md](DeepSelect-deep-dive.md) |
  [中文](DeepSelect-deep-dive.zh.md)
- 用例矩阵与度量口径：`tests/test.py:137-160`（计时与带宽）、`tests/test.py:225-242`（性能用例定义）、
  `tests/kernelkit/bench.py:110-164`（kineto 计时与 L2 冲刷）
