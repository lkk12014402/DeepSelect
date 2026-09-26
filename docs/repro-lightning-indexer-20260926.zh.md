# Lightning Indexer Scenario 复现记录（2026-09-26，第二次独立复现）

本文记录 2026-09-26 在 **NVIDIA B300** 上复现 README 中
[Lightning Indexer Scenario](../README.md#lightning-indexer-scenario) 性能数据
（即 `assets/perf_bf16.png`）的完整过程：每一步的决策思路、命令、脚本、数据，
以及遇到的问题与解决办法。

本次是该仓库的**第二次独立复现**。第一次复现（2026-09-24）的记录见
[repro-lightning-indexer.zh.md](repro-lightning-indexer.zh.md)，其中已包含环境原理、
度量口径、故障排查等通用内容；本文聚焦本次复现特有的事实，并给出
**两次复现数据的逐点对比**（可重复性证据）。两次复现使用的是同一代码版本
（git rev `0f03b68`）、同一台机器、同一 conda 环境。

文中所有版本号、耗时、数据均为本次实测结果，非估算。

---

## 0. TL;DR

```bash
source /models/miniforge3/etc/profile.d/conda.sh
conda activate muse_verify            # 可用，无需新建环境

cd /models/DeepSelect
MAX_JOBS=18 NVCC_THREADS=4 python setup.py bdist_wheel
pip install --no-deps --force-reinstall dist/deep_select-*.whl

CUDA_VISIBLE_DEVICES=0 python -u tests/test.py --perf-only --dtype bf16 -rf \
    | tee /tmp/perf_bf16_20260926.log

python3 benchmarks/parse_perf_bf16.py /tmp/perf_bf16_20260926.log ./bf16_perf_20260926.csv
python3 benchmarks/plot_perf_bf16.py ./bf16_perf_20260926.csv ./perf_bf16_repro_20260926.png
```

本次实测结论：

- 90 个 bf16 性能用例**全部通过正确性检查**（`All 90 cases passed!`）。
- DeepSelect 有效带宽 **0.006 ~ 5.314 TB/s**，相对 `torch.topk` 加速 **2.36x ~ 20.31x**，
  与 README 宣称的 "2 ~ 20x speedup" 一致。
- 与 2026-09-24 的首次复现逐点对比：DeepSelect 带宽最大相对偏差 **7.69%**
  （出现在 `batch=6, vocab=4K` 的 µs 级小 case，绝对差仅 0.001 TB/s），
  speedup 最大相对偏差 **3.03%** —— 数据跨日可重复。
- 跑分全程 **37.5 秒**；构建因 ninja 缓存命中秒级完成（`ninja: no work to do.`）。

---

## 1. 复现思路（每一步的决策记录）

1. **先读 README 定位复现目标。** `README.md:14-24` 定义 Lightning Indexer Scenario
   （bf16、topk ≤ 4096、任意 batch/vocab），`README.md:36-45` 指明度量方式为
   `python3 tests/test.py --perf-only`，指标是有效内存带宽（TB/s），图表为
   `assets/perf_bf16.png`（bf16, topk=512, 按 batch 分子图）。
2. **读测试入口确定用例矩阵。** `tests/test.py:225-237`：Lightning Indexer 性能用例为
   `topk ∈ {512, 1024}` × `batch ∈ {6, 256, 512, 768, 4096}` ×
   `vocab ∈ {256, 1K, 4K, 16K, 64K, 128K, 256K, 512K, 1M}`，共 2×5×9 = **90 条**，
   配置 `sorted_value=False, sorted_index=False, return_value=False`，int32 索引，
   输入 `randn` 分布。计时口径见 `tests/test.py:137-160`（kineto kernel 时间；
   torch.topk 取多 kernel 的 e2e 跨度；每次迭代前 8 GB `zero_()` 冲 L2）。
3. **发现仓库已有 09-24 的复现产物**（`docs/repro-lightning-indexer.zh.md`、
   `bf16_perf.csv`、`perf_bf16_repro.png`，均未被 git 跟踪）。因此本次定位调整为：
   不是"从零开荒"，而是**第二次独立复现**——重走完整流程，并新增两次数据的
   逐点对比作为可重复性证据。
4. **检查 `muse_verify` 环境**（用户指定）：逐项验证 torch / CUDA / nvcc / triton /
   matplotlib / 已装 deep_select → 全部满足 `setup.py:116-127` 的硬性前提
   （NVCC ≥ 12.9、sm_100a/sm_103a），**可用，无需新建环境**（§2）。
5. **构建安装**：git 工作区干净（HEAD=`0f03b68`，与已装 wheel 版本一致），
   仍完整重走 `bdist_wheel` + 强制重装，验证构建链路在当前环境依然成立（§3）。
6. **跑分**：与首次复现同口径（`CUDA_VISIBLE_DEVICES=0`、不加 `-nc`），
   本次 4 张 B300 全部空闲（首次复现时有 3 张被占用，只能固定 0 号卡）（§4）。
7. **解析与对拍**：日志 → CSV，抽样与原始日志逐字段对拍（§5.1）。
8. **出图与对比**：复刻 README 版式出图，与 `assets/perf_bf16.png` 及
   09-24 CSV 双重对比（§6、§7）。

## 2. 环境检查：`muse_verify` 是否可用？

**结论：可用。** 逐项实测输出：

```bash
$ source /models/miniforge3/etc/profile.d/conda.sh && conda activate muse_verify
$ python --version                  # Python 3.12.14
$ python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
torch 2.13.0+cu130  True  (10, 3)   # sm_103 → 满足 sm_100a/sm_103a 要求
$ python -c "import triton; print(triton.__version__)"       # 3.7.1（tests/kernelkit/bench.py 依赖）
$ python -c "import matplotlib; print(matplotlib.__version__)"  # 3.11.2（出图依赖）
$ pip show deep_select              # 1.0.0+0f03b68.20260924.151248（首次复现时安装）
$ nvcc --version                    # release 13.3, V13.3.33（≥ 12.9 ✓）
$ echo $CUDA_HOME                   # /usr/local/cuda
$ git submodule status              # cutlass ae6bccf3 (v4.4.0-35-gae6bccf3) 已就位
$ git status --porcelain            # 干净（仅 09-24 复现留下的未跟踪文档/png）
$ nvidia-smi                        # 4 × NVIDIA B300 SXM6 AC（275 GB），本次全部空闲
$ nproc                             # 70
```

硬件/软件基线与首次复现完全一致：GPU NVIDIA B300 SXM6 AC（sm_103，148 SM，
L2 = 126 MB），驱动 580.159.04，CUDA Toolkit 13.3。

**关于"新建环境"**：因 `muse_verify` 可用，本次未新建。若需干净环境，参考命令
（与首次复现文档 §2.2 相同，本次未实测）：

```bash
conda create -y -n deepselect python=3.12 && conda activate deepselect
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install ninja triton matplotlib
```

### 2.1 踩坑：在仓库根目录 `import deep_select` 会失败

检查环境时先在 `/models/DeepSelect` 下执行 `python -c "import deep_select"`，报：

```
ImportError: cannot import name 'deep_select_cuda' from partially initialized module 'deep_select'
```

**原因**：仓库根目录下有同名源码包 `deep_select/`（纯 Python，无 `.so`），
`sys.path[0]` 使其遮蔽 site-packages 里已安装的完整包。这正是首次复现文档 §4.4
记录过的坑，本次再次遇到。**解决**：`cd /tmp` 后再 import 即正常：

```
module: /models/miniforge3/envs/muse_verify/lib/python3.12/site-packages/deep_select/__init__.py
stride requirement (bytes): (1024, 32)
sorted-value max abs diff: 0.0      # 与 torch.topk 的值多重集一致
gather consistency: True            # input[indices] == values
```

## 3. 构建与安装

```bash
cd /models/DeepSelect
MAX_JOBS=18 NVCC_THREADS=4 python setup.py bdist_wheel 2>&1 | tee /tmp/deepselect_bdist_20260926.log
pip install --no-deps --force-reinstall dist/deep_select-1.0.0+0f03b68.20260926.153723-*.whl
```

实测要点：

- 日志确认 `Build target: Platform.CUDA`、`Compiling using NVCC 13.3`，
  链接后寄存器溢出检查通过：`No register spills detected.`
- 因代码自首次构建后无任何改动（git HEAD 仍为 `0f03b68`），ninja 命中缓存，
  输出 `ninja: no work to do.`，**秒级完成**（首次全量构建为 2 分 32 秒）。
- 产物 `dist/deep_select-1.0.0+0f03b68.20260926.153723-cp312-cp312-linux_x86_64.whl`，
  强制重装后替换掉 09-24 的旧 wheel。
- 仍走 `bdist_wheel` 而非 README 的 `pip install -v .`：`setup.py:204` 的版本号
  含构建时刻时间戳，会让 pip 的 PEP 517 双次元数据查询自相矛盾而报错
  （详见首次复现文档 §4.2，该问题与代码/环境无关，本次未复测）。

## 4. 跑 Lightning Indexer 基准

```bash
cd /models/DeepSelect
CUDA_VISIBLE_DEVICES=0 python -u tests/test.py --perf-only --dtype bf16 -rf 2>&1 | tee /tmp/perf_bf16_20260926.log
```

- `--perf-only`：只跑 `num_runs > 0` 的性能用例；`--dtype bf16`：仅 Lightning Indexer
  （排除 5 个 fp32 Sampler 用例）；`-rf`：失败也跑完再汇总；不加 `-nc`，保留每条
  用例间 0.2 s 冷却，让 GPU 时钟/温度稳定。
- 用例矩阵 90 条（§1 第 2 点）。注意 `--perf-only` **不关**正确性检查
  （`check_correctness` 默认 `True`），跑通即结果正确。
- 本次 4 张卡全空闲，仍固定 `CUDA_VISIBLE_DEVICES=0` 与首次复现口径一致。
- 实测耗时 **37.5 秒**（15:38:06.7 → 15:38:44.2，按日志文件 birth/mtime），
  与首次的 44 秒相当。结尾输出 `All 90 cases passed!`。

## 5. 数据

### 5.1 解析脚本与对拍

解析脚本 [`benchmarks/parse_perf_bf16.py`](../benchmarks/parse_perf_bf16.py)
（正则提取 `Running on TestParam(...)` / `topk : ... us, ... TB/s` /
`torch.topk : ... (speedup ...x)` 三类行）：

```bash
python3 benchmarks/parse_perf_bf16.py /tmp/perf_bf16_20260926.log /models/DeepSelect/bf16_perf_20260926.csv
# 输出：90 cases -> /models/DeepSelect/bf16_perf_20260926.csv
```

抽取 3 条与原始日志逐字段对拍，全部一致：

```
日志: topk : 11.408 us, 0.070 TB/s / torch.topk : 50.917 us, 0.016 TB/s (speedup 4.46x)
CSV : 512,6,65536,11.408,0.07,50.917,0.016,4.46                     ✓

日志: topk : 1618.059 us, 5.314 TB/s / torch.topk : 32863.470 us, 0.262 TB/s (speedup 20.31x)
CSV : 512,4096,1048576,1618.059,5.314,32863.47,0.262,20.31           ✓

日志: topk : 256.875 us, 4.188 TB/s / torch.topk : 3684.540 us, 0.292 TB/s (speedup 14.34x)
CSV : 1024,512,1048576,256.875,4.188,3684.54,0.292,14.34             ✓
```

### 5.2 topk = 512（README 图表口径）

**DeepSelect 有效带宽 (TB/s)**

| batch \ vocab | 256 | 1K | 4K | 16K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| **6** | 0.01 | 0.01 | 0.01 | 0.03 | 0.07 | 0.10 | 0.12 | 0.34 | 0.55 |
| **256** | 0.36 | 0.24 | 0.51 | 0.94 | 1.80 | 2.49 | 3.18 | 3.84 | 4.47 |
| **512** | 0.64 | 0.30 | 0.59 | 1.07 | 2.01 | 2.70 | 3.36 | 3.97 | 4.55 |
| **768** | 0.80 | 0.33 | 0.63 | 1.16 | 2.13 | 2.79 | 3.43 | 4.04 | 4.59 |
| **4096** | 1.45 | 0.42 | 0.82 | 1.51 | 2.59 | 3.32 | 4.03 | 4.70 | **5.31** |

**torch.topk 有效带宽 (TB/s)**

| batch \ vocab | 256 | 1K | 4K | 16K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| **6** | — | 0.002 | 0.003 | 0.004 | 0.016 | 0.027 | 0.047 | 0.079 | 0.123 |
| **256** | — | 0.053 | 0.047 | 0.107 | 0.212 | 0.248 | 0.237 | 0.270 | 0.283 |
| **512** | — | 0.058 | 0.078 | 0.151 | 0.248 | 0.238 | 0.270 | 0.283 | 0.292 |
| **768** | — | 0.061 | 0.098 | 0.160 | 0.233 | 0.251 | 0.276 | 0.289 | 0.296 |
| **4096** | — | 0.171 | 0.231 | 0.241 | 0.284 | 0.292 | 0.293 | 0.261 | 0.262 |

**speedup（DeepSelect vs torch.topk）**

| batch \ vocab | 256 | 1K | 4K | 16K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| **6** | — | 2.6x | 4.6x | 9.2x | 4.5x | 3.5x | 2.6x | 4.3x | 4.5x |
| **256** | — | 4.6x | 10.8x | 8.8x | 8.5x | 10.0x | 13.5x | 14.2x | 15.8x |
| **512** | — | 5.2x | 7.5x | 7.1x | 8.1x | 11.3x | 12.4x | 14.0x | 15.6x |
| **768** | — | 5.4x | 6.4x | 7.3x | 9.1x | 11.1x | 12.4x | 14.0x | 15.5x |
| **4096** | — | 2.5x | 3.5x | 6.3x | 9.1x | 11.4x | 13.7x | 18.0x | **20.3x** |

（"—"：`vocab_size < topk` 时 `tests/test.py:147` 主动跳过 torch.topk 对比。）

### 5.3 topk = 1024（同一命令附带产出）

格式为 `DeepSelect TB/s / speedup`：

| batch \ vocab | 256 | 1K | 4K | 16K | 64K | 128K | 256K | 512K | 1M |
|---|---|---|---|---|---|---|---|---|---|
| **6** | 0.01 / — | 0.02 / 5.1x | 0.02 / 4.5x | 0.04 / 9.5x | 0.07 / 4.0x | 0.09 / 3.1x | 0.11 / 2.4x | 0.33 / 4.2x | 0.50 / 4.0x |
| **256** | 0.65 / — | 0.86 / 10.5x | 0.58 / 10.3x | 0.96 / 8.6x | 1.70 / 8.0x | 2.25 / 9.1x | 2.92 / 12.3x | 3.55 / 13.2x | 4.11 / 14.4x |
| **512** | 1.14 / — | 1.51 / 16.6x | 0.67 / 7.2x | 1.12 / 7.1x | 1.88 / 7.6x | 2.38 / 9.9x | 3.06 / 11.3x | 3.67 / 12.9x | 4.19 / 14.3x |
| **768** | 1.40 / — | 1.87 / 19.6x | 0.72 / 6.0x | 1.21 / 7.2x | 1.97 / 8.4x | 2.49 / 9.9x | 3.12 / 11.3x | 3.72 / 12.8x | 4.21 / 14.3x |
| **4096** | 2.49 / — | 3.31 / 12.9x | 0.94 / 3.4x | 1.58 / 6.2x | 2.36 / 8.2x | 2.94 / 10.1x | 3.65 / 12.4x | 4.31 / 16.6x | 4.84 / 18.5x |

完整 8 列数值（含 µs 级耗时）见 `bf16_perf_20260926.csv`。

## 6. 与官方图 `assets/perf_bf16.png` 的对比

官方图：3 子图分组柱状图（batch=6/512/4096），x 轴类目 16K~1M，共享 0~7 TB/s 轴，
配色 DeepSelect=#66CCFE / torch.topk=#ED0000。本次复现图
`perf_bf16_repro_20260926.png` 用相同版式绘制（脚本见 §7），逐点对比：

- 首次复现（09-24）已对官方 PNG 做过像素反读（y 轴约 59.6 px/TB/s），实测与官方图
  最大偏差约 0.04 TB/s（读数分辨率之内吻合），见首次文档 §6.3。
- 本次数据与首次数据逐点几乎相同（§8），因此与官方图的吻合结论原样成立。
  两个代表性读数：官方图 `batch=4096, vocab=1M` ≈ 5.3 TB/s，本次 **5.314**；
  官方图 `batch=512, vocab=1M` ≈ 4.6 TB/s，本次 **4.547**。
- 两图目视对比（同版式、同配色、柱高一致）确认无异常点。

## 7. 出图脚本

[`benchmarks/plot_perf_bf16.py`](../benchmarks/plot_perf_bf16.py)（matplotlib，Agg 后端；
与首次复现文档 §8 相同）：

```bash
python3 benchmarks/plot_perf_bf16.py ./bf16_perf_20260926.csv ./perf_bf16_repro_20260926.png
```

要点：`topk=512`、`batch ∈ {6, 512, 4096}`、`vocab ∈ {16K, 64K, 128K, 256K, 512K, 1M}`，
分组柱状图，共享 0~7 TB/s y 轴，配色 `#66CCFE` / `#ED0000`。

## 8. 与 2026-09-24 首次复现的逐点对比（可重复性）

对两次 CSV 的全部 90 条按 `(topk, batch, vocab)` 对齐比较：

```
cases compared: 90
DeepSelect TB/s max rel diff vs 09-24: 7.69%   at (topk=512, batch=6, vocab=4096): 0.013 -> 0.014
speedup       max rel diff vs 09-24: 3.03%   at (topk=1024, batch=6, vocab=1024): 5.28 -> 5.12
this run: DeepSelect 0.006 ~ 5.314 TB/s, speedup 2.36x ~ 20.31x
```

- 最大偏差出现在 `batch=6` 的 µs 级小 case：kernel 只有 2~25 µs，耗时由 launch
  开销主导，0.001 TB/s 的绝对抖动就被放大成 7.69% 的相对偏差，属正常测量噪声。
- 带宽受限区（batch 大、vocab 大）两次几乎逐位一致，例如
  `batch=4096, vocab=1M, topk=512`：5.313 → 5.314 TB/s，speedup 20.31x → 20.31x。
- 对比脚本核心：`csv.DictReader` 读两份 CSV → 按 key 对齐 → 逐字段算相对偏差。

## 9. 遇到的问题与解决办法（本次实录）

| 问题 | 原因 | 解决 |
|---|---|---|
| 仓库根目录下 `import deep_select` 报 `cannot import name 'deep_select_cuda'` | 源码包 `deep_select/`（无 `.so`）遮蔽 site-packages 已装包；首次文档 §4.4 已记录 | `cd /tmp` 后再 import；跑测试无此问题（`tests/` 下 `sys.path[0]` 是 tests 目录） |
| 写 `/tmp/parse_perf_bf16.py` 时提示文件已存在 | 首次复现遗留的同名脚本 | 本次所有脚本/日志/CSV/图一律加 `_20260926` 后缀，保留两次产物互相对照 |
| 担心共享 GPU 导致带宽失真 | 首次复现时 1/2/3 号卡被其他进程占用 | 本次实测 4 张卡全空闲（`nvidia-smi` 显存 0 MiB），仍固定 `CUDA_VISIBLE_DEVICES=0` 保持口径一致 |

除此之外全程无新问题：构建无寄存器溢出、90/90 用例通过、解析 90 行零缺失。
首次复现记录的 `pip install -v .` 版本号问题（`setup.py:204` 时间戳）本次未复测，
直接沿用了 `bdist_wheel` + `pip install dist/*.whl` 的规避路径。

## 10. 产出物清单

| 路径 | 说明 |
|---|---|
| `/tmp/perf_bf16_20260926.log` | 90 条用例原始跑分日志，**唯一数据源** |
| `/tmp/deepselect_bdist_20260926.log` | 构建日志（含 spill 检查：`No register spills detected.`） |
| [`benchmarks/parse_perf_bf16.py`](../benchmarks/parse_perf_bf16.py)、[`benchmarks/plot_perf_bf16.py`](../benchmarks/plot_perf_bf16.py) | 解析 / 出图脚本（已收入仓库） |
| `bf16_perf_20260926.csv` | 结构化数据（90 行 × 8 列，已对拍） |
| `perf_bf16_repro_20260926.png` | 复刻 README 版式的图 |
| `dist/deep_select-1.0.0+0f03b68.20260926.153723-*.whl` | 本次构建产物 |

以上 CSV/PNG/dist 均被 `.gitignore` 或未跟踪规则覆盖，仓库原有代码零改动
（`git status --porcelain` 仅新增本文档及上述复现产物）。

## 11. 结论

1. `muse_verify` 环境**可直接使用**（torch 2.13.0+cu130 / CUDA 13.3 / sm_103 /
   triton / matplotlib 齐备），无需新建环境。
2. README 的 Lightning Indexer 数据在 B300 上**完整复现**：90/90 用例通过，
   带宽 0.006 ~ 5.314 TB/s，加速比 2.36x ~ 20.31x，落在官方宣称的 2 ~ 20x 区间。
3. 与首次复现（09-24）逐点对比偏差 ≤ 7.69%（且最大偏差为 launch 开销主导的
   µs 级小 case），带宽受限区两次结果几乎逐位一致 —— **数据跨日可重复**。
