"""Plot the bf16 Lightning Indexer CSV (from parse_perf_bf16.py) in the same
layout as README's assets/perf_bf16.png: 3 subplots (batch=6/512/4096),
categorical vocab axis 16K..1M, shared 0-7 TB/s y-axis.

Usage:
    python3 benchmarks/plot_perf_bf16.py bf16_perf.csv perf_bf16_repro.png

Reference: docs/repro-lightning-indexer-20260926.zh.md
"""
import sys, csv, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "bf16_perf.csv"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "perf_bf16_repro.png"

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
    tt = [data[b][v][1] or 0.0 for v in VOCABS]   # no torch baseline when vocab<topk; not in VOCABS anyway
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
