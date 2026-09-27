"""Plot bf16 Lightning Indexer bandwidth in the README layout (3 subplots
batch=6/512/4096, categorical vocab axis 16K..1M, shared 0-7 TB/s y-axis),
with funnel-topk (pre- and post-optimization) added alongside DeepSelect and
torch.topk.

Bars per vocab group (4): DeepSelect, funnel-topk best-of-3-modes BEFORE
optimization, funnel-topk best-of-3-modes AFTER optimization, torch.topk.
"best" = lowest latency among standard/fast/turbo for that config; the bar
height is the corresponding effective bandwidth.

Data sources (both produced by benchmarks/compare_ds_vs_funnel.py):
  compare_ds_funnel_20260926.json            — baseline (pre-optimization)
  compare_ds_funnel_after_opt_20260926.json  — after guard + segmented path

Usage:
    python3 benchmarks/plot_perf_bf16_with_funnel.py [before.json] [after.json] [out.png]

Reference: docs/optimize-funnel-topk-20260926.zh.md
"""
import sys, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
BEFORE = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "compare_ds_funnel_20260926.json")
AFTER  = sys.argv[2] if len(sys.argv) > 2 else str(REPO_ROOT / "compare_ds_funnel_after_opt_20260926.json")
OUT    = sys.argv[3] if len(sys.argv) > 3 else str(REPO_ROOT / "perf_bf16_with_funnel.png")

BATCHES = [6, 512, 4096]
VOCABS  = [16384, 65536, 131072, 262144, 524288, 1048576]
XLABELS = ["16K", "64K", "128K", "256K", "512K", "1M"]
MODES   = ["standard", "fast", "turbo"]

def load(path):
    d = {}
    for r in json.load(open(path)):
        d[(r["batch"], r["vocab"])] = r
    return d

before, after = load(BEFORE), load(AFTER)

def ds_tbps(row):  return float(row["deep_select"][1])
def tt_tbps(row):  return float(row["torch"][1])
def funnel_best(row):
    """(tbps, mode) of the fastest funnel mode for this config."""
    best_mode = min(MODES, key=lambda m: float(row[m][0]))
    return float(row[best_mode][1]), best_mode

fig, axes = plt.subplots(1, len(BATCHES), figsize=(12.8, 3.5), sharey=True)
COLORS = [("#66CCFE", "DeepSelect"),
          ("#E69F00", "funnel-topk (before opt)"),
          ("#009E73", "funnel-topk (after opt)"),
          ("#ED0000", "torch.topk")]
width = 0.2
for ax, b in zip(axes, BATCHES):
    x = range(len(VOCABS))
    series = [
        [ds_tbps(after[(b, v)]) for v in VOCABS],
        [funnel_best(before[(b, v)])[0] for v in VOCABS],
        [funnel_best(after[(b, v)])[0] for v in VOCABS],
        [tt_tbps(after[(b, v)]) for v in VOCABS],
    ]
    for k, (vals, (color, label)) in enumerate(zip(series, COLORS)):
        ax.bar([i + (k - 1.5) * width for i in x], vals, width, color=color, label=label)
    ax.set_xticks(list(x)); ax.set_xticklabels(XLABELS, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"batch={b}", fontsize=9)
    ax.set_xlabel("vocab size", fontsize=8)
    ax.set_ylim(0, 7); ax.set_yticks(range(8))
    ax.grid(True, axis="y", color="#DDDDDD", lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(labelsize=8)
axes[0].set_ylabel("Effective bandwidth (TB/s)", fontsize=8)
axes[0].legend(fontsize=7.5, loc="upper left", framealpha=1.0)
fig.suptitle("bf16, topk=512: DeepSelect vs funnel-topk (before/after opt) vs torch.topk  (NVIDIA B300)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(OUT, dpi=200)
print("wrote", OUT)

# also print the winning mode per config for the caption
for b in BATCHES:
    parts = []
    for v in VOCABS:
        m0 = funnel_best(before[(b, v)])[1]
        m1 = funnel_best(after[(b, v)])[1]
        parts.append(f"{v}:{m0}->{m1}")
    print(f"batch={b} best-mode before->after: " + ", ".join(parts))
