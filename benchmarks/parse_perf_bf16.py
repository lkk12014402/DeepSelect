"""Turn the stdout of `tests/test.py --perf-only --dtype bf16` into a tidy CSV.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -u tests/test.py --perf-only --dtype bf16 -rf | tee /tmp/perf_bf16.log
    python3 benchmarks/parse_perf_bf16.py /tmp/perf_bf16.log bf16_perf.csv

Reference: docs/repro-lightning-indexer-20260926.zh.md
"""
import re, csv, sys, os

LOG = sys.argv[1] if len(sys.argv) > 1 else "/tmp/perf_bf16.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "bf16_perf.csv"

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
print(f"{len(rows)} cases -> {os.path.abspath(OUT)}")
