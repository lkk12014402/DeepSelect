"""Round-1 guard optimization check: recall must be IDENTICAL to the
pre-optimization build (the guard is provably lossless), speed should improve.

Compares against the baseline numbers recorded in compare_ds_funnel_20260926.json.
"""
import json, sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
import kernelkit as kk
from funnel_topk import topk as funnel_topk_fn

K = 512
BATCHES = [6, 256, 512, 4096]
VOCABS = [16384, 65536, 131072, 262144, 524288, 1048576]
NUM_RUNS = 10

base = {(r["batch"], r["vocab"]): r for r in json.load(open(REPO_ROOT / "compare_ds_funnel_20260926.json"))}

_EXCLUDE_PREFIX = ("cuda", "Memset")
_EXCLUDE_SUBSTR = ("FillFunctor", "profiler_range")

def bench_call(fn):
    res = kk.bench(fn, NUM_RUNS)
    names = [n for n in res.get_kernel_names()
             if not n.startswith(_EXCLUDE_PREFIX) and not any(e in n for e in _EXCLUDE_SUBSTR)]
    return res.get_e2e_time(names) * 1e6  # us

def index_recall(idx_ref, idx):
    return (idx_ref.unsqueeze(-1) == idx.unsqueeze(-2)).any(dim=-1).float().mean().item() * 100.0

print(f"{'cfg':>18} | {'mode':>9} | {'old us':>9} {'new us':>9} {'speedup':>7} | {'rc old':>6} {'rc new':>6}")
max_rc_delta = 0.0
for b in BATCHES:
    for v in VOCABS:
        torch.manual_seed(0)
        x = torch.randn(b, v, dtype=torch.bfloat16, device="cuda")
        idx_ref = torch.topk(x, K, dim=1, sorted=True).indices
        for mode in ["standard", "fast", "turbo"]:
            old = base[(b, v)][mode]
            old_us, old_rc = float(old[0]), float(old[2])
            new_us = bench_call(lambda m=mode: funnel_topk_fn(x, K, mode=m, sorted=False))
            _, fi = funnel_topk_fn(x, K, mode=mode, sorted=False)
            new_rc = index_recall(idx_ref, fi)
            max_rc_delta = max(max_rc_delta, abs(new_rc - old_rc))
            print(f"B={b:5d} V={v:8d} | {mode:>9} | {old_us:9.1f} {new_us:9.1f} {old_us/new_us:6.2f}x | {old_rc:5.1f}% {new_rc:5.1f}%", flush=True)
        del x, idx_ref
        torch.cuda.empty_cache()
print(f"\nmax |Δrecall| vs pre-optimization build: {max_rc_delta:.3f} pp (expect ~0: guard is lossless)")
