"""Round-2 check: segmented clean-funnel path (small-batch parallelism fix).

Verifies for the segmented path (active when B < 2*SMs, bf16):
  - sanity checks: index range / uniqueness / gather consistency / topk condition
  - recall@K vs torch.topk (expect >= single-block turbo/fast)
  - latency vs the recorded pre-segmentation numbers

Usage: python3 benchmarks/verify_round2_segmented.py
"""
import json, sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))
import kernelkit as kk
from funnel_topk import topk as funnel_topk_fn
from funnel_topk import _kernels

K = 512
BATCHES = [6, 256]          # segmented path active (B < 2*148)
VOCABS = [16384, 65536, 131072, 262144, 524288, 1048576]
NUM_RUNS = 10

base = {(r["batch"], r["vocab"]): r for r in json.load(open(REPO_ROOT / "compare_ds_funnel_20260926.json"))}

_EXCLUDE_PREFIX = ("cuda", "Memset")
_EXCLUDE_SUBSTR = ("FillFunctor", "profiler_range")

def bench_call(fn):
    res = kk.bench(fn, NUM_RUNS)
    names = [n for n in res.get_kernel_names()
             if not n.startswith(_EXCLUDE_PREFIX) and not any(e in n for e in _EXCLUDE_SUBSTR)]
    return res.get_e2e_time(names) * 1e6

def index_recall(idx_ref, idx):
    return (idx_ref.unsqueeze(-1) == idx.unsqueeze(-2)).any(dim=-1).float().mean().item() * 100.0

def sanity(x, values, indices):
    B, N = x.shape
    idx = indices.long()
    ok_range = bool(((idx >= 0) & (idx < N)).all())
    srt = idx.sort(dim=1).values
    ok_uniq = bool((srt[:, 1:] != srt[:, :-1]).all())
    ok_gather = bool(torch.equal(x.gather(1, idx), values)) if values is not None else None
    return ok_range, ok_uniq, ok_gather

for b in BATCHES:
    for v in VOCABS:
        torch.manual_seed(0)
        x = torch.randn(b, v, dtype=torch.bfloat16, device="cuda")
        idx_ref = torch.topk(x, K, dim=1, sorted=True).indices
        for mode in ["fast", "turbo"]:
            old = base[(b, v)][mode]
            old_us, old_rc = float(old[0]), float(old[2])
            us = bench_call(lambda m=mode: funnel_topk_fn(x, K, mode=m, sorted=False))
            fv, fi = funnel_topk_fn(x, K, mode=mode, sorted=False)
            rc = index_recall(idx_ref, fi)
            ok = sanity(x, fv, fi)
            print(f"B={b:4d} V={v:8d} {mode:6s}: {old_us:8.1f} -> {us:7.1f} us ({old_us/us:5.2f}x) "
                  f"| rc {old_rc:5.1f}% -> {rc:5.1f}% | range={ok[0]} uniq={ok[1]} gather={ok[2]}", flush=True)
        del x, idx_ref
        torch.cuda.empty_cache()
