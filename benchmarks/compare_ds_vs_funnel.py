"""Apples-to-apples comparison: deep_select vs funnel_topk vs torch.topk.

Timing: DeepSelect's own harness (tests/kernelkit `bench`): kineto/CUPTI kernel
time under an 8 GB L2 flush before every iteration — the same methodology behind
README's perf_bf16.png. For each candidate we take the e2e span over ALL device
kernels of one call (minus the L2-flush memset and CPU-side runtime API events),
which is exactly what tests/test.py does for torch.topk and reduces to the
single-kernel time for deep_select.

Workload: Lightning Indexer shape — bf16, k=512, batch in {6,256,512,4096},
vocab in {16K..1M}. recall@k is index-based against torch.topk (bf16 has many
ties, so even an exact kernel may score < 100% index recall; deep_select is
additionally verified exact via the topk condition min(selected) >= max(unselected)).

Prerequisites:
    pip install -e .                          # DeepSelect itself
    TORCH_CUDA_ARCH_LIST="10.3" pip install -e . --no-build-isolation
        # in the funnel-topk checkout, see docs/compare-funnel-topk-20260926.zh.md §4

Usage (from anywhere):
    python3 benchmarks/compare_ds_vs_funnel.py [out.json]

Reference: docs/compare-funnel-topk-20260926.zh.md
"""
import os, sys, json
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))  # kernelkit
import kernelkit as kk

import deep_select
try:
    from funnel_topk import topk as funnel_topk_fn
    from funnel_topk import _kernels as funnel_kernels
except ImportError:
    sys.exit("funnel_topk is not installed. See docs/compare-funnel-topk-20260926.zh.md §4:\n"
             "  cd /path/to/funnel-topk && TORCH_CUDA_ARCH_LIST=\"10.3\" pip install -e . --no-build-isolation")

OUT_JSON = sys.argv[1] if len(sys.argv) > 1 else str(REPO_ROOT / "compare_ds_funnel.json")

DEVICE = "cuda"
K = 512
BATCHES = [6, 256, 512, 4096]
VOCABS = [16384, 65536, 131072, 262144, 524288, 1048576]
NUM_RUNS = 10

print(f"deep_select OK, funnel CUDA ext: {funnel_kernels.HAS_CUDA_EXT}, "
      f"Triton backend: {__import__('funnel_topk.funnel', fromlist=['HAS_FUNNEL_TRITON']).HAS_FUNNEL_TRITON}")
if not funnel_kernels.HAS_CUDA_EXT:
    sys.exit("funnel_topk CUDA extension (_C) is unavailable; GPU runs would silently fall "
             "back to torch.topk. Rebuild per docs/compare-funnel-topk-20260926.zh.md §4.")

# CPU-side runtime API events (cudaLaunchKernel / cudaDeviceSynchronize / ...)
# show up as bare names; device kernels all start with "void " (C++ signature),
# so match runtime APIs by PREFIX to avoid killing kernels whose template args
# contain "cuda" (e.g. sbtopk::gatherTopK<...>(at::cuda::detail::TensorInfo...)).
# The 8 GB L2-flush memset shows up as a FillFunctor elementwise kernel.
_EXCLUDE_PREFIX = ("cuda", "Memset")
_EXCLUDE_SUBSTR = ("FillFunctor", "profiler_range")

def bench_call(fn, nbytes, num_runs=NUM_RUNS):
    """Return (e2e_us, tbps) over all device kernels of fn(), excluding the L2-flush memset."""
    res = kk.bench(fn, num_runs)
    names = [n for n in res.get_kernel_names()
             if not n.startswith(_EXCLUDE_PREFIX) and not any(e in n for e in _EXCLUDE_SUBSTR)]
    t = res.get_e2e_time(names)
    return t * 1e6, nbytes / t / 1e12

def index_recall(idx_ref, idx, k):
    found = (idx_ref.unsqueeze(-1) == idx.unsqueeze(-2)).any(dim=-1)
    return found.float().mean().item() * 100.0

results = []
for b in BATCHES:
    for v in VOCABS:
        torch.manual_seed(0)
        x = torch.randn(b, v, dtype=torch.bfloat16, device=DEVICE)
        in_bytes = b * v * 2
        row = {"batch": b, "vocab": v}

        # --- torch.topk baseline (sorted=False, same as DeepSelect's harness) ---
        nb = in_bytes + b * K * (2 + 8)
        us, tbps = bench_call(lambda: torch.topk(x, K, dim=1, sorted=False), nb)
        row["torch"] = (us, tbps)
        idx_ref = torch.topk(x, K, dim=1, sorted=True).indices  # ref for recall

        # --- deep_select (README config: sv=False, si=False, rv=False, int32) ---
        ds_idx_bytes = b * K * 4
        nb = in_bytes + ds_idx_bytes
        us, tbps = bench_call(
            lambda: deep_select.topk(x, K, sorted=False, begin=None, end=None,
                                     indices_type=torch.int32, sorted_index=False,
                                     hint=None, output_idx=None, output_idx_offset=None,
                                     idx_oob_fill_value=-1, value_oob_fill_value=0.0,
                                     return_value=False, abort_when_nan_found=False), nb)
        row["deep_select"] = (us, tbps)
        _, ds_idx = deep_select.topk(x, K, sorted=False, begin=None, end=None,
                                     indices_type=torch.int64, sorted_index=False,
                                     hint=None, output_idx=None, output_idx_offset=None,
                                     idx_oob_fill_value=-1, value_oob_fill_value=0.0,
                                     return_value=True, abort_when_nan_found=False)
        row["ds_recall"] = index_recall(idx_ref, ds_idx, K)
        # exactness witness: min(selected) >= max(unselected)
        gathered = x.gather(1, ds_idx)
        masked = x.clone(); masked.scatter_(1, ds_idx, float("-inf"))
        row["ds_exact"] = bool((gathered.amin(1) >= masked.amax(1)).all())

        # --- funnel_topk, three modes ---
        for mode in ["standard", "fast", "turbo"]:
            try:
                probe = funnel_kernels.funnel_cuda_topk(x[:1].contiguous(), K, mode=mode)
                if probe is None:
                    row[mode] = None; continue
                keep = probe[0].shape[-1]
                nb = in_bytes + b * keep * (2 + 8)
                us, tbps = bench_call(lambda m=mode: funnel_topk_fn(x, K, mode=m, sorted=False), nb)
                _, f_idx = funnel_topk_fn(x, K, mode=mode, sorted=False)
                row[mode] = (us, tbps, index_recall(idx_ref, f_idx, K), keep)
            except Exception as e:
                row[mode] = f"ERR: {type(e).__name__}: {e}"

        results.append(row)
        def fmt(r):
            if r is None: return "N/A"
            if isinstance(r, str): return r
            if len(r) == 4: return f"{r[0]:8.1f}us {r[1]:5.2f}TB/s rc={r[2]:5.1f}% keep={r[3]}"
            return f"{r[0]:8.1f}us {r[1]:5.2f}TB/s"
        print(f"B={b:5d} V={v:8d} | ds {fmt(row['deep_select'])} (rc={row['ds_recall']:.1f}% exact={row['ds_exact']}) | "
              f"torch {fmt(row['torch'])} | std {fmt(row['standard'])} | fast {fmt(row['fast'])} | turbo {fmt(row['turbo'])}",
              flush=True)
        del x, idx_ref, ds_idx, gathered, masked
        torch.cuda.empty_cache()

with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=1, default=str)
print("saved", OUT_JSON)
