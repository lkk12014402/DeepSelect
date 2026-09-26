"""Correctness verification: apply DeepSelect's own test-suite checks
(tests/test.py) to funnel_topk's three modes, plus deep_select as control.

Checks per row (input x of shape (B, N), k=512, bf16):
  1. index_range      : 0 <= idx < N for every returned index
  2. unique_index     : no duplicate indices within a row
  3. gather_consistent: values == x.gather(1, idx)   (reported value is real)
  4. topk_condition   : min(selected_values) >= max(unselected_values)
                        (the definition of exact top-k; an approximate kernel
                        is expected to FAIL this on some rows)

For topk_condition we report the fraction of rows satisfying it (row_rate);
the other three are hard assertions reported as all-pass booleans.

Usage:
    python3 benchmarks/verify_correctness_vs_funnel.py

Reference: docs/compare-funnel-topk-20260926.zh.md
"""
import sys, json
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tests"))

import deep_select
from funnel_topk import topk as funnel_topk_fn

K = 512
BATCHES = [6, 256, 4096]
VOCABS = [16384, 262144, 1048576]

def checks(x, values, indices, k):
    B, N = x.shape
    idx = indices.long()
    out = {}
    out["index_range"] = bool(((idx >= 0) & (idx < N)).all())
    srt = idx.sort(dim=1).values
    out["unique_index"] = bool((srt[:, 1:] != srt[:, :-1]).all())
    gathered = x.gather(1, idx)
    if values is not None:
        out["gather_consistent"] = bool(torch.equal(gathered, values))
    masked = x.clone()
    masked.scatter_(1, idx, float("-inf"))
    ok_rows = gathered.amin(1) >= masked.amax(1)
    out["topk_cond_row_rate"] = ok_rows.float().mean().item() * 100.0
    return out

results = []
for b in BATCHES:
    for v in VOCABS:
        torch.manual_seed(0)
        x = torch.randn(b, v, dtype=torch.bfloat16, device="cuda")
        row = {"batch": b, "vocab": v}

        # deep_select (control): README config, return_value=True for value checks
        ds_v, ds_i = deep_select.topk(
            x, K, sorted=False, begin=None, end=None, indices_type=torch.int64,
            sorted_index=False, hint=None, output_idx=None, output_idx_offset=None,
            idx_oob_fill_value=-1, value_oob_fill_value=0.0,
            return_value=True, abort_when_nan_found=False)
        row["deep_select"] = checks(x, ds_v, ds_i, K)

        for mode in ["standard", "fast", "turbo"]:
            fv, fi = funnel_topk_fn(x, K, mode=mode, sorted=False)
            row[mode] = checks(x, fv, fi, K)

        results.append(row)
        def s(c):
            return (f"range={c['index_range']} uniq={c['unique_index']} "
                    f"gather={c['gather_consistent']} topk_rows={c['topk_cond_row_rate']:5.1f}%")
        print(f"B={b:5d} V={v:8d} | ds: {s(row['deep_select'])}")
        for mode in ["standard", "fast", "turbo"]:
            print(f"{'':19s}| {mode:9s}: {s(row[mode])}", flush=True)

out = REPO_ROOT / "verify_correctness_vs_funnel.json"
with open(out, "w") as f:
    json.dump(results, f, indent=1)
print("saved", out)
