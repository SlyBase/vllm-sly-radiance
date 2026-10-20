#!/usr/bin/env python3
"""Per-kernel time and post-kernel idle gaps from torch.profiler traces (ci/prefill_torch_profile.py output).

For every kernel name family: count, total and median duration, and how many times the GPU stayed idle
>= --stall us after it (the ~123 us platform stall that follows kernels longer than ~1.1 ms), with the
idle time attributed to the kernel that precedes the gap. Median over traces.

  prefill_kernel_gaps.py TRACE_OR_DIR [...] [--stall 100] [--top 25]
"""
import argparse, glob, gzip, json, os, re, statistics
from collections import defaultdict

FAM = [
    ("gemm_atiled", r"gemm_atiled"), ("gemm_folded", r"gemm_folded"), ("gemm_decode", r"gemm_decode"),
    ("silu_mul_quant", r"silu_mul_quant"), ("add_rms_quant", r"add_rms_quant"),
    ("gdn_norm_quant", r"gdn_norm_quant"), ("w4a16", r"w4a16"), ("r4d_gdn", r"r4d_gdn|gdn"),
    ("r4d_attn", r"r4d_attn"), ("triton", r"^triton_"), ("unified_attn(drafter)", r"kernel_unified_attention"),
]


def fam(n):
    for f, rx in FAM:
        if re.search(rx, n):
            return f
    return n[:40]


def one(path, stall):
    ev = json.load(gzip.open(path) if path.endswith(".gz") else open(path))["traceEvents"]
    k = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    out = defaultdict(lambda: [0, 0.0, [], 0, 0.0])  # n, total, durs, stalls, stall_us
    for i, e in enumerate(k):
        f = out[fam(e["name"])]
        f[0] += 1
        f[1] += e["dur"]
        f[2].append(e["dur"])
        if i + 1 < len(k):
            gap = k[i + 1]["ts"] - (e["ts"] + e["dur"])
            if stall <= gap <= 1000:
                f[3] += 1
                f[4] += gap
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--stall", type=float, default=100.0)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "trace_*.json*"))) if os.path.isdir(p) else [p]
    runs = [one(f, a.stall) for f in files]
    names = sorted({n for r in runs for n in r}, key=lambda n: -statistics.median(r[n][1] for r in runs if n in r))
    print(f"{len(runs)} trace(s); stall = idle {a.stall:.0f}..1000 us after the kernel")
    print(f"{'kernel':28s} {'n':>5s} {'ms':>8s} {'med us':>8s} {'stalls':>6s} {'stall ms':>8s}")
    for n in names[:a.top]:
        rs = [r[n] for r in runs if n in r]
        med = lambda i: statistics.median(x[i] for x in rs)
        print(f"{n:28s} {med(0):5.0f} {med(1) / 1e3:8.2f} {statistics.median(rs[0][2]):8.1f} {med(3):6.0f} "
              f"{med(4) / 1e3:8.2f}")


if __name__ == "__main__":
    main()
