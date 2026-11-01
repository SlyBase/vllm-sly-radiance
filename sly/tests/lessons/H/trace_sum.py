#!/usr/bin/env python3
"""trace_sum.py TRACE.json[.gz] [TOPN] -- GPU kernel summary of a torch-profiler chrome trace of a decode run:
GPU busy time, span of the GPU activity, the idle share between kernels, and the top kernels by total time.
Used to tell WHY a decode step is slow: attention kernel time (kernel), idle gaps (CPU / launch bound),
or a pile of extra small kernels (an eager / piecewise path instead of a replayed graph)."""
import collections
import gzip
import json
import sys

p = sys.argv[1]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 18
op = gzip.open if p.endswith(".gz") else open
ev = json.load(op(p, "rt")).get("traceEvents", [])
k = [e for e in ev if e.get("cat") == "kernel" and "dur" in e]
if not k:
    sys.exit("no kernel events in " + p)
t0 = min(e["ts"] for e in k)
t1 = max(e["ts"] + e["dur"] for e in k)
busy = sum(e["dur"] for e in k)
print(f"{len(k)} kernels, GPU busy {busy / 1e3:.1f} ms of span {(t1 - t0) / 1e3:.1f} ms ({100 * busy / (t1 - t0):.0f} %), idle {(t1 - t0 - busy) / 1e3:.1f} ms")
agg = collections.defaultdict(lambda: [0.0, 0])
for e in k:
    a = agg[e["name"][:90]]
    a[0] += e["dur"]
    a[1] += 1
print(f"{'kernel':92s} {'total ms':>9s} {'calls':>7s} {'avg us':>8s}")
for n, (d, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
    print(f"{n:92s} {d / 1e3:9.2f} {c:7d} {d / c:8.1f}")
attn = sum(d for n, (d, c) in agg.items() if "attn" in n.lower() or "attention" in n.lower() or "r4d" in n.lower())
print(f"attention-named kernels: {attn / 1e3:.1f} ms ({100 * attn / busy:.0f} % of busy)")
