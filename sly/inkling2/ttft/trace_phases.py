#!/usr/bin/env python3
"""rocprofv3 kernel_trace.csv -> one line per request phase: kernel count, GPU busy, span, idle, top kernels.

A phase = kernels separated from the next by less than --gap-ms (default 500). Run the profiled server with one
request every few seconds (max_tokens=1) and each phase is that request's prefill step. busy = union of kernel
intervals (overlapping streams are not double counted); idle = span - busy = launch / host gaps.

  python3 trace_phases.py kernel_trace.csv [--gap-ms 500] [--top 8]
"""
import argparse
import csv
from collections import defaultdict


def phases(rows, gap_ms=500.0, top=8):
    ks = sorted(((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Kernel_Name"]) for r in rows))
    out, cur = [], []
    for k in ks:
        if cur and (k[0] - max(c[1] for c in cur)) / 1e6 > gap_ms:
            out.append(cur)
            cur = []
        cur.append(k)
    if cur:
        out.append(cur)
    res = []
    for ph in out:
        busy, end = 0, ph[0][0]
        for s, e, _ in ph:  # sorted by start: sweep the union
            s2 = max(s, end)
            busy += max(0, e - s2)
            end = max(end, e)
        span = max(e for _, e, _ in ph) - ph[0][0]
        per = defaultdict(int)
        for s, e, n in ph:
            per[n] += e - s
        res.append({"kernels": len(ph), "busy_ms": busy / 1e6, "span_ms": span / 1e6, "idle_ms": (span - busy) / 1e6,
                    "top": [(n, t / 1e6) for n, t in sorted(per.items(), key=lambda x: -x[1])[:top]]})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--gap-ms", type=float, default=500)
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    with open(a.csv) as f:
        ps = phases(list(csv.DictReader(f)), a.gap_ms, a.top)
    for i, p in enumerate(ps):
        print(f"phase {i}: kernels={p['kernels']} span={p['span_ms']:.1f} ms busy={p['busy_ms']:.1f} ms idle={p['idle_ms']:.1f} ms "
              f"busy_ratio={p['busy_ms'] / p['span_ms'] if p['span_ms'] else 0:.2f}")
        for n, t in p["top"]:
            print(f"    {t:8.2f} ms  {n[:110]}")
