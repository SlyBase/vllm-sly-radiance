#!/usr/bin/env python3
"""Split a rocprofv3 kernel trace into GEMM / attention / GDN / norm-quant / other / launch gaps.

Input: a rocprofv3 `--kernel-trace --output-format csv` run: <pid>_kernel_trace.csv (columns incl.
Queue_Id, Kernel_Name, Start_Timestamp, End_Timestamp, Grid_Size_X/Y/Z, Workgroup_Size_X/Y/Z) and,
optionally, <pid>_kernel_stats.csv (Name, Calls, TotalDurationNs, AverageNs, ...). Pass a file or a
directory (largest *kernel_trace.csv below it wins: the EngineCore process holds the GPU work).

The trace of a whole vLLM run holds start-up, warmup and the measured requests. The measured
requests are cut out by idle time: bursts of kernels separated by > --idle-gap-s of GPU silence;
the last --segments bursts are the measured ones, labelled --labels (default 2k,64k) in time order.
The driver (ci/prefill_diag.sh) sleeps several seconds between requests for exactly this.

Launch gap = idle time between consecutive kernels on the SAME queue, inside a burst (gaps < idle
gap). With the profiler on, every kernel carries ~56 us of its own overhead (rocprofv3-profiling.md):
read the gap column as an upper bound, and its share, not the absolute us.

  prefill_trace_split.py TRACE_DIR_OR_CSV [--segments 2] [--labels 2k,64k] [--forwards 2k=1,64k=32]
                         [--json out.json] [--top 15]
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

# First match wins. Order follows the diagnosis plan: GEMM, attention, GDN, norm-quant, rest.
CLASSES = [
    ("GEMM", r"gemm|w4a8|mxfp4|wvsplitk|hgemm|matmul|Cijk_"),
    ("attention", r"unified_attention|r4d|paged|flash_attn|fmha|flash_fwd|attn_fwd"),
    ("GDN", r"gdn|chunk|conv1d|delta|recurrent|linear_attn"),
    ("norm-quant", r"rms|quant|silu|layernorm|layer_norm"),
]
CLASS_ORDER = [c for c, _ in CLASSES] + ["other"]
_RX = [(c, re.compile(p, re.I)) for c, p in CLASSES]


def classify(name):
    for c, rx in _RX:
        if rx.search(name):
            return c
    return "other"


def find_trace(path):
    if os.path.isfile(path):
        return path
    c = glob.glob(os.path.join(path, "**", "*kernel_trace.csv"), recursive=True)
    if not c:
        sys.exit(f"no *kernel_trace.csv below {path}")
    return max(c, key=os.path.getsize)


def load_trace(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            def g(k):
                return int(r.get(k) or 0)
            rows.append(dict(
                q=(r.get("Agent_Id", ""), r.get("Queue_Id", "")), name=r["Kernel_Name"],
                s=g("Start_Timestamp"), e=g("End_Timestamp"),
                grid=(g("Grid_Size_X"), g("Grid_Size_Y"), g("Grid_Size_Z")),
                wg=(g("Workgroup_Size_X"), g("Workgroup_Size_Y"), g("Workgroup_Size_Z"))))
    rows.sort(key=lambda r: r["s"])
    return rows


def bursts(rows, idle_ns):
    out, cur, hi = [], [], None
    for r in rows:
        if cur and r["s"] - hi > idle_ns:
            out.append(cur)
            cur, hi = [], None
        cur.append(r)
        hi = r["e"] if hi is None else max(hi, r["e"])
    if cur:
        out.append(cur)
    return out


def union_ns(rows):
    tot, end = 0, None
    for r in sorted(rows, key=lambda r: r["s"]):
        if end is None or r["s"] > end:
            tot += r["e"] - r["s"]
            end = r["e"]
        elif r["e"] > end:
            tot += r["e"] - end
            end = r["e"]
    return tot


def analyse(rows, idle_ns):
    """One burst -> dict of totals (ns)."""
    t0, t1 = min(r["s"] for r in rows), max(r["e"] for r in rows)
    cls = defaultdict(lambda: [0, 0])          # class -> [ns, calls]
    kern = defaultdict(lambda: [0, 0])         # kernel name -> [ns, calls]
    gemm = defaultdict(lambda: [0, 0])         # (name, grid, wg) -> [ns, calls]
    perq = defaultdict(list)
    for r in rows:
        d, c = r["e"] - r["s"], classify(r["name"])
        cls[c][0] += d; cls[c][1] += 1
        kern[r["name"]][0] += d; kern[r["name"]][1] += 1
        if c == "GEMM":
            k = (r["name"], r["grid"], r["wg"])
            gemm[k][0] += d; gemm[k][1] += 1
        perq[r["q"]].append(r)
    gaps = []
    for rs in perq.values():
        rs.sort(key=lambda r: r["s"])
        for a, b in zip(rs, rs[1:]):
            g = b["s"] - a["e"]
            if 0 < g < idle_ns:
                gaps.append(g)
    gaps.sort()
    return dict(span_ns=t1 - t0, busy_ns=union_ns(rows), n=len(rows),
                cls={c: cls[c] for c in CLASS_ORDER if c in cls},
                gap_ns=sum(gaps), gap_n=len(gaps),
                gap_p50=gaps[len(gaps) // 2] if gaps else 0,
                gap_p90=gaps[int(len(gaps) * .9)] if gaps else 0,
                kern=sorted(kern.items(), key=lambda kv: -kv[1][0]),
                gemm=sorted(gemm.items(), key=lambda kv: -kv[1][0]))


def load_stats(trace_csv):
    p = trace_csv.replace("kernel_trace", "kernel_stats")
    if p == trace_csv or not os.path.exists(p):
        return None
    with open(p, newline="") as f:
        return sorted(csv.DictReader(f), key=lambda r: -float(r.get("TotalDurationNs") or 0))


def short(n, w=70):
    n = re.sub(r"\s+", " ", n)
    return n if len(n) <= w else n[:w - 1] + "…"


def report(label, a, top, forwards=None):
    ms = lambda ns: ns / 1e6
    span = a["span_ns"]
    L = [f"=== {label}: span {ms(span):.1f} ms, GPU-busy {ms(a['busy_ns']):.1f} ms, {a['n']} kernels ==="]
    L.append(f"{'class':<12}{'ms':>10}{'% span':>9}{'calls':>9}")
    for c, (ns, n) in a["cls"].items():
        L.append(f"{c:<12}{ms(ns):>10.1f}{100 * ns / span:>8.1f}%{n:>9}")
    L.append(f"{'launch gap':<12}{ms(a['gap_ns']):>10.1f}{100 * a['gap_ns'] / span:>8.1f}%{a['gap_n']:>9}"
             f"   (p50 {a['gap_p50'] / 1e3:.0f} us, p90 {a['gap_p90'] / 1e3:.0f} us; incl. ~56 us profiler overhead)")
    f = forwards or 0
    L.append(f"-- GEMM by (kernel, grid, workgroup), top {top}"
             + (f"; calls/fwd with {f} forward(s) (64 layers: 64 = once per layer, e.g. down)" if f else ""))
    L.append(f"{'ms':>9}{'%span':>7}{'calls':>7}{'avg us':>9}{'c/fwd':>7}  kernel  grid  wg")
    for (nm, grid, wg), (ns, n) in a["gemm"][:top]:
        L.append(f"{ms(ns):>9.1f}{100 * ns / span:>6.1f}%{n:>7}{ns / n / 1e3:>9.1f}"
                 f"{(n / f if f else float('nan')):>7.1f}  {short(nm, 50)}  {grid}  {wg}")
    L.append(f"-- top {top} kernels by total time")
    L.append(f"{'ms':>9}{'%span':>7}{'calls':>7}{'avg us':>9}  class       kernel")
    for nm, (ns, n) in a["kern"][:top]:
        L.append(f"{ms(ns):>9.1f}{100 * ns / span:>6.1f}%{n:>7}{ns / n / 1e3:>9.1f}  {classify(nm):<11} {short(nm)}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace")
    ap.add_argument("--segments", type=int, default=2)
    ap.add_argument("--labels", default="2k,64k")
    ap.add_argument("--idle-gap-s", type=float, default=2.0)
    ap.add_argument("--forwards", default="", help="label=N,... forward passes per request (calls/fwd column)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    path = find_trace(a.trace)
    rows = load_trace(path)
    idle = int(a.idle_gap_s * 1e9)
    bs = bursts(rows, idle)
    print(f"{path}: {len(rows)} kernels, {len(bs)} bursts (idle gap > {a.idle_gap_s}s); "
          f"burst sizes: {[len(b) for b in bs][-8:]}")
    sel = bs[-a.segments:]
    labels = a.labels.split(",")
    if len(sel) < a.segments or len(labels) < len(sel):
        sys.exit(f"need {a.segments} bursts and labels, have {len(sel)} bursts / {len(labels)} labels")
    fw = {k: int(v) for k, v in (p.split("=") for p in a.forwards.split(",") if p)}
    res = {}
    for lab, b in zip(labels, sel):
        res[lab] = analyse(b, idle)
        print(report(lab, res[lab], a.top, fw.get(lab)))
        print()
    st = load_stats(path)
    if st:
        print(f"-- kernel_stats.csv (WHOLE run incl. start-up/warmup), top 5 of {len(st)}")
        for r in st[:5]:
            print(f"{float(r['TotalDurationNs']) / 1e6:>10.1f} ms {r['Calls']:>8} calls  {short(r['Name'])}")
    if a.json:
        slim = {k: {kk: vv for kk, vv in v.items() if kk not in ("kern", "gemm")} for k, v in res.items()}
        for k, v in res.items():
            slim[k]["top_kernels"] = [[n, ns, c] for n, (ns, c) in v["kern"][:a.top]]
            slim[k]["top_gemm"] = [[n, list(g), list(w), ns, c] for (n, g, w), (ns, c) in v["gemm"][:a.top]]
        json.dump(slim, open(a.json, "w"), indent=1)
    return res


if __name__ == "__main__":
    main()
