#!/usr/bin/env python3
"""Time split of one prefill request from a torch.profiler (kineto) chrome trace, no rocprofv3.

Pairs with ci/prefill_torch_profile.py (writes trace_<tokens>_<i>.json.gz + requests.json). Each GPU kernel is
classified by the innermost custom CPU op that launched it (kineto correlation id -> runtime call -> enclosing
radiance::/vllm::/_C:: op); kernels outside those ops fall back to kernel-name rules. Name-only rules
(ci/prefill_trace_split.py) misfile r4d_gdn_* as attention and drop the drafter's w4a16 GEMMs into other.

Idle time is split into gaps <= 1 ms (inside a forward: launch/dependency) and > 1 ms (step boundaries, CPU).
GEMM TF/s: N from grid.x (64 columns per block at TN=2, 128 at TN=4), K=5120 except N=5120 (down K=17408 when
the previous kernel is silu_mul_quant, else o_proj/out_proj K=6144); M per chunk from the drafter's
`hybrid_triton_w4a16 Mx5120x25600` annotation (or --m).

  prefill_torch_split.py DIR_OR_TRACE [...] [--m 1514] [--json out.json]
"""
import argparse
import glob
import gzip
import json
import os
import re
import statistics
from collections import defaultdict

OPCLS = [
    ("GEMM", r"radiance::mxfp4_linear"),
    ("norm-quant", r"radiance::(add_rms_quant|silu_mul_quant|gdn_norm_quant)|_C::(dynamic_per_token|rms_norm|fused_add_rms)"),
    ("GDN", r"vllm::qwen_gdn_attention_core"),
    ("attention", r"vllm::unified_attention|vllm::unified_kv_cache_update|_C::rotary"),
    ("drafter-w4a16", r"vllm::rdna_hybrid_w4a16|vllm::radiance_w4a16"),
]
NAMECLS = [
    ("drafter-w4a16", r"w4a16|kernel_unified_attention|reshape_and_cache_kernel_flash"),
    ("GDN", r"gdn"),
    ("attention", r"r4d_attn|reshape_and_cache_flash"),
    ("inductor-fused", r"^triton_(poi|red|per)_"),
]
ORDER = [c for c, _ in OPCLS] + ["inductor-fused", "other", "idle<=1ms", "idle>1ms"]
SHAPES = {34816: "gate_up", 16384: "qkvz", 14336: "qkv", 128: "ba"}


def load(path):
    ev = json.load(gzip.open(path) if path.endswith(".gz") else open(path))["traceEvents"]
    kern = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
    rt = {e["args"]["correlation"]: e for e in ev
          if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})}
    ops = sorted((e for e in ev if e.get("cat") == "cpu_op" and re.match(r"(radiance|vllm|_C)::", e["name"])),
                 key=lambda e: e["ts"])
    ms = [int(m.group(1)) for e in sorted(ev, key=lambda e: e.get("ts", 0)) if e.get("cat") == "user_annotation"
          for m in [re.match(r"hybrid_triton_w4a16 (\d+)x5120x25600", e.get("name", ""))] if m]
    return kern, rt, ops, ms


def classify(k, rt, ops):
    r = rt.get(k["args"].get("correlation"))
    if r is not None:
        t, best = r["ts"], None
        for o in ops:
            if o["ts"] > t:
                break
            if t <= o["ts"] + o["dur"] and (best is None or o["dur"] < best["dur"]):
                best = o
        if best is not None:
            for c, rx in OPCLS:
                if re.search(rx, best["name"]):
                    return c
    for c, rx in NAMECLS:
        if re.search(rx, k["name"], re.I):
            return c
    return "other"


def gemm_shape(k, prev):
    gx = (k["args"].get("grid") or [0])[0]
    tn = re.search(r"gemm_\w+<(\d+),", k["name"])
    n = gx * (128 if tn and tn.group(1) == "4" else 64)
    if n == 5120:
        return ("down", n, 17408) if "silu_mul_quant" in prev else ("o/out_proj", n, 6144)
    return SHAPES.get(n, f"N{n}"), n, 5120


def split(path, m_override=None):
    kern, rt, ops, ms = load(path)
    if m_override:
        ms = [m_override]
    cls, gem = defaultdict(float), defaultdict(lambda: [0.0, 0, 0, 0])
    small = big = 0.0
    prev = ""
    for i, k in enumerate(kern):
        c = classify(k, rt, ops)
        cls[c] += k["dur"]
        if c == "GEMM" and "gemm" in k["name"]:
            nm, n, kk = gemm_shape(k, prev)
            kind = "atiled" if "atiled" in k["name"] else "folded" if "folded" in k["name"] else "other"
            g = gem[(nm, kind)]
            g[0] += k["dur"]; g[1] += 1; g[2], g[3] = n, kk
        if i + 1 < len(kern):
            gap = kern[i + 1]["ts"] - (k["ts"] + k["dur"])
            if gap > 1000:
                big += gap
            elif gap > 0:
                small += gap
        prev = k["name"]
    cls["idle<=1ms"], cls["idle>1ms"] = small, big
    span = kern[-1]["ts"] + kern[-1]["dur"] - kern[0]["ts"] if kern else 0
    tfs = {}
    for (nm, kind), (us, calls, n, kk) in gem.items():
        if ms and n:
            tfs[f"{nm}:{kind}"] = dict(ms=us / 1e3, calls=calls,
                                       tfs=2 * sum(ms) * n * kk * calls / len(ms) / (us * 1e-6) / 1e12)
    return dict(span_ms=span / 1e3, cls_ms={c: cls[c] / 1e3 for c in ORDER if cls.get(c)}, M=ms, gemm=tfs,
                n_kernels=len(kern))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--m", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "trace_*.json*"))) if os.path.isdir(p) else [p]
    groups = defaultdict(list)
    for f in files:
        m = re.search(r"trace_(\d+)_\d+", f)
        groups[m.group(1) if m else f].append(split(f, a.m))
    out = {}
    for key, runs in groups.items():
        # median run by span (the first profiled run of a cold length can carry a CPU stall)
        r = sorted(runs, key=lambda r: r["span_ms"])[len(runs) // 2]
        tot = sum(r["cls_ms"].values())
        print(f"=== {key}: {len(runs)} run(s), median span {r['span_ms']:.1f} ms "
              f"(spans {[round(x['span_ms'], 1) for x in runs]}), chunks M={r['M']}, {r['n_kernels']} kernels")
        for c, v in r["cls_ms"].items():
            print(f"  {c:<16}{v:9.1f} ms {100 * v / tot:6.1f} %")
        print(f"  {'non-GEMM':<16}{tot - r['cls_ms'].get('GEMM', 0):9.1f} ms "
              f"{100 * (1 - r['cls_ms'].get('GEMM', 0) / tot):6.1f} %")
        for g, v in sorted(r["gemm"].items(), key=lambda kv: -kv[1]["ms"]):
            print(f"  GEMM {g:<20}{v['ms']:8.1f} ms {v['calls']:5d} calls {v['tfs']:6.0f} TF/s")
        out[key] = dict(median=r, spans=[x["span_ms"] for x in runs])
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)
    return out


if __name__ == "__main__":
    main()
