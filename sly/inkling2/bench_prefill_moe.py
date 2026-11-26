"""GPU sweep of Inkling-2-Mini's Triton expert GEMMs at the real PREFILL shapes (run in the image).

Inkling reads no vLLM fused_moe config file: its expert GEMMs (`radiance_inkling2_moe._expert_gemm` for
NVFP4 layers 0-19, vLLM's `_mxfp8_gemm` for MXFP8 layers 20-23) are launched with fixed tiles. This tool
times them at a prefill chunk (default 4096 and 8192 tokens = token tile 128 and 256, 128 experts, top-8,
random routing), ranks launch configs and writes the winner as the table that
RADIANCE_INKLING2_MOE_PREFILL_CFG points to.

    python bench_prefill_moe.py [--tokens 4096,8192] [--fmt nvfp4,mxfp8] [--budget-s 600]
                                [--table-out cfg.json] [--dq-estimate]
    python bench_prefill_moe.py --cpu-smoke --table-out /tmp/t.json     # Triton interpreter, tiny shape

Per config the time of w13 (N 2304, K 2304) + w2 (N 2304, K 1152) is summed (one table entry serves both).
TFLOP/s counts the real routed rows only. `--dq-estimate`: time of "dequantize all experts to bf16 once,
then one bf16 bmm" (uniform rows per expert, an UPPER BOUND on what a hipBLASLt route could reach).
"""
import argparse
import itertools
import json
import math
import os
import sys
import time
import types
from dataclasses import dataclass

if "--cpu-smoke" in sys.argv:
    os.environ.setdefault("TRITON_INTERPRET", "1")  # before triton is imported

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

EXPERTS, TOPK = 128, 8
V2 = "--v2" in sys.argv  # sweep the v2 kernels instead of v1
SHAPES = {"w13": (2304, 2304), "w2": (2304, 1152)}  # (N, K) of the two projections
PEAK_TF = 101.0  # bf16 WMMA dense, 64 CU x 1024 FLOP/clk x 1.546 GHz (the clock the card holds under load)
_TILES = ((8, 8), (128, 16), (384, 32), (768, 64), (4096, 128), (1 << 31, 256))  # vLLM trtllm_bmm._TILES


def token_tile(tokens):
    return next(t for max_tokens, t in _TILES if tokens <= max_tokens)


@dataclass
class Routing:
    tiles: list      # [expert id per CTA | row end per CTA | CTA count | padded rows]
    count: int       # CTAs with rows
    max_ctas: int    # the launch grid (expert_plan's upper bound)
    rows_real: int
    tile: int


def make_routing(tokens, topk, experts, tile, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.rand(tokens, experts, generator=g).topk(topk, dim=1).indices
    counts = torch.bincount(ids.flatten(), minlength=experts).tolist()
    rows = tokens * topk
    max_ctas = min(experts, rows) + (rows - min(experts, rows)) // tile
    ctas, ends = [], []
    for e, c in enumerate(counts):
        for j in range(-(-c // tile)):
            ctas.append(e)
            ends.append(len(ends) * tile + min(tile, c - j * tile))
    count = len(ctas)
    pad = max_ctas - count
    return Routing(ctas + [0] * pad + ends + [0] * pad + [count, tile * max_ctas], count, max_ctas, rows, tile)


def flops(rows, n, k):
    return 2 * rows * n * k


def pick_best(results):
    """[(cfg, [us per projection or None])] -> (cfg, summed us) of the fastest config that ran on all."""
    ok = [(sum(us), i, cfg) for i, (cfg, us) in enumerate(results) if all(u is not None for u in us)]
    if not ok:
        return None, math.inf
    total, _, cfg = min(ok)
    return cfg, total


def merge_table(path, fmt, tile, cfg):
    table = json.load(open(path)) if os.path.exists(path) else {}
    table.setdefault(fmt, {})[str(tile)] = cfg
    with open(path, "w") as f:
        json.dump(table, f, indent=1, sort_keys=True)


class Clock:
    def __init__(self, cpu):
        self.cpu = cpu

    def us(self, fn, reps):
        fn()  # compile + warm
        if self.cpu:
            t = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - t) / reps * 1e6
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        torch.cuda.synchronize()
        return a.elapsed_time(b) / reps * 1000


# ---- NVFP4 -----------------------------------------------------------------------------------

def nvfp4_inputs(r, n, k, experts, dev):
    rows = r.tile * r.max_ctas
    w = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8, device=dev)
    ws = (torch.rand(experts, n, k // 16, device=dev) * 3.75 + 0.25).to(torch.float8_e4m3fn)
    act = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, device=dev)
    asc = (torch.rand(rows * (k // 16) + 4096, device=dev) * 3.75 + 0.25).to(torch.float8_e4m3fn)
    rs, es = torch.rand(rows, device=dev) + 0.5, torch.rand(experts, device=dev) + 0.5
    tiles = torch.tensor(r.tiles, dtype=torch.int32, device=dev)
    plan = types.SimpleNamespace(tile=r.tile, max_ctas=r.max_ctas, rows=rows)
    return act, asc, rs, w, ws, es, tiles, plan


def nvfp4_candidates(tile, cpu):
    if cpu:
        yield dict(block_m=16, block_n=32, block_k=64), None
        yield dict(block_m=16, block_n=64, block_k=64), None
        return
    if V2:  # v2 kernels: full grid; v2_safe=True avoids the denormal operand of the bit-trick decode
        yield dict(block_m=128, block_n=128, block_k=64, num_warps=4, num_stages=2), "A"  # v1 as tuned by the table
        for tr, bm, bn, bk, nw, ns in itertools.product((False, True), (64, 128), (64, 128), (32, 64), (4, 8), (2, 3)):
            if bm <= max(tile, 64):
                yield dict(block_m=bm, block_n=bn, block_k=bk, num_warps=nw, num_stages=ns, v2=True, v2_trunc=tr), "A"
        return
    stage_a = [dict(block_m=bm, block_n=bn, block_k=bk, num_warps=4, num_stages=2)
               for bm, bn, bk in itertools.product((32, 64, 128), (64, 128), (64, 128, 256)) if bm <= tile]
    for cfg in stage_a:
        yield cfg, "A"


def split_v2_time(m, r, data, cfg, clock, reps):
    """us of the activation dequant kernel alone per projection (the rest of a v2 call is the GEMM)."""
    out = []
    for d in data:
        act, asc, _rs, w, _ws, _es, tiles, plan = d
        k = w.shape[2] * 2
        dq = torch.empty((plan.rows, k), dtype=torch.bfloat16, device=act.device)
        br = min(16, cfg["block_m"])
        out.append(clock.us(lambda: m._dequant_act[(plan.max_ctas, triton.cdiv(plan.tile, br))](
            act, asc, tiles, dq, MAX_CTAS=plan.max_ctas, K=k, TILE=plan.tile, BK=cfg["block_k"], BR=br,
            DT=tl.bfloat16, SAFE=cfg.get("v2_safe", False), TRUNC=cfg.get("v2_trunc", False)), reps))
    print(json.dumps(dict(dequant_act_us=out)), flush=True)


def run_nvfp4(r, experts, dev, clock, reps, budget_end, cpu):
    import radiance_inkling2_moe as m

    m._hip = None
    kw = dict(dot_dtype=tl.float32) if cpu else {}
    shapes = list(SHAPES.values()) if not cpu else [(48, 192), (48, 128)]
    data = [nvfp4_inputs(r, n, k, experts, dev) for n, k in shapes]
    valid = torch.zeros(r.tile * r.max_ctas, dtype=torch.bool, device=dev)
    for t in range(r.count):
        valid[t * r.tile: r.tiles[r.max_ctas + t]] = True

    def time_cfg(cfg):
        try:
            return [clock.us(lambda d=d: m.nvfp4_expert_gemm(*d, **cfg, **kw), reps) for d in data]
        except Exception as e:  # OutOfResources, LDS overflow ...
            print(json.dumps(dict(fmt="nvfp4", cfg=cfg, error=f"{type(e).__name__}: {str(e)[:120]}")), flush=True)
            return [None] * len(data)

    default_us = [clock.us(lambda d=d: m.nvfp4_expert_gemm(*d, **kw), reps) for d in data]
    results, seen = [], set()

    def add(cfg):
        key = tuple(sorted(cfg.items()))
        if key in seen or time.time() > budget_end:
            return
        seen.add(key)
        us = time_cfg(cfg)
        results.append((cfg, us))
        print(json.dumps(dict(fmt="nvfp4", tile=r.tile, cfg=cfg, us=us)), flush=True)

    for cfg, _ in nvfp4_candidates(r.tile, cpu):
        add(cfg)
    if not cpu:  # stage B: warps / stages around the three best tiles
        top = sorted((sum(us), i) for i, (c, us) in enumerate(results) if all(u is not None for u in us))[:3]
        for _, i in top:
            for nw, ns in itertools.product((2, 4, 8), (1, 2, 3)):
                add({**results[i][0], "num_warps": nw, "num_stages": ns})
    best, best_us = pick_best(results)
    if V2 and best and not cpu:
        split_v2_time(m, r, data, best, clock, reps)
    ref = m.nvfp4_expert_gemm(*data[0], **kw)[valid].float()
    out = m.nvfp4_expert_gemm(*data[0], **best, **kw)[valid].float() if best else ref
    rel = float(((out - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item())
    rows_real = r.rows_real
    tf = sum(flops(rows_real, n, k) for n, k in shapes) / (best_us * 1e6) if best else 0.0
    tf_def = sum(flops(rows_real, n, k) for n, k in shapes) / (sum(default_us) * 1e6)
    res = dict(fmt="nvfp4", tile=r.tile, best=best, best_us=best_us, default_us=sum(default_us),
               per_proj_default_us=default_us, tflops_best=round(tf, 2), tflops_default=round(tf_def, 2),
               pct_peak_best=round(100 * tf / PEAK_TF, 1), max_rel_vs_default=rel,
               speedup=round(sum(default_us) / best_us, 2) if best else None)
    print(json.dumps(dict(summary=res)), flush=True)
    return res


# ---- MXFP8 -----------------------------------------------------------------------------------

def run_mxfp8(r, experts, dev, clock, reps, budget_end):
    import radiance_inkling2_moe as m

    m._hip = None
    from vllm.models.inkling2.nvidia.ops.moe_sm120 import _mxfp8_gemm

    rows = r.tile * r.max_ctas
    tiles = torch.tensor(r.tiles, dtype=torch.int32, device=dev)
    ones = torch.ones(experts, device=dev)
    data = []
    for n, k in SHAPES.values():
        w = torch.randint(0, 120, (experts, n, k), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        ws = torch.randint(120, 130, (experts, n, k // 32), dtype=torch.uint8, device=dev)
        act = torch.randint(0, 120, (rows, k), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        asc = torch.randint(120, 130, (rows * (k // 32) + 4096,), dtype=torch.uint8, device=dev)
        data.append((n, k, w, ws, act, asc, torch.empty(rows, n, dtype=torch.bfloat16, device=dev)))
    valid = torch.zeros(rows, dtype=torch.bool, device=dev)
    for t in range(r.count):
        valid[t * r.tile: r.tiles[r.max_ctas + t]] = True

    def launch(d, c):
        n, k, w, ws, act, asc, out = d
        _mxfp8_gemm[(triton.cdiv(n, c["block_n"]), r.max_ctas * triton.cdiv(r.tile, c["block_t"]))](
            act, asc, w, ws, ones, out, tiles, MAX_CTAS=r.max_ctas, N=n, K=k, TILE=r.tile,
            BN=c["block_n"], BT=c["block_t"], BK=c["block_k"], num_warps=c["num_warps"],
            num_stages=c["num_stages"])
        return out

    default = m.mxfp8_launch(r.tile, True)
    default_us = [clock.us(lambda d=d: launch(d, default), reps) for d in data]
    results, seen = [], set()

    def add(cfg):
        key = tuple(sorted(cfg.items()))
        if key in seen or time.time() > budget_end:
            return
        seen.add(key)
        try:
            us = [clock.us(lambda d=d: launch(d, cfg), reps) for d in data]
        except Exception as e:
            print(json.dumps(dict(fmt="mxfp8", cfg=cfg, error=f"{type(e).__name__}: {str(e)[:120]}")), flush=True)
            us = [None] * len(data)
        results.append((cfg, us))
        print(json.dumps(dict(fmt="mxfp8", tile=r.tile, cfg=cfg, us=us)), flush=True)

    for bn, bt in itertools.product((32, 64, 128), (16, 32, 64)):
        if bt <= r.tile:
            add(dict(block_n=bn, block_t=bt, block_k=128, num_warps=4, num_stages=2))
    top = sorted((sum(us), i) for i, (c, us) in enumerate(results) if all(u is not None for u in us))[:3]
    for _, i in top:
        for nw, ns in itertools.product((2, 4, 8), (1, 2, 3)):
            add({**results[i][0], "num_warps": nw, "num_stages": ns})
    best, best_us = pick_best(results)
    ref = launch(data[0], default)[valid].float().clone()
    out = launch(data[0], best)[valid].float() if best else ref
    rel = float(((out - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item())
    tf = sum(flops(r.rows_real, n, k) for n, k in SHAPES.values()) / (best_us * 1e6) if best else 0.0
    res = dict(fmt="mxfp8", tile=r.tile, best=best, best_us=best_us, default_us=sum(default_us),
               per_proj_default_us=default_us, tflops_best=round(tf, 2), pct_peak_best=round(100 * tf / PEAK_TF, 1),
               max_rel_vs_default=rel, speedup=round(sum(default_us) / best_us, 2) if best else None)
    print(json.dumps(dict(summary=res)), flush=True)
    return res


# ---- "dequantize once, then bf16 bmm" upper bound -------------------------------------------

@triton.jit
def _dq_kernel(Packed, Scale, Out, K: tl.constexpr, BR: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    p = tl.load(Packed + rows[:, None].to(tl.int64) * (K // 2) + tl.arange(0, K // 2)[None, :])
    s = tl.load(Scale + rows[:, None].to(tl.int64) * (K // 16) + tl.arange(0, K // 16)[None, :])
    tl.store(Out + rows[:, None].to(tl.int64) * K + tl.arange(0, K)[None, :],
             _dequant_rows(p, s.to(tl.float32), tl.bfloat16))


def run_dq_estimate(r, experts, dev, clock, reps):
    """us of dequant(all experts' weights + the routed activation rows) + one bf16 bmm, per projection."""
    global _dequant_rows
    from radiance_inkling2_moe import _dequant_rows  # a jit function resolves it as a module global
    total = 0.0
    for n, k in SHAPES.values():
        w = torch.randint(0, 256, (experts * n, k // 2), dtype=torch.uint8, device=dev)
        ws = (torch.rand(experts * n, k // 16, device=dev) + 0.5).to(torch.float8_e4m3fn)
        out = torch.empty(experts * n, k, dtype=torch.bfloat16, device=dev)
        per = r.rows_real // experts
        act = torch.randint(0, 256, (experts * per, k // 2), dtype=torch.uint8, device=dev)
        asc = (torch.rand(experts * per, k // 16, device=dev) + 0.5).to(torch.float8_e4m3fn)
        aout = torch.empty(experts * per, k, dtype=torch.bfloat16, device=dev)
        dq_w = clock.us(lambda: _dq_kernel[(experts * n // 16,)](w, ws, out, K=k, BR=16), reps)
        dq_a = clock.us(lambda: _dq_kernel[(experts * per // 16,)](act, asc, aout, K=k, BR=16), reps)
        a3, w3 = aout.view(experts, per, k), out.view(experts, n, k)
        bmm = clock.us(lambda: torch.bmm(a3, w3.transpose(1, 2)), reps)
        total += dq_w + dq_a + bmm
        print(json.dumps(dict(dq_estimate=dict(n=n, k=k, dq_weights_us=round(dq_w), dq_act_us=round(dq_a),
                                               bmm_us=round(bmm), bmm_tflops=round(flops(per * experts, n, k) / bmm / 1e6, 1)))),
              flush=True)
        del w, ws, out, act, asc, aout, a3, w3
    tf = sum(flops(r.rows_real, n, k) for n, k in SHAPES.values()) / (total * 1e6)
    res = dict(fmt="nvfp4-dq-bmm-estimate", tile=r.tile, us=round(total), tflops=round(tf, 2))
    print(json.dumps(dict(summary=res)), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="4096,8192")
    ap.add_argument("--fmt", default="nvfp4,mxfp8")
    ap.add_argument("--budget-s", type=float, default=600, help="wall budget per (format, tile)")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--table-out")
    ap.add_argument("--dq-estimate", action="store_true")
    ap.add_argument("--cpu-smoke", action="store_true")
    ap.add_argument("--v2", action="store_true", help="sweep the v2 prefill kernels")
    a = ap.parse_args(argv)
    import radiance_inkling2_moe as m

    os.environ[m._CFG_ENV] = "0"  # the default column must be the historical launch, not the built-in table
    os.environ[m._V2_ENV] = "0"  # explicit v2=True entries opt in; a bare launch must stay v1
    if a.cpu_smoke:
        dev, experts, clock = "cpu", 4, Clock(True)
        r = make_routing(24, 2, experts, 16, seed=0)
        res = {"nvfp4": run_nvfp4(r, experts, dev, clock, 1, time.time() + 60, True)}
        if a.table_out and res["nvfp4"]["best"]:
            merge_table(a.table_out, "nvfp4", r.tile, res["nvfp4"]["best"])
        return res
    dev, clock, out = "cuda", Clock(False), {}
    for tokens in [int(x) for x in a.tokens.split(",")]:
        tile = token_tile(tokens)
        r = make_routing(tokens, TOPK, EXPERTS, tile, seed=0)
        print(json.dumps(dict(routing=dict(tokens=tokens, tile=tile, ctas=r.count, max_ctas=r.max_ctas,
                                           rows=r.rows_real))), flush=True)
        for fmt in a.fmt.split(","):
            end = time.time() + a.budget_s
            res = run_nvfp4(r, EXPERTS, dev, clock, a.reps, end, False) if fmt == "nvfp4" \
                else run_mxfp8(r, EXPERTS, dev, clock, a.reps, end)
            out[f"{fmt}-{tile}"] = res
            if a.table_out and res["best"] and (res["speedup"] or 0) > 1.0:
                merge_table(a.table_out, fmt, tile, res["best"])
        if a.dq_estimate and "nvfp4" in a.fmt:
            out[f"dq-{tile}"] = run_dq_estimate(r, EXPERTS, dev, clock, a.reps)
    return out


if __name__ == "__main__":
    main()
