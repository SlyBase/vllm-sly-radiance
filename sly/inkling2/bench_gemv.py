"""Micro-benchmark: Inkling-2-Mini low-M GEMV, vLLM Triton bf16 vs the HIP int8 (Q8_0) kernel.

  triton   vLLM's _gemv_kernel on the bf16 weight (what runs without RADIANCE_INKLING2_W8)
  hip-q8   radiance_inkling2_gemv.gemv_q8 on the int8 copy + fp32 scale per 32 K values

Weights ROTATE through a pool of >= 280 MB (4x the R9700's 64 MB Infinity Cache), so every call
streams cold weight bytes. Calls are captured into one CUDA graph over the whole pool and the
replay is timed with events: no Python launch cost, like the graph-captured decode (split-K's
second kernel therefore counts). GB/s is bytes actually read: N*K*2 (triton), N*K*1.125 (q8).
`roofline` is a plain streaming read of the same pool size (what the card can deliver at all).
Every timed cell is first checked against an fp32 torch reference (first 8192 rows).

  python3 bench_gemv.py [--cfgs] [--shapes N,K ...] [--ms 1,2,4,8,16] [--json out.json]
    --cfgs   also time every launch config (R, U) of the kernel, not only the auto choice
Output: one flushed line per cell: `N K M arm cfg us GBps ok`.
"""
import argparse
import json
import math
import sys
import time

import torch

import radiance_inkling2_gemv as ext
import radiance_inkling2_gemv_hip as disp

SHAPES = [(8600, 2304), (6552, 2304), (1024, 2304), (2304, 3072), (5120, 2304),
          (18432, 2304), (2304, 9216), (201024, 2304)]
M_VALUES = (1, 2, 4, 8, 16)
POOL_BYTES = 280 * 1024 * 1024
CHECK_ROWS = 8192


def graph_us(fns, min_s=0.15):
    """Mean us per call of fns (one kernel each), replayed as one CUDA graph."""
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for f in fns:
            f()  # warmup / compile
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for f in fns:
            f()
    g.replay()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    g.replay()
    e1.record()
    torch.cuda.synchronize()
    one = e0.elapsed_time(e1) / 1e3
    reps = max(3, min(300, int(min_s / max(one, 1e-6))))
    e0.record()
    for _ in range(reps):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / reps / len(fns)


def triton_call(x, w, y):
    import triton
    from vllm.models.inkling2.nvidia.ops.gemv import _BK, _BN, _gemv_kernel
    m, k = x.shape
    n = w.shape[0]
    _gemv_kernel[(triton.cdiv(n, _BN),)](x, w, None, y, m, n, k, x.stride(0), w.stride(0),
                                         y.stride(0), HAS_BIAS=False, BN=_BN, BK=_BK, MP=16,
                                         num_warps=4)


def hip_call(x, wq, scale, y, scratch, cfg):  # int8 [N,K] + fp32 [N,K/32]
    m, k = x.shape
    n = wq.shape[0]
    ext.gemv_q8(x.data_ptr(), wq.data_ptr(), scale.data_ptr(), 0, y.data_ptr(), m, n, k,
                 x.stride(0), wq.stride(0), y.stride(0), int(y.dtype == torch.float32),
                 scratch.data_ptr(), scratch.numel() * 4, torch.cuda.current_stream().cuda_stream,
                 cfg)


def ok(y, x, deq):  # deq: [rows, K] fp32 dequantised/bf16->fp32 weight slice
    ref = x.float() @ deq.t()
    got = y[:, :deq.shape[0]].float()
    return bool(torch.allclose(got, ref, rtol=2**-6, atol=2e-3))


def roofline(emit):
    n = POOL_BYTES // 4
    pool = [torch.randint(0, 2**31 - 1, (n,), dtype=torch.int32, device="cuda") for _ in range(2)]
    us = graph_us([lambda t=t: t.sum() for t in pool for _ in range(4)])
    emit(f"roofline sum-int32 {n * 4 * 1e-3 / us:.0f} GB/s ({us:.0f} us per {n * 4 >> 20} MiB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", action="store_true")
    ap.add_argument("--shapes", nargs="*")
    ap.add_argument("--ms", default=",".join(map(str, M_VALUES)))
    ap.add_argument("--json")
    a = ap.parse_args()
    shapes = [tuple(map(int, s.split(","))) for s in a.shapes] if a.shapes else SHAPES
    ms = [int(v) for v in a.ms.split(",")]
    rows = []

    def emit(line):
        print(line, flush=True)

    emit(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)} CFGS {ext.CFGS}")
    try:
        roofline(emit)
    except Exception as e:
        emit(f"roofline ERR {e!r}")
    scratch = torch.empty(1 << 20, dtype=torch.float32, device="cuda")
    for n, k in shapes:
        out32 = n == 201024
        g = torch.Generator(device="cuda").manual_seed(0)
        base = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) / k**0.5
        wq0, sc0 = disp.to_q8(base)
        deq = wq0[:CHECK_ROWS].float() * sc0[:CHECK_ROWS].repeat_interleave(32, dim=1)
        bf_deq = base[:CHECK_ROWS].float()
        pb = max(1, math.ceil(POOL_BYTES / (n * k * 2)))
        pf = max(1, math.ceil(POOL_BYTES / (n * k * 1.125)))
        pool_b = [base] + [base.clone() for _ in range(pb - 1)]
        pool_q = [(wq0, sc0)] + [(wq0.clone(), sc0.clone()) for _ in range(pf - 1)]
        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
            y = torch.empty(m, n, device="cuda", dtype=torch.float32 if out32 else torch.bfloat16)
            cells = []
            try:
                triton_call(x, base, y)
                torch.cuda.synchronize()
                good = ok(y, x, bf_deq)
                us = graph_us([lambda w=w: triton_call(x, w, y) for w in pool_b])
                cells.append(("triton", "-", us, n * k * 2, good))
            except Exception as e:  # keep the sweep alive, report
                emit(f"{n} {k} {m} triton ERR {e!r}"[:200])
            cfgs = list(range(ext.NCFG)) if a.cfgs else []
            for cfg in [-1] + cfgs:
                try:
                    hip_call(x, wq0, sc0, y, scratch, cfg)
                    torch.cuda.synchronize()
                    good = ok(y, x, deq)
                    us = graph_us([lambda q=q, s=s: hip_call(x, q, s, y, scratch, cfg)
                                   for q, s in pool_q])
                    cells.append(("hip-q8", "auto" if cfg < 0 else cfg, us, n * k + 4 * n * (k // 32), good))
                except Exception as e:
                    emit(f"{n} {k} {m} hip-q8 cfg={cfg} ERR {e!r}"[:200])
            for arm, cfg, us, byt, good in cells:
                row = dict(n=n, k=k, m=m, arm=arm, cfg=cfg, us=round(us, 2),
                           gbps=round(byt / us * 1e-3, 1), ok=good)
                rows.append(row)
                emit(f"{n} {k} {m} {arm} {cfg} {us:.1f} {row['gbps']:.0f} {'ok' if good else 'WRONG'}")
        del pool_b, pool_q, base, wq0
        torch.cuda.empty_cache()
        if a.json:
            json.dump(rows, open(a.json, "w"))


if __name__ == "__main__":
    main()
