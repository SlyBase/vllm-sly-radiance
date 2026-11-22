"""Micro-benchmark: Inkling-2-Mini low-M GEMV, vLLM Triton bf16 vs the HIP bf16 kernel (and int8).

  triton    vLLM's _gemv_kernel on the bf16 weight (what runs with RADIANCE_INKLING2_HIP_GEMV=0)
  hip-bf16  radiance_inkling2_gemv.gemv_bf16 on the SAME bf16 weight (the default route)
  hipblas   torch.mm(x, W^T) = hipBLASLt on the same bf16 weight, M >= 2 (bf16 out; fp32-out shapes via
            torch.mm(out_dtype=float32) when this torch has it, else an ERR line)
  hip-q8    radiance_inkling2_gemv.gemv_q8 on the tiled int8 copy + fp32 scale per 32 K (--q8 only)

Weights ROTATE through a pool of >= 280 MB (4x the R9700's 64 MB Infinity Cache), so every call
streams cold weight bytes. Calls are captured into one CUDA graph over the whole pool and the
replay is timed with events: no Python launch cost, like the graph-captured decode. GB/s is bytes
actually read: N*K*2 (bf16 arms), N*K*1.125 (q8: int8 bytes + scales). `roofline` is a plain streaming
read of the same pool size (what the card can deliver at all). Every timed cell is first checked
against an fp32 torch reference (first 8192 rows).

  python3 bench_gemv.py [--cfgs] [--q8] [--shapes N,K ...] [--ms 1,2,4,8,16]
                        [--json out.json] [--big GB] [--skip triton,hipblas]
    --cfgs   also time every launch config (T, W / T, W, D) of the kernels under test (the int8 kernel with
             --q8, else the bf16 one), not only the auto choice
    --skip   leave these arms out (the sweeps of the int8 kernel need bf16 only as the bar: keep hip-bf16)
    --big    M=8 only: weights spread over a pool of GB gigabytes (a model's real address range)
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

# fused local qkvr, global q-side, k/v, wo / DSpark o, DSpark qkv, DSpark kv_ctx, gate_up, down,
# unembed (fp32 out), markov rnn_proj (bias), MoE router
SHAPES = [(8600, 2304), (6552, 2304), (1024, 2304), (2304, 3072), (5120, 2304), (2048, 2304),
          (18432, 2304), (2304, 9216), (201024, 2304), (768, 5120), (128, 2304)]
M_VALUES = tuple(range(1, 17))
POOL_BYTES = 280 * 1024 * 1024
CHECK_ROWS = 8192


def graph_us(fns, min_s=0.1):
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


def triton_call(x, w, y, b=None):
    import triton
    from vllm.models.inkling2.nvidia.ops.gemv import _BK, _BN, _gemv_kernel
    m, k = x.shape
    n = w.shape[0]
    _gemv_kernel[(triton.cdiv(n, _BN),)](x, w, b, y, m, n, k, x.stride(0), w.stride(0),
                                         y.stride(0), HAS_BIAS=b is not None, BN=_BN, BK=_BK, MP=16,
                                         num_warps=4)


def bf16_call(x, w, y, cfg, b=None):
    m, k = x.shape
    ext.gemv_bf16(x.data_ptr(), w.data_ptr(), b.data_ptr() if b is not None else 0, y.data_ptr(),
                  m, w.shape[0], k, x.stride(0), w.stride(0), y.stride(0),
                  int(y.dtype == torch.float32), torch.cuda.current_stream().cuda_stream, cfg)


def blas_call(x, w, y):
    if y.dtype == torch.bfloat16:
        torch.mm(x, w.t(), out=y)
    else:
        y.copy_(torch.mm(x, w.t(), out_dtype=torch.float32))


def q8_call(x, qt, st, n, y, cfg, b=None):  # tiled int8 [NT, K/64, 1024] + fp32 [NT, K/64, 32]
    m, k = x.shape
    ext.gemv_q8(x.data_ptr(), qt.data_ptr(), st.data_ptr(), b.data_ptr() if b is not None else 0, y.data_ptr(), m, n, k,
                x.stride(0), y.stride(0), int(y.dtype == torch.float32), qt.shape[0],
                torch.cuda.current_stream().cuda_stream, cfg)


def ok(y, x, deq, extra=0.0):  # deq: [rows, K] fp32 weight slice, extra: bias slice
    ref = x.float() @ deq.t() + extra
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
    ap.add_argument("--q8", action="store_true")
    ap.add_argument("--skip", default="")
    ap.add_argument("--shapes", nargs="*")
    ap.add_argument("--ms", default=",".join(map(str, M_VALUES)))
    ap.add_argument("--cfgs-at", default="", help="M values the --cfgs sweep runs at (default: all --ms)")
    ap.add_argument("--json")
    ap.add_argument("--big", type=float, default=0.0)
    a = ap.parse_args()
    shapes = [tuple(map(int, s.split(","))) for s in a.shapes] if a.shapes else SHAPES
    ms = [int(v) for v in a.ms.split(",")]
    cfg_ms = [int(v) for v in a.cfgs_at.split(",")] if a.cfgs_at else ms
    rows = []

    def emit(line):
        print(line, flush=True)

    emit(f"torch {torch.__version__} device {torch.cuda.get_device_name(0)} BF16_CFGS {ext.BF16_CFGS}")
    try:
        roofline(emit)
    except Exception as e:
        emit(f"roofline ERR {e!r}")
    skip = set(a.skip.split(","))
    for n, k in shapes:
        out32 = n == 201024
        has_bias = n == 768
        g = torch.Generator(device="cuda").manual_seed(0)
        base = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) / k**0.5
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g) if has_bias else None
        bf_deq = base[:CHECK_ROWS].float()
        if a.q8:
            wq0, sc0 = disp.to_q8(base)
            deq = wq0[:CHECK_ROWS].float() * sc0[:CHECK_ROWS].repeat_interleave(32, dim=1)
            pf = max(1, math.ceil(POOL_BYTES / (n * k * 1.125)))
            qt0, st0 = disp.pack_q8(wq0, sc0)
            pool_q = [(qt0, st0)] + [(qt0.clone(), st0.clone()) for _ in range(pf - 1)]
        pool_bytes = int(a.big * 2**30) if a.big else POOL_BYTES
        pb = max(1, math.ceil(pool_bytes / (n * k * 2)))
        pool_b = [base] + [base.clone() for _ in range(pb - 1)]
        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
            y = torch.empty(m, n, device="cuda", dtype=torch.float32 if out32 else torch.bfloat16)
            cells = []
            ref_extra = bias[:CHECK_ROWS].float() if has_bias else 0.0

            def check(yy):
                ref = x.float() @ bf_deq.t() + ref_extra
                return bool(torch.allclose(yy[:, :bf_deq.shape[0]].float(), ref, rtol=2**-6, atol=2e-3))

            if "triton" not in skip:
                try:
                    triton_call(x, base, y, bias)
                    torch.cuda.synchronize()
                    good = check(y)
                    us = graph_us([lambda w=w: triton_call(x, w, y, bias) for w in pool_b])
                    cells.append(("triton", "-", us, n * k * 2, good))
                except Exception as e:  # keep the sweep alive, report
                    emit(f"{n} {k} {m} triton ERR {e!r}"[:200])
            cfgs = list(range(ext.BF16_NCFG)) if a.cfgs and not a.q8 and m in cfg_ms else []
            for cfg in [-1] + cfgs:
                try:
                    bf16_call(x, base, y, cfg, bias)
                    torch.cuda.synchronize()
                    good = check(y)
                    us = graph_us([lambda w=w: bf16_call(x, w, y, cfg, bias) for w in pool_b])
                    cells.append(("hip-bf16", "auto" if cfg < 0 else cfg, us, n * k * 2, good))
                except Exception as e:
                    emit(f"{n} {k} {m} hip-bf16 cfg={cfg} ERR {e!r}"[:200])
            if m >= 2 and not has_bias and "hipblas" not in skip:
                try:
                    blas_call(x, base, y)
                    torch.cuda.synchronize()
                    good = check(y)
                    us = graph_us([lambda w=w: blas_call(x, w, y) for w in pool_b])
                    cells.append(("hipblas", "-", us, n * k * 2, good))
                except Exception as e:
                    emit(f"{n} {k} {m} hipblas ERR {e!r}"[:200])
            if a.q8 and m <= 16:
                for cfg in [-1] + (list(range(ext.NCFG)) if a.cfgs and m in cfg_ms else []):
                    try:
                        q8_call(x, qt0, st0, n, y, cfg, bias)
                        torch.cuda.synchronize()
                        good = ok(y, x, deq, ref_extra)
                        us = graph_us([lambda q=q, s=s: q8_call(x, q, s, n, y, cfg, bias) for q, s in pool_q])
                        cells.append(("hip-q8", "auto" if cfg < 0 else cfg, us,
                                      n * k + 4 * n * (k // 32), good))
                    except Exception as e:
                        emit(f"{n} {k} {m} hip-q8 cfg={cfg} ERR {e!r}"[:200])
            for arm, cfg, us, byt, good in cells:
                row = dict(n=n, k=k, m=m, arm=arm, cfg=cfg, us=round(us, 2),
                           gbps=round(byt / us * 1e-3, 1), ok=good, pool_mb=pb * n * k * 2 >> 20)
                rows.append(row)
                emit(f"{n} {k} {m} {arm} {cfg} {us:.1f} {row['gbps']:.0f} {'ok' if good else 'WRONG'}")
        del pool_b, base
        torch.cuda.empty_cache()
        if a.json:
            json.dump(rows, open(a.json, "w"))


if __name__ == "__main__":
    main()
