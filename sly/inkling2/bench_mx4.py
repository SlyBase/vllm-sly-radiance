"""Rotated-pool CUDA-GRAPH bench of the dense decode GEMVs on Inkling-2-Mini's real shapes (gfx1201).

Arms per (N, K, M): mx4 (radiance_inkling2_mx4, MXFP4 W4A16), w4a8 (the radiance_mxfp4_fp8 op incl. its fp8
activation-quant launch), int8 (gemv_q8), bf16 (gemv_bf16). Every arm is a list of calls over a weight pool of
>= 300 MB (nothing is cache-served), captured into ONE graph and replayed: no Python launch cost, which is what
the model's decode graphs see. (The eager loop of bench_mxfp4dense.py is host bound: ~29 us per W4A8 call
whatever the shape.) GB/s = bytes of the arm's weight format read per call / time: mx4 and w4a8 count codes +
E8M0 scales (K/2 + K/32 per row), int8 K + K/8 (fp32 scale per 32), bf16 2 K.

  python3 bench_mx4.py                       table, M = 1 8 16
  python3 bench_mx4.py --cfgs [--m 1 8 16]   gemv_mx4 launch-config sweep (cfg table of pick_cfg_mx4)
  python3 bench_mx4.py --shapes 8600x2304 384x2304 --arms mx4 w4a8
Env for the w4a8 arm: RADIANCE_MXFP4_W4A8=1 RADIANCE_MXFP4_W4A8_MIN_M=0 RADIANCE_MXFP4_DECODE_MAX_M=16
RADIANCE_MXFP4_WPERM=0 RADIANCE_MXFP4_AT_PSW=0.
"""
import argparse
import os

import torch

import radiance_inkling2_mxfp4dense as MD

dev = "cuda"
POOL = 300 << 20
_SCRATCH = []
SHAPES = [(384, 2304), (1024, 2304), (2304, 3072), (3072, 2304), (6552, 2304), (8600, 2304), (201024, 2304)]


def graph_us(fns, reps=20):
    for f in fns:
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in fns:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1000 / (reps * len(fns))


def copies(nbytes):
    return min(512, max(2, -(-POOL // nbytes)))


def stream():
    return torch.cuda.current_stream().cuda_stream  # inside a capture: the capture stream


def arm_mx4(n, k, m, cfg, w):
    """-> (fns for launch config cfg, bytes); `mx4_pool` builds the weight pool once for a sweep."""
    mk, nbytes = mx4_pool(n, k, m, w)
    return mk(cfg), nbytes


def mx4_pool(n, k, m, w):
    import radiance_inkling2_mx4 as ext
    packed, scale, _ = MD.quantize_mxfp4(w)
    qt, st = MD.pack_mx4(packed, scale)
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    y = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    pool = [(qt.clone(), st.clone()) for _ in range(copies(qt.numel() + st.numel()))]
    nt = qt.shape[0]
    return (lambda cfg: [(lambda q=q, s=s: ext.gemv_mx4(x.data_ptr(), q.data_ptr(), s.data_ptr(), y.data_ptr(), m, n,
                                                       k, k, n, 0, nt, stream(), cfg)) for q, s in pool],
            n * k // 2 + n * k // 32)


def arm_w4a8(n, k, m, cfg, w):
    import radiance_mxfp4 as R
    import radiance_mxfp4_fp8 as ext
    packed, scale, wref = MD.quantize_mxfp4(w)
    if not _SCRATCH:  # the kernel keeps raw pointers into these: they must outlive every captured graph
        _SCRATCH.extend((torch.empty(4 * 64 * 36864, dtype=torch.float32, device=dev),
                         torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)))
        ext.set_decode_scratch(_SCRATCH[0].data_ptr(), _SCRATCH[0].numel() * 4, _SCRATCH[1].data_ptr())
    assert R.MIN_M <= 0, "RADIANCE_MXFP4_W4A8_MIN_M must be 0"
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    op = torch.ops.radiance.mxfp4_linear
    pool = [(packed.clone(), scale.clone(), wref.clone()) for _ in range(copies(packed.numel() + scale.numel()))]
    return [(lambda p=p, s=s, r=r: op(x, p, s, r)) for p, s, r in pool], n * k // 2 + n * k // 32


def arm_int8(n, k, m, cfg, w):
    import radiance_inkling2_gemv as ext
    import radiance_inkling2_gemv_hip as GH
    qt, st = GH.pack_q8(*GH.to_q8(w))
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    y = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    pool = [(qt.clone(), st.clone()) for _ in range(copies(qt.numel() + 4 * st.numel()))]
    nt = qt.shape[0]
    return [(lambda q=q, s=s: ext.gemv_q8(x.data_ptr(), q.data_ptr(), s.data_ptr(), 0, y.data_ptr(), m, n, k, k, n,
                                          0, nt, stream(), -1)) for q, s in pool], n * k + n * k // 8


def arm_bf16(n, k, m, cfg, w):
    import radiance_inkling2_gemv as ext
    x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    y = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
    pool = [w.clone() for _ in range(copies(w.numel() * 2))]
    return [(lambda wc=wc: ext.gemv_bf16(x.data_ptr(), wc.data_ptr(), 0, y.data_ptr(), m, n, k, k, k, n, 0,
                                         stream(), -1)) for wc in pool], n * k * 2


ARMS = {"mx4": arm_mx4, "w4a8": arm_w4a8, "int8": arm_int8, "bf16": arm_bf16}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", action="store_true", help="sweep the gemv_mx4 launch configs")
    ap.add_argument("--m", type=int, nargs="+", default=[1, 8, 16])
    ap.add_argument("--shapes", nargs="+", help="NxK ...")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    a = ap.parse_args()
    shapes = [tuple(map(int, s.split("x"))) for s in a.shapes] if a.shapes else SHAPES
    if a.cfgs:
        import radiance_inkling2_mx4 as ext
    print(f"{'N':>7} {'K':>5} {'M':>3} {'arm':>10} {'us/call':>8} {'GB/s':>7}", flush=True)
    for n, k in shapes:
        w = torch.randn(n, k, device=dev, dtype=torch.bfloat16) / 48
        for m in a.m:
            if a.cfgs:
                best = None
                mk, nbytes = mx4_pool(n, k, m, w)
                for cfg in range(ext.NCFG):
                    us = graph_us(mk(cfg))
                    print(f"{n:7d} {k:5d} {m:3d} {'mx4 c%d' % cfg:>10} {us:8.1f} {nbytes / us / 1e3:7.1f}  "
                          f"(T,W,D)={ext.CFGS[cfg]}", flush=True)
                    best = min(best or (us, cfg), (us, cfg))
                print(f"{n:7d} {k:5d} {m:3d}   BEST cfg {best[1]} {best[0]:.1f} us  auto={ext.auto_cfg_mx4(m, n, k)}",
                      flush=True)
                continue
            for name in a.arms:
                fns, nbytes = ARMS[name](n, k, m, -1, w)
                us = graph_us(fns)
                print(f"{n:7d} {k:5d} {m:3d} {name:>10} {us:8.1f} {nbytes / us / 1e3:7.1f}", flush=True)
                del fns
                torch.cuda.empty_cache()
        del w
        torch.cuda.empty_cache()
    print("BENCH_DONE", flush=True)


if __name__ == "__main__":
    main()
