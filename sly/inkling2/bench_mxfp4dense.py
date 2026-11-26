"""GPU micro-bench + correctness of the MXFP4 dense path on Inkling-2-Mini's real shapes (gfx1201).

Runs inside the radiance image with the GPU. Validates the reused radiance_mxfp4_fp8 kernel against
radiance_mxfp4._exact_ref (fp32 ground truth) on the actual dense shapes, then benches GB/s on the
PACKED 4-bit bytes over a rotated weight pool (>=256 MB, never a single cached matrix), M=1..16.
Also compares against the bf16 HIP GEMV (radiance_inkling2_gemv) at the decode M band.

NOTE: the loop here is EAGER, so every W4A8 call reads ~29 us whatever the shape: that is the host (Python custom-op
dispatch + fp8 quant launch + empty + launch, ~28.7 us of CPU per call), not the kernel. The model runs the op
inside CUDA graphs; bench_mx4.py times it (and the HIP W4A16 GEMV, the int8 and bf16 GEMVs) as a replayed graph.

Env: RADIANCE_MXFP4_W4A8=1 RADIANCE_MXFP4_W4A8_MIN_M=0 RADIANCE_MXFP4_DECODE_MAX_M=16
     RADIANCE_MXFP4_WPERM=0 RADIANCE_MXFP4_AT_PSW=0 RADIANCE_INKLING2_MXFP4_DENSE=1
"""
import os
import sys
import time

import torch

import radiance_mxfp4 as R
import radiance_mxfp4_fp8 as ext
import radiance_inkling2_mxfp4dense as MD
from vllm import _custom_ops as ops

dev = "cuda"
print("ext:", ext.__file__, flush=True)
print("mxfp4dense:", MD.__file__, flush=True)

# Decode scratch (the op's kernel_class normally allocates this at load; we call the op directly).
DECODE_MAX_M = int(os.environ.get("RADIANCE_MXFP4_DECODE_MAX_M", "16"))
partial = torch.empty(4 * max(64, DECODE_MAX_M) * 36864, dtype=torch.float32, device=dev)
cnt = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)
ext.set_decode_scratch(partial.data_ptr(), partial.numel() * 4, cnt.data_ptr())
print(f"decode scratch {partial.numel()*4>>20} MiB, DECODE_MAX_M={DECODE_MAX_M}", flush=True)

# Real dense shapes (N, K) from the checkpoint: fused qkvr + its global-layer slices, o_proj, head.
SHAPES = [(8600, 2304), (6552, 2304), (3072, 2304), (1024, 2304), (384, 2304),
          (2304, 3072), (201024, 2304)]
POOL_BYTES = 300 << 20  # rotated weight pool >= 256 MB so no weight is cache-served


def make_pool(n, k):
    packed_one = n * (k // 2)
    copies = max(2, (POOL_BYTES + packed_one - 1) // packed_one)
    pool = []
    g = torch.Generator(device=dev).manual_seed(1234)
    for _ in range(copies):
        w = torch.randn(n, k, generator=g, device=dev, dtype=torch.bfloat16)
        p, s, r = MD.quantize_mxfp4(w)
        pool.append((p, s, r))
    return pool, copies * packed_one


def fp8_quant(x):
    q, s = ops.scaled_fp8_quant(x, scale=None, use_per_token_if_dynamic=True)
    return q, s.view(-1).float().contiguous()


def correctness():
    print("\n=== CORRECTNESS (kernel vs _exact_ref fp32) ===", flush=True)
    bad = 0
    for n, k in SHAPES:
        if k % 64:
            print(f"  N={n} K={k}: SKIP K%64", flush=True)
            continue
        g = torch.Generator(device=dev).manual_seed(7)
        w = torch.randn(n, k, generator=g, device=dev, dtype=torch.bfloat16)
        p, s, r = MD.quantize_mxfp4(w)
        for m in (1, 8, 16):
            x = torch.randn(m, k, generator=g, device=dev, dtype=torch.bfloat16)
            xq, xs = fp8_quant(x)
            out = torch.ops.radiance.mxfp4_linear(x, p, s, r)
            ref = R._exact_ref(xq, xs, p, s, n, k)
            rel = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
            ok = rel < 0.02
            bad += not ok
            print(f"  N={n:6d} K={k} M={m:2d} rel={rel:.5f} {'ok' if ok else '**WRONG**'}", flush=True)
    print("CORRECTNESS", "FAIL" if bad else "PASS", f"({bad} wrong)", flush=True)
    return bad


def bench_mxfp4():
    print("\n=== MXFP4 W4A8 GB/s on packed bytes (rotated pool) ===", flush=True)
    print(f"{'N':>7} {'K':>5} {'M':>3} {'us/call':>9} {'GB/s':>8}", flush=True)
    results = {}
    for n, k in SHAPES:
        if k % 64:
            continue
        pool, pool_bytes = make_pool(n, k)
        nc = len(pool)
        for m in (1, 2, 4, 8, 16):
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            # warmup (also fills the op's internal state)
            for i in range(nc):
                p, s, r = pool[i]
                torch.ops.radiance.mxfp4_linear(x, p, s, r)
            torch.cuda.synchronize()
            iters = 60
            t0 = time.perf_counter()
            for it in range(iters):
                p, s, r = pool[it % nc]
                torch.ops.radiance.mxfp4_linear(x, p, s, r)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / iters
            gbs = (n * (k // 2)) / dt / 1e9
            results[(n, k, m)] = (dt * 1e6, gbs)
            print(f"{n:7d} {k:5d} {m:3d} {dt*1e6:9.1f} {gbs:8.1f}", flush=True)
        del pool
        torch.cuda.empty_cache()
    return results


def bench_bf16():
    print("\n=== bf16 HIP GEMV GB/s (rotated pool, decode band) ===", flush=True)
    try:
        import radiance_inkling2_gemv as gext
    except ImportError:
        print("  bf16 GEMV ext missing, skip", flush=True)
        return {}
    print(f"{'N':>7} {'K':>5} {'M':>3} {'us/call':>9} {'GB/s':>8}", flush=True)
    res = {}
    for n, k in SHAPES:
        if n > 36864 or k % 8:
            continue
        copies = max(2, (POOL_BYTES + n * k * 2 - 1) // (n * k * 2))
        g = torch.Generator(device=dev).manual_seed(99)
        pool = [torch.randn(n, k, generator=g, device=dev, dtype=torch.bfloat16) for _ in range(copies)]
        for m in (1, 4, 8):
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            out = torch.empty(m, n, device=dev, dtype=torch.bfloat16)
            for w in pool:
                gext.gemv_bf16(x.data_ptr(), w.data_ptr(), 0, out.data_ptr(), m, n, k,
                               x.stride(0), w.stride(0), out.stride(0), 0,
                               torch.cuda.current_stream().cuda_stream, -1)
            torch.cuda.synchronize()
            iters = 60
            t0 = time.perf_counter()
            for it in range(iters):
                w = pool[it % copies]
                gext.gemv_bf16(x.data_ptr(), w.data_ptr(), 0, out.data_ptr(), m, n, k,
                               x.stride(0), w.stride(0), out.stride(0), 0,
                               torch.cuda.current_stream().cuda_stream, -1)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / iters
            gbs = (n * k * 2) / dt / 1e9
            res[(n, k, m)] = (dt * 1e6, gbs)
            print(f"{n:7d} {k:5d} {m:3d} {dt*1e6:9.1f} {gbs:8.1f}", flush=True)
        del pool
        torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    bad = correctness()
    mx = bench_mxfp4()
    bf = bench_bf16()
    print("\n=== SPEEDUP MXFP4 vs bf16 (decode M) ===", flush=True)
    for (n, k, m), (us, gbs) in sorted(mx.items()):
        if m in (1, 4, 8) and (n, k, m) in bf:
            bus, bgbs = bf[(n, k, m)]
            print(f"  N={n:6d} K={k} M={m:2d}: bf16 {bus:7.1f}us {bgbs:6.1f}GB/s -> "
                  f"mxfp4 {us:7.1f}us {gbs:6.1f}GB/s  {bus/us:.2f}x", flush=True)
    print("MICRODONE", "FAIL" if bad else "PASS", flush=True)