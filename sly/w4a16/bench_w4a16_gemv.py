#!/usr/bin/env python3
"""GPU microbench: gemv_u4 (sly/w4a16/radiance_w4a16_gemv.hip) vs the stock `rdna_hybrid_w4a16_apply` op on the
Qwen3.8-27B DFlash2 shapes (G = 128; the last one is the int4 lm_head), DRAM-cold (a pool of weight copies larger
than the caches, called round robin), max |diff| against the stock result, GB/s of the bytes each route streams.

  python3 sly/w4a16/bench_w4a16_gemv.py [--ms 1 6 8 16] [--shapes 6144x5120 ...] [--cfgs]

--cfgs also times every launch config per (shape, M) and prints the winner (the cfg -1 table in the .hip is taken
from this). Needs the built radiance_w4a16_gemv.so and the patched vLLM (rdna_hybrid_w4a16 op) on the GPU host.
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("RADIANCE_W4A16_HIP_GEMV", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import radiance_w4a16_gemv_hip as W  # noqa: E402

SHAPES = ["6144x5120", "5120x4096", "5120x17408", "5120x25600", "248320x5120"]  # N x K
POOL_BYTES = 768 << 20  # > the caches: every call streams its weight from DRAM


def exllama_pack(u):  # [N, K] nibbles -> [N, K/8] int32, the ExLlama shuffle
    n, k = u.shape
    g = u.view(n, k // 8, 8).to(torch.int64)
    p = (g[..., 0] | (g[..., 2] << 4) | (g[..., 4] << 8) | (g[..., 6] << 12) | (g[..., 1] << 16) | (g[..., 3] << 20)
         | (g[..., 5] << 24) | (g[..., 7] << 28))
    return ((p + 2 ** 31) % 2 ** 32 - 2 ** 31).to(torch.int32)


def timed(fn, iters):
    for _ in range(3):
        fn(0)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for i in range(iters):
        fn(i)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters * 1e3  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="*", default=SHAPES)
    ap.add_argument("--ms", nargs="*", type=int, default=[1, 6, 8, 16])
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--cfgs", action="store_true")
    a = ap.parse_args()
    assert W._ext is not None, "radiance_w4a16_gemv.so not importable"
    stock_op = torch.ops.vllm.rdna_hybrid_w4a16_apply
    cu = torch.cuda.get_device_properties(0).multi_processor_count
    dev, g = "cuda", a.group
    print(f"{'shape':>12} {'M':>3} {'stock us':>9} {'GB/s':>6} {'u4 us':>8} {'GB/s':>6} {'x':>5} {'max|diff|':>10}")
    for sh in a.shapes:
        n, k = map(int, sh.split("x"))
        gen = torch.Generator(device=dev).manual_seed(0)
        u = torch.randint(0, 16, (n, k), device=dev, dtype=torch.uint8, generator=gen)
        s = (torch.rand(n, k // g, device=dev, generator=gen) * 0.05 + 0.005).to(torch.bfloat16)
        wq = torch.cat([exllama_pack(u[r:r + 8192]) for r in range(0, n, 8192)]).view(torch.int8)  # chunked: lm_head
        qt, st = W.pack_u4(wq, s, g)
        del u
        stock_bytes = wq.numel() + s.numel() * 2
        u4_bytes = qt.numel() + st.numel() * 2
        copies = max(2, -(-POOL_BYTES // max(stock_bytes, u4_bytes)))
        pool = [(wq.clone(), s.clone(), qt.clone(), st.clone()) for _ in range(copies - 1)] + [(wq, s, qt, st)]
        for m in a.ms:
            x = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
            y = torch.empty(m, n, device=dev, dtype=torch.bfloat16)

            def u4(i, cfg=-1):
                _, _, q_, s_ = pool[i % copies]
                W._ext.gemv_u4(x.data_ptr(), q_.data_ptr(), s_.data_ptr(), y.data_ptr(), 0, m, n, k, x.stride(0),
                               y.stride(0), 0, qt.shape[0], torch.cuda.current_stream().cuda_stream, cfg)

            def stock(i):
                w_, s_, _, _ = pool[i % copies]
                return stock_op(x, w_, s_, None, None, cu, g)

            ref = stock(copies - 1)
            u4(copies - 1)
            torch.cuda.synchronize()
            diff = (y.float() - ref.float()).abs().max().item()
            t_s, t_u = timed(stock, a.iters), timed(u4, a.iters)
            print(f"{sh:>12} {m:>3} {t_s:9.1f} {stock_bytes / t_s / 1e3:6.0f} {t_u:8.1f} {u4_bytes / t_u / 1e3:6.0f} "
                  f"{t_s / t_u:5.2f} {diff:10.4f}")
            if a.cfgs:
                ts = {c: timed(lambda i, c=c: u4(i, c), a.iters // 2) for c in range(W._ext.NCFG)}
                best = min(ts, key=ts.get)
                print(f"{'':>12} cfgs us: " + " ".join(f"{c}:{t:.1f}" for c, t in ts.items()) + f"  best {best} "
                      f"{W._ext.CFGS[best]} (auto {W._ext.auto_cfg_u4(m, n, k)})")
        del pool


if __name__ == "__main__":
    main()
