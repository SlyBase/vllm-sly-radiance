#!/usr/bin/env python3
"""P4(c): Tile-Sweep fuer _triton_w4a16_skinny_fmt_kernel (vLLM 0.29 rdna_hybrid_w4a16)
auf den DFlash2-Draft-Shapes (gs=128, symmetrisch). Misst DRAM-kalt (Gewichts-Rotation
>= 128 MB) je (Shape, M) alle Tile-Konfigurationen und vergleicht mit der heutigen
gfx12x-Heuristik. Laeuft im Wegwerf-Container aus dem Runtime-Image (GPU exklusiv).

  python3 bench_w4a16_tiles.py [--quick] [--out /root/w4a16_tiles.json]
"""
import argparse, itertools, json, sys, time
import torch, triton
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as H

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="/root/w4a16_tiles.json")
ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--fc", action="store_true", help="only the DFlash2 fc layer (K = 5 x 5120 aux -> N 5120, ReplicatedLinear)")
args = ap.parse_args()

GS = 128
# (Name, N, K) des Drafts: 5 Layer, hidden 5120, 32x128 q / 8x128 kv, inter 17408
SHAPES = [("qkv", 6144, 5120), ("o", 5120, 4096), ("gate_up", 34816, 5120), ("down", 5120, 17408)]
if args.fc:
    SHAPES = [("fc", 5120, 25600)]  # combine_hidden_states -> self.model.fc, M = Target-Token je Step (8 x Seqs)
MS = [8, 16, 32, 40, 64] if not args.quick else [8, 40]
dev = torch.device("cuda")
torch.manual_seed(0)

def make(N, K, copies):
    ws, ss = [], []
    for _ in range(copies):
        ws.append(torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev))
        ss.append((torch.rand(N, K // GS, device=dev, dtype=torch.bfloat16) * 0.02 + 0.001))
    return ws, ss

def run(a, w, s, cfg):
    M, K = a.shape; N = w.shape[0]
    BM, BN, BK, NW, NS = cfg
    c = torch.empty((M, N), dtype=a.dtype, device=dev)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    kw = {} if NS is None else {"num_stages": NS}
    H._triton_w4a16_skinny_fmt_kernel[grid](
        a, w, s, s, c, M, N, K, K // 8, K // GS, group_size=GS, ZP_BIAS=8, HAS_ZP=False,
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_warps=NW, **kw)
    return c

def heuristic(M, N, K):
    # Kopie des gfx12x-Zweigs aus triton_w4a16_skinny_fmt_gemm (vLLM 0.29.0)
    if M <= 32: return (16, 16, 128, 4, None)
    if M <= 64:
        if K >= 2 * N: return (64, 32, 128, 8, None)
        if N > K: return (64, 32, 64, 8, None)
        return (32, 64, 128, 4, None)
    if M <= 128:
        if K >= 2 * N: return (64, 16, 64, 1, None)
        if N >= 2 * K: return (64, 128, 64, 8, None)
        return (64, 64, 64, 8, None)
    return (128, 64, 64, 8, None)

def bench(a, ws, ss, cfg, iters):
    n = len(ws)
    for i in range(3): run(a, ws[i % n], ss[i % n], cfg)
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for i in range(iters):
        ev[i][0].record(); run(a, ws[i % n], ss[i % n], cfg); ev[i][1].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000 for s, e in ev)
    return t[len(t) // 2]  # median us

def cfgs_for(M):
    bms = [16] if M <= 16 else ([16, 32] if M <= 32 else [16, 32, 64])
    bns = [16, 32, 64, 128]
    bks = [64, 128]
    nws = [1, 2, 4, 8]
    nss = [None, 1, 3] if not args.quick else [None, 2]
    for bm, bn, bk, nw, ns in itertools.product(bms, bns, bks, nws, nss):
        if nw == 1 and bm * bn > 1024: continue   # 1 Warp nur fuer kleine Tiles
        if nw == 8 and bm * bn < 512: continue
        yield (bm, bn, bk, nw, ns)

results = {}
for name, N, K in SHAPES:
    wbytes = N * K // 2 + N * (K // GS) * 2
    copies = max(2, (160 << 20) // wbytes)
    ws, ss = make(N, K, copies)
    for M in MS:
        a = (torch.randn(M, K, device=dev, dtype=torch.bfloat16))
        hcfg = heuristic(M, N, K)
        ref = run(a, ws[0], ss[0], hcfg).float()
        t_h = bench(a, ws, ss, hcfg, args.iters)
        best = (t_h, hcfg); rows = []
        t0 = time.time(); ncf = 0
        for cfg in cfgs_for(M):
            try:
                out = run(a, ws[0], ss[0], cfg).float()
                err = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
                if err > 2e-2: rows.append((cfg, None, err)); continue
                t = bench(a, ws, ss, cfg, args.iters)
            except Exception as e:  # z.B. LDS-Overflow / Compile-Fehler
                rows.append((cfg, None, str(e)[:60])); continue
            ncf += 1
            rows.append((cfg, t, err))
            if t < best[0]: best = (t, cfg)
        gbs = wbytes / (best[0] * 1e-6) / 1e9; gbs_h = wbytes / (t_h * 1e-6) / 1e9
        print(f"{name:8s} N={N:5d} K={K:5d} M={M:2d}: heuristic {hcfg} {t_h:7.1f} us ({gbs_h:5.0f} GB/s)"
              f" -> best {best[1]} {best[0]:7.1f} us ({gbs:5.0f} GB/s) {t_h/best[0]:.2f}x  [{ncf} cfgs, {time.time()-t0:.0f}s]",
              flush=True)
        top = sorted([r for r in rows if r[1] is not None], key=lambda r: r[1])[:5]
        results[f"{name}:{M}"] = {"N": N, "K": K, "M": M, "heuristic": [hcfg, t_h],
                                  "best": [best[1], best[0]], "top5": top}
    del ws, ss; torch.cuda.empty_cache()
json.dump(results, open(args.out, "w"), indent=1)
print("written", args.out)
