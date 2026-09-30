#!/usr/bin/env python3
"""P4(c): Tile-Sweep fuer _triton_w4a16_skinny_fmt_kernel (vLLM 0.29 rdna_hybrid_w4a16)
auf den DFlash2-Draft-Shapes (gs=128, symmetrisch). Misst DRAM-kalt (Gewichts-Rotation
>= 128 MB) je (Shape, M) alle Tile-Konfigurationen und vergleicht mit der heutigen
gfx12x-Heuristik. Laeuft im Wegwerf-Container aus dem Runtime-Image (GPU exklusiv).

  python3 bench_w4a16_tiles.py [--quick] [--out /root/w4a16_tiles.json]

--splitk (0.4.1): Sweep des Split-K-/Dequant-Pfads aus sly/patch_w4a16_tiles.py (Patch vorher im
Container anwenden) ueber alle Target-, Draft- und Fusions-Shapes, auf dem gekachelten Gewichtslayout
(RADIANCE_W4A16_TILED, `--rows` = Zeilenlayout). Split-K laeuft im Serial-Modus (mode 2, kein zweiter
Launch; `--modes0` misst den Puffer-Reduce zusaetzlich), KSTEP 2 als zweite Stufe auf den Top-5. Gemessen
wird der ganze Aufruf. Ausgabe je Fall: 0.4.0-Stand (Tile-Tabelle, Zeilenlayout), Tile-Tabelle gekachelt,
Default-Config der Fusionspfade, Bestwert gekachelt und derselbe Bestwert im Zeilenlayout (#3-Effekt);
am Ende die _GFX12X_SPLITK-Zeilen fuer Eintraege >= 5 % schneller als min(Tiles, Default).
"""
import argparse, itertools, json, sys, time
import torch, triton
from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as H

ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--out", default="/root/w4a16_tiles.json")
ap.add_argument("--iters", type=int, default=40)
ap.add_argument("--fc", action="store_true", help="only the DFlash2 fc layer (K = 5 x 5120 aux -> N 5120, ReplicatedLinear)")
ap.add_argument("--lmhead", action="store_true", help="only the int4 lm_head (N 248320, K 5120; sly/mxfp4/radiance_lmhead_int4.py)")
ap.add_argument("--target", action="store_true", help="only the INT4 target shapes the drafter does not share (RedHatAI/Qwen3.8-27B-INT4)")
ap.add_argument("--splitk", action="store_true", help="split-K sweep over target + drafter shapes (needs the patched module)")
ap.add_argument("--shapes", default="", help="comma list of shape names to keep (with --splitk)")
ap.add_argument("--atomic", action="store_true", help="also sweep the atomic split-K variant (with --splitk)")
ap.add_argument("--ms", default="", help="comma list of M buckets (default 8,16,32,40,64)")
ap.add_argument("--max-partial-mib", type=float, default=1.0,
                help="skip split-K configs whose fp32 partials exceed this (KV pool, see patch_w4a16_tiles.py)")
ap.add_argument("--rows", action="store_true", help="sweep on the row layout instead of the tiled one")
ap.add_argument("--modes0", action="store_true", help="also sweep the partial-buffer reduce (mode 0)")
ap.add_argument("--table-out", default="", help="write the resulting _GFX12X_SPLITK as JSON (RADIANCE_W4A16_SPLITK_TABLE)")
args = ap.parse_args()

GS = 128
# (Name, N, K) des Drafts: 5 Layer, hidden 5120, 32x128 q / 8x128 kv, inter 17408
SHAPES = [("qkv", 6144, 5120), ("o", 5120, 4096), ("gate_up", 34816, 5120), ("down", 5120, 17408)]
if args.fc:
    SHAPES = [("fc", 5120, 25600)]  # combine_hidden_states -> self.model.fc, M = Target-Token je Step (8 x Seqs)
if args.lmhead:
    SHAPES = [("lm_head", 248320, 5120)]  # RadianceLMHeadInt4: verify M = 8 x Seqs, draft M = 7 x Seqs
if args.target:
    # INT4-Target (compressed-tensors W4A16 g128): gate_up/down teilt es mit dem Draft, diese nicht.
    # GDN in_proj_qkvz = qkv 10240 + z 6144; Attention qkv = q 12288 (mit Gate) + k,v 1024;
    # GDN out_proj und Attention o_proj haben beide N 5120 x K 6144. Reihenfolge nach Bytes x Calls je Step
    # (qkvz 48x43 MB, out_o 64x16 MB, attn_qkv 16x38 MB), damit ein abgebrochener Sweep die wichtigsten hat.
    SHAPES = [("qkvz", 16384, 5120), ("out_o", 5120, 6144), ("attn_qkv", 14336, 5120)]
if args.splitk:
    # nach erwartetem Gewinn geordnet: N=5120 mit langem K zuerst (160-320 Workgroups ohne Split-K)
    # + the fused call sites: GDN qkvz + ba merged (16384 + 96 rows), DFlash context-KV (5 layers x K/V
    # rows), DFlash2 grouped-conv kernel_projection (2 x taps 2 x 320 groups)
    SHAPES = [("down", 5120, 17408), ("out_o", 5120, 6144), ("fc", 5120, 25600), ("o", 5120, 4096),
              ("qkvz_ba", 16480, 5120), ("qkvz", 16384, 5120), ("attn_qkv", 14336, 5120), ("qkv", 6144, 5120),
              ("gate_up", 34816, 5120), ("ctx_kv", 10240, 5120), ("conv", 1280, 5120),
              ("lm_head", 248320, 5120)]  # int4 head, 2 calls per step (verify + DFlash2 candidates)
    if args.shapes:
        SHAPES = [x for x in SHAPES if x[0] in args.shapes.split(",")]
MS = [8, 16, 32, 40, 64] if not args.quick else ([8, 16] if args.splitk else [8, 40])
if args.ms:
    MS = [int(x) for x in args.ms.split(",")]
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

def run_sk(a, w, s, cfg):
    return H.triton_w4a16_splitk_gemm(a, w, s, GS, cfg)


def bench_sk(a, ws, ss, cfg, iters):
    n = len(ws)
    for i in range(3): run_sk(a, ws[i % n], ss[i % n], cfg)
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for i in range(iters):
        ev[i][0].record(); run_sk(a, ws[i % n], ss[i % n], cfg); ev[i][1].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000 for s, e in ev)
    return t[len(t) // 2]


def splitk_cfgs(M):
    # (deq, unpack): 0/0 stock dequant, 1 scale after the dot, 2 magic-number + folded zero point,
    # unpack 1 interleave-free 8-dot. Split-K as the serial reduce (mode 2): it never costs more than the
    # buffer reduce (same partials, one launch less); atomic never won (window A).
    # BLOCK_M 16 at every M: M = 40 in 64-row tiles computes 37 % padding; 16-row tiles re-read the
    # weights per M tile, but from L2 (the sweep decides)
    bms = [16] if M <= 16 else ([16, 32] if M <= 32 else [16, 32, 64])
    sks = [1, 2, 4, 6, 8, 12, 16] if not args.quick else [1, 4, 8]
    dus = [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)]
    mds = [2, 0] if args.modes0 else [2]
    for bm, bn, nw, sk, (dq, up), md in itertools.product(bms, [32, 64, 128], [2, 4, 8], sks, dus, mds):
        if (sk, dq, up) == (1, 0, 0) or (nw == 8 and bn < 64) or (sk == 1 and md != 2):
            continue
        yield (bm, bn, 128, nw, None, sk, md if sk > 1 else 0, dq, up)


def partial_bytes(cfg, M, N):
    return 0 if cfg[5] <= 1 else (M * N * 4 if cfg[6] == 1 else cfg[5] * M * N * 4)


if args.splitk:
    results, table = {}, []
    new_table = {k: v for k, v in H._GFX12X_SPLITK.items() if k[3] not in MS}  # buckets not swept keep theirs
    for name, N, K in SHAPES:
        wbytes = N * K // 2 + N * (K // GS) * 2
        copies = max(2, (160 << 20) // wbytes)
        wr, ss = make(N, K, copies)
        wt = [H.radiance_w4a16_tile(w).view(torch.int32) for w in wr]
        ws = wr if args.rows else wt
        for M in MS:
            a = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            cur = H._gfx12x_draft_override(GS, K, N, M) or heuristic(M, N, K)
            cur7 = tuple(cur) + (1, 0)
            ref = run_sk(a, wr[0], ss[0], cur7).float()
            t_040 = bench_sk(a, wr, ss, cur7, args.iters)          # 0.4.0: tile table, row layout
            t_c = bench_sk(a, ws, ss, cur7, args.iters)            # tile table on the swept layout
            dflt = H._radiance_default_cfg(M)
            t_d = bench_sk(a, ws, ss, dflt, args.iters) if dflt else float("inf")
            base = min(t_c, t_d)
            best = (t_c, cur7); rows = []
            t0 = time.time()
            for cfg in splitk_cfgs(M):
                if partial_bytes(cfg, M, N) > args.max_partial_mib * (1 << 20):
                    continue
                try:
                    out = run_sk(a, ws[0], ss[0], cfg).float()
                    err = (out - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
                    if err > 2e-2: rows.append((cfg, None, err)); continue
                    t = bench_sk(a, ws, ss, cfg, args.iters)
                except Exception as e:
                    rows.append((cfg, None, str(e)[:60])); continue
                rows.append((cfg, t, err))
                if t < best[0]: best = (t, cfg)
            # stage 2: KSTEP 2 on the five fastest
            top = sorted([r for r in rows if r[1] is not None], key=lambda r: r[1])[:5]
            for cfg, _t, _e in top:
                c2 = tuple(cfg[:7]) + (cfg[7] if len(cfg) > 7 else 0, cfg[8] if len(cfg) > 8 else 0, 2)
                try:
                    t = bench_sk(a, ws, ss, c2, args.iters)
                except Exception as e:
                    rows.append((c2, None, str(e)[:60])); continue
                rows.append((c2, t, None))
                if t < best[0]: best = (t, c2)
            t_rows = bench_sk(a, wr, ss, best[1], args.iters)  # the same winner on the row layout (#3)
            def gb(t): return wbytes / (t * 1e-6) / 1e9
            print(f"{name:8s} N={N:5d} K={K:5d} M={M:2d}: 0.4.0 {t_040:6.1f} | tiles {t_c:6.1f} | default "
                  f"{t_d:6.1f} -> best {best[1]} {best[0]:6.1f} us ({gb(best[0]):4.0f} GB/s), rows {t_rows:6.1f}"
                  f"  {t_040 / best[0]:.2f}x vs 0.4.0  [{time.time() - t0:.0f}s]", flush=True)
            top = sorted([r for r in rows if r[1] is not None], key=lambda r: r[1])[:5]
            results[f"{name}:{M}"] = {"N": N, "K": K, "M": M, "v040": [cur7, t_040], "today": [cur7, t_c],
                                      "default": [dflt, t_d], "best": [best[1], best[0]], "best_rows": t_rows,
                                      "top5": top}
            if best[1] != cur7 and best[0] <= 0.95 * base:
                table.append(f"    ({GS}, {K}, {N}, {M}): {best[1]},  # {name}: {base:.0f} -> {best[0]:.0f} us")
                new_table[(GS, K, N, M)] = tuple(best[1])
        del wr, wt, ws, ss; torch.cuda.empty_cache()
    json.dump(results, open(args.out, "w"), indent=1)
    print("written", args.out)
    print("\n_GFX12X_SPLITK entries (>= 5 % faster than min(tile table, default)):")
    print("\n".join(table) if table else "    (none)")
    if args.table_out:
        json.dump({",".join(map(str, k)): list(v) for k, v in sorted(new_table.items())}, open(args.table_out, "w"),
                  indent=1)
        print("table written", args.table_out, len(new_table), "entries")
    sys.exit(0)

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
