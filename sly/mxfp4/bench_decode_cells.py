#!/usr/bin/env python3
"""Cell sweep of the W4A8 decode kernel over the M=16..64 band (radiance_mxfp4_fp8.hip, launch()).

Times every (shape, M, split-K, BK) cell of the decode kernel, DRAM-fed, and reports the best cell
per shape x M against what the shipped policy (split_k_for + decode_bk64) picks. Step 7 / P3 of the
vllm7 plan: the M<=8 band and the M>64 band were tuned cell by cell (tier7), the band in between --
2..8 concurrent DFlash sequences at SPEC=7 -- only by the fill rule.

Both knobs are cached statics in the .so, so every (ks, bk) combination runs in its own process:
the parent spawns one worker per combination with RADIANCE_MXFP4_DECODE_KS / _BK set, collects the
CSV rows and prints the table. Weights rotate over >= --rotate-mb of distinct copies per shape so
the 64 MB infinity cache never serves a repeat (a single gate_up buffer measured 2x the DRAM peak).

Throw-away container from the runtime image, GPU exclusive, no model needed (random weights;
timing does not depend on the values). The .so under test may be bind-mounted ahead of
site-packages (PYTHONPATH) -- radiance_mxfp4 prints which one it loaded.

  RADIANCE_MXFP4=1 RADIANCE_MXFP4_W4A8=1 RADIANCE_MXFP4_W4A8_MIN_M=0 RADIANCE_MXFP4_DECODE_MAX_M=128 \\
  python3 bench_decode_cells.py --csv /out/s7_cells.csv [--ms 16,24,32,40,48,56,64] [--iters 200]
"""
import argparse, csv, io, os, subprocess, sys

# (name, N, K): the six per-layer GEMMs of Qwen3.8-27B at TP=1 that reach the decode kernel.
# in_proj_ba (N=96, nblk 1) is a single n-block and always takes the widest split; not swept.
SHAPES = [
    ("gate_up", 34816, 5120),   # nblk 272
    ("qkvz",    16384, 5120),   # nblk 128  (in_proj_qkv + in_proj_z)
    ("qkv",     14336, 5120),   # nblk 112  (q + k + v, 16 attention layers)
    ("o_proj",   5120, 6144),   # nblk  40, K 6144
    ("out_proj", 5120, 6144),   # nblk  40, K 6144 (same shape as o_proj, distinct weights)
    ("down",     5120, 17408),  # nblk  40, K 17408
]
# BK=64 is instantiated for split 1 and 4 only (RAD_DEC_BY_KS); split 2 is BK=128 only.
COMBOS = [(1, 64), (1, 128), (2, 128), (4, 64), (4, 128)]
DEC_KS = 4


TUNE16 = os.environ.get("RADIANCE_MXFP4_DECODE_TUNE16", "1") != "0"


def policy(nblk, M, K):
    """Python replica of launch()'s (8, 64] band: split_k_for() + decode_bk64(), and on top of
    that the 0.2.1 cell table unless RADIANCE_MXFP4_DECODE_TUNE16=0 (which the worker inherits,
    so the measured "policy" column and this label always describe the same cell)."""
    fill = 110 if M <= 24 else 78
    ks = DEC_KS
    for c in (1, 2):
        if nblk * c >= fill:
            ks = c
            break
    tm = (M + 15) // 16
    bk = 64 if (tm == 4 and ks == 1) else 128
    if TUNE16 and M > 8:
        bk = 128
        if ks == DEC_KS and nblk > 8 and K < 8192 and 16 <= M <= 24:
            ks = 2
    return ks, bk


def worker(args):
    import torch
    import radiance_mxfp4 as R          # noqa  (loads the .so, honours the env knobs)
    assert R._ext is not None, "radiance_mxfp4_fp8 extension not loaded"
    assert R.DECODE_MAX_M >= 64, "set RADIANCE_MXFP4_DECODE_MAX_M=128"
    dev = torch.device("cuda")
    if not R._decode_scratch_ready[0]:
        R._decode_scratch_ready[0] = True
        R._decode_scratch[0] = torch.empty(4 * max(64, R.DECODE_MAX_M) * 36864, dtype=torch.float32, device=dev)
        R._decode_scratch[1] = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)
        R._ext.set_decode_scratch(R._decode_scratch[0].data_ptr(), R._decode_scratch[0].numel() * 4,
                                  R._decode_scratch[1].data_ptr())
    combo = args.combo
    ms = [int(m) for m in args.ms.split(",")]
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(0)
    out = csv.writer(sys.stdout)
    for name, N, K in SHAPES:
        wbytes = N * K // 2 + N * K // 32
        copies = max(2, -(-args.rotate_mb * (1 << 20) // wbytes))
        W = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev) for _ in range(copies)]
        WS = [torch.randint(120, 131, (K // 32, N), dtype=torch.uint8, device=dev) for _ in range(copies)]
        WR = [R.make_row_ref(ws) for ws in WS]
        for M in ms:
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            xq, xs = R._traced_quant(x)
            xs = xs.reshape(-1).contiguous().float()
            c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            def call(i):
                R._ext.launch(xq.data_ptr(), W[i].data_ptr(), WS[i].data_ptr(), WR[i].data_ptr(),
                              xs.data_ptr(), c.data_ptr(), M, N, K, stream)
            for i in range(20):
                call(i % copies)
            torch.cuda.synchronize()
            if args.check:
                ref = R._exact_ref(xq, xs, W[0], WS[0], N, K)
                call(0)
                torch.cuda.synchronize()
                rel = ((c.float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
                if rel > 0.02 or not torch.isfinite(c.float()).all():
                    print(f"WRONG {combo} {name} M={M} rel={rel:.4f}", file=sys.stderr)
                    sys.exit(2)
            reps = []
            for _ in range(args.reps):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for i in range(args.iters):
                    call(i % copies)
                e1.record()
                torch.cuda.synchronize()
                reps.append(e0.elapsed_time(e1) * 1000.0 / args.iters)
            us = sorted(reps)[len(reps) // 2]
            out.writerow([combo, name, N, K, M, f"{us:.2f}", f"{wbytes / us / 1e3:.1f}"])
            sys.stdout.flush()
        W = WS = WR = None       # free the rotation set before the next shape
        torch.cuda.empty_cache()


def parent(args):
    rows = []
    combos = [("policy", {})] + [(f"d{ks}/b{bk}", {"RADIANCE_MXFP4_DECODE_KS": str(ks),
                                                   "RADIANCE_MXFP4_DECODE_BK": str(bk)})
                                 for ks, bk in COMBOS]
    for combo, env in combos:
        e = dict(os.environ, **env)
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--combo", combo,
               "--ms", args.ms, "--iters", str(args.iters), "--reps", str(args.reps),
               "--rotate-mb", str(args.rotate_mb)] + (["--check"] if args.check else [])
        print(f"== {combo}: {env}", file=sys.stderr)
        p = subprocess.run(cmd, env=e, stdout=subprocess.PIPE, text=True)
        if p.returncode != 0:
            print(f"worker {combo} failed rc={p.returncode}", file=sys.stderr)
            sys.exit(p.returncode)
        rows += [r for r in csv.reader(io.StringIO(p.stdout)) if r]
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["combo", "shape", "N", "K", "M", "us", "GBps"])
            w.writerows(rows)
    # table: per shape x M, policy cell vs best cell
    cell = {(r[0], r[1], int(r[4])): (float(r[5]), float(r[6])) for r in rows}
    ms = [int(m) for m in args.ms.split(",")]
    print(f"{'shape':9s} {'M':>3s}  {'policy':>10s} {'us':>7s} {'GB/s':>6s} | "
          f"{'best':>10s} {'us':>7s} {'GB/s':>6s}  {'gain':>6s} | " +
          " ".join(f"{c:>8s}" for c in [f'd{k}/b{b}' for k, b in COMBOS]))
    for name, N, K in SHAPES:
        nblk = (N + 127) // 128
        for M in ms:
            pk, pb = policy(nblk, M, K)
            plabel = f"d{pk}/b{pb}"
            pus = cell.get(("policy", name, M))
            cells = {f"d{k}/b{b}": cell[(f"d{k}/b{b}", name, M)] for k, b in COMBOS
                     if (f"d{k}/b{b}", name, M) in cell}
            best = min(cells, key=lambda k: cells[k][0])
            base = pus[0] if pus else cells[plabel][0]
            gain = (base - cells[best][0]) / base * 100
            print(f"{name:9s} {M:3d}  {plabel:>10s} {base:7.1f} {(pus or cells[plabel])[1]:6.0f} | "
                  f"{best:>10s} {cells[best][0]:7.1f} {cells[best][1]:6.0f}  {gain:5.1f}% | " +
                  " ".join(f"{cells[c][0]:8.1f}" if c in cells else f"{'-':>8s}"
                           for c in [f'd{k}/b{b}' for k, b in COMBOS]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="16,24,32,40,48,56,64")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rotate-mb", type=int, default=160)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--check", action="store_true", help="verify each cell against the fp32 reference")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--combo", default="policy")
    a = ap.parse_args()
    worker(a) if a.worker else parent(a)
