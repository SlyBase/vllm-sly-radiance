#!/usr/bin/env python3
"""Isolated kernel bench for the wide decode band (RADIANCE_MXFP4_WIDE_MAX_M) and the neighbours.

One worker process per ARM (the knobs are cached statics in the .so). Every arm times
_ext.launch() on the real production GEMM shapes, DRAM-fed (weights rotate over >= --rotate-mb of
distinct copies), and compares its output with the first arm ("cur" = unchanged image behaviour):

  cur    production env as is (DECODE_MAX_M=128, WIDE off)           -> the control
  wide   WIDE_MAX_M=256, table split-K                                -> the new band (M > 128 only differs)
  w1/w2/w4  WIDE_MAX_M=256 + WIDE_KS=k (forced split)                 -> cell sweep for the table (M > 128)
  nt     WIDE_MAX_M=256 + DECODE_NT=1                                 -> streaming loads (all M)
  tn4    TN4_MIN_M=129                                                -> folded kernel with the 128-wide tile

Per (shape, M): us per arm, % vs cur, the best arm, and a per-step weighted GEMM sum (layer counts of
Qwen3.8-27B at TP=1: gate_up/down 64, qkvz/in_proj_ba/out_proj 48, qkv/o_proj 16) -- that sum is the
decision metric, as in Radiance's per-prefill-step GEMM total.

Exactness (every cell): relative error against the fp32 reference (<0.02, same gate as the image's
CHECKALL), and against the cur arm's output (<=3e-3 = one bf16 ulp; bit-equal is reported when it
happens). ks=1 should be bit-equal to the folded kernel's order of accumulation only by luck, so the
gate is the tolerance, not equality.

Run (GPU exclusive, no model):  python3 bench_wide_cells.py --out /out [--ms ...] [--iters 100]
"""
import argparse, csv, io, os, subprocess, sys

SHAPES = [   # name, N, K, layers per step
    ("gate_up",  34816, 5120, 64),
    ("qkvz",     16384, 5120, 48),
    ("qkv",      14336, 5120, 16),
    ("o_proj",    5120, 6144, 16),
    ("out_proj",  5120, 6144, 48),
    ("down",      5120, 17408, 64),
    ("in_proj_ba",  96, 5120, 48),
]
DEFAULT_MS = "16,24,32,48,64,72,96,128,160,192,256,384,512"
ARMS = [
    ("cur",  {}),
    ("wide", {"RADIANCE_MXFP4_WIDE_MAX_M": "256"}),
    ("w1",   {"RADIANCE_MXFP4_WIDE_MAX_M": "256", "RADIANCE_MXFP4_WIDE_KS": "1"}),
    ("w2",   {"RADIANCE_MXFP4_WIDE_MAX_M": "256", "RADIANCE_MXFP4_WIDE_KS": "2"}),
    ("w4",   {"RADIANCE_MXFP4_WIDE_MAX_M": "256", "RADIANCE_MXFP4_WIDE_KS": "4"}),
    ("nt",   {"RADIANCE_MXFP4_WIDE_MAX_M": "256", "RADIANCE_MXFP4_DECODE_NT": "1"}),
    ("tn4",  {"RADIANCE_MXFP4_TN4_MIN_M": "129"}),
]
ONLY_ABOVE_128 = {"w1", "w2", "w4", "tn4"}      # arms that cannot differ from cur at M <= 128


def worker(a):
    import torch
    import radiance_mxfp4 as R
    assert R._ext is not None, "radiance_mxfp4_fp8 extension not loaded"
    dev = torch.device("cuda")
    R._decode_scratch[0] = torch.empty(4 * max(64, R.DECODE_MAX_M) * 36864, dtype=torch.float32, device=dev)
    R._decode_scratch[1] = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)
    R._ext.set_decode_scratch(R._decode_scratch[0].data_ptr(), R._decode_scratch[0].numel() * 4,
                              R._decode_scratch[1].data_ptr())
    ms = [int(m) for m in a.ms.split(",")]
    stream = torch.cuda.current_stream().cuda_stream
    w = csv.writer(sys.stdout)
    refdir = os.path.join(a.out, "ref"); os.makedirs(refdir, exist_ok=True)
    for name, N, K, _ in SHAPES:
        g = torch.Generator(device=dev); g.manual_seed(1234)
        wbytes = N * K // 2 + N * K // 32
        copies = max(2, -(-a.rotate_mb * (1 << 20) // wbytes))
        W = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev, generator=g) for _ in range(copies)]
        WS = [torch.randint(120, 131, (K // 32, N), dtype=torch.uint8, device=dev, generator=g) for _ in range(copies)]
        WR = [R.make_row_ref(s) for s in WS]
        for M in ms:
            if a.arm in ONLY_ABOVE_128 and M <= 128:
                continue
            g.manual_seed(1000 + M)          # same input per (shape, M) in every arm, whatever was skipped
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16, generator=g)
            xq, xs = R._traced_quant(x)
            xs = xs.reshape(-1).contiguous().float()
            c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            def call(i):
                R._ext.launch(xq.data_ptr(), W[i].data_ptr(), WS[i].data_ptr(), WR[i].data_ptr(),
                              xs.data_ptr(), c.data_ptr(), M, N, K, stream)
            call(0); torch.cuda.synchronize()
            ref = R._exact_ref(xq, xs, W[0], WS[0], N, K)
            rel_ref = ((c.float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
            finite = bool(torch.isfinite(c.float()).all())
            path = os.path.join(refdir, f"{name}_{M}.pt")
            if a.arm == "cur":
                torch.save(c.cpu(), path); rel_cur, bit, mx = 0.0, 1, 0.0
            else:
                r0 = torch.load(path).to(dev)
                d = (c.float() - r0.float())
                rel_cur = (d.norm() / r0.float().norm().clamp_min(1e-9)).item()
                bit = int(bool((c == r0).all())); mx = d.abs().max().item()
            for i in range(20):
                call(i % copies)
            torch.cuda.synchronize()
            reps = []
            for _ in range(a.reps):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                for i in range(a.iters):
                    call(i % copies)
                e1.record(); torch.cuda.synchronize()
                reps.append(e0.elapsed_time(e1) * 1000.0 / a.iters)
            us = sorted(reps)[len(reps) // 2]
            w.writerow([a.arm, name, N, K, M, f"{us:.2f}", f"{rel_ref:.5f}", f"{rel_cur:.6f}", bit,
                        f"{mx:.4g}", int(finite)])
            sys.stdout.flush()
        W = WS = WR = None
        torch.cuda.empty_cache()


def parent(a):
    rows = []
    for arm, env in ARMS:
        e = dict(os.environ, **env)
        print(f"== arm {arm}: {env}", file=sys.stderr, flush=True)
        p = subprocess.run([sys.executable, os.path.abspath(__file__), "--worker", "--arm", arm,
                            "--ms", a.ms, "--iters", str(a.iters), "--reps", str(a.reps),
                            "--rotate-mb", str(a.rotate_mb), "--out", a.out],
                           env=e, stdout=subprocess.PIPE, text=True)
        if p.returncode != 0:
            print(f"arm {arm} failed rc={p.returncode}", file=sys.stderr); sys.exit(2)
        rows += [r for r in csv.reader(io.StringIO(p.stdout)) if r]
    with open(os.path.join(a.out, "cells.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["arm", "shape", "N", "K", "M", "us", "rel_ref", "rel_cur", "bit_equal", "maxabs", "finite"])
        cw.writerows(rows)
    cell = {(r[0], r[1], int(r[4])): r for r in rows}
    ms = [int(m) for m in a.ms.split(",")]
    arms = [x for x, _ in ARMS]
    layers = {n: l for n, _, _, l in SHAPES}
    bad = []
    print("\n#### per cell: us (cur) and delta % vs cur per arm (negative = faster); [ref|cur] = errors")
    print(f"{'shape':10s} {'M':>4s} {'cur':>8s} | " + " ".join(f"{x:>9s}" for x in arms[1:]) + " | best")
    for name, N, K, _ in SHAPES:
        for M in ms:
            c0 = cell.get(("cur", name, M))
            if not c0:
                continue
            base = float(c0[5]); parts = []; best, bu = "cur", base
            for x in arms[1:]:
                r = cell.get((x, name, M))
                if not r:
                    parts.append(f"{'-':>9s}"); continue
                u = float(r[5]); parts.append(f"{(u - base) / base * 100:+8.1f}%")
                if u < bu: best, bu = x, u
                if float(r[6]) > 0.02 or float(r[7]) > 3e-3 or not int(r[10]):
                    bad.append((x, name, M, r[6], r[7]))
            print(f"{name:10s} {M:4d} {base:8.1f} | " + " ".join(parts) + f" | {best} {(bu - base) / base * 100:+.1f}%")
    print("\n#### per-step weighted GEMM sum (us, one forward over the 7 shapes x layer counts)")
    print(f"{'M':>4s} {'cur':>9s} | " + " ".join(f"{x:>9s}" for x in arms[1:]) + " | pick(wide) vs cur")
    for M in ms:
        def tot(x):
            t = 0.0
            for name, _, _, _ in SHAPES:
                r = cell.get((x, name, M)) or cell.get(("cur", name, M))
                t += float(r[5]) * layers[name]
            return t
        base = tot("cur")
        print(f"{M:4d} {base:9.0f} | " + " ".join(f"{tot(x):9.0f}" for x in arms[1:]) +
              f" | wide {(tot('wide') - base) / base * 100:+.1f}%  tn4 {(tot('tn4') - base) / base * 100:+.1f}%"
              f"  nt {(tot('nt') - base) / base * 100:+.1f}%")
    print("\n#### exactness: " + ("FAIL " + str(len(bad)) + " cells (arm shape M rel_ref rel_cur): " + str(bad[:12])
                                   if bad else "PASS (all cells rel_ref < 0.02, rel_cur <= 3e-3, finite)"))
    nbit = sum(1 for r in rows if r[0] != "cur" and r[8] == "1"); nall = sum(1 for r in rows if r[0] != "cur")
    print(f"bit-equal to cur: {nbit}/{nall} cells")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default=DEFAULT_MS)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rotate-mb", type=int, default=160)
    ap.add_argument("--out", default="/out")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--arm", default="cur")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    worker(a) if a.worker else parent(a)
