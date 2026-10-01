#!/usr/bin/env python3
"""Row-major vs fragment-tiled cell sweep of the W4A8 GEMM over the prefill band M=96..8192.

bench_decode_cells.py covers the decode kernel M=16..64 only. Nothing covers prefill: the
A-tiled path is documented as a single "12-16% faster at M >= 2048" spot check, the mid-M band
129..512 (transfer plan P2, where production hands rows to the row-major kernel below the
A_TILED_MIN_M=513 threshold) has no data at all, so this sweep starts at min_m=96 to cover the full band below and
across the threshold.
This script times _ext.launch (row-major A) against _ext.launch_at (A-tiled) per (shape, M) on
the six real GEMMs of Qwen3.8-27B at TP=1.

Tiled activations come from the REAL producer -- torch.ops.radiance.add_rms_quant with the
tiled flag forced on by patching radiance_fused_norm._rmx_tiled_wanted -- so the fragment
layout is whatever the .hip expects, never re-implemented. "Values are identical in both
layouts" (radiance_fused_norm.py) is exactly what --check verifies per cell: both arms are
produced from the same bf16 input, the row-major arm's (q, scale) feeds the fp32 reference,
and the tiled arm's output must match it.

Shape-specific notes (production wiring, radiance_fused_norm.py):
  - qkvz/qkv/gate_up (K=5120): add_rms_quant wired, tiled in production from M=513.
  - o_proj/out_proj (K=6144): gdn_norm_quant has no tiled variant, production stays row-major;
    measured tiled here anyway (K=6144 fits the add_rms K<=10240 limit) to show what a tiled
    producer would buy.
  - down (K=17408): NO tiled producer exists (add_rms K<=10240, and silu_mul_quant's N<=18432
    excludes down's 34816-wide gate_up strip at TP=1). Tiled arm skipped; that gap is itself a
    finding -- a MAXG-16 silu or a second-stage producer is the only way to tile down.

Weights rotate over >= --rotate-mb distinct copies per shape so the 64 MB infinity cache never
serves a weight repeat (as in bench_decode_cells.py). A stays cache-resident in both arms; W
dominates DRAM at these shapes, so the arms stay comparable and GBps_w matches the decode
bench's accounting. GBps_total adds A and the bf16 output.

M<=128 takes the decode split-K branch inside launch() exactly as production does
(DECODE_MAX_M=128); the scratch setup mirrors bench_decode_cells.py.

Throw-away container from the runtime image, GPU exclusive, no model needed. The .so under
test may be bind-mounted ahead of site-packages (PYTHONPATH) -- radiance_mxfp4 prints which
one it loaded. Run --check first; every cell must validate against the fp32 reference before
timings mean anything. If cells report WRONG, your env perms weights inconsistently with the
W construction (see RADIANCE_MXFP4_WPERM) -- match whatever your bench_decode_cells runs use.

  RADIANCE_MXFP4=1 RADIANCE_MXFP4_W4A8=1 RADIANCE_MXFP4_W4A8_MIN_M=0 \
  RADIANCE_MXFP4_DECODE_MAX_M=128 RADIANCE_MXFP4_A_TILED_MIN_M=0 RADIANCE_FUSED_NORM_QUANT=1 \
  python3 bench_prefill_cells.py --csv /out/prefill_cells.csv --check \
      [--ms 96,128,160,192,256,320,384,512,576,768,1024,1536,2048,3072,4096,6144,8192] \
      [--iters 100] [--reps 3] [--rotate-mb 256]
"""
import argparse
import csv
import os
import sys

import torch

# (name, N, K): the six per-layer GEMMs of Qwen3.8-27B at TP=1 that reach the W4A8 kernel.
# Same set as bench_decode_cells.py. TILED_OK = shapes the add_rms producer can tile:
# K % 128 == 0 (the fragment layout) and K <= 10240 (the launcher's _ADD_RMS_MAX_K).
SHAPES = [
    ("gate_up", 34816, 5120),
    ("qkvz",    16384, 5120),
    ("qkv",     14336, 5120),
    ("o_proj",   5120, 6144),
    ("out_proj", 5120, 6144),
    ("down",     5120, 17408),
]
TILED_OK = {"gate_up", "qkvz", "qkv", "o_proj", "out_proj"}
# Production tile wiring: add_rms-quant consumers only. down/o_proj/out_proj stay row-major.
PROD_TILED = {"gate_up", "qkvz", "qkv"}
PROD_AT_MIN_M = 513  # RADIANCE_MXFP4_A_TILED_MIN_M in the README quickstart


def produce(fn, y, res, wgt, eps):
    """One add_rms_quant call; the tiled flag is whatever radiance_fused_norm decides now."""
    q, s, _ = fn(y, res, wgt, eps)
    return q, s


def time_arm(call, copies, iters, reps, check=None):
    for i in range(20):
        call(i % copies if copies else 0)
    torch.cuda.synchronize()
    if check is not None:
        check()
    times = []
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for i in range(iters):
            call(i % copies if copies else 0)
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1) * 1000.0 / iters)
    return sorted(times)[len(times) // 2]


def main(a):
    import radiance_mxfp4 as R
    import radiance_fused_norm as F
    assert R._ext is not None, "radiance_mxfp4_fp8 extension not loaded"
    assert F._ext is not None, ("RADIANCE_FUSED_NORM_QUANT=1 required "
                                "(the tiled arm needs the real producer)")
    dev = torch.device("cuda")
    if R.DECODE_MAX_M and not R._decode_scratch_ready[0]:
        R._decode_scratch_ready[0] = True
        R._decode_scratch[0] = torch.empty(4 * max(64, R.DECODE_MAX_M) * 36864,
                                           dtype=torch.float32, device=dev)
        R._decode_scratch[1] = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)
        R._ext.set_decode_scratch(R._decode_scratch[0].data_ptr(),
                                  R._decode_scratch[0].numel() * 4,
                                  R._decode_scratch[1].data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    ms = [int(m) for m in a.ms.split(",")]
    rows = []
    torch.manual_seed(0)
    for name, N, K in SHAPES:
        wbytes = N * K // 2 + N * K // 32
        copies = max(2, -(-a.rotate_mb * (1 << 20) // wbytes))
        W = [torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
             for _ in range(copies)]
        WS = [torch.randint(120, 131, (K // 32, N), dtype=torch.uint8, device=dev)
              for _ in range(copies)]
        WR = [R.make_row_ref(ws) for ws in WS]
        print(f"== {name}: N={N} K={K} ({copies} weight copies, {wbytes >> 20} MB each)",
              file=sys.stderr)
        for M in ms:
            y = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            res = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            wgt = torch.ones(K, device=dev, dtype=torch.bfloat16)
            out = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
            fn = torch.ops.radiance.add_rms_quant
            # row-major arm: producer unforced (RADIANCE_MXFP4_A_TILED_MIN_M=0 -> row-major)
            q_rm, s_rm = produce(fn, y, res, wgt, 1e-6)
            # tiled arm: force the producer's tiled decision for this call, then restore
            saved = F._rmx_tiled_wanted
            F._rmx_tiled_wanted = lambda M_: True
            try:
                q_at = s_at = None
                if name in TILED_OK:
                    q_at, s_at = produce(fn, y, res, wgt, 1e-6)
            finally:
                F._rmx_tiled_wanted = saved

            def rm_call(i):
                R._ext.launch(q_rm.data_ptr(), W[i].data_ptr(), WS[i].data_ptr(),
                              WR[i].data_ptr(), s_rm.data_ptr(), out.data_ptr(),
                              M, N, K, stream)

            ref = None
            rm_check = None
            if a.check:
                ref = R._exact_ref(q_rm, s_rm, W[0], WS[0], N, K)

                def rm_check():
                    rm_call(0)
                    torch.cuda.synchronize()
                    rel = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
                    if rel > 0.02 or not torch.isfinite(out.float()).all():
                        print(f"WRONG {name} M={M} arm=rowmajor rel={rel:.4f}",
                              file=sys.stderr)
                        sys.exit(2)

            us_rm = time_arm(rm_call, copies, a.iters, a.reps, check=ref is not None
                             and rm_check or None)
            rows.append((name, N, K, M, "rowmajor", us_rm, wbytes, wbytes + M * K + M * N * 2))

            us_at = None
            if q_at is not None:
                def at_call(i):
                    R._ext.launch_at(q_at.data_ptr(), W[i].data_ptr(), WS[i].data_ptr(),
                                     WR[i].data_ptr(), s_at.data_ptr(), out.data_ptr(),
                                     M, N, K, stream)

                if a.check:
                    def at_check():
                        at_call(0)
                        torch.cuda.synchronize()
                        rel = ((out.float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
                        if rel > 0.02 or not torch.isfinite(out.float()).all():
                            print(f"WRONG {name} M={M} arm=atiled rel={rel:.4f}",
                                  file=sys.stderr)
                            sys.exit(2)
                else:
                    at_check = None
                us_at = time_arm(at_call, copies, a.iters, a.reps, check=at_check)
                rows.append((name, N, K, M, "atiled", us_at, wbytes,
                             wbytes + M * K + M * N * 2))
            # out / q_* / s_* stay referenced by the closures above (rm_call, at_call),
            # so they free when the closures die at the end of this iteration
            del y, res, wgt
        W = WS = WR = None
        torch.cuda.empty_cache()

    if a.csv:
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["shape", "N", "K", "M", "arm", "us", "GBps_w", "GBps_total"])
            for name, N, K, M, arm, us, wb, tb in rows:
                w.writerow([name, N, K, M, arm, f"{us:.2f}", f"{wb / us / 1e3:.1f}",
                            f"{tb / us / 1e3:.1f}"])

    cell = {(r[0], r[3], r[4]): r[5] for r in rows}
    print(f"{'shape':9s} {'M':>5s}  {'rowmajor':>9s} {'atiled':>9s} {'delta':>7s}  "
          "prod choice (A_TILED_MIN_M=513, wired producers only)")
    for name, N, K in SHAPES:
        for M in ms:
            rm = cell.get((name, M, "rowmajor"))
            at = cell.get((name, M, "atiled"))
            if rm is None:
                continue
            if name in PROD_TILED:
                prod = f"{'atiled' if M >= PROD_AT_MIN_M else 'rowmajor'}"
            else:
                prod = "rowmajor" + ("" if name in TILED_OK else "  (no tiled producer)")
            if at is None:
                print(f"{name:9s} {M:5d}  {rm:9.1f} {'n/a':>9s} {'-':>7s}  {prod}")
            else:
                d = (rm - at) / rm * 100.0
                print(f"{name:9s} {M:5d}  {rm:9.1f} {at:9.1f} {d:+6.1f}%  {prod}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="96,128,160,192,256,320,384,512,576,768,1024,1536,"
                                    "2048,3072,4096,6144,8192")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rotate-mb", type=int, default=256)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--check", action="store_true",
                    help="verify both arms against the fp32 reference before timing")
    a = ap.parse_args()
    main(a)
