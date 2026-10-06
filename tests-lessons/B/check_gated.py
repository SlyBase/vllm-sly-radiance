#!/usr/bin/env python3
"""RADIANCE_MXFP4_GATED_FOLD isolated check + microbench (GPU, no model).

For M in 8..64 (every M, so the DTM 1..4 tiles and the partial last fragment are all hit) and the
production gate_up shape (N=34816, K=5120) plus a narrower one (N=16384, K=2048, still nblk >= 110 so DKS=1), compares the fused pair
  launch_gated + launch_quant_rows
against the unfused pair the production graph runs today
  launch (decode GEMM, bf16 gate_up) + launch_silu_mul_quant
BYTE for BYTE (q e4m3 [M, H] and scale bits [M]). Then times both, DRAM-fed (weight copies rotated
over >= 512 MB so the 64 MB MALL never serves a repeat), A/B/A.

Env: same as production (W4A8, DECODE_MAX_M=128, WPERM=1); the script sets them if unset.
  python3 check_gated.py [--iters 200] [--rotate-mb 512] [--ms 8,16,...]
Exit code 1 on any mismatch.
"""
import argparse
import os
import sys

for k, v in (("RADIANCE_MXFP4", "1"), ("RADIANCE_MXFP4_W4A8", "1"), ("RADIANCE_MXFP4_W4A8_MIN_M", "0"),
             ("RADIANCE_MXFP4_DECODE_MAX_M", "128"), ("RADIANCE_MXFP4_WPERM", "1")):
    os.environ.setdefault(k, v)

import torch  # noqa: E402
import radiance_mxfp4 as R  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--iters", type=int, default=200)
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--rotate-mb", type=int, default=512)
ap.add_argument("--ms", default=",".join(str(m) for m in range(8, 65)))
ap.add_argument("--bench-ms", default="8,16,24,32,40,48,56,64")
args = ap.parse_args()

E = R._ext
assert E is not None and hasattr(E, "launch_gated"), "extension missing launch_gated (mount the new .so)"
dev = torch.device("cuda")
if not R._decode_scratch_ready[0]:
    R._decode_scratch_ready[0] = True
    R._decode_scratch[0] = torch.empty(4 * 128 * 36864, dtype=torch.float32, device=dev)
    R._decode_scratch[1] = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)
    E.set_decode_scratch(R._decode_scratch[0].data_ptr(), R._decode_scratch[0].numel() * 4,
                         R._decode_scratch[1].data_ptr())
stream = torch.cuda.current_stream().cuda_stream
torch.manual_seed(0)
WPERM = R.WPERM
bad = 0


def make_w(N, K):
    w = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
    ws = torch.randint(120, 131, (K // 32, N), dtype=torch.uint8, device=dev)
    wr = R.make_row_ref(ws)
    if WPERM:
        w = R.permute_w(w, N, K)
    return w, ws, wr


def quant(M, K):
    # activations with a few outliers so the per-token scale is not trivially 1
    x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    x[:, ::97] *= 12
    xq, xs = R._traced_quant(x)
    return xq, xs.reshape(-1).contiguous().float()


def unfused(xq, xs, w, ws, wr, M, N, K, gu, q, s):
    E.launch(xq.data_ptr(), w.data_ptr(), ws.data_ptr(), wr.data_ptr(), xs.data_ptr(), gu.data_ptr(),
             M, N, K, stream)
    E.launch_silu_mul_quant(gu=gu.data_ptr(), q=q.data_ptr(), scale=s.data_ptr(), M=M, N=N // 2,
                            stream=stream, tiled=0)


def fused(xq, xs, w, ws, wr, M, N, K, p, q, s):
    ok = E.launch_gated(xq.data_ptr(), w.data_ptr(), ws.data_ptr(), wr.data_ptr(), xs.data_ptr(),
                        p.data_ptr(), M, N, K, stream)
    assert ok, f"launch_gated declined M={M} N={N} K={K}"
    E.launch_quant_rows(p.data_ptr(), q.data_ptr(), s.data_ptr(), M, N // 2, stream)


SHAPES = [("gate_up", 34816, 5120), ("mid", 16384, 2048)]
print(f"# WPERM={int(WPERM)} DECODE_MAX_M={R.DECODE_MAX_M}")
for name, N, K in SHAPES:
    H = N // 2
    w, ws, wr = make_w(N, K)
    nbad = 0
    for M in [int(m) for m in args.ms.split(",")]:
        xq, xs = quant(M, K)
        gu = torch.zeros(M, N, device=dev, dtype=torch.bfloat16)
        q0 = torch.zeros(M, H, device=dev, dtype=torch.float8_e4m3fn)
        s0 = torch.zeros(M, device=dev, dtype=torch.float32)
        p = torch.zeros(M, H, device=dev, dtype=torch.bfloat16)
        q1 = torch.zeros(M, H, device=dev, dtype=torch.float8_e4m3fn)
        s1 = torch.zeros(M, device=dev, dtype=torch.float32)
        unfused(xq, xs, w, ws, wr, M, N, K, gu, q0, s0)
        fused(xq, xs, w, ws, wr, M, N, K, p, q1, s1)
        torch.cuda.synchronize()
        qd = (q0.view(torch.uint8) != q1.view(torch.uint8)).sum().item()
        sd = (s0.view(torch.int32) != s1.view(torch.int32)).sum().item()
        # the bf16 product against torch's own silu*mul of the unfused gu (informational: expf vs
        # torch's silu may differ by an ulp before the bf16 rounding; the gate is q/scale above)
        g, u = gu[:, :H].float(), gu[:, H:].float()
        pr = (torch.nn.functional.silu(g).to(torch.bfloat16).float() * u).to(torch.bfloat16)
        pd = (pr.view(torch.int16) != p.view(torch.int16)).sum().item()
        if qd or sd:
            nbad += 1
            print(f"MISMATCH {name} M={M}: q bytes differ {qd}/{M * H}, scale bits differ {sd}/{M}"
                  f" (product vs torch ref differs {pd})")
        elif M in (8, 24, 64):
            print(f"  ok {name} M={M}: q {M * H} B identical, scale identical "
                  f"(product vs torch silu*mul: {pd} of {M * H} bf16 differ, informational)")
    print(f"{name}: {'EXACT for all M' if not nbad else f'{nbad} M values MISMATCH'}")
    bad += nbad
    del w, ws, wr

# ---- microbench (gate_up only, DRAM-fed) ------------------------------------------------------
name, N, K = SHAPES[0]
H = N // 2
wbytes = N * K // 2 + N * K // 32
copies = max(2, -(-args.rotate_mb * (1 << 20) // wbytes))
Ws = [make_w(N, K) for _ in range(copies)]
print(f"\n# microbench {name} N={N} K={K}, {copies} weight copies, {args.iters} iters x {args.reps} reps, median")
print(f"{'M':>3} {'unfused us':>11} {'fused us':>9} {'delta us':>9} {'delta %':>8}   (x64 layers = ms/step)")
tot = 0.0
bms = [int(m) for m in args.bench_ms.split(",")]
for M in bms:
    xq, xs = quant(M, K)
    gu = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
    p = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    q = torch.empty(M, H, device=dev, dtype=torch.float8_e4m3fn)
    s = torch.empty(M, device=dev, dtype=torch.float32)

    def run(fn, buf):
        for i in range(20):
            fn(xq, xs, *Ws[i % copies], M, N, K, buf, q, s)
        torch.cuda.synchronize()
        out = []
        for _ in range(args.reps):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for i in range(args.iters):
                fn(xq, xs, *Ws[i % copies], M, N, K, buf, q, s)
            e1.record()
            torch.cuda.synchronize()
            out.append(e0.elapsed_time(e1) * 1000.0 / args.iters)
        return sorted(out)[len(out) // 2]
    # A/B/A so drift shows
    a1 = run(unfused, gu)
    b = run(fused, p)
    a2 = run(unfused, gu)
    a = 0.5 * (a1 + a2)
    d = b - a
    tot += d
    print(f"{M:3d} {a:11.2f} {b:9.2f} {d:+9.2f} {100 * d / a:+7.1f}%   {d * 64 / 1000:+.3f} ms  (A1 {a1:.2f} A2 {a2:.2f})")
print(f"\nmean delta over the bench Ms: {tot / len(bms):+.2f} us/layer")
print("RESULT:", "EXACT (fused == unfused bytes for every M in the list)" if not bad else f"FAIL ({bad} mismatching cells)")
sys.exit(1 if bad else 0)
