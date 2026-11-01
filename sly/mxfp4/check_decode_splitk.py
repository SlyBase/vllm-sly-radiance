#!/usr/bin/env python3
"""Cross-CU visibility check for the W4A8 decode kernel's fused split-K reduction (0.7.1).

radiance_mxfp4_fp8_gemm_decode writes DKS fp32 partials per n-block and the last block to bump the
counter sums them. Up to 0.7.0 only thread 0 fenced before the bump, so the reducer could read
partials the other threads of a sibling block had not made visible yet -- i.e. whatever the slot
held from the PREVIOUS call. With identical inputs every call a stale slot holds the same numbers,
so the race is invisible; this check therefore alternates two activations (and two outputs) on
every launch, back to back without a sync, so a stale partial is always a wrong one.

The reduction order is fixed (k = 0..DKS-1), so a correct kernel is bit-for-bit repeatable:
every output is compared with torch.equal against that input's golden (taken after a full sync,
and itself checked against the fp32 reference). One worker process per split (the KS knob is a
cached static in the .so): "policy" = the production routing, then DKS forced to 2 and 4 for
M <= 64 (the M > 64 band keeps its table). With RADIANCE_MXFP4_WIDE_MAX_M set (1.0 production: 192)
the wide band M in (128, WIDE_MAX_M] runs the same kernel template and is checked too (its split is
forced through RADIANCE_MXFP4_WIDE_KS); M above both bands is skipped. The scratch is sized exactly as
radiance_mxfp4.py sizes it, so a split that does not fit takes the same fallback as in production.

Throw-away container from the image under test, GPU exclusive, no model (random weights):

  RADIANCE_MXFP4=1 RADIANCE_MXFP4_W4A8=1 RADIANCE_MXFP4_W4A8_MIN_M=0 RADIANCE_MXFP4_DECODE_MAX_M=128 \\
  RADIANCE_MXFP4_WPERM=1 RADIANCE_MXFP4_WIDE_MAX_M=192 \\
  python3 check_decode_splitk.py [--iters 2000] [--ms 5,8,16,24,40,64,96,128,160,192]

Exit 0 = every launch bit-identical; 2 = a mismatch (printed with shape/M/split/count).
"""
import argparse, os, subprocess, sys

# (name, N, K): the per-layer GEMMs of Qwen3.8-27B at TP=1 that reach the decode kernel.
SHAPES = [
    ("gate_up", 34816, 5120),    # nblk 272
    ("qkvz",    16384, 5120),    # nblk 128
    ("qkv",     14336, 5120),    # nblk 112
    ("o_proj",   5120, 6144),    # nblk  40 -> DKS 2/4 at M <= 64
    ("down",     5120, 17408),   # nblk  40, 136 k-slabs
    ("ba",         96, 5120),    # nblk   1 -> always the widest split
]
COMBOS = [("policy", {}),
          ("ks2", {"RADIANCE_MXFP4_DECODE_KS": "2", "RADIANCE_MXFP4_WIDE_KS": "2"}),
          ("ks4", {"RADIANCE_MXFP4_DECODE_KS": "4", "RADIANCE_MXFP4_WIDE_KS": "4"})]
BATCH = 32          # outputs kept per input before they are compared


def worker(args):
    import torch
    import radiance_mxfp4 as R
    assert R._ext is not None, "radiance_mxfp4_fp8 extension not loaded"
    ms = [int(m) for m in args.ms.split(",")]
    skip = [m for m in ms if not (m <= R.DECODE_MAX_M or 128 < m <= R.WIDE_MAX_M)]
    if skip:
        print(f"skip M={skip}: outside DECODE_MAX_M={R.DECODE_MAX_M} / WIDE_MAX_M={R.WIDE_MAX_M}",
              file=sys.stderr)
    ms = [m for m in ms if m not in skip]
    assert ms, "no M inside the decode bands: raise RADIANCE_MXFP4_DECODE_MAX_M / _WIDE_MAX_M"
    dev = torch.device("cuda")
    if not R._decode_scratch_ready[0]:
        R._decode_scratch_ready[0] = True
        R._decode_scratch[0] = torch.empty(4 * max(64, R.DECODE_MAX_M) * 36864, dtype=torch.float32, device=dev)
        R._decode_scratch[1] = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)
        R._ext.set_decode_scratch(R._decode_scratch[0].data_ptr(), R._decode_scratch[0].numel() * 4,
                                  R._decode_scratch[1].data_ptr())
    stream = torch.cuda.current_stream().cuda_stream
    torch.manual_seed(0)
    bad = 0
    for name, N, K in SHAPES:
        W = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
        if R.WPERM:
            W = R.permute_w(W, N, K)
        WS = torch.randint(120, 131, (K // 32, N), dtype=torch.uint8, device=dev)
        WR = R.make_row_ref(WS)
        for M in ms:
            xs_ = []
            for _ in range(2):
                xq, xsc = R._traced_quant(torch.randn(M, K, device=dev, dtype=torch.bfloat16))
                xs_.append((xq, xsc.reshape(-1).contiguous().float()))
            outs = [torch.empty(BATCH, M, N, device=dev, dtype=torch.bfloat16) for _ in range(2)]

            def call(j, c):
                xq, xsc = xs_[j]
                R._ext.launch(xq.data_ptr(), W.data_ptr(), WS.data_ptr(), WR.data_ptr(),
                              xsc.data_ptr(), c.data_ptr(), M, N, K, stream)

            gold = []
            for j in range(2):
                g = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                call(j, g)
                torch.cuda.synchronize()
                ref = R._exact_ref(xs_[j][0], xs_[j][1], W, WS, N, K)
                rel = ((g.float() - ref).norm() / ref.norm().clamp_min(1e-9)).item()
                if rel > 0.02 or not torch.isfinite(g.float()).all():
                    print(f"WRONG {args.combo} {name} M={M} golden rel={rel:.4f}", file=sys.stderr)
                    bad += 1
                gold.append(g)
            miss = 0
            for _ in range(max(1, args.iters // (2 * BATCH))):
                for b in range(BATCH):        # x0, x1, x0, x1 ... no sync in between
                    call(0, outs[0][b])
                    call(1, outs[1][b])
                torch.cuda.synchronize()
                for j in range(2):
                    miss += int((outs[j] != gold[j]).flatten(1).any(1).sum().item())
            launches = 2 * BATCH * max(1, args.iters // (2 * BATCH))
            print(f"{args.combo:6s} {name:8s} M={M:3d}  {miss}/{launches} launches differ", flush=True)
            bad += miss
        W = WS = WR = None
        torch.cuda.empty_cache()
    sys.exit(2 if bad else 0)


def parent(args):
    rc = 0
    for combo, env in COMBOS:
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", "--combo", combo,
               "--ms", args.ms, "--iters", str(args.iters)]
        print(f"== {combo}: {env}", file=sys.stderr, flush=True)
        p = subprocess.run(cmd, env=dict(os.environ, **env))
        rc = rc or p.returncode
    print("PASS: every launch bit-identical" if rc == 0 else f"FAIL rc={rc}")
    sys.exit(rc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="5,8,16,24,40,64,96,128,160,192")
    ap.add_argument("--iters", type=int, default=2048, help="launches per (shape, M, split)")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--combo", default="policy")
    a = ap.parse_args()
    worker(a) if a.worker else parent(a)
