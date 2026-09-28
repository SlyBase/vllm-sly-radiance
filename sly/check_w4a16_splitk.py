#!/usr/bin/env python3
"""Numerics check for the W4A16 split-K path of sly/patch_w4a16_tiles.py (hunks 3 + 4).

Against an fp32 reference (unpacked ExLlama nibbles, (q - 8) * scale or (q - zp) * scale, fp32 matmul)
and against the stock _triton_w4a16_skinny_fmt_kernel with the same tile:

  * split_k 1 must be bit-identical to the stock kernel (it IS the stock kernel),
  * split_k 2..8, partial buffer and atomic, must sit at the bf16 output floor of the reference,
  * the fast-dequant schemes (deq 1 scale-after-dot, deq 2 magic-number + folded zero point, unpack 1
    interleave-free 8-dot) alone and split, same floor,
  * uneven splits (K tiles not divisible by split_k), M not a multiple of BLOCK_M, asymmetric zp,
  * the dispatch: a _GFX12X_SPLITK entry is taken by triton_w4a16_skinny_fmt_gemm and
    RADIANCE_W4A16_SPLITK=0 falls back to the tile table.

CPU (Triton interpreter, no GPU; reduced N, real K), in a plain `docker run` without device flags:
  BENCH_DEVICE=cpu TRITON_INTERPRET=1 python3 check_w4a16_splitk.py
GPU (full production shapes):
  python3 check_w4a16_splitk.py --full
The patch has to be applied in the container first (python3 sly/patch_w4a16_tiles.py).
"""
import argparse
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--full", action="store_true", help="real N as well (GPU; far too slow interpreted)")
args = ap.parse_args()

from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as H  # noqa: E402

dev = torch.device(os.environ.get("BENCH_DEVICE", "cuda"))
GS = 128
# The Triton interpreter mis-casts int -> bf16 (outputs ~1e11 on the stock kernel), fp16 is exact;
# the kernel body is dtype-generic, so the CPU run checks fp16 and the GPU run bf16 (production).
DT = torch.float16 if dev.type == "cpu" else torch.bfloat16
torch.manual_seed(0)

# (name, N, K): INT4 target + DFlash2 drafter shapes; N cut to 96/160 on the CPU (K is what split-K cuts)
SHAPES = [("down", 5120, 17408), ("out_o", 5120, 6144), ("o", 5120, 4096), ("fc", 5120, 25600),
          ("qkvz", 16384, 5120), ("gate_up", 34816, 5120)]
if not args.full:
    SHAPES = [(n, 96 if i % 2 else 160, k) for i, (n, _, k) in enumerate(SHAPES) if n in ("down", "out_o", "gate_up")]
    SHAPES += [("k_odd", 64, 1152)]
MS = [1, 5, 8, 13, 16, 32] if args.full else [1, 8, 13]
# (deq, unpack, split_k, atomic): the stock dequant through every split mode, then each fast-dequant
# scheme alone (split_k 1 = DIRECT store) and split (buffer and atomic)
VARIANTS = [(0, 0, sk, at) for sk in (1, 2, 3, 4, 8) for at in ((0,) if sk == 1 else (0, 1))]
VARIANTS += [(d, u, sk, at) for d, u in ((1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
             for sk, at in ((1, 0), (3, 0), (8, 0), (3, 1))]

SHIFTS = torch.tensor([(j // 2) * 4 + (j % 2) * 16 for j in range(8)], dtype=torch.int32)


def unpack_nibbles(b_q):  # [N, K//8] int32 ExLlama shuffle -> [N, K] nibbles
    w = (b_q.cpu()[:, :, None] >> SHIFTS[None, None, :]) & 0xF
    return w.reshape(b_q.shape[0], -1)


def reference(a, b_q, scales, zp):
    q = unpack_nibbles(b_q).float()
    N, K = q.shape
    s = scales.cpu().float().repeat_interleave(GS, dim=1)
    if zp is None:
        w = (q - 8) * s
    else:
        z = (zp.cpu()[:, None, :] >> (4 * torch.arange(8, dtype=torch.int32))[None, :, None]) & 0xF
        w = (q - z.reshape(N, -1).float().repeat_interleave(GS, dim=1)) * s
    return a.cpu().float() @ w.t()


fails = 0
checked = 0
for name, N, K in SHAPES:
    b_q = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev)
    scales = (torch.rand(N, K // GS, dtype=torch.float32) * 0.02 + 0.001).to(DT).to(dev)
    zp = torch.randint(-2**31, 2**31 - 1, (N // 8, K // GS), dtype=torch.int32, device=dev)
    for M in MS:
        a = torch.randn(M, K, dtype=torch.float32).to(DT).to(dev)
        for use_zp in (False, True) if name in ("down", "k_odd") else (False,):
            z = zp if use_zp else None
            ref = reference(a, b_q, scales, z)
            floor = ref.abs().max().item() * 2 ** -7  # output rounding (bf16 2^-9 rel.), fp32 reassociation
            bm = 16 if M <= 16 else 32
            base = None
            for deq, unpack, sk, atomic in VARIANTS:
                cfg = (bm, 32, 128, 2, None, sk, atomic, deq, unpack)
                out = H.triton_w4a16_splitk_gemm(a, b_q, scales, GS, cfg, zp=z).cpu()
                err = (out.float() - ref).abs().max().item()
                ok = err <= floor and out.shape == (M, N)
                if (deq, unpack, sk) == (0, 0, 1):
                    base = out
                tag = (f"{name:7s} N={N:5d} K={K:5d} M={M:2d} zp={int(use_zp)} deq={deq} unpack={unpack} "
                       f"sk={sk} at={atomic}")
                checked += 1
                if not ok:
                    fails += 1
                    print(f"FAIL {tag}: max err {err:.3e} > floor {floor:.3e}", flush=True)
                elif (deq, unpack, sk) != (0, 0, 1) and M in (8, 13):
                    d = (out.float() - base.float()).abs().max().item()
                    print(f"ok   {tag}: err {err:.2e} (floor {floor:.2e}), vs sk1 {d:.2e}", flush=True)

# split_k 1 through the public path == the stock kernel launched directly (bit-identical)
name, N, K = SHAPES[0]
b_q = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev)
scales = (torch.rand(N, K // GS) * 0.02 + 0.001).to(DT).to(dev)
a = torch.randn(8, K).to(DT).to(dev)
c0 = torch.empty((8, N), dtype=a.dtype, device=dev)
H._triton_w4a16_skinny_fmt_kernel[(1, (N + 31) // 32)](
    a, b_q, scales, scales, c0, 8, N, K, K // 8, K // GS, group_size=GS, ZP_BIAS=8, HAS_ZP=False,
    BLOCK_M=16, BLOCK_N=32, BLOCK_K=128, num_warps=2)
c1 = H.triton_w4a16_splitk_gemm(a, b_q, scales, GS, (16, 32, 128, 2, None, 1, 0))
ident = torch.equal(c0.cpu(), c1.cpu())
print(f"split_k 1 bit-identical to the stock kernel: {ident}")
fails += not ident

# dispatch: a table entry is taken, RADIANCE_W4A16_SPLITK=0 falls back
H._on_gfx12x = lambda: True  # the CPU has no platform; the branch itself is what is tested
key = (GS, K, N, 8)
prev = H._GFX12X_SPLITK.get(key)
H._GFX12X_SPLITK[key] = (16, 32, 128, 2, None, 4, 0)
calls = []
orig = H.triton_w4a16_splitk_gemm
H.triton_w4a16_splitk_gemm = lambda *x, **kw: calls.append(1) or orig(*x, **kw)
os.environ["RADIANCE_W4A16_SPLITK"] = "1"
d1 = H.triton_w4a16_skinny_fmt_gemm(a, b_q, scales, GS)
os.environ["RADIANCE_W4A16_SPLITK"] = "0"
d0 = H.triton_w4a16_skinny_fmt_gemm(a, b_q, scales, GS)
H.triton_w4a16_splitk_gemm = orig
if prev is None:
    del H._GFX12X_SPLITK[key]
else:
    H._GFX12X_SPLITK[key] = prev
taken = calls == [1]
ref = reference(a, b_q, scales, None)
fl = ref.abs().max().item() * 2 ** -7
dispatch_ok = taken and (d1.cpu().float() - ref).abs().max().item() <= fl \
    and (d0.cpu().float() - ref).abs().max().item() <= fl
print(f"dispatch: split-K entry taken once and knob=0 falls back: {dispatch_ok}")
fails += not dispatch_ok

print(f"{checked} split-K cases, {fails} failures")
sys.exit(1 if fails else 0)
