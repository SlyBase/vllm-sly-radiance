#!/usr/bin/env python3
"""Numerics check for the W4A16 kernels of sly/patch_w4a16_tiles.py.

Against an fp32 reference (unpacked ExLlama nibbles, (q - 8) * scale or (q - zp) * scale, fp32 matmul):

  * split_k 1 through the public path is the stock kernel, bit-identical,
  * every dequant / unpack / split / mode combination at the output floor of the reference,
  * bit-identical pairs that pin the new code paths to the old ones: the tiled weight layout (LAYOUT 1)
    against the row layout, the serial reduce (mode 2) against the partial-buffer reduce (mode 0),
    KSTEP 2 against KSTEP 1, the split epilogue (EPI 1) against the plain output,
  * the fused silu epilogue (EPI 2, interleaved gate/up rows) against silu(gate) * up of the reference,
  * the fused-op entry (_radiance_fused_gemm) with a table config and on its torch fallback,
  * tile/untile round trip, the int4 re-quantization of the post-load transforms, the table file
    (RADIANCE_W4A16_SPLITK_TABLE), the dispatch and the 1 MiB partial cap of the built-in table.

CPU (Triton interpreter, no GPU; reduced N, real K), in a plain `docker run` without device flags:
  BENCH_DEVICE=cpu TRITON_INTERPRET=1 python3 check_w4a16_splitk.py
GPU (full production shapes):
  python3 check_w4a16_splitk.py --full
The patch has to be applied in the container first (python3 sly/patch_w4a16_tiles.py).
"""
import argparse
import json
import os
import sys
import tempfile

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
# (deq, unpack, split_k, mode, kstep): stock dequant through every split mode, each fast-dequant scheme
# alone (split_k 1 = direct epilogue) and split, KSTEP 2 on a split and a direct config
VARIANTS = [(0, 0, 1, 0, 1)] + [(0, 0, sk, md, 1) for sk in (2, 3, 8) for md in (0, 1, 2)]
VARIANTS += [(d, u, sk, md, 1) for d, u in ((1, 0), (2, 0), (0, 1), (1, 1), (2, 1))
             for sk, md in ((1, 0), (3, 0), (3, 2), (8, 1))]
VARIANTS += [(2, 0, 3, 2, 2), (1, 0, 1, 0, 2)]

SHIFTS = torch.tensor([(j // 2) * 4 + (j % 2) * 16 for j in range(8)], dtype=torch.int32)


def unpack_nibbles(b_q):  # [N, K//8] int32 ExLlama shuffle -> [N, K] nibbles
    w = (b_q.cpu()[:, :, None] >> SHIFTS[None, None, :]) & 0xF
    return w.reshape(b_q.shape[0], -1)


def dequant(b_q, scales, zp):
    q = unpack_nibbles(b_q).float()
    N, K = q.shape
    s = scales.cpu().float().repeat_interleave(GS, dim=1)
    if zp is None:
        return (q - 8) * s
    z = (zp.cpu()[:, None, :] >> (4 * torch.arange(8, dtype=torch.int32))[None, :, None]) & 0xF
    return (q - z.reshape(N, -1).float().repeat_interleave(GS, dim=1)) * s


def reference(a, b_q, scales, zp):
    return a.cpu().float() @ dequant(b_q, scales, zp).t()


def tiled(b_q):
    return H.radiance_w4a16_tile(b_q).view(torch.int32)


fails = 0
checked = 0


def report(ok, tag, detail=""):
    global fails, checked
    checked += 1
    if not ok:
        fails += 1
        print(f"FAIL {tag} {detail}", flush=True)


# --- tile / untile round trip ---
w = torch.randint(-2**31, 2**31 - 1, (64, 1152 // 8), dtype=torch.int32)
t = H.radiance_w4a16_tile(w)
report(t.shape == (4, 9, 16, 64) and torch.equal(H.radiance_w4a16_untile(t).view(torch.int32), w),
       "tile/untile round trip")
print(f"tile/untile round trip: {not fails}")

for name, N, K in SHAPES:
    b_q = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev)
    b_t = tiled(b_q)
    scales = (torch.rand(N, K // GS, dtype=torch.float32) * 0.02 + 0.001).to(DT).to(dev)
    zp = torch.randint(-2**31, 2**31 - 1, (N // 8, K // GS), dtype=torch.int32, device=dev)
    for M in MS:
        a = torch.randn(M, K, dtype=torch.float32).to(DT).to(dev)
        for use_zp in (False, True) if name in ("down", "k_odd") else (False,):
            z = zp if use_zp else None
            ref = reference(a, b_q, scales, z)
            floor = ref.abs().max().item() * 2 ** -7  # output rounding (bf16 2^-9 rel.), fp32 reassociation
            bm = 16 if M <= 16 else 32
            outs = {}
            for deq, unpack, sk, md, ks in VARIANTS:
                cfg = (bm, 32, 128, 2, None, sk, md, deq, unpack, ks)
                tag = f"{name:7s} N={N:5d} K={K:5d} M={M:2d} zp={int(use_zp)} deq={deq} up={unpack} sk={sk} md={md} ks={ks}"
                out = H.triton_w4a16_splitk_gemm(a, b_q, scales, GS, cfg, zp=z).cpu()
                err = (out.float() - ref).abs().max().item()
                report(err <= floor and out.shape == (M, N), tag, f"max err {err:.3e} > floor {floor:.3e}")
                out_t = H.triton_w4a16_splitk_gemm(a, b_t, scales, GS, cfg, zp=z).cpu()
                # atomic (mode 1) has no fixed summation order on a GPU: tolerance there, identity elsewhere
                t_err = (out_t.float() - ref).abs().max().item()
                report(torch.equal(out, out_t) or (md == 1 and t_err <= floor), tag,
                       "tiled layout differs from row layout")
                outs[(deq, unpack, sk, md, ks)] = out
            # serial reduce == buffer reduce, KSTEP 2 == KSTEP 1 (same summation order)
            for (deq, unpack, sk, md, ks), out in outs.items():
                if md == 2 and (deq, unpack, sk, 0, ks) in outs:
                    report(torch.equal(out, outs[(deq, unpack, sk, 0, ks)]), f"{name} M={M} deq={deq} sk={sk}",
                           "serial reduce differs from the buffer reduce")
                if ks == 2 and (deq, unpack, sk, md, 1) in outs:
                    report(torch.equal(out, outs[(deq, unpack, sk, md, 1)]), f"{name} M={M} deq={deq} sk={sk}",
                           "KSTEP 2 differs from KSTEP 1")
            if M in (8, 13):
                print(f"ok   {name:7s} N={N:5d} K={K:5d} M={M:2d} zp={int(use_zp)}: {len(VARIANTS)} variants x 2 layouts"
                      f" ({fails} failures so far)", flush=True)
    # epilogues (symmetric weights): EPI 1 == plain slices, EPI 2 vs silu(gate) * up of the reference
    M = 8
    a = torch.randn(M, K, dtype=torch.float32).to(DT).to(dev)
    ref = reference(a, b_q, scales, None)
    ref_silu = torch.nn.functional.silu(ref[:, 0::2]) * ref[:, 1::2]
    fl2 = ref_silu.abs().max().item() * 2 ** -6
    n1 = N - 32
    for cfg in ((16, 32, 128, 2, None, 1, 0, 2, 0), (16, 32, 128, 2, None, 3, 2, 2, 0),
                (16, 32, 128, 2, None, 3, 0, 1, 0), (16, 32, 128, 2, None, 3, 1, 0, 0)):
        plain = H.triton_w4a16_splitk_gemm(a, b_t, scales, GS, cfg).cpu()
        c1, c2 = H.triton_w4a16_splitk_gemm(a, b_t, scales, GS, cfg, epi=1, n1=n1)
        both = torch.cat([c1.cpu(), c2.cpu()], 1)
        exact = torch.equal(both, plain)  # atomic (mode 1) has no fixed order on a GPU: tolerance there
        close = (both.float() - ref).abs().max().item() <= ref.abs().max().item() * 2 ** -7
        report(c1.is_contiguous() and c2.is_contiguous() and (exact or (cfg[6] == 1 and close)),
               f"{name} EPI 1 {cfg}", "split epilogue differs from the plain output")
        sil = H.triton_w4a16_splitk_gemm(a, b_t, scales, GS, cfg, epi=2).cpu()
        err = (sil.float() - ref_silu).abs().max().item()
        report(sil.shape == (M, N // 2) and err <= fl2, f"{name} EPI 2 {cfg}", f"err {err:.3e} > {fl2:.3e}")

fails_before_epi = fails  # (the epilogue checks run per shape above; this counts the fused entry only)
# fused-op entry: table config (EPI kernel) and the torch fallback (no table entry, e.g. prefill M)
name, N, K = SHAPES[0]
b_q = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev)
scales = (torch.rand(N, K // GS) * 0.02 + 0.001).to(DT).to(dev)
w_q8 = H.radiance_w4a16_tile(b_q)  # int8 [N/16, K/128, 16, 64], as stored after load
on_gfx12x = H._on_gfx12x
for M in (8, 70):
    a = torch.randn(M, K).to(DT).to(dev)
    ref = reference(a, b_q, scales, None)
    ref_silu = torch.nn.functional.silu(ref[:, 0::2]) * ref[:, 1::2]
    for on_gfx in (True, False):
        H._on_gfx12x = (lambda v: (lambda: v))(on_gfx)
        H._RADIANCE_SK_TABLE = {(GS, K, N, 8): (16, 32, 128, 2, None, 3, 2, 2, 0)}
        os.environ["RADIANCE_W4A16_SPLITK"] = "1"
        s_out = H._radiance_fused_gemm(a, w_q8, scales, GS, epi=2).cpu()
        c1, c2 = H._radiance_fused_gemm(a, w_q8, scales, GS, epi=1, n1=N - 32)
        err = (s_out.float() - ref_silu).abs().max().item()
        e1 = (torch.cat([c1.cpu(), c2.cpu()], 1).float() - ref).abs().max().item()
        report(err <= ref_silu.abs().max().item() * 2 ** -6 and e1 <= ref.abs().max().item() * 2 ** -7,
               f"fused entry M={M} gfx12x={on_gfx}", f"silu err {err:.3e}, split err {e1:.3e}")
H._on_gfx12x = on_gfx12x
H._RADIANCE_SK_TABLE = None
print(f"epilogues and fused entry: {fails == fails_before_epi}")

# int4 re-quantization (post-load: GDN ba rows, drafter kernel_projection): relative RMS error of the
# dequantized weight, same recipe as the int4 lm_head (~10 % expected for a symmetric g128 int4)
wf = torch.randn(96, 1024, dtype=torch.float32).to(torch.bfloat16)
q8, s8 = H._radiance_quant_rows(wf, GS)
deq = dequant(q8.view(torch.int32), s8, None)
rel = ((deq - wf.float()).square().mean() / wf.float().square().mean()).sqrt().item()
report(q8.shape == (96, 512) and s8.shape == (96, 8) and rel < 0.13, "int4 re-quantization", f"rel rms {rel:.3f}")
print(f"int4 re-quantization rel rms {rel:.3f}")

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
report(ident, "split_k 1 == stock")

# dispatch: a table entry is taken, RADIANCE_W4A16_SPLITK=0 falls back (to the tiled stock kernel)
H._on_gfx12x = lambda: True  # the CPU has no platform; the branch itself is what is tested
key = (GS, K, N, 8)
H._RADIANCE_SK_TABLE = {key: (16, 32, 128, 2, None, 4, 2)}
calls = []
orig = H.triton_w4a16_splitk_gemm
H.triton_w4a16_splitk_gemm = lambda *x, **kw: calls.append(1) or orig(*x, **kw)
os.environ["RADIANCE_W4A16_SPLITK"] = "1"
d1 = H.triton_w4a16_skinny_fmt_gemm(a, b_q, scales, GS)
os.environ["RADIANCE_W4A16_SPLITK"] = "0"
d0 = H.triton_w4a16_skinny_fmt_gemm(a, tiled(b_q), scales, GS)
H.triton_w4a16_splitk_gemm = orig
H._RADIANCE_SK_TABLE = None
H._on_gfx12x = on_gfx12x
ref = reference(a, b_q, scales, None)
fl = ref.abs().max().item() * 2 ** -7
dispatch_ok = calls == [1] and (d1.cpu().float() - ref).abs().max().item() <= fl \
    and (d0.cpu().float() - ref).abs().max().item() <= fl
print(f"dispatch: split-K entry taken once, knob=0 falls back (tiled stock kernel): {dispatch_ok}")
report(dispatch_ok, "dispatch")

# table file (RADIANCE_W4A16_SPLITK_TABLE) replaces the built-in table
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
    json.dump({"128,1152,64,8": [16, 32, 128, 2, None, 2, 2, 2, 0, 2]}, tf)
os.environ["RADIANCE_W4A16_SPLITK_TABLE"] = tf.name
os.environ["RADIANCE_W4A16_SPLITK"] = "1"
H._RADIANCE_SK_TABLE = None
e8 = H._gfx12x_splitk_override(GS, 1152, 64, 8)
e16 = H._gfx12x_splitk_override(GS, 1152, 64, 16)
del os.environ["RADIANCE_W4A16_SPLITK_TABLE"]
H._RADIANCE_SK_TABLE = None
tbl_ok = e8 == (16, 32, 128, 2, None, 2, 2, 2, 0, 2) and e16 is None
print(f"table file replaces the built-in table: {tbl_ok}")
report(tbl_ok, "table file")

# built-in table: every split-K partial within the KV cap
cap_ok = all(v[5] <= 1 or (b * n * 4 if v[6] == 1 else v[5] * b * n * 4) <= H._RADIANCE_SK_MAX_PARTIAL
             for (_g, _k, n, b), v in H._GFX12X_SPLITK.items())
print(f"built-in table partials <= {H._RADIANCE_SK_MAX_PARTIAL >> 20} MiB: {cap_ok}")
report(cap_ok, "partial cap")

print(f"{checked} checks, {fails} failures")
sys.exit(1 if fails else 0)
