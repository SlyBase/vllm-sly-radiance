"""CPU emulation of the gemv_mx4 kernel (radiance_inkling2_mx4.hip): decode chain, tile layout, lane addressing.

The kernel (W4A16 on the matrix unit) rests on facts that are each emulated here with plain loops, so an
indexing slip shows up before a GPU window:
  * e2m1 code -> bf16 by byte LUTs (v_perm_b32) + sign OR + byte interleave, EXACT for every code;
  * a dword of 8 nibbles is ONE wmma operand (8 bf16 slots); slot s sits at nibble position NIBPOS[s];
  * the tile layout produced by pack_mx4 is read at the offsets documented in the .hip header;
  * wmma gfx12 wave32 lane layout (as in test_gemv_q8_layout.py);
  * the per-32 scale 2^(e - 127) is applied to the exact K = 32 partial sum.
The emulation must equal dequantize-then-dot for every M 1..16, odd N, any T / W, K not a multiple of 128.
Nothing here imports the decode helpers of the module under test: they are re-written from the .hip header.
"""
import struct
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import radiance_inkling2_mxfp4dense as MD  # noqa: E402

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
NIBPOS = [0, 2, 4, 6, 1, 3, 5, 7]  # slot s of a wmma operand <- nibble NIBPOS[s] of the code dword
# byte LUTs of the kernel: bf16 high / low byte of the magnitude, entries 0..3 in the low dword
LUT_H = (0x40404040, 0x3F3F3F00)  # (entries 4..7, entries 0..3)
LUT_L = (0xC0804000, 0xC0800000)
IL_LO, IL_HI = 0x05010400, 0x07030602  # interleave selectors: [lo0 hi0 lo1 hi1] / [lo2 hi2 lo3 hi3]


def perm(s0, s1, sel):
    """v_perm_b32: result byte j = byte sel_j of the 8 bytes {s0 (4..7), s1 (0..3)}."""
    src = [(s1 >> (8 * i)) & 0xFF for i in range(4)] + [(s0 >> (8 * i)) & 0xFF for i in range(4)]
    return sum(src[(sel >> (8 * j)) & 0xFF] << (8 * j) for j in range(4))


def decode8(w):
    """One code dword -> 4 dwords of bf16 pairs (the wmma operand), following the kernel."""
    se = w & 0x07070707
    so = (w >> 4) & 0x07070707
    he = perm(LUT_H[0], LUT_H[1], se) | ((w << 4) & 0x80808080)
    ho = perm(LUT_H[0], LUT_H[1], so) | (w & 0x80808080)
    le = perm(LUT_L[0], LUT_L[1], se)
    lo = perm(LUT_L[0], LUT_L[1], so)
    return [perm(he, le, IL_LO), perm(he, le, IL_HI), perm(ho, lo, IL_LO), perm(ho, lo, IL_HI)]


def bf16_pairs_to_floats(dwords):
    out = []
    for d in dwords:
        for half in (d & 0xFFFF, d >> 16):
            out.append(struct.unpack("<f", struct.pack("<I", half << 16))[0])
    return out


def nib(w, p):
    return (w >> (4 * p)) & 0xF


def test_decode_chain_is_exact_for_every_nibble_in_every_position():
    for p in range(8):
        for code in range(16):
            w = code << (4 * p)
            vals = bf16_pairs_to_floats(decode8(w))
            for s in range(8):
                want = (E2M1[code & 7] * (-1 if code & 8 else 1)) if NIBPOS[s] == p else 0.0
                assert vals[s] == want and (vals[s] != 0 or want == 0), (p, code, s)


def test_decode_chain_random_dwords():
    g = torch.Generator().manual_seed(0)
    for w in torch.randint(0, 2**32, (3000,), generator=g, dtype=torch.int64).tolist():
        vals = bf16_pairs_to_floats(decode8(w))
        for s in range(8):
            c = nib(w, NIBPOS[s])
            assert vals[s] == E2M1[c & 7] * (-1 if c & 8 else 1)


def test_negative_zero_is_a_zero_product():
    assert bf16_pairs_to_floats(decode8(0x88888888)) == [0.0] * 8  # -0 is -0.0 == 0.0


# ---- wmma / kernel emulation (lane by lane) ----------------------------------------------------


def wmma(a_lanes, b_lanes, c_lanes):
    """a_lanes[lane][j] = A[lane % 16][8 * (lane // 16) + j]; b_lanes[lane][j] = B[8 * (lane // 16) + j][lane % 16];
    returns c_lanes[lane][e] = C[8 * (lane // 16) + e][lane % 16] + sum_k A[8 h + e][k] B[k][lane % 16]."""
    A, B, C = torch.zeros(16, 16), torch.zeros(16, 16), torch.zeros(16, 16)
    for lane in range(32):
        h, r = lane // 16, lane % 16
        for j in range(8):
            A[r, 8 * h + j] = a_lanes[lane][j]
            B[8 * h + j, r] = b_lanes[lane][j]
            C[8 * h + j, r] = c_lanes[lane][j]
    D = A @ B + C
    return [[float(D[8 * (lane // 16) + e, lane % 16]) for e in range(8)] for lane in range(32)]


def emulate(x, packed, scale, T, W):
    """x [M, K] bf16-valued fp32, (packed [N, K/2], scale [K/32, N]) -> y [M, N], following the kernel."""
    M, K = x.shape
    N = packed.shape[0]
    qt, st = MD.pack_mx4(packed, scale)
    qf, sf = qt.reshape(-1).tolist(), st.reshape(-1).tolist()
    ngroups, ntiles = K // 64, -(-N // 16)
    y = torch.zeros(M, N)
    for blk in range(-(-ntiles // T)):
        red = [[[0.0] * 8 for _ in range(32)] for _ in range(W * T)]
        for wv in range(W):
            g0, g1 = ngroups * wv // W, ngroups * (wv + 1) // W
            acc = [[[0.0] * 8 for _ in range(32)] for _ in range(T)]
            for t in range(T):
                tile = min(blk * T + t, ntiles - 1)
                for g in range(g0, g1):
                    for b in range(2):  # the two 32-K blocks of the group
                        a, bw, sc = [[], []], [[], []], []
                        for lane in range(32):
                            c, h = lane % 16, lane // 16
                            xrow = x[c] if c < M else torch.zeros(K)
                            xs = xrow[g * 64 + b * 32 + 16 * h:][:16]
                            off = (tile * ngroups + g) * 512 + lane * 16
                            dw = [int.from_bytes(bytes(qf[off + 4 * i:off + 4 * i + 4]), "little") for i in range(4)]
                            for j in range(2):
                                a[j].append(xs[8 * j:8 * j + 8].tolist())
                                bw[j].append(bf16_pairs_to_floats(decode8(dw[2 * b + j])))
                            so = (tile * ngroups + g) * 32 + c * 2
                            sc.append(2.0 ** (sf[so + b] - 127))
                        tmp = wmma(a[0], bw[0], [[0.0] * 8] * 32)
                        tmp = wmma(a[1], bw[1], tmp)
                        for lane in range(32):
                            for e in range(8):
                                acc[t][lane][e] += tmp[lane][e] * sc[lane]
                for lane in range(32):
                    red[wv * T + t][lane] = acc[t][lane]
        for t in range(T):
            for lane in range(32):
                for e in range(8):
                    v = sum(red[w * T + t][lane][e] for w in range(W))
                    m, n = 8 * (lane // 16) + e, (blk * T + t) * 16 + lane % 16
                    if m < M and n < N:
                        y[m, n] = v
    return y


@pytest.mark.parametrize("T,W", [(1, 2), (1, 8), (2, 4)])
@pytest.mark.parametrize("m", [1, 5, 8, 9, 16])
def test_emulated_kernel_equals_dequantize_then_dot(m, T, W):
    g = torch.Generator().manual_seed(m)
    n, k = 37, 320  # 3 tiles (last one 5 rows), 5 groups: not a multiple of 128 K, uneven wave shares
    x = torch.randn(m, k, generator=g).bfloat16().float()
    packed, scale, _ = MD.quantize_mxfp4(torch.randn(n, k, generator=g).bfloat16())
    ref = x @ MD.dequant_mxfp4(packed, scale).float().t()
    torch.testing.assert_close(emulate(x, packed, scale, T, W), ref, rtol=1e-4, atol=1e-4)


def test_pack_shapes_and_zero_padding_of_the_last_tile():
    packed, scale, _ = MD.quantize_mxfp4(torch.randn(37, 192).bfloat16())
    qt, st = MD.pack_mx4(packed, scale)
    assert qt.dtype == st.dtype == torch.uint8
    assert tuple(qt.shape) == (3, 3, 512) and tuple(st.shape) == (3, 3, 32)
    # rows 37..47 of the last tile: code 0 and scale byte 0 (a zero contribution whatever x holds)
    last = qt[2].reshape(3, 2, 16, 16)  # [g, h, c, 16 B]
    assert (last[:, :, 5:, :] == 0).all()
    assert (st[2].reshape(3, 16, 2)[:, 5:, :] == 0).all()


def test_pack_is_a_pure_permutation_of_the_codes():
    packed, scale, _ = MD.quantize_mxfp4(torch.randn(48, 256).bfloat16())
    qt, st = MD.pack_mx4(packed, scale)
    assert sorted(qt.reshape(-1).tolist()) != [] and qt.numel() == packed.numel()
    nibbles = lambda t: sorted(torch.cat([t & 15, t >> 4]).reshape(-1).tolist())  # noqa: E731
    assert nibbles(qt) == nibbles(packed)
    assert sorted(st.reshape(-1).tolist()) == sorted(scale.reshape(-1).tolist())
