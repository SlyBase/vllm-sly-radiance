"""CPU emulation of the gemv_q8 kernel (radiance_inkling2_gemv.hip): lane addressing + wmma operand layout.

The kernel is written against three facts, each emulated here lane by lane with plain loops, no vectorised
shortcut (so an indexing slip in the design shows up before a GPU window):
  * wmma 16x16x16 on gfx12 wave32: lane 16 h + r holds row r of A and column r of B for K 8 h .. 8 h + 7 of the
    step (element j = K 8 h + j); the accumulator lane 16 h + c holds C[row 8 h + e][column c], e = 0..7.
    (This is what the working gemv_bf16 kernel relies on.)
  * gemv_q8: A = x (lane 16 h + i = x row i), B = weights (lane 16 h + c = weight row c); per block of 32 K two steps
    take the 16 K 32 b + 16 h .. +15 of a lane as 8 + 8 elements, for x and for the weights alike.
  * tiles come from pack_q8 and are read at the offsets documented in the .hip header.
The emulation must equal dequantize-then-dot for every M 1..16, odd N, any T / W, K not a multiple of 128.
"""
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import radiance_inkling2_gemv_hip as disp  # noqa: E402

T_CFG = [(1, 2), (1, 8), (2, 4)]  # (T tiles per wave, W waves per block)


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


def emulate(x, wq, scale, T, W):
    """x [M, K] bf16-valued fp32, wq int8 [N, K], scale [N, K/32] -> y [M, N], following the kernel."""
    M, K = x.shape
    N = wq.shape[0]
    qt, st = disp.pack_q8(wq, scale)
    qf, sf = qt.reshape(-1), st.reshape(-1)
    ngroups, ntiles = K // 64, -(-N // 16)
    y = torch.zeros(M, N)
    for blk in range(-(-ntiles // T)):
        red = [[[0.0] * 8 for _ in range(32)] for _ in range(W * T)]  # LDS [wave][t][lane][e]
        for wv in range(W):
            g0, g1 = ngroups * wv // W, ngroups * (wv + 1) // W
            acc = [[[0.0] * 8 for _ in range(32)] for _ in range(T)]
            for t in range(T):
                tile = min(blk * T + t, ntiles - 1)
                for g in range(g0, g1):
                    for k in range(2):  # the two 32-K blocks of the group
                        a0, a1, b0, b1, sc = [], [], [], [], []
                        for lane in range(32):
                            c, h = lane % 16, lane // 16
                            xrow = x[c] if c < M else torch.zeros(K)
                            xs = xrow[g * 64 + k * 32 + 16 * h:][:16]  # 16 consecutive K of this lane
                            off = (tile * ngroups + g) * 1024 + k * 512 + lane * 16
                            ws = qf[off:off + 16].float()
                            so = (tile * ngroups + g) * 32 + c * 2
                            sc.append(float(sf[so + k]))
                            a0.append(xs[:8].tolist()); a1.append(xs[8:].tolist())
                            b0.append(ws[:8].tolist()); b1.append(ws[8:].tolist())
                        tmp = wmma(a0, b0, [[0.0] * 8] * 32)
                        tmp = wmma(a1, b1, tmp)
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


@pytest.mark.parametrize("T,W", T_CFG)
@pytest.mark.parametrize("m", [1, 5, 8, 9, 16])
def test_emulated_kernel_equals_dequantize_then_dot(m, T, W):
    g = torch.Generator().manual_seed(m)
    n, k = 37, 320  # 3 tiles (last one 5 rows), 5 groups: not a multiple of 128 K, uneven wave shares
    x = torch.randn(m, k, generator=g).bfloat16().float()
    w = torch.randn(n, k, generator=g).bfloat16()
    wq, scale = disp.to_q8(w)
    ref = x @ (wq.float() * scale.repeat_interleave(32, dim=1)).t()
    got = emulate(x, wq, scale, T, W)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_int8_to_bf16_magic_is_exact_for_every_code():
    """The kernel's int8 -> bf16: float(0x4B000000 | (q ^ 0x80 as unsigned)) - (2^23 + 128), high 16 bits."""
    import struct

    for q in range(-128, 128):
        u = (q & 0xFF) ^ 0x80
        f = struct.unpack("<f", struct.pack("<I", 0x4B000000 | u))[0] - 8388736.0
        bits = struct.unpack("<I", struct.pack("<f", f))[0]
        assert bits & 0xFFFF == 0 and f == q, q  # nothing lost by taking the high half
        assert torch.tensor([bits >> 16], dtype=torch.int32).to(torch.int16).view(torch.bfloat16).item() == q
