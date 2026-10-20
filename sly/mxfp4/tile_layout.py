"""Index math of the WMMA-fragment-tiled fp8 activation layout (radiance_mxfp4_fp8.hip).

Fragment (mt, ks) = rows mt*16..+15 x cols ks*16..+15 = 256 contiguous bytes in lane order:
lane l owns bytes 8l..8l+7 = A[mt*16 + l%16][ks*16 + (l//16)*8 + 0..7]. Pure torch, any dtype
of 1 byte; used by the CPU test and by bench_prefill_cells.py --check.
"""
import torch


def tile(a: torch.Tensor) -> torch.Tensor:
    """Row-major [M, K] (K % 16 == 0) -> flat tiled storage of ceil(M/16)*16*K bytes (pad = 0)."""
    M, K = a.shape
    assert K % 16 == 0
    mp = (M + 15) // 16 * 16
    p = torch.zeros((mp, K), dtype=a.dtype, device=a.device)
    p[:M] = a
    # [mt, lrow, ks, half, e] -> [mt, ks, half, lrow, e]; lane = lrow + 16*half
    return p.view(mp // 16, 16, K // 16, 2, 8).permute(0, 2, 3, 1, 4).contiguous().view(-1)


def untile(t: torch.Tensor, M: int, K: int) -> torch.Tensor:
    """Inverse of tile(): flat tiled storage -> row-major [M, K]."""
    mp = (M + 15) // 16 * 16
    r = t.reshape(-1)[: mp * K].view(mp // 16, K // 16, 2, 16, 8).permute(0, 3, 1, 2, 4)
    return r.reshape(mp, K)[:M].contiguous()


def kernel_offset(row: int, col: int, K: int) -> int:
    """Byte offset the producer kernels compute for element (row, col): the scalar form of the
    `(mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)` uint2 store index."""
    mt, lrow, kst, g, e = row >> 4, row & 15, K >> 4, col >> 3, col & 7
    return ((mt * kst + (g >> 1)) * 32 + lrow + 16 * (g & 1)) * 8 + e

