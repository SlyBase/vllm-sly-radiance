"""Inkling 2 NVFP4 expert GEMM for gfx12 (RDNA4), which has no FP4 MMA.

vLLM's sm120 path computes `tl.dot_scaled(a, a_scale, "e2m1", w, w_scale, "e2m1")` -- a Blackwell
instruction.  Here both operands are dequantized per K chunk (e2m1 x e4m3 block scale, 16 columns per
scale) and fed to a bf16 `tl.dot` with fp32 accumulation.  The dequantized product has at most 6
significant bits, so bf16 holds it exactly: the result equals the native MMA up to accumulation
order.  Weights stay packed in memory (the ~20 GB checkpoint fits a 32 GB card).

Layouts are vLLM's (models/inkling2/nvidia/ops/moe_sm120.py): packed uint8 with the even column in
the low nibble, e4m3 weight scales row-major [E, N, K/16], activation scales in the tile-padded
R8c4/R128c4 layout (`_scale_offsets`), `tiles` = [expert per CTA | row end per CTA | CTA count].
"""
import torch
from vllm.models.inkling2.nvidia.ops.block_scales import _scale_offsets
from vllm.triton_utils import tl, triton


@triton.jit
def _e2m1(code):
    """4-bit e2m1 code (sign in bit 3) -> fp32."""
    mag = code & 7
    exp = (mag >> 1).to(tl.float32)
    man = (mag & 1).to(tl.float32)
    val = tl.where(mag < 2, man * 0.5, (1.0 + man * 0.5) * tl.exp2(exp - 1.0))
    return tl.where((code >> 3) == 1, -val, val)


@triton.jit
def _dequant_rows(packed, scales, DT: tl.constexpr):
    """[R, K/2] uint8 + [R, K/16] fp32 scales -> [R, K] in DT (exact in bf16)."""
    r: tl.constexpr = packed.shape[0]
    h: tl.constexpr = packed.shape[1]
    g: tl.constexpr = scales.shape[1]
    p = packed.to(tl.int32)
    vals = tl.reshape(tl.join(_e2m1(p & 0xF), _e2m1(p >> 4)), (r, h * 2))
    s = tl.reshape(tl.broadcast_to(scales[:, :, None], (r, g, 16)), (r, g * 16))
    return (vals * s).to(DT)


@triton.jit
def _expert_gemm(
    Act, ActScale, RowScale, Weight, WeightScale, ExpertScale, Out, Tiles,
    MAX_CTAS: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TILE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DT: tl.constexpr,
):
    tile = tl.program_id(0)
    if tile >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    row_end = tl.load(Tiles + MAX_CTAS + tile)
    rows = tile * TILE + tl.program_id(1) * BM + tl.arange(0, BM)
    valid = rows < tl.minimum(row_end, (tile + 1) * TILE)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        packed = k * (BK // 2) + tl.arange(0, BK // 2)
        groups = k * (BK // 16) + tl.arange(0, BK // 16)
        a = tl.load(Act + rows[:, None].to(tl.int64) * (K // 2) + packed[None, :],
                    valid[:, None] & (packed[None, :] < K // 2), other=0)
        a_scale = tl.load(ActScale + _scale_offsets(rows[:, None], groups[None, :], K // 16, TILE),
                          valid[:, None] & (groups[None, :] < K // 16), other=0.0)
        w = tl.load(Weight + (expert * N + cols[:, None]) * (K // 2) + packed[None, :],
                    (cols[:, None] < N) & (packed[None, :] < K // 2), other=0)
        w_scale = tl.load(WeightScale + (expert * N + cols[:, None]) * (K // 16) + groups[None, :],
                          (cols[:, None] < N) & (groups[None, :] < K // 16), other=0.0)
        acc = tl.dot(_dequant_rows(a, a_scale.to(tl.float32), DT),
                     tl.trans(_dequant_rows(w, w_scale.to(tl.float32), DT)), acc)
    acc *= tl.load(ExpertScale + expert)
    acc *= tl.load(RowScale + rows, valid, other=0)[:, None]
    tl.store(Out + rows[:, None].to(tl.int64) * N + cols[None, :], acc,
             valid[:, None] & (cols[None, :] < N))


def nvfp4_expert_gemm(act, act_scale, row_scale, weight, weight_scale, expert_scale, tiles, plan,
                      block_m=None, block_n=None, block_k=None, dot_dtype=tl.bfloat16):
    """bf16 `[plan.rows, N]` = routed rows x NVFP4 expert weights `[E, N, K/2]` (rows past a tile's
    `row_end` are left unwritten).  `dot_dtype`: bf16 is exact for these products; the Triton CPU
    interpreter mis-computes bf16 dots, so the tests pass float32 (same values)."""
    n, k = weight.shape[1], weight.shape[2] * 2
    bm = block_m or min(32, triton.next_power_of_2(plan.tile))  # arange needs a power of 2;
    bn = block_n or min(64, triton.next_power_of_2(n))          # the masks cover the overhang
    bk = block_k or min(128, triton.next_power_of_2(k))
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    _expert_gemm[(plan.max_ctas, triton.cdiv(plan.tile, bm), triton.cdiv(n, bn))](
        act, act_scale, row_scale, weight, weight_scale, expert_scale, out, tiles,
        MAX_CTAS=plan.max_ctas, N=n, K=k, TILE=plan.tile, BM=bm, BN=bn, BK=bk, DT=dot_dtype)
    return out
