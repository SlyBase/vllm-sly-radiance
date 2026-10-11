"""KV4 attention: AITER's Triton unified attention with a 4-bit K/V tile loader and fp8 compute.

Forked from aiter/ops/triton/_triton_kernels/attention/unified_attention.py (kernel_unified_attention_2d,
kernel_unified_attention_3d; reduce_segments is reused as is) and the Triton launchers in
aiter/ops/triton/attention/unified_attention.py. What changed, and only that:

  1. The K/V tile load and the two products. A tile is TILE_SIZE cells of 272 bytes
     (radiance_kv4_format): per side the e4m3 scales (K: one per half, V: one per token) and
     128 packed nibbles, packed byte p holding dim p (low nibble) and dim p + 128 (high nibble).
     The loader reads K as (128, TILE) and V as (TILE, 128) bytes -- half the bytes of the fp8
     kernel. K (uniform codebook) goes to the int8 WMMA as the nibble itself; V (geo codebook)
     goes to the fp8 WMMA as a bit shuffle of the nibble (the geo values are chosen so that their
     e4m3 bit patterns are the nibble's bits moved into place). Neither scale touches the tile:

        S      = sum_h  s_h(t) * ( dot_i8(Q_h, N_h) - 7.5 * rowsum(Q_h) )     h in {lo, hi}
        acc_h += dot_fp8( fp8(P * vs(t) * 16), V_h ) / 16

     The per-(token, half) K scale and the uniform codebook's offset land on the (BLOCK_M, TILE)
     score tile; the per-token V scale multiplies P before its single fp8 conversion; the dots
     are the fp8 kernel's own two K=128 products per side, the score ones on the integer units.
     Measured on the way here (verify-8, 32k, one sequence, 8 warps, fp8 kernel ~ 475 us): a
     per-element dequant with the scale gathered per element 25x, broadcast through LDS 9x;
     eight 32-dim groups as K=32 dots 2.3x before any arithmetic; the uniform codebook as fp8
     by bit formula with three selects 7x, by hardware int -> fp8 conversion 6.5x, as four
     bit-plane dots 6x; a single select per tile element 2.1x, the nibble mask alone 1.5x, the
     bytes bitcast with no arithmetic 1.42x. geo bits on both sides 2.46x; K on int8 1.81x.
  2. q_descale is a [num_tokens, num_query_heads] fp32 tensor (dynamic per-(token, head) query
     quantization), applied per row; k/v descales are gone -- the scales live in the cache.

The kernel runs uniform K / geo V; lloyd is a reference format.

Dropped, because no full-attention layer of this model uses them and the backend refuses them:
sliding window, alibi, qq_bias, sinks, softcap, shuffled KV, fused fp8 output.

The launch side mirrors AITER's: `_UAParams` with q and KV dtype reported as fp8 so the gfx1201
config tables and `radiance_attn_decode`'s verify-batch tune (BLOCK_M 64, TILE 32, split counts)
apply unchanged; AITER's `use_2d_kernel` (wrapped by the tune) decides 2D vs 3D.
"""
import os

import torch
import triton
import triton.language as tl

import aiter.ops.triton.attention.unified_attention as UA
import aiter.ops.triton.utils.unified_attention_utils as UU
from aiter.ops.triton._triton_kernels.attention.unified_attention import (
    cdiv_fn,
    find_seq_idx,
    reduce_segments,
)
from aiter.ops.triton.utils.types import e4m3_dtype

import radiance_kv4_format as F

FP8_TL = tl.float8e4nv
FP8_MAX = 448.0


# ---------------------------------------------------------------------------------------------
# tile pieces
# ---------------------------------------------------------------------------------------------

@triton.jit
def _scale(cache_ptr, cell, tile_mask, OFF: tl.constexpr, FULL_TILE: tl.constexpr):
    """One side's e4m3 scale byte at OFF for every cell of the tile, as (TILE,) fp32."""
    if FULL_TILE:
        b = tl.load(cache_ptr + cell + OFF)
    else:
        b = tl.load(cache_ptr + cell + OFF, mask=tile_mask, other=0)
    return b.to(FP8_TL, bitcast=True).to(tl.float32)


@triton.jit
def _nib(packed, hi: tl.constexpr, ABLATE: tl.constexpr):
    """One nibble n = s t2 t1 t0 of every packed byte as the fp8 value (-1)^s * {2,3,4,6,8,12,16,24}[t]
    (the geo codebook): e4m3 bits 0x40 | s << 7 | t << 2, i.e. two shifts, two ands, two ors on
    the packed byte. Nothing per element that is not a plain bit operation: on the R9700 a select
    per tile element costs 60 % of the fp8 attention kernel and a dtype conversion 5x of it."""
    if ABLATE & 1:   # timing ablation: no arithmetic, the bytes bitcast as fp8
        if hi:
            return (packed >> 4).to(FP8_TL, bitcast=True)
        return packed.to(FP8_TL, bitcast=True)
    if ABLATE & 16:  # timing ablation: the mask only
        if hi:
            return ((packed >> 4) & 0xF).to(FP8_TL, bitcast=True)
        return (packed & 0xF).to(FP8_TL, bitcast=True)
    if hi:
        bits = (packed & 0x80) | ((packed >> 2) & 0x1C) | 0x40
    else:
        bits = ((packed << 4) & 0x80) | ((packed << 2) & 0x1C) | 0x40
    return bits.to(FP8_TL, bitcast=True)


@triton.jit
def _p8(P, s, ABLATE: tl.constexpr):
    """P scaled per token by the V scale, lifted by 16 so P * s stays out of e4m3's subnormals
    (threshold 2^-6 -> 2^-10); returns (P8, 1 / 16). The clamp sits on the TILE scales, not on P
    (a select per P element costs 180 us per call): a V scale above 28 saturates, i.e. rotated V
    magnitudes of a few hundred. One conversion per tile, shared by both halves of V."""
    if ABLATE & 4:   # timing ablation: plain P conversion, no scaling
        return P.to(FP8_TL), 1.0
    return (P * tl.minimum(s * 16.0, 448.0)[None, :]).to(FP8_TL), 1.0 / 16.0


@triton.jit
def _tile_step(
    cache_ptr, cell, tile_mask, pk, seq_offset, context_len, query_pos, q_mask,
    Q_lo, Q_hi, qs_lo, qs_hi, row_scale, M, L, acc_lo, acc_hi,
    K_SCALE_OFF: tl.constexpr, K_PACK_OFF: tl.constexpr,
    V_SCALE_OFF: tl.constexpr, V_PACK_OFF: tl.constexpr,
    FULL_TILE: tl.constexpr, ABLATE: tl.constexpr, ZP: tl.constexpr = 7.5,
):
    # K: (128, TILE) packed bytes, two nibble streams, one scale per stream and token
    if FULL_TILE:
        k_pack = tl.load(cache_ptr + cell[None, :] + K_PACK_OFF + pk[:, None])
    else:
        k_pack = tl.load(cache_ptr + cell[None, :] + K_PACK_OFF + pk[:, None],
                         mask=tile_mask[None, :], other=0)
    sk_lo = _scale(cache_ptr, cell, tile_mask, K_SCALE_OFF, FULL_TILE)
    sk_hi = _scale(cache_ptr, cell, tile_mask, K_SCALE_OFF + 1, FULL_TILE)
    # K on the int8 WMMA: the nibble (0..15) is the int8 operand, the uniform codebook's -7.5
    # offset is one correction per row (sum of the int8 query row), the half scale lands on the
    # score tile. Nothing per element but the mask.
    S = (tl.dot(Q_lo, (k_pack & 0xF).to(tl.int8)).to(tl.float32) - ZP * qs_lo[:, None]) * sk_lo[None, :]
    S += (tl.dot(Q_hi, ((k_pack >> 4) & 0xF).to(tl.int8)).to(tl.float32) - ZP * qs_hi[:, None]) * sk_hi[None, :]
    S = S * row_scale[:, None]
    seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
    S = tl.where(q_mask & seq_mask, S, float("-inf"))

    m_j = tl.maximum(M, tl.max(S, axis=1))
    m_j = tl.where(m_j > float("-inf"), m_j, 0.0)
    P = tl.math.exp2(S - m_j[:, None])
    l_j = tl.sum(P, axis=1)
    alpha = tl.math.exp2(M - m_j)
    L = L * alpha + l_j
    M = m_j
    acc_lo = acc_lo * alpha[:, None]
    acc_hi = acc_hi * alpha[:, None]

    # V: (TILE, 128) packed bytes
    if FULL_TILE:
        v_pack = tl.load(cache_ptr + cell[:, None] + V_PACK_OFF + pk[None, :])
    else:
        v_pack = tl.load(cache_ptr + cell[:, None] + V_PACK_OFF + pk[None, :],
                         mask=tile_mask[:, None], other=0)
    sv = _scale(cache_ptr, cell, tile_mask, V_SCALE_OFF, FULL_TILE)
    P8, f = _p8(P, sv, ABLATE)
    acc_lo = acc_lo + tl.dot(P8, _nib(v_pack, False, ABLATE)) * f
    acc_hi = acc_hi + tl.dot(P8, _nib(v_pack, True, ABLATE)) * f
    return M, L, acc_lo, acc_hi


@triton.jit
def _prologue(query_ptr, q_descale_ptr, query_start_len_ptr, q_block_global_idx, num_seqs,
              kv_head_idx, qk_scale, pk, offs_m,
              num_query_heads: tl.constexpr, num_queries_per_kv: tl.constexpr,
              query_stride_0, query_stride_1, BLOCK_Q: tl.constexpr, ALL_DECODE: tl.constexpr):
    """Shared by the 2D and 3D kernels: which sequence and query rows this program owns, the two
    int8 query streams with their row sums, and the per-row scale."""
    if ALL_DECODE:
        seq_idx = q_block_global_idx
        q_block_local_idx = 0
        cur_batch_query_len = 1
        cur_batch_in_all_start_index = q_block_global_idx
    else:
        seq_idx = find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)
        q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
        q_block_local_idx = q_block_global_idx - q_block_start_idx
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv
    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_mask_0 = query_pos < cur_batch_query_len
    query_mask_1 = query_offset_1 < num_query_heads
    q_mask = query_mask_0[:, None] & query_mask_1[:, None]
    q_row = query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1
    Q_lo = tl.load(query_ptr + q_row + pk[None, :], mask=q_mask, other=0)
    Q_hi = tl.load(query_ptr + q_row + 128 + pk[None, :], mask=q_mask, other=0)
    qs_lo = tl.sum(Q_lo.to(tl.float32), axis=1)      # row sums for the K zero-point correction
    qs_hi = tl.sum(Q_hi.to(tl.float32), axis=1)
    q_descale = tl.load(q_descale_ptr + query_offset_0 * num_query_heads + query_offset_1,
                        mask=query_mask_0 & query_mask_1, other=0.0)
    row_scale = qk_scale * q_descale
    return (seq_idx, q_block_local_idx, cur_batch_query_len, query_pos, query_offset_0,
            query_offset_1, query_mask_0, query_mask_1, q_mask, Q_lo, Q_hi, qs_lo, qs_hi, row_scale)


# ---------------------------------------------------------------------------------------------
# 2D kernel: one program per (kv head, q block), the whole KV range
# ---------------------------------------------------------------------------------------------

@triton.jit
def kv4_attention_2d(
    output_ptr,          # [num_tokens, num_query_heads, HEAD]  bf16
    query_ptr,           # [num_tokens, num_query_heads, HEAD]  int8
    q_descale_ptr,       # [num_tokens, num_query_heads]        fp32
    cache_ptr,           # [num_blocks, num_kv_heads, BLOCK_SIZE, CONTENT] uint8
    block_tables_ptr,    # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,        # [num_seqs]
    query_start_len_ptr, # [num_seqs + 1]
    scale,               # float32, already divided by HEAD for the rotation
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    stride_c0: tl.int64,
    stride_c1: tl.int64,
    stride_c2: tl.int64,
    num_seqs: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    K_SCALE_OFF: tl.constexpr, K_PACK_OFF: tl.constexpr,
    V_SCALE_OFF: tl.constexpr, V_PACK_OFF: tl.constexpr,
    ALL_DECODE: tl.constexpr = False,
    ABLATE: tl.constexpr = 0,
    num_warps: tl.constexpr = 4,
    waves_per_eu: tl.constexpr = 2,
    num_stages: tl.constexpr = 1,
):
    kv_head_idx = tl.program_id(0)
    q_block_global_idx = tl.program_id(1)
    RCP_LN2 = 1.4426950408889634
    offs_m = tl.arange(0, BLOCK_M)
    pk = tl.arange(0, 128)
    offs_t = tl.arange(0, TILE_SIZE)

    (seq_idx, q_block_local_idx, cur_batch_query_len, query_pos, query_offset_0, query_offset_1,
     query_mask_0, query_mask_1, q_mask, Q_lo, Q_hi, qs_lo, qs_hi, row_scale) = _prologue(
        query_ptr, q_descale_ptr, query_start_len_ptr, q_block_global_idx, num_seqs, kv_head_idx,
        scale * RCP_LN2, pk, offs_m, num_query_heads, num_queries_per_kv, query_stride_0,
        query_stride_1, BLOCK_Q, ALL_DECODE)
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc_lo = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_M, 128], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    for j in range(0, num_tiles):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)
        cell = (physical_block_idx * stride_c0 + kv_head_idx * stride_c1
                + (seq_offset % BLOCK_SIZE) * stride_c2)
        M, L, acc_lo, acc_hi = _tile_step(
            cache_ptr, cell, tile_mask, pk, seq_offset, context_len, query_pos, q_mask,
            Q_lo, Q_hi, qs_lo, qs_hi, row_scale, M, L, acc_lo, acc_hi,
            K_SCALE_OFF, K_PACK_OFF, V_SCALE_OFF, V_PACK_OFF, TILE_SIZE == BLOCK_SIZE, ABLATE)

    one_over_L = 1.0 / L[:, None]
    out_row = query_offset_0[:, None] * output_stride_0 + query_offset_1[:, None] * output_stride_1
    tl.store(output_ptr + out_row + pk[None, :], acc_lo * one_over_L, mask=q_mask)
    tl.store(output_ptr + out_row + 128 + pk[None, :], acc_hi * one_over_L, mask=q_mask)


# ---------------------------------------------------------------------------------------------
# 3D kernel: one program per (q block, kv head, KV segment), reduced by reduce_segments
# ---------------------------------------------------------------------------------------------

@triton.jit
def kv4_attention_3d(
    segm_output_ptr,     # [num_tokens, num_query_heads, NUM_SEGMENTS, HEAD] fp32
    segm_max_ptr,        # [num_tokens, num_query_heads, NUM_SEGMENTS]
    segm_expsum_ptr,     # [num_tokens, num_query_heads, NUM_SEGMENTS]
    query_ptr,
    q_descale_ptr,
    cache_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    stride_c0: tl.int64,
    stride_c1: tl.int64,
    stride_c2: tl.int64,
    num_seqs: tl.int32,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    K_SCALE_OFF: tl.constexpr, K_PACK_OFF: tl.constexpr,
    V_SCALE_OFF: tl.constexpr, V_PACK_OFF: tl.constexpr,
    ALL_DECODE: tl.constexpr = False,
    ABLATE: tl.constexpr = 0,
    num_warps: tl.constexpr = 4,
    waves_per_eu: tl.constexpr = 2,
    num_stages: tl.constexpr = 1,
):
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)
    RCP_LN2 = 1.4426950408889634
    offs_m = tl.arange(0, BLOCK_M)
    pk = tl.arange(0, 128)
    offs_t = tl.arange(0, TILE_SIZE)

    (seq_idx, q_block_local_idx, cur_batch_query_len, query_pos, query_offset_0, query_offset_1,
     query_mask_0, query_mask_1, q_mask, Q_lo, Q_hi, qs_lo, qs_hi, row_scale) = _prologue(
        query_ptr, q_descale_ptr, query_start_len_ptr, q_block_global_idx, num_seqs, kv_head_idx,
        scale * RCP_LN2, pk, offs_m, num_query_heads, num_queries_per_kv, query_stride_0,
        query_stride_1, BLOCK_Q, ALL_DECODE)
    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return
    seq_len = tl.load(seq_lens_ptr + seq_idx)
    tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    block_table_offset = seq_idx * block_table_stride
    M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc_lo = tl.zeros([BLOCK_M, 128], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_M, 128], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len
    max_seq_prefix_len = context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = cdiv_fn(max_seq_prefix_len, TILE_SIZE)

    for j in range(segm_idx * tiles_per_segment,
                   min((segm_idx + 1) * tiles_per_segment, num_tiles)):
        seq_offset = j * TILE_SIZE + offs_t
        if TILE_SIZE == BLOCK_SIZE:
            tile_mask = tl.full((1,), 1, dtype=tl.int1)
        else:
            tile_mask = seq_offset < max_seq_prefix_len
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)
        cell = (physical_block_idx * stride_c0 + kv_head_idx * stride_c1
                + (seq_offset % BLOCK_SIZE) * stride_c2)
        M, L, acc_lo, acc_hi = _tile_step(
            cache_ptr, cell, tile_mask, pk, seq_offset, context_len, query_pos, q_mask,
            Q_lo, Q_hi, qs_lo, qs_hi, row_scale, M, L, acc_lo, acc_hi,
            K_SCALE_OFF, K_PACK_OFF, V_SCALE_OFF, V_PACK_OFF, TILE_SIZE == BLOCK_SIZE, ABLATE)

    if NUM_SEGMENTS_PER_SEQ == 1:
        one_over_L = 1.0 / L[:, None]
        acc_lo = acc_lo * one_over_L
        acc_hi = acc_hi * one_over_L

    HEAD: tl.constexpr = 256
    segm_row = (query_offset_0[:, None].to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD)
                + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD)
                + segm_idx * HEAD)
    tl.store(segm_output_ptr + segm_row + pk[None, :], acc_lo, mask=q_mask)
    tl.store(segm_output_ptr + segm_row + 128 + pk[None, :], acc_hi, mask=q_mask)
    if NUM_SEGMENTS_PER_SEQ > 1:
        segm_offset = (query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
                       + query_offset_1 * NUM_SEGMENTS_PER_SEQ + segm_idx)
        tl.store(segm_max_ptr + segm_offset, M, mask=query_mask_0 & query_mask_1)
        tl.store(segm_expsum_ptr + segm_offset, L, mask=query_mask_0 & query_mask_1)


# ---------------------------------------------------------------------------------------------
# host side
# ---------------------------------------------------------------------------------------------

LAST_LAUNCH: dict = {}   # diagnostics: regs / spills / cell of the most recent launches
KERNEL_CODEBOOK = (F.CODEBOOK_K, F.CODEBOOK_V)


def _note(name, kernel):
    try:
        LAST_LAUNCH[name] = dict(n_regs=kernel.n_regs, n_spills=kernel.n_spills)
    except Exception:  # noqa: BLE001
        pass


def quantize_query(query: torch.Tensor):
    """query (num_tokens, num_query_heads, HEAD) any float -> (q int8 rotated, descale fp32
    [num_tokens, num_query_heads]). Symmetric per-(token, head) scale, the score side's int8 WMMA
    operand."""
    n, h, _ = query.shape
    q_rot = F.rotate(query.reshape(-1, F.HEAD).to(torch.float32)).reshape(n, h, F.HEAD)
    amax = q_rot.abs().amax(-1).clamp_min(1e-12)
    descale = amax / 127.0
    q8 = torch.round(q_rot / descale[..., None]).clamp(-127, 127).to(torch.int8)
    return q8, descale


def unrotate_output(out_rot: torch.Tensor) -> torch.Tensor:
    """(num_tokens, num_query_heads, HEAD) rotated attention output -> original basis."""
    n, h, _ = out_rot.shape
    return F.unrotate(out_rot.reshape(-1, F.HEAD).to(torch.float32)).reshape(n, h, F.HEAD)


def _lds_fit(config: dict, tile: int) -> int:
    """AITER's R9700 LDS rule for fp8 tiles (K and V fp8 tiles of HEAD bytes per token)."""
    config.setdefault("num_stages", 1)
    while config["num_stages"] > 1 and tile * F.HEAD * config["num_stages"] + 256 > 65536:
        config["num_stages"] -= 1
    while tile > 16 and tile * F.HEAD * config["num_stages"] + 256 > 65536:
        tile //= 2
    return tile


def _kernel_config(config: dict) -> dict:
    cfg = dict(config)
    for k in ("TILE_SIZE_MIN", "TILE_SIZE_MAX"):
        cfg.pop(k, None)
    return cfg


_LAYOUT = dict(
    K_SCALE_OFF=F.K_SCALE_OFF, K_PACK_OFF=F.K_PACK_OFF,
    V_SCALE_OFF=F.V_SCALE_OFF, V_PACK_OFF=F.V_PACK_OFF,
)


def kv4_attention(
    q8: torch.Tensor,            # (num_tokens, num_query_heads, HEAD) int8, rotated
    q_descale: torch.Tensor,     # (num_tokens, num_query_heads) fp32
    kv_cache: torch.Tensor,      # (num_blocks, num_kv_heads, block_size, CONTENT) uint8
    out: torch.Tensor,           # (num_tokens, num_query_heads, HEAD) bf16, rotated on return
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    seqused_k: torch.Tensor,
    max_seqlen_k: int,
    softmax_scale: float,        # the model's scale; the rotation's 1/HEAD is applied here
    block_table: torch.Tensor,
    codebook=(F.CODEBOOK_K, F.CODEBOOK_V),
) -> torch.Tensor:
    assert codebook == KERNEL_CODEBOOK, "the kernel runs the (uniform K, geo V) format only"
    assert q8.dtype == torch.int8 and kv_cache.dtype == torch.uint8
    num_tokens, num_query_heads, head_size = q8.shape
    assert head_size == F.HEAD
    num_blocks, num_kv_heads, block_size, content = kv_cache.shape
    assert content == F.CONTENT
    num_seqs = len(seqused_k)
    num_queries_per_kv = num_query_heads // num_kv_heads
    scale = softmax_scale / F.HEAD

    # AITER's launch bookkeeping, with q and KV reported as fp8 so the config tables (and the
    # fork's verify-batch tune) treat this like the fp8 kernel it is on the compute side.
    BLOCK_M = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    cu_count = UA.get_num_sms()
    all_decode = int(max_seqlen_q) == 1
    total_num_q_blocks = num_seqs if all_decode else num_tokens // BLOCK_Q + num_seqs
    params = UA._UAParams(
        q=q8, k=kv_cache, v=kv_cache, out=out,
        cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k, block_table=block_table,
        softmax_scale=scale, softcap=0.0, causal=True, sliding_window=0,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k,
        num_tokens=num_tokens, num_query_heads=num_query_heads, num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv, head_size=F.HEAD, head_size_v=F.HEAD,
        num_seqs=num_seqs, total_num_q_blocks=total_num_q_blocks,
        num_2d_prgms=total_num_q_blocks * num_kv_heads,
        num_blocks=num_blocks, block_size=block_size, k_width=16, scale_k_width=4,
        block_scales_size=16, q_dtype=e4m3_dtype, kv_cache_dtype=e4m3_dtype,
        all_decode=all_decode, shuffled_kv_cache=False, use_alibi_slopes=False, use_qq_bias=False,
        num_sms=cu_count, target_num_prgms=cu_count * 4,
    )
    common = dict(
        query_ptr=q8, q_descale_ptr=q_descale, cache_ptr=kv_cache,
        block_tables_ptr=block_table, seq_lens_ptr=seqused_k, query_start_len_ptr=cu_seqlens_q,
        scale=scale, num_query_heads=num_query_heads, num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q8.stride(0), query_stride_1=q8.stride(1),
        stride_c0=kv_cache.stride(0), stride_c1=kv_cache.stride(1), stride_c2=kv_cache.stride(2),
        num_seqs=num_seqs, BLOCK_SIZE=block_size, ALL_DECODE=all_decode,
        ABLATE=int(os.environ.get("RADIANCE_KV4_ABLATE", "0")),
        **_LAYOUT,
    )

    if UA.use_2d_kernel(params, "triton"):
        config = UU.get_unified_attention_config("attn_2d", params, backend="triton")
        config["BLOCK_M"] = max(config["BLOCK_M"], triton.next_power_of_2(num_queries_per_kv))
        config["BLOCK_Q"] = config["BLOCK_M"] // num_queries_per_kv
        config["TILE_SIZE"] = _lds_fit(config, config["TILE_SIZE"])
        blocks = num_seqs if all_decode else num_tokens // config["BLOCK_Q"] + num_seqs
        kern = kv4_attention_2d[(num_kv_heads, blocks)](
            output_ptr=out, output_stride_0=out.stride(0), output_stride_1=out.stride(1),
            **common, **_kernel_config(config),
        )
        _note("2d", kern)
        LAST_LAUNCH["cfg"] = dict(kind="2d", **{k: config[k] for k in ("TILE_SIZE", "BLOCK_M", "num_warps", "num_stages", "waves_per_eu") if k in config})
        return out

    split = UU.get_unified_attention_config("kv_split", params, backend="triton")
    NUM_SEGMENTS = split["NUM_SEGMENTS"]
    TILE_SIZE = split["TILE_SIZE"]
    config = UU.get_unified_attention_config("attn_3d", params, backend="triton")
    if not all_decode:
        # The verify-batch cell. The fork's fp8 tune (BLOCK_M 64, TILE 32, 4 warps, 2 waves) spills
        # here -- the nibble streams and the score-side scales are extra live tiles -- and runs
        # 2.6x the fp8 kernel; 8 warps at TILE 16 hold the registers and run 1.8x (bench_kv4_attn,
        # 32k, one sequence). The split count stays the tune's.
        config.update(BLOCK_M=64, num_warps=8, waves_per_eu=1, num_stages=1)
        TILE_SIZE = 16
    config["BLOCK_M"] = max(config["BLOCK_M"], triton.next_power_of_2(num_queries_per_kv))
    config["BLOCK_Q"] = config["BLOCK_M"] // num_queries_per_kv
    TILE_SIZE = _lds_fit(config, TILE_SIZE)
    blocks = num_seqs if all_decode else num_tokens // config["BLOCK_Q"] + num_seqs
    if NUM_SEGMENTS > 1:
        segm_output = torch.empty(num_tokens, num_query_heads, NUM_SEGMENTS, F.HEAD,
                                  dtype=torch.float32, device=q8.device)
        segm_max = torch.empty(num_tokens, num_query_heads, NUM_SEGMENTS,
                               dtype=torch.float32, device=q8.device)
        segm_expsum = torch.empty_like(segm_max)
    else:
        segm_output = out
        segm_max = out
        segm_expsum = out
    kern = kv4_attention_3d[(blocks, num_kv_heads, NUM_SEGMENTS)](
        segm_output_ptr=segm_output, segm_max_ptr=segm_max, segm_expsum_ptr=segm_expsum,
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS, TILE_SIZE=TILE_SIZE,
        **common, **_kernel_config(config),
    )
    _note("3d", kern)
    LAST_LAUNCH["cfg"] = dict(kind="3d", TILE=TILE_SIZE, SEG=NUM_SEGMENTS, **{k: config[k] for k in ("BLOCK_M", "num_warps", "num_stages", "waves_per_eu") if k in config})
    if NUM_SEGMENTS == 1:
        return out
    rconf = UU.get_unified_attention_config("reduce", params, backend="triton")
    reduce_segments[(num_tokens, num_query_heads)](
        output_ptr=out, segm_output_ptr=segm_output, segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum, seq_lens_ptr=seqused_k, num_seqs=num_seqs,
        num_query_heads=num_query_heads, out_scale_ptr=None,
        output_stride_0=out.stride(0), output_stride_1=out.stride(1),
        block_table_stride=block_table.stride(0),
        HEAD_SIZE=F.HEAD, HEAD_SIZE_PADDED=F.HEAD, query_start_len_ptr=cu_seqlens_q,
        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS, TILE_SIZE=TILE_SIZE, BLOCK_Q=None, **rconf,
    )
    return out
