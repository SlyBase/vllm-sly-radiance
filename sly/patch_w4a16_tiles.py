#!/usr/bin/env python3
"""Per-shape Triton tile overrides for the DFlash2 W4A16 drafter on gfx1201 (vLLM 0.29.0).

vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py routes M <= 5 to the
HIP skinny kernel and everything else to _triton_w4a16_skinny_fmt_kernel. A DFlash drafter
never sees M <= 5 (block_size 8 -> M = 8 x running sequences), so every drafter GEMM takes the
Triton path, and the gfx12x tile heuristic there was tuned on Llama-3.1-8B AWQ shapes: at
M <= 32 it picks BLOCK 16x16x128 / 4 warps -- 2176 tiny workgroups for the 34816-wide gate_up,
267 GB/s on the R9700 (profile D, vllm7, 2026-09-15). The per-shape override table that the
file already carries (_GFX1X_PREFILL_OVERRIDES) is consulted in the gfx1151 branch only.

This adds a gfx12x override table keyed by (group_size, K, N, M bucket) for the four drafter
shapes of syvai/Qwen3.8-27B-DFlash2-W4A16 (hidden 5120, 32x128 q / 8x128 kv, intermediate
17408, gs=128), its fc layer, and the int4 lm_head of sly/mxfp4/radiance_lmhead_int4.py
(N=248320: the stock 16-column tiles mean 15520 workgroups, 250 GB/s -- slower than the fp8
hipBLASLt head it replaces; with the table 502 GB/s at M=8), plus the three target shapes of a
compressed-tensors INT4 Qwen3.8-27B that the drafter does not share (qkvz 16384x5120, attention
qkv 14336x5120, out/o 5120x6144: stock 1.2-2.9x slower), all measured DRAM-cold with
sly/bench_w4a16_tiles.py on the R9700. Any shape or M bucket not in the table falls through
to the stock heuristic unchanged.
RADIANCE_W4A16_TILES=0 disables the table (A/B control, no rebuild).

Split-K (hunks 3 + 4, 0.4.1): the stock kernel runs the whole K loop in one workgroup, so an N=5120
shape at M <= 16 gets 160-320 workgroups of 2 waves for 64 CUs -- latency-bound at 266-331 GB/s
(INT4 target down_proj K=17408: 139 us against 82 us for the split-K MXFP4 decode kernel on the same
shape). _radiance_w4a16_splitk_kernel adds grid axis 2 over group-aligned K ranges; the fp32 partials
go either to a [SPLIT_K, M, N] buffer summed and cast by _radiance_w4a16_splitk_reduce (deterministic,
2 launches) or via tl.atomic_add into a zeroed fp32 [M, N] (3 launches, order not fixed). The same kernel
carries two cheaper dequant schemes (DEQ 1: scale applied to the fp32 tile result; DEQ 2: bf16 magic-number
nibble conversion with zero point and scale folded into the tile result) and an interleave-free unpack
(UNPACK 1: 8 dots against stride-8 activation columns), selectable per table entry. Entries of
_GFX12X_SPLITK take precedence over the tile table; split_k 1 with the stock dequant runs the stock kernel,
so such a row is bit-identical to the tile table. RADIANCE_W4A16_SPLITK=0 disables the table.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py"

# --- 1. the table, placed right after _GFX1X_PREFILL_OVERRIDES ---
apply(F,
      'def triton_w4a16_skinny_fmt_gemm(\n',
      '# --- radiance (sly/patch_w4a16_tiles.py): gfx1201 tiles for the DFlash2 W4A16 drafter ---\n'
      '# (group_size, K, N, M bucket) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages).\n'
      '# M bucket = smallest of (8, 16, 32, 40, 64) that is >= M; larger M keeps the heuristic.\n'
      '# Measured DRAM-cold on the R9700 (sly/bench_w4a16_tiles.py, 2026-09-15).\n'
      '_GFX12X_DRAFT_OVERRIDES: dict[tuple[int, int, int, int], tuple[int, int, int, int, int | None]] = {\n'
      '    # qkv_proj  N=6144  K=5120  (stock at M=8/16/32: 77/78/137 us -> 52/52/58)\n'
      '    (128, 5120, 6144, 8): (16, 32, 128, 2, None),\n'
      '    (128, 5120, 6144, 16): (16, 32, 128, 2, None),\n'
      '    (128, 5120, 6144, 32): (32, 32, 128, 4, None),\n'
      '    (128, 5120, 6144, 40): (64, 64, 64, 4, None),\n'
      '    (128, 5120, 6144, 64): (64, 64, 64, 4, None),\n'
      '    # o_proj  N=5120  K=4096  (57/57/96 us -> 47/46/47)\n'
      '    (128, 4096, 5120, 8): (16, 16, 128, 2, 1),\n'
      '    (128, 4096, 5120, 16): (16, 16, 128, 2, 3),\n'
      '    (128, 4096, 5120, 32): (32, 32, 128, 4, None),\n'
      '    # gate_up_proj  N=34816  K=5120  (369/454/944/453/516 us -> 219/224/272/347/361)\n'
      '    (128, 5120, 34816, 8): (16, 64, 128, 4, None),\n'
      '    (128, 5120, 34816, 16): (16, 64, 128, 4, None),\n'
      '    (128, 5120, 34816, 32): (32, 64, 128, 4, None),\n'
      '    (128, 5120, 34816, 40): (64, 64, 128, 4, None),\n'
      '    (128, 5120, 34816, 64): (64, 64, 128, 4, None),\n'
      '    # down_proj  N=5120  K=17408  (248/212/412/233/245 us -> 139/141/161/208/218)\n'
      '    (128, 17408, 5120, 8): (16, 32, 128, 2, None),\n'
      '    (128, 17408, 5120, 16): (16, 32, 128, 2, None),\n'
      '    (128, 17408, 5120, 32): (32, 32, 128, 4, None),\n'
      '    (128, 17408, 5120, 40): (64, 32, 128, 4, None),\n'
      '    (128, 17408, 5120, 64): (64, 32, 128, 4, None),\n'
      '    # fc (combine_hidden_states, ReplicatedLinear)  N=5120  K=25600  -- runs BEFORE the\n'
      '    # draft padding on the target-scheduled tokens, so M = 8 x seqs at conc <= 4\n'
      '    # (288/293/575/316/326 us -> 205/205/205/286/294; every config bit-identical)\n'
      '    (128, 25600, 5120, 8): (16, 32, 128, 2, None),\n'
      '    (128, 25600, 5120, 16): (16, 32, 128, 2, None),\n'
      '    (128, 25600, 5120, 32): (32, 32, 128, 4, None),\n'
      '    (128, 25600, 5120, 40): (64, 32, 128, 4, None),\n'
      '    (128, 25600, 5120, 64): (64, 32, 128, 4, None),\n'
      '    # lm_head int4 (sly/mxfp4/radiance_lmhead_int4.py, RADIANCE_LMHEAD_INT4=1)  N=248320\n'
      '    # K=5120 -- verify M = 8 x seqs, draft bootstrap M = 7 x seqs; 656 MB per call\n'
      '    # (2623/2709/5327/2715/2824 us -> 1305/1333/1585/2099/2185; fp8 hipBLASLt: 2459-2623)\n'
      '    (128, 5120, 248320, 8): (16, 64, 128, 4, 1),\n'
      '    (128, 5120, 248320, 16): (16, 64, 128, 4, 1),\n'
      '    (128, 5120, 248320, 32): (32, 128, 64, 8, None),\n'
      '    (128, 5120, 248320, 40): (64, 128, 64, 8, None),\n'
      '    (128, 5120, 248320, 64): (64, 64, 64, 4, 1),\n'
      '    # INT4 target (RedHatAI/Qwen3.8-27B-INT4, compressed-tensors W4A16 g128): the shapes the drafter\n'
      '    # does not share -- GDN in_proj_qkvz, attention qkv (q with gate), GDN out_proj = attention o_proj\n'
      '    # (N 5120 x K 6144); bench_w4a16_tiles.py --target, 2026-09-17\n'
      '    # INT4 target qkvz  N=16384  K=5120  (stock at M=8/16/32/40/64: 183/183/331/184/183 us -> 107/106/119/152/155)\n'
      '    (128, 5120, 16384, 8): (16, 64, 128, 4, 1),\n'
      '    (128, 5120, 16384, 16): (16, 32, 128, 2, 1),\n'
      '    (128, 5120, 16384, 32): (32, 32, 128, 4, None),\n'
      '    (128, 5120, 16384, 40): (64, 128, 128, 8, None),\n'
      '    (128, 5120, 16384, 64): (64, 128, 64, 8, None),\n'
      '    # INT4 target out_o  N=5120  K=6144  (stock at M=8/16/32/40/64: 78/80/142/105/97 us -> 61/61/61/79/81)\n'
      '    (128, 6144, 5120, 8): (16, 16, 128, 2, 3),\n'
      '    (128, 6144, 5120, 16): (16, 16, 128, 2, 3),\n'
      '    (128, 6144, 5120, 32): (32, 32, 128, 4, None),\n'
      '    (128, 6144, 5120, 40): (64, 32, 64, 4, None),\n'
      '    (128, 6144, 5120, 64): (64, 32, 64, 4, None),\n'
      '    # INT4 target attn_qkv  N=14336  K=5120  (stock at M=8/16/32/40/64: 157/163/330/169/185 us -> 93/97/116/145/146)\n'
      '    (128, 5120, 14336, 8): (16, 64, 128, 4, 1),\n'
      '    (128, 5120, 14336, 16): (16, 64, 128, 2, 1),\n'
      '    (128, 5120, 14336, 32): (32, 64, 128, 4, None),\n'
      '    (128, 5120, 14336, 40): (64, 64, 128, 4, 1),\n'
      '    (128, 5120, 14336, 64): (64, 64, 128, 4, None),\n'
      '}\n'
      '_GFX12X_DRAFT_BUCKETS = (8, 16, 32, 40, 64)\n'
      '\n'
      '\n'
      'def _gfx12x_draft_override(group_size, K, N, M):\n'
      '    import os\n'
      '\n'
      '    if os.environ.get("RADIANCE_W4A16_TILES", "1") != "1":\n'
      '        return None\n'
      '    for b in _GFX12X_DRAFT_BUCKETS:\n'
      '        if M <= b:\n'
      '            return _GFX12X_DRAFT_OVERRIDES.get((group_size, K, N, b))\n'
      '    return None\n'
      '\n'
      '\n'
      'def triton_w4a16_skinny_fmt_gemm(\n',
      '_GFX12X_DRAFT_OVERRIDES',
      'rdna_hybrid_w4a16: gfx12x drafter tile table')

# --- 2. consult it at the top of the gfx12x branch ---
apply(F,
      '    if _on_gfx12x():\n'
      '        # Tuned on gfx1201 (Radeon AI PRO R9700, 32 CUs, 32-wide wavefronts)\n'
      '        # using Llama-3.1-8B AWQ weight shapes with group_size=128.\n'
      '        if M <= 32:\n',
      '    if _on_gfx12x():\n'
      '        # Tuned on gfx1201 (Radeon AI PRO R9700, 32 CUs, 32-wide wavefronts)\n'
      '        # using Llama-3.1-8B AWQ weight shapes with group_size=128.\n'
      '        _radiance_override = _gfx12x_draft_override(group_size, K, N, M)\n'
      '        if _radiance_override is not None:\n'
      '            BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages = _radiance_override\n'
      '        elif M <= 32:\n',
      '_radiance_override = _gfx12x_draft_override(',
      'rdna_hybrid_w4a16: gfx12x branch consults the drafter tile table')

# --- 3. split-K / fast-dequant kernel, reduce and table, placed after the drafter tile table ---
# Plain source (not a chain of string literals) so it reads like the kernel it becomes.
_SPLITK_SRC = r'''

# --- radiance (sly/patch_w4a16_tiles.py): split-K + fast dequant for skinny M on gfx1201 ---
@triton.jit
def _radiance_w4a16_splitk_kernel(
    a_ptr, b_ptr, scales_ptr, zp_ptr, p_ptr,
    M, N, K, K8, num_groups, tiles_per_split,
    group_size,
    ZP_BIAS: tl.constexpr,
    HAS_ZP: tl.constexpr,
    ATOMIC: tl.constexpr,
    DIRECT: tl.constexpr,
    DEQ: tl.constexpr,
    UNPACK: tl.constexpr,
    MAGIC: tl.constexpr,
    MAGIC_F: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # _triton_w4a16_skinny_fmt_kernel over the K tiles [k_lo, k_hi) of split pid_k. BLOCK_K <= group_size,
    # so one scale (and zero point) per row covers a whole K tile, which DEQ 1/2 exploit:
    #   DEQ 0  as stock: w = (q - zp) * scale in the activation dtype, one dot per tile
    #   DEQ 1  w = q - zp (exact), the scale multiplies the fp32 tile result instead of every weight
    #   DEQ 2  w = bits(MAGIC | q) = MAGIC_F + q exactly (no int->float convert, no subtract), zero point
    #          and scale folded into the tile result: (dot - rowsum(a) * (MAGIC_F + zp)) * scale
    #   UNPACK 0  3x tl.interleave + per-column shifts back to natural K order (stock)
    #   UNPACK 1  no interleave: nibble j of every packed word is K index 8i+j, so 8 dots over K/8 against
    #             the matching stride-8 columns of A (needs BLOCK_K // 8 >= 16)
    # Output: DIRECT -> cast into p_ptr (= c, split_k 1); ATOMIC -> fp32 atomic_add into [M, N];
    # else the fp32 partial into slice pid_k of [SPLIT_K, M, N].
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    exllama_shifts_row = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16
    shifts_1d = tl.reshape(
        tl.broadcast_to(exllama_shifts_row[None, :], (BLOCK_K // 8, 8)),
        (BLOCK_K,),
    )
    shifts_full = tl.broadcast_to(shifts_1d[None, :], (BLOCK_N, BLOCK_K))

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k_lo = pid_k * tiles_per_split
    k_hi = tl.minimum(k_lo + tiles_per_split, tl.cdiv(K, BLOCK_K))
    for k_start in range(k_lo, k_hi):
        offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)
        b_ptrs = b_ptr + offs_n[:, None] * K8 + offs_k8[None, :]
        mask_b = mask_n[:, None] & (offs_k8[None, :] < K8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        group_idx = (k_start * BLOCK_K) // group_size
        scales = tl.load(scales_ptr + offs_n * num_groups + group_idx, mask=mask_n, other=1.0)
        if HAS_ZP:
            zp_word = tl.load(zp_ptr + (offs_n // 8) * num_groups + group_idx, mask=mask_n, other=0)
            zp_raw = (zp_word >> (4 * (offs_n % 8))) & 0xF
            zp_col = zp_raw[:, None]
        else:
            zp_col = ZP_BIAS

        tile = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        rowsum = tl.zeros((BLOCK_M,), dtype=tl.float32)
        if UNPACK == 0:
            offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
            a = tl.load(a_ptr + offs_m[:, None] * K + offs_k[None, :],
                        mask=mask_m[:, None] & (offs_k[None, :] < K), other=0.0)
            b = tl.interleave(b_packed, b_packed)
            b = tl.interleave(b, b)
            b = tl.interleave(b, b)
            b = (b >> shifts_full) & 0xF
            if DEQ == 0:
                w = (b - zp_col).to(scales.dtype) * scales[:, None]
            elif DEQ == 1:
                w = (b - zp_col).to(scales.dtype)
            else:
                w = (b | MAGIC).to(tl.int16).to(scales.dtype, bitcast=True)
                rowsum += tl.sum(a.to(tl.float32), axis=1)
            tile += tl.dot(a, tl.trans(w), out_dtype=tl.float32)
        else:
            for j in tl.static_range(8):
                offs_kj = k_start * BLOCK_K + tl.arange(0, BLOCK_K // 8) * 8 + j
                a_j = tl.load(a_ptr + offs_m[:, None] * K + offs_kj[None, :],
                              mask=mask_m[:, None] & (offs_kj[None, :] < K), other=0.0)
                b_j = (b_packed >> ((j // 2) * 4 + (j % 2) * 16)) & 0xF
                if DEQ == 0:
                    w_j = (b_j - zp_col).to(scales.dtype) * scales[:, None]
                elif DEQ == 1:
                    w_j = (b_j - zp_col).to(scales.dtype)
                else:
                    w_j = (b_j | MAGIC).to(tl.int16).to(scales.dtype, bitcast=True)
                    rowsum += tl.sum(a_j.to(tl.float32), axis=1)
                tile += tl.dot(a_j, tl.trans(w_j), out_dtype=tl.float32)

        if DEQ == 0:
            accumulator += tile
        elif DEQ == 1:
            accumulator += tile * scales.to(tl.float32)[None, :]
        else:
            if HAS_ZP:
                zf = MAGIC_F + zp_raw.to(tl.float32)
                tile = tile - rowsum[:, None] * zf[None, :]
            else:
                tile = tile - rowsum[:, None] * (MAGIC_F + ZP_BIAS)
            accumulator += tile * scales.to(tl.float32)[None, :]

    mask_c = mask_m[:, None] & mask_n[None, :]
    if DIRECT:
        tl.store(p_ptr + offs_m[:, None] * N + offs_n[None, :],
                 accumulator.to(p_ptr.dtype.element_ty), mask=mask_c)
    elif ATOMIC:
        tl.atomic_add(p_ptr + offs_m[:, None] * N + offs_n[None, :], accumulator, mask=mask_c)
    else:
        tl.store(p_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :], accumulator, mask=mask_c)


@triton.jit
def _radiance_w4a16_splitk_reduce(p_ptr, c_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    # c = sum over the SPLIT_K fp32 partials, cast to the output dtype (split order fixed).
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.load(p_ptr + offs, mask=mask, other=0.0)
    for s in tl.static_range(1, SPLIT_K):
        acc += tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(c_ptr + offs, acc.to(c_ptr.dtype.element_ty), mask=mask)


# (group_size, K, N, M bucket) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, split_k, atomic
# [, deq, unpack]) -- deq/unpack default 0 (stock dequant). Consulted before the tile table; from
# sly/bench_w4a16_tiles.py --splitk (R9700, DRAM-cold, "today" = the tile table above).
_GFX12X_SPLITK: dict[tuple[int, int, int, int], tuple] = {
@@SPLITK_TABLE@@}


def _gfx12x_splitk_override(group_size, K, N, M):
    import os

    if os.environ.get("RADIANCE_W4A16_SPLITK", "1") != "1":
        return None
    for b in _GFX12X_DRAFT_BUCKETS:
        if M <= b:
            return _GFX12X_SPLITK.get((group_size, K, N, b))
    return None


# One fp32 split-K workspace per device, shared by every call: GEMM and reduce are stream-ordered, so a
# call's partials are consumed before the next call writes. Sized on first use to the largest table need,
# so production never reallocates; per-call buffers inside CUDA-graph capture had cost ~13 MB of graph pool
# (KV pool 384,316 -> 383,911 tokens, window B 2026-09-28).
_RADIANCE_SK_WS: dict = {}


def _radiance_sk_ws_table_max():
    need = 0
    for (_gs, _k, n, bucket), e in _GFX12X_SPLITK.items():
        if e[5] > 1 and not e[6]:
            need = max(need, e[5] * bucket * n)
    return need


def _radiance_sk_workspace(device, numel):
    ws = _RADIANCE_SK_WS.get(device)
    if ws is not None and ws.numel() >= numel:
        return ws[:numel]
    if ws is not None and device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        # an off-table shape bigger than the shared buffer: a captured graph may already point at the
        # shared one, so never replace it during capture -- this call gets its own (graph-pool) buffer
        return torch.empty(numel, dtype=torch.float32, device=device)
    ws = torch.empty(max(numel, _radiance_sk_ws_table_max()), dtype=torch.float32, device=device)
    _RADIANCE_SK_WS[device] = ws
    return ws[:numel]


def triton_w4a16_splitk_gemm(a, b_q, scales, group_size, cfg, zp_bias=8, zp=None, c=None):
    """triton_w4a16_skinny_fmt_gemm with an explicit (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages,
    split_k, atomic[, deq, unpack]) config; split_k 1 with deq 0 / unpack 0 runs the stock kernel."""
    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, split_k, atomic = cfg[:7]
    deq, unpack = (tuple(cfg[7:9]) + (0, 0))[:2]
    M, K = a.shape
    N = b_q.shape[0]
    K8 = K // 8
    num_groups = K // group_size
    BLOCK_K = min(BLOCK_K, group_size)
    if unpack:
        assert BLOCK_K // 8 >= 16, "UNPACK 1 needs BLOCK_K >= 128 (tl.dot K >= 16)"
    has_zp = zp is not None
    if c is None:
        c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    extra_kwargs = {} if num_stages is None else {"num_stages": num_stages}
    k_tiles = triton.cdiv(K, BLOCK_K)
    tiles_per_split = triton.cdiv(k_tiles, max(1, split_k))
    split_k = triton.cdiv(k_tiles, tiles_per_split)  # no empty splits
    if split_k <= 1 and not deq and not unpack:
        _triton_w4a16_skinny_fmt_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
            a, b_q, scales, zp if has_zp else scales, c, M, N, K, K8, num_groups,
            group_size=group_size, ZP_BIAS=zp_bias, HAS_ZP=has_zp,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=num_warps, **extra_kwargs)
        return c
    direct = split_k <= 1
    if direct:
        p = c
    elif atomic:
        p = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    else:
        p = _radiance_sk_workspace(a.device, split_k * M * N)
    _radiance_w4a16_splitk_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)](
        a, b_q, scales, zp if has_zp else scales, p, M, N, K, K8, num_groups, tiles_per_split,
        group_size, ZP_BIAS=zp_bias, HAS_ZP=has_zp, ATOMIC=bool(atomic and not direct), DIRECT=direct,
        DEQ=deq, UNPACK=unpack,
        # MAGIC | q is MAGIC_F + q exactly: fp16 1024.0 (10 mantissa bits), bf16 128.0 (7)
        MAGIC=0x6400 if a.dtype == torch.float16 else 0x4300,
        MAGIC_F=1024.0 if a.dtype == torch.float16 else 128.0,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=num_warps, **extra_kwargs)
    if not direct:
        MN = M * N
        _radiance_w4a16_splitk_reduce[(triton.cdiv(MN, 1024),)](
            p, c, MN, SPLIT_K=1 if atomic else split_k, BLOCK=1024, num_warps=4)
    return c
'''

# window A2 sweep, 2026-09-28 (R9700, 300 W, DRAM-cold; entries >= 5 % faster than the tile table).
# Fields 8/9 = deq/unpack; deq 2 (magic-number bf16, folded zero point + scale) wins at M <= 32 almost
# everywhere, split-K on the K-heavy N=5120 shapes, deq 1 at M = 40/64.
_SPLITK_TABLE = [
    # down  N=5120  K=17408: M=8 150->95 / M=16 130->99 / M=32 151->124 / M=40 203->184 us
    ((128, 17408, 5120, 8), (16, 32, 128, 2, None, 2, 0, 2, 0)),
    ((128, 17408, 5120, 16), (16, 128, 128, 4, None, 6, 0, 2, 0)),
    ((128, 17408, 5120, 32), (32, 128, 128, 4, None, 8, 0, 2, 0)),
    ((128, 17408, 5120, 40), (64, 64, 128, 4, None, 12, 0, 1, 0)),
    # out_o  N=5120  K=6144: M=8 61->48 / M=16 61->51 us
    ((128, 6144, 5120, 8), (16, 128, 128, 8, None, 8, 0, 2, 1)),
    ((128, 6144, 5120, 16), (16, 32, 128, 2, None, 2, 0, 2, 0)),
    # fc  N=5120  K=25600: M=8 197->131 / M=16 201->134 / M=32 208->169 / M=40 294->269 us
    ((128, 25600, 5120, 8), (16, 128, 128, 4, None, 2, 0, 2, 0)),
    ((128, 25600, 5120, 16), (16, 64, 128, 4, None, 2, 0, 2, 0)),
    ((128, 25600, 5120, 32), (32, 128, 128, 4, None, 8, 0, 2, 0)),
    ((128, 25600, 5120, 40), (64, 64, 128, 4, None, 6, 0, 1, 0)),
    # o  N=5120  K=4096: M=8 45->38 / M=16 42->39 / M=40 59->54 / M=64 60->55 us
    ((128, 4096, 5120, 8), (16, 64, 128, 4, None, 6, 0, 2, 0)),
    ((128, 4096, 5120, 16), (16, 128, 128, 8, None, 4, 0, 2, 1)),
    ((128, 4096, 5120, 40), (64, 32, 128, 4, None, 1, 0, 1, 0)),
    ((128, 4096, 5120, 64), (64, 32, 128, 4, None, 1, 0, 1, 0)),
    # qkvz  N=16384  K=5120: M=8 106->82 / M=16 110->81 / M=32 128->101 / M=40 174->158 us
    ((128, 5120, 16384, 8), (16, 128, 128, 2, None, 1, 0, 2, 0)),
    ((128, 5120, 16384, 16), (16, 128, 128, 2, None, 1, 0, 2, 0)),
    ((128, 5120, 16384, 32), (32, 128, 128, 4, None, 1, 0, 2, 0)),
    ((128, 5120, 16384, 40), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    # attn_qkv  N=14336  K=5120: M=8 89->76 / M=16 111->76 / M=32 117->99 / M=40 157->144 / M=64 164->145 us
    ((128, 5120, 14336, 8), (16, 64, 128, 4, None, 1, 0, 2, 0)),
    ((128, 5120, 14336, 16), (16, 64, 128, 4, None, 1, 0, 2, 0)),
    ((128, 5120, 14336, 32), (32, 128, 128, 4, None, 2, 0, 2, 0)),
    ((128, 5120, 14336, 40), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    ((128, 5120, 14336, 64), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    # qkv  N=6144  K=5120: M=8 53->39 / M=16 46->40 / M=40 74->67 / M=64 75->70 us
    ((128, 5120, 6144, 8), (16, 64, 128, 4, None, 1, 0, 2, 0)),
    ((128, 5120, 6144, 16), (16, 64, 128, 4, None, 1, 0, 2, 0)),
    ((128, 5120, 6144, 40), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    ((128, 5120, 6144, 64), (64, 64, 128, 4, None, 1, 0, 1, 0)),
    # gate_up  N=34816  K=5120: M=8 200->178 / M=16 203->181 / M=32 258->226 us
    ((128, 5120, 34816, 8), (16, 128, 128, 8, None, 1, 0, 2, 0)),
    ((128, 5120, 34816, 16), (16, 128, 128, 8, None, 1, 0, 2, 0)),
    ((128, 5120, 34816, 32), (32, 128, 128, 4, None, 1, 0, 2, 0)),
]

apply(F,
      '_GFX12X_DRAFT_BUCKETS = (8, 16, 32, 40, 64)\n',
      '_GFX12X_DRAFT_BUCKETS = (8, 16, 32, 40, 64)\n' + _SPLITK_SRC.replace(
          '@@SPLITK_TABLE@@', ''.join(f'    {k}: {v},\n' for k, v in _SPLITK_TABLE)),
      '_radiance_w4a16_splitk_kernel',
      'rdna_hybrid_w4a16: split-K / fast-dequant kernel + table')

# --- 4. the dispatch takes a split-K entry before the stock launch ---
apply(F,
      '    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))\n',
      '    if _on_gfx12x():\n'
      '        _radiance_sk = _gfx12x_splitk_override(group_size, K, N, M)\n'
      '        if _radiance_sk is not None:\n'
      '            return triton_w4a16_splitk_gemm(a, b_q, scales, group_size, _radiance_sk,\n'
      '                                            zp_bias=zp_bias, zp=zp, c=c)\n'
      '\n'
      '    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))\n',
      '_radiance_sk = _gfx12x_splitk_override(',
      'rdna_hybrid_w4a16: dispatch consults the split-K table')
