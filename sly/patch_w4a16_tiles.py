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
2 launches) or via tl.atomic_add into a zeroed fp32 [M, N] (3 launches, order not fixed). Entries of
_GFX12X_SPLITK take precedence over the tile table; split_k 1 always runs the stock kernel, so a table
row with split_k 1 is bit-identical to the tile table. RADIANCE_W4A16_SPLITK=0 disables it.
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

# --- 3. split-K kernel, reduce and table, placed after the drafter tile table ---
apply(F,
      '_GFX12X_DRAFT_BUCKETS = (8, 16, 32, 40, 64)\n',
      '_GFX12X_DRAFT_BUCKETS = (8, 16, 32, 40, 64)\n'
      '\n'
      '\n'
      '# --- radiance (sly/patch_w4a16_tiles.py): split-K for skinny M on gfx1201 ---\n'
      '@triton.jit\n'
      'def _radiance_w4a16_splitk_kernel(\n'
      '    a_ptr, b_ptr, scales_ptr, zp_ptr, p_ptr,\n'
      '    M, N, K, K8, num_groups, tiles_per_split,\n'
      '    group_size,\n'
      '    ZP_BIAS: tl.constexpr,\n'
      '    HAS_ZP: tl.constexpr,\n'
      '    ATOMIC: tl.constexpr,\n'
      '    BLOCK_M: tl.constexpr,\n'
      '    BLOCK_N: tl.constexpr,\n'
      '    BLOCK_K: tl.constexpr,\n'
      '):\n'
      '    # Body of _triton_w4a16_skinny_fmt_kernel over K tiles [k_lo, k_hi) of this split; the fp32\n'
      '    # partial goes to p_ptr ([SPLIT_K, M, N] slice pid_k, or atomically into [M, N]).\n'
      '    pid_m = tl.program_id(0)\n'
      '    pid_n = tl.program_id(1)\n'
      '    pid_k = tl.program_id(2)\n'
      '\n'
      '    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)\n'
      '    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)\n'
      '\n'
      '    exllama_shifts_row = (tl.arange(0, 8) // 2) * 4 + (tl.arange(0, 8) % 2) * 16\n'
      '    shifts_1d = tl.reshape(\n'
      '        tl.broadcast_to(exllama_shifts_row[None, :], (BLOCK_K // 8, 8)),\n'
      '        (BLOCK_K,),\n'
      '    )\n'
      '    shifts_full = tl.broadcast_to(shifts_1d[None, :], (BLOCK_N, BLOCK_K))\n'
      '\n'
      '    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)\n'
      '\n'
      '    k_lo = pid_k * tiles_per_split\n'
      '    k_hi = tl.minimum(k_lo + tiles_per_split, tl.cdiv(K, BLOCK_K))\n'
      '    for k_start in range(k_lo, k_hi):\n'
      '        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)\n'
      '        mask_k = offs_k < K\n'
      '\n'
      '        a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]\n'
      '        mask_a = (offs_m[:, None] < M) & mask_k[None, :]\n'
      '        a = tl.load(a_ptrs, mask=mask_a, other=0.0)\n'
      '\n'
      '        offs_k8 = k_start * (BLOCK_K // 8) + tl.arange(0, BLOCK_K // 8)\n'
      '        b_ptrs = b_ptr + offs_n[:, None] * K8 + offs_k8[None, :]\n'
      '        mask_b = (offs_n[:, None] < N) & (offs_k8[None, :] < K8)\n'
      '        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)\n'
      '\n'
      '        b = tl.interleave(b_packed, b_packed)\n'
      '        b = tl.interleave(b, b)\n'
      '        b = tl.interleave(b, b)\n'
      '        b = (b >> shifts_full) & 0xF\n'
      '\n'
      '        group_idx = (k_start * BLOCK_K) // group_size\n'
      '        scale_ptrs = scales_ptr + offs_n * num_groups + group_idx\n'
      '        scale_mask = offs_n < N\n'
      '        scales = tl.load(scale_ptrs, mask=scale_mask, other=1.0)\n'
      '\n'
      '        if HAS_ZP:\n'
      '            zp_ptrs = zp_ptr + (offs_n // 8) * num_groups + group_idx\n'
      '            zp_word = tl.load(zp_ptrs, mask=scale_mask, other=0)\n'
      '            zp_raw = (zp_word >> (4 * (offs_n % 8))) & 0xF\n'
      '            b_fp = (b - zp_raw[:, None]).to(scales.dtype) * scales[:, None]\n'
      '        else:\n'
      '            b_fp = (b - ZP_BIAS).to(scales.dtype) * scales[:, None]\n'
      '\n'
      '        b_fp_t = tl.trans(b_fp)\n'
      '        accumulator += tl.dot(a, b_fp_t, out_dtype=tl.float32)\n'
      '\n'
      '    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)\n'
      '    if ATOMIC:\n'
      '        tl.atomic_add(p_ptr + offs_m[:, None] * N + offs_n[None, :], accumulator, mask=mask_c)\n'
      '    else:\n'
      '        p_ptrs = p_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]\n'
      '        tl.store(p_ptrs, accumulator, mask=mask_c)\n'
      '\n'
      '\n'
      '@triton.jit\n'
      'def _radiance_w4a16_splitk_reduce(p_ptr, c_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):\n'
      '    # c = sum over the SPLIT_K fp32 partials, cast to the output dtype (split order fixed).\n'
      '    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n'
      '    mask = offs < MN\n'
      '    acc = tl.load(p_ptr + offs, mask=mask, other=0.0)\n'
      '    for s in tl.static_range(1, SPLIT_K):\n'
      '        acc += tl.load(p_ptr + s * MN + offs, mask=mask, other=0.0)\n'
      '    tl.store(c_ptr + offs, acc.to(c_ptr.dtype.element_ty), mask=mask)\n'
      '\n'
      '\n'
      '# (group_size, K, N, M bucket) -> (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, split_k, atomic).\n'
      '# Consulted before the tile table; filled from sly/bench_w4a16_tiles.py --splitk.\n'
      '_GFX12X_SPLITK: dict[tuple[int, int, int, int], tuple[int, int, int, int, int | None, int, int]] = {\n'
      '}\n'
      '\n'
      '\n'
      'def _gfx12x_splitk_override(group_size, K, N, M):\n'
      '    import os\n'
      '\n'
      '    if os.environ.get("RADIANCE_W4A16_SPLITK", "1") != "1":\n'
      '        return None\n'
      '    for b in _GFX12X_DRAFT_BUCKETS:\n'
      '        if M <= b:\n'
      '            return _GFX12X_SPLITK.get((group_size, K, N, b))\n'
      '    return None\n'
      '\n'
      '\n'
      'def triton_w4a16_splitk_gemm(a, b_q, scales, group_size, cfg, zp_bias=8, zp=None, c=None):\n'
      '    """triton_w4a16_skinny_fmt_gemm with an explicit (BLOCK_M, BLOCK_N, BLOCK_K, num_warps,\n'
      '    num_stages, split_k, atomic) config; split_k 1 runs the stock kernel unchanged."""\n'
      '    BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, split_k, atomic = cfg\n'
      '    M, K = a.shape\n'
      '    N = b_q.shape[0]\n'
      '    K8 = K // 8\n'
      '    num_groups = K // group_size\n'
      '    BLOCK_K = min(BLOCK_K, group_size)\n'
      '    has_zp = zp is not None\n'
      '    if c is None:\n'
      '        c = torch.empty((M, N), dtype=a.dtype, device=a.device)\n'
      '    extra_kwargs = {} if num_stages is None else {"num_stages": num_stages}\n'
      '    k_tiles = triton.cdiv(K, BLOCK_K)\n'
      '    tiles_per_split = triton.cdiv(k_tiles, max(1, split_k))\n'
      '    split_k = triton.cdiv(k_tiles, tiles_per_split)  # no empty splits\n'
      '    if split_k <= 1:\n'
      '        _triton_w4a16_skinny_fmt_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](\n'
      '            a, b_q, scales, zp if has_zp else scales, c, M, N, K, K8, num_groups,\n'
      '            group_size=group_size, ZP_BIAS=zp_bias, HAS_ZP=has_zp,\n'
      '            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=num_warps, **extra_kwargs)\n'
      '        return c\n'
      '    if atomic:\n'
      '        p = torch.zeros((M, N), dtype=torch.float32, device=a.device)\n'
      '    else:\n'
      '        p = torch.empty((split_k, M, N), dtype=torch.float32, device=a.device)\n'
      '    _radiance_w4a16_splitk_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), split_k)](\n'
      '        a, b_q, scales, zp if has_zp else scales, p, M, N, K, K8, num_groups, tiles_per_split,\n'
      '        group_size, ZP_BIAS=zp_bias, HAS_ZP=has_zp, ATOMIC=bool(atomic),\n'
      '        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, num_warps=num_warps, **extra_kwargs)\n'
      '    MN = M * N\n'
      '    _radiance_w4a16_splitk_reduce[(triton.cdiv(MN, 1024),)](\n'
      '        p, c, MN, SPLIT_K=1 if atomic else split_k, BLOCK=1024, num_warps=4)\n'
      '    return c\n',
      '_radiance_w4a16_splitk_kernel',
      'rdna_hybrid_w4a16: split-K kernel + table')

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
