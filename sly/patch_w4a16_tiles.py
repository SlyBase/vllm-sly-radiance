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
17408, gs=128), measured DRAM-cold with sly/bench_w4a16_tiles.py on the R9700. Any shape or
M bucket not in the table falls through to the stock heuristic unchanged.
RADIANCE_W4A16_TILES=0 disables the table (A/B control, no rebuild).
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
