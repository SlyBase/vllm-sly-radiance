#!/usr/bin/env python3
"""Build-time source patch for AITER's unified attention on RDNA (gfx1201 / R9700).
Six idempotent string replacements on the installed site-packages copy of
unified_attention.py.

AITER rebuilt this file's config machinery between the elif-chain API this patch was
originally written against and v0.1.21.post2: `select_3d_config`/`select_2d_config`
(Python elif ladders keyed on head_size/dtype, living right here in this file) are
gone. In their place: `get_unified_attention_config(op, params, backend, arch)` in
`aiter.ops.triton.utils.unified_attention_utils` looks a tuning dict up from a JSON
table per op ("attn_2d" | "attn_3d" | "reduce" | "kv_split"), keyed by axes
(head_size, max_seqlen_q, dtype, ...) with LEQ/GEQ/"any" fallback -- see that module's
own docstring. Two generic post-processing steps then run on whatever the table
returns: `compute_tile_params()` derives TILE_SIZE from TILE_SIZE_MIN/MAX bounds and
the runtime page size, and `compute_segment_params()` derives NUM_SEGMENTS. The tables
themselves are loaded with `json.load()` (see `config_utils.load_config_json`), i.e.
they are data files, not Python source -- `_patchlib.apply()`'s `ast.parse()`
validation would reject a JSON literal (`true`/`false`/`null` are not valid Python),
so this patch cannot live in the table. It has to live in the Python call sites that
consume `get_unified_attention_config()`'s result, which is exactly where AITER's own
elif ladders used to sit, so the shape of this file's fix is not that different from
before -- only where the values come from changed.

1. LDS fit (correctness, unconditional, both the 2D and 3D triton paths). The
   attention kernels stage a TILE_SIZE x next_pow2(head_size) K/V tile num_stages
   deep in shared memory, at the KV cache element size, plus ~256 B (see
   kernel_unified_attention_2d/_3d in _triton_kernels/attention/unified_attention.py:
   the `for j in range(...)` loop over KV tiles is what Triton's `num_stages`
   compile-time pipelining depth double/triple-buffers into LDS). AITER's attn_2d/
   attn_3d tables size that for CDNA's much larger LDS; the R9700 has 64 KiB, so
   several of its own picks do not fit and Triton raises OutOfResources at cudagraph
   capture:
     head_size 256, 2-byte KV : 64*256*2*2 + 256 = 65792
     head_size 512, fp8    KV : 64*512*1*2 + 256 = 65792   (Gemma4's global-attention layers)
   Both call sites now step num_stages, then TILE_SIZE, down until the tile fits.
   This is a hard requirement, not a preference, so it lives in source and applies
   whether or not a runtime tuned-config hook is installed (that hook is a tune;
   this is a correctness clamp) -- and it runs no matter what numbers the JSON table
   (or a future retune of it) happens to carry, which is the whole point of putting
   it in the Python side of the boundary instead.

   Consistency note (this is the part that is genuinely new architecture, not just a
   renamed old fix): for the 3D/split-KV path, TILE_SIZE is decided ONCE in
   `unified_attention()` from the "kv_split" op's config, then handed as a plain
   int argument to BOTH `_unified_attention_3d_triton()` (the attention kernel
   launch) and `_reduce_segments_triton()` (the segment-merge kernel launch). Both
   kernels independently recompute `tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS *
   TILE_SIZE)` from that same TILE_SIZE (see kernel_unified_attention_3d and
   reduce_segments in the _triton_kernels module -- literally the same expression in
   both) to know which KV tiles landed in which segment. If the two kernels ever see
   different TILE_SIZE values, reduce_segments walks a different tile/segment
   partition than the one attention actually wrote, silently merging the wrong
   segments. Because TILE_SIZE is a plain Python int (not a mutable container),
   clamping it *inside* `_unified_attention_3d_triton()` on its own local copy would
   never reach the `unified_attention()` variable that gets forwarded to the reduce
   call afterwards -- the two callers would quietly diverge. So `_unified_attention_
   3d_triton()` now returns its (possibly clamped, possibly re-tuned by patch 2
   below) TILE_SIZE, and the one call site in `unified_attention()` captures that
   return value into the same local variable that flows into `_reduce_segments_
   triton()` right after. The 2D path has no such coupling (no reduce step), so its
   fit clamp stays fully local to `_unified_attention_2d_triton()`.

2. bf16/fp16 (2-byte, incl. --kv-cache-dtype auto) 3D-decode tune. do_bench-optimal
   at head_size 256: TILE 16, warps 4, stages 2, waves 2, reduce warps 4 (warps=4 is
   the lever: +14% decode, 4-7x prefill). This must be a source patch rather than a
   new JSON table entry for the same reason as (1): the table is data, and a
   gfx1201-only override sitting in a shared-arch data file would need its own
   axis/schema support the table format doesn't have for this case. It also must
   stay a source patch rather than part of the tuned-config wrapper because that
   wrapper is bypassed for the bf16 3D path in-serve. Gated on DEVICE_ARCH ==
   "gfx1201" (the module already computes this at import time) rather than applied
   unconditionally: unlike the LDS-fit clamp, this is a performance tune for this one
   card, not a correctness floor, and must not silently override another arch's own
   attn_3d/reduce table entries if this same forked wheel is ever run elsewhere.
   The TILE_SIZE=16 override goes through the same return-value channel as patch 1,
   so reduce_segments always sees whichever TILE_SIZE attention actually used.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
F = SP / "aiter/ops/triton/attention/unified_attention.py"


# 1a. _unified_attention_2d_triton(): the attn_2d config's TILE_SIZE (always present,
# required by kernel_unified_attention_2d's launch) and num_stages (NOT required by
# that kernel's declared signature, since num_stages/num_warps/waves_per_eu are pure
# Triton compile kwargs -- absent from the table, they would silently fall through to
# Triton's own per-target default instead of anything we can see or clamp here, so
# `setdefault` pins one before the fit loop runs). Placed before the shuffled-KV
# override below it: a shuffled tile is pinned to block_size by kernel layout, not
# tunable, so the fit only ever needs to consider the table's own (non-shuffled) pick.
A_2D_PRE = (
    '    config = get_unified_attention_config("attn_2d", params, backend="triton")\n'
    "    config[\"BLOCK_M\"] = max(\n"
    '        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)\n'
    "    )\n"
    '    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv\n'
    '    assert config["BLOCK_Q"] >= 1\n'
    "    if params.shuffled_kv_cache:\n"
    '        config["TILE_SIZE"] = params.block_size\n'
)
INSERT_2D_FIT = (
    "    # --- RADIANCE LDS fit (2D): shrink the staged K/V tile into the R9700's\n"
    "    # 64 KiB LDS. See the module docstring for the full rationale.\n"
    '    config.setdefault("num_stages", 2)\n'
    "    _rad_el = 2 if params.kv_cache_dtype in (torch.bfloat16, torch.float16) else 1\n"
    "    _rad_hs = triton.next_power_of_2(params.head_size)\n"
    "    while (\n"
    '        config["num_stages"] > 1\n'
    '        and config["TILE_SIZE"] * _rad_hs * _rad_el * config["num_stages"] + 256 > 65536\n'
    "    ):\n"
    '        config["num_stages"] -= 1\n'
    "    while (\n"
    '        config["TILE_SIZE"] > 16\n'
    '        and config["TILE_SIZE"] * _rad_hs * _rad_el * config["num_stages"] + 256 > 65536\n'
    "    ):\n"
    '        config["TILE_SIZE"] //= 2\n'
    "\n"
)

# 1b. _unified_attention_3d_triton(): same fit, but on the TILE_SIZE parameter (the
# 3D path's TILE_SIZE lives outside the attn_3d config -- see the kv_split note in the
# module docstring) and config["num_stages"] (a required kernel argument here, so
# always present -- no setdefault needed, unlike 2D).
A_3D_PRE = (
    '    config = get_unified_attention_config("attn_3d", params, backend="triton")\n'
    "    config[\"BLOCK_M\"] = max(\n"
    '        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)\n'
    "    )\n"
    '    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv\n'
    '    assert config["BLOCK_Q"] >= 1\n'
    "\n"
    "    if params.all_decode:\n"
)
INSERT_3D_FIT = (
    '    config = get_unified_attention_config("attn_3d", params, backend="triton")\n'
    "    config[\"BLOCK_M\"] = max(\n"
    '        config["BLOCK_M"], triton.next_power_of_2(params.num_queries_per_kv)\n'
    "    )\n"
    '    config["BLOCK_Q"] = config["BLOCK_M"] // params.num_queries_per_kv\n'
    '    assert config["BLOCK_Q"] >= 1\n'
    "\n"
    "    # --- RADIANCE LDS fit (3D): shrink the staged K/V tile into the R9700's\n"
    "    # 64 KiB LDS. See the module docstring for the full rationale, including why\n"
    "    # the resulting TILE_SIZE has to be returned by this function rather than\n"
    "    # just reassigned locally.\n"
    "    _rad_el = 2 if params.kv_cache_dtype in (torch.bfloat16, torch.float16) else 1\n"
    "    _rad_hs = triton.next_power_of_2(params.head_size)\n"
    "    while (\n"
    '        config["num_stages"] > 1\n'
    '        and TILE_SIZE * _rad_hs * _rad_el * config["num_stages"] + 256 > 65536\n'
    "    ):\n"
    '        config["num_stages"] -= 1\n'
    "    while (\n"
    "        TILE_SIZE > 16\n"
    '        and TILE_SIZE * _rad_hs * _rad_el * config["num_stages"] + 256 > 65536\n'
    "    ):\n"
    "        TILE_SIZE //= 2\n"
    "\n"
    "    if params.all_decode:\n"
)

# 2a. Same function, right before the kernel launch: the gfx1201 bf16/fp16 tune.
A_3D_LAUNCH = (
    "    kernel_unified_attention_3d[\n"
    "        (total_num_q_blocks, params.num_kv_heads, NUM_SEGMENTS)\n"
    "    ](\n"
)
INSERT_3D_TUNE = (
    "    # --- RADIANCE bf16 3D-decode tune (2-byte KV, gfx1201) ---\n"
    "    # do_bench-optimal at head_size 256: TILE16 warps4 stages2 waves2 (warps4 is\n"
    "    # the lever: +14% decode, 4-7x prefill). Already LDS-safe by construction\n"
    "    # (16 * 256 * 2 * 2 + 256 = 16896 <= 65536), so no further clamping needed.\n"
    "    # Gated to this card: a tune for gfx1201, not a general RDNA default.\n"
    "    if (\n"
    '        DEVICE_ARCH == "gfx1201"\n'
    "        and params.head_size == 256\n"
    "        and params.kv_cache_dtype in (torch.bfloat16, torch.float16)\n"
    "    ):\n"
    "        TILE_SIZE = 16\n"
    '        config["num_warps"] = 4\n'
    '        config["num_stages"] = 2\n'
    '        config["waves_per_eu"] = 2\n'
    "\n"
    "    kernel_unified_attention_3d[\n"
    "        (total_num_q_blocks, params.num_kv_heads, NUM_SEGMENTS)\n"
    "    ](\n"
)

# 2b. End of _unified_attention_3d_triton(): return the (possibly clamped/tuned)
# TILE_SIZE so the caller can thread it into the matching reduce_segments call.
A_3D_TAIL = (
    "        IS_Q_FP8=(params.q_dtype == e4m3_dtype),\n"
    "        IS_KV_FP8=(params.kv_cache_dtype == e4m3_dtype),\n"
    "        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,\n"
    "        TILE_SIZE=TILE_SIZE,\n"
    "        **config,\n"
    "    )\n"
    "\n"
    "\n"
    "def _reduce_segments_triton(\n"
)
INSERT_3D_RETURN = (
    "        IS_Q_FP8=(params.q_dtype == e4m3_dtype),\n"
    "        IS_KV_FP8=(params.kv_cache_dtype == e4m3_dtype),\n"
    "        NUM_SEGMENTS_PER_SEQ=NUM_SEGMENTS,\n"
    "        TILE_SIZE=TILE_SIZE,\n"
    "        **config,\n"
    "    )\n"
    "    # RADIANCE: hand the final TILE_SIZE back so unified_attention() can forward\n"
    "    # the exact value this launch used into the paired reduce_segments call.\n"
    "    return TILE_SIZE\n"
    "\n"
    "\n"
    "def _reduce_segments_triton(\n"
)

# 3. unified_attention()'s 3D call site: capture the returned TILE_SIZE. This is the
# other half of the return added in 2b -- without it the fix above would only ever
# affect the attention kernel's own launch, while reduce_segments kept using the
# original (possibly too-large / not-yet-tuned) value.
A_CALL_SITE = (
    "        else:\n"
    "            _unified_attention_3d_triton(\n"
    "                params,\n"
    "                segm_output,\n"
    "                segm_max,\n"
    "                segm_expsum,\n"
    "                NUM_SEGMENTS,\n"
    "                TILE_SIZE,\n"
    "            )\n"
)
INSERT_CALL_SITE = (
    "        else:\n"
    "            # RADIANCE: capture the (possibly LDS-clamped / gfx1201-tuned)\n"
    "            # TILE_SIZE this launch actually used, so the reduce_segments call\n"
    "            # below stays in lockstep with it instead of silently diverging.\n"
    "            TILE_SIZE = _unified_attention_3d_triton(\n"
    "                params,\n"
    "                segm_output,\n"
    "                segm_max,\n"
    "                segm_expsum,\n"
    "                NUM_SEGMENTS,\n"
    "                TILE_SIZE,\n"
    "            )\n"
)

# 4. _reduce_segments_triton(): the paired reduce-warps tune for the same gfx1201
# bf16/head_size-256 case as patch 2a. Independent of the TILE_SIZE-propagation
# machinery above (num_warps here is purely this kernel's own launch tuning, read
# from its own "reduce" op config -- it does not need anything threaded in from the
# 3D launch), but it is part of the same do_bench-measured tuple, so it lives next to
# it in the module docstring's patch 2.
A_REDUCE_PRE = (
    "    head_size_padded = triton.next_power_of_2(params.head_size)\n"
    '    config = get_unified_attention_config("reduce", params, backend="triton")\n'
    "\n"
    "    reduce_segments[(params.num_tokens, params.num_query_heads)](\n"
)
INSERT_REDUCE_TUNE = (
    "    head_size_padded = triton.next_power_of_2(params.head_size)\n"
    '    config = get_unified_attention_config("reduce", params, backend="triton")\n'
    "\n"
    "    # --- RADIANCE reduce-warps tune (pairs with the attn_3d bf16 tune above) ---\n"
    "    if (\n"
    '        DEVICE_ARCH == "gfx1201"\n'
    "        and params.head_size == 256\n"
    "        and params.kv_cache_dtype in (torch.bfloat16, torch.float16)\n"
    "    ):\n"
    '        config["num_warps"] = 4\n'
    "\n"
    "    reduce_segments[(params.num_tokens, params.num_query_heads)](\n"
)


def main():
    apply(F, A_2D_PRE, A_2D_PRE + INSERT_2D_FIT, "RADIANCE LDS fit (2D)", "unified_attention LDS fit (2D)")
    apply(F, A_3D_PRE, INSERT_3D_FIT, "RADIANCE LDS fit (3D)", "unified_attention LDS fit (3D)")
    apply(F, A_3D_LAUNCH, INSERT_3D_TUNE, "RADIANCE bf16 3D-decode tune (2-byte KV, gfx1201)", "unified_attention bf16 3D-decode tune")
    apply(F, A_3D_TAIL, INSERT_3D_RETURN, "RADIANCE: hand the final TILE_SIZE back", "unified_attention 3D TILE_SIZE return")
    apply(F, A_CALL_SITE, INSERT_CALL_SITE, "RADIANCE: capture the (possibly LDS-clamped", "unified_attention 3D TILE_SIZE propagation")
    apply(F, A_REDUCE_PRE, INSERT_REDUCE_TUNE, "RADIANCE reduce-warps tune", "unified_attention reduce-warps tune")


if __name__ == "__main__":
    main()
