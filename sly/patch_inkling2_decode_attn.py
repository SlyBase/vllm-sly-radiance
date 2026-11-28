#!/usr/bin/env python3
"""Inkling 2 decode/verify attention on gfx12: fill the card and skip dead work at long context.

Root cause (rocprofv3 + sly/inkling2/bench_decode_attn.py + offline gfx1201 ISA, R9700):
  1. `plan_decode_attention` sizes the split-KV grid as `num_sms // 8` per KV head, tuned for 188 NVIDIA SMs.
     `torch` reports 32 compute units for the R9700 (WGPs), so one sequence launched 32 programs walking ~37k
     keys serially at 150k context (6.7 ms/call, 46 GB/s). `CTAS_PER_SM * num_sms // work` launches 64 splits
     x 8 KV heads = 512 programs.
  2. `_rel_attention_kernel` masked every key block even though a full layer's relative bias only covers the
     last E keys. Key blocks older than `pos_lo - E` run unmasked (`RADIANCE_INKLING2_ATTN_BULK=1`).
  3. Because `PAGE % BLOCK_N == 0`, every key block sits inside one physical 128-token page at contiguous slots
     `slot0 .. slot0 + BLOCK_N - 1`. Without a scalar page lookup and 16-byte / 128-byte stride alignment hints,
     Triton emitted 126-254 `global_load_u8` 1-byte loads and spilled 149-662 VGPRs per thread on gfx1201.
     `_load_fp8_kv_block` (`RADIANCE_INKLING2_VEC_KV=1`) emits 4 x `global_load_b128` with 0 VGPR spills in both
     the unmasked bulk loop and the masked loop (serving all 40 local SWA layer visits + global masked tails),
     paired with `(BLOCK_N=32, warps=8)` on global decode and `(BLOCK_N=32, warps=4, 33 splits)` on local SWA.

Knobs (RADIANCE_INKLING2_*): DECODE_SPLITS (1; 0 = stock decode plan, tile, masked loop and scalar loads),
VEC_KV (1), DECODE_BLOCK_N (32), DECODE_WARPS (8), LOCAL_BLOCK_N (32), LOCAL_WARPS (4), ATTN_BULK (1),
DECODE_CTAS_PER_SM (16).Idempotent (sentinel per hunk). Runs after patch_inkling2_gfx12 and before
patch_inkling2_prefill_attn.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
OPS = SP / "vllm/models/inkling2/nvidia/ops"
ATTN = OPS / "attention.py"
SM90 = OPS / "attention_sm90.py"

BULK_LOOP = '''    # Key blocks entirely before `pos_lo - E` (relative-bias extent): visible to every row, no bias, in
    # range. RADIANCE_INKLING2_ATTN_BULK: these run unmasked, the last ~E keys take the masked loop below.
    # VEC (RADIANCE_INKLING2_VEC_KV): a block lies in one page, so one scalar page lookup and a 16-byte-aligned
    # tile load replace the per-key gather.
    VEC: tl.constexpr = (
        VEC_KV
        and K.dtype.element_ty.primitive_bitwidth == 8
        and V.dtype.element_ty.primitive_bitwidth == 8
        and PAGE % BLOCK_N == 0
        and D % 16 == 0
    )
    BULK_ON: tl.constexpr = BULK and WINDOW == 0 and not DRAFT and PAGE % BLOCK_N == 0
    num_bulk = num_iters * 0
    if BULK_ON:
        n_bulk = tl.maximum(pos_lo - E + 1, 0) // BLOCK_N
        num_bulk = tl.minimum(tl.maximum(n_bulk - split + num_splits - 1, 0) // num_splits, num_iters)
        for i in range(num_bulk):
            n = split + i * num_splits
            page = tl.load(block_table + n * BLOCK_N // PAGE)
            if VEC:
                kv_base = tl.multiple_of(
                    page.to(tl.int64) * stride_kv_blk
                    + tl.multiple_of(n * BLOCK_N % PAGE, BLOCK_N) * stride_kv_slot
                    + kv_off,
                    16,
                )
                k = _load_fp8_kv_block(K, kv_base, stride_kv_slot, None, False, q.dtype, BLOCK_N, D).T
            else:
                kv_ptrs = (
                    page.to(tl.int64) * stride_kv_blk
                    + (n * BLOCK_N % PAGE + tl.arange(0, BLOCK_N)) * stride_kv_slot
                    + kv_off
                )
                k = _load_fp8_kv_all(K, kv_ptrs, q.dtype, D).T
            s = tl.dot(q, k) * qk_scale
            m_new = tl.maximum(m_i, tl.max(s, 1))
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(s - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, 1)
            v = (
                _load_fp8_kv_block(V, kv_base, stride_kv_slot, None, False, q.dtype, BLOCK_N, D)
                if VEC
                else _load_fp8_kv_all(V, kv_ptrs, q.dtype, D)
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
    for i in range(num_bulk, num_iters):
        n = n_lo + split + i * num_splits
        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
        key_valid = keys < seqlen_k
'''

MASKED_PAGE = '''        if VEC:
            kv_base = tl.multiple_of(
                tl.load(block_table + n * BLOCK_N // PAGE).to(tl.int64) * stride_kv_blk
                + tl.multiple_of(n * BLOCK_N % PAGE, BLOCK_N) * stride_kv_slot
                + kv_off,
                16,
            )
        else:
            page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)
            kv_ptrs = (
                page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off
            )
'''

LOAD_ALL = '''@triton.jit
def _load_fp8_kv_all(ptr, offsets, dtype: tl.constexpr, D: tl.constexpr):
    # unmasked rows of a bulk key block (every key valid): same load and convert as `_load_fp8_kv`
    d = tl.arange(0, D)
    return tl.load(ptr + offsets[:, None] + d[None, :]).to(dtype)


@triton.jit
def _load_fp8_kv_block(
    ptr, base_off, stride_slot, mask, has_mask: tl.constexpr, dtype: tl.constexpr, BLOCK_N: tl.constexpr, D: tl.constexpr
):
    # BLOCK_N consecutive slots of one page: `base_off` is 16-byte aligned and `stride_slot` a multiple of 128,
    # so every row is D contiguous bytes at a 16-byte aligned address (global_load_b128)
    ptr = tl.multiple_of(ptr, 16)
    stride_slot = tl.multiple_of(stride_slot, 128)
    n = tl.arange(0, BLOCK_N)
    d = tl.max_contiguous(tl.multiple_of(tl.arange(0, D), 16), 128)
    p = tl.max_contiguous(tl.multiple_of(ptr + base_off + n[:, None] * stride_slot + d[None, :], (1, 16)), (1, 128))
    if has_mask:
        return tl.load(p, mask=mask[:, None], other=0.0).to(dtype)
    return tl.load(p).to(dtype)


@triton.jit
def _rel_attention_kernel(
'''

KNOBS = '''import os

import torch

from vllm.triton_utils import tl, triton

# RADIANCE_INKLING2_DECODE_SPLITS=0 is the stock decode path: stock plan, tile, masked loop, no vector load.
_RAD_DEC_SPLITS = os.environ.get("RADIANCE_INKLING2_DECODE_SPLITS", "1") != "0"
# RADIANCE_INKLING2_ATTN_BULK: unmasked loop over key blocks older than the relative-bias extent.
_BULK = os.environ.get("RADIANCE_INKLING2_ATTN_BULK", "1") != "0"
# RADIANCE_INKLING2_VEC_KV: one page lookup and a 16-byte-aligned fp8 tile load per key block.
_RAD_VEC_KV = _RAD_DEC_SPLITS and os.environ.get("RADIANCE_INKLING2_VEC_KV", "1") != "0"
# ROCm decode tile (block_n, warps): global layers (window 0) and local sliding-window layers.
_RAD_DEC_BN = int(os.environ.get("RADIANCE_INKLING2_DECODE_BLOCK_N", "32"))
_RAD_DEC_WARPS = int(os.environ.get("RADIANCE_INKLING2_DECODE_WARPS", "2"))
_RAD_LOC_BN = int(os.environ.get("RADIANCE_INKLING2_LOCAL_BLOCK_N", "32"))
_RAD_LOC_WARPS = int(os.environ.get("RADIANCE_INKLING2_LOCAL_WARPS", "4"))
print(
    f"[radiance.inkling2] decode_attn: splits={int(_RAD_DEC_SPLITS)} vec_kv={int(_RAD_VEC_KV)} bulk={int(_BULK)} "
    f"global(block_n={_RAD_DEC_BN},warps={_RAD_DEC_WARPS}) local(block_n={_RAD_LOC_BN},warps={_RAD_LOC_WARPS})",
    flush=True,
)
'''

LAUNCH_TILE = '''    # radiance decode tile (patch_inkling2_decode_attn): prefill (MODE_OUT_LSE) keeps its tile
    block_n = (32 if torch.version.hip else 64) if block_m >= 128 else 128 if block_m <= 32 else 64
    num_warps = 8 if block_m >= 128 else 4
    if torch.version.hip and mode == MODE_PARTIAL and block_m <= 32 and _RAD_DEC_SPLITS:
        block_n, num_warps = (_RAD_DEC_BN, _RAD_DEC_WARPS) if window == 0 else (_RAD_LOC_BN, _RAD_LOC_WARPS)
    # the vector block load needs 16-byte aligned rows: strides and base pointers of the KV cache
    vec = (
        _RAD_VEC_KV
        and mode != MODE_OUT_LSE
        and key_cache.stride(0) % 128 == 0
        and key_cache.stride(1) % 128 == 0
        and key_cache.stride(2) % 128 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
    )
    _rel_attention_kernel[grid](
'''


def main():
    if not SM90.exists():
        print("  NOOP  inkling2 decode attention (models/inkling2 absent)")
        return
    apply(
        ATTN,
        "from dataclasses import dataclass\n",
        "import os\nfrom dataclasses import dataclass\n",
        "import os\nfrom dataclasses import dataclass\n",
        "inkling2 decode splits: import os",
    )
    apply(
        ATTN,
        "from .attention_sm90 import MODE_GATED, MODE_PARTIAL, rel_attention\n",
        "from .attention_sm90 import MODE_GATED, MODE_PARTIAL, _RAD_DEC_SPLITS, _RAD_LOC_BN, rel_attention\n",
        "_RAD_DEC_SPLITS, _RAD_LOC_BN, rel_attention",
        "inkling2 decode splits: import knobs",
    )
    apply(
        ATTN,
        "_VERIFY_STAGES = 3\n",
        "_VERIFY_STAGES = 3\n"
        "# RADIANCE_INKLING2_DECODE_CTAS_PER_SM: gfx12 reports 32 compute units, not 188 SMs; 16 programs per\n"
        "# unit keeps the long-context decode launch bandwidth-bound (0 = the stock plan below).\n"
        "_ROCM_CTAS_PER_SM = (\n"
        "    int(os.environ.get(\"RADIANCE_INKLING2_DECODE_CTAS_PER_SM\", \"16\")) if torch.version.hip and _RAD_DEC_SPLITS else 0\n"
        ")\n",
        "_ROCM_CTAS_PER_SM = (",
        "inkling2 decode splits: knob",
    )
    apply(
        ATTN,
        "    return max(num_sms // work, min(num_sms // 8, 6 * num_sms // work))\n",
        "    if _ROCM_CTAS_PER_SM:\n"
        "        return max(1, _ROCM_CTAS_PER_SM * num_sms // work)\n"
        "    return max(num_sms // work, min(num_sms // 8, 6 * num_sms // work))\n",
        "if _ROCM_CTAS_PER_SM:",
        "inkling2 decode splits: full-layer plan",
    )
    apply(
        ATTN,
        "            return triton.cdiv(rel_extent - 1, _FA4_BLOCK_N) + 1\n",
        "            if torch.version.hip and _RAD_DEC_SPLITS:  # one split per BLOCK_N of the window\n"
        "                return min(33, triton.cdiv(rel_extent - 1, _RAD_LOC_BN) + 1)\n"
        "            return triton.cdiv(rel_extent - 1, _FA4_BLOCK_N) + 1\n",
        "return min(33, triton.cdiv(rel_extent - 1, _RAD_LOC_BN) + 1)",
        "inkling2 decode splits: local-layer plan",
    )
    apply(
        SM90,
        "import torch\n\nfrom vllm.triton_utils import tl, triton\n",
        KNOBS,
        "_RAD_DEC_SPLITS = os.environ.get(",
        "inkling2 attention decode: knobs and proof line",
    )
    apply(
        SM90,
        "@triton.jit\ndef _rel_attention_kernel(\n",
        LOAD_ALL,
        "def _load_fp8_kv_block(",
        "inkling2 attention decode: block loads",
    )
    apply(
        SM90,
        "    BLOCK_N: tl.constexpr,\n):\n    TOKENS: tl.constexpr = BLOCK_M // G\n",
        "    BLOCK_N: tl.constexpr,\n    BULK: tl.constexpr,\n    VEC_KV: tl.constexpr,\n):\n    TOKENS: tl.constexpr = BLOCK_M // G\n",
        "    VEC_KV: tl.constexpr,\n",
        "inkling2 attention decode: signature",
    )
    apply(
        SM90,
        "    for i in range(num_iters):\n        n = n_lo + split + i * num_splits\n"
        "        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)\n        key_valid = keys < seqlen_k\n",
        BULK_LOOP,
        "BULK_ON: tl.constexpr",
        "inkling2 attention decode: bulk loop",
    )
    apply(
        SM90,
        "        page = tl.load(block_table + keys // PAGE, mask=key_valid, other=0)\n"
        "        kv_ptrs = (\n            page.to(tl.int64) * stride_kv_blk + (keys % PAGE) * stride_kv_slot + kv_off\n        )\n",
        MASKED_PAGE,
        "kv_base = tl.multiple_of(\n                tl.load(",
        "inkling2 attention decode: masked loop page",
    )
    for name, var in (("K", "k"), ("V", "v")):
        tail = ".T" if name == "K" else ""
        apply(
            SM90,
            f"        if {name}.dtype.element_ty.primitive_bitwidth == 8:\n"
            f"            {var} = _load_fp8_kv({name}, kv_ptrs, key_valid, q.dtype, D){tail}\n",
            f"        if VEC:\n"
            f"            {var} = _load_fp8_kv_block({name}, kv_base, stride_kv_slot, key_valid, True, q.dtype, BLOCK_N, D){tail}\n"
            f"        elif {name}.dtype.element_ty.primitive_bitwidth == 8:\n"
            f"            {var} = _load_fp8_kv({name}, kv_ptrs, key_valid, q.dtype, D){tail}\n",
            f"{var} = _load_fp8_kv_block({name}, kv_base, stride_kv_slot, key_valid",
            f"inkling2 attention decode: masked loop {name} load",
        )
    apply(
        SM90,
        "    _rel_attention_kernel[grid](\n",
        LAUNCH_TILE,
        "# radiance decode tile (patch_inkling2_decode_attn)",
        "inkling2 attention decode: tile choice",
    )
    apply(
        SM90,
        "        BLOCK_N=(32 if torch.version.hip else 64) if block_m >= 128 else 128 if block_m <= 32 else 64,\n"
        "        num_warps=8 if block_m >= 128 else 4,\n",
        "        BLOCK_N=block_n,\n"
        "        BULK=_RAD_DEC_SPLITS and _BULK and mode != MODE_OUT_LSE,\n"
        "        VEC_KV=vec,\n"
        "        num_warps=num_warps,\n",
        "VEC_KV=vec,",
        "inkling2 attention decode: launch",
    )


if __name__ == "__main__":
    main()
