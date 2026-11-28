#!/usr/bin/env python3
"""Inkling 2 prefill attention on gfx12: run the unmasked bulk loop in the prefill launch too.

`sly/patch_inkling2_decode_attn.py` made key blocks that lie entirely before `pos_lo - E` (E = relative-bias
extent, 1024) run without per-key masks, distance matrix and bias gather, for decode launches
(MODE_GATED / MODE_PARTIAL). The prefill launch (MODE_OUT_LSE, 128-row tiles of 42 tokens x 3 GQA heads) kept the
masked loop over every key block. rocprofv3 of a cold 150k prompt (R9700, 210 W): `_rel_attention_kernel` is 48.6 s
of 69.8 s GPU time (70 %), 12.2 of 27.3 s at 64k; almost all of its key blocks are such bulk blocks.

  * `RADIANCE_INKLING2_PREFILL_BULK` (default 1, 0 = masked loop): BULK=True for MODE_OUT_LSE launches of full
    layers. The decode knob `RADIANCE_INKLING2_ATTN_BULK=0` still turns every bulk loop off.
  * `RADIANCE_INKLING2_PREFILL_BLOCK_N` (default 16; stock ROCm 32), `RADIANCE_INKLING2_PREFILL_BLOCK_M` (0 = stock 128
    rows) and `RADIANCE_INKLING2_PREFILL_WARPS` (default 4; 0 = stock 8): tile of that launch.
  * `RADIANCE_INKLING2_PREFILL_LAZY` (default 0): conditional rescale of the running output in the bulk loop.

Idempotent (sentinel per hunk). A vLLM without models/inkling2 is a NOOP. Runs after patch_inkling2_decode_attn.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
SM90 = SP / "vllm/models/inkling2/nvidia/ops/attention_sm90.py"

OLD_BM = "    block_m = _block_m(max_query_len, group)\n"
NEW_BM = (
    "    block_m = _block_m(max_query_len, group)\n"
    "    prefill_bulk = (\n"
    "        _BULK and _PREFILL_BULK and torch.version.hip and mode == MODE_OUT_LSE and window == 0 and block_m >= 128\n"
    "    )\n"
    "    if prefill_bulk and _PREFILL_BLOCK_M:\n"
    "        block_m = _PREFILL_BLOCK_M\n"
)
OLD_LAUNCH = (
    "        BLOCK_N=block_n,\n"
    "        BULK=_RAD_DEC_SPLITS and _BULK and mode != MODE_OUT_LSE,\n"
    "        VEC_KV=vec,\n"
    "        num_warps=num_warps,\n"
)
NEW_LAUNCH = (
    "        BLOCK_N=_PREFILL_BLOCK_N if prefill_bulk else block_n,\n"
    "        BULK=_BULK and (_PREFILL_BULK if mode == MODE_OUT_LSE else _RAD_DEC_SPLITS),\n"
    "        VEC_KV=vec,\n"
    "        LAZY=_PREFILL_LAZY and bool(prefill_bulk),\n"
    "        num_warps=_PREFILL_WARPS if prefill_bulk and _PREFILL_WARPS else num_warps,\n"
)

OLD_SIG = "    BULK: tl.constexpr,\n    VEC_KV: tl.constexpr,\n):\n"
NEW_SIG = "    BULK: tl.constexpr,\n    VEC_KV: tl.constexpr,\n    LAZY: tl.constexpr,\n):\n"

V_EXPR = (
    "(\n"
    "                    _load_fp8_kv_block(V, kv_base, stride_kv_slot, None, False, q.dtype, BLOCK_N, D)\n"
    "                    if VEC\n"
    "                    else _load_fp8_kv_all(V, kv_ptrs, q.dtype, D)\n"
    "                )"
)
OLD_STEP = (
    "            m_new = tl.maximum(m_i, tl.max(s, 1))\n"
    "            alpha = tl.exp2(m_i - m_new)\n"
    "            p = tl.exp2(s - m_new[:, None])\n"
    "            l_i = l_i * alpha + tl.sum(p, 1)\n"
    "            v = (\n"
    "                _load_fp8_kv_block(V, kv_base, stride_kv_slot, None, False, q.dtype, BLOCK_N, D)\n"
    "                if VEC\n"
    "                else _load_fp8_kv_all(V, kv_ptrs, q.dtype, D)\n"
    "            )\n"
    "            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)\n"
    "            m_i = m_new\n"
)
NEW_STEP = (
    "            if LAZY:\n"
    "                # rescale only when a row's max rose by more than 8 (log2): p <= 256, softmax unchanged\n"
    "                m_blk = tl.max(s, 1)\n"
    "                if tl.max(m_blk - m_i, 0) > 8.0:\n"
    "                    m_new = tl.maximum(m_i, m_blk)\n"
    "                    alpha = tl.exp2(m_i - m_new)\n"
    "                    l_i = l_i * alpha\n"
    "                    acc = acc * alpha[:, None]\n"
    "                    m_i = m_new\n"
    "                p = tl.exp2(s - m_i[:, None])\n"
    "                l_i = l_i + tl.sum(p, 1)\n"
    f"                v = {V_EXPR}\n"
    "                acc = acc + tl.dot(p.to(v.dtype), v)\n"
    "            else:\n"
    "                m_new = tl.maximum(m_i, tl.max(s, 1))\n"
    "                alpha = tl.exp2(m_i - m_new)\n"
    "                p = tl.exp2(s - m_new[:, None])\n"
    "                l_i = l_i * alpha + tl.sum(p, 1)\n"
    f"                v = {V_EXPR}\n"
    "                acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)\n"
    "                m_i = m_new\n"
)


def main():
    if not SM90.exists():
        print("  NOOP  inkling2 prefill attention (models/inkling2 absent)")
        return
    apply(
        SM90,
        '_BULK = os.environ.get("RADIANCE_INKLING2_ATTN_BULK", "1") != "0"\n',
        '_BULK = os.environ.get("RADIANCE_INKLING2_ATTN_BULK", "1") != "0"\n'
        "# RADIANCE_INKLING2_PREFILL_BULK: the same unmasked loop in the prefill launch (MODE_OUT_LSE) of full layers;\n"
        "# RADIANCE_INKLING2_PREFILL_BLOCK_N / _BLOCK_M / _WARPS: tile of that launch (defaults 16 / 128 / 4; stock ROCm 32 / 128 / 8).\n"
        '_PREFILL_BULK = os.environ.get("RADIANCE_INKLING2_PREFILL_BULK", "1") != "0"\n'
        '_PREFILL_BLOCK_N = int(os.environ.get("RADIANCE_INKLING2_PREFILL_BLOCK_N", "16"))\n'
        '_PREFILL_BLOCK_M = int(os.environ.get("RADIANCE_INKLING2_PREFILL_BLOCK_M", "0"))\n'
        '_PREFILL_WARPS = int(os.environ.get("RADIANCE_INKLING2_PREFILL_WARPS", "4"))\n'
        '_PREFILL_LAZY = os.environ.get("RADIANCE_INKLING2_PREFILL_LAZY", "0") == "1"\n',
        "_PREFILL_BULK = os.environ.get(",
        "inkling2 prefill attention: knobs",
    )
    apply(SM90, OLD_SIG, NEW_SIG, "    LAZY: tl.constexpr,\n", "inkling2 prefill attention: signature")
    apply(SM90, OLD_STEP, NEW_STEP, "            if LAZY:\n", "inkling2 prefill attention: lazy rescale")
    apply(SM90, OLD_BM, NEW_BM, "    prefill_bulk = (\n", "inkling2 prefill attention: tile gate")
    apply(SM90, OLD_LAUNCH, NEW_LAUNCH, "LAZY=_PREFILL_LAZY and bool(prefill_bulk)", "inkling2 prefill attention: launch")


if __name__ == "__main__":
    main()
