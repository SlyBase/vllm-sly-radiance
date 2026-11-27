#!/usr/bin/env python3
"""Inkling 2 decode/verify attention on gfx12: fill the card and skip dead work at long context.

Root cause (rocprofv3 + sly/inkling2/bench_decode_attn.py, R9700): the global (full-attention) layers run
8x per decode step (layers 5 and 23 once, 11 and 17 over 3 loop passes). `plan_decode_attention` sizes the
split-KV grid as `num_sms // 8` per KV head, tuned for 188 NVIDIA SMs. `torch` reports 32 compute units for
the R9700 (WGPs), so a 1-sequence verify step launched 4 splits x 8 KV heads = 32 programs of 4 waves, each
walking ~37k keys serially at 150k context: 6.7 ms per call, 46 GB/s of ~640. 8 calls = 53 ms of the 73 ms
step, and the whole reason step time grows with context (26 ms @ 8k -> 73 ms @ 150k).

Two edits, both in the installed vLLM `models/inkling2/nvidia/ops`:

  1. attention.py     `plan_decode_attention` (full layers, ROCm only): `CTAS_PER_SM * num_sms // work`
                      splits instead of `num_sms // 8` -- 32 splits for one sequence, so 256 programs.
                      Knob RADIANCE_INKLING2_DECODE_CTAS_PER_SM (default 8, 0 = stock plan). The count still
                      depends only on the batch, never on KV lengths.
  2. attention_sm90.py `_rel_attention_kernel`: a full layer's relative bias only covers the last E keys.
                      Every key block that lies entirely before `pos_lo - E` is fully visible, unbiased and
                      in range for every row of the tile, so a first loop runs those blocks without the
                      per-key masks, the distance matrix and the bias gather, with one page lookup per block
                      (PAGE % BLOCK_N == 0). Only the last ~E keys take the original masked loop. Decode
                      launches only (MODE_GATED / MODE_PARTIAL): prefill (MODE_OUT_LSE) is untouched.
                      Same arithmetic per key, so results match the masked loop bit for bit within one split.
                      Knob RADIANCE_INKLING2_ATTN_BULK (default 1, 0 = masked loop only).

Idempotent (sentinel per hunk). A vLLM without models/inkling2 is a NOOP. Runs after patch_inkling2_gfx12
(whose `_load_fp8_kv` and `num_stages` text the second hunk anchors on).
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
    BULK_ON: tl.constexpr = BULK and WINDOW == 0 and not DRAFT and PAGE % BLOCK_N == 0
    num_bulk = num_iters * 0
    if BULK_ON:
        n_bulk = tl.maximum(pos_lo - E + 1, 0) // BLOCK_N
        num_bulk = tl.minimum(tl.maximum(n_bulk - split + num_splits - 1, 0) // num_splits, num_iters)
        for i in range(num_bulk):
            n = split + i * num_splits
            page = tl.load(block_table + n * BLOCK_N // PAGE)
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
            v = _load_fp8_kv_all(V, kv_ptrs, q.dtype, D)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
    for i in range(num_bulk, num_iters):
        n = n_lo + split + i * num_splits
        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)
        key_valid = keys < seqlen_k
'''

LOAD_ALL = '''@triton.jit
def _load_fp8_kv_all(ptr, offsets, dtype: tl.constexpr, D: tl.constexpr):
    # unmasked rows of a bulk key block (every key valid): same load and convert as `_load_fp8_kv`
    d = tl.arange(0, D)
    return tl.load(ptr + offsets[:, None] + d[None, :]).to(dtype)


@triton.jit
def _rel_attention_kernel(
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
        "_VERIFY_STAGES = 3\n",
        "_VERIFY_STAGES = 3\n"
        "# RADIANCE_INKLING2_DECODE_CTAS_PER_SM: gfx12 reports 32 compute units, not 188 SMs; 16 programs per\n"
        "# unit keeps the long-context decode launch bandwidth-bound (0 = the stock plan below).\n"
        "_ROCM_CTAS_PER_SM = (\n"
        "    int(os.environ.get(\"RADIANCE_INKLING2_DECODE_CTAS_PER_SM\", \"16\")) if torch.version.hip else 0\n"
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
        SM90,
        "import torch\n\nfrom vllm.triton_utils import tl, triton\n",
        "import os\n\nimport torch\n\nfrom vllm.triton_utils import tl, triton\n\n"
        "# RADIANCE_INKLING2_ATTN_BULK: unmasked loop over key blocks older than the relative-bias extent.\n"
        "_BULK = os.environ.get(\"RADIANCE_INKLING2_ATTN_BULK\", \"1\") != \"0\"\n",
        "_BULK = os.environ.get(",
        "inkling2 attention bulk: knob",
    )
    apply(
        SM90,
        "@triton.jit\ndef _rel_attention_kernel(\n",
        LOAD_ALL,
        "def _load_fp8_kv_all(",
        "inkling2 attention bulk: unmasked load",
    )
    apply(
        SM90,
        "    BLOCK_N: tl.constexpr,\n):\n    TOKENS: tl.constexpr = BLOCK_M // G\n",
        "    BLOCK_N: tl.constexpr,\n    BULK: tl.constexpr,\n):\n    TOKENS: tl.constexpr = BLOCK_M // G\n",
        "    BULK: tl.constexpr,\n",
        "inkling2 attention bulk: signature",
    )
    apply(
        SM90,
        "    for i in range(num_iters):\n        n = n_lo + split + i * num_splits\n"
        "        keys = n * BLOCK_N + tl.arange(0, BLOCK_N)\n        key_valid = keys < seqlen_k\n",
        BULK_LOOP,
        "BULK_ON: tl.constexpr",
        "inkling2 attention bulk: loop",
    )
    apply(
        SM90,
        "        BLOCK_N=(32 if torch.version.hip else 64) if block_m >= 128 else 128 if block_m <= 32 else 64,\n",
        "        BLOCK_N=64\n"
        "        if _BULK and torch.version.hip and mode != MODE_OUT_LSE and window == 0 and block_m <= 32\n"
        "        else (32 if torch.version.hip else 64) if block_m >= 128 else 128 if block_m <= 32 else 64,\n"
        "        BULK=_BULK and mode != MODE_OUT_LSE,\n",
        "BULK=_BULK and mode",
        "inkling2 attention bulk: launch (BLOCK_N 64 on full decode layers, bulk flag)",
    )


if __name__ == "__main__":
    main()
