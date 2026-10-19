#!/usr/bin/env python3
"""UltraQuant continuation-prefill SDPA fallback assumed Hk == 1 (vLLM 0.32.0).

Found live on the first 1.6.0 UltraQuant + DFlash serve (R9700, gfx1201):
a >10k-token prompt is prefilled in max_num_batched_tokens=2048 chunks. The
first chunk runs the dense per-sequence path, which calls SDPA with
`enable_gqa=use_gqa` and is fine. Every following chunk of the same request is
a continuation prefill and lands in `UltraQuantAttention._ultraquant_continuation_prefill`,
which has three backends: flash-attn varlen and the Triton unified kernel both
take q with Hq heads against k/v with Hk heads natively -- but the last fallback
does

    if Hq > Hk:
        k_t = k_t.expand(1, Hq, -1, -1)

and `tensor.expand` can only grow a head dim that is 1. On flash-attn-less RDNA
builds (`_HAS_FLASH_ATTN` False) with Qwen3.8-27B (24 q-heads / 4 kv-heads)
this dies on the second chunk with

    RuntimeError: The expanded size of the tensor (24) must match the existing
    size (4) at non-singleton dimension 1.

killing the engine mid-request (0.31 never had this code: `ultraquant_4bit`
kv dtype is new in 0.32). Fix: SDPA handles grouped-query attention natively
via `enable_gqa=True` -- the exact plumbing the first-chunk call in the same
file already uses and that was verified live on this set-up. It is a no-op
when Hq == Hk, so pass it unconditionally and drop the expand entirely (no
6x materialization of K/V on the workspace buffers).
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / \
    "vllm/v1/attention/backends/ultraquant_attn.py"

apply(F,
       '''        q_t = q_rot.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        if Hq > Hk:
            k_t = k_t.expand(1, Hq, -1, -1)
            v_t = v_t.expand(1, Hq, -1, -1)
        return (
            F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=self.scale
            )''',
       '''        q_t = q_rot.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        # --- RADIANCE FIX (continuation SDPA fallback assumed Hk == 1) ---
        # tensor.expand() only works when Hk == 1; a GQA model (Qwen3.8-27B:
        # 24 q-heads / 4 kv-heads) with a >threshold continuation chunk on a
        # flash-attn-less build died with "expanded size of the tensor (24)
        # must match the existing size (4)", killing the engine mid-request.
        # SDPA supports grouped-query attention natively (enable_gqa), the
        # same plumbing the dense first-chunk SDPA call above uses; it is a
        # no-op when Hq == Hk, so pass it unconditionally and drop the expand
        # (avoids materializing K/V 6x per layer on the workspace buffers).
        return (
            F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                is_causal=True,
                scale=self.scale,
                enable_gqa=Hq > Hk,
            )''',
       'enable_gqa=Hq > Hk',
       'ultraquant continuation SDPA GQA')
