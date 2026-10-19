#!/usr/bin/env python3
"""UltraQuant continuation-prefill flash-attn-less fallback: GQA shape + OOM (vLLM 0.32.0).

Found live on the first 1.6.0 UltraQuant + DFlash serve (R9700, gfx1201):
a >10k-token prompt is prefilled in max_num_batched_tokens=2048 chunks. The
first chunk runs the dense per-sequence path, which calls SDPA with
`enable_gqa=use_gqa` and is fine. Every following chunk of the same request is
a continuation prefill and lands in `UltraQuantAttention._ultraquant_continuation_prefill`,
whose non-sinks / D<=256 tail has two exits: flash-attn varlen (handles 24 q
/ 4 kv heads natively) and a torch-SDPA fallback with

    if Hq > Hk:
        k_t = k_t.expand(1, Hq, -1, -1)

Two live failures on a flash-attn-less RDNA build (Qwen3.8-27B, 24 q-heads /
4 kv-heads, GQA ratio 6):

1. `tensor.expand` can only grow a head dim that is 1 -> the second chunk died
   with "expanded size of the tensor (24) must match the existing size (4)".
2. Using torch SDPA's `enable_gqa` instead fixed the shape but then OOM-killed
   the engine on a long cached_len: with no flash-attn (FA2/FA3 both absent on
   RDNA), torch routes grouped-query SDPA through the *math* backend, which
   materializes the Hq x cache_len score tensor -- 746 MiB for one layer of a
   ~40k-token prompt -- during the FP4-dequant workspace phase, while the KV
   pool holds the GPU at 0 B free (lowering --gpu-memory-utilization does not
   help; the pool is sized to the util in both cases). CPython's SDPA math
   backend is not tileable from outside; the memory is inherent to the backend
   choice, not to the pool size.

Fix: fall back to vLLM's Triton `unified_attention` kernel -- the same kernel
and the same dense call shape the function above already uses for the
sinks / head_dim>256 branch. It handles GQA natively
(`num_queries_per_kv = num_query_heads // num_kv_heads`) with online-softmax
tiles (O(1) score memory per layer), which is also exactly what the plain
`TRITON_ATTN` backend runs for this model's full-attention layers. The
flash-attn varlen branch is kept for flash-attn-present builds (MI3xx), where
FA2 still beats the Triton kernel. The `expand` SDPA fallback is gone entirely:
0.31 never had this code path (ultraquant_4bit is new in 0.32), so there is no
old shape to preserve.

Hunk 2 removes the per-layer worst-case `k_full_buf`/`v_full_buf` concat
buffers underneath it (512 MiB per side per full-attention layer: 16 GiB worst
case, found as the next follow-up OOM live); both concats now come from the
shared workspace manager in the same call as the dequant K/V targets.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / \
    "vllm/v1/attention/backends/ultraquant_attn.py"

apply(F,
       '''        if self.sinks is not None or D > _CK_MAX_HEAD_DIM:
            from vllm.v1.attention.ops.triton_unified_attention import (
                unified_attention,
            )

            out = torch.empty_like(q_rot)
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=device)
            block_table_single = torch.zeros((1, 1), dtype=torch.int32, device=device)
            window = (
                (self.sliding_window - 1, 0)
                if self.sliding_window and self.sliding_window > 0
                else (-1, -1)
            )
            unified_attention(
                q=q_rot,
                k=k_full.unsqueeze(0),
                v=v_full.unsqueeze(0),
                out=out,
                cu_seqlens_q=cu_q,
                max_seqlen_q=q_len,
                seqused_k=seqused_k,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=window,
                block_table=block_table_single,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                sinks=self.sinks,
            )
            return out.to(qdtype)
        if _HAS_FLASH_ATTN:
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            return self._flash_attn_varlen(
                q=q_rot,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )
        q_t = q_rot.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        if Hq > Hk:
            k_t = k_t.expand(1, Hq, -1, -1)
            v_t = v_t.expand(1, Hq, -1, -1)
        return (
            F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=self.scale
            )
            .squeeze(0)
            .transpose(0, 1)
            .to(qdtype)
        )''',
       '''        if self.sinks is not None or D > _CK_MAX_HEAD_DIM:
            from vllm.v1.attention.ops.triton_unified_attention import (
                unified_attention,
            )

            out = torch.empty_like(q_rot)
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=device)
            block_table_single = torch.zeros((1, 1), dtype=torch.int32, device=device)
            window = (
                (self.sliding_window - 1, 0)
                if self.sliding_window and self.sliding_window > 0
                else (-1, -1)
            )
            unified_attention(
                q=q_rot,
                k=k_full.unsqueeze(0),
                v=v_full.unsqueeze(0),
                out=out,
                cu_seqlens_q=cu_q,
                max_seqlen_q=q_len,
                seqused_k=seqused_k,
                max_seqlen_k=seq_len,
                softmax_scale=self.scale,
                causal=True,
                window_size=window,
                block_table=block_table_single,
                softcap=0.0,
                q_descale=None,
                k_descale=None,
                v_descale=None,
                sinks=self.sinks,
            )
            return out.to(qdtype)
        if _HAS_FLASH_ATTN:
            cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
            cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
            return self._flash_attn_varlen(
                q=q_rot,
                k=k_full,
                v=v_full,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=q_len,
                max_seqlen_k=seq_len,
            )
        # --- RADIANCE FIX (continuation SDPA fallback: GQA shape + OOM) ---
        # tensor.expand() could only grow a head dim that is 1 (Qwen3.8-27B:
        # 24 q / 4 kv heads died with "expanded size of the tensor (24) must
        # match the existing size (4)"). SDPA's enable_gqa fixed the shape but
        # then OOM-killed the engine on a long cached_len: with flash-attn
        # absent (RDNA), torch routes grouped-query SDPA through the math
        # backend, which materializes the Hq x cache_len score tensor (746 MiB
        # for one layer of a ~40k-token prompt) on top of the dequant KV
        # workspace with the pool at 0 B free. The Triton unified kernel
        # handles GQA natively with online-softmax tiles (same call shape as
        # the sinks / head_dim>256 branch above, and what the TRITON_ATTN
        # backend runs for these layers anyway), so that is the fallback.
        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention,
        )

        out = torch.empty_like(q_rot)
        cu_q = torch.tensor([0, q_len], dtype=torch.int32, device=device)
        seqused_k = torch.tensor([seq_len], dtype=torch.int32, device=device)
        block_table_single = torch.zeros((1, 1), dtype=torch.int32, device=device)
        window = (
            (self.sliding_window - 1, 0)
            if self.sliding_window and self.sliding_window > 0
            else (-1, -1)
        )
        unified_attention(
            q=q_rot,
            k=k_full.unsqueeze(0),
            v=v_full.unsqueeze(0),
            out=out,
            cu_seqlens_q=cu_q,
            max_seqlen_q=q_len,
            seqused_k=seqused_k,
            max_seqlen_k=seq_len,
            softmax_scale=self.scale,
            causal=True,
            window_size=window,
            block_table=block_table_single,
            softcap=0.0,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            sinks=self.sinks,
        )
        return out.to(qdtype)''',
       'RADIANCE FIX (continuation SDPA fallback: GQA shape + OOM)',
       'ultraquant continuation fallback: Triton unified attention')

# --- hunk 2: the per-layer worst-case concat buffers OOM at request time ---
# "Layer-cached prefix+chunk concatenation, grown once to worst case": cap =
# block_table capacity = 128 x 2048 tokens at the 262k setting -> 512 MiB per
# side, cached on EVERY full-attention layer (16 layers -> 16 GiB worst case),
# with a 512 MiB first-touch alloc at request time while the pool holds the
# GPU at 0 B free. Found live 2026-10-19 in the follow-up crash after hunk 1
# (torch.OutOfMemoryError trying 512 MiB at `k_full_buf = torch.empty(cap,
# Hk, D, ...)`). Upstream already has the right mechanism two lines above:
# the dequant K/V targets come from the *shared* workspace manager. Allocate
# the concats from the same manager, in the same call as the dequant targets
# so the four views are adjacent partitions of one scratch tensor (no
# overlap); the manager keeps the high-water mark across layers and requests,
# so a long context pays once, not per layer -- and only once total, instead
# of 16 GiB of per-layer dead weight. Growth at request time is allowed: the
# workspace is only locked during graph capture, and the dequant call in the
# same function proves request-time growth runs.
apply(F,
       '''        buf_shape = (1, Hk, alloc_len, D)
        k_buf, v_buf = current_workspace_manager().get_simultaneous(
            (buf_shape, qdtype),
            (buf_shape, qdtype),
        )
        ultraquant_full_dequant_kv(
            kv_cache=kv_cache,
            block_table=block_table,
            k_out=k_buf,
            v_out=v_buf,
            alloc_len=alloc_len,
        )

        # Layer-cached prefix+chunk concatenation, grown once to worst case.
        cap = block_table.shape[1] * block_size
        k_full_buf = getattr(layer, "_uq_kfull_buf", None)
        if (
            k_full_buf is None
            or k_full_buf.shape[0] < cap
            or k_full_buf.dtype != qdtype
        ):
            k_full_buf = torch.empty(cap, Hk, D, dtype=qdtype, device=device)
            layer._uq_kfull_buf = k_full_buf
        v_full_buf = getattr(layer, "_uq_vfull_buf", None)
        if (
            v_full_buf is None
            or v_full_buf.shape[0] < cap
            or v_full_buf.dtype != qdtype
        ):
            v_full_buf = torch.empty(cap, Hk, D, dtype=qdtype, device=device)
            layer._uq_vfull_buf = v_full_buf
        k_full = k_full_buf[:seq_len]
        v_full = v_full_buf[:seq_len]''',
       '''        buf_shape = (1, Hk, alloc_len, D)
        # --- RADIANCE FIX (continuation workspace: no per-layer worst-case
        # buffers): the upstream per-layer k_full/v_full buffers were "grown
        # once to worst case" = block_table capacity == 128 x 2048 tokens at
        # the 262k setting -> 512 MiB per side cached on EVERY full-attention
        # layer (16 layers -> 16 GiB), with a 512 MiB first-touch alloc at
        # request time while the pool holds 0 B free. Allocate the concats
        # from the shared workspace manager instead, in the same call as the
        # dequant targets, so the four views are adjacent partitions of one
        # scratch (no overlap); the manager keeps the high-water mark across
        # layers and requests, so a long context pays once in total, not per
        # layer. Request-time growth is allowed: only graph capture locks
        # the workspace, and the dequant call itself already grows it per
        # chunk while serving.
        full_shape = (seq_len, Hk, D)
        k_buf, v_buf, k_full, v_full = \
            current_workspace_manager().get_simultaneous(
                (buf_shape, qdtype),
                (buf_shape, qdtype),
                (full_shape, qdtype),
                (full_shape, qdtype),
            )
        ultraquant_full_dequant_kv(
            kv_cache=kv_cache,
            block_table=block_table,
            k_out=k_buf,
            v_out=v_buf,
            alloc_len=alloc_len,
        )''',
       'RADIANCE FIX (continuation workspace: no per-layer worst-case',
       'ultraquant continuation workspace: shared manager, no 16 GiB layer buffers')
