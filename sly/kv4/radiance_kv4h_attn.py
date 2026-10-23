"""TetraAttention: 4-bit KV attention for RDNA4 (gfx1201). ROCM_AITER_UNIFIED_ATTN's metadata, layout
and graph support over the 4-bit KV cache of radiance_kv4h_format, read by the HIP int4/int8-WMMA
kernel (radiance_kv4h.hip) and written by radiance_kv4h_store. Its working name was KV4H: the
files, the RADIANCE_KV4H_* knobs and the KV4H backend name (an alias) keep it.

Select it with `--attention-backend TETRA --kv-cache-dtype int4_per_token_head` (see
radiance_kv4_attn for why that dtype string). Serves the target's head-256 layers (272-byte cells)
and the DFlash drafter's head-128 sliding-window, non-causal layers (136-byte cells = vLLM's own
int4 cell size, so the pages line up) -- `--speculative-config.attention_backend TETRA`. The query
is rotated and quantized per (token, head) by a Triton kernel (int4 by default, int8 as two int4
planes with RADIANCE_KV4H_QPLANES=2), the output un-rotated inside the segment merge. The kernel extension is compiled by hipcc on first use (install_kv4.py does it at container
start so the first forward does not wait ~20 s).
"""
from dataclasses import replace
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionBackend,
    RocmAiterUnifiedAttentionImpl,
    RocmAiterUnifiedAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.rocm_attn import RocmAttentionMetadata
from vllm.v1.kv_cache_interface import AttentionSpec

import radiance_kv4_format as F
import radiance_kv4h_format as H
from radiance_kv4_attn import KV_DTYPE
from radiance_kv4h_kernel import kv4h_attention
from radiance_kv4h_store import kv4h_store

logger = init_logger("vllm.radiance_kv4h")   # under vllm.* so the engine log shows it

assert 2 * H.layout(256).TPT == F.CONTENT, "KV4H head-256 cells must match KV4's 272 bytes"
HEADS = tuple(H.LAYOUTS)

# RADIANCE_KV4H_TRACE=1 (or the marker file next to this module's source, /patches/sly/kv4/TRACE):
# log every distinct launch shape once (tokens, max q / kv len, split count, graph capture)
import os  # noqa: E402
import sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
TRACE = os.environ.get("RADIANCE_KV4H_TRACE", "") == "1" or _Path("/patches/sly/kv4/TRACE").exists()
_SEEN: set = set()


class TetraAttentionBackend(RocmAiterUnifiedAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [KV_DTYPE]

    @staticmethod
    def get_name() -> str:
        return "TETRA"

    @staticmethod
    def get_impl_cls() -> type["TetraAttentionImpl"]:
        return TetraAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[RocmAiterUnifiedAttentionMetadataBuilder]:
        return RocmAiterUnifiedAttentionMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """One (head, slot) cell: 2 x TPT bytes of the head size's layout (272 for 256, 136 for 128)."""
        spec = super().customize_spec(spec)
        if spec.state_content_bytes is not None:
            return spec
        if spec.head_size not in HEADS:
            raise ValueError(f"TetraAttention supports head sizes {HEADS}, got {spec.head_size}")
        return replace(spec, state_content_bytes=2 * H.layout(spec.head_size, min(H.GROUP, spec.head_size)).TPT)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size in HEADS

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True           # the DFlash drafter's block attention (kernel causal flag)

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER


class TetraAttentionImpl(RocmAiterUnifiedAttentionImpl):
    def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window,
                 kv_cache_dtype, logits_soft_cap=None, attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, sinks=None):
        super().__init__(num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window,
                         kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name,
                         sinks)
        if kv_cache_dtype != KV_DTYPE:
            raise ValueError(f"TetraAttention needs --kv-cache-dtype {KV_DTYPE}, got {kv_cache_dtype}")
        if head_size not in HEADS:
            raise ValueError(f"TetraAttention supports head sizes {HEADS} only")
        if alibi_slopes is not None:
            raise ValueError("TetraAttention: alibi is not implemented")
        if logits_soft_cap is not None and logits_soft_cap > 0:
            raise ValueError("TetraAttention: softcap is not implemented")
        if sinks is not None:
            raise ValueError("TetraAttention: sinks are not implemented")
        if attn_type != AttentionType.DECODER:
            raise ValueError("TetraAttention: decoder attention only")
        self.window = int(sliding_window) if sliding_window is not None and sliding_window > 0 else 0
        self.head = head_size
        # the query is quantized here, per (token, head), after the rotation
        self.supports_quant_query_input = False
        logger.info_once("[tetra] TetraAttention installed: HIP iu4/iu8 WMMA kernel, K group %d, V per token",
                         H.GROUP)
        logger.info("[tetra] layer: head %d, %d-byte cells, window %d, K group %d, query planes %d", head_size,
                    2 * H.layout(head_size, min(H.GROUP, head_size)).TPT, self.window, min(H.GROUP, head_size),
                    int(os.environ.get("RADIANCE_KV4H_QPLANES", "") or 1))

    # -- what vLLM may fuse into this layer ---------------------------------------------------
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return False          # the output is un-rotated after the kernel, so no fused fp8 store

    def fused_rope_kvcache_supported(self):
        return False          # rope in vLLM's generic path, then the store

    def fused_qk_norm_rope_kvcache_supported(self):
        return False

    def fused_qk_norm_mrope_kvcache_supported(self) -> bool:
        return False

    # -- cache write --------------------------------------------------------------------------
    def do_kv_cache_update(self, layer: AttentionLayer, key: torch.Tensor, value: torch.Tensor,
                           kv_cache: torch.Tensor, slot_mapping: torch.Tensor):
        kv4h_store(key, value, kv_cache, slot_mapping)

    # -- attention ----------------------------------------------------------------------------
    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("TetraAttention: fused output quantization is not supported")
        if attn_metadata is None:
            return output.fill_(0)     # profiling run
        n = attn_metadata.num_actual_tokens
        out_rot = torch.empty(n, self.num_heads, self.head, dtype=torch.bfloat16, device=query.device)
        kv4h_attention(
            query[:n], kv_cache, out_rot,
            attn_metadata.query_start_loc, attn_metadata.max_query_len,
            attn_metadata.seq_lens, attn_metadata.max_seq_len,
            self.scale, attn_metadata.block_table,
            causal=attn_metadata.causal,         # False: the DFlash drafter's block attention
            window=self.window,
            output=output[:n],                   # un-rotated in the segment merge
        )
        if TRACE:
            from radiance_kv4h_kernel import LAST_LAUNCH
            cfg = LAST_LAUNCH.get("cfg", {})
            key = (self.head, n, int(attn_metadata.max_query_len), int(attn_metadata.max_seq_len),
                   int(attn_metadata.seq_lens.shape[0]), cfg.get("nseg"), torch.cuda.is_current_stream_capturing())
            if key not in _SEEN:
                _SEEN.add(key)
                # stderr: module loggers outside the vllm.* tree have no handler in the engine log
                sys.stderr.write("[tetra] launch head=%d tokens=%d max_q=%d max_k=%d seqs=%d nseg=%s capturing=%s\n" % key)
                sys.stderr.flush()
        return output


class KV4HAttentionBackend(TetraAttentionBackend):
    """The working name (`--attention-backend KV4H`), kept so existing launch lines keep working."""

    @staticmethod
    def get_name() -> str:
        return "KV4H"


KV4HAttentionImpl = TetraAttentionImpl
