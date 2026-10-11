"""KV4 attention backend: ROCM_AITER_UNIFIED_ATTN's metadata, layout and graph support over a
4-bit KV cache (radiance_kv4_format), read by radiance_kv4_kernel and written by radiance_kv4_store.

Select it with `--attention-backend KV4 --kv-cache-dtype int4_per_token_head`. The dtype string is
vLLM's own: CacheConfig accepts it, the spec gets KVQuantMode.INT4_PER_TOKEN_HEAD and a uint8
cache, kv_cache.py zeroes the checkpoint's static K/V scales and attention.py skips the static
query quant -- all of which is what this backend wants, because its scales live in the cache and
the query is quantized per (token, head) here. The bytes themselves are KV4's, not vLLM's: the
layer's cache is only ever read by this backend. (The DFlash drafter's layers run on TRITON_ATTN
with vLLM's int4 mode; see patch_int4_pth_rocm.py.)

What is inherited on purpose: the builder (query_start_loc, seq_lens, block_table, the
cudagraph-capture metadata), supported layouts (LBHNC), block-size rules, cudagraph support.
What this backend changes:
  * customize_spec: 272 bytes of content per (head, slot) instead of 2 x head_size x fp8.
  * do_kv_cache_update: kv4_store (rotate, quantize, pack).
  * forward: rotate + fp8-quantize the query, kv4_attention, un-rotate the output.
  * refuses what the kernel does not implement: sliding window, alibi, sinks, softcap, non-decoder
    attention, head sizes other than 256, fused fp8 output (the output must be un-rotated first)
    and the fused rope+cache update (rope runs in vLLM's generic path, then the store).

The format is fixed (radiance_kv4_format: K uniform per half, V geo per token); there is no knob.
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
from radiance_kv4_kernel import kv4_attention, quantize_query, unrotate_output
from radiance_kv4_store import kv4_store

logger = init_logger(__name__)

KV_DTYPE = "int4_per_token_head"


class KV4AttentionBackend(RocmAiterUnifiedAttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [KV_DTYPE]

    @staticmethod
    def get_name() -> str:
        return "KV4"

    @staticmethod
    def get_impl_cls() -> type["KV4AttentionImpl"]:
        return KV4AttentionImpl

    @staticmethod
    def get_builder_cls() -> type[RocmAiterUnifiedAttentionMetadataBuilder]:
        return RocmAiterUnifiedAttentionMetadataBuilder

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """One (head, slot) cell is the 272-byte KV4 content, K side then V side."""
        spec = super().customize_spec(spec)
        if spec.state_content_bytes is not None:
            return spec
        if spec.head_size != F.HEAD:
            raise ValueError(f"KV4 supports head_size {F.HEAD} only, got {spec.head_size}")
        return replace(spec, state_content_bytes=F.CONTENT)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size == F.HEAD

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return False

    @classmethod
    def supports_sink(cls) -> bool:
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return False

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

class KV4AttentionImpl(RocmAiterUnifiedAttentionImpl):
    def __init__(self, num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window,
                 kv_cache_dtype, logits_soft_cap=None, attn_type=AttentionType.DECODER,
                 kv_sharing_target_layer_name=None, sinks=None):
        super().__init__(num_heads, head_size, scale, num_kv_heads, alibi_slopes, sliding_window,
                         kv_cache_dtype, logits_soft_cap, attn_type, kv_sharing_target_layer_name,
                         sinks)
        if kv_cache_dtype != KV_DTYPE:
            raise ValueError(f"KV4 needs --kv-cache-dtype {KV_DTYPE}, got {kv_cache_dtype}")
        if head_size != F.HEAD:
            raise ValueError(f"KV4 supports head_size {F.HEAD} only")
        if alibi_slopes is not None or (sliding_window is not None and sliding_window > 0):
            raise ValueError("KV4: alibi and sliding window are not implemented")
        if logits_soft_cap is not None and logits_soft_cap > 0:
            raise ValueError("KV4: softcap is not implemented")
        if sinks is not None:
            raise ValueError("KV4: sinks are not implemented")
        if attn_type != AttentionType.DECODER:
            raise ValueError("KV4: decoder attention only")
        # the query is quantized here, per (token, head), after the rotation
        self.supports_quant_query_input = False
        logger.info_once("[kv4] backend installed: %d-byte cells, K %s / V %s", F.CONTENT, F.CODEBOOK_K, F.CODEBOOK_V)

    # -- what vLLM may fuse into this layer ---------------------------------------------------
    def fused_output_quant_supported(self, quant_key: QuantKey):
        return False          # the output is un-rotated after the kernel, so no fused fp8 store

    def fused_rope_kvcache_supported(self):
        return False          # rope in vLLM's generic path, then kv4_store

    def fused_qk_norm_rope_kvcache_supported(self):
        return False

    def fused_qk_norm_mrope_kvcache_supported(self) -> bool:
        return False

    # -- cache write --------------------------------------------------------------------------
    def do_kv_cache_update(self, layer: AttentionLayer, key: torch.Tensor, value: torch.Tensor,
                           kv_cache: torch.Tensor, slot_mapping: torch.Tensor):
        kv4_store(key, value, kv_cache, slot_mapping)

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
            raise NotImplementedError("KV4: fused output quantization is not supported")
        if attn_metadata is None:
            return output.fill_(0)     # profiling run
        assert attn_metadata.use_cascade is False and attn_metadata.causal
        n = attn_metadata.num_actual_tokens
        q8, q_descale = quantize_query(query[:n])
        out_rot = torch.empty_like(output[:n])
        kv4_attention(
            q8, q_descale, kv_cache, out_rot,
            attn_metadata.query_start_loc, attn_metadata.max_query_len,
            attn_metadata.seq_lens, attn_metadata.max_seq_len,
            self.scale, attn_metadata.block_table,
        )
        output[:n].copy_(unrotate_output(out_rot))
        return output
