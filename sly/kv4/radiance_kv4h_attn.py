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
import os    # noqa: E402  (the module's own env knobs read below, before the imports further down)
import sys   # noqa: E402

# R4D_KV4I=1 routes the PREFILL launches of the head-256 layers to the libr4d int4 kernel
# (r4d_attn_prefill_h256_gqa6_int4kv, sly/r4d/r4d_kv4i.patch): 1.01x of the fp8 libr4d kernel at
# past 49152 against 1.81x for the TetraAttention kernel itself, see docs/R4D-INT4-PREFILL.md.
# Decode, verify, the head-128 drafter and every batch this guard declines stay on TetraAttention.
# ponytail: a whole launch is routed or none of it -- a batch that mixes long prefill runs with decode
# keeps Tetra for all of it, because the kernel takes ONE q_len per launch.
import r4d as _r4d
import kv4i_route as R4
KV4I_ON = os.environ.get("R4D_KV4I", "") == "1"
KV4I_MIN_Q = max(16, int(os.environ.get("RADIANCE_KV4I_MIN_Q", str(R4.MIN_Q_DEFAULT))))
KV4I_MIN_CTX = max(0, int(os.environ.get("RADIANCE_KV4I_MIN_CTX", str(R4.MIN_CTX_DEFAULT))))
# rotate(q) is up to 16x the magnitude of q and e4m3 clips at 448: store rotate(q) / GAIN, the kernel
# multiplies GAIN back in fp32 (its q_scale), so the numbers are the same with 16x the headroom.
KV4I_QGAIN = float(os.environ.get("RADIANCE_KV4I_QGAIN", "16"))
# R4D_KV4I_CHECK=N: the first N routed launches also run the TetraAttention kernel and log the relative
# L2 between the two outputs per layer (host reads: a debug aid for an accuracy hunt, never for serving)
KV4I_CHECK = int(os.environ.get("R4D_KV4I_CHECK", "0") or 0)
_KV4I_CHECKED = [0]
# R4D_KV4I_DUMP=<dir>: the first R4D_KV4I_DUMP_N (1) routed launches of each layer whose name contains one
# of R4D_KV4I_DUMP_LAYERS (comma list, e.g. "layers.47.") are saved as <dir>/<layer>.<i>.pt: query, e4m3
# query, the cache blocks of sequence 0, metadata, routed out_rot and the Tetra reference (debug only).
KV4I_DUMP = os.environ.get("R4D_KV4I_DUMP", "")
KV4I_DUMP_LAYERS = tuple(s for s in os.environ.get("R4D_KV4I_DUMP_LAYERS", "").split(",") if s)
KV4I_DUMP_N = int(os.environ.get("R4D_KV4I_DUMP_N", "1") or 1)
_KV4I_DUMPED: dict = {}
KV4I_GPB, KV4I_BLOCK = R4.KV4_GPB, R4.KV4_BLOCK
KV4I_ENTRY = getattr(_r4d, "attn_prefill_h256_gqa6_int4kv", None)
_KV4I_SAID: set = set()


def _kv4i_say(key: str, msg: str) -> None:
    """Log a distinct route/decline reason once (stderr: module loggers outside vllm.* have no handler)."""
    if key in _KV4I_SAID:
        return
    _KV4I_SAID.add(key)
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

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
        self._kv4i_scratch = None      # 256-byte libr4d scratch, allocated on the first routed launch
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

    # -- the int4 libr4d prefill route (R4D_KV4I=1, opt-in) ------------------------------------
    def _kv4i_decline(self, query, kv_cache, m, output, output_scale, output_block_scale) -> str | None:
        """Why this launch must stay on TetraAttention, or None when the int4 libr4d kernel takes it."""
        # during graph capture a device->host copy is illegal, so decline before reading any tensor
        if not KV4I_ON:
            return "R4D_KV4I unset"
        if torch.cuda.is_current_stream_capturing():
            return "graph capture"
        # Python ints only: this runs in every attention layer of every step, and a device->host read
        # here is a stream sync per layer (equal query lengths <=> tokens == seqs * longest query).
        return R4.decline(
            enabled=KV4I_ON, has_entry=KV4I_ENTRY is not None,
            head=int(self.head), k_group=int(H.GROUP),
            cache_block=int(kv_cache.shape[2]) if kv_cache is not None and kv_cache.dim() == 4 else -1,
            cache_shape_ok=kv_cache is not None and kv_cache.dim() == 4,
            cache_strides_ok=(kv_cache is not None and kv_cache.dim() == 4
                              and kv_cache.stride(3) == 1 and kv_cache.stride(2) == kv_cache.shape[3]),
            window=int(self.window), causal=m.causal,
            fused_out=output_scale is not None or output_block_scale is not None,
            out_dtype=str(output.dtype).replace("torch.", ""),
            capturing=False,   # already declined above, before the host copy
            max_query_len=int(m.max_query_len), max_seq_len=int(m.max_seq_len),
            num_tokens=int(m.num_actual_tokens), num_seqs=int(m.seq_lens.shape[0]),
            min_q=KV4I_MIN_Q, min_ctx=KV4I_MIN_CTX,
        )

    def _kv4i_block_table(self, m) -> torch.Tensor:
        """vLLM block ids -> libr4d's 16-token group ids (group G = block G // 104, group G % 104).

        Padded by 3 entries: the kernel's double-buffered prefetch reads up to three groups past the
        end of a row (the same padding the bench arm uses).
        """
        ids = m.block_table.to(torch.int32) * KV4I_GPB
        sub = torch.arange(KV4I_GPB, dtype=torch.int32, device=ids.device)
        table = (ids.unsqueeze(-1) + sub).reshape(ids.shape[0], ids.shape[1] * KV4I_GPB).contiguous()
        return torch.nn.functional.pad(table, (0, 3))

    def _kv4i_launch(self, layer, query, kv_cache, m, output):
        n = int(m.num_actual_tokens)
        # TetraAttention stores K/V rotated and quantised, so the query is rotated first, scaled and
        # handed over as e4m3 (the kernel's O_QF8 path), exactly as the bench arm does.
        q_scale = float(getattr(layer, "_q_scale_float", 1.0) or 1.0) * KV4I_QGAIN
        q = (F.rotate(query[:n].float()) / q_scale).to(torch.float8_e4m3fn)
        _r4d.set_q_fp8(q_scale)
        table = self._kv4i_block_table(m)
        out_rot = torch.empty(n, self.num_heads, self.head, dtype=torch.bfloat16, device=query.device)
        if self._kv4i_scratch is None:
            self._kv4i_scratch = torch.empty(256, dtype=torch.uint8, device=query.device)
        KV4I_ENTRY(
            q.data_ptr(), kv_cache.data_ptr(), table.data_ptr(), m.seq_lens.data_ptr(), out_rot.data_ptr(),
            0, 0, self._kv4i_scratch.data_ptr(),
            int(table.shape[0]), int(m.max_query_len), self.num_heads, self.num_kv_heads, self.head,
            16, int(table.shape[1]), kv_cache.stride(0), kv_cache.stride(1),
            self.scale / self.head,   # K and q are both rotated (x16 each): fold it, as kv4h_attention does
            0, int(m.max_seq_len), torch.cuda.current_stream().cuda_stream, H.GROUP,
        )
        # un-rotation is a 256x256 matmul per launch: ponytail ceiling, it is not in the kernel
        output[:n].copy_(F.unrotate(out_rot.float()))
        name = getattr(layer, "layer_name", "?")
        dump = (KV4I_DUMP and any(s in name for s in KV4I_DUMP_LAYERS)
                and _KV4I_DUMPED.get(name, 0) < KV4I_DUMP_N)
        check = _KV4I_CHECKED[0] < KV4I_CHECK
        ref = None
        if dump or check:
            ref = torch.empty(n, self.num_heads, self.head, dtype=torch.bfloat16, device=query.device)
            kv4h_attention(query[:n], kv_cache, torch.empty_like(ref), m.query_start_loc, m.max_query_len,
                           m.seq_lens, m.max_seq_len, self.scale, m.block_table, causal=True, window=0, output=ref)
        if dump:
            i = _KV4I_DUMPED[name] = _KV4I_DUMPED.get(name, 0) + 1
            nblk = -(-int(m.seq_lens[0]) // int(kv_cache.shape[2]))
            torch.save(dict(
                query=query[:n].cpu(), q8=q.cpu(), q_scale=q_scale, scale=float(self.scale),
                kv=kv_cache[m.block_table[0, :nblk].long()].cpu(), block_table=m.block_table.cpu(),
                seq_lens=m.seq_lens.cpu(), query_start_loc=m.query_start_loc.cpu(),
                num_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
                out_rot=out_rot.cpu(), out=output[:n].cpu(), ref=ref.cpu()),
                os.path.join(KV4I_DUMP, f"{name}.{i - 1}.pt"))
            sys.stderr.write(f"[kv4i-dump] {name}.{i - 1}.pt n={n} ctx={int(m.max_seq_len)}\n")
        if check:
            _KV4I_CHECKED[0] += 1
            d = output[:n].float() - ref.float()
            sys.stderr.write("[kv4i-check] %s n=%d ctx=%d rel_l2=%.4f max_abs=%.3f ref_absmax=%.3f qrot_amax=%.1f\n" % (
                getattr(layer, "layer_name", "?"), n, int(m.max_seq_len), float(d.norm() / ref.float().norm()),
                float(d.abs().max()), float(ref.float().abs().max()), float(F.rotate(query[:n].float()).abs().max())))
            sys.stderr.flush()
        _kv4i_say("route", f"[kv4i] prefill runs >= {KV4I_MIN_Q} tokens at context >= {KV4I_MIN_CTX} go to "
                           f"libr4d int4 (head 256, K group 256, block {KV4I_BLOCK}); decode, verify and the "
                           f"drafter stay on TETRA")
        return output

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
        if KV4I_ON:
            why = self._kv4i_decline(query, kv_cache, attn_metadata, output, output_scale, output_block_scale)
            if why is None:
                return self._kv4i_launch(layer, query, kv_cache, attn_metadata, output)
            _kv4i_say(why, f"[kv4i] TETRA, not libr4d int4: {why}")
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
