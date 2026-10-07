"""R4D_HYBRID attention: libr4d's paged PREFILL kernel for long prefill runs, ROCM_AITER_UNIFIED_ATTN
for everything else.

Select it with `--attention-backend R4D_HYBRID` (the drafter keeps its own backend through
`--speculative-config.attention_backend TRITON_ATTN`, exactly as with ROCM_AITER_UNIFIED_ATTN).

WHY A HYBRID. Measured at 210 W on the production launch, `--attention-backend R4D` against
ROCM_AITER_UNIFIED_ATTN: prefill +19 % at 64K and +9 % at 32K (the AITER 2D kernel is 49 % of the
prefill GPU time there), but greedy decode 60.9 ms/step against 34.9. The R4D backend subclasses the
TRITON backend, so it brings its own metadata builder, its own cudagraph class, a 16-token kernel block
and bf16 queries -- and it does not carry this image's verify-batch decode tune
(sly/radiance_attn_decode.py), which only applies to fp8 q + fp8 KV launches of the AITER kernel.
This backend does the opposite: it IS the AITER backend (class, builder, KV layout, preferred block
size, cudagraph support, fp8 query quantisation, fused rope+KV-cache update, decode tune) and only the
forward() of a step that contains a long prefill run is touched.

ROUTING (decided per step on the CPU from query_start_loc_cpu, never from device memory):
  * the step has no request with q_len >= RADIANCE_R4D_PREFILL_MIN_Q -- every decode / verify batch,
    every short prefill: forward() is `super().forward()`, byte for byte what ROCM_AITER_UNIFIED_ATTN
    runs. This includes every cudagraph-captured launch (capture batches are uniform decode shapes,
    and the builder additionally forces the plan off while capturing).
  * otherwise the batch is cut into maximal runs. A run of equal-length requests with q_len >= MIN_Q goes
    to libr4d's paged prefill kernel (one launch per run); the remaining requests (decodes riding along
    with a prefill chunk, a short tail chunk) go to the AITER kernel as ONE sub-batch.

KV LAYOUT. One layout, the one ROCM_AITER_UNIFIED_ATTN already has: per layer a (block, kv head, slot,
2*head_dim) view, K and V packed per slot, with the kernel block equal to the manager block (896 tokens
here, not 16). libr4d only understands 16-slot blocks, but it takes the block and head STRIDES as
arguments, so no second layout is needed: block b of the cache is read as `ratio = N/16` sub-blocks, the
sub-block (b, j) lives at element b*S0 + j*16*C, so with block_stride := 16*C the kernel's block id for it
is b*(S0/(16*C)) + j. The step builds that expanded 16-slot block table once (a handful of tiny device ops,
only on steps that have a long run) and shares it across the 16 attention layers.

QUERY. The attention layer quantises q to fp8 (static scale) before the op for every impl that sets
supports_quant_query_input, and the AITER decode path needs that. libr4d has no 8-bit query variant, so
for the R4D runs q is widened back to bf16 (exact: e4m3 values are bf16-representable, times the static
q scale). It carries the same fp8 rounding the AITER kernel has anyway; the R4D kernel then runs its
f16 legs (R4D_ATTN_FP8=0), i.e. is more accurate than the kernel it replaces for the rest of the chain.

Anything libr4d cannot take falls back to the AITER path on the whole step: no libr4d, geometry other
than head_dim 256 / GQA 6, sliding window, alibi, sinks, logits soft cap, non-causal, a fused output
quantisation (`output_scale`), a cache whose strides do not admit the sub-block trick, graph capture.
The first fallback reason is logged once.

Knobs (read at import):
  RADIANCE_R4D_HYBRID_ROUTE       1 (default) | 0 = never route to libr4d: the backend is then exactly
                                  ROCM_AITER_UNIFIED_ATTN under another name (A/B control of the class)
  RADIANCE_R4D_PREFILL_MIN_Q      smallest per-request q_len that goes to libr4d (default 512). The R4D
                                  prefill kernel owns 64 query rows per workgroup, so one 256-token request
                                  fills 16 workgroups of a 64-CU part; equal-length requests batch into one
                                  launch, a lone 512-token chunk is the practical floor.
  RADIANCE_USE_R4D                0 = libr4d off (shared with the R4D backend and the GDN hooks); the hybrid
                                  then refuses to load like `--attention-backend R4D` does
  R4D_ATTN_FP8                    0 (default) | 1 | 2 | 3: libr4d's own opt-in 8-bit legs of the prefill kernel
                                  (QK8 / PV8 / both). Orthogonal; see NOTES-H.md for what it costs in accuracy.
"""

import dataclasses
import os
import sys

import torch

from vllm.v1.attention.backends.rocm_aiter_unified_attn import (
    RocmAiterUnifiedAttentionBackend,
    RocmAiterUnifiedAttentionImpl,
    RocmAiterUnifiedAttentionMetadataBuilder,
)

# radiance_r4d_attn owns the RTLD_DEEPBIND import of libr4d and the registry lookup of the kernels, so
# the hybrid binds exactly the entry points the R4D backend would and cannot drift from it.
import radiance_r4d_attn as _r4d_attn

r4d = _r4d_attn.r4d
BLOCK_SIZE = _r4d_attn.BLOCK_SIZE          # 16: the only block libr4d knows
HEAD_DIM = _r4d_attn.HEAD_DIM
GQA = _r4d_attn.GQA
_PREFILL = _r4d_attn._PREFILL              # (fp8 KV, bf16 KV)
_KV_DTYPE_NAMES = _r4d_attn._KV_DTYPE_NAMES
USE_R4D = _r4d_attn.USE_R4D

ROUTE = os.environ.get("RADIANCE_R4D_HYBRID_ROUTE", "1") != "0"
MIN_Q = max(16, int(os.environ.get("RADIANCE_R4D_PREFILL_MIN_Q", "512")))


def _say(msg: str) -> None:
    sys.stderr.write(f"[radiance] {msg}\n")


_logged: set = set()


def _say_once(key: str, msg: str) -> None:
    if key not in _logged:
        _logged.add(key)
        _say(msg)


def _plan(query_start_loc_cpu: torch.Tensor, num_reqs: int, min_q: int):
    """Cut the batch into runs; None when no request is long enough for libr4d.

    One tuple per run:
      ("r4d",   first request, request count, query length, first token)   equal-length long requests
      ("aiter", first request, request count, first token, token count, longest query)
    Requests with no tokens (padding of a graph capture) join whatever short run they border and are
    dropped from a long one.
    """
    qs = query_start_loc_cpu.tolist()
    runs = []
    short = None  # [first req, req count, first tok, max q] of the open AITER run
    i = 0
    any_long = False
    while i < num_reqs:
        n = qs[i + 1] - qs[i]
        if n >= min_q:
            j = i + 1
            while j < num_reqs and qs[j + 1] - qs[j] == n:
                j += 1
            if short is not None:
                runs.append(("aiter", short[0], short[1], short[2], qs[i] - short[2], short[3]))
                short = None
            runs.append(("r4d", i, j - i, n, qs[i]))
            any_long = True
            i = j
        else:
            if short is None:
                short = [i, 0, qs[i], 0]
            short[1] += 1
            short[3] = max(short[3], n)
            i += 1
    if not any_long:
        return None
    if short is not None and short[3] > 0:
        runs.append(("aiter", short[0], short[1], short[2], qs[num_reqs] - short[2], short[3]))
    return tuple(runs)


class R4DHybridMetadataBuilder(RocmAiterUnifiedAttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        m = super().build(common_prefix_len, common_attn_metadata, fast_build)
        m.hyb_runs = None
        m.hyb_cache = None   # the expanded 16-slot block table, built by the first layer that needs it
        m.hyb_sub = {}       # run index -> metadata of an AITER sub-batch, built by the first layer
        # max_query_len is a host int: a pure decode / verify step never reaches the tolist() below.
        if ROUTE and common_attn_metadata.max_query_len >= MIN_Q:
            m.hyb_runs = _plan(
                common_attn_metadata.query_start_loc_cpu, common_attn_metadata.num_reqs, MIN_Q
            )
        return m

    def build_for_cudagraph_capture(self, common_attn_metadata):
        m = super().build_for_cudagraph_capture(common_attn_metadata)
        m.hyb_runs = None  # a captured launch bakes its pointers: only the AITER path may be captured
        return m


class R4DHybridAttentionBackend(RocmAiterUnifiedAttentionBackend):
    # Everything else -- block size preference, kernel block sizes, KV layouts, cudagraph support,
    # supports_* -- is inherited on purpose: the KV cache is the one ROCM_AITER_UNIFIED_ATTN allocates.

    @staticmethod
    def get_name() -> str:
        return "R4D_HYBRID"

    @staticmethod
    def get_impl_cls() -> type["R4DHybridAttentionImpl"]:
        return R4DHybridAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[R4DHybridMetadataBuilder]:
        return R4DHybridMetadataBuilder

    @classmethod
    def supports_combination(cls, *args, **kwargs) -> str | None:
        if r4d is None:
            return f"R4D kernels are not built into this image ({_r4d_attn._IMPORT_ERROR})"
        if not USE_R4D:
            return "RADIANCE_USE_R4D=0 turns the R4D kernel library off; drop --attention-backend R4D_HYBRID too"
        return None


class R4DHybridAttentionImpl(RocmAiterUnifiedAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if r4d is None:
            raise RuntimeError(f"R4D kernels are not built into this image ({_r4d_attn._IMPORT_ERROR})")
        if not USE_R4D:
            raise RuntimeError("RADIANCE_USE_R4D=0 turns the R4D kernel library off; drop R4D_HYBRID too")
        # Why this layer can never use libr4d (None = it can). Decided once; the layer then runs the
        # AITER path unconditionally and the hybrid is exactly ROCM_AITER_UNIFIED_ATTN for it.
        self._why_not = None
        geometry = dict(
            head_dim=self.head_size,
            gqa=self.num_queries_per_kv,
            block_size=BLOCK_SIZE,
            causal=1,
            q_dtype="bf16",
            kv_dtype=_KV_DTYPE_NAMES.get(self.kv_cache_dtype, self.kv_cache_dtype),
        )
        reasons = []
        if r4d.select("attn_prefill_paged", **geometry) is None:
            reasons.append("no libr4d prefill kernel for this geometry")
        if getattr(self, "attn_type", None) != _r4d_attn.AttentionType.DECODER:
            reasons.append("not decoder attention")
        if self.sliding_window != (-1, -1):
            reasons.append("sliding window")
        if self.alibi_slopes is not None:
            reasons.append("alibi slopes")
        if self.logits_soft_cap:
            reasons.append("logits soft cap")
        if self.sinks is not None:
            reasons.append("attention sinks")
        if reasons:
            self._why_not = ", ".join(reasons)
            _say_once(f"geo:{self._why_not}", f"R4D_HYBRID: a layer runs AITER only ({self._why_not})")
        from vllm.config import get_current_vllm_config

        self._descale_len = get_current_vllm_config().scheduler_config.max_num_seqs * self.num_kv_heads
        self._geom = None       # (variant, S0 in units of 16*C, head stride) once checked
        self._descales = None
        self._dummy = None      # the prefill kernel never touches the split-KV scratch, but the ABI has it

    # -- helpers -----------------------------------------------------------------------------
    def _geometry(self, kv_cache: torch.Tensor):
        """(variant, block-id multiplier, head stride) for this layer's cache, or a reason string."""
        if self._geom is None:
            if kv_cache.dim() != 4:
                self._geom = "cache is not a (block, head, slot, content) view"
            else:
                _, heads, n, c = kv_cache.shape
                unit = BLOCK_SIZE * c
                if c != 2 * self.head_size or kv_cache.stride(3) != 1 or kv_cache.stride(2) != c:
                    self._geom = f"cache strides {tuple(kv_cache.stride())} are not K/V-packed slots"
                elif n % BLOCK_SIZE != 0:
                    self._geom = f"cache block of {n} slots is not a multiple of {BLOCK_SIZE}"
                elif kv_cache.stride(0) % unit != 0:
                    self._geom = "block stride is not a multiple of one 16-slot sub-block"
                else:
                    variant = 1 if kv_cache.element_size() == 2 else 0
                    self._geom = (variant, kv_cache.stride(0) // unit, kv_cache.stride(1), n // BLOCK_SIZE)
        return self._geom

    def _descale_ptrs(self, layer, device):
        ks = float(getattr(layer, "_k_scale_float", 1.0))
        vs = float(getattr(layer, "_v_scale_float", 1.0))
        if ks == 1.0 and vs == 1.0:
            return 0, 0
        if self._descales is None:
            n = self._descale_len
            self._descales = (
                torch.empty(n, dtype=torch.float32, device=device),
                torch.empty(n, dtype=torch.float32, device=device),
                None,
            )
        kbuf, vbuf, cached = self._descales
        if cached != (ks, vs):
            kbuf.fill_(ks)
            vbuf.fill_(vs)
            self._descales = (kbuf, vbuf, (ks, vs))
        return kbuf.data_ptr(), vbuf.data_ptr()

    @staticmethod
    def _block_table16(m, runs, mult: int, ratio: int):
        """The 16-slot block table of the requests the R4D runs cover, built once per step.

        Row r (r = request - lo) lists, for each 16-slot sub-block of the request, libr4d's block id:
        manager block b, sub-block j -> b * mult + j, with block_stride = 16 * C elements. Returns
        (table, lo).
        """
        key = (mult, ratio)
        cache = m.hyb_cache
        if cache is not None and cache[0] == key:
            return cache[1], cache[2]
        lo = min(r[1] for r in runs if r[0] == "r4d")
        hi = max(r[1] + r[2] for r in runs if r[0] == "r4d")
        bt = m.block_table
        # The whole row, not just the columns max_seq_len says are live: a few hundred thousand int32 on a
        # step that already moves 2048 tokens through 64 layers, and no dependence on that bound being tight.
        nblk = bt.shape[1]
        ids = bt[lo:hi, :nblk].to(torch.int32) * mult
        sub = torch.arange(ratio, dtype=torch.int32, device=bt.device)
        table = (ids.unsqueeze(-1) + sub).reshape(hi - lo, nblk * ratio).contiguous()
        m.hyb_cache = (key, table, lo)
        return table, lo

    def _aiter(self, layer, query, key, value, kv_cache, m, output, output_scale, output_block_scale):
        return super().forward(
            layer, query, key, value, kv_cache, m, output, output_scale, output_block_scale
        )

    # -- forward -----------------------------------------------------------------------------
    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        runs = getattr(attn_metadata, "hyb_runs", None) if attn_metadata is not None else None
        if runs is None:
            return self._aiter(
                layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
            )
        why = self._why_not
        if why is None:
            if attn_metadata.causal is not True:
                why = "non-causal batch"
            elif output_scale is not None or output_block_scale is not None:
                why = "fused output quantisation"
            elif output.dtype != torch.bfloat16:
                why = f"output dtype {output.dtype}"
            elif torch.cuda.is_current_stream_capturing():
                why = "graph capture"
        geom = None
        if why is None:
            geom = self._geometry(kv_cache)
            if isinstance(geom, str):
                why = geom
        if why is not None:
            _say_once(f"fb:{why}", f"R4D_HYBRID: step runs AITER only ({why})")
            return self._aiter(
                layer, query, key, value, kv_cache, attn_metadata, output, output_scale, output_block_scale
            )

        variant, mult, head_stride, ratio = geom
        launch = _PREFILL[variant]
        k_descale, v_descale = self._descale_ptrs(layer, query.device)
        table, lo = self._block_table16(attn_metadata, runs, mult, ratio)
        max_blocks = table.shape[1]
        block_stride = BLOCK_SIZE * kv_cache.shape[3]
        o_row = self.num_heads * self.head_size * output.element_size()
        bt_base, sl_base = table.data_ptr(), attn_metadata.seq_lens.data_ptr()
        kv_ptr, o_base = kv_cache.data_ptr(), output.data_ptr()
        max_ctx = attn_metadata.max_seq_len
        stream = torch.cuda.current_stream().cuda_stream
        if self._dummy is None:
            self._dummy = torch.empty(256, dtype=torch.uint8, device=query.device)
        scratch = self._dummy.data_ptr()
        q_scale = float(getattr(layer, "_q_scale_float", 1.0))
        _say_once(
            "route",
            f"R4D_HYBRID: long prefill runs go to libr4d (min q {MIN_Q}, cache block {ratio * BLOCK_SIZE} "
            f"slots read as {ratio} x {BLOCK_SIZE}, block-id multiplier {mult})",
        )

        for idx, run in enumerate(runs):
            if run[0] == "aiter":
                _, first_req, n_req, first_tok, n_tok, max_q = run
                if n_tok == 0:
                    continue
                sub = attn_metadata.hyb_sub.get(idx)
                if sub is None:
                    sub = attn_metadata.hyb_sub[idx] = dataclasses.replace(
                        attn_metadata,
                        num_actual_tokens=n_tok,
                        max_query_len=max_q,
                        query_start_loc=attn_metadata.query_start_loc[first_req : first_req + n_req + 1]
                        - first_tok,
                        seq_lens=attn_metadata.seq_lens[first_req : first_req + n_req],
                        block_table=attn_metadata.block_table[first_req : first_req + n_req],
                    )
                self._aiter(
                    layer,
                    query[first_tok : first_tok + n_tok],
                    key,
                    value,
                    kv_cache,
                    sub,
                    output[first_tok : first_tok + n_tok],
                    None,
                    None,
                )
                continue
            _, first_req, n_seqs, q_len, first_tok = run
            q = query[first_tok : first_tok + n_seqs * q_len]
            if q.dtype != torch.bfloat16:
                q = q.to(torch.bfloat16)
                if q_scale != 1.0:
                    q = q * q_scale
            launch(
                q.data_ptr(),
                kv_ptr,
                bt_base + (first_req - lo) * max_blocks * 4,
                sl_base + first_req * 4,
                o_base + first_tok * o_row,
                k_descale,
                v_descale,
                scratch,
                n_seqs,
                q_len,
                self.num_heads,
                self.num_kv_heads,
                self.head_size,
                BLOCK_SIZE,
                max_blocks,
                block_stride,
                head_stride,
                self.scale,
                0,
                max_ctx,
                stream,
            )
        return output
