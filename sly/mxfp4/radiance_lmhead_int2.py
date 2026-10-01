"""Two-stage greedy lm_head: a 2-bit coarse pass plus an exact top-K rerank.

Off by default; RADIANCE_LMHEAD_INT2=1 switches it on. Read at import time, like the other
RADIANCE_* knobs. Precedence: with RADIANCE_LMHEAD_INT4=1 the int4 head keeps precedence
(quant_method_for declines -- the rerank here needs a floating-point weight in place, and
composing over the int4 packed layout is future work); with RADIANCE_LMHEAD_FP8=1 the head
composes: the fp8 head packs the weight as always and the rerank scores the dequantized fp8
rows exactly.

Why: greedy decoding only needs the exact argmax, and the argmax does not need every logit
to be exact -- only the winner's row. The head keeps the weight it already has loaded (bf16
[vocab, hidden], 2.54 GB at the production vocab) and adds a compact 2-bit copy of it
(per-group absmax scales along the hidden dim, group size RADIANCE_LMHEAD_INT2_GS = 128,
four 2-bit codes packed per byte; 0.34 GB at the same vocab). Per call:

  * stage 1: coarse logits from the 2-bit copy -- the only full-vocabulary read,
  * stage 2: the top RADIANCE_LMHEAD_INT2_TOPK = 16 coarse candidates per row are rescored
    exactly against the higher-precision source (a few hundred KB of row gather), and the
    argmax over those rescored candidates is the answer.

Correctness: whenever the true argmax row is inside the coarse top-16, the two-stage result
IS the exact greedy argmax -- stage 2 recomputes that row exactly, and no other candidate
can beat it. sly/check_lmhead_int2.py measures how often the exact top-1 lands inside the
coarse top-16 (the residual risk) plus the coarse-vs-exact top-16 overlap, mirroring the
methodology of sly/check_lmhead_int4.py.

How: `RadianceLMHeadInt2` subclasses UnquantizedEmbeddingMethod so create_weights (and with
it the bf16 weight loader) stays untouched. process_weights_after_loading quantises in row
chunks (bounded transients, like the int4 head): per group of GROUP_SIZE input columns one
bf16 scale, symmetric q in [-2, 1] (scale = amax/1 -- the int4 head's convention at 2
bits, and the scale is rounded to bf16 before q is chosen so q is optimal for the scale
the dequant will actually use). Four codes share one byte. `layer.weight` is NOT replaced:
it is the exact stage-2 source, which is the whole point of keeping it; the packed copy
lands in `layer.weight_int2` (uint8 [N, K//4]) and `layer.weight_int2_scale` (bf16
[N, K//G]).

The greedy guard. The int4 head needs none: its output is a full row of approximately
exact logits, which is still a legitimate distribution a sampler may draw from. This head
is not: apply() returns a row whose top-K entries are exact and whose every other entry is
clamped below the rerank minimum, so the argmax of the returned row is by construction the
two-stage decision -- and sampling from that row would be wrong by construction. So the
knob is a greedy-only deployment contract, enforced in two layers:

  * note_sampling(): call it from wherever per-request sampling parameters become known.
    While set, apply() serves the exact full-width row instead (the stock GEMM over the
    kept weight; over the fp8 head when composed), which is valid to sample from. This
    covers every eager caller (the checker, offline tools, tests).
  * The served engine runs the head inside the captured graph, where a Python flag cannot
    be re-read per batch. There the guard has to live in the sampler, which runs eagerly:
    while this head is armed, the sampler must refuse any row that is not plain greedy
    (temperature 0, no penalties, no logit bias -- anything that post-processes logits
    could legitimately promote a non-candidate, which the clamped row cannot represent).
    Wire that refusal next to the existing radiance block in
    vllm/v1/worker/gpu/sample/sampler.py before serving anything but greedy traffic.

Consumers of the shared head: with DFlash the drafter reads its top-k bootstrap from this
same head. The clamped row's eligible set is exactly the rerank pool, so a top-16 consumer
draws from the exactly-scored candidates (bounded by the coarse pass's recall, which the
checker measures); if accept-length regresses, raise ..._TOPK or turn the knob off.
Logprobs: the row below the winner is meaningless, so a greedy serve that requests logprobs
gets garbage values for the non-winners -- do not enable where logprobs matter.

Not for `--hf-overrides '{"head_dtype": "float32"}}'` (LogitsProcessor._apply_head would
bypass apply(), same limitation as the fp8 and int4 heads).

Layout note: LAYOUT="row" packs codes k = 4j..4j+3 into byte j. LAYOUT="quarter" stores the
interleave the drafter's int2 Triton kernel reads (byte j carries k = j, K/4+j, K/2+j,
3K/4+j; see radiance_drafthead.py) -- set it when the GPU kernel that replaces the
reference stage 1 wants that layout. The reference stage 1 here (dequant in row chunks +
fp32 matmul) is the numerics definition and the CPU fallback; the GPU window wires the
2-bit GEMM kernel with the same arithmetic.

Hooked in by sly/patch_lmhead_int2.py (QuarkConfig.get_quant_method, in front of the int4
hook).
"""

import os

import torch
from torch.nn import Parameter

# The vllm imports tie the class into model loading; everything below them (packing, the
# two-stage forward, the checker's numerics) runs on plain torch, so a missing vllm (CPU
# test box, offline checker) degrades to a logging fallback instead of an ImportError.
try:
    from vllm.logger import init_logger
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead,
        UnquantizedEmbeddingMethod,
    )

    # Under the vllm.* namespace: vLLM's logging config only attaches handlers there, a
    # top-level "radiance_lmhead_int2" logger would stay silent in the EngineCore process.
    logger = init_logger("vllm.radiance.lmhead_int2")
    _HAVE_VLLM = True
except Exception:  # offline: no vllm package
    import logging

    ParallelLMHead, UnquantizedEmbeddingMethod = None, object
    logger = logging.getLogger("radiance_lmhead_int2")
    _HAVE_VLLM = False

ENABLED = os.environ.get("RADIANCE_LMHEAD_INT2", "0") == "1"
GROUP_SIZE = int(os.environ.get("RADIANCE_LMHEAD_INT2_GS", "128"))
TOPK = int(os.environ.get("RADIANCE_LMHEAD_INT2_TOPK", "16"))
# "row": byte j holds codes for k = 4j..4j+3; "quarter": the drafter kernel's interleave
# (see the module docstring). Applies to both pack and unpack.
LAYOUT = os.environ.get("RADIANCE_LMHEAD_INT2_LAYOUT", "row")
# Rows per quantisation / dequantisation chunk: 8192 x 5120 fp32 = 168 MB transient
# instead of the ~5 GB a whole-matrix fp32 would take (same bound as the int4 head).
CHUNK_ROWS = 8192
# Rows of x per rerank gather: 256 x 16 x 5120 fp32 = 84 MB transient at TOPK=16.
CHUNK_M = 256
QMAX = 1  # symmetric int2: q in [-2, 1], scale = amax / 1 (the int4 convention at 2 bits)
ZP = 2   # unsigned code = q + 2 in [0, 3]

_LOGGED: set = set()


def _info_once(key, msg, *args):
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(msg, *args)


# --- the sampling guard (see the module docstring) -------------------------------
SAMPLING_SEEN = False


def note_sampling():
    """Runtime guard: call when any row of the batch will be sampled rather than greedy.

    From then on apply() serves the exact full-width row (valid to sample from) instead of
    the clamped two-stage row. Idempotent. reset_sampling() exists for the checker/tests.
    """
    global SAMPLING_SEEN
    if not SAMPLING_SEEN:
        SAMPLING_SEEN = True
        logger.warning(
            "[radiance] int2 lm_head: sampling seen -> serving exact full logits from now "
            "on (the two-stage row is only valid for greedy argmax)")


def reset_sampling():
    global SAMPLING_SEEN
    SAMPLING_SEEN = False


# --- 2-bit packing ---------------------------------------------------------------
def pack_int2_rows(codes: torch.Tensor, layout: str = "row") -> torch.Tensor:
    """uint8 codes [rows, K] in [0, 3] -> packed uint8 [rows, K//4], four codes per byte."""
    r, k = codes.shape
    if k % 4:
        raise ValueError(f"hidden {k} not divisible by 4")
    c = codes.to(torch.int32)
    if layout == "row":
        c4 = c.reshape(r, k // 4, 4)
        b = c4[..., 0] | (c4[..., 1] << 2) | (c4[..., 2] << 4) | (c4[..., 3] << 6)
    elif layout == "quarter":
        p = k // 4
        b = (c[:, :p] | (c[:, p:2 * p] << 2) | (c[:, 2 * p:3 * p] << 4) | (c[:, 3 * p:] << 6))
    else:
        raise ValueError(f"unknown int2 layout {layout!r}")
    return b.to(torch.uint8)


def unpack_int2_rows(packed: torch.Tensor, layout: str = "row") -> torch.Tensor:
    """packed uint8 [rows, K//4] -> uint8 codes [rows, K] in [0, 3] (inverse of pack)."""
    r, q = packed.shape
    k = q * 4
    b = packed.to(torch.int32)
    if layout == "row":
        codes = torch.stack((b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3), dim=-1)
        return codes.reshape(r, k).to(torch.uint8)
    if layout == "quarter":
        p = k // 4
        codes = torch.empty((r, k), dtype=torch.uint8, device=packed.device)
        for t in range(4):
            codes[:, t * p:(t + 1) * p] = ((b >> (2 * t)) & 3).to(torch.uint8)
        return codes
    raise ValueError(f"unknown int2 layout {layout!r}")


def quantize_int2_rows(w: torch.Tensor, group_size: int, layout: str = "row"):
    """bf16/fp32 [rows, K] -> (packed uint8 [rows, K//4], bf16 scales [rows, K//G]).

    Symmetric per-group absmax quantisation; the scale is rounded to bf16 *before* q is
    chosen, so q is optimal for the scale the dequant will actually multiply with (the
    int4 head's convention). All-zero (padded vocab) rows: amax 0 -> clamp keeps the
    division finite, 0/s = 0.
    """
    r, k = w.shape
    if k % group_size or k % 4:
        raise ValueError(f"hidden {k} not divisible by group {group_size} or by 4")
    blk = w.float().reshape(r, k // group_size, group_size)
    amax = blk.abs().amax(dim=-1, keepdim=True)
    s = amax.clamp_(min=1e-8).to(torch.bfloat16).float()
    q = torch.round(blk / s).clamp_(-2, QMAX)
    codes = (q + ZP).to(torch.uint8).reshape(r, k)
    return pack_int2_rows(codes, layout), s.reshape(r, k // group_size).to(torch.bfloat16)


def dequant_int2_rows(packed: torch.Tensor, scales: torch.Tensor, group_size: int,
                      layout: str = "row") -> torch.Tensor:
    """(packed, scales) -> fp32 [rows, K]; exact inverse of quantize_int2_rows' arithmetic."""
    codes = unpack_int2_rows(packed, layout)
    s = scales.to(torch.float32)
    if s.shape[-1] * group_size != codes.shape[-1]:
        raise ValueError(f"scales {tuple(s.shape)} do not cover K={codes.shape[-1]} at g{group_size}")
    return (codes.to(torch.float32) - ZP) * s.repeat_interleave(group_size, dim=1)


# --- the two-stage forward -------------------------------------------------------
def coarse_logits(x2d: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                  group_size: int, layout: str = "row", chunk_rows: int = CHUNK_ROWS,
                  out: torch.Tensor | None = None) -> torch.Tensor:
    """Stage 1: coarse logits fp32 [M, N] = x @ dequant(2-bit copy), chunked over N.

    Reference arithmetic (and the CPU fallback): dequantise `chunk_rows` rows at a time
    into a bounded fp32 transient and matmul. The GPU window replaces this with the 2-bit
    GEMM kernel -- same inputs, same math, the packed layout flag matching the kernel.
    """
    m, k = x2d.shape
    n = packed.shape[0]
    if packed.shape[1] * 4 != k:
        raise ValueError(f"packed {tuple(packed.shape)} does not match K={k}")
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=packed.device)
    xf = x2d.float()
    for i in range(0, n, chunk_rows):
        j = min(i + chunk_rows, n)
        deq = dequant_int2_rows(packed[i:j], scales[i:j], group_size, layout=layout)
        out[:, i:j] = xf @ deq.t()
    return out


def rerank_exact(x2d: torch.Tensor, src, cand_idx: torch.Tensor, chunk_m: int = CHUNK_M):
    """Stage 2: exact fp32 scores [M, R] for the candidate rows, straight off the source.

    `src` is (weight [N, K], per-row scale [N] fp32 or None) -- the floating-point weight
    the head kept (bf16 by default), or the fp8 head's e4m3 rows plus their per-channel
    scale, which dequantize exactly. Gathered in M chunks to bound the transient.
    """
    m = x2d.shape[0]
    w, wsc = src
    cand = torch.empty((m, cand_idx.shape[1]), dtype=torch.float32, device=x2d.device)
    # advanced indexing of fp8 is not implemented on every backend; a bitcast view to
    # uint8 is free and indexable everywhere, so gather through it
    wv = w.view(torch.uint8) if w.dtype == torch.float8_e4m3fn else w
    for m0 in range(0, m, chunk_m):
        m1 = min(m0 + chunk_m, m)
        idx = cand_idx[m0:m1]
        rows = wv[idx]
        if rows.dtype == torch.uint8:
            rows = rows.view(torch.float8_e4m3fn)
        rowsf = rows.float()
        if wsc is not None:
            rowsf = rowsf * wsc[idx].float().unsqueeze(-1)
        cand[m0:m1] = torch.einsum("mk,mrk->mr", x2d[m0:m1].float(), rowsf)
    return cand


def greedy_two_stage(x2d: torch.Tensor, packed: torch.Tensor, scales: torch.Tensor,
                     group_size: int, topk: int, src, bias: torch.Tensor | None = None,
                     layout: str = "row"):
    """The two-stage greedy forward on a [M, K] batch.

    Returns (mixture fp32 [M, N], cand_idx int64 [M, R], cand fp32 [M, R]): `cand` holds
    the exact greedy scores (bias folded in when one is given) of the R coarse
    candidates, and `mixture` is the returned-logits encoding of the decision: exact
    scores at the candidate positions, every other entry clamped below the candidates'
    minimum, so argmax(mixture) == argmax(cand) always -- the row is valid for greedy
    argmax and for nothing else.
    """
    n = packed.shape[0]
    r = min(topk, n)
    coarse = coarse_logits(x2d, packed, scales, group_size, layout=layout)
    if bias is not None:
        coarse = coarse + bias
    cand_idx = coarse.topk(r, dim=1).indices
    cand = rerank_exact(x2d, src, cand_idx)
    if bias is not None:
        # fold the bias into the rerank scores too: the pool was selected under the
        # biased coarse scores, so the exact decision must rank under the same bias
        cand = cand + bias.reshape(-1)[cand_idx]
    floor = cand.min(dim=1, keepdim=True).values
    # clamp non-candidates one ulp strictly below the pool minimum: an exact tie
    # between a non-candidate and the weakest candidate could otherwise let argmax's
    # first-index tie-break promote a row the rerank never scored
    mixture = torch.minimum(coarse, torch.nextafter(floor, torch.full_like(floor,
                                                                       float("-inf"))))
    mixture.scatter_(1, cand_idx, cand)
    return mixture, cand_idx, cand


# --- the vLLM quant method --------------------------------------------------------
def _src_rows(layer: torch.nn.Module):
    """(weight [N, K], per-row scale [N] fp32 or None) for the exact stage-2 source.

    bf16/fp16/fp32: the weight itself (the default head). float8_e4m3fn with a per-row
    `weight_scale` (the fp8 head's storage, [N, K] rows + [1, N] scale): the rows plus
    their scale, which dequantize exactly. Anything else (e.g. the int4 head's packed int8,
    whose layout this module must not guess at): None -> apply() falls back.
    """
    w = layer.weight
    if w.dim() != 2:
        return None
    if w.dtype in (torch.bfloat16, torch.float16, torch.float32):
        return w, None
    if w.dtype == torch.float8_e4m3fn:
        sc = getattr(layer, "weight_scale", None)
        if sc is not None and sc.numel() == w.shape[0]:
            return w, sc.detach().reshape(-1).float()
    return None


def quant_method_for(layer: torch.nn.Module, prefix: str):
    """get_quant_method hook: our method for an enabled ParallelLMHead, else None.

    RADIANCE_LMHEAD_INT4=1 keeps precedence: the int4 head is the shipped one, and this
    head cannot rerank over its packed layout, so with both knobs set we decline. With
    RADIANCE_LMHEAD_FP8=1 the method composes: the fp8 head packs the weight as it always
    does and the rerank scores the dequantized fp8 rows (exact), keeping the fp8 head's
    own memory win while the coarse pass reads the 2-bit copy.
    """
    if not ENABLED or not _HAVE_VLLM or not isinstance(layer, ParallelLMHead):
        return None
    if os.environ.get("RADIANCE_LMHEAD_INT4", "0") == "1":
        _info_once(
            prefix, "[radiance] %s -> int2 declined: RADIANCE_LMHEAD_INT4=1 (the int4 head "
            "keeps precedence; reranking over its packed layout is future work)", prefix)
        return None
    if os.environ.get("RADIANCE_LMHEAD_FP8", "0") == "1":
        import radiance_lmhead_fp8

        _info_once(prefix, "[radiance] %s -> int2 coarse + exact rerank over the fp8 head "
                           "(group %d, top-%d, greedy-only)", prefix, GROUP_SIZE, TOPK)
        return RadianceLMHeadInt2OverFp8(radiance_lmhead_fp8.RadianceLMHeadFp8())
    _info_once(prefix, "[radiance] %s -> int2 two-stage greedy head (group %d, top-%d, "
                       "greedy-only)", prefix, GROUP_SIZE, TOPK)
    return RadianceLMHeadInt2()


class RadianceLMHeadInt2(UnquantizedEmbeddingMethod):
    """create_weights/embedding as the base class; the loaded weight stays, apply() changes."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "weight_int2", None) is not None:
            return  # already built (a second pass over shared modules)
        w = layer.weight
        assert w.dim() == 2, f"lm_head weight must be [vocab, hidden], got {tuple(w.shape)}"
        n, k = w.shape
        g = GROUP_SIZE
        assert k % g == 0 and k % 4 == 0, f"hidden {k} not divisible by group {g} or by 4"
        sc = None
        if w.dtype == torch.float8_e4m3fn:
            # a head somebody else already packed per-channel (the fp8 head's storage):
            # build the coarse copy from the dequantized rows -- exact w.r.t. that head
            sc = getattr(layer, "weight_scale", None)
            assert sc is not None and sc.numel() == n, "fp8 lm_head without a per-row scale"
            sc = sc.detach().reshape(-1).float()
        elif not w.dtype.is_floating_point:
            # e.g. the int4 head's packed int8: this module must not guess at that layout
            logger.warning("[radiance] lm_head is %s; the int2 head needs a floating-point "
                           "weight for its exact rerank -- keeping the loaded head as is",
                           w.dtype)
            return
        packed = torch.empty((n, k // 4), dtype=torch.uint8, device=w.device)
        scale = torch.empty((n, k // g), dtype=torch.bfloat16, device=w.device)
        for i in range(0, n, CHUNK_ROWS):
            j = min(i + CHUNK_ROWS, n)
            blk = w[i:j].float()
            if sc is not None:
                blk = blk * sc[i:j, None]
            packed[i:j], scale[i:j] = quantize_int2_rows(blk, g, LAYOUT)
            del blk
        layer.weight_int2 = Parameter(packed.contiguous(), requires_grad=False)
        layer.weight_int2_scale = Parameter(scale.contiguous(), requires_grad=False)
        logger.info(
            "[radiance] lm_head int2 coarse copy built: [%d, %d] group %d, %s layout, "
            "+%.2f GiB resident; the %s weight stays as the exact rerank source",
            n, k, g, LAYOUT, packed.numel() / 2**30 + scale.numel() * 2 / 2**30, w.dtype,
        )

    def _exact_full(self, layer: torch.nn.Module, x: torch.Tensor,
                    bias: torch.Tensor | None) -> torch.Tensor:
        """The sampling fallback: a full row of (approximately) exact logits, valid to
        sample from. Over the fp8 head when composed; otherwise the stock GEMM over the
        weight this head kept."""
        inner = getattr(self, "inner", None)
        if inner is not None:
            return inner.apply(layer, x, bias)
        x2d = x.reshape(-1, x.shape[-1])
        out = torch.matmul(x2d, layer.weight.t())
        if bias is not None:
            out = out + bias
        return out.reshape(x.shape[:-1] + (out.shape[-1],))

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        src = _src_rows(layer)
        if SAMPLING_SEEN or src is None or getattr(layer, "weight_int2", None) is None:
            # sampling seen (eager callers), or nothing this head can do -- never serve
            # the clamped row where it would be wrong; serve the exact full row instead
            return self._exact_full(layer, x, bias)
        out, _, _ = greedy_two_stage(x_2d, layer.weight_int2, layer.weight_int2_scale,
                                     GROUP_SIZE, TOPK, src, bias, LAYOUT)
        return out.to(x.dtype).reshape(x.shape[:-1] + (out.shape[-1],))


class RadianceLMHeadInt2OverFp8(RadianceLMHeadInt2):
    """int2 two-stage head composed over the fp8 head (RADIANCE_LMHEAD_FP8=1).

    Loading is untouched: create_weights is the same base (one bf16 parameter), the loader
    fills bf16. process_weights_after_loading builds the 2-bit coarse copy from the bf16
    weight FIRST (it must come from the unpacked weight), then hands the layer to the fp8
    head, which packs it exactly as it does without this knob. apply(): coarse pass on
    the 2-bit copy, rerank on the dequantized fp8 rows -- slightly EXACTER than the fp8
    head's own output, which additionally quantizes the activation per token. The sampling
    fallback is the fp8 head itself (a full row of approximately exact logits).
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if getattr(layer, "weight_int2", None) is not None:
            return  # a second pass over shared modules; the fp8 head skips itself likewise
        super().process_weights_after_loading(layer)  # coarse copy from the bf16 weight
        self.inner.process_weights_after_loading(layer)  # then pack the source it packs
