"""lm_head in int4 (W4A16, symmetric, per-group scales) for the Quark MXFP4 target.

Off by default; RADIANCE_LMHEAD_INT4=1 switches it on and takes precedence over
RADIANCE_LMHEAD_FP8 (the int4 hook runs first in QuarkConfig.get_quant_method). Read at
import time, like the other RADIANCE_* knobs.

Why: with RADIANCE_LMHEAD_FP8=1 the shared DFlash head already runs at the bandwidth ceiling
(hipBLASLt fp8 GEMM, 1.27 GB per call at 543 GB/s = 2.34 ms, twice per step = 11 % of the
42 ms DFlash step on vllm7, 2026-09-15). The only lever left is fewer bytes: int4 with
group-128 bf16 scales is 656 MB per call. The kernel is the one the W4A16 DFlash drafter
already runs on (vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py:
HIP wvSplitK_int4_g for M <= 5, otherwise _triton_w4a16_skinny_fmt_kernel with the gfx1201
tile table from sly/patch_w4a16_tiles.py), so no new kernel is involved and the activation
stays bf16 (no per-token quant launch).

How: `RadianceLMHeadInt4` subclasses UnquantizedEmbeddingMethod so create_weights (and with it
the bf16 weight loader) stays untouched. process_weights_after_loading quantises in row
chunks: per group of RADIANCE_LMHEAD_INT4_GS (default 128) input columns one bf16 scale,
symmetric q in [-8, 7]; with RADIANCE_LMHEAD_INT4_CLIP=mse (default) the scale of every group
is picked from a small clip-ratio search (1.0 .. 0.8 x amax/7) by least squared error --
that is what AWQ/GPTQ-style tooling does for the output layer and it lowers the
quantisation error by 10-20 % at < 1 s of load time. The nibbles are packed with vLLM's own
pack_int4_exllama_shuffle into the [N, K//8] int32 layout the kernel reads, stored as the
int8 view the kernel path expects, and `layer.weight` is replaced in place (same name, so
everything that introspects a loaded lm_head still works); `layer.weight_scale` [N, K//G]
bf16 is added. The freed 1.88 GiB (vs bf16; 0.61 GiB vs fp8) is seen by the KV-cache
profiling that runs later.

apply() calls torch.ops.vllm.rdna_hybrid_w4a16_apply exactly like
RDNAHybridW4A16LinearKernel.apply_weights does for the drafter.

Accuracy: int4-g128 is ~3x the per-weight RMS error of per-channel fp8 on this matrix;
the logit error averages out over K = 5120 but the layer decides the target argmax and the
drafter's top-k candidates, so it ships only behind the GSM8K / accept-length gate (plan
"Schritt 5"). sly/check_lmhead_int4.py measures argmax flips and top-16 overlap offline.

Not for `--hf-overrides '{"head_dtype": "float32"}'` (LogitsProcessor._apply_head would
bypass apply(), same limitation as the fp8 head).

Hooked in by sly/patch_lmhead_int4.py (QuarkConfig.get_quant_method, before the fp8 hook).
"""

import os

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)

logger = init_logger("vllm.radiance.lmhead_int4")

ENABLED = os.environ.get("RADIANCE_LMHEAD_INT4", "0") == "1"
GROUP_SIZE = int(os.environ.get("RADIANCE_LMHEAD_INT4_GS", "128"))
# "mse": per-group clip-ratio search (default); "0"/"rtn": plain round-to-nearest at amax/7.
CLIP = os.environ.get("RADIANCE_LMHEAD_INT4_CLIP", "mse")
CLIP_RATIOS = (1.0, 0.95, 0.9, 0.85, 0.8)
# Rows per quantisation chunk: 8192 x 5120 fp32 = 168 MB per transient (a few of them live
# during the clip search) instead of the 5 GB a whole-matrix .float() would take.
CHUNK_ROWS = 8192
QMAX = 7  # symmetric int4: q in [-8, 7], scale = amax / 7 (compressed-tensors convention)
ZP_BIAS = 8  # unsigned nibble = q + 8, the constant the kernel subtracts (HAS_ZP=False)


def quant_method_for(layer: torch.nn.Module, prefix: str):
    """QuarkConfig.get_quant_method hook: our method for an enabled ParallelLMHead, else None."""
    if not ENABLED or not isinstance(layer, ParallelLMHead):
        return None
    logger.info_once(
        "[radiance] %s -> int4 W4A16 (group %d, clip=%s)", prefix, GROUP_SIZE, CLIP
    )
    return RadianceLMHeadInt4()


def quantize_int4_rows(w: torch.Tensor, group_size: int, clip: str | None = None):
    """bf16/fp32 [rows, K] -> (packed int32 [rows, K//8] ExLlama-shuffled, bf16 scales [rows, K//G]).

    Symmetric per-group quantisation; the scale is rounded to bf16 *before* q is chosen, so q
    is optimal for the scale the kernel will actually multiply with.
    """
    from vllm.model_executor.kernels.linear.mixed_precision.rdna_hybrid_w4a16 import (
        pack_int4_exllama_shuffle,
    )

    clip = CLIP if clip is None else clip
    rows, k = w.shape
    ng = k // group_size
    blk = w.float().view(rows, ng, group_size)
    amax = blk.abs().amax(dim=-1, keepdim=True)
    ratios = CLIP_RATIOS if clip == "mse" else (1.0,)
    best_err = None
    best_q = None
    best_s = None
    for r in ratios:
        # Padded (all-zero) vocab rows: amax 0 -> clamp keeps the division finite, 0/s = 0.
        s = (amax * (r / QMAX)).clamp_(min=1e-8).to(torch.bfloat16).float()
        q = torch.round(blk / s).clamp_(-ZP_BIAS, QMAX)
        err = (q * s - blk).square_().sum(dim=-1, keepdim=True)
        if best_err is None:
            best_err, best_q, best_s = err, q, s
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best_q = torch.where(better, q, best_q)
            best_s = torch.where(better, s, best_s)
        del s, q, err
    nib = (best_q.view(rows, k) + ZP_BIAS).to(torch.uint8)
    packed = pack_int4_exllama_shuffle(nib)
    scales = best_s.view(rows, ng).to(torch.bfloat16)
    return packed, scales


class RadianceLMHeadInt4(UnquantizedEmbeddingMethod):
    """create_weights/embedding as the base class; only the loaded weight and apply() change."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight
        if w.dtype == torch.int8:  # already packed (a second pass over shared modules)
            return
        assert w.dim() == 2, f"lm_head weight must be [vocab, hidden], got {tuple(w.shape)}"
        n, k = w.shape
        g = GROUP_SIZE
        assert k % g == 0 and k % 8 == 0, f"hidden {k} not divisible by group {g}"
        w_q = torch.empty((n, k // 8), dtype=torch.int32, device=w.device)
        scale = torch.empty((n, k // g), dtype=torch.bfloat16, device=w.device)
        for i in range(0, n, CHUNK_ROWS):
            packed, s = quantize_int4_rows(w[i:i + CHUNK_ROWS], g)
            w_q[i:i + CHUNK_ROWS] = packed
            scale[i:i + CHUNK_ROWS] = s
            del packed, s
        # int8 view of the [N, K//8] int32 packing = [N, K//2]; the kernel path re-views it as
        # int32 (that is how RDNAHybridW4A16LinearKernel stores the drafter's weights too).
        layer.weight = Parameter(w_q.view(torch.int8).contiguous(), requires_grad=False)
        layer.weight_scale = Parameter(scale.contiguous(), requires_grad=False)
        logger.info(
            "[radiance] lm_head quantised to int4 g%d: [%d, %d], %.2f GiB freed",
            g, n, k, (n * k * w.element_size() - n * k // 2 - scale.numel() * 2) / 2**30,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.utils.platform_utils import num_compute_units

        x_2d = x.reshape(-1, x.shape[-1])
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        out = torch.ops.vllm.rdna_hybrid_w4a16_apply(
            x_2d, layer.weight, layer.weight_scale, None, bias, num_compute_units(), GROUP_SIZE
        )
        return out.reshape(x.shape[:-1] + (out.shape[-1],))
