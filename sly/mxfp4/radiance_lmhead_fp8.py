"""lm_head in fp8 (per-output-channel weight, per-token activation) for the Quark MXFP4 target.

Off by default; RADIANCE_LMHEAD_FP8=1 switches it on. Read at import time, like the other
RADIANCE_* knobs in radiance_mxfp4.py.

Why: the Quark checkpoint lists lm_head in its `exclude` list, so QuarkConfig.get_quant_method
hands out UnquantizedLinearMethod and the projection runs as a bf16 hipBLASLt GEMM over the
full 248320 x 5120 vocabulary matrix -- 2.54 GB of weight traffic per call, 4.17 ms at the
609 GB/s that GEMM reaches on the R9700. With DFlash the shared head runs twice per step
(target verify + draft): 8.6 of ~49 ms (profile D, vllm7, 2026-09-15). fp8 halves the traffic;
per-output-channel scales keep the quantisation error of every vocabulary row independent of
the others, which is what makes fp8 on the output layer practically lossless.

How: `RadianceLMHeadFp8` subclasses UnquantizedEmbeddingMethod so create_weights (and with it
the bf16 weight loader) stays untouched. process_weights_after_loading quantises in row chunks
(bounded transient), replaces `layer.weight` with the fp8 tensor and adds `layer.weight_scale`
(fp32, one per row). The freed 1.27 GiB is seen by the KV-cache profiling that runs later.
apply() quantises the activation per token with the same dynamic_scaled_fp8_quant the W4A8
decode kernel uses and runs torch._scaled_mm row-wise on hipBLASLt (vLLM's own
RowWiseTorchFP8ScaledMMLinearKernel declares gfx12x supported and bf16 the only output dtype;
bf16 is the model dtype here).

DFlash shares the target lm_head with the drafter (v1/worker/gpu/spec_decode/dflash/utils.py,
load_dflash_model), so both calls per step take this path. The drafter's own ParallelLMHead
(compressed-tensors config, no weights in the DFlash2 checkpoint) is deleted on sharing and
never sees this class.

Not for `--hf-overrides '{"head_dtype": "float32"}'`: LogitsProcessor._apply_head would then
bypass apply() and torch.mm the fp8 tensor, which fails loudly at the first request.

Hooked in by sly/patch_lmhead_fp8.py (QuarkConfig.get_quant_method, before the exclude check).
"""

import os

import torch
from torch.nn import Parameter

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.platforms import current_platform

# Under the vllm.* namespace: vLLM's logging config only attaches handlers there, a
# top-level "radiance_lmhead_fp8" logger would stay silent in the EngineCore process.
logger = init_logger("vllm.radiance.lmhead_fp8")

ENABLED = os.environ.get("RADIANCE_LMHEAD_FP8", "0") == "1"
# e4m3fn on gfx12x; the same dtype the activation quant below produces.
FP8_DTYPE = current_platform.fp8_dtype()
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)
# Rows per quantisation chunk: 16384 x 5120 fp32 = 320 MB transient instead of the 5 GB a
# whole-matrix .float() would take while the drafter and the graph pool still need headroom.
CHUNK_ROWS = 16384
# hipBLASLt's _scaled_mm prefers M >= 16 (vLLM pads its fp8 linears for the same reason);
# DFlash's verify batch is 8 tokens per request, so single-stream decode sits below that.
# 0 disables the padding; tunable for A/B without a rebuild.
MIN_M = int(os.environ.get("RADIANCE_LMHEAD_FP8_MIN_M", "16"))


def quant_method_for(layer: torch.nn.Module, prefix: str):
    """QuarkConfig.get_quant_method hook: our method for an enabled ParallelLMHead, else None."""
    if not ENABLED or not isinstance(layer, ParallelLMHead):
        return None
    logger.info_once(
        "[radiance] %s -> fp8 (per-channel weight, per-token activation, min_m=%d)",
        prefix, MIN_M,
    )
    return RadianceLMHeadFp8()


class RadianceLMHeadFp8(UnquantizedEmbeddingMethod):
    """create_weights/embedding as the base class; only the loaded weight and apply() change."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight
        if w.dtype == FP8_DTYPE:  # already quantised (a second pass over shared modules)
            return
        assert w.dim() == 2, f"lm_head weight must be [vocab, hidden], got {tuple(w.shape)}"
        n, k = w.shape
        w_fp8 = torch.empty((n, k), dtype=FP8_DTYPE, device=w.device)
        scale = torch.empty((n,), dtype=torch.float32, device=w.device)
        for i in range(0, n, CHUNK_ROWS):
            blk = w[i:i + CHUNK_ROWS].float()
            # Padded (all-zero) vocab rows: amax 0 -> clamp keeps the division finite, 0/s = 0.
            s = (blk.abs().amax(dim=1) / FP8_MAX).clamp_(min=1e-12)
            w_fp8[i:i + CHUNK_ROWS] = (blk / s[:, None]).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
            scale[i:i + CHUNK_ROWS] = s
            del blk
        # Keep the parameter name: everything that introspects a loaded lm_head looks at
        # `.weight`. (1, N) is the row-wise scale_b layout torch._scaled_mm wants.
        layer.weight = Parameter(w_fp8, requires_grad=False)
        layer.weight_scale = Parameter(scale.view(1, n), requires_grad=False)
        logger.info(
            "[radiance] lm_head quantised to fp8: [%d, %d], %.2f GiB freed",
            n, k, n * k * (w.element_size() - 1) / 2**30,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = x.shape[0]
        pad = MIN_M if 0 < m < MIN_M else None
        x_fp8, x_scale = ops.scaled_fp8_quant(
            x, scale=None, num_token_padding=pad, use_per_token_if_dynamic=True
        )
        # weight is [N, K] row-major; .t() is the column-major [K, N] operand B expects.
        out = torch._scaled_mm(
            x_fp8,
            layer.weight.t(),
            scale_a=x_scale,
            scale_b=layer.weight_scale,
            bias=bias,
            out_dtype=x.dtype,
        )
        return out[:m] if pad else out
