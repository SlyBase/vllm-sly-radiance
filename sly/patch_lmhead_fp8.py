#!/usr/bin/env python3
"""Hook radiance_lmhead_fp8 into QuarkConfig.get_quant_method (vLLM 0.29.0).

The Quark checkpoint (amd/Qwen3.8-27B-Quark-AWQ-MXFP4) lists lm_head in its `exclude`
list; the exclude branch of get_quant_method returns UnquantizedLinearMethod for it and
the logits run as a bf16 hipBLASLt GEMM over the whole vocabulary (2.54 GB per call,
twice per step with DFlash -- 8.6 of ~49 ms on vllm7). This inserts one guarded call at
the top of get_quant_method, before the exclude check, that returns RadianceLMHeadFp8
for a ParallelLMHead when RADIANCE_LMHEAD_FP8=1 and leaves every other layer alone.

Lazy import: get_quant_method runs in the worker at model construction; the module only
imports vLLM layers and torch (no HIP init), but keeping it out of quark.py's import
graph means a broken radiance_lmhead_fp8 can never take Quark itself down.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply_any

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/layers/quantization/quark/quark.py"

SIG = (
    '    def get_quant_method(\n'
    '        self, layer: torch.nn.Module, prefix: str\n'
    '    ) -> "QuantizeMethodBase | None":\n'
)
RAD = (
    '        # --- radiance (sly/patch_lmhead_fp8.py): fp8 lm_head, RADIANCE_LMHEAD_FP8=1 ---\n'
    '        # Before the exclude check: the checkpoint excludes lm_head, and that branch hands\n'
    '        # out UnquantizedLinearMethod (bf16 GEMM over the whole vocabulary) for it.\n'
    '        try:\n'
    '            import radiance_lmhead_fp8 as _radiance_lmhead\n'
    '\n'
    '            _radiance_method = _radiance_lmhead.quant_method_for(layer, prefix)\n'
    '            if _radiance_method is not None:\n'
    '                return _radiance_method\n'
    '        except Exception as _radiance_exc:  # never block model load on our own head\n'
    '            logger.warning_once("[radiance] fp8 lm_head unavailable: %r", _radiance_exc)\n'
)
# vLLM 0.29.0: the body opens with the exclude check.
BODY_029 = (
    '        # Check if the layer is skipped for quantization.\n'
    '        exclude_layers = cast(list[str], self.quant_config.get("exclude"))\n'
)
# vLLM 0.30.0: the body opens with get_quant_method_target(), which resolves the exclude list
# (lm_head -> no method at all, i.e. a bf16 ParallelLMHead); the hook goes in front of it.
BODY_030 = (
    '        weight_quant_key, activation_quant_key, method_cls = (\n'
    '            self.get_quant_method_target(prefix, type(layer))\n'
    '        )\n'
)

apply_any(F, [(SIG + BODY_029, SIG + RAD + BODY_029), (SIG + BODY_030, SIG + RAD + BODY_030)],
          '    # --- radiance (sly/patch_lmhead_fp8.py): fp8 lm_head, RADIANCE_LMHEAD_FP8=1 ---',
          'quark: RadianceLMHeadFp8 for ParallelLMHead (RADIANCE_LMHEAD_FP8=1)')
