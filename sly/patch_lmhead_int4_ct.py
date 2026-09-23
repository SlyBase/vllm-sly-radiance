#!/usr/bin/env python3
"""Hook radiance_lmhead_int4 into CompressedTensorsConfig.get_quant_method (vLLM 0.29.0).

Why: sly/patch_lmhead_int4.py only hooks QuarkConfig, so a compressed-tensors W4A16 target
(RedHatAI/Qwen3.8-27B-INT4: lm_head is in the checkpoint's `ignore` list, i.e. bf16) keeps a
2.37 GiB bf16 head -- 2x the bytes per call of the int4 head and ~1.9 GiB less KV cache.
With RADIANCE_LMHEAD_INT4=1 a ParallelLMHead now gets RadianceLMHeadInt4 here too, on the same
W4A16 kernel path and tile table as the Quark case.

Hook point: the ParallelLMHead branch, right after the scheme lookup and before its result is
used. An `ignore`d head (scheme None) gets RadianceLMHeadInt4 on the bf16 weight as before. A head
the checkpoint stores quantized (unsloth/Qwen3.8-27B-NVFP4: FP8 per-channel) gets
RadianceLMHeadInt4Over wrapping the checkpoint's own CompressedTensorsLinearMethod: that one loads
the fp8 weight + scale and (radiance_nvfp4, RADIANCE_NVFP4_LMHEAD=bf16) dequantizes it at load,
then the int4 pass runs. Before 0.3.6 the hook ran ahead of the lookup and gave every head a bf16
parameter, so an fp8 checkpoint head would have loaded as unscaled e4m3 codes. The DFlash drafter
builds its ParallelLMHead without a quant_config (and shares the target head), so only the target
head is affected. RADIANCE_LMHEAD_INT4 unset -> quant_method_for returns None -> stock behaviour.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = (Path(sysconfig.get_paths()["purelib"])
     / "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py")

apply(F,
      '        if isinstance(layer, ParallelLMHead):\n'
      '            try:\n'
      '                quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)\n'
      '            except ValueError:\n'
      '                quant_scheme = None\n'
      '            if quant_scheme is not None:\n',
      '        if isinstance(layer, ParallelLMHead):\n'
      '            try:\n'
      '                quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)\n'
      '            except ValueError:\n'
      '                quant_scheme = None\n'
      '            # --- radiance (sly/patch_lmhead_int4_ct.py): int4 lm_head, RADIANCE_LMHEAD_INT4=1 ---\n'
      '            try:\n'
      '                import radiance_lmhead_int4 as _radiance_lmhead_int4\n'
      '\n'
      '                _radiance_method = _radiance_lmhead_int4.quant_method_for(\n'
      '                    layer, prefix,\n'
      '                    inner=None if quant_scheme is None else CompressedTensorsLinearMethod(self))\n'
      '                if _radiance_method is not None:\n'
      '                    if quant_scheme is not None:\n'
      '                        layer.scheme = quant_scheme\n'
      '                    return _radiance_method\n'
      '            except Exception as _radiance_exc:  # never block model load on our own head\n'
      '                logger.warning_once("[radiance] int4 lm_head unavailable: %r", _radiance_exc)\n'
      '            if quant_scheme is not None:\n',
      '# --- radiance (sly/patch_lmhead_int4_ct.py): int4 lm_head, RADIANCE_LMHEAD_INT4=1 ---',
      'compressed_tensors: RadianceLMHeadInt4 for ParallelLMHead (RADIANCE_LMHEAD_INT4=1)')
