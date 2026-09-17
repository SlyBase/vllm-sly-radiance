#!/usr/bin/env python3
"""Hook radiance_lmhead_int4 into CompressedTensorsConfig.get_quant_method (vLLM 0.29.0).

Why: sly/patch_lmhead_int4.py only hooks QuarkConfig, so a compressed-tensors W4A16 target
(RedHatAI/Qwen3.8-27B-INT4: lm_head is in the checkpoint's `ignore` list, i.e. bf16) keeps a
2.37 GiB bf16 head -- 2x the bytes per call of the int4 head and ~1.9 GiB less KV cache.
With RADIANCE_LMHEAD_INT4=1 a ParallelLMHead now gets RadianceLMHeadInt4 here too, on the same
W4A16 kernel path and tile table as the Quark case.

Hook point: first thing in the ParallelLMHead branch, before the scheme lookup whose `ignore`
match would hand the head to the unquantized embedding method. The DFlash drafter builds its
ParallelLMHead without a quant_config (and shares the target head), so only the target head is
affected. RADIANCE_LMHEAD_INT4 unset -> quant_method_for returns None -> stock behaviour.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = (Path(sysconfig.get_paths()["purelib"])
     / "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py")

apply(F,
      '        if isinstance(layer, ParallelLMHead):\n'
      '            try:\n'
      '                quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)\n',
      '        if isinstance(layer, ParallelLMHead):\n'
      '            # --- radiance (sly/patch_lmhead_int4_ct.py): int4 lm_head, RADIANCE_LMHEAD_INT4=1 ---\n'
      '            try:\n'
      '                import radiance_lmhead_int4 as _radiance_lmhead_int4\n'
      '\n'
      '                _radiance_method = _radiance_lmhead_int4.quant_method_for(layer, prefix)\n'
      '                if _radiance_method is not None:\n'
      '                    return _radiance_method\n'
      '            except Exception as _radiance_exc:  # never block model load on our own head\n'
      '                logger.warning_once("[radiance] int4 lm_head unavailable: %r", _radiance_exc)\n'
      '            try:\n'
      '                quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)\n',
      '# --- radiance (sly/patch_lmhead_int4_ct.py): int4 lm_head, RADIANCE_LMHEAD_INT4=1 ---',
      'compressed_tensors: RadianceLMHeadInt4 for ParallelLMHead (RADIANCE_LMHEAD_INT4=1)')
