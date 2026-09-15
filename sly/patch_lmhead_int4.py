#!/usr/bin/env python3
"""Hook radiance_lmhead_int4 into QuarkConfig.get_quant_method (vLLM 0.29.0).

Same hook point as sly/patch_lmhead_fp8.py (whose block this one is inserted in front of,
so it must run after that patch): before the exclude check that would hand out
UnquantizedLinearMethod for the Quark-excluded lm_head. With RADIANCE_LMHEAD_INT4=1 a
ParallelLMHead gets RadianceLMHeadInt4 (int4 W4A16 on the drafter's kernel path, 656 MB
per call instead of fp8's 1.27 GB); everything else, and the fp8 hook, stays as is. With
both envs set int4 wins because its block comes first.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/layers/quantization/quark/quark.py"

FP8_BLOCK = '        # --- radiance (sly/patch_lmhead_fp8.py): fp8 lm_head, RADIANCE_LMHEAD_FP8=1 ---\n'

apply(F,
      FP8_BLOCK,
      '        # --- radiance (sly/patch_lmhead_int4.py): int4 lm_head, RADIANCE_LMHEAD_INT4=1 ---\n'
      '        # Before the fp8 block: int4 takes precedence when both knobs are set.\n'
      '        try:\n'
      '            import radiance_lmhead_int4 as _radiance_lmhead_int4\n'
      '\n'
      '            _radiance_method = _radiance_lmhead_int4.quant_method_for(layer, prefix)\n'
      '            if _radiance_method is not None:\n'
      '                return _radiance_method\n'
      '        except Exception as _radiance_exc:  # never block model load on our own head\n'
      '            logger.warning_once("[radiance] int4 lm_head unavailable: %r", _radiance_exc)\n'
      + FP8_BLOCK,
      '    # --- radiance (sly/patch_lmhead_int4.py): int4 lm_head, RADIANCE_LMHEAD_INT4=1 ---',
      'quark: RadianceLMHeadInt4 for ParallelLMHead (RADIANCE_LMHEAD_INT4=1)')
