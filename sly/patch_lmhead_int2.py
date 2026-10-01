#!/usr/bin/env python3
"""Hook radiance_lmhead_int2 into QuarkConfig.get_quant_method (vLLM 0.29.0).

Same hook point and same ordering trick as sly/patch_lmhead_int4.py, one slot further
in front: the int2 block is inserted before the int4 block, so with
RADIANCE_LMHEAD_INT4=1 AND RADIANCE_LMHEAD_INT2=1 the int2 hook runs first -- and
declines (quant_method_for returns None), because the int4 head replaces the bf16
weight the int2 rerank needs. Precedence stays with the shipped head; int2 applies
only when it is the enabled lm-head knob (alone, or composed over
RADIANCE_LMHEAD_FP8=1, whose per-channel fp8 rows the rerank scores exactly). With
the knob unset the hook returns None before touching anything and the load path is
byte-identical to stock.

RADIANCE_LMHEAD_INT2=1 is a greedy-only deployment contract: the head returns a
logits row whose argmax is the two-stage decision and whose remaining entries are
clamped below the rerank minimum, so sampling from it would be wrong by construction.
The module carries the runtime guard for eager callers (note_sampling()); the served
engine runs the head inside the captured graph, where a Python flag cannot be re-read
per batch, so the sampler-side refusal documented in the module docstring must be
wired before serving anything but plain-greedy traffic.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/layers/quantization/quark/quark.py"

INT4_BLOCK = '        # --- radiance (sly/patch_lmhead_int4.py): int4 lm_head, RADIANCE_LMHEAD_INT4=1 ---\n'

apply(F,
      INT4_BLOCK,
      '        # --- radiance (sly/patch_lmhead_int2.py): int2 lm_head, RADIANCE_LMHEAD_INT2=1 ---\n'
      '        # In front of the int4 block; quant_method_for declines when RADIANCE_LMHEAD_INT4=1,\n'
      '        # so with both knobs the shipped int4 head keeps precedence.\n'
      '        try:\n'
      '            import radiance_lmhead_int2 as _radiance_lmhead_int2\n'
      '\n'
      '            _radiance_method = _radiance_lmhead_int2.quant_method_for(layer, prefix)\n'
      '            if _radiance_method is not None:\n'
      '                return _radiance_method\n'
      '        except Exception as _radiance_exc:  # never block model load on our own head\n'
      '            logger.warning_once("[radiance] int2 lm_head unavailable: %r", _radiance_exc)\n'
      + INT4_BLOCK,
      '    # --- radiance (sly/patch_lmhead_int2.py): int2 lm_head, RADIANCE_LMHEAD_INT2=1 ---',
      'quark: RadianceLMHeadInt2 for ParallelLMHead (RADIANCE_LMHEAD_INT2=1)')
