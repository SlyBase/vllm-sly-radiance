#!/usr/bin/env python3
"""Hook radiance_embed_int8 into QuarkConfig.get_quant_method (vLLM 0.29.0) + compile key.

Same hook point as sly/patch_lmhead_int4.py, inserted in front of its block (so this runs
after it in the Dockerfile loop): with RADIANCE_EMBED_INT8=1 the Quark-excluded, bf16
embed_tokens (248320 x 5120 = 2.37 GiB on Qwen3.8-27B) is stored as int8 rows (1.18 GiB
back for the KV cache) or, with RADIANCE_EMBED_BITS=4, as group-128 nibbles (1.74 GiB back).
Only VocabParallelEmbedding that is not a ParallelLMHead is touched; the lm_head hooks stay.

envs.compile_factors(): the knobs change the traced graph (int8 gather + rescale instead of
F.embedding), so they join the RADIANCE_FUSED_NORM_QUANT* entries in the torch.compile cache
key. Unset knobs add nothing.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
QUARK = SP / "vllm/model_executor/layers/quantization/quark/quark.py"
ENVS = SP / "vllm/envs.py"

INT4_BLOCK = '        # --- radiance (sly/patch_lmhead_int4.py): int4 lm_head, RADIANCE_LMHEAD_INT4=1 ---\n'

apply(QUARK,
      INT4_BLOCK,
      '        # --- radiance (sly/patch_embed_int8.py): int8/int4 embed_tokens, RADIANCE_EMBED_INT8=1 ---\n'
      '        try:\n'
      '            import radiance_embed_int8 as _radiance_embed_int8\n'
      '\n'
      '            _radiance_method = _radiance_embed_int8.quant_method_for(layer, prefix)\n'
      '            if _radiance_method is not None:\n'
      '                return _radiance_method\n'
      '        except Exception as _radiance_exc:  # never block model load on our own embedding\n'
      '            logger.warning_once("[radiance] int8 embedding unavailable: %r", _radiance_exc)\n'
      + INT4_BLOCK,
      '    # --- radiance (sly/patch_embed_int8.py): int8/int4 embed_tokens, RADIANCE_EMBED_INT8=1 ---',
      'quark: RadianceEmbedInt8 for VocabParallelEmbedding (RADIANCE_EMBED_INT8=1)')

A = '        if _radiance_k.startswith("RADIANCE_FUSED_NORM_QUANT"):\n'
apply(ENVS, A,
      '        # sly/patch_embed_int8.py: RADIANCE_EMBED_* change the embedding subgraph as well\n'
      '        if _radiance_k.startswith(("RADIANCE_FUSED_NORM_QUANT", "RADIANCE_EMBED_")):\n',
      'RADIANCE_EMBED_', 'envs: RADIANCE_EMBED_* in compile_factors')
