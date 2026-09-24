#!/usr/bin/env python3
"""Add RADIANCE_NVFP4_* to vLLM's torch.compile cache key (vLLM 0.29.0).

patch_nvfp4_mxfp4.py decides at load which linears of a compressed-tensors checkpoint leave the
stock schemes for the radiance W4A8 kernel (RADIANCE_NVFP4_MXFP4, _FP8_LAYERS, _BF16_LAYERS) and
what the lm_head becomes (_LMHEAD). Each of those changes the traced graph for the SAME checkpoint,
while envs.compile_factors() hashes VLLM_* only -- a knob flip would replay a stale AOT graph.
Same mechanism as the RADIANCE_FUSED_NORM_QUANT* / RADIANCE_EMBED_* entries; unset knobs add
nothing, so the Quark/INT4 cache keys are unchanged. Anchors on sly/patch_embed_int8.py's line, so
it runs after that one in the Dockerfile loop.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

ENVS = Path(sysconfig.get_paths()["purelib"]) / "vllm/envs.py"

apply(ENVS,
      '        if _radiance_k.startswith(("RADIANCE_FUSED_NORM_QUANT", "RADIANCE_EMBED_")):\n',
      '        # sly/patch_nvfp4_compile_key.py: RADIANCE_NVFP4_* pick the scheme per linear at load\n'
      '        if _radiance_k.startswith(("RADIANCE_FUSED_NORM_QUANT", "RADIANCE_EMBED_", "RADIANCE_NVFP4_")):\n',
      'RADIANCE_NVFP4_', 'envs: RADIANCE_NVFP4_* in compile_factors')
