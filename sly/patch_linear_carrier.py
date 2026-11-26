#!/usr/bin/env python3
"""Let the fused-norm (q, scale) carrier through vLLM main's `apply_flattened`.

sly/patch_fused_norm_quant.py (RADIANCE_FUSED_NORM_QUANT=1, production) hands the W4A8 GEMMs a
`(q, scale)` tuple instead of a bf16 tensor. vLLM 0.34 passed the layer input straight to
`quant_method.apply`; main (1bdc437) wraps every call in `apply_flattened(fn, x)`, which reads
`x.ndim` first -> `AttributeError: 'tuple' object has no attribute 'ndim'` in profile_run
(Qwen3MoeMLP.down_proj). The carrier is always a 2D token batch, so it goes to `fn` as is.

NOOP on a tree without `apply_flattened` (vLLM <= 0.34). Runs after sly/patch_fused_norm_quant.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
LINEAR = SP / "vllm/model_executor/layers/linear.py"
LABEL = "linear: apply_flattened passes the radiance (q, scale) carrier"

if "def apply_flattened(" not in LINEAR.read_text():
    print(f"  SKIP  {LABEL}: no apply_flattened in this vLLM (the layers call quant_method.apply directly)")
else:
    A = "    if x.ndim <= 2 or not x.is_contiguous():\n"
    apply(LINEAR, A,
          "    if isinstance(x, tuple):  # radiance fused-norm carrier (sly/patch_linear_carrier.py)\n"
          "        return fn(x)\n" + A,
          "radiance fused-norm carrier (sly/patch_linear_carrier.py)", LABEL)
