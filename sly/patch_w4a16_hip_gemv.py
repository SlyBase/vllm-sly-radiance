#!/usr/bin/env python3
"""Route the symmetric W4A16 low-M calls to the HIP GEMV (sly/w4a16/radiance_w4a16_gemv.hip).

rdna_hybrid_w4a16.py's registered custom op `rdna_hybrid_w4a16_apply` serves BOTH the W4A16 linear kernel
(RDNAHybridW4A16LinearKernel.apply_weights: the DFlash2 drafter, the int4 target) and the int4 lm_head
(RadianceLMHeadInt4.apply). Its impl sends M > MAX_SKINNY_BATCH_SIZE (5) to the Triton kernel; a DFlash2 step is
M = 8. This puts one hook at the top of the impl: sly/w4a16/radiance_w4a16_gemv_hip.try_w4a16_hip_gemv returns the
result for M 1..16 (symmetric, bf16, RADIANCE_W4A16_HIP_GEMV=1) or None, and None falls into the unchanged dispatch.

The hook lives inside the op implementation, which torch.compile / piecewise / full CUDA graphs already treat as one
opaque call: the fake impl and the registration are untouched, so tracing, the AOT cache key and the graph shapes
do not change with the knob.

Order: any time after sly/patch_w4a16_tiles (the two anchor on different lines of the impl, both orders apply).
The module is installed next to the other radiance_*.py files (radiance_w4a16_gemv_hip.py), the kernel as
radiance_w4a16_gemv.so; a missing module leaves the stock path.
RADIANCE_PATCH_TARGET_DIR overrides the site-packages dir.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
F = SP / "vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py"

apply(F,
      'def _rdna_hybrid_w4a16_apply_impl(\n',
      'try:  # --- radiance (sly/patch_w4a16_hip_gemv.py): pre-tiled HIP GEMV for symmetric W4A16 at M 1..16 ---\n'
      '    import radiance_w4a16_gemv_hip as _radiance_hip_gemv\n'
      'except Exception:  # never block the import of the stock kernel on it\n'
      '    _radiance_hip_gemv = None\n'
      '\n\n'
      'def _rdna_hybrid_w4a16_apply_impl(\n',
      '    import radiance_w4a16_gemv_hip as _radiance_hip_gemv',
      'rdna_hybrid_w4a16: HIP GEMV module import')

apply(F,
      '    import vllm._custom_ops as ops\n\n    M = x_2d.shape[0]\n',
      '    import vllm._custom_ops as ops\n\n'
      '    # radiance (RADIANCE_W4A16_HIP_GEMV=1): symmetric W4A16 at M 1..16 on the pre-tiled HIP GEMV; None = stock path\n'
      '    if _radiance_hip_gemv is not None:\n'
      '        _radiance_y = _radiance_hip_gemv.try_w4a16_hip_gemv(x_2d, w_q, w_s, w_zp, group_size, bias)\n'
      '        if _radiance_y is not None:\n'
      '            return _radiance_y\n\n'
      '    M = x_2d.shape[0]\n',
      '_radiance_hip_gemv.try_w4a16_hip_gemv(x_2d, w_q, w_s, w_zp, group_size, bias)',
      'rdna_hybrid_w4a16: HIP GEMV hook at the top of the apply op')

apply(F,
      '        kernel_config = get_current_vllm_config().kernel_config\n',
      '        if _radiance_hip_gemv is not None:\n'
      '            _hq, _hs, _hz = self._get_weight_params(layer)\n'
      '            _radiance_hip_gemv.prepack_weights(\n'
      '                _hq.data, _hs.data, _hz.data if _hz is not None else None, c.group_size, getattr(layer, "prefix", "")\n'
      '            )\n'
      '        kernel_config = get_current_vllm_config().kernel_config\n',
      '_radiance_hip_gemv.prepack_weights(',
      'rdna_hybrid_w4a16: prepack HIP GEMV tiles at weight load before KV sizing')
