#!/usr/bin/env python3
"""DFlash drafter with a compressed-tensors W4A16 qkv_proj (vLLM 0.29.0).

`syvai/Qwen3.8-27B-DFlash2-W4A16` ships its qkv_proj as compressed-tensors
`weight_packed` (+ `weight_scale`, `weight_shape`); the layer has NO raw
`.weight` attribute. vllm/model_executor/models/qwen3_dflash.py reads
`qkv_proj.weight` in two places to build the fused context-KV projection
(`_dflash_kv_weight_rows` and the load-time "deferred" decision in
`load_weights`) and crashes engine init with AttributeError.

Fix: treat "no raw .weight" exactly like the fp8 case that already exists
(patch_dflash_fused_kv_fp8): defer, then dequantise through the layer's own
forward on an identity matrix -- quant-method-agnostic, so it covers W4A16,
W8A8 and anything else compressed-tensors packs.

Ported from the vllm5 runtime bind-mount patch (vLLM 0.27.1) that ran the same
drafter on the INT4 target; here baked into the image (vllm7 has no
runtime-patch mounts). Verified 2026-09-15 on vllm7: k=7 DFlash2 on the MXFP4
target, single-stream 84 t/s / conc 8 229 t/s, see homelab plan
groovy-floating-twilight.md ("Lauf F").

Runs AFTER patch_dflash_fused_kv_fp8 and patch_dflash_w4 (anchors are the
post-patch text of those two).
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/models/qwen3_dflash.py"

apply(F,
      '    weight = qkv_proj.weight\n'
      '    if weight.dtype not in _DFLASH_FP8:\n'
      '        return weight[q_size:]\n'
      '    dtype = getattr(qkv_proj, "orig_dtype", torch.bfloat16)\n'
      '    eye = torch.eye(\n'
      '        qkv_proj.input_size_per_partition, dtype=dtype, device=weight.device\n'
      '    )\n',
      '    # --- SLY W4-PACKED: a quantised qkv_proj without a raw `.weight`\n'
      '    # (compressed-tensors W4A16: weight_packed) is handled like the fp8 case --\n'
      '    # dequantise through the layer\'s own forward on an identity matrix.\n'
      '    weight = getattr(qkv_proj, "weight", None)\n'
      '    if weight is not None and weight.dtype not in _DFLASH_FP8:\n'
      '        return weight[q_size:]\n'
      '    dtype = getattr(qkv_proj, "orig_dtype", torch.bfloat16)\n'
      '    device = weight.device if weight is not None else next(qkv_proj.parameters()).device\n'
      '    eye = torch.eye(\n'
      '        qkv_proj.input_size_per_partition, dtype=dtype, device=device\n'
      '    )\n',
      '    # --- SLY W4-PACKED: a quantised qkv_proj without a raw `.weight`',
      'dflash W4-packed: _dflash_kv_weight_rows without .weight')

apply(F,
      '        self._kv_source_attn = layers_attn\n'
      '        if layers_attn[0].qkv_proj.weight.dtype in _DFLASH_FP8:\n',
      '        self._kv_source_attn = layers_attn\n'
      '        # --- SLY W4-PACKED: no raw .weight (W4A16 packed) is deferred as well.\n'
      '        qkv_weight = getattr(layers_attn[0].qkv_proj, "weight", None)\n'
      '        if qkv_weight is None or qkv_weight.dtype in _DFLASH_FP8:\n',
      '        # --- SLY W4-PACKED: no raw .weight (W4A16 packed) is deferred as well.',
      'dflash W4-packed: load_weights defers packed qkv_proj')
