#!/usr/bin/env python3
"""Make vLLM's fused GDN MTP decode path available on gfx1201 (sly/gdn/radiance_gdn_decode.py).

qwen_gdn_linear_attn.py decides per layer, at construction, whether its forward takes the fused CUDA
decode path; one of the conditions is that `torch.ops._C.fused_gdn_decode_post_conv_mtp` exists. This
imports radiance_gdn_decode right after the module's own imports and registers the HIP port under that
name, so the decision sees it. Everything else (VLLM_GDN_DECODE_KERNEL, the shape/dtype gates, the
packed forward) is vLLM's own code, unchanged.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

GDN = Path(sysconfig.get_paths()["purelib"]) / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"

A = "MAX_FUSED_GDN_MTP_TOKENS = 8\n"
apply(GDN, A,
      "try:  # radiance (sly/patch_gdn_fused_decode.py): HIP port of fused_gdn_decode_post_conv_mtp\n"
      "    import radiance_gdn_decode as _radiance_gdn_decode\n"
      "\n"
      "    _radiance_gdn_decode.install()\n"
      "except Exception as _radiance_gdn_decode_exc:  # never block model import on our module\n"
      "    import sys as _rgd_sys\n"
      "\n"
      "    _rgd_sys.stderr.write(\n"
      "        f\"[radiance.gdn_decode] not installed: {_radiance_gdn_decode_exc!r}\\n\"\n"
      "    )\n"
      + A,
      "import radiance_gdn_decode as _radiance_gdn_decode",
      "gdn: register the HIP fused_gdn_decode_post_conv_mtp op")
