#!/usr/bin/env python3
"""Inkling-2-Mini coda experts MXFP8 -> NVFP4 at load, opt-in source patch (lossy).

RADIANCE_INKLING2_CODA_FP4=1 (or a layer list) serves the checkpoint's MXFP8 expert layers (20-23) as
NVFP4: fewer weight bytes per decode step, the NVFP4 kernels already in the image.  Off = inert.

Two hooks: `Inkling2DecoderLayer.__init__` marks the MXFP8 layers the knob selects, and
`Inkling2Experts.process_weights_after_loading` converts the marked ones (module radiance_inkling2_coda_fp4).
NOOP on a vLLM without models/inkling2.  Idempotent.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
ROOT = SP / "vllm/models/inkling2"

if not ROOT.is_dir():
    print("  NOOP  inkling2 coda_fp4: this vLLM has no models/inkling2")
    raise SystemExit(0)

apply(
    ROOT / "nvidia/model.py",
    "        self.mlp = Inkling2MoE(config, expert_format, config.layer_visit_ids(layer_id))\n",
    "        self.mlp = Inkling2MoE(config, expert_format, config.layer_visit_ids(layer_id))\n"
    "        if torch.version.hip and expert_format == \"mxfp8\":  # opt-in RADIANCE_INKLING2_CODA_FP4\n"
    "            from radiance_inkling2_coda_fp4 import wants\n\n"
    "            self.mlp.experts.rad_requant = wants(layer_id)\n",
    "from radiance_inkling2_coda_fp4 import wants",
    "inkling2 coda_fp4: mark the MXFP8 expert layers the knob selects",
)
apply(
    ROOT / "nvidia/moe.py",
    "    def process_weights_after_loading(self) -> None:\n"
    "        if self.sm100 or (self.sm120 and not self.humming_nvfp4):\n",
    "    def process_weights_after_loading(self) -> None:\n"
    "        if getattr(self, \"rad_requant\", False):  # RADIANCE_INKLING2_CODA_FP4: MXFP8 -> NVFP4\n"
    "            from radiance_inkling2_coda_fp4 import requant_experts\n\n"
    "            requant_experts(self)\n"
    "        if self.sm100 or (self.sm120 and not self.humming_nvfp4):\n",
    "from radiance_inkling2_coda_fp4 import requant_experts",
    "inkling2 coda_fp4: convert the marked experts after loading",
)
