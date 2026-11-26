#!/usr/bin/env python3
"""Qwen3.5 on ROCm keeps the legacy torch.compile model (vLLM main moved it to an eager one).

vLLM main@1bdc437 routes Qwen3.5 on ROCm to `vllm/models/qwen3_5/amd` (its own decoder forward, its own
`Qwen3_5RMSNorm`) and lists the architecture in `default_breakable_cudagraph_architectures()`, which turns
torch.compile OFF (mode NONE, breakable CUDA graphs). All radiance hooks sit in the legacy
`model_executor/models/qwen3_next.py` layer, so on that model `radiance::add_rms_quant` never ran and the
A-tiled prefill GEMMs lost their tiled activations: Qwen prod config -4..-6 % decode (four categories) and
-6 % prefill at 8k/16k against 1.13.4 (release check of 1.14.0, QFIX).

Two hunks: (1) vllm/models/qwen3_5/__init__.py picks the AMD model only with RADIANCE_QWEN35_AMD=1;
(2) vllm/config/vllm.py drops the Qwen3.5 architectures from the ROCm breakable-graph default under the same
switch. Other platforms and the other ROCm architectures (DeepSeek V4.1, HYV4) are untouched.
SKIP on a vLLM without `vllm/models/qwen3_5` (<= 0.34).
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
INIT = SP / "vllm/models/qwen3_5/__init__.py"
CONFIG = SP / "vllm/config/vllm.py"
ENV = 'environ.get("RADIANCE_QWEN35_AMD") == "1"'

if not INIT.parent.is_dir():
    print("  SKIP  qwen3_5 legacy: no vllm/models/qwen3_5 in this vLLM (the legacy model is the only one)")
else:
    apply(INIT, "elif current_platform.is_rocm():\n",
          f'elif current_platform.is_rocm() and __import__("os").{ENV}:  # radiance: legacy model by default\n',
          "# radiance: legacy model by default", "qwen3_5: ROCm uses the legacy model unless RADIANCE_QWEN35_AMD=1")
    apply(CONFIG, "            | _QWEN3_5_ARCHITECTURES\n        )\n",
          f"            | (_QWEN3_5_ARCHITECTURES if os.{ENV} else frozenset())"
          "  # radiance: legacy model keeps torch.compile\n        )\n",
          "# radiance: legacy model keeps torch.compile",
          "config: Qwen3.5 on ROCm defaults to torch.compile unless RADIANCE_QWEN35_AMD=1")
