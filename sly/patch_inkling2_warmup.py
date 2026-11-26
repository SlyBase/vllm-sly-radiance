#!/usr/bin/env python3
"""Inkling-2-Mini prefill-shape warmup hook (module: sly/inkling2/radiance_inkling2_warmup.py).

One hook: `Worker._compile_or_warm_up_model` calls `warmup_prefill_ladder` right after vLLM's `warmup_kernels`, before the
CUDA graph capture. Inert for any model but Inkling (the module checks `model_type`) and with
RADIANCE_INKLING2_PREFILL_WARMUP=0. NOOP on a vLLM without models/inkling2. Idempotent.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])

if not (SP / "vllm/models/inkling2").is_dir():
    print("  NOOP  inkling2 prefill warmup: this vLLM has no models/inkling2")
    raise SystemExit(0)

ANCHOR = "            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)\n"
apply(
    SP / "vllm/v1/worker/gpu_worker.py",
    ANCHOR,
    ANCHOR + "            from radiance_inkling2_warmup import warmup_prefill_ladder\n\n"
    "            warmup_prefill_ladder(self)  # Inkling only: real prompt lengths, before the API is up\n",
    "warmup_prefill_ladder(self)",
    "inkling2 prefill warmup hook",
)
