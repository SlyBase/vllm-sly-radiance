#!/usr/bin/env python3
"""Skip the DFlash2 drafter's work on old prefill chunks (RADIANCE_DFLASH_PREFILL_WINDOW_SKIP=1).

One hook at the top of `DFlashSpeculator.propose` (vllm/v1/worker/gpu/spec_decode/dflash/speculator.py):
when `radiance_dflash_skip.skip_context` says every request is on a prefill chunk that ends more
than `sliding_window + margin` tokens before its prompt end, return before `combine_hidden_states`
(`fc`), the context-K/V fill and the draft step. Why that is lossless, the margin and the gates
(prefix caching, non-sliding layers, DP) are in sly/radiance_dflash_skip.py. Default off (the hook
reads the knob once and returns False). RADIANCE_PATCH_TARGET_DIR overrides the site-packages root
(sly/tests/test_dflash_prefill_skip_cpu.py); the Dockerfile build never sets it.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

TARGET = (
    Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
    / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "dflash" / "speculator.py"
)

apply(TARGET,
      "        num_reqs = input_batch.num_reqs\n        num_target_tokens = input_batch.num_tokens\n",
      "        num_reqs = input_batch.num_reqs\n"
      "        # radiance (sly/patch_dflash_prefill_skip.py): old prefill chunks need no drafter work.\n"
      "        from radiance_dflash_skip import skip_context\n"
      "\n"
      "        if skip_context(self, input_batch, dummy_run, is_profile):\n"
      "            return self.draft_tokens[:num_reqs]\n"
      "        num_target_tokens = input_batch.num_tokens\n",
      "from radiance_dflash_skip import skip_context",
      "dflash: skip drafter work on old prefill chunks")
