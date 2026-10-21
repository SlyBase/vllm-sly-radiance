#!/usr/bin/env python3
"""Skip the DFlash2 drafter's work on old prefill chunks (RADIANCE_DFLASH_PREFILL_WINDOW_SKIP=1).

One hook at the top of `DFlashSpeculator.propose` (vllm/v1/worker/gpu/spec_decode/dflash/speculator.py):
when `radiance_dflash_skip.skip_context` says every request is on a prefill chunk that ends more
than `sliding_window + margin` tokens before its prompt end, return before `combine_hidden_states`
(`fc`), the context-K/V fill and the draft step. Why that is lossless, the margin and the gates
(prefix caching, non-sliding layers, DP) are in sly/radiance_dflash_skip.py. Two more hunks serve
RADIANCE_DFLASH_PREFILL_WINDOW_SKIP_APC=1 (prefix caching on): `KVCacheManager.allocate_slots` notes
each chunk before it is cached and `SingleTypeKVCacheManager.cache_blocks` drops the sliding-window
blocks that touch a possibly skipped position from the hash map, so no hit can read an unwritten
drafter slot. Default off (every hook reads the knobs once and returns the stock value). RADIANCE_PATCH_TARGET_DIR overrides the site-packages root
(sly/tests/test_dflash_prefill_skip_cpu.py); the Dockerfile build never sets it.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SITE = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"]) / "vllm" / "v1"
TARGET = SITE / "worker" / "gpu" / "spec_decode" / "dflash" / "speculator.py"
CORE = SITE / "core"

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

apply(CORE / "kv_cache_manager.py",
      "        self.coordinator.cache_blocks(request, num_tokens_to_cache)\n\n        return self.create_kv_cache_blocks(new_blocks)",
      "        # radiance (sly/patch_dflash_prefill_skip.py): record chunks whose drafter K/V may be skipped.\n"
      "        from radiance_dflash_skip import note_chunk\n"
      "\n"
      "        note_chunk(self, request, total_computed_tokens + num_new_tokens)\n"
      "        self.coordinator.cache_blocks(request, num_tokens_to_cache)\n\n"
      "        return self.create_kv_cache_blocks(new_blocks)",
      "note_chunk(self, request",
      "dflash: note possibly skipped chunks before they are cached")

apply(CORE / "single_type_kv_cache_manager.py",
      "            dcp_world_size=self.dcp_world_size,\n        )\n        self.block_pool.cache_full_blocks(\n"
      "            request=request,\n            blocks=self.req_to_blocks[request.request_id],\n            num_cached_blocks=num_cached_blocks,",
      "            dcp_world_size=self.dcp_world_size,\n        )\n"
      "        # radiance (sly/patch_dflash_prefill_skip.py): blocks with unwritten drafter K/V never hit.\n"
      "        from radiance_dflash_skip import mask_unwritten\n"
      "\n"
      "        block_mask = mask_unwritten(self, request, num_cached_blocks, num_full_blocks, block_mask)\n"
      "        self.block_pool.cache_full_blocks(\n"
      "            request=request,\n            blocks=self.req_to_blocks[request.request_id],\n            num_cached_blocks=num_cached_blocks,",
      "mask_unwritten(self, request",
      "dflash: keep skipped drafter blocks out of the prefix cache")
