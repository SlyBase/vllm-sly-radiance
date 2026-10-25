#!/usr/bin/env python3
"""Run the DFlash2 drafter only over the rows its window reads (RADIANCE_DFLASH_PREFILL_PARTIAL_SKIP=1).

Row-granular companion of sly/patch_dflash_prefill_skip.py, in the same `DFlashSpeculator.propose`
(vllm/v1/worker/gpu/spec_decode/dflash/speculator.py), anchored after that patch's hook (apply it
first). `radiance_dflash_skip.context_rows` returns (r0, draft) for a batch of one request in prefill:

  1. a whole old chunk (r0 == num_tokens) returns before `fc`, as the whole skip does;
  2. `fc` (`combine_hidden_states`) runs over rows r0.. only, into `self.hidden_states[r0:]`;
  3. the context slot mappings of rows < r0 are PAD_SLOT_ID, so neither the eager context-K/V fill
     nor the FULL graph's own context rows store K/V from the stale hidden states below r0;
  4. the eager context-K/V fill starts at r0 (FULL: at max(num_context, r0));
  5. draft == False (not the prompt's last chunk; the scheduler drops its draft tokens): store the
     context rows the skipped FULL replay would have stored, eagerly and with the same M, and return
     before the draft step.

Rows >= r0 go through the same GEMM tiles as the whole chunk (r0 keeps >= 513 rows), so their K/V are
bit-identical; the final chunk's draft reads only those rows. Why the cut is lossless, and the gates
(prefix caching, non-sliding layers, DP), are in sly/radiance_dflash_skip.py. Default off: every hunk
takes the stock path when `context_rows` returns None. RADIANCE_PATCH_TARGET_DIR overrides the
site-packages root (sly/tests/test_dflash_prefill_partial_cpu.py); the Dockerfile build never sets it.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

TARGET = (
    Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
    / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "dflash" / "speculator.py"
)

# 1. the decision, right after the whole skip's hook
apply(TARGET,
      "        if skip_context(self, input_batch, dummy_run, is_profile):\n"
      "            return self.draft_tokens[:num_reqs]\n"
      "        num_target_tokens = input_batch.num_tokens\n",
      "        if skip_context(self, input_batch, dummy_run, is_profile):\n"
      "            return self.draft_tokens[:num_reqs]\n"
      "        # radiance (sly/patch_dflash_prefill_partial.py): prefill rows below the drafter's window\n"
      "        # get no fc / context K/V, and a chunk that is not the prompt's last no draft step.\n"
      "        from radiance_dflash_skip import context_rows\n"
      "\n"
      "        _rad_rows = context_rows(self, input_batch, dummy_run, is_profile)\n"
      "        _rad_r0 = _rad_rows[0] if _rad_rows else 0\n"
      "        if _rad_r0 >= input_batch.num_tokens:\n"
      "            return self.draft_tokens[:num_reqs]\n"
      "        num_target_tokens = input_batch.num_tokens\n",
      "from radiance_dflash_skip import context_rows",
      "dflash: partial skip, decision")

# 2. fc over rows r0.. only
apply(TARGET,
      "        if aux_hidden_states:\n"
      "            hidden_states = self.model.combine_hidden_states(\n"
      "                torch.cat(aux_hidden_states, dim=-1)\n"
      "            )\n"
      "        else:\n"
      "            hidden_states = last_hidden_states\n"
      "        self.hidden_states[:num_target_tokens].copy_(hidden_states[:num_target_tokens])\n",
      "        if aux_hidden_states:\n"
      "            # radiance partial skip: fc over the rows the drafter's window can read\n"
      "            hidden_states = self.model.combine_hidden_states(\n"
      "                torch.cat([h[_rad_r0:num_target_tokens] for h in aux_hidden_states], dim=-1)\n"
      "                if _rad_r0\n"
      "                else torch.cat(aux_hidden_states, dim=-1)\n"
      "            )\n"
      "        else:\n"
      "            hidden_states = last_hidden_states[_rad_r0:]\n"
      "        self.hidden_states[_rad_r0:num_target_tokens].copy_(\n"
      "            hidden_states[: num_target_tokens - _rad_r0]\n"
      "        )\n",
      "# radiance partial skip: fc over the rows",
      "dflash: partial skip, fc rows")

# 3. no K/V stored from the stale rows below r0 (eager fill and the FULL graph's context rows)
apply(TARGET,
      "                self.sample_from_anchor,\n"
      "            )\n"
      "\n"
      "        batch_sync, num_batch_tokens = (\n",
      "                self.sample_from_anchor,\n"
      "            )\n"
      "        if _rad_r0:\n"
      "            # radiance partial skip: rows below r0 hold stale hidden states, store no K/V for them\n"
      "            self._context_slot_mappings[:, :_rad_r0].fill_(PAD_SLOT_ID)\n"
      "\n"
      "        batch_sync, num_batch_tokens = (\n",
      "# radiance partial skip: rows below r0",
      "dflash: partial skip, pad stale rows")

# 4. + 5. context K/V from r0 on; a non-final chunk returns before the draft step
apply(TARGET,
      "                # Prefill context beyond the graph's rows is stored before replay.\n"
      "                self._precompute_context_kv(num_context, num_target_tokens)\n"
      "        else:\n"
      "            self._precompute_context_kv(0, num_target_tokens, dummy_run)\n",
      "                # Prefill context beyond the graph's rows is stored before replay.\n"
      "                self._precompute_context_kv(max(num_context, _rad_r0), num_target_tokens)\n"
      "        else:\n"
      "            self._precompute_context_kv(_rad_r0, num_target_tokens, dummy_run)\n"
      "        if _rad_rows and not _rad_rows[1]:\n"
      "            # radiance partial skip: not the prompt's last chunk, its draft would be dropped. Store\n"
      "            # the context rows the skipped FULL replay would have stored (same rows, same M).\n"
      "            if batch_desc.cg_mode == CUDAGraphMode.FULL and _rad_r0 < num_context:\n"
      "                self._precompute_context_kv(0, num_context)\n"
      "            return self.draft_tokens[:num_reqs]\n",
      "# radiance partial skip: not the prompt's last chunk",
      "dflash: partial skip, context rows + no draft on non-final chunks")
