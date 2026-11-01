#!/usr/bin/env python3
"""Nucleus Speculative Decoding (NSD, arxiv 2610.07822) in vLLM's spec-decode rejection kernel.

NSD widens the verifier's acceptance rule for a drafted token x: x is accepted when the
standard test passes (u <= min(1, p(x)/q(x))) OR x lies in the target nucleus S_theta(p), i.e.
the mass of the tokens with a strictly higher target logit than x is <= theta. The residual
distribution and the bonus token are unchanged. Reference: base_evaluator.py of
EIT-NLP/Nucleus-Speculative-Decoding (cleanNSD: `standard_accept_mask | nucleus_mask`).

Where: vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py (V2 model runner, the path the
production Qwen3.8 + DFlash2 config runs). The target logits that reach `rejection_sample` are
already post temperature / min-p / top-k / top-p, i.e. the distribution that is actually
sampled, which is the p the rule needs. Deterministic drafts (point-mass q) and probabilistic
drafts both go through the same Leviathan branch; greedy rows, synthetic-rate mode and block
verification are not touched.

How (graph-safe, no host sync, no sort of the vocabulary):
  1. `_nsd_nucleus_mass_kernel`, grid (requests, vocab blocks), after the stock per-block
     logsumexp stats: per draft step, the block's partial of sum(exp(logit - lse) where
     logit > logit[x]). One streaming pass over the logits, no sort.
  2. In `_rejection_kernel` the Leviathan branch ORs `sum(partials) <= theta` into `accepted`
     (constexpr USE_NSD: with the knob off the NSD branch is compiled out; the kernel signature gains two args, so the Triton cache key differs from stock but the arithmetic is the same).

RADIANCE_NSD         0|1, default 0 (off: nothing extra is launched, the NSD branch is compiled out).
RADIANCE_NSD_THETA   nucleus mass in [0, 1), default 0.95. One module-level read; both values
                     are per-process, they cannot change mid-serve.
Mutually exclusive with RADIANCE_CACTUS_DELTA (sly/patch_cactus.py): the paper's single-step
bound does not cover a union of both rules, so setting both fails at import, loudly.

The anchors stay clear of the Cactus hunks (either patch can come first in the loop).
Idempotent; fatal on drift. RADIANCE_PATCH_TARGET_DIR overrides the site-packages root
(sly/tests/test_nsd_cpu.py applies the patch to a scratch copy; the Dockerfile build never sets it).
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

TARGET = (
    Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
    / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "rejection_sampler_utils.py"
)

# 1. env knobs + exclusivity check, after the module's imports
apply(TARGET,
      "from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_block_argmax\n",
      "from vllm.v1.worker.gpu.sample.watermark import philox_gumbel_block_argmax\n"
      "\n"
      "# radiance nsd: env knobs (sly/patch_nsd.py)\n"
      "import os as _nsd_os\n"
      "\n"
      "NSD_ENABLED = _nsd_os.environ.get(\"RADIANCE_NSD\", \"0\") == \"1\"\n"
      "NSD_THETA = float(_nsd_os.environ.get(\"RADIANCE_NSD_THETA\", \"0.95\"))\n"
      "if NSD_ENABLED:\n"
      # The Cactus knob name is split on purpose: patch_cactus.py uses the whole name as its
      # "already applied" sentinel, and a literal here would make it skip its own env-read hunk.
      # The default is the Cactus patch's own value (0.5, ON in this image): reading 0 here let
      # RADIANCE_NSD=1 run the union Cactus(0.5) OR NSD, which the paper's bound does not cover.
      "    if float(_nsd_os.environ.get(\"RADIANCE_CACTUS\" + \"_DELTA\", \"0.5\") or 0) != 0.0:\n"
      "        raise RuntimeError(\n"
      "            \"RADIANCE_NSD=1 and the Cactus rule are mutually exclusive acceptance rules; \"\n"
      # split like the read above: a literal name here would trip patch_cactus.py's sentinel.
      "            \"set RADIANCE_CACTUS\" + \"_DELTA=0 to switch Cactus off\"\n"
      "        )\n"
      "    if not 0.0 <= NSD_THETA < 1.0:\n"
      "        raise RuntimeError(f\"RADIANCE_NSD_THETA must be in [0, 1), got {NSD_THETA}\")\n",
      "# radiance nsd: env knobs",
      "nsd: env knobs + exclusivity")

# 2. the nucleus-mass kernel, ahead of _rejection_kernel
apply(TARGET,
      "@triton.jit\ndef _rejection_kernel(\n",
      "@triton.jit\n"
      "def _nsd_nucleus_mass_kernel(\n"
      "    # radiance nsd: per-block partial of the target mass above the draft token (sly/patch_nsd.py)\n"
      "    # [num_logits, num_blocks]\n"
      "    nsd_partial_ptr,\n"
      "    nsd_partial_stride,\n"
      "    # [num_logits, V]\n"
      "    target_logits_ptr,\n"
      "    target_logits_stride,\n"
      "    # [num_logits, num_blocks]\n"
      "    target_local_max_ptr,\n"
      "    target_local_max_stride,\n"
      "    # [num_logits, num_blocks]\n"
      "    target_local_sumexp_ptr,\n"
      "    target_local_sumexp_stride,\n"
      "    # [num_logits]\n"
      "    draft_sampled_ptr,\n"
      "    # [num_reqs + 1]\n"
      "    cu_num_logits_ptr,\n"
      "    # [num_reqs]\n"
      "    idx_mapping_ptr,\n"
      "    # [max_num_reqs]\n"
      "    temp_ptr,\n"
      "    vocab_size,\n"
      "    vocab_num_blocks,\n"
      "    BLOCK_SIZE: tl.constexpr,\n"
      "    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,\n"
      "):\n"
      "    req_idx = tl.program_id(0)\n"
      "    block_idx = tl.program_id(1)\n"
      "    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)\n"
      "    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)\n"
      "    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)\n"
      "    num_draft_tokens = end_idx - start_idx - 1\n"
      "    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)\n"
      "    if temp == 0.0:\n"
      "        # Greedy rows keep the argmax test.\n"
      "        return\n"
      "\n"
      "    offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
      "    mask = offsets < vocab_size\n"
      "    for i in range(num_draft_tokens):\n"
      "        logit_idx = start_idx + i\n"
      "        draft_token = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)\n"
      "        # -1 placeholders end verification; the rejection kernel never reads their partials.\n"
      "        if draft_token >= 0:\n"
      "            token_logit = tl.load(\n"
      "                target_logits_ptr + logit_idx * target_logits_stride + draft_token\n"
      "            ).to(tl.float32)\n"
      "            lse = _compute_global_logsumexp(\n"
      "                target_local_max_ptr,\n"
      "                target_local_max_stride,\n"
      "                target_local_sumexp_ptr,\n"
      "                target_local_sumexp_stride,\n"
      "                logit_idx,\n"
      "                vocab_num_blocks,\n"
      "                PADDED_VOCAB_NUM_BLOCKS,\n"
      "            )\n"
      "            logits = tl.load(\n"
      "                target_logits_ptr + logit_idx * target_logits_stride + offsets,\n"
      "                mask=mask,\n"
      "                other=float(\"-inf\"),\n"
      "            ).to(tl.float32)\n"
      "            # Strictly higher logit only (ties are not 'above'), as in the reference.\n"
      "            mass = tl.sum(tl.where(logits > token_logit, tl.exp(logits - lse), 0.0), axis=0)\n"
      "            # A token the sampler has already removed (-inf) is never in the nucleus.\n"
      "            mass = tl.where(token_logit > float(\"-inf\"), mass, 2.0)\n"
      "            tl.store(nsd_partial_ptr + logit_idx * nsd_partial_stride + block_idx, mass)\n"
      "\n\n"
      "@triton.jit\ndef _rejection_kernel(\n",
      "def _nsd_nucleus_mass_kernel",
      "nsd: nucleus-mass kernel")

# 3. _rejection_kernel signature: runtime args + constexpr (all passed positionally at the launch)
apply(TARGET,
      "    local_residual_mass_ptr,\n"
      "    local_residual_mass_stride,\n"
      "    vocab_num_blocks,\n",
      "    local_residual_mass_ptr,\n"
      "    local_residual_mass_stride,\n"
      "    # radiance nsd: kernel args (sly/patch_nsd.py)\n"
      "    # [num_logits, num_blocks]\n"
      "    nsd_partial_ptr,\n"
      "    nsd_partial_stride,\n"
      "    nsd_theta,\n"
      "    USE_NSD: tl.constexpr,\n"
      "    vocab_num_blocks,\n",
      "# radiance nsd: kernel args",
      "nsd: kernel signature")

# 4. Leviathan branch: OR the nucleus test into `accepted`
apply(TARGET,
      "                verifying = accepted\n"
      "                accepted_length += accepted\n"
      "                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)\n"
      "\n"
      "    tl.store(rejected_steps_ptr + req_idx, accepted_length)\n",
      "                if USE_NSD:\n"
      "                    # radiance nsd: accept a draft token inside the target nucleus (sly/patch_nsd.py)\n"
      "                    nsd_blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)\n"
      "                    nsd_mass = tl.sum(\n"
      "                        tl.load(\n"
      "                            nsd_partial_ptr + logit_idx * nsd_partial_stride + nsd_blocks,\n"
      "                            mask=nsd_blocks < vocab_num_blocks,\n"
      "                            other=0.0,\n"
      "                        ),\n"
      "                        axis=0,\n"
      "                    )\n"
      "                    accepted = accepted | (nsd_mass <= nsd_theta)\n"
      "                verifying = accepted\n"
      "                accepted_length += accepted\n"
      "                tl.store(sampled_ptr + req_idx * sampled_stride + i, draft_sampled)\n"
      "\n"
      "    tl.store(rejected_steps_ptr + req_idx, accepted_length)\n",
      "# radiance nsd: accept a draft token",
      "nsd: Leviathan acceptance")

# 5. wrapper: launch the mass kernel after the stock stats, before the rejection kernel
apply(TARGET,
      "    # Sample up until the first rejected/bonus token, and store\n"
      "    # the step.\n",
      "    # radiance nsd: nucleus mass of every draft token (sly/patch_nsd.py). Off (default): nothing\n"
      "    # launched, USE_NSD=False compiles the stock kernel. Not for block verification / synthetic rates.\n"
      "    use_nsd = (\n"
      "        NSD_ENABLED and not use_block_verification and synthetic_conditional_rates is None\n"
      "    )\n"
      "    nsd_partial = None\n"
      "    if use_nsd:\n"
      "        nsd_partial = target_logits.new_empty(\n"
      "            num_logits, vocab_num_blocks, dtype=torch.float32\n"
      "        )\n"
      "        _nsd_nucleus_mass_kernel[(num_reqs, vocab_num_blocks)](\n"
      "            nsd_partial,\n"
      "            nsd_partial.stride(0),\n"
      "            target_logits,\n"
      "            target_logits.stride(0),\n"
      "            target_local_max,\n"
      "            target_local_max.stride(0),\n"
      "            target_local_sumexp,\n"
      "            target_local_sumexp.stride(0),\n"
      "            draft_sampled,\n"
      "            cu_num_logits,\n"
      "            idx_mapping,\n"
      "            temperature,\n"
      "            vocab_size,\n"
      "            vocab_num_blocks,\n"
      "            BLOCK_SIZE=VOCAB_BLOCK_SIZE,\n"
      "            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,\n"
      "        )\n"
      "\n"
      "    # Sample up until the first rejected/bonus token, and store\n"
      "    # the step.\n",
      "# radiance nsd: nucleus mass of every draft token",
      "nsd: kernel launch (mass)")

# 6. rejection kernel launch args (before vocab_num_blocks, so independent of the Cactus launch arg)
apply(TARGET,
      "        local_residual_mass.stride(0) if local_residual_mass is not None else 0,\n"
      "        vocab_num_blocks,\n",
      "        local_residual_mass.stride(0) if local_residual_mass is not None else 0,\n"
      "        nsd_partial,  # radiance nsd: launch args (sly/patch_nsd.py)\n"
      "        nsd_partial.stride(0) if nsd_partial is not None else 0,\n"
      "        NSD_THETA,\n"
      "        use_nsd,\n"
      "        vocab_num_blocks,\n",
      "# radiance nsd: launch args",
      "nsd: kernel launch (rejection)")
