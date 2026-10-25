#!/usr/bin/env python3
"""Cactus acceptance transform in vLLM's spec-decode rejection kernel.

Cactus (arxiv 2604.04987) raises the verifier's acceptance threshold for the drafted
token from the target probability q_n to h_n = clamp(q_n + sqrt(2*delta*q_n*(1-q_n)),
0, 1), trading a bounded KL budget (delta) for a higher acceptance rate. delta=0 is
the identity, so RADIANCE_CACTUS_DELTA=0 is bit-exact stock (the image default is 0.5).

The vLLM 0.31 kernel does the acceptance test in the Leviathan branch of
_rejection_kernel: `target_logprob > tl.log(u) + draft_logprob`, i.e. accept iff
u < q_n/p_n. The hunk swaps the threshold for log(h_n), with a where-guard that keeps
the stock comparison bit-exact at delta=0 (an exp/log round-trip is not guaranteed
bit-identical in fp32). Greedy requests keep their own branch (argmax agreement);
synthetic-rate and block-verification modes are untouched.

RADIANCE_CACTUS_DELTA   divergence budget (default 0.5; set 0 for the bit-exact stock
                        comparison). One module-level read; the value is a per-process
                        compile constant of the Triton kernel, so it cannot change
                        mid-serve.
RADIANCE_CACTUS_FADE_START  length fade (issue #164): the request's delta falls linearly from
RADIANCE_CACTUS_FADE_END    `delta` at START output tokens to 0 at END (defaults 4096 / 16384);
                            END <= START = hard cut at START, END = 0 = no fade (1.8.1 behaviour).

Only the V2 model runner path is patched: vLLM 0.31 defaults ROCm Qwen3.8 + dflash
to the V2 runner (vllm/config/vllm.py), and the V1 sample/rejection_sampler.py is a
separate implementation this image's production config never uses.

Idempotent; fatal on drift. RADIANCE_PATCH_TARGET_DIR overrides the site-packages
root (sly/tests/test_cactus_cpu.py applies the patch to a scratch copy; the
Dockerfile build never sets it).
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply, apply_any

TARGET_CALL = (
    Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
    / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "rejection_sampler.py"
)
TARGET = (
    Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
    / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "rejection_sampler_utils.py"
)

apply(TARGET,
      "import torch\n\nfrom vllm.triton_utils import tl, tldevice, triton\n",
      "import os\n\nimport torch\n\nfrom vllm.triton_utils import tl, tldevice, triton\n"
      "\n"
      "# radiance cactus: env knob (sly/patch_cactus.py)\n"
      "CACTUS_DELTA = float(os.environ.get(\"RADIANCE_CACTUS_DELTA\", \"0.5\"))\n",
      "RADIANCE_CACTUS_DELTA",
      "cactus: import + env read")

# Length fade (issue #164). Long sampled generations (>~16k output tokens) entered repetition loops ~3x as
# often with delta 0.5 as with the stock rule; Cactus only in the first 4096 output tokens removed them
# (0/16 vs 11/32 runs). The fade keeps short outputs untouched and most of the gain of long ones.
apply(TARGET,
      "CACTUS_DELTA = float(os.environ.get(\"RADIANCE_CACTUS_DELTA\", \"0.5\"))\n",
      "CACTUS_DELTA = float(os.environ.get(\"RADIANCE_CACTUS_DELTA\", \"0.5\"))\n"
      "# radiance cactus fade: env knobs (sly/patch_cactus.py, issue #164)\n"
      "CACTUS_FADE_START = int(os.environ.get(\"RADIANCE_CACTUS_FADE_START\", \"4096\"))\n"
      "CACTUS_FADE_END = int(os.environ.get(\"RADIANCE_CACTUS_FADE_END\", \"16384\"))\n",
      "RADIANCE_CACTUS_FADE_START",
      "cactus fade: env knobs")

apply(TARGET,
      "    vocab_num_blocks,\n"
      "    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,\n"
      "    HAS_DRAFT_LOGITS: tl.constexpr,\n"
      "    SYNTHETIC_MODE: tl.constexpr,\n"
      "    USE_BLOCK_VERIFICATION: tl.constexpr,\n"
      "):\n",
      "    vocab_num_blocks,\n"
      "    cactus_delta,  # radiance cactus: kernel arg (sly/patch_cactus.py)\n"
      "    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,\n"
      "    HAS_DRAFT_LOGITS: tl.constexpr,\n"
      "    SYNTHETIC_MODE: tl.constexpr,\n"
      "    USE_BLOCK_VERIFICATION: tl.constexpr,\n"
      "):\n",
      "# radiance cactus: kernel arg",
      "cactus: kernel signature")

apply(TARGET,
      "    cactus_delta,  # radiance cactus: kernel arg (sly/patch_cactus.py)\n",
      "    cactus_delta,  # radiance cactus: kernel arg (sly/patch_cactus.py)\n"
      "    cactus_prompt_len_ptr,  # radiance cactus fade: kernel args\n"
      "    cactus_fade_start,\n"
      "    cactus_fade_end,\n",
      "# radiance cactus fade: kernel args",
      "cactus fade: kernel signature")

# Threshold. Shape 2 is the 1.5.1-1.8.1 hunk, so a launcher re-run on an older image upgrades it.
_STOCK_TEST = (
    "                    # Probability ratio test: p(x) > u * q(x)\n"
    "                    # Equivalent log form: log_p(x) > log(u) + log_q(x)\n"
)
_THRESHOLD = (
    _STOCK_TEST +
    "                    # radiance cactus: threshold (sly/patch_cactus.py): q_n -> h_n,\n"
    "                    # delta=0 stays the stock comparison, bit-exact.\n"
    "                    # radiance cactus fade: delta falls to 0 over the output length (#164)\n"
    "                    out_len = (pos - tl.load(cactus_prompt_len_ptr + req_state_idx)).to(tl.float32)\n"
    "                    fade = tl.minimum(tl.maximum((cactus_fade_end - out_len)\n"
    "                                      / tl.maximum((cactus_fade_end - cactus_fade_start) * 1.0, 1.0), 0.0), 1.0)\n"
    "                    req_delta = tl.where(cactus_fade_end > 0, cactus_delta * fade, cactus_delta)\n"
    "                    q_n = tl.exp(target_logprob)\n"
    "                    h_n = tl.minimum(\n"
    "                        tl.maximum(q_n + tl.sqrt(2.0 * req_delta * q_n * (1.0 - q_n)), 0.0), 1.0\n"
    "                    )\n"
    "                    log_h_n = tl.where(req_delta == 0.0, target_logprob, tl.log(h_n))\n"
    "                    accepted = log_h_n > tl.log(u) + draft_logprob\n"
)
apply_any(TARGET,
      [
      (_STOCK_TEST + "                    accepted = target_logprob > tl.log(u) + draft_logprob\n", _THRESHOLD),
      (_STOCK_TEST +
       "                    # radiance cactus: threshold (sly/patch_cactus.py): q_n -> h_n,\n"
       "                    # delta=0 stays the stock comparison, bit-exact.\n"
       "                    q_n = tl.exp(target_logprob)\n"
       "                    h_n = tl.minimum(\n"
       "                        tl.maximum(q_n + tl.sqrt(2.0 * cactus_delta * q_n * (1.0 - q_n)), 0.0), 1.0\n"
       "                    )\n"
       "                    log_h_n = tl.where(cactus_delta == 0.0, target_logprob, tl.log(h_n))\n"
       "                    accepted = log_h_n > tl.log(u) + draft_logprob\n",
       _THRESHOLD)],
      "# radiance cactus fade: delta falls",
      "cactus: Leviathan acceptance")

# vLLM 0.32.0 inserts watermarking_skip_mask between watermarking and watermark_key, so the
# cactus_delta wrapper arg anchors on the skip-mask + watermark_key tail (unique in the file).
# The old shape stays for the launchers patching older images.
apply_any(TARGET,
      [
      (
      "    watermarking_skip_mask: torch.Tensor | None = None,\n"
      "    watermark_key: int | None = None,\n"
      ") -> tuple[torch.Tensor, torch.Tensor]:\n",
      "    watermarking_skip_mask: torch.Tensor | None = None,\n"
      "    watermark_key: int | None = None,\n"
      "    cactus_delta: float = CACTUS_DELTA,  # radiance cactus: wrapper arg (sly/patch_cactus.py)\n"
      ") -> tuple[torch.Tensor, torch.Tensor]:\n"
      ),
      (
      "    watermarking: torch.Tensor | None = None,\n"
      "    watermark_key: int | None = None,\n"
      ") -> tuple[torch.Tensor, torch.Tensor]:\n",
      "    watermarking: torch.Tensor | None = None,\n"
      "    watermark_key: int | None = None,\n"
      "    cactus_delta: float = CACTUS_DELTA,  # radiance cactus: wrapper arg (sly/patch_cactus.py)\n"
      ") -> tuple[torch.Tensor, torch.Tensor]:\n"
      )],
      "# radiance cactus: wrapper arg",
      "cactus: rejection_sample signature")

apply(TARGET,
      "        vocab_num_blocks,\n"
      "        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,\n",
      "        vocab_num_blocks,\n"
      "        cactus_delta,  # radiance cactus: launch arg (sly/patch_cactus.py)\n"
      "        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,\n",
      "# radiance cactus: launch arg",
      "cactus: kernel launch")

# Wrapper arg: the per-slot prompt lengths. None (a caller that does not pass them) = no fade.
apply(TARGET,
      "    cactus_delta: float = CACTUS_DELTA,  # radiance cactus: wrapper arg (sly/patch_cactus.py)\n",
      "    cactus_delta: float = CACTUS_DELTA,  # radiance cactus: wrapper arg (sly/patch_cactus.py)\n"
      "    cactus_prompt_len: torch.Tensor | None = None,  # radiance cactus fade: wrapper arg\n",
      "# radiance cactus fade: wrapper arg",
      "cactus fade: rejection_sample signature")

apply(TARGET,
      "        cactus_delta,  # radiance cactus: launch arg (sly/patch_cactus.py)\n",
      "        cactus_delta,  # radiance cactus: launch arg (sly/patch_cactus.py)\n"
      "        # radiance cactus fade: launch args. Without prompt lengths the kernel never reads the\n"
      "        # pointer (fade end 0), so `seed` ([max_num_reqs], in bounds) only fills the slot.\n"
      "        cactus_prompt_len if cactus_prompt_len is not None else seed,\n"
      "        CACTUS_FADE_START,\n"
      "        CACTUS_FADE_END if cactus_prompt_len is not None else 0,\n",
      "# radiance cactus fade: launch args",
      "cactus fade: kernel launch")

# The V2 rejection sampler passes the prompt lengths it already holds (req_states.prompt_len, per slot).
apply(TARGET_CALL,
      "            use_block_verification=self.use_block_verification,\n",
      "            use_block_verification=self.use_block_verification,\n"
      "            cactus_prompt_len=self.sampler.req_states.prompt_len.gpu,  # radiance cactus fade (sly/patch_cactus.py)\n",
      "# radiance cactus fade (sly/patch_cactus.py)",
      "cactus fade: prompt lengths from the rejection sampler")
