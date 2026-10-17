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

from _patchlib import apply

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
      "                    # Probability ratio test: p(x) > u * q(x)\n"
      "                    # Equivalent log form: log_p(x) > log(u) + log_q(x)\n"
      "                    accepted = target_logprob > tl.log(u) + draft_logprob\n",
      "                    # Probability ratio test: p(x) > u * q(x)\n"
      "                    # Equivalent log form: log_p(x) > log(u) + log_q(x)\n"
      "                    # radiance cactus: threshold (sly/patch_cactus.py): q_n -> h_n,\n"
      "                    # delta=0 stays the stock comparison, bit-exact.\n"
      "                    q_n = tl.exp(target_logprob)\n"
      "                    h_n = tl.minimum(\n"
      "                        tl.maximum(q_n + tl.sqrt(2.0 * cactus_delta * q_n * (1.0 - q_n)), 0.0), 1.0\n"
      "                    )\n"
      "                    log_h_n = tl.where(cactus_delta == 0.0, target_logprob, tl.log(h_n))\n"
      "                    accepted = log_h_n > tl.log(u) + draft_logprob\n",
      "# radiance cactus: threshold",
      "cactus: Leviathan acceptance")

apply(TARGET,
      "    watermarking: torch.Tensor | None = None,\n"
      "    watermark_key: int | None = None,\n"
      ") -> tuple[torch.Tensor, torch.Tensor]:\n",
      "    watermarking: torch.Tensor | None = None,\n"
      "    watermark_key: int | None = None,\n"
      "    cactus_delta: float = CACTUS_DELTA,  # radiance cactus: wrapper arg (sly/patch_cactus.py)\n"
      ") -> tuple[torch.Tensor, torch.Tensor]:\n",
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
