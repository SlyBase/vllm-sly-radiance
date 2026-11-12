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
RADIANCE_CACTUS_CAP_R       acceptance cap, default 0.5 (0 = off): h_n is additionally limited to
                            1 - r*(1 - q_n). h >= q always (never stricter than stock) and, at the
                            drafter's p ~ 1, it keeps r of the stock rejection chance at every q < 1 --
                            Cactus itself has none above q = 1/(1+2*delta). A second, length-independent
                            guarantee next to the fade; it costs gain mainly in the q 0.4-0.7 band.
RADIANCE_CACTUS_ESCAPE      escape mixture, default 0 (off): with probability eps the EXACT stock
                            acceptance test runs instead of the Cactus test (same uniform u).
                            Per-step acceptance becomes (1-eps)*min(1,h/p) + eps*min(1,q/p), so the
                            output distribution is exactly (1-eps)*out_cactus + eps*q and the escape
                            chance is >= eps * stock at EVERY (q,p) -- the cap only guarantees that
                            at p ~ 1, and a confident drafter inside a repetition loop is exactly
                            the p -> 1, q -> 1 corner where threshold-only Cactus has zero escape
                            (issue #164, docs/ANALYSIS-164-cactus-looping.md). eps=1 is bit-exact
                            stock; eps=0 and delta=0 keep the previous comparison bit-exact.

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
      "CACTUS_FADE_END = int(os.environ.get(\"RADIANCE_CACTUS_FADE_END\", \"16384\"))\n"
      "CACTUS_CAP_R = float(os.environ.get(\"RADIANCE_CACTUS_CAP_R\", \"0.5\"))\n"
      "if not 0.0 <= CACTUS_CAP_R <= 1.0:\n"
      "    raise RuntimeError(f\"RADIANCE_CACTUS_CAP_R must be in [0, 1], got {CACTUS_CAP_R}\")\n",
      "RADIANCE_CACTUS_CAP_R",
      "cactus fade + cap: env knobs")

# Escape mixture (issue #164). The cap keeps an escape only while the drafter is unsure (p < 1);
# inside a repetition loop the drafter is confident (p -> 1) AND the target is confident (q -> 1),
# and there threshold-only Cactus accepts with probability 1: the loop becomes absorbing. Mixing
# the exact stock test in with probability eps restores escape >= eps * stock at every (q, p).
apply(TARGET,
      "if not 0.0 <= CACTUS_CAP_R <= 1.0:\n"
      "    raise RuntimeError(f\"RADIANCE_CACTUS_CAP_R must be in [0, 1], got {CACTUS_CAP_R}\")\n",
      "if not 0.0 <= CACTUS_CAP_R <= 1.0:\n"
      "    raise RuntimeError(f\"RADIANCE_CACTUS_CAP_R must be in [0, 1], got {CACTUS_CAP_R}\")\n"
      "# radiance cactus escape mixture: env knob (sly/patch_cactus.py, issue #164)\n"
      "CACTUS_ESCAPE = float(os.environ.get(\"RADIANCE_CACTUS_ESCAPE\", \"0\"))\n"
      "if not 0.0 <= CACTUS_ESCAPE <= 1.0:\n"
      "    raise RuntimeError(f\"RADIANCE_CACTUS_ESCAPE must be in [0, 1], got {CACTUS_ESCAPE}\")\n",
      "RADIANCE_CACTUS_ESCAPE",
      "cactus escape mixture: env knob")

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
      "    cactus_fade_end,\n"
      "    cactus_cap_r,\n",
      "# radiance cactus fade: kernel args",
      "cactus fade: kernel signature")

apply(TARGET,
      "    cactus_cap_r,\n",
      "    cactus_cap_r,\n"
      "    cactus_escape,  # radiance cactus escape mixture: kernel arg (#164)\n",
      "# radiance cactus escape mixture: kernel arg",
      "cactus escape mixture: kernel signature")

# Threshold. Shape 2 is the 1.5.1-1.8.1 hunk, shape 3 the 1.9.0-1.11.0 fade hunk, so a launcher
# re-run on an older image upgrades it to the escape mixture in one pass.
_STOCK_TEST = (
    "                    # Probability ratio test: p(x) > u * q(x)\n"
    "                    # Equivalent log form: log_p(x) > log(u) + log_q(x)\n"
)
_ACCEPT_TAIL = (
    "                    log_h_n = tl.where(req_delta == 0.0, target_logprob, tl.log(h_n))\n"
    "                    # radiance cactus escape mixture: acceptance (#164). With probability\n"
    "                    # CACTUS_ESCAPE the exact stock test runs instead (same u). Acceptance\n"
    "                    # becomes the convex mix (1-eps)*min(1,h/p) + eps*min(1,q/p):\n"
    "                    # out = (1-eps)*out_cactus + eps*q, so the escape chance is >= eps*stock\n"
    "                    # at EVERY (q,p), not only at p~1 like the cap. eps=0 and delta=0 keep\n"
    "                    # the previous comparison bit-exact; eps=1 is the stock log test itself.\n"
    "                    acc_cactus = log_h_n > tl.log(u) + draft_logprob\n"
    "                    a_h = tl.exp(tl.minimum(log_h_n - draft_logprob, 0.0))\n"
    "                    a_q = tl.exp(tl.minimum(target_logprob - draft_logprob, 0.0))\n"
    "                    acc_mix = u < (1.0 - cactus_escape) * a_h + cactus_escape * a_q\n"
    "                    acc_stock = target_logprob > tl.log(u) + draft_logprob\n"
    "                    accepted = tl.where(cactus_escape == 1.0, acc_stock,\n"
    "                                        tl.where((cactus_escape > 0.0) & (req_delta > 0.0),\n"
    "                                                 acc_mix, acc_cactus))\n"
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
    "                    # radiance cactus cap: keeps an escape at every q (h >= q, so never below stock)\n"
    "                    h_n = tl.minimum(h_n, 1.0 - cactus_cap_r * (1.0 - q_n))\n"
    + _ACCEPT_TAIL
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
       _THRESHOLD),
      ("                    log_h_n = tl.where(req_delta == 0.0, target_logprob, tl.log(h_n))\n"
       "                    accepted = log_h_n > tl.log(u) + draft_logprob\n",
       _ACCEPT_TAIL)],
      "# radiance cactus escape mixture: acceptance",
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
      "    cactus_prompt_len: torch.Tensor | None = None,  # radiance cactus fade: wrapper arg\n",
      "    cactus_prompt_len: torch.Tensor | None = None,  # radiance cactus fade: wrapper arg\n"
      "    cactus_escape: float = CACTUS_ESCAPE,  # radiance cactus escape mixture: wrapper arg (#164)\n",
      "# radiance cactus escape mixture: wrapper arg",
      "cactus escape mixture: rejection_sample signature")

apply(TARGET,
      "        cactus_delta,  # radiance cactus: launch arg (sly/patch_cactus.py)\n",
      "        cactus_delta,  # radiance cactus: launch arg (sly/patch_cactus.py)\n"
      "        # radiance cactus fade: launch args. Without prompt lengths the kernel never reads the\n"
      "        # pointer (fade end 0), so `seed` ([max_num_reqs], in bounds) only fills the slot.\n"
      "        cactus_prompt_len if cactus_prompt_len is not None else seed,\n"
      "        CACTUS_FADE_START,\n"
      "        CACTUS_FADE_END if cactus_prompt_len is not None else 0,\n"
      "        CACTUS_CAP_R,\n",
      "# radiance cactus fade: launch args",
      "cactus fade: kernel launch")

apply(TARGET,
      "        CACTUS_CAP_R,\n",
      "        CACTUS_CAP_R,\n"
      "        CACTUS_ESCAPE,  # radiance cactus escape mixture: launch arg (#164)\n",
      "# radiance cactus escape mixture: launch arg",
      "cactus escape mixture: kernel launch")

# The V2 rejection sampler passes the prompt lengths it already holds (req_states.prompt_len, per slot).
apply(TARGET_CALL,
      "            use_block_verification=self.use_block_verification,\n",
      "            use_block_verification=self.use_block_verification,\n"
      "            cactus_prompt_len=self.sampler.req_states.prompt_len.gpu,  # radiance cactus fade (sly/patch_cactus.py)\n",
      "# radiance cactus fade (sly/patch_cactus.py)",
      "cactus fade: prompt lengths from the rejection sampler")
