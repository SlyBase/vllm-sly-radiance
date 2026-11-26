#!/usr/bin/env python3
"""Opt-in MXFP4 dense routing for Inkling 2 on gfx12 (RADIANCE_INKLING2_MXFP4_DENSE=1).

SEPARATE from patch_inkling2_gfx12.py (which owns the gates/PDL/PTX/expert/GEMV hunks): this patch
only adds the two MXFP4-dense interception hooks and touches none of those hunks. It runs AFTER
patch_inkling2_gfx12 in the Dockerfile loop, so it anchors on the gfx12-patched model files (the
form the shipped image carries). The dispatch + quantizer live in radiance_inkling2_mxfp4dense.py;
with the env knob off (the default) every hook returns None and the lossless bf16 path runs unchanged.

  1. ops/linear.py linear(): the dense bf16 matmul entry (attention qkvr/o_proj, the dense MLP, the
     DSpark drafter's linears + head). The hook tries the MXFP4 copy first; None falls through.
  2. model.py Inkling2LogitsProcessor._apply_head: the lm_head/unembed matmul (which bypasses
     linear() via sm12x_bf16_gemv / the quant method). The hook routes it through the MXFP4 copy.

  3. Load time (finalize_model_weights runs each module's process_weights_after_loading, before vLLM
     profiles memory): Inkling2ForConditionalGeneration and Inkling2DSparkModel quantize their dense
     weights and release the bf16 originals; Inkling2DSparkModel.attach_target frees the shared
     lm_head after the fold has read it. Nothing is quantized lazily, so there is no bf16 fallback.

A vLLM without models/inkling2 is a NOOP.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
ROOT = SP / "vllm/models/inkling2"


def main():
    if not ROOT.is_dir():
        print("  NOOP  inkling2 mxfp4dense: this vLLM has no models/inkling2")
        return
    apply(
        ROOT / "nvidia/ops/linear.py",
        '    """bf16 `x @ weight.T` with fp32 accumulation (into `out` if given,\n'
        '    batches over LOW_M_MAX_TOKENS only)."""\n'
        "    if out is not None:\n",
        '    """bf16 `x @ weight.T` with fp32 accumulation (into `out` if given,\n'
        '    batches over LOW_M_MAX_TOKENS only)."""\n'
        "    if torch.version.hip:  # opt-in MXFP4 dense (RADIANCE_INKLING2_MXFP4_DENSE=1)\n"
        "        from radiance_inkling2_mxfp4dense import mxfp4_dense_linear\n\n"
        "        _y = mxfp4_dense_linear(x, weight, out)\n"
        "        if _y is not None:\n"
        "            return _y\n"
        "    if out is not None:\n",
        "from radiance_inkling2_mxfp4dense import mxfp4_dense_linear",
        "inkling2 mxfp4dense: MXFP4 dense linear hook (opt-in)",
    )
    apply(
        ROOT / "nvidia/model.py",
        "    def _apply_head(self, lm_head, hidden_states, embedding_bias):\n"
        "        from vllm.model_executor.layers.linear import UnquantizedLinearMethod\n",
        "    def _apply_head(self, lm_head, hidden_states, embedding_bias):\n"
        "        if torch.version.hip:  # opt-in MXFP4 dense head (RADIANCE_INKLING2_MXFP4_DENSE=1)\n"
        "            from radiance_inkling2_mxfp4dense import mxfp4_dense_head\n\n"
        "            _y = mxfp4_dense_head(lm_head.weight, hidden_states, self.head_dtype)\n"
        "            if _y is not None:\n"
        "                return _y\n"
        "        from vllm.model_executor.layers.linear import UnquantizedLinearMethod\n",
        "from radiance_inkling2_mxfp4dense import mxfp4_dense_head",
        "inkling2 mxfp4dense: MXFP4 lm_head hook (opt-in)",
    )
    # --- load time: quantize and release the bf16 originals before vLLM sizes the KV cache ---
    apply(
        ROOT / "nvidia/model.py",
        "    def get_language_model(self) -> nn.Module:\n"
        "        return self\n",
        "    def process_weights_after_loading(self) -> None:\n"
        "        if torch.version.hip:  # opt-in MXFP4 dense (RADIANCE_INKLING2_MXFP4_DENSE=1)\n"
        "            from radiance_inkling2_mxfp4dense import process_target\n\n"
        "            process_target(self)\n\n"
        "    def get_language_model(self) -> nn.Module:\n"
        "        return self\n",
        "from radiance_inkling2_mxfp4dense import process_target",
        "inkling2 mxfp4dense: quantize the target's dense weights at load (opt-in)",
    )
    apply(
        ROOT / "nvidia/dspark.py",
        "    def get_draft_kv_cache_layer_names(self) -> list[str]:\n",
        "    def process_weights_after_loading(self) -> None:\n"
        "        if torch.version.hip:  # opt-in MXFP4 dense (RADIANCE_INKLING2_MXFP4_DENSE=1)\n"
        "            from radiance_inkling2_mxfp4dense import process_draft\n\n"
        "            process_draft(self)\n\n"
        "    def get_draft_kv_cache_layer_names(self) -> list[str]:\n",
        "from radiance_inkling2_mxfp4dense import process_draft",
        "inkling2 mxfp4dense: quantize the draft's dense weights at load (opt-in)",
    )
    apply(
        ROOT / "nvidia/dspark.py",
        "        head.rnn_bias = torch.empty_like(head.rnn_proj.bias, dtype=torch.float32)\n"
        "        self.refresh_derived_weights()\n",
        "        head.rnn_bias = torch.empty_like(head.rnn_proj.bias, dtype=torch.float32)\n"
        "        self.refresh_derived_weights()\n"
        "        if torch.version.hip:  # the fold above was the last bf16 read of the head\n"
        "            from radiance_inkling2_mxfp4dense import release_head\n\n"
        "            release_head(self.lm_head.weight)\n",
        "from radiance_inkling2_mxfp4dense import release_head",
        "inkling2 mxfp4dense: free the bf16 lm_head after the draft folded it (opt-in)",
    )


if __name__ == "__main__":
    main()