#!/usr/bin/env python3
"""vLLM's native DFlash2 speculator for the Inkling-2-Mini target (`incoai/Inkling-2-Mini-DFlash2`).

Opt-in by serving flag only: `--speculative-config '{"method":"dflash","model":<dir containing
"dflash2" in its name>,"num_speculative_tokens":7,"draft_sample_method":"probabilistic",
"attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto"}'`. The default DSpark route is untouched,
and every hook is inert for a non-Inkling target (the Qwen DFlash2 prod). Module with the logic:
inkling2/radiance_inkling2_dflash2.py. Hooks:

  1. eagle3_utils.get_eagle3_aux_layers_from_config   capture ids = Inkling layer visits (no +1)
  2. qwen3_dflash2.CandidateSelector                  `after_correction` softcap on unary + edge score
  3. qwen3_dflash2 candidate head                     Inkling2LogitsProcessor, cap moved off the unary
  4. qwen3_dflash2.DFlash2Qwen3Model.embed_input_ids  target embed_norm (shared in dflash/utils.py)

NOOP on a vLLM without `models/inkling2`.

Reference: the vLLM diffs of github.com/tonyd2wild/Inkling-2-Mini-NVFP4-DGX-Spark (dflash2/patches, Apache-2.0,
GB10 / RTX 3090, 48/48 greedy outputs byte-identical to DSpark) make the same three Inkling changes (ids as-is,
after_correction softcap, target embed_norm on the draft inputs). Re-anchored here to the ink-k9 files; their SM<90
torch fallbacks do not apply to ROCm (`current_platform.is_cuda()` is False there).
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
EAGLE3 = SP / "vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py"
QWEN3 = SP / "vllm/model_executor/models/qwen3_dflash2.py"
LOADER = SP / "vllm/v1/worker/gpu/spec_decode/dflash/utils.py"

IMPORT = "from radiance_inkling2_dflash2 import"


def main():
    if not (SP / "vllm/models/inkling2").is_dir():
        print("  NOOP  inkling2 dflash2: this vLLM has no models/inkling2")
        return
    apply(
        EAGLE3,
        '            layer_ids = [i + 1 for i in (dflash_config.get("target_layer_ids") or [])]\n',
        "            from radiance_inkling2_dflash2 import aux_layer_ids\n\n"
        '            layer_ids = aux_layer_ids(spec_config, dflash_config.get("target_layer_ids") or [])\n',
        "aux_layer_ids(",
        "inkling2 dflash2 capture ids",
    )
    apply(
        QWEN3,
        "from .utils import maybe_prefix\n",
        "from radiance_inkling2_dflash2 import (\n"
        "    candidate_head_cls,\n"
        "    embed_with_norm,\n"
        "    selector_softcaps,\n"
        "    softcap_after,\n"
        ")\n\n"
        "from .utils import maybe_prefix\n",
        IMPORT,
        "inkling2 dflash2 import",
    )
    apply(
        QWEN3,
        "        params_dtype: torch.dtype,\n"
        "        prefix: str,\n"
        "    ) -> None:\n"
        "        super().__init__()\n"
        "        self.top_k = top_k\n",
        "        params_dtype: torch.dtype,\n"
        "        prefix: str,\n"
        "        post_softcap: float | None = None,\n"
        "    ) -> None:\n"
        "        super().__init__()\n"
        "        self.top_k = top_k\n"
        "        self.post_softcap = post_softcap\n",
        "self.post_softcap = post_softcap",
        "inkling2 dflash2 selector softcap field",
    )
    apply(
        QWEN3,
        "        return _score_edges(\n"
        "            self.predecessor_codebook,\n"
        "            self.successor_codebook,\n"
        "            candidate_ids,\n"
        "            unary_logits,\n"
        "            hidden,\n"
        "            anchor_token_ids,\n"
        "            self.top_k,\n"
        "        )\n",
        "        scores = _score_edges(\n"
        "            self.predecessor_codebook,\n"
        "            self.successor_codebook,\n"
        "            candidate_ids,\n"
        "            unary_logits,\n"
        "            hidden,\n"
        "            anchor_token_ids,\n"
        "            self.top_k,\n"
        "        )\n"
        "        return softcap_after(scores, self.post_softcap)\n",
        "softcap_after(scores",
        "inkling2 dflash2 selector softcap",
    )
    apply(
        QWEN3,
        '                top_k=int(draft_config["selector_top_k"]),\n',
        '                top_k=int(draft_config["selector_top_k"]),\n'
        "                post_softcap=selector_softcaps(draft_config)[1],\n",
        "post_softcap=selector_softcaps(",
        "inkling2 dflash2 selector cap config",
    )
    apply(
        QWEN3,
        '        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)\n',
        "        softcap = selector_softcaps(draft_config)[0] or 0.0\n",
        "softcap = selector_softcaps(",
        "inkling2 dflash2 unary cap",
    )
    apply(
        QWEN3,
        "        self.candidate_logits_processor = LogitsProcessor(\n",
        "        self.candidate_logits_processor = candidate_head_cls(vllm_config)(\n",
        "candidate_head_cls(vllm_config)(",
        "inkling2 dflash2 candidate head",
    )
    apply(
        QWEN3,
        "        return super().embed_input_ids(input_ids) * self.input_embedding_scale\n",
        "        embeds = embed_with_norm(self, super().embed_input_ids(input_ids))\n"
        "        return embeds * self.input_embedding_scale\n",
        "embed_with_norm(self,",
        "inkling2 dflash2 embed norm",
    )
    apply(
        LOADER,
        "    maybe_share_target_embed(dflash_model, draft_inner, target_inner)\n",
        "    maybe_share_target_embed(dflash_model, draft_inner, target_inner)\n"
        "    from radiance_inkling2_dflash2 import attach_embed_norm\n\n"
        "    attach_embed_norm(draft_inner, target_inner)\n",
        "attach_embed_norm(draft_inner",
        "inkling2 dflash2 embed norm share",
    )


if __name__ == "__main__":
    main()
