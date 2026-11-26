"""Glue that lets vLLM's native DFlash2 speculator draft for the Inkling-2-Mini target.

The checkpoint is `incoai/Inkling-2-Mini-DFlash2` (5 sliding-window Qwen3 layers, grouped convs, a
candidate selector); the vLLM speculator (`spec_decode/dflash2`) and model (`qwen3_dflash2.py`) are
used unchanged except for the four Inkling-specific points the vendor's SGLang patch lists. Hooks:
sly/patch_inkling2_dflash2.py. Every function is inert for a non-Inkling target (Qwen prod).

  1. capture ids   `aux_layer_ids`      layer VISITS of the looped target, the last id = the final-norm
                                        output (num_layer_visits); no DFlash "+1" shift.
  2. vocab         (upstream already)   candidates are clamped to min(target vocab, draft vocab).
  3. softcap mode  `selector_softcaps`, `softcap_after`   `selector_softcap_mode: after_correction`
                                        caps unary + edge score, not the unary logit alone.
  4. explicit ids  `aux_layer_ids`      the config's target_layer_ids, validated against the target.
  +  head / embed  `candidate_head_cls`, `attach_embed_norm`, `embed_with_norm`: the candidate logits
                   run through Inkling's own logits processor (its MXFP4 dense head replaces the bf16
                   weights at load), the draft embeds tokens with the target's embed_norm
                   (RADIANCE_INKLING2_DFLASH2_EMBED_NORM=0 = raw embedding).
"""
import os

import torch

_MODES = ("before_correction", "after_correction")


def _is_inkling2(spec_config) -> bool:
    target = getattr(spec_config, "target_model_config", None)
    return str(getattr(getattr(target, "hf_config", None), "model_type", "")).startswith("inkling2")


def aux_layer_ids(spec_config, target_layer_ids):
    """Target layers whose hidden state the draft reads (what `set_aux_hidden_state_layers` takes)."""
    ids = list(target_layer_ids)
    if not _is_inkling2(spec_config):
        return [i + 1 for i in ids]  # DFlash ids are "after layer i"; eagle3 ids "input of layer i+1"
    visits = spec_config.target_model_config.hf_config.text_config.num_layer_visits
    if not ids or ids != sorted(set(ids)) or ids[0] < 0 or ids[-1] != visits:
        raise ValueError(
            f"Inkling draft capture ids must be sorted, unique, >= 0 and end at the final-norm "
            f"visit {visits}; got {ids}"
        )
    return ids


def selector_softcaps(dflash_config):
    """(cap on the unary logits, cap on the corrected score); at most one is set."""
    mode = dflash_config.get("selector_softcap_mode", "before_correction")
    if mode not in _MODES:
        raise ValueError(f"selector_softcap_mode must be one of {_MODES}, got {mode!r}")
    cap = float(dflash_config.get("final_logit_softcapping") or 0.0) or None
    return (cap, None) if mode == "before_correction" else (None, cap)


def softcap_after(scores, cap):
    """cap * tanh(scores / cap) on the corrected selector scores; no cap = unchanged."""
    return scores if not cap else cap * torch.tanh(scores / cap)


def candidate_head_cls(vllm_config):
    """Inkling's own logits processor for a Inkling target (it owns the MXFP4 dense head route)."""
    if _is_inkling2(vllm_config.speculative_config):
        from vllm.models.inkling2.nvidia.model import Inkling2LogitsProcessor

        return Inkling2LogitsProcessor
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    return LogitsProcessor


def attach_embed_norm(draft_inner, target_inner):
    """Share the target's embed_norm with the draft: Inkling's input embedding is norm(embed(token))."""
    norm = getattr(target_inner, "embed_norm", None)
    if norm is not None and os.environ.get("RADIANCE_INKLING2_DFLASH2_EMBED_NORM", "1") == "1":
        draft_inner.embed_norm = norm


def embed_with_norm(draft_inner, embeds):
    norm = getattr(draft_inner, "embed_norm", None)
    return embeds if norm is None else norm(embeds)
