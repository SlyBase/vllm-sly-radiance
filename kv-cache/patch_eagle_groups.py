#!/usr/bin/env python3
"""R3.13a -- stop mislabelling all nine KV groups as EAGLE/MTP draft groups.

WHAT IS WRONG TODAY

This engine boots with:

  scheduler.py:243 KV offloading: EAGLE/MTP draft attention groups
                   [0, 1, 2, 3, 4, 5, 6, 7, 8] detected.

All nine. That is every group in the model -- the six Mamba/GDN groups, both
full-attention groups, and the actual draft group. Only one of them is a draft
group. The line is not cosmetic: `is_eagle_group` changes both the store and the
load path.

  * RequestOffloadState.storable_chunks() drops the trailing chunk of an eagle
    group while decoding, because a draft layer's KV for the last accepted
    position can still be rewritten when spec tokens are rejected. Applied to all
    nine groups, it means the newest 1,648-token chunk of every conversation is
    withheld from the offload store for the whole of a decode -- the single chunk
    a follow-up turn is most likely to ask for.
  * _lookup() queries one extra chunk for an eagle group and then pops it, so the
    servable prefix is one chunk shorter than it needs to be, nine times over.

WHY IT HAPPENS

vllm/v1/core/kv_cache_utils.py has exactly one annotator,
`_annotate_eagle_groups_deepseek_v4`, and it is gated twice:

  1. It is only *called* from the `group_and_unify_kv_cache_specs` branch of
     get_kv_cache_groups() -- the DeepSeek-V4 path. A hybrid Mamba+attention model
     like ours falls through to `_get_kv_cache_groups_uniform_page_size` at the
     bottom of the function, which never annotates anything.
  2. Even if it were called, it returns early unless some spec carries
     `model_version == "deepseek_v4"`.

So no group is ever annotated, and the offload scheduler's fallback

     if use_eagle and not eagle_groups:
         eagle_groups = set(range(len(kv_cache_config.kv_cache_groups)))

flags the lot. That fallback is a reasonable fail-safe -- over-excluding is safer
than serving a volatile draft chunk -- but it is meant to be unreachable.

Upstream knows. The function carries its own note:

     # FIXME(yifan): avoid/generalize this hacky check.

and vllm-project/vllm#52047 (NOT merged as #55390; #52047 does not cover this model) ("Annotate MTP draft KV groups positionally") is the
open PR for it. Only ~11% of that PR's diff applies to our tree -- it targets a
different grouping path -- so this patch ports the *idea*, not the diff.

WHAT THIS CHANGES

Two one-line-shaped edits, both in kv_cache_utils.py:

  A. Drop the `model_version == "deepseek_v4"` gate. The rule the function
     implements -- "the draft model's attention layer is registered last, so flag
     whichever group holds the last layer" -- is a fact about how vLLM registers a
     draft model, not a fact about DeepSeek.
  B. Call the annotator on the general hybrid path too, right after
     `_get_kv_cache_groups_uniform_page_size` builds the groups.

WHY THE LAST-LAYER RULE IS CORRECT FOR THIS MODEL

Qwen3.8-27B-MXFP4-mtpfp8 is 64 layers with full_attention_interval=4: 48 linear
(GDN) + 16 full attention. The DFlash2 draft adds 5 sliding-attention layers,
registered after the main model. vLLM groups at most 8 layers per group, giving
the 9 groups we see: 48/8 = 6 Mamba groups (g0-g5), 16/8 = 2 full-attention
groups (g6-g7), and the 5 draft layers as g8. The last registered layer is a
draft layer, so the rule flags g8 and nothing else.

VERIFICATION -- this patch is self-checking. After a restart the boot line must read

  KV offloading: EAGLE/MTP draft attention groups [8] detected.

If it still says [0, 1, ..., 8], the annotation did not take and the engine is
running upstream behaviour. Nothing else needs inspecting.

SAFETY. This only ever *narrows* the eagle set, and a narrower set means more
chunks stored and longer prefixes served -- it cannot cause a volatile draft
chunk to be served, because g8 stays flagged. If the rule ever picked no group at
all, the scheduler's fallback still flags everything, i.e. today's behaviour.

REVERTING: set RADIANCE_OFFLOAD_EAGLE_GROUPS=0 and restart. The patch stays
applied and the upstream DeepSeek-only gate comes back.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _patchlib import apply, apply_any, skip_if_upstream  # noqa: E402

VLLM = Path(sys.prefix) / "lib" / f"python3.{sys.version_info.minor}" / "site-packages" / "vllm"
if not VLLM.exists():
    import vllm as _v
    VLLM = Path(_v.__file__).parent

KVU = VLLM / "v1" / "core" / "kv_cache_utils.py"

print("[radiance] R3.13a eagle-group annotation")

# --- A. remove the DeepSeek-only gate -------------------------------------------------
# Three shapes of the same rule. vLLM 0.27.1: one annotator, `_annotate_eagle_groups_deepseek_v4`,
# gated on the spec's model_version. vLLM >= 0.30: `_annotate_eagle_groups` runs on every
# grouping path (hunk B is upstream), but its positional rule -- the draft model's attention
# layer is registered last, so flag the group holding the last layer -- is reached only through
# a fallback argument the uniform-page-size path never passes (0.30: `use_deepseek_v4_fallback`;
# 0.31: `use_trailing_layer_fallback=_uses_trailing_mtp_layers(...)`, True for `mtp` and
# DeepSeek V4 only, False for dflash). The fallback for an unannotated model flipped too:
# 0.30+ treats every group as NON-draft ("fail toward serving"), so the drafter's volatile
# trailing chunk can be stored and served one chunk stale, and -- measured 2026-10-08,
# kv-cache/README.md -- the tier hit then lands 4-5 turns back. So the rule is re-enabled here
# the same way: gated on the knob and run with filtered_spec (registration order without the
# hidden-state layers that were appended as their own groups above; 0.31 additionally checks
# that the groups partition the spec exactly, which holds for filtered_spec). The block goes
# BETWEEN upstream's annotate call and `_warn_if_unannotated_eagle_mamba`, so the warning does
# not fire for a model this patch annotates a line later.
_BLOCK_030 = '''    # radiance R3.13a: run the positional draft-group rule past the model-type gate.
    # The draft model's attention layer is registered last, so the group holding the
    # last layer is the drafter's -- a fact about how vLLM registers a draft model, not
    # about one model family. Upstream otherwise treats every group as non-draft and
    # stores the drafter's volatile trailing chunk. filtered_spec keeps registration
    # order without the hidden-state layers appended as their own groups above, which
    # is what makes the rule valid on this path. RADIANCE_OFFLOAD_EAGLE_GROUPS unset/0 = upstream.
    if os.environ.get("RADIANCE_OFFLOAD_EAGLE_GROUPS", "0") == "1" and not any(
        group.is_eagle_group for group in groups
    ):
        _annotate_eagle_groups(
            vllm_config, filtered_spec, groups, %s=True
        )
'''
_CALL_030 = "    _annotate_eagle_groups(vllm_config, kv_cache_spec, groups)\n"
# the preceding line makes the uniform-path call unique (the packed path has the same call)
_CALL_031 = (
    "            groups.append(KVCacheGroupSpec([name], aligned))\n"
    "\n"
    "    _annotate_eagle_groups(\n"
    "        vllm_config,\n"
    "        kv_cache_spec,\n"
    "        groups,\n"
    "        use_trailing_layer_fallback=_uses_trailing_mtp_layers(vllm_config),\n"
    "    )\n"
)
_ANCHOR_027 = '''    # Detection uses the merged MLA spec's model_version.
    if not any(
        getattr(spec, "model_version", None) == "deepseek_v4"
        for spec in kv_cache_spec.values()
    ):
        return'''
_NEW_027 = '''    # Detection uses the merged MLA spec's model_version.
    # The rule this function applies -- the draft model's attention layer is
    # registered last, so flag whichever group holds the last layer -- is a fact
    # about how vLLM registers a draft model, not anything specific to DeepSeek.
    # Gating it on model_version leaves every other speculative model unannotated,
    # which trips the offload scheduler's flag-them-all fallback. Setting
    # RADIANCE_OFFLOAD_EAGLE_GROUPS=0 restores the DeepSeek-only gate.
    if os.environ.get("RADIANCE_OFFLOAD_EAGLE_GROUPS", "0") != "1" and not any(
        getattr(spec, "model_version", None) == "deepseek_v4"
        for spec in kv_cache_spec.values()
    ):
        return'''
apply_any(
    KVU,
    [
        (_ANCHOR_027, _NEW_027),
        (_CALL_030, _CALL_030 + _BLOCK_030 % "use_deepseek_v4_fallback"),
        (_CALL_031, _CALL_031 + _BLOCK_030 % "use_trailing_layer_fallback"),
    ],
    sentinel="RADIANCE_OFFLOAD_EAGLE_GROUPS",
    label="A kv_cache_utils: generalize eagle annotation past the model-type gate",
)

# --- B. call it on the general hybrid grouping path ------------------------------------
if not skip_if_upstream(
    KVU,
    "_warn_if_unannotated_eagle_mamba(vllm_config, groups)",
    "B kv_cache_utils: annotate eagle groups on the hybrid page-size path",
    "vLLM >= 0.30 calls the annotator on every grouping path (A covers the gate)",
):
    apply(
        KVU,
        anchor="    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec, vllm_config)",
        new='''    groups = _get_kv_cache_groups_uniform_page_size(filtered_spec, vllm_config)

    # Annotate the EAGLE/MTP draft group on the uniform-page-size path too. Upstream
    # annotates only on the DeepSeek-V4 branch above, so a hybrid Mamba+attention model
    # lands here with nothing annotated and the offload scheduler flags all of its groups
    # as draft groups. filtered_spec (not kv_cache_spec) keeps registration order while
    # excluding the hidden-state layers that are not in `groups` yet.
    _annotate_eagle_groups_deepseek_v4(vllm_config, filtered_spec, groups)''',
        sentinel="_annotate_eagle_groups_deepseek_v4(vllm_config, filtered_spec, groups)",
        label="B kv_cache_utils: annotate eagle groups on the hybrid page-size path",
    )

print("[radiance] R3.13a applied -- expect 'draft attention groups [8] detected' at boot")
