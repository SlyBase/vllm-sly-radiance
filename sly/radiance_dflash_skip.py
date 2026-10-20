"""Drafter sliding-window prefill skip: the decision (RADIANCE_DFLASH_PREFILL_WINDOW_SKIP).

Hooked at the top of `DFlashSpeculator.propose` by sly/patch_dflash_prefill_skip.py. While a request
is still prefilling, the scheduler drops the draft tokens (`update_draft_token_ids`: "Ignore draft
tokens for prefill chunks"), and the DFlash2 drafter only reads context K/V inside its sliding
window (all layers `sliding_attention`; its conv and candidate selector see query rows only). So a
chunk that ends `window + margin` tokens before the end of the prompt needs no `fc`, no context-K/V
fill and no draft step: `propose` returns early. The final chunks, decode and spec-verify steps
never match the rule and run unchanged.

margin = max(RADIANCE_DFLASH_PREFILL_WINDOW_SKIP_MARGIN, 2 KV blocks): the block that holds the
last skipped position (partly unwritten) must lie wholly below the first block the attention can
touch, whatever its tile size is.

Off (stays stock) with prefix caching on (a cached drafter block of a skipped position would hold
garbage for the next request that shares the prefix), with any non-sliding drafter layer, without a
configured window, and with data parallelism (the draft step is a collective).
"""
import os
import sys

_SW = "sliding_attention"


def _config(spec):
    """(window, margin), or () when the skip is off or unsafe. Decided once per speculator."""
    if os.environ.get("RADIANCE_DFLASH_PREFILL_WINDOW_SKIP", "0") != "1":
        return ()
    hf = spec.draft_model_config.hf_config
    dflash = getattr(hf, "dflash_config", None) or {}
    window = dflash.get("swa_window_size", getattr(hf, "sliding_window", None))
    layer_types = getattr(hf, "layer_types", None)
    why = (
        "prefix caching is on" if spec.vllm_config.cache_config.enable_prefix_caching
        else "data parallel" if spec.dp_size != 1
        else "the drafter has non-sliding layers" if not layer_types or set(layer_types) != {_SW}
        else "the drafter has no sliding window" if not window
        else ""
    )
    if why:
        sys.stderr.write(f"[radiance.dflash_skip] RADIANCE_DFLASH_PREFILL_WINDOW_SKIP ignored: {why}\n")
        return ()
    block = max(g.kv_cache_spec.block_size for g in spec.kv_cache_config.kv_cache_groups)
    margin = max(int(os.environ.get("RADIANCE_DFLASH_PREFILL_WINDOW_SKIP_MARGIN", "512")), 2 * block)
    sys.stderr.write(f"[radiance.dflash_skip] ON: window {window}, margin {margin}\n")
    return int(window), margin


def skip_context(spec, input_batch, dummy_run=False, is_profile=False):
    """True when this step is a prefill chunk of every request and all its positions lie below
    prompt_len - window - margin."""
    if dummy_run or is_profile:  # before _config: the profile run precedes set_attn (no kv_cache_config yet)
        return False
    cfg = getattr(spec, "_radiance_pf_skip", None)
    if cfg is None:
        cfg = spec._radiance_pf_skip = _config(spec)
    if not cfg:
        return False
    n = input_batch.num_reqs
    end = input_batch.num_computed_prefill_tokens_np[:n] + input_batch.num_scheduled_tokens[:n]
    return bool((end <= input_batch.prefill_len_np[:n] - cfg[0] - cfg[1]).all())
