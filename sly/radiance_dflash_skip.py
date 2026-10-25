"""Drafter sliding-window prefill skip: the decision (RADIANCE_DFLASH_PREFILL_WINDOW_SKIP).

Hooked at the top of `DFlashSpeculator.propose` by sly/patch_dflash_prefill_skip.py. While a request
is still prefilling, the scheduler drops the draft tokens (`update_draft_token_ids`: "Ignore draft
tokens for prefill chunks"), and the DFlash2 drafter only reads context K/V inside its sliding
window (all layers `sliding_attention`; its conv and candidate selector see query rows only). So a
chunk that ends `window + margin` tokens before the end of the prompt needs no `fc`, no context-K/V
fill and no draft step: `propose` returns early. The final chunks, decode and spec-verify steps
never match the rule and run unchanged.

margin = max(RADIANCE_DFLASH_PREFILL_WINDOW_SKIP_MARGIN, 2 drafter KV blocks): the block that holds the
last skipped position (partly unwritten) must lie wholly below the first block the attention can
touch, whatever its tile size is.

A step also never skips when its prefill chunk is shorter than `window + margin`: the chunk after a
skipped one must fill the whole window, or its draft read reaches back into the skipped range (a
discarded draft only, but it could read unwritten - possibly NaN - slots).

Off (stays stock) with prefix caching on (a cached drafter block of a skipped position would hold
garbage for the next request that shares the prefix), with any non-sliding drafter layer, without a
configured window, and with data parallelism (the draft step is a collective).

Partial skip (RADIANCE_DFLASH_PREFILL_PARTIAL_SKIP, `context_rows`, hooked by
sly/patch_dflash_prefill_partial.py): the same rule at row granularity, for a batch of one request
in prefill. Only positions >= prompt_len - window - margin are ever read by the final chunk's draft, so
`fc` and the context-K/V fill run over the chunk's rows from there on, and a chunk that is not the
prompt's last skips its (dropped) draft step. With the DFlash2 drafter (window 2048, block 880, so margin
1760) a 6,000-token prompt at chunk 4096 sends 3,808 rows through the drafter instead of 6,000; the whole
skip never acts there. Kept rows are never fewer than
`_min_rows()` (513, the large-M tile of every drafter GEMM), unless the chunk itself is shorter: so the
kept rows take the same kernel config as the whole chunk and come out bit-identical. Same gates
as the whole skip; the knobs are independent and compose (whole skip first).
"""
import os
import sys

_SW = "sliding_attention"
# The large-M GEMM regime of the drafter's W4A16 linears (fc and the fused context-K/V projection):
# M > 512 leaves the M buckets of `triton_w4a16_skinny_fmt_gemm`, M >= RADIANCE_DFLASH_FC_TILED_MIN_M
# takes the one wide tile. Rows sliced off a chunk stay at or above both so the tile does not change.
_LARGE_M = 513


def _drafter_block(spec):
    """Block size of the KV group(s) holding the drafter's attention layers. The target's hybrid
    (mamba-aligned) group can have a 64k block and says nothing about the drafter's window. Without
    a reliable match: the smallest sliding-window group, else 0 (the env margin alone)."""
    groups = spec.kv_cache_config.kv_cache_groups
    mine = getattr(spec, "draft_attn_layer_names", None) or ()
    sizes = [g.kv_cache_spec.block_size for g in groups if set(g.layer_names) & set(mine)]
    if not sizes:
        sizes = [g.kv_cache_spec.block_size for g in groups
                 if getattr(g.kv_cache_spec, "sliding_window", None)]
        sys.stderr.write(f"[radiance.dflash_skip] drafter KV group not identified; sliding-group blocks {sizes}\n")
    return max(sizes, default=0)


def _config(spec):
    """(window, margin), or () when the skip is off or unsafe. Decided once per speculator."""
    if os.environ.get("RADIANCE_DFLASH_PREFILL_WINDOW_SKIP", "0") != "1":
        return ()
    return _window_margin(spec, "RADIANCE_DFLASH_PREFILL_WINDOW_SKIP")


def _partial_config(spec):
    """(window, margin, min_rows), or () when the partial skip is off or unsafe."""
    if os.environ.get("RADIANCE_DFLASH_PREFILL_PARTIAL_SKIP", "0") != "1":
        return ()
    cfg = _window_margin(spec, "RADIANCE_DFLASH_PREFILL_PARTIAL_SKIP")
    return cfg + (_min_rows(),) if cfg else ()


def _min_rows():
    try:
        tiled = int(os.environ.get("RADIANCE_DFLASH_FC_TILED_MIN_M", "0"))
    except ValueError:
        tiled = 0
    return max(_LARGE_M, tiled)


def _window_margin(spec, knob):
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
        sys.stderr.write(f"[radiance.dflash_skip] {knob} ignored: {why}\n")
        return ()
    window = int(window or 0)  # checked above; the `or` narrows the config's Optional type
    block = _drafter_block(spec)
    margin = max(int(os.environ.get("RADIANCE_DFLASH_PREFILL_WINDOW_SKIP_MARGIN", "512")), 2 * block)
    sys.stderr.write(f"[radiance.dflash_skip] {knob} ON: window {window}, margin {margin}"
                     + (f", chunks >= {window + margin} tokens\n" if knob.endswith("WINDOW_SKIP") else "\n"))
    return window, margin


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
    # A skipped chunk must be covered by the next written chunk inside the window: with fewer than
    # window + margin tokens this step, that chunk's draft read would reach into the skipped range.
    if int(input_batch.num_scheduled_tokens[:n].max()) < cfg[0] + cfg[1]:
        return False
    end = input_batch.num_computed_prefill_tokens_np[:n] + input_batch.num_scheduled_tokens[:n]
    return bool((end <= input_batch.prefill_len_np[:n] - cfg[0] - cfg[1]).all())


def context_rows(spec, input_batch, dummy_run=False, is_profile=False):
    """(r0, draft) when the partial skip changes this step, else None (stock path).

    r0: the first batch row whose context K/V the final draft can read (rows below it get no `fc`
    and no K/V); r0 == num_tokens means the whole chunk is old. draft: False on a chunk that is not
    the prompt's last, whose draft tokens the scheduler drops. Acts only on a batch of one request
    in prefill (its rows are then positions c0..c1-1 in order); decode, spec-verify and
    multi-request steps return None.
    """
    if dummy_run or is_profile:  # before _partial_config, as in skip_context
        return None
    cfg = getattr(spec, "_radiance_pf_partial", None)
    if cfg is None:
        cfg = spec._radiance_pf_partial = _partial_config(spec)
    if not cfg or input_batch.num_reqs != 1:
        return None
    window, margin, min_rows = cfg
    c0 = int(input_batch.num_computed_prefill_tokens_np[0])
    n = int(input_batch.num_scheduled_tokens[0])
    prompt = int(input_batch.prefill_len_np[0])
    if c0 >= prompt or n != int(input_batch.num_tokens):
        return None
    c1 = c0 + n
    draft = c1 >= prompt
    kept = c1 - max(c0, prompt - window - margin)
    if kept <= 0:  # a whole old chunk (the whole skip's case, without its chunk-size floor)
        return (n, False)
    # Never fewer than min_rows unless the chunk is shorter: same GEMM tile as the whole chunk.
    r0 = n - min(n, max(kept, min_rows))
    return None if r0 == 0 and draft else (r0, draft)
