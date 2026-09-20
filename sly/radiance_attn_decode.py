"""Decode-attention tune for the DFlash/MTP verify batch: gfx1201 (R9700), aiter >= 0.1.21 config API.

Replaces the decode half of radiance_kernels.install_attn_config_hook. That hook wrapped
`select_3d_config` / `select_2d_config`, which aiter 0.1.21.post2 no longer has (its config lookup is
now `get_unified_attention_config(op, params, backend, arch)` over JSON tables), so on this image it
died with AttributeError at every start and the tune never applied; the message sat in the accept
gate's `known_warnings` from 0.2.3 on and was read as noise. The verify batch therefore ran on aiter's
flat gfx1201 table: BLOCK_M 16, TILE 64 (from kv_split), 2 warps, NUM_SEGMENTS 32.

What is wrong with that table for this workload (fp8 q + fp8 KV, head 256, 24 q / 4 kv heads, verify
width 8; measured with rocprofv3 on 0.2.6, one request, k=7):

    ctx        0.1k   32.8k   65.6k    ~98k
    attn/call  21 us  1.63 ms 3.48 ms  ~5.1 ms   x16 full-attention layers = the whole step growth

1. BLOCK_M. unified_attention() derives it from the GQA ratio alone (16 while the ratio is <= 16), so
   BLOCK_Q = 16 // 6 = 2 and a width-8 verify is chopped into 5 q-blocks per sequence, each of which
   streams that sequence's whole KV cache. A 64-row block holds all 8 tokens x 6 heads and reads the
   KV once. Widening pays from a verify width of 6 (36 of 64 rows, 56%) up; below that a 32-row block
   is as good or better, and a 16-row block never won. (The retired hook widened only at >= 80% fill;
   with the rest of the cell tuned the crossover measured at ~50%.)
2. The shape of the kernel itself (TILE, warps, stages, waves_per_eu; reduce warps): occupancy, not
   arithmetic, limits it -- a 64-token fp8 K/V tile double-buffered is 32 KB of the 64 KB LDS, and
   aiter's flat table (TILE 64, 2 warps, 2 stages) leaves the machine mostly idle. TILE 32 / 4 warps /
   1 stage / waves_per_eu 2 for the 64-row block, TILE 16 / 2 warps / 1 stage / waves 2 for the
   32-row one; waves_per_eu 6, which the retired hook carried, is 5-15% slower.
3. The split-KV count. kv_split sizes NUM_SEGMENTS from the launch's program count (taken at the
   *stock* BLOCK_Q, so off as soon as (1) changes the launch) and aiter's 16-workgroups-per-CU target is
   4x too many: every split writes and re-reads tokens x 24 heads x 1 KB of partials. Measured over
   4k-128k in a graph captured at max_model_len, the best fixed count is 32 with one sequence (4
   programs) and 16 from two up; 128 splits (what stock would pick for the wide launch) costs
   10-25% at 1 sequence.

Every one of those is a config value, so the tune lives in the config lookup and nowhere else: this
module wraps `get_unified_attention_config` and overrides the "attn_3d", "kv_split" and "reduce"
results for the regime below. `params` carries everything the decision needs (num_tokens, num_seqs,
num_queries_per_kv, all_decode, max_seqlen_q/k, dtypes), so there is no frame inspection and no launch
shim; unified_attention()'s own wrapper derives BLOCK_Q from the BLOCK_M returned here and sizes the
grid from it. TILE_SIZE and NUM_SEGMENTS are decided once in kv_split and handed to both the attention
and the reduce launch, so the two can not drift apart (patch_unified_attention_lds returns the
LDS-clamped TILE_SIZE the same way).

`use_2d_kernel` is wrapped as well. It runs *before* any config is looked up and reads
`params.num_2d_prgms`, again at the stock BLOCK_Q, so with 7 or 8 verifying sequences (5 q-blocks x 4
kv heads x 7 = 140 > 4 x 32 CUs) stock sends the batch down the 2D kernel -- no KV split at all -- at
every context length, from the CUDA-graph capture size that first crosses the line.

Capture semantics (read this before changing a number): FULL CUDA graphs are captured with
`max_seqlen_k = max_model_len` (gpu_model_runner, for_cudagraph_capture) and seq_lens filled with 1
(RocmAiterUnifiedAttentionMetadataBuilder). The launch geometry -- BLOCK_M, NUM_SEGMENTS, the 2D/3D
choice -- is therefore fixed at capture for the shape (num_tokens, num_seqs) and replayed at every
depth from 1 token to max_model_len. A split count "derived from the depth" only ever sees the depth
limit; what matters is the count that is good over the whole range, and bench_decode_attn.py measures
exactly that (capture at max_model_len, replay at 4k..128k). Eager calls (piecewise steps) get the same
rule with their real max_seqlen_k.

Untouched, deliberately:
  * ALL_DECODE (max_seqlen_q == 1: the first decode step after a prefill, a target without drafts). It
    stays on aiter's table (the drafter's own attention is vLLM's TRITON_ATTN, not this file).
  * every other dtype / head size / sliding-window / shuffled-KV layer, and any batch whose longest query
    does not fit one 64-row block (max_seqlen_q x 6 > 64: prefill chunks): plan() returns None.
  * the 2D prefill config. The retired hook's fp8 prefill tune (TILE 16, waves 1) measured 2-12% slower
    than aiter's Q_GEQ_256 entry on this card (2048-token chunks at 0 / 32k / 98k of past KV), so it is
    not ported.

Knobs (read at import):
  RADIANCE_ATTN_DECODE_TUNE      1 (default) | 0 = aiter's stock tables (A/B control)
  RADIANCE_ATTN_DECODE_WIDE      auto (default) | 0 = never widen | 1 = widen every verify batch that fits
  RADIANCE_ATTN_DECODE_MIN_FILL  fraction of the 64 rows a verify batch must fill for `auto` (0.5)
  RADIANCE_ATTN_DECODE_3D        1 (default) = verify batches take the 3D kernel above 512 tokens when the
                                 launch the plan makes is small enough; 0 = aiter's 2D/3D choice
"""
import functools
import os
import sys

ENABLED = os.environ.get("RADIANCE_ATTN_DECODE_TUNE", "1") != "0"
WIDE = os.environ.get("RADIANCE_ATTN_DECODE_WIDE", "auto")
MIN_FILL = float(os.environ.get("RADIANCE_ATTN_DECODE_MIN_FILL", "0.5"))
FORCE_3D = os.environ.get("RADIANCE_ATTN_DECODE_3D", "1") != "0"

HEAD = 256                 # only head size these numbers were measured at
WIDE_ROWS = 64             # rows of the wide block (BLOCK_M)

# One cell per regime: the attention kernel's (BLOCK_M, TILE, num_warps, num_stages, waves_per_eu).
# Measured with sly/bench_decode_attn.py on the R9700 (README, "Decode attention"): the wide cell for a
# verify batch of >= MIN_FILL * 64 rows, the 32-row cell below that. The reduce kernel wants 4+ warps
# when it merges many splits (1 warp is 5x slower at 128, 2 is 25% slower) and does not care at 16-32.
_WIDE = dict(block_m=64, tile=32, warps=4, stages=1, waves=2)
_NARROW = dict(block_m=32, tile=16, warps=2, stages=1, waves=2)
_REDUCE = dict(r_warps=4, r_stages=1, r_waves=2)

# Split-KV count = machine-fill over the launch's own programs, ~SPLIT_PER_CU workgroups per CU, a power
# of two inside [SPLIT_MIN, SPLIT_MAX]: 32 at 4 programs (one sequence), 16 beyond.
SPLIT_PER_CU = 4
SPLIT_MIN = 16
SPLIT_MAX = 32

# Bench-only overrides (bench_decode_attn.py), never set in a serve. FORCE is a dict of plan fields
# (block_m, tile, warps, stages, waves, segments, r_warps, r_stages, r_waves, force3d) that bypasses the
# regime gate so a sweep can put any cell on any shape; "segments" is an int, "auto" for the shipped
# rule or "pcu<N>" for the machine-fill rule at N workgroups per CU (unclamped). FORCE_2D updates the
# "attn_2d" config (prefill / 2D decode).
FORCE = None
FORCE_2D = None

_state = {"fp8": None, "logged": False, "failed": False}


def _next_pow2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def _regime(params):
    """True when this launch is a verify batch the tune covers."""
    fp8 = _state["fp8"]
    return (fp8 is not None
            and params.q_dtype == fp8 and params.kv_cache_dtype == fp8
            and params.head_size == HEAD
            and not params.all_decode
            and params.max_seqlen_q > 1
            and params.max_seqlen_q * params.num_queries_per_kv <= WIDE_ROWS   # one 64-row block holds a verify batch
            and params.sliding_window <= 0
            and not params.shuffled_kv_cache
            and not params.use_qq_bias and not params.use_alibi_slopes
            and params.num_queries_per_kv > 0)


def _prgms(block_m, num_tokens, num_seqs, nqpkv, num_kv_heads):
    """Programs of the 3D launch at this BLOCK_M -- unified_attention()'s own bound, floor(tokens /
    BLOCK_Q) + seqs q-blocks per kv head."""
    return (num_tokens // max(1, block_m // nqpkv) + num_seqs) * num_kv_heads


def _segments(prgms, max_seqlen_k, tile, num_sms, per_cu=None):
    """Machine-fill split count: ~num_sms * per_cu workgroups over the launch's programs, a power of two
    in [SPLIT_MIN, SPLIT_MAX], never more than the KV has tiles (real depth in eager calls, max_model_len
    in a captured graph). Same shape as aiter's compute_segment_params, retuned."""
    want = -(-int(num_sms) * int(per_cu or SPLIT_PER_CU) // max(1, int(prgms)))
    limit = min(SPLIT_MAX, -(-int(max_seqlen_k) // int(tile)))
    return _next_pow2(max(min(SPLIT_MIN, limit), min(limit, want)))


@functools.lru_cache(maxsize=512)
def _plan(num_tokens, num_seqs, max_seqlen_q, max_seqlen_k, nqpkv, num_kv_heads, num_sms):
    rows = max_seqlen_q * nqpkv
    if WIDE == "0":
        wide = False
    elif WIDE == "1":
        wide = rows <= WIDE_ROWS
    else:
        wide = MIN_FILL * WIDE_ROWS <= rows <= WIDE_ROWS
    cell = dict(_WIDE if wide else _NARROW)
    cell["prgms"] = _prgms(cell["block_m"], num_tokens, num_seqs, nqpkv, num_kv_heads)
    cell["segments"] = _segments(cell["prgms"], max_seqlen_k, cell["tile"], num_sms)
    cell.update(_REDUCE)
    cell["force3d"] = FORCE_3D
    return cell


def plan(params):
    """The plan (a dict of launch fields) for this launch, or None to leave aiter's config alone."""
    if not ENABLED:
        return None
    if FORCE is not None:
        p = dict(FORCE)
        p.setdefault("force3d", True)
        p["prgms"] = _prgms(p["block_m"], params.num_tokens, params.num_seqs,
                            params.num_queries_per_kv, params.num_kv_heads)
        if isinstance(p["segments"], str):        # "auto" (shipped rule) or "pcu<N>"
            per_cu = int(p["segments"][3:]) if p["segments"].startswith("pcu") else None
            p["segments"] = _segments(p["prgms"], params.max_seqlen_k, p["tile"], params.num_sms, per_cu)
        return p
    if not _regime(params):
        return None
    # the depth only matters below SPLIT_MAX * TILE tokens (eager calls); capping it keeps the cache small
    return _plan(params.num_tokens, params.num_seqs, params.max_seqlen_q, min(int(params.max_seqlen_k), 4096),
                 params.num_queries_per_kv, params.num_kv_heads, params.num_sms)


def _apply(op, cfg, p, params):
    if op == "attn_3d":
        cfg["BLOCK_M"] = p["block_m"]
        cfg["num_warps"] = p["warps"]
        cfg["num_stages"] = p["stages"]
        cfg["waves_per_eu"] = p["waves"]
    elif op == "kv_split":
        cfg["TILE_SIZE"] = p["tile"]
        cfg["NUM_SEGMENTS"] = p["segments"]
        if not _state["logged"]:
            _state["logged"] = True
            sys.stderr.write(
                f"[radiance] decode attn plan: BLOCK_M={p['block_m']} TILE={p['tile']} "
                f"splits={p['segments']} warps={p['warps']} stages={p['stages']} waves={p['waves']} "
                f"reduce={p['r_warps']}/{p['r_stages']}/{p['r_waves']} "
                f"(seqs={params.num_seqs} tokens={params.num_tokens} kv={params.max_seqlen_k})\n")
            sys.stderr.flush()
    elif op == "reduce":
        cfg["num_warps"] = p["r_warps"]
        cfg["num_stages"] = p["r_stages"]
        cfg["waves_per_eu"] = p["r_waves"]


def install():
    """Wrap the config lookup and the 2D/3D switch. True when this aiter has the config-table API (the
    tune is then installed, or deliberately skipped/off and logged), False when it does not: the caller
    keeps its select_3d_config path."""
    import torch
    try:
        import aiter.ops.triton.attention.unified_attention as UA
        import aiter.ops.triton.utils.unified_attention_utils as UU
    except Exception:
        return False
    if not hasattr(UU, "get_unified_attention_config") or not hasattr(UA, "use_2d_kernel"):
        return False
    if getattr(UU.get_unified_attention_config, "_radiance_decode_tune", False):
        return True
    if getattr(UA, "DEVICE_ARCH", None) != "gfx1201":
        sys.stderr.write("[radiance] decode attn tune skipped: not gfx1201\n")
        return True
    if not ENABLED:
        sys.stderr.write("[radiance] decode attn tune off (RADIANCE_ATTN_DECODE_TUNE=0): aiter stock tables\n")
        return True
    _state["fp8"] = torch.float8_e4m3fn

    orig_cfg, orig_2d = UU.get_unified_attention_config, UA.use_2d_kernel

    def get_unified_attention_config(op, params, backend="triton", arch=None):
        cfg = orig_cfg(op, params, backend, arch)
        if backend == "triton" and arch is None and op == "attn_2d" and FORCE_2D is not None:
            cfg.update(FORCE_2D)
            return cfg
        if backend != "triton" or arch is not None or op not in ("attn_3d", "kv_split", "reduce"):
            return cfg
        try:
            p = plan(params)
            if p is not None:
                _apply(op, cfg, p, params)
        except Exception as e:  # a tune must never take a serve down: fall back to aiter's own numbers
            if not _state["failed"]:
                _state["failed"] = True
                sys.stderr.write(f"[radiance] decode attn tune fell back to stock config: {e!r}\n")
            return orig_cfg(op, params, backend, arch)
        return cfg

    def use_2d_kernel(params):
        use_2d = orig_2d(params)
        if use_2d and FORCE_3D and params.max_seqlen_k > 512:
            try:
                p = plan(params)
                # judged at the launch the plan will actually make, not at the stock BLOCK_Q
                if p is not None and p["force3d"] and p["prgms"] <= params.target_num_prgms:
                    return False
            except Exception:
                pass
        return use_2d

    get_unified_attention_config._radiance_decode_tune = True
    UU.get_unified_attention_config = get_unified_attention_config
    UA.use_2d_kernel = use_2d_kernel
    rebound = 0
    for mod in list(sys.modules.values()):
        if mod is not None and getattr(mod, "get_unified_attention_config", None) is orig_cfg:
            mod.get_unified_attention_config = get_unified_attention_config
            rebound += 1
    sys.stderr.write(f"[radiance] decode attn tune installed (verify-batch config, {rebound} module binding"
                     f"{'s' if rebound != 1 else ''}, wide={WIDE} min_fill={MIN_FILL})\n")
    sys.stderr.flush()
    return True
