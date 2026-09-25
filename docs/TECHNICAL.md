# Technical changes in detail

The deep dives behind the short list in the [README](../README.md#what-this-image-changes-and-why): what each change does, why, and how it was measured. Chronological by image version; the [CHANGELOG](../CHANGELOG.md) has the per-release summary.

## MXFP4 on gfx1201 (0.1.0 – 0.1.2)

- **Quark/MXFP4 loader for vLLM 0.29.0** (`sly/patch_quark_mxfp4.py`, 5 hunks). ggz14's patches
  targeted 0.27.1; vLLM 0.29 added two more CDNA-only gates around the AITER custom op, so two hunks
  are new. `RADIANCE_MXFP4=1` unlocks AITER's Triton `gemm_afp4wfp4` (native W4A4) — without it every
  MXFP4 linear layer silently falls back to `EmulationMxfp4LinearKernel` (bf16 dequant + `F.linear`).
- **AITER `gemm_afp4wfp4` config for gfx1201** (`sly/mxfp4-configs/`). AITER has no gfx1201 tuning
  for this GEMM family and aborts with `AssertionError` without a config file; the gfx950 default it
  would otherwise use needs 67–100 KiB of LDS and crashes the EngineCore on the first request
  (`OutOfResources`, RDNA4 has 64 KiB). Fixed by probing every M bucket on the card
  (`num_stages` 3→2), then tuned the M ≤ 8 decode bucket.
- **W4A8 HIP kernel layered on top** (`sly/mxfp4/radiance_mxfp4_fp8.hip`,
  `sly/mxfp4/radiance_mxfp4.py`). `RADIANCE_MXFP4_W4A8=1` registers ggz14's fp8-WMMA kernel at the
  head of vLLM's ROCm kernel list; it takes the shapes it supports and everything else falls back to
  the AITER path — both are needed, layered, no either/or.
- **`DEC_MAX_N` 32768 → 36864.** Qwen3.8-27B at TP=1 has a fused `gate_up_proj` with N = 34816, one
  above the old limit, so the decode-critical GEMM was silently routed to the folded prefill kernel
  (64 calls/step at ~420 µs instead of ~190 µs = 30 of 72 ms per DFlash step). Scratch and
  block-counter sizing in the plugin grew to match. DFlash step 67 → 48 ms.
- **"Track A" decode routing.** `RADIANCE_MXFP4_W4A8_MIN_M=0` + `RADIANCE_MXFP4_DECODE_MAX_M=64` send
  the decode shapes (M ≤ 64 = 8 sequences × 8 tokens at k=7) to the HIP split-K decode kernel instead
  of AITER's Triton GEMM: 1.85× on the decode GEMMs (6.9 → 12.9 tok/s in the isolated A/B, before
  the other work below). Raised to `128` afterwards: the kernel choice is baked in at CUDA-graph
  capture time and the capture sizes 72–128 (prefill+decode mixed steps, short prefills — ~20 % of
  the iterations at concurrency 8/16) were still hitting the folded prefill kernel. Conc 8/16
  348/357 → 368/371 tok/s, TTFT p50 −14…−23 %, exact-reference check 0 wrong, +36 MiB scratch.

## DFlash2 speculative decoding with a W4A16 drafter (0.1.2, 0.1.4)

- **`sly/patch_dflash_w4_packed.py`** — lets the DFlash drafter be a compressed-tensors W4A16
  checkpoint (`qkv_proj` has `weight_packed`, no raw `.weight`; deferred and dequantised through its
  own forward like the fp8 case).
- **`sly/patch_gdn_nonspec_mask.py`** — upstream's `patch_gdn_metadata` numpy path leaves
  `non_spec_sequence_masks_cpu` unset → `UnboundLocalError` at engine init as soon as
  `--speculative-config` is given.
- **gfx1201 tile table for the drafter GEMMs** (`sly/patch_w4a16_tiles.py`,
  `sly/bench_w4a16_tiles.py`). vLLM's `rdna_hybrid_w4a16.py` routes M ≤ 5 to the HIP skinny kernel and
  everything else to a Triton kernel whose gfx12x heuristic was tuned on Llama-3.1-8B: at M ≤ 32 it
  picks 16×16 tiles (2176 workgroups for the 34816-wide `gate_up`, 267 GB/s). A DFlash drafter never
  sees M ≤ 5 (block size 8 → M = 8 × sequences). The table is keyed by `(group_size, K, N, M bucket
  8/16/32/40/64)` for the four drafter shapes and was measured DRAM-cold on the R9700: `gate_up`
  1.7–3.5×, `down` 1.5–2.6×, `qkv` 1.5–2.4× at M ≤ 32. Step gap 43.4 → 42.0 ms.
  `RADIANCE_W4A16_TILES=0` restores the stock heuristic (A/B without a rebuild).

## fp8 lm_head (0.1.3)

`sly/mxfp4/radiance_lmhead_fp8.py` + `sly/patch_lmhead_fp8.py`. The Quark checkpoint lists
`lm_head` in its `exclude` list, so the 248320 × 5120 vocabulary projection ran as a bf16 GEMM —
2.54 GB of weight traffic per call, and DFlash calls the shared head twice per step (8.6 of ~49 ms).
`RADIANCE_LMHEAD_FP8=1` quantises the weight per output channel to fp8 after loading and runs it
row-wise through `torch._scaled_mm` (hipBLASLt) with per-token fp8 activations. Step 48 → 44.5 ms,
1.27 GiB returned to the KV cache. GSM8K (cot, zero-shot, greedy, 200 tasks): 0.830 ± 0.027 vs
0.835 ± 0.026 for bf16 — no measurable loss.

## int4 lm_head (0.1.5)

`sly/mxfp4/radiance_lmhead_int4.py` + `sly/patch_lmhead_int4.py`, opt-in via `RADIANCE_LMHEAD_INT4=1`
(takes precedence over the fp8 knob). The fp8 head already runs at the bandwidth ceiling (1.27 GB per
call at 543 GB/s = 2.34 ms, twice per DFlash step = 11 % of the 42 ms step), so the only lever left
is bytes: int4 with group-128 bf16 scales is 656 MB per call. The weight is quantised after loading
(symmetric, per-group MSE clip search over ratios 1.0…0.8, 1.6 s), packed with vLLM's
`pack_int4_exllama_shuffle` and applied on the same W4A16 kernel path as the drafter
(`torch.ops.vllm.rdna_hybrid_w4a16_apply`: HIP skinny kernel at M ≤ 5, else the Triton kernel with
five lm_head rows added to the gfx1201 tile table — the stock 16-column tiles would make it *slower*
than fp8: 2623 vs 2459 µs at M = 8, with the table 1305 µs / 502 GB/s). No activation quant launch.
Offline error on the real weight: 10.8 % of the logit RMS vs 3.7 % for fp8 (int4 ≈ 3× fp8, as
expected from the formats). The error does not reach the argmax on real decode distributions: in a
same-window A/B against the fp8 head GSM8K went 0.850 → 0.840 (5 vs 3 paired flips, noise) and
the mean accepted tokens per step 4.33 → 4.31. The freed 0.6 GB goes to the KV cache (133k → 141k
tokens).

## Fused norm / activation + fp8 quant (0.1.6)

`sly/mxfp4/radiance_fused_norm.py` + `sly/patch_fused_norm_quant.py`, opt-in via
`RADIANCE_FUSED_NORM_QUANT=1`. Every W4A8 GEMM input used to be quantised by its own
`scaled_fp8_quant` launch after an Inductor-generated norm/activation kernel. Three hand-written HIP
kernels in `radiance_mxfp4_fp8.hip` do the elementwise step and the per-token fp8 quant (scale =
max(amax/448, 1/(448·512)), e4m3 codes) in one pass and hand `(q, scale)` straight to
`torch.ops.radiance.mxfp4_linear_pq`:

- `radiance_add_rms_quant` — decoder `input_layernorm` / `post_attention_layernorm` (Gemma 1+w,
  residual add in fp32) → `qkv_proj`, `in_proj_qkvz` + `in_proj_ba`, `gate_up_proj`.
- `radiance_silu_mul_quant` — MLP `silu(gate) * up` → `down_proj` (MAXG raised 5 → 9 for the
  17408-wide intermediate; scratch sizing only).
- `radiance_gdn_norm_quant` — GDN gated per-head RMS norm (`((x·rsqrt)·w)·silu(z)`) → `out_proj`.

A site is fused only if every consumer is a folded radiance W4A8 layer at TP=1; the per-fusion switches
`RADIANCE_FUSED_NORM_QUANT_ADD_RMS/_SILU/_GDN` isolate one kernel. The knobs are added to vLLM's
`compile_factors()` so a flip never replays a stale AOT graph. Production since 2026-09-16; measurements:
see *Results*.

## upstream/ggz14 merged (0.2.0)

`main` carries the full [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4)
history (243 commits since the common base `e1c99aa`). What is **active** in the image did not change:
the Dockerfile keeps our pins, our patch loop and the `sly/` MXFP4 stack; ggz14's own top-level
`radiance_mxfp4.py` / `patch_quark_mxfp4.py` / `radiance_mxfp4_fp8.hip` / `mxfp4-configs/` stay in the
tree unused (the `sly/` copies are the evolved, production-validated variants). ggz14's ~24 extra
`patch_*.py` are listed with a reason in `ci/unused_patches.txt`; most of them are applied by ggz14 at
container start (`serve-mxfp4.sh`) together with libr4d additions from `r4d_radiance_extras.patch`
(narrow bf16/fp16 SSM state, fused GDN decode step, 3-rank all-reduce), which this image does not
build. Auto-merged changes to files that *are* in the image (`radiance_gdn.py` narrow-state /
fused-update binding — resolves to `None` on the pinned libr4d, so bf16 SSM state keeps falling back
to FLA exactly as before; `patch_gdn_metadata.py` / `patch_r4d.py` gained 0.29 anchor variants via
`_patchlib.apply_any`; `radiance_r4d_attn.py` 0.29 KV-layout hook; `radiance_gemm.py` paroquant
fallback that is never taken; `radiance_drafthead.py` fp8-head support behind the unset
`RADIANCE_FAST_DRAFT`) were reviewed as behaviour-neutral for this configuration. Kept ours instead
of ggz14's: `patch_dflash_fused_kv_fp8.py` (`sly/patch_dflash_w4_packed.py` anchors on its
`_DFLASH_FP8` text) and the `radiance_preamble.py` banner. Candidates for follow-up steps, each with
its own A/B: libr4d extras (bf16 SSM state on R4D instead of FLA), `patch_kv_group_size`,
`patch_gdn_shared_build`, `patch_topk_*`, `patch_dflash_selector_topk`, `patch_dynwidth`.

## Gated-delta-net / attention

- **`sly/patch_short_prefill.py`** — a 1-token prefill was misclassified as decode in the GDN
  metadata builder (wrong state for the first token of a short prompt).
- **`patch_unified_attention_lds.py` ported to aiter 0.1.21.post2** — AITER replaced its
  `select_3d_config`/`select_2d_config` elif chains with JSON-driven config tables between the
  version upstream patched and 0.1.21.post2; the patch was rewritten (6 hunks): LDS clamp for gfx1201,
  `TILE_SIZE` return for `reduce_segments`, gfx1201-gated bf16 3D decode tuning.
- **bf16 SSM state** (runtime flag, `--mamba-ssm-cache-dtype bfloat16`): halves the GDN state pages
  (attention block 1664 → 896 tokens), KV 101k → 133k tokens on the 32 GB card, concurrent-request
  ceiling ~6 → ~12. GSM8K unchanged (0.835), needle-in-haystack 3/3 at 12k and 24k tokens.

## Decode attention: the verify batch on aiter's config tables (0.2.7)

Single-stream decode at long context was attention-bound: one kernel, `kernel_unified_attention_3d`
(head 256, 16 calls per step for the 16 full-attention layers), carried all of the growth. rocprofv3 on
0.2.6 (one request, k=7): 21 µs per call at 0.1k, 1.63 ms at 32.8k, 3.48 ms at 65.6k, ~5.1 ms at 98k,
i.e. step 42 → 82 → 110 → 147 ms — and a kernel at ~24 % of the DRAM bandwidth even after counting that
every q-block re-streams the sequence's KV.

The fix already existed: `radiance_kernels.install_attn_config_hook()` is the upstream tune for exactly
this shape, but it wrapped `UA.select_3d_config`, which aiter 0.1.21.post2 no longer has, so every start
logged `install_attn_config_hook failed: AttributeError(… 'select_3d_config')` and the tune never
applied. The accept gate had that line on its `known_warnings` list since 0.2.3, so a dead feature read
as noise (it is off the list now, and `[radiance] decode attn tune installed` is a *required* log marker
instead). `sly/radiance_attn_decode.py` ports the tune to the new API — a wrapper around
`get_unified_attention_config` and `use_2d_kernel`, no frame inspection or launch shim — and measures
what the old numbers only claimed:

- **`BLOCK_M`.** aiter derives it from the GQA ratio alone (16), so a width-8 verify (8 tokens × 6 heads)
  is 5 q-blocks per sequence and each re-reads the whole KV. A 64-row block reads it once. Measured with
  every block size on its own tuned cell and split count, the crossover is at **width 6 (56 % fill)**,
  not the retired hook's 80 %: the 64-row block wins 1.05–1.22× from width 6 to 9 (DFlash k=7 is width
  8, 75 %), the 32-row block wins or ties below, the 16-row block never wins.
- **Kernel shape.** aiter's flat table (TILE 64, 2 warps, 2 stages) leaves the machine idle. TILE 32,
  4 warps, 1 stage, **`waves_per_eu` 2** for the 64-row block — the retired hook's 6 is 5–15 % slower —
  and reduce with 4 warps (1 warp is 5× slower at 128 splits, 2 is 25 % slower).
- **Split-KV count.** FULL CUDA graphs are captured with `max_seqlen_k = max_model_len` (262144 here), so
  the geometry is fixed at capture and replayed at every depth; a "shape-derived" count only ever sees
  that. The bench therefore captures at 262144 and replays 4k…128k. Best fixed count: **32 with one
  sequence, 16 from two up** (each split writes and re-reads tokens × 24 heads × 1 KB of partials;
  stock's 16 workgroups per CU, i.e. 128 splits for the wide launch, costs 10–25 % at one sequence).
- **7–8 sequences.** `num_2d_prgms` is computed at the stock BLOCK_Q, so from 7 verifying sequences stock
  takes the 2D kernel (no KV split) at every depth; the wide 3D plan is 1.7–2.7× faster there.
- **Untouched:** ALL_DECODE, other dtypes/head sizes, prefill chunks. The retired hook's fp8 prefill tune
  (TILE 16, waves 1) measured 2–12 % *slower* than aiter's `Q_GEQ_256` entry at 0 / 32k / 98k of past KV,
  so the 2D prefill config stays aiter's.

Micro, `sly/bench_decode_attn.py` (µs per attention call incl. reduce, stock → tuned, width 8; the bench
reproduces the trace: stock 1.75 / 3.37 / 4.94 ms at 32k / 64k / 96k against 1.63 / 3.48 / ~5.1):

| sequences | 4k | 16k | 32k | 64k | 96k | 128k |
|---|---|---|---|---|---|---|
| 1 | 242 → 39 (6.2×) | 870 → 84 (10.3×) | 1752 → 146 (12.0×) | 3365 → 279 (12.1×) | 4941 → 410 (12.1×) | 6562 → 541 (12.1×) |
| 4 | 880 → 100 (8.8×) | 3405 → 307 (11.1×) | 6658 → 591 (11.3×) | 13142 → 1164 (11.3×) | 19679 → 1708 (11.5×) | 26248 → 2238 (11.7×) |
| 6 | 1391 → 152 (9.1×) | 5024 → 485 (10.4×) | 9881 → 923 (10.7×) | 19598 → 1809 (10.8×) | 29438 → 2633 (11.2×) | 39210 → 3492 (11.2×) |
| 8 (stock = 2D kernel) | 336 → 201 (1.7×) | 1369 → 648 (2.1×) | 2748 → 1246 (2.2×) | 5546 → 2299 (2.4×) | 8360 → 3387 (2.5×) | 11164 → 4447 (2.5×) |

Width 6 and 9 are 8.5–13.7× at 1–6 sequences (the same shape of table); the tuned kernel reaches ~490 GB/s
at 96k (77 % of the read peak).

End to end on the R9700 (2026-09-20, production arguments, same window, image 0.2.6 with the two files
bind-mounted over the image's — the 0.2.7 image itself is built by CI —, `RADIANCE_ATTN_DECODE_TUNE=0`
vs `1`; one stream, greedy, 256 tokens, `step` = decode seconds / spec-decode drafts, i.e. the engine's
step gap without any profiler):

| context | TG stock → tuned | step stock → tuned | accepted tokens/step |
|---|---|---|---|
| 0.1k | 113.0 → 113.5 tok/s | 37.6 → 37.5 ms | 4.30 → 4.27 |
| 33k | 42.2 → **65.9** tok/s (+56 %) | 65.0 → 40.7 ms (−24) | 2.77 → 2.71 |
| 65k | 32.9 → **72.9** tok/s (+122 %) | 93.3 → 42.7 ms (−51) | 3.13 → 3.12 |
| 99k | 27.5 → **68.0** tok/s (+147 %) | 120.2 → 44.7 ms (−76) | 3.32 → 3.05 |

The step grows by 7 ms over 99k of context instead of 83. In-situ kernel times (rocprofv3, tuned side; the
64-token windows also hold a ~1.4k-token uncached prefill tail, so only the decode kernel rows count):
`kernel_unified_attention_3d` (grid 128×4×32, i.e. one q-block and 32 splits) 5.8 / 129 / 245 / 363 µs per
call at 0.1k / 32.8k / 65.6k / 98.4k against 21 / 1630 / 3480 / ~5100 µs stock — 0.09 / 2.1 / 3.9 / 5.8 ms
per step (16 calls) against 0.34 / 26 / 56 / 82 — plus 0.06–0.14 ms of reduce. Concurrency and quality
unchanged: BetterBench fast config (conc-only) 104.5 → 106.8 tok/s at conc 1 and 369.7 → 381.8 at conc
8, all 24/24 requests OK; GSM8K (cot, zero-shot, greedy, 200 items) 0.865 → 0.850 ± 0.025 with 4 vs 1
paired flips, against the recorded 0.835. The accepted-tokens/step differences above are single greedy
streams (±0.2); the numerics differ only through split-KV order and the fp8 rounding of P per tile
(relative error against an fp32 reference 2.1–2.4 %, stock 2.1–2.7 %).

## Drafter attention: split-KV over the window (0.2.8)

Follow-up to the decode tune: what else grows (or costs) per step at long context? A single stream (k=7,
greedy, prod arguments) goes from 38.3 ms per step at 1k of context to 44.6 ms at 99k; the target's
attention is what 0.2.7 flattened, the rest is the DFlash2 drafter. Its five layers are all sliding-window
2048 (32 q / 8 kv heads, head 128, fp8 KV) and call vLLM's `unified_attention` once per layer per step with 8
queries per sequence. That launch is BLOCK_M 16 = BLOCK_Q 4 for GQA 4 with the 3D (split-KV) path closed for
more than one query, i.e. 24 workgroups on 32 CUs, each walking the whole window: 133 µs per call for one
sequence, 344 µs for eight, ~20–40 GB/s for 4 MB of K/V.

`sly/radiance_attn_drafter.py` rebinds `unified_attention` in `vllm.v1.attention.backends.triton_attn` and, for
that call only, (1) cuts the block table and `seq_len` down to the window in one tiny kernel (whole leading
blocks are dropped; positions are only used relative to the sequence end and RoPE is already in the cached K)
and (2) launches vLLM's own kernels in 3D mode over the slice: BLOCK_M 32 (= the 8 queries × 4 heads of a k=7
verify), TILE 32, 4 warps, 64 / 32 / 16 splits by sequence count (1 / 2–3 / more), then vLLM's `reduce_segments`.
The drafter's call is **non-causal** (`causal=False`: a key counts when it is `< seq_len` and within the window
of its query on either side); the first version of the gate assumed causal, declined the real call, and only
the "declined" log line gave that away — the launch now passes the flag through, and the accept gate requires
the plan line (see `ci/accept/README.md`). Untouched: every other head size / GQA ratio / dtype, per-sequence
causal, alibi / sinks / softcap / mm-prefix / chunked, prefill chunks (more than 16 queries per sequence), no
window; an exception in the tuned path falls back to vLLM's own launch for the rest of the process.

Micro, `sly/bench_drafter_attn.py` (µs per call and layer, window 2048, non-causal; CUDA graph captured at
`max_seqlen_k = 262144`, replayed at 1k…96k; the value is flat from 2k on):

| sequences | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| stock | 133 | 205 | 196 | 344 |
| split-KV | 38 (3.5×) | 70 (2.9×) | 105 (1.9×) | 188 (1.8×) |

Five layers per step: 0.5 ms (one sequence) to 0.8 ms (eight) shorter. Numerics are the stock kernel's
(relative error against an fp32 reference 0.0020–0.0022 against 0.0022–0.0023, window 2048 and 8192).

End to end (R9700, production arguments, same window, one stream, greedy, 200 tokens; mean of two corpus
slices and three tasks; prod = 0.2.7 without the tune, candidate = the same image with the file bind-mounted):

| context | 0.1k | 1k | 8k | 32k | 96k |
|---|---|---|---|---|---|
| step, prod | 37.3 ms | 38.3 | 39.5 | 41.7 | 44.6 |
| step, tuned | 37.1 ms | 38.0 | 38.6 | 40.0 | 43.8 |

The step is 0.2–1.6 ms (~1–4 %) shorter and the accepted tokens/step are unchanged within the ±0.2–0.6 a single
greedy stream scatters; KV pool unchanged (384,316 tokens). That is all the drafter's *kernel* has to give: the
decline of tokens/step over context (generation 4.6 → 3.7, summarising 3.4 → 2.8 from 1k to 99k; copying
collapses from 7.3 to 3.3 once the source is out of the drafter's window) is the drafter's acceptance, not its
attention time.

Negative result, kept so it is not tried again: with this launch a wider drafter window is cheap (+0.3–0.4 ms
per step at 8192, KV pool 384,316 → 356,996 tokens), but the checkpoint was trained with 2048 and gets worse,
not better, outside it. `dflash_config.swa_window_size` 8192 (a config copy of the drafter with hardlinked
weights) lowers the accepted tokens/step against window 2048 by 25–43 % for generation and 24–28 % for
summarising at 8k / 32k / 99k; copying is mixed (+6 % at 33k, −19 % at 99k). The window stays 2048.

## Prompt lookup on top of the DFlash draft (0.2.9)

What is left of the long-context TG decline after the two attention tunes is the drafter's acceptance: DFlash2
attends to the last 2048 tokens only, so text that repeats something further back — an edit's `old_string`, a
quoted file, a re-emitted block — is a copy the drafter cannot see. Single stream, greedy, tokens/step of the
running image: verbatim copy of the first 30 lines of the context 7.3 at 1k of context, 3.6 / 4.4 / 3.2 at 8k /
32k / 99k (the source has left the window), against 4.4–3.9 for free generation, which does not care.

`sly/radiance_lookup_draft.py` (`RADIANCE_LOOKUP_DRAFT`, default on) lets a suffix n-gram lookup over the *whole*
context replace the DFlash draft for a step, when it is very likely to win. It hooks the V2 model runner's DFlash2
speculator (the repo's older n-gram / dynamic-draft hooks attach to the V1 MTP proposer and are inert with DFlash2):

- **Scan.** One kernel per step looks in `req_states.all_token_ids` (vLLM keeps it in UVA host memory; 72 µs at 99k
  of context for one request, bounded by the KV pool at ~0.25 ms for any batch) for the longest earlier occurrence
  (up to 24 tokens, most recent on a tie) of the current suffix — only occurrences whose continuation starts more
  than the drafter's window (2048, from its config) back. A source inside the window is one the drafter sees,
  and it is better at it: without this rule lookup cost 5–8 % on edits at 1k of context.
- **Policy** (`ENTER` 8, `STAY` 3, `HOT` 3). Enter lookup mode on a match of ≥ ENTER tokens; stay in it while the
  last lookup step accepted ≥ HOT draft tokens and a match of ≥ STAY exists; otherwise the DFlash draft stays
  exactly as the graph wrote it. "Any match ≥ 3" loses 1–17 % on free generation (short matches rarely
  continue), simulated on recorded traces and the reason for the entry threshold.
- **Lossless.** The step verifies 1 + 7 rows as before. In lookup mode the tokens are the continuation of the match
  and the cached draft distribution `draft_logits` (the "probabilistic" draft method's q) is rewritten to a point
  mass on them, i.e. a deterministic draft: accept with p(token), on rejection resample from p without it.
  Any exception turns the override off for the process (`lookup draft: off after an error`).

Results (R9700, production arguments, image 0.2.8 with the file bind-mounted, one server, the override switched
by a file between runs of the same prompt — off, on, off — so the only difference is the override; 200 tokens):

| context | far copy (30 lines) | edit (30 lines, `self`→`this`) | JSON list of names | generation | summary |
|---|---|---|---|---|---|
| 1k | 7.31 → 7.31 | 6.23 → 6.23 | 5.60 → 5.60 | 4.47 → 4.47 | 3.43 → 3.43 |
| 8k | 3.64 → **6.98** (TG +89 %) | 3.67 → **5.83** (+57 %) | 4.98 → 4.98 | 4.44 → 4.44 | 3.62 → 3.62 |
| 32k | 4.42 → **7.02** (+61 %) | 4.00 → **6.51** (+61 %) | 3.60 → 3.51 (−3 %) | 4.44 → 4.26 (−5 %) | 3.37 → 3.37 |
| 99k | 3.24 → **7.12** (+116 %) | 3.25 → **6.75** (+105 %) | 4.92 → 4.72 (−3 %) | 3.91 → 3.91 | 2.80 → 2.80 |

(accepted tokens/step, mean of two corpus slices; the step time is 38.4 / 39.1 / 40.5 / 44.1 ms with and without the
override, unchanged within noise). A lookup step accepts 6.6–6.9 of 7 draft tokens. Where the override is not taken (generation, summary) the
tokens are identical to the run without it; the copy and edit generations are bit-identical to it in every case.
The few differences that exist (JSON list, one generation) sit at positions where the target's own distribution is
flat (p(top1) 0.19–0.68, top-2 gap 0.13–0.94 nat) and appear between two runs *without* the override as well
(verify-batch composition, the same ulp noise as across concurrency levels).

Lossless checks: the V2 rejection sampler with the point-mass distribution against the target distribution
(`sly/bench_lookup_draft.py --rejection`, 200k trials, drafts accepted 28 %, three token positions: max |z| 2.5, the
same as vLLM's own deterministic-draft path; a control with the point mass on the wrong tokens is off by |z| > 700);
temperature 1.0, top_p 1.0, 320 samples per mode over 8 concurrent requests at 8k of context asking for a far
copy: common-prefix length with the greedy copy 42.6 → 41.6 tokens, per-token deviation rate 1.37 % → 1.44 %
(z = +0.5, Mann-Whitney z = −0.07), accepted tokens/step 3.08 → 6.13. Eight concurrent requests on a 33k context
(four copies, two edits, one generation, one summary), greedy: accepted tokens/step of the batch 3.84 → 5.33,
6 of 8 generations identical to lookup-off (the same 6 of 8 between two runs without the override).
BetterBench fast config (conc 1 / 8, sampling at temperature 0.7 / top_p 0.95 / top_k 20; contexts under 2k tokens, so
the override is idle there), override off / on / off: conc 1 108.2 / 110.1 / 107.0 tok/s, conc 8 384.5 / 370.5 / 372.7,
24/24 requests each, accepted tokens/step 4.281 / 4.281 / 4.226 — inside the run-to-run spread of the two off runs;
GSM8K (200 items, greedy, chat) 0.830 off, 0.835 on (± 0.026).

Not done: the drafter is unchanged, so the loss of free generation with context (4.6 → 3.9 tokens/step from 1k to
99k) stays; this only helps where the text repeats. Knobs: see *Environment knobs*; `RADIANCE_LOOKUP_SWITCH=<file>`
(override only while the file exists) is the A/B tool used above.

## Prefill GEMM on fragment-tiled activations, fragment-order weights (0.3.0, 0.3.1)

Two switches that were in the image all along, neither of which did anything in production:

- **`RADIANCE_MXFP4_A_TILED_MIN_M=513`** (0.3.0). `radiance_mxfp4_fp8_gemm_atiled` reads the
  activation straight into the WMMA registers instead of staging a 256-row A tile through LDS (the
  largest single cost of the folded kernel, ablated at 24–32 %). It needs the activation in
  fragment-tiled layout, which only ggz14's `radiance_arnq.py` producers emitted — this image's
  fused norm/quant ops (`sly/mxfp4/radiance_fused_norm.py`) never asked for it, so the knob was
  inert. 0.3.0 makes `add_rms_quant` and `silu_mul_quant` write the tiled layout at M ≥ the
  threshold and register it for the consumer. The registry no longer pops on first use: the
  gated-delta-net input norm feeds two GEMMs (`in_proj_qkvz`, `in_proj_ba`), and a popped entry would
  have made the second one read the tiled buffer as row-major.
- **`RADIANCE_MXFP4_WPERM=1`** (0.3.1). Weights in WMMA fragment order, so a wave's weight read is
  128 contiguous bytes instead of sixteen rows K/2 apart. It cost 405 KV tokens (383,911 instead of
  384,316): the permute made a new parameter per layer while the checkpoint copy was still alive and
  left holes between the weights. 0.3.1 permutes back into the weight's own storage.

A/B on the production arguments (BetterBench decode + prefill + concurrency 1/4/8, 8 passes, 210 W,
same window, ABAB for the tiled path): prefill +11 … +13 % at 2k–64k from the tiled GEMM
(2,041 → 2,315 tok/s at 2k, 1,674 → 1,857 at 64k, repeat within 0.3 %), decode unchanged; WPERM on
top: step gap 38.5 → 37.2 ms, weighted decode +3.7 %, conc 1 +3.7 %, prefill +1 %. KV pool 384,316
in every warm start, greedy output (four prompts up to 7k tokens) byte-identical in every arm. Rejected in
the same windows: `RADIANCE_MXFP4_DECODE_NT=1` on top of WPERM (step 37.6 ms), libr4d's decode kernel
(`RADIANCE_MXFP4_R4D_DECODE_MAX_M=64`: not bit-identical, 810 KV tokens), `RADIANCE_MXFP4_TN4_MIN_M=4096`
(neutral), `patch_dflash_selector_topk` with K 24/32 (see `ci/unused_patches.txt`), `--no-async-scheduling`
(neutral), and a 36-cell sweep of aiter's 2D prefill attention config (stock cell within 1–5 % of the
best, see `sly/radiance_attn_decode.py`).

## Fused GDN decode on ROCm (HIP port of vLLM's kernel, default off)

vLLM 0.29 has a fused kernel for the post-conv half of the gated-delta-net decode of a speculative
verify batch (q/k l2norm, gating, delta-rule state update with the per-position write-back, gated
RMSNorm in one launch per request and value head), but builds it for NVIDIA archs only; on ROCm the
layer logs `Falling back to the Triton GDN decode path` and runs the FLA update kernel plus its glue
— about six extra launches per GDN layer, 48 layers per target forward. `sly/gdn/radiance_gdn_decode.hip`
is a HIP port of that kernel (cp.async → register prefetch, `__shfl_*_sync` → wave32 `__shfl_*`,
`__nv_bfloat16` → `__hip_bfloat16`; the math is unchanged), registered as
`torch.ops._C.fused_gdn_decode_post_conv_mtp` so vLLM's own switch picks it up.

Status: in the image, **off by default** (`RADIANCE_GDN_FUSED_DECODE=1` plus
`VLLM_GDN_DECODE_KERNEL=cuda` enables it). vLLM's reference test for the kernel passes on gfx1201
(22 cases: head ratios 1/2/3/4/8, bf16 and fp32 state, ragged batches, silu and sigmoid gate) plus
four production shapes in both gate modes — 30 of 30. One serving arm so far (300 W, prod args): step gap 35.32 → 34.59 ms
(−2.1 %), weighted decode 129.8 → 132.0 tok/s, prefill −0.5 %, concurrency within noise; the warm-start
KV check and a repeat arm are still open before it goes into the production unit. The fused path hands
`out_proj` a bf16 activation, so the fp8 `gdn_norm_quant` fusion is bypassed there.

## libr4d extras: GDN kernels for the bf16 state cache (0.3.5)

`--mamba-ssm-cache-dtype bfloat16` halves the gated-delta-net state traffic, but the pinned libr4d
only carried fp32-state GDN kernels, so `radiance_gdn` declined every GDN layer to the FLA Triton
path (`falling back to FLA ... state dtype torch.bfloat16` in the log). ggz14's libr4d extras carry
the narrow-state kernels (bf16 / fp16 state, fp32 accumulate, round-to-nearest-even stores). They
were written against libr4d b9e42ab; `sly/r4d/r4d_extras_rx10.patch` is the same patch rebased onto
our pin 5dc6302 (kernel sources unchanged, the four registry files merged by hand), applied before
`build.sh`. Nothing to switch on: the kernels bind as soon as the cache is 16-bit, and the log
prints `all-R4D decode path live` / `all-R4D prefill path live`.

Measured 2026-09-24 (300 W, production args, 0.3.4 → 0.3.5, BetterBench ab.json + long-context probe; a
repeat of the 0.3.4 arm at the end of the window reproduced its prefill within 0.1 % and its step gap
within 0.03 ms):

| | 0.3.4 | 0.3.5 | Δ |
|---|---|---|---|
| prefill 2k / 8k / 16k / 32k / 64k (tok/s) | 3144 / 3127 / 3037 / 2817 / 2431 | 3297 / 3299 / 3221 / 2980 / 2555 | +4.9 / +5.5 / +6.0 / +5.8 / +5.1 % |
| weighted decode (tok/s) | 129.4 | 130.6 | +1.0 % |
| step gap, 37-token / 28k-token prompt (ms) | 35.57 / 37.49 | 34.85 / 36.83 | −0.7 ms |
| conc 4 / 8 (tok/s) | 334.7 / 390.9 (repeat 342.3 / 399.8) | 340.2 / 392.1 | within noise |
| KV pool (tokens) | 384,316 | 384,316 | – |
| GSM8K (200, cot zero-shot) | 0.835 (baseline) | 0.845 | – |

Not bit-identical to the FLA path (different rounding of the bf16 state), so greedy text drifts
after a few sentences; accuracy is unchanged. The extras' other kernels stay opt-in:
`RADIANCE_GDN_FUSED_UPDATE=1` (fused decode step, measured: no further gain) and `R4D_ATTN_FP8`
(8-bit legs of the R4D prefill attention, R4D backend only). The lazy-snapshot kernels are built
but not wired (`patch_gdn_lazy` is not applied, see *Considered and not adopted*).

## NVFP4 checkpoints via load-time MXFP4 requantization (0.3.6)

ggz14's `radiance_nvfp4.py` + `patch_nvfp4_mxfp4.py` (merged with upstream/ggz14 in 0.2.0, left out of
the loop until now) are in the image. With `RADIANCE_NVFP4_MXFP4=1` a compressed-tensors NVFP4
checkpoint (`unsloth/Qwen3.8-27B-NVFP4`) is loaded as stored and every linear is requantized to
MXFP4 (e2m1 + e8m0/32) at load: the NVFP4 MLPs, the FP8 per-channel attention / GDN / last-8-layer
MLPs (`RADIANCE_NVFP4_FP8_LAYERS=mxfp4`) and the bf16 GDN a/b gates (`RADIANCE_NVFP4_BF16_LAYERS=in_proj_ba`).
The result goes through `init_mxfp4_linear_kernel`, i.e. onto **this image's** `RadianceMxfp4W4A8LinearKernel`,
exactly like a Quark layer. There is no NVFP4 kernel: NVFP4 is a storage format here.

What that means for the tuning in this image: the Quark checkpoint excludes only `lm_head` (plus mtp /
visual), so with the defaults above the NVFP4 model ends up with the **same set of MXFP4 linears at the
same (N, K)** as the Quark one. Everything keyed on the kernel, the shape or the layer type applies
unchanged: decode cell table / split-K, A-tiled prefill and WPERM (0.3.0/0.3.1), fused add-rms / silu /
GDN-norm quant (every consumer is a radiance W4A8 layer, so every site fuses), decode / drafter
attention tunes, prompt lookup, int8/int4 embed (bf16 in both checkpoints), KV group size, the load-time
allocator scope. Expected speed is therefore the Quark number, give or take the lm_head below; ggz14's
+5 % decode for NVFP4 on TP=2 was measured against the ParoQuant unit and came from the GDN in_proj
merge, which this image rejected (see `ci/unused_patches.txt`).

Differences that do matter:

- **lm_head.** The unsloth checkpoint stores it FP8 per-channel. `RADIANCE_NVFP4_LMHEAD=bf16` dequantizes
  it at load; with `RADIANCE_LMHEAD_INT4=1` the int4 head is then built from that bf16 weight
  (`sly/patch_lmhead_int4_ct.py`, now behind the scheme lookup; before 0.3.6 the int4 hook ran first and
  would have loaded the e4m3 codes unscaled). `RADIANCE_LMHEAD_FP8` is a Quark-only hook and does nothing here.
- **Accuracy.** Requantizing is a second rounding: 0.158 relRMS against bf16 vs 0.113 for NVFP4 itself
  (ggz14's study; direct bf16 → MXFP4 is 0.112), and the FP8 layers lose their 8-bit precision. The
  Quark checkpoint is AWQ-calibrated direct MXFP4. ggz14 measured GSM8K 500q 97.6 % on TP=2 — the gate
  here is the usual GSM8K run against the Quark baseline.
- **Load and memory.** ~10 s of requantization; the fp8 → bf16 → int4 head has a 2.4 GiB bf16 transient
  at load. The KV pool has to be measured (not assumed to be 384k).
- **KV scales.** The checkpoint carries static fp8 KV scales, which vLLM applies with `--kv-cache-dtype fp8`
  (the Quark checkpoint has none, i.e. 1.0).

Measured 2026-09-24 (300 W, production args, 0.3.6, second start, against Quark MXFP4 on the
same stack): load 461 s cold incl. ~21 s requantization (304 layers, relRMS 0.113–0.116 against
the NVFP4 weights), KV pool **374,202** tokens (Quark 384,316: −2.6 %, the bf16 → int4 head's
transient), GSM8K 0.825 (Quark 0.835–0.845, ±0.027 at 200), weighted decode 133.4 tok/s
against Quark's 132.7 at the same 34.8 ms step gap (128 fresh BetterBench runs each; a first
64-run sample had read 136.6 vs 130.6, which was trajectory noise, see *Other checkpoints*) — prefill 3160 / 2504 tok/s at 2k / 64k (Quark 3297 / 2555), conc 4 / 8
352.6 / 402.2 (Quark 340.2 / 392.1). A fix was needed on the way: `radiance_nvfp4.py` built its
e2m1 grid as a module-level tensor, which vLLM's meta-device model construction turned into a meta
tensor ("Cannot copy out of meta tensor" at the first requant).

Launch: the production command below with `-e RADIANCE_NVFP4_MXFP4=1`, `--model unsloth/Qwen3.8-27B-NVFP4`
and `--quantization compressed-tensors` instead of `quark`; the other flags stay. The boot log prints one
`[radiance.nvfp4]` line per converted layer with its requant error.

**INT4 (compressed-tensors W4A16, `RedHatAI/Qwen3.8-27B-INT4`)** runs on vLLM's `rdna_hybrid_w4a16` kernels,
not on the W4A8 kernel, so the MXFP4 GEMM work (cells, A-tiled, WPERM, fused norm quant) does not apply.
What applies already: the target tile table (`sly/patch_w4a16_tiles.py --target`), int4 lm_head (CT hook),
int8/int4 embed, the attention / drafter / lookup tunes, KV groups and the load-time allocator scope. A
load-time INT4 → MXFP4 requant (the NVFP4 approach) would put it on the tuned kernel, but g128 uniform
int4 → e2m1/e8m0 per 32 is a coarser double rounding than NVFP4's, and the same base model exists as
direct Quark MXFP4 — not planned.

## ParoQuant, AutoRound and escha checkpoints (0.3.6, opt-in)

ggz14's three other weight formats are in the image, each as a single-file HIP extension built in
the assemble stage plus a quantization config that registers only when asked. Dispatch is by the
checkpoint's `quant_method`; without the switch nothing is imported.

| Switch | Registers | Checkpoints | Tested here |
|---|---|---|---|
| `RADIANCE_PAROQUANT=1` | `paroquant`, `paroquant_mxfp4` (`radiance_quant_plugins.pth`) | [z-lab/Qwen3.8-27B-PARO](https://huggingface.co/z-lab/Qwen3.8-27B-PARO) (int4 g128 asym + learned Givens rotations, W4A8 on gfx1201; see `PAROQUANT.md`), PARO-MXFP4 variants | yes, see below |
| `RADIANCE_AUTOROUND=1` | `auto-round` (`patch_autoround.py`; claims int4 g128 sym before vLLM's INC config, which refuses ROCm) | Intel AutoRound int4 exports | registration only (no checkpoint on this box) |
| `RADIANCE_ESCHA=1` | `escha` (`patch_escha.py`, ExLlamaV3-derived trellis format, `escha/FORMAT.md`) | ggz14's escha exports | registration only |

ParoQuant on one R9700 (2026-09-24, 300 W, production args with `--model z-lab/Qwen3.8-27B-PARO`,
no `--quantization`, `RADIANCE_LMHEAD_FP8=0 RADIANCE_LMHEAD_INT4=0`, same DFlash2 W4A16 drafter):
loads and serves, GSM8K 0.84 (Quark MXFP4: 0.835–0.845), but it is **not a speed option on this
stack**: weighted decode 67.7 tok/s against 130.6 (step gap 70.6 ms against 34.8), prefill 2313 /
2004 tok/s at 2k / 64k against 3297 / 2555, conc 8 238.8 against 392.1, KV pool 1.20× instead of
1.46× of 262k (the checkpoint's lm_head stays bf16). ggz14's numbers (combined decode 226 t/s
against MXFP4's 186) are from 2 × R9700 with their own launcher profile (R4D attention, verify
head, dynamic width); none of that tuning is carried over to the single-card path here.

Launch: the production command with `-e RADIANCE_PAROQUANT=1` (or `RADIANCE_AUTOROUND=1` /
`RADIANCE_ESCHA=1`), `--model <checkpoint>` and **no** `--quantization` flag (the config is picked
from the checkpoint).

## Build details

- ROCm base: `ARG ROCM_BASE` defaults to `rocm/dev-ubuntu-24.04:10.0.0-full@sha256:…` (the
  production image is built on exactly that digest, Renovate tracks it); overridable with
  `--build-arg ROCM_BASE=…`.
- PyTorch 2.14.0, torchvision 0.29.0, Triton 3.8.0, aiter 0.1.22.post1, transformers 5.17.0, vLLM 0.30.0; libr4d pinned
  by commit.
- `MAX_JOBS` capped (PyTorch compile OOM-killed the host at 16 jobs), retry loop around PyTorch's
  submodule clone, torch wheel build fixed for 2.14's deprecated `setup.py bdist_wheel`.
- The HIP kernel is a single-file pybind11 extension compiled with the image's own `hipcc` in the
  assemble stage; the patch scripts (`/opt/patches`) never reach the final image.
