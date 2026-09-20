# vllm-sly-radiance

A vLLM inference image for the **AMD Radeon AI PRO R9700 (gfx1201 / RDNA4)** that runs
**MXFP4 (Quark) checkpoints** — built, measured and operated on a single R9700 with
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` plus DFlash2 speculative decoding.

> **Status: experimental, single-GPU, single-model.** Everything below was measured on exactly one
> setup: one R9700 (32 GB, 32 CUs), `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` (gated-delta-net hybrid) with
> the `syvai/Qwen3.8-27B-DFlash2-W4A16` drafter, fp8 KV cache, ROCm 10.0, vLLM 0.29.0. Other MXFP4
> models, tensor parallel and non-R9700 hardware are untested. Expect breaking changes.

## Lineage

This repository is a fork that stacks three layers:

| Layer | Source | What it contributes |
|---|---|---|
| **vLLM 0.29.0** | [vllm-project/vllm](https://github.com/vllm-project/vllm) (tag `v0.29.0`) | The inference engine. Built from source for `PYTORCH_ROCM_ARCH=gfx1201`. |
| **vllm-radiance** | [StillDeadcode/vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance) (Codeberg) | The RDNA4 image: ROCm + PyTorch + Triton + AITER + vLLM build pipeline, the gfx1201 correctness patches (`patch_gfx1201.py`, `patch_unified_attention_lds.py`, …), the [libr4d](https://codeberg.org/StillDeadcode/libr4d) hand-written kernel library (paged attention, fused GDN prefill scan, skinny bf16 GEMM, P2P all-reduce), DFlash/MTP draft support, the entrypoint and bandwidth sweep. Upstream targets **FP8** checkpoints on two R9700 (TP=2). |
| **radiance-vllm-mxfp4** | [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4) (Codeberg) | MXFP4 on RDNA4: the Quark-loader gates that unlock AITER's native Triton MXFP4 GEMM (`gemm_afp4wfp4`) on gfx1201, and the hand-written fp8-WMMA W4A8 HIP kernel (`radiance_mxfp4_fp8.hip`) with its `RadianceMxfp4W4A8LinearKernel` plugin. Written against vLLM 0.27.1. |

**`main`** is the integration branch (upstream vllm-radiance + everything SlyBase adds); the two
upstreams are mirrored read-only as `upstream/stilldeadcode` and `upstream/ggz14` (see
[Branches, CI and upstream sync](#branches-ci-and-upstream-sync)). Everything SlyBase
adds lives in [`sly/`](sly/) (patches, kernels, configs, benches) and is wired into the `Dockerfile`
via the same `_patchlib.apply()` mechanism upstream uses — anchor-based, idempotent, verified with
`ast.parse()` at build time. `sly/README.md` (German) is the per-patch reference.

## What was changed on top

Chronological summary of the tunings and kernel work on `main` (versions = `VERSION` file /
image tag `vllm-sly-radiance:<version>-rocm10.0`).

### MXFP4 on gfx1201 (0.1.0 – 0.1.2)

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

### DFlash2 speculative decoding with a W4A16 drafter (0.1.2, 0.1.4)

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

### fp8 lm_head (0.1.3)

`sly/mxfp4/radiance_lmhead_fp8.py` + `sly/patch_lmhead_fp8.py`. The Quark checkpoint lists
`lm_head` in its `exclude` list, so the 248320 × 5120 vocabulary projection ran as a bf16 GEMM —
2.54 GB of weight traffic per call, and DFlash calls the shared head twice per step (8.6 of ~49 ms).
`RADIANCE_LMHEAD_FP8=1` quantises the weight per output channel to fp8 after loading and runs it
row-wise through `torch._scaled_mm` (hipBLASLt) with per-token fp8 activations. Step 48 → 44.5 ms,
1.27 GiB returned to the KV cache. GSM8K (cot, zero-shot, greedy, 200 tasks): 0.830 ± 0.027 vs
0.835 ± 0.026 for bf16 — no measurable loss.

### int4 lm_head (0.1.5)

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

### Fused norm / activation + fp8 quant (0.1.6)

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

### upstream/ggz14 merged (0.2.0)

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

### Gated-delta-net / attention

- **`sly/patch_short_prefill.py`** — a 1-token prefill was misclassified as decode in the GDN
  metadata builder (wrong state for the first token of a short prompt).
- **`patch_unified_attention_lds.py` ported to aiter 0.1.21.post2** — AITER replaced its
  `select_3d_config`/`select_2d_config` elif chains with JSON-driven config tables between the
  version upstream patched and 0.1.21.post2; the patch was rewritten (6 hunks): LDS clamp for gfx1201,
  `TILE_SIZE` return for `reduce_segments`, gfx1201-gated bf16 3D decode tuning.
- **bf16 SSM state** (runtime flag, `--mamba-ssm-cache-dtype bfloat16`): halves the GDN state pages
  (attention block 1664 → 896 tokens), KV 101k → 133k tokens on the 32 GB card, concurrent-request
  ceiling ~6 → ~12. GSM8K unchanged (0.835), needle-in-haystack 3/3 at 12k and 24k tokens.

### Decode attention: the verify batch on aiter's config tables (0.2.7)

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

### Drafter attention: split-KV over the window (0.2.8)

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

### Prompt lookup on top of the DFlash draft (0.2.9)

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

### Build

- ROCm base: `ARG ROCM_BASE` defaults to `rocm/dev-ubuntu-24.04:10.0.0-full@sha256:…` (the
  production image is built on exactly that digest, Renovate tracks it); overridable with
  `--build-arg ROCM_BASE=…`.
- PyTorch 2.14.0, Triton 3.8.0, aiter 0.1.21.post2, transformers 5.17.0, vLLM 0.29.0; libr4d pinned
  by commit.
- `MAX_JOBS` capped (PyTorch compile OOM-killed the host at 16 jobs), retry loop around PyTorch's
  submodule clone, torch wheel build fixed for 2.14's deprecated `setup.py bdist_wheel`.
- The HIP kernel is a single-file pybind11 extension compiled with the image's own `hipcc` in the
  assemble stage; the patch scripts (`/opt/patches`) never reach the final image.

## Results

BetterBench 0.4.0 `--decode --concurrency` against the production container (2026-09-15, image
0.1.4, all knobs from the *Options* section below). Reference: the same model family as INT4
(`w4a16`) on stock vLLM on the same card.

| Concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| **MXFP4 + DFlash2 k=7 (this image)** tok/s | **99** | **177** | **276** | **348** | **361** |
| INT4 reference tok/s | 77 | 114 | 175 | 214 | 205 |
| Δ | +29 % | +56 % | +58 % | +62 % | +76 % |
| … with `DECODE_MAX_M=128` (conc-only run) | 99 | 178 | 280 | **368** | **371** |
| … + int4 lm_head (0.1.5, conc-only A/B run; fp8 head in the same window: 99 / 180 / 286 / 356 / 364) | **105** | **187** | **293** | 354 | **371** |
| … + fused norm/act + fp8 quant (0.1.6, conc-only A/B run; 0.1.5 in the same window: 105.0 / 186.9 / 292.3 / 356.5 / 364.0) | **108** | **190** | **298** | **376** | 363 |

Single-stream median 120 tok/s, DFlash step gap 42.0 ms, KV cache 8.57 GiB / 133k fp8 tokens at
`--max-model-len 32768 --gpu-memory-utilization 0.95`. Profile of one step (conc 1): decode GEMMs
26.7 ms at ~540 GB/s (84 % of the R9700's read peak — the remaining gap is launch overhead and small
elementwise kernels, not the GEMMs).

Progression: 0.1.0 → 0.1.2 (DEC_MAX_N) step 67 → 48 ms; 0.1.3 (fp8 lm_head) 44.5 ms; bf16 SSM state
+ 0.1.4 (drafter tiles) 42.0 ms; 0.1.5 (int4 lm_head) 39.7 ms at conc 1 (41.9 ms at conc 2, conc 8
unchanged), KV 141k tokens; 0.1.6 (fused norm/act + fp8 quant) +3.0 / +1.9 / +1.8 / +5.6 / −0.2 %
at conc 1/2/4/8/16 (≈1 ms/step at conc 1), per-stream decode 128.7 → 133.5 tok/s at conc 1, TTFT p50
105 → 102 ms, KV 146k tokens (+3.8 %, fewer Inductor intermediates), GSM8K 0.840 → 0.845 ± 0.026,
mean tokens/step 4.312 → 4.314 (acceptance unchanged). Numerics (`sly/check_fused_norm.py`): the
fused chain equals the unfused pq chain exactly; against eager/Inductor only fp8 rounding flips.

## Options

### Environment knobs (read at import time)

Production values first; everything else is tuning/diagnostic and off by default.

| Variable | Default | Production | Meaning |
|---|---|---|---|
| `RADIANCE_MXFP4` | `0` | `1` | Unlock AITER's Triton `gemm_afp4wfp4` (native MXFP4 W4A4) on gfx1201. **Required** for any MXFP4 model; without it vLLM emulates in bf16. |
| `RADIANCE_MXFP4_W4A8` | `0` | `1` | Register `RadianceMxfp4W4A8LinearKernel` (the HIP fp8-WMMA kernel) ahead of the AITER path. |
| `RADIANCE_MXFP4_W4A8_MIN_M` | `256` | `0` | Smallest M the HIP kernel accepts for the folded (prefill) path. `0` = also small M. |
| `RADIANCE_MXFP4_DECODE_MAX_M` | `0` | `128` | Largest M routed to the HIP split-K decode kernel; above it the folded kernel takes over. Must cover `max_num_seqs × (num_speculative_tokens + 1)` = 64 for pure decode; 128 also covers the CUDA-graph capture sizes of the mixed prefill+decode steps. |
| `RADIANCE_MXFP4_SANITIZE` | `0` | `0` | `nan_to_num` on activations before the quant kernel. Was needed before the libr4d GDN path was fixed; costs 0.9 ms/step. |
| `RADIANCE_LMHEAD_FP8` | `0` | `1` | fp8 per-output-channel lm_head via `torch._scaled_mm` (see above); fallback when `RADIANCE_LMHEAD_INT4` is unset. Not compatible with `--hf-overrides '{"head_dtype": "float32"}'`. |
| `RADIANCE_LMHEAD_FP8_MIN_M` | `16` | – | Pads M below this for hipBLASLt. |
| `RADIANCE_W4A16_TILES` | `1` | – | `0` disables the gfx1201 drafter tile table (stock heuristic). |
| `RADIANCE_LMHEAD_INT4` | `0` | `1` | int4 (W4A16, group-128 bf16 scales) lm_head on the drafter's Triton/HIP kernel path (see above); takes precedence over `RADIANCE_LMHEAD_FP8`. Same `head_dtype` limitation. |
| `RADIANCE_LMHEAD_INT4_GS` | `128` | – | Group size of the int4 lm_head (64 measured: −8 % error for 2× scale bytes, not worth it). |
| `RADIANCE_LMHEAD_INT4_CLIP` | `mse` | – | Per-group scale search over clip ratios 1.0…0.8 by least squared error; `rtn` = plain amax/7 (−14 % vs +0 % error, 1.6 s vs 0.3 s at load). |
| `RADIANCE_FUSED_NORM_QUANT` | `0` | `1` | Fused add+rms_norm / silu·mul / GDN gated norm + per-token fp8 quant in front of the W4A8 GEMMs (see above). Needs `RADIANCE_MXFP4_W4A8=1`, `RADIANCE_MXFP4_W4A8_MIN_M=0`, `RADIANCE_MXFP4_SANITIZE=0`; part of the torch.compile cache key. |
| `RADIANCE_FUSED_NORM_QUANT_ADD_RMS` / `_SILU` / `_GDN` | `1` | – | Per-fusion switches (only read when `RADIANCE_FUSED_NORM_QUANT=1`). |
| `GPU_MAX_HW_QUEUES` | ROCm default | `2` | ROCm HW queue count; 2 measured best for this single-process setup. |
| `RADIANCE_MXFP4_DECODE_TUNE16` | `1` | – | 0.2.1 cell table for decode M in (8, 64] (2–8 concurrent sequences): BK=128 everywhere in the band and split-K 2 for o_proj/out_proj at M=16/24, measured with `sly/mxfp4/bench_decode_cells.py`. `0` = pre-0.2.1 fill rule + `decode_bk64` (A/B control). M ≤ 8 is never touched. |
| `RADIANCE_MXFP4_DECODE_KS` | auto | – | Force the decode kernel's split-K factor (sweep/A-B knob, overrides the tables). |
| `RADIANCE_MXFP4_DECODE_BK` | auto | – | `128` pins BK=128, `64` forces the BK=64 instantiation where one exists (split 1 and 4) — sweep knobs, never a production setting. |
| `RADIANCE_MXFP4_DECODE_NT` | `0` | – | Non-temporal weight loads in the decode kernel. |
| `RADIANCE_MXFP4_TN4_MIN_M` | `2048` | – | M from which the folded kernel uses the wide TN=4 tile. |
| `RADIANCE_MXFP4_A_TILED_MIN_M` | `0` | – | Tiled-A layout for very large M (must exceed 512 and `DECODE_MAX_M`). |
| `RADIANCE_MXFP4_WPERM` / `RADIANCE_MXFP4_R4D_DECODE_MAX_M` | `0` / `0` | – | Experimental: fragment-order weights + libr4d's `gemm_mxfp4a8_nt_m64` decode kernel. |
| `RADIANCE_MXFP4_MHIST` | `0` | – | Print every distinct `(N, K, M)` the plugin sees once (which M the decode path really issues). |
| `RADIANCE_MXFP4_DEBUG`, `_CHECKX`, `_CHECKALL`, `_REFLINEAR`, `_SHADOW`, `_SYNC`, `_KERNEL_N`, `_KERNEL_NK` | off | – | Correctness/diagnostic switches, see the header of `sly/mxfp4/radiance_mxfp4.py`. |
| `RADIANCE_ATTN_DECODE_TUNE` | `1` | – | 0.2.7: the fp8-q + fp8-KV, head-256 verify-batch tune of aiter's unified attention (`sly/radiance_attn_decode.py`, see *Decode attention* below). `0` = aiter's stock tables (A/B control; baked in at CUDA-graph capture, so it needs a restart). |
| `RADIANCE_ATTN_DECODE_WIDE` | `auto` | – | `auto` = 64-row `BLOCK_M` when the verify batch fills ≥ `MIN_FILL` of the 64 rows, 32 below; `0` = never widen; `1` = widen every batch that fits. |
| `RADIANCE_ATTN_DECODE_MIN_FILL` | `0.5` | – | Fill fraction for `WIDE=auto`: width ≥ 6 at GQA 6 (measured crossover; the retired hook's 0.8 would not have widened DFlash k=7's 48 of 64 rows). |
| `RADIANCE_ATTN_DECODE_3D` | `1` | – | Verify batches take the 3D split-KV kernel above 512 tokens when the plan's own launch is small enough (stock sends 7–8 sequences down the 2D kernel). `0` = aiter's 2D/3D choice. |
| `RADIANCE_ATTN_DRAFTER_TUNE` | `1` | – | 0.2.8: split-KV launch for the DFlash2 drafter's sliding-window attention (fp8 KV, head 128, GQA 4, 2–16 queries per sequence; `sly/radiance_attn_drafter.py`, see *Drafter attention* above). `0` = vLLM's launch (A/B control; baked in at CUDA-graph capture, so it needs a restart). |
| `RADIANCE_ATTN_DRAFTER_SEGMENTS` | `auto` | – | `auto` = 64 splits with one sequence, 32 with two or three, 16 beyond; `N` (a power of two) forces the split count. |
| `RADIANCE_LOOKUP_DRAFT` | `1` | – | 0.2.9: prompt-lookup override of the DFlash2 draft (`sly/radiance_lookup_draft.py`, see *Prompt lookup* above): a suffix n-gram match over the whole context replaces the DFlash draft when it is very likely to win. `0` = the DFlash draft as the graph writes it (A/B control; the hook is not installed, so it needs a restart). |
| `RADIANCE_LOOKUP_ENTER` / `_STAY` / `_HOT` | `8` / `3` / `3` | – | Enter lookup mode on a match of ≥ ENTER tokens; stay in it while the last lookup step accepted ≥ HOT draft tokens and a match of ≥ STAY exists. "Any match ≥ 3" (ENTER 3) loses 1–17 % on free generation. |
| `RADIANCE_LOOKUP_MIN_DIST` | `auto` | – | Only sources whose continuation starts more than this many tokens back count; `auto` = the drafter's sliding window from its config (2048). `0` = any (costs 5–8 % on edits inside the drafter's window). |
| `RADIANCE_LOOKUP_SWITCH` / `_STATS` | unset / `0` | – | Debug: the override runs only while the file `SWITCH` exists (A/B inside one process without a restart); `STATS=N` logs the lookup share and accepted tokens per lookup step every N draft steps. |

Inherited from upstream vllm-radiance (see its `DOCKERHUB.md`): `RADIANCE_GFX_ARCH`,
`RADIANCE_NUMA_BIND`, `RADIANCE_RUN_BWTEST`, `RADIANCE_BANNER_PLAIN`, `RADIANCE_RMS_QUANT_FUSION`
and the draft-controller knobs.

### Launch arguments (production)

The image sets `vllm serve` as `ENTRYPOINT`; pass vLLM arguments directly.

```bash
docker run --rm --name vllm7-mxfp4 \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --ipc=host \
  -e HIP_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 \
  -e GPU_MAX_HW_QUEUES=2 \
  -e RADIANCE_MXFP4=1 \
  -e RADIANCE_MXFP4_W4A8=1 \
  -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
  -e RADIANCE_MXFP4_DECODE_MAX_M=128 \
  -e RADIANCE_MXFP4_SANITIZE=0 \
  -e RADIANCE_LMHEAD_FP8=1 \
  -e RADIANCE_LMHEAD_INT4=1 \
  -e RADIANCE_FUSED_NORM_QUANT=1 \
  -p 8000:8000 \
  -v /root/hf-cache:/root/.cache/huggingface \
  -v /root/vllm7-cache/triton:/root/.triton \
  -v /root/vllm7-cache/torch_compile:/root/.cache/vllm/torch_compile_cache \
  -v /root/vllm7-cache/aiter:/root/.aiter \
  vllm-sly-radiance:$(cat VERSION)-rocm10.0 \
  --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 \
  --quantization quark \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config.method dflash \
  --speculative-config.model syvai/Qwen3.8-27B-DFlash2-W4A16 \
  --speculative-config.num_speculative_tokens 7 \
  --speculative-config.draft_sample_method probabilistic \
  --speculative-config.attention_backend TRITON_ATTN \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN \
  --compilation-config.cudagraph_mode FULL_AND_PIECEWISE \
  --enable-prefix-caching \
  --skip-mm-profiling --enable-mm-embeds \
  --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
  --port 8000
```

Why these values:

| Argument | Note |
|---|---|
| `--quantization quark` | The checkpoint's quant method (fp4 + e8m0 scales); set explicitly rather than trusting auto-detect. |
| `--kv-cache-dtype fp8` | 133k tokens of KV on 32 GB; `auto` (bf16) halves that. |
| `--mamba-ssm-cache-dtype bfloat16` | Halves the GDN state pages (see above). The model config defaults to float32. Changing this invalidates the compile cache. |
| `--speculative-config.*` | DFlash2 with the W4A16 drafter. `num_speculative_tokens 7` is the drafter's maximum (block size 8). `TRITON_ATTN` for the drafter; the target uses `ROCM_AITER_UNIFIED_ATTN`. |
| `--max-num-seqs 8` | `8 × (7 + 1) = 64` decode tokens per step, well inside `RADIANCE_MXFP4_DECODE_MAX_M=128`. In practice the KV cache caps 20k-context requests at ~12. |
| `--max-num-batched-tokens 2048` | Halves peak activation memory during profiling (KV 3.2 → 7.45 GiB before the fp8/bf16 wins); prefill chunking costs nothing measurable here. |
| `--compilation-config.cudagraph_mode FULL_AND_PIECEWISE` | Full CUDA graphs for decode, piecewise for prefill. |
| `--skip-mm-profiling --enable-mm-embeds --limit-mm-per-prompt.* 0` | Text-only serving of a VL-capable architecture; skips the vision profiling allocation. |
| `HIP_VISIBLE_DEVICES=0` | Hosts with an iGPU (gfx1103) next to the R9700 would otherwise initialise both. |

**Compile cache:** the first start after an image or config change compiles fresh and the memory
profiler sees ~2 GiB more peak → ~2 GiB less KV cache. Restart once with a warm cache. Startup is
4–6 minutes with a warm cache.

## Build

Everything the build needs is in this directory (flat Docker context). Multi-stage: **builder**
(PyTorch, Triton, torchvision, AITER, vLLM from source for gfx1201) → **rocmprune** → **assemble**
(wheels, upstream RDNA4 patches, then the `sly/` patches, libr4d, the HIP kernel) → **final**
(clean `ubuntu:24.04` + pruned ROCm + venv + entrypoint).

```bash
git clone https://github.com/SlyBase/vllm-sly-radiance.git
cd vllm-sly-radiance

docker build -t vllm-sly-radiance:$(cat VERSION)-rocm10.0 .
```

A cold build compiles PyTorch and takes hours (`MAX_JOBS=4` by default — raise it on a box with
RAM to spare). With the builder stage cached, a change to the `sly/` layer rebuilds in ~10 minutes.
The build needs no GPU, so it can run next to a serving container.

Smoke test:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri -e HIP_VISIBLE_DEVICES=0 \
  --entrypoint python3 vllm-sly-radiance:$(cat VERSION)-rocm10.0 \
  -c "from vllm.model_executor.layers.quantization.quark import QuarkConfig; print('OK')"
```

## Repository layout

```
Dockerfile                    build pipeline (upstream + sly/ patch loop + HIP kernel compile)
VERSION                       image version (0.1.4)
_patchlib.py                  anchor-based, idempotent patch helper (upstream)
patch_*.py, radiance_*.py     upstream vllm-radiance RDNA4 patches and runtime modules
sly/
  README.md                   per-patch reference (German)
  patch_quark_mxfp4.py        Quark/MXFP4 loader gates for vLLM 0.29.0 + kernel plugin registration
  patch_short_prefill.py      GDN 1-token-prefill fix
  patch_dflash_w4_packed.py   W4A16 (compressed-tensors) DFlash drafter
  patch_gdn_nonspec_mask.py   non_spec_sequence_masks_cpu on the numpy path
  patch_lmhead_fp8.py         hook radiance_lmhead_fp8 into QuarkConfig
  patch_w4a16_tiles.py        gfx1201 tile table for the drafter GEMMs
  bench_w4a16_tiles.py        tile sweep that produced the table
  mxfp4/radiance_mxfp4.py     RadianceMxfp4W4A8LinearKernel plugin (dispatch, scratch, knobs)
  mxfp4/radiance_mxfp4_fp8.hip  fp8-WMMA W4A8 GEMM: folded prefill + split-K decode kernels
  mxfp4/radiance_lmhead_fp8.py  fp8 lm_head
  mxfp4-configs/              AITER gemm_afp4wfp4 config for gfx1201
```

## Branches, CI and upstream sync

```
main                    integration branch = upstream vllm-radiance + sly/ (protected: PR + green `ci`)
upstream/stilldeadcode  read-only mirror of StillDeadcode/vllm-radiance `main` (Codeberg)
upstream/ggz14          read-only mirror of ggz14/radiance-vllm-mxfp4 `main` (Codeberg)
archive/*               tags freezing the pre-2026-09 layout (sly/main, sly/b-*, sly/e2-*) — kept forever
v0.1.1 … v0.1.6         annotated tags = the VERSION history (v0.1.0 has no unambiguous commit)
```

Remotes on a dev machine: `origin` (GitHub) plus the two Codeberg upstreams:

```bash
git remote add stilldeadcode https://codeberg.org/StillDeadcode/vllm-radiance.git
git remote add ggz14 https://codeberg.org/ggz14/radiance-vllm-mxfp4.git
```

Workflows (`.github/workflows/`):

| Workflow | Trigger | What it does |
|---|---|---|
| `ci` | PR, push to `main`, manual | `lint` (ruff E9/F63/F7/F82, shellcheck, hadolint, actionlint, `docker buildx build --check`), `patch-dryrun` (`ci/patch_dryrun.sh`: the pinned upstream sources from the Dockerfile ARGs in a venv, then the Dockerfile patch loop twice — pass 1 must apply every hunk, pass 2 must be all NOOP; `ci/patch_dryrun_skip.txt` is the documented skip allowlist), `consistency` (`ci/check_consistency.py`: every patch file is in the loop or in `ci/unused_patches.txt`, every `sly/patch_*.py` is documented in `sly/README.md`, image changes bump `VERSION`). The aggregate status **`ci`** is the required check on `main`. |
| `build` | push to `main` touching `VERSION`, tags `v*`, manual | Self-hosted runner (`rocm-build`, LXC 2408, CPU only — no `--device`, no GPU test, no deploy): `docker build` → import smoke test → push to `ghcr.io/slybase/vllm-sly-radiance:<VERSION>-rocm10.0` only for `v*` tags or the `push_ghcr` input. Build log is an artifact. |
| `upstream-sync` | daily 04:00 UTC, manual | Fast-forwards `upstream/*` from Codeberg (never force) and opens/updates a PR `upstream/<name>` → `main` (label `upstream-sync`) listing the new commits, the test-merge conflict status and the image-relevant files. Never merges. Needs the `SYNC_TOKEN` secret (fine-grained PAT, contents + pull-requests write) to create PRs. |

Renovate (`renovate.json`) tracks every Dockerfile ARG pin via the `# renovate:` markers above the
ARGs (ROCm base image tag + digest, ubuntu digest, torch/triton/torchvision as one group, vLLM,
AITER, transformers, rocm_bandwidth_test, libr4d commit) and the CI tool versions. No automerge —
every bump goes through `ci`, and vLLM/AITER/ROCm bumps additionally need a build and an A/B run.

Changing code:

```bash
git switch -c feat/my-change main
# edit; bump VERSION for anything that changes the image (consistency enforces it on PRs)
python3 ci/check_consistency.py --base origin/main && ci/patch_dryrun.sh
git push -u origin feat/my-change && gh pr create
# merge when `ci` is green — the VERSION bump on main then triggers `build`
```

Merging upstream: take the `upstream-sync` PR (or `git merge origin/upstream/<name>` on a branch),
resolve the usual conflicts (Dockerfile pins + patch loop, `README.md`, `VERSION`) and re-verify
every `sly/` anchor — `ci/patch_dryrun.sh` fails hard when an anchor is gone.

## Credits

- [StillDeadcode](https://codeberg.org/StillDeadcode) — vllm-radiance and libr4d, the RDNA4
  foundation this image is built on.
- [ggz14](https://codeberg.org/ggz14) — radiance-vllm-mxfp4: the MXFP4 loader work and the W4A8
  HIP kernel.
- [vLLM](https://github.com/vllm-project/vllm), [AITER](https://github.com/ROCm/aiter),
  [DFlash](https://github.com/vllm-project/vllm/pull/52816).

License: same as upstream vllm-radiance (see `LICENSE`).
