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

### Prefill GEMM on fragment-tiled activations, fragment-order weights (0.3.0, 0.3.1)

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

### Fused GDN decode on ROCm (HIP port of vLLM's kernel, default off)

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

### libr4d extras: GDN kernels for the bf16 state cache (0.3.5)

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

### NVFP4 checkpoints via load-time MXFP4 requantization (0.3.6)

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

### ParoQuant, AutoRound and escha checkpoints (0.3.6, opt-in)

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

### Current image: 0.3.1 (2026-09-23)

BetterBench 0.4.0, default config, all three phases (single-stream decode 3 warmup + 20 passes per
category, prefill sweep, concurrency 1/2/4/8/16 × 48 requests; temperature 0.7 / top_p 0.95 / top_k 20),
against the production container: image 0.3.1 with `RADIANCE_MXFP4_A_TILED_MIN_M=513` and
`RADIANCE_MXFP4_WPERM=1`, otherwise the same arguments as the 0.2.9 run below (KV pool 384,316 tokens,
`--max-model-len 262144`). Two runs back to back, no other traffic: **300 W with the firmware fan curve**
(no acoustic limit) and the production setting **210 W / fan curve capped at 2,800 rpm**.

**Single-stream decode** (tok/s ± 95 % CI):

| Category | 300 W | 210 W | 0.2.9, 210 W |
|---|---|---|---|
| chat | 90.8 ± 5.7 | 85.5 ± 4.2 | 82.1 |
| code | 145.7 ± 9.1 | 136.1 ± 9.2 | 130.5 |
| file_edit | 175.2 ± 6.8 | 164.6 ± 6.8 | 157.0 |
| json | 166.7 ± 11.2 | 153.8 ± 10.0 | 153.5 |
| math | 170.4 ± 7.3 | 157.2 ± 6.8 | 158.0 |
| prose | 82.9 ± 3.3 | 77.7 ± 2.9 | 77.7 |
| reasoning | 105.1 ± 13.6 | 105.6 ± 13.5 | 102.7 |
| summarization | 128.7 ± 9.1 | 121.5 ± 6.4 | 122.8 |
| **weighted** | **132.6** | **125.3** | 122.4 |
| step gap p50 | 35.40 ms | 37.52 ms | 38.23 ms |
| tokens/update | 4.61 | 4.63 | 4.60 |

**Prefill** (unique prompts, no prefix cache, 16 output tokens):

| Prompt tokens | 1,514 | 5,918 | 11,794 | 23,543 | 47,056 |
|---|---|---|---|---|---|
| **300 W**, tok/s | **3,040** | **3,029** | **2,974** | **2,786** | **2,424** |
| **210 W**, tok/s | 2,431 | 2,409 | 2,360 | 2,225 | 1,961 |
| 0.2.9, 210 W | 2,073 | 2,053 | 2,020 | 1,913 | 1,713 |
| Δ 210 W vs 0.2.9 | +17.3 % | +17.3 % | +16.8 % | +16.3 % | +14.5 % |
| TTFT 300 W / 210 W | 0.50 / 0.62 s | 1.95 / 2.46 s | 3.97 / 5.00 s | 8.45 / 10.58 s | 19.4 / 24.0 s |

**Concurrency** (48 requests per level, all ok, 0 preemptions):

| Concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| **300 W**, aggregate tok/s | **116.8** | **213.8** | **329.4** | **400.9** | **398.5** |
| **210 W**, aggregate tok/s | 110.0 | 196.5 | 311.2 | 359.8 | 369.8 |
| 0.2.9, 210 W | 105.6 | 199.6 | 294.7 | 393.7 | 382.3 |
| TTFT p50 300 W / 210 W, ms | 108 / 111 | 135 / 149 | 157 / 173 | 184 / 228 | 183 / 204 |

**Card** (5 s samples over the whole run):

| | 300 W | 210 W |
|---|---|---|
| board power avg / max | 298 / 417 W | 209 / 295 W |
| junction avg / max | 99 / 102 °C | 94 / 98 °C |
| memory avg / max | 87 / 90 °C | 90 / 94 °C |
| fan avg / max | 3,756 / 4,058 rpm | 2,334 / 2,445 rpm |
| sclk avg | 3,005 MHz | 2,344 MHz |

Draft acceptance (server counters over each run): 4.23 / 4.25 tokens per step, acceptance rate 0.462 / 0.464
— unchanged against 0.2.9 (4.20 / 0.457), the gains are step time. What 300 W buys over 210 W: +5.8 %
weighted decode (step 37.5 → 35.4 ms), +24 … +26 % prefill, +6 … +11 % concurrency, for 42 % more power and
about 1,400 rpm more fan. Against 0.2.9 at the same 210 W the new image gains +2.4 % decode and +14.5 …
+17.3 % prefill (the tiled GEMM plus WPERM); conc 8/16 came out 8.6 / 3.3 % below the 0.2.9 run, which is
larger than the ±4 % conc-8 spread seen between identical arms in the A/B above but was not repeated, so
it is recorded here and not explained. Raw JSON: `/opt/accept/betterbench/out/full/full{300,210}.json` on
the CI LXC, fan/power samples in `/root/gpu_full{300,210}.csv` on the Proxmox host.

### Other checkpoints on one R9700 (0.3.6, 2026-09-24)

The same image and the same production arguments (DFlash2 W4A16 drafter, k = 7, fp8 KV, bf16 SSM
cache, `--max-model-len 262144`), only `--model` / `--quantization` and the format switch changed;
300 W, fan on auto, BetterBench `ab.json` (weighted serial decode, concurrency 1/4/8, prefill at
2k/64k context), GSM8K 200 questions (cot zero-shot, greedy, ±0.027).

| Checkpoint | Switch | Decode tok/s | Step gap | Conc 1 / 4 / 8 tok/s | Prefill 2k / 64k tok/s | KV (262k requests) | GSM8K |
|---|---|---|---|---|---|---|---|
| [amd/Qwen3.8-27B-Quark-AWQ-MXFP4](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) (production) | – | 130.4 | 34.8 ms | 118 / 344 / 412 | 3300–3385 / 2510–2555 | 1.46× (384,316) | 0.845 |
| [unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) | `RADIANCE_NVFP4_MXFP4=1`, `--quantization compressed-tensors` | 133.4 ¹ | 34.8 ms | 126 / 353 / 402 | ~3160 / 2504 | 1.43× (374,202) | 0.825 |
| [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4) (W4A16) | `--quantization compressed-tensors` | 107.8 | 42.9 ms | 100 / 282 / 290 | 1628 / 1439 | 1.47× | 0.820 |
| [z-lab/Qwen3.8-27B-PARO](https://huggingface.co/z-lab/Qwen3.8-27B-PARO) | `RADIANCE_PAROQUANT=1`, no `--quantization` | 67.7 | 70.6 ms | 60 / 194 / 239 | 2313 / 2004 | 1.20× | 0.840 |
| [amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16) | – | does not load | | | | | |

- **MXFP4 (Quark)** is the best all-round choice: fastest prefill, the full KV pool at maximum
  context, best accuracy. Every kernel in this image was tuned on it.
- **NVFP4** is requantized to MXFP4 at load and runs on the same kernels (same step gap), so decode
  is the same as Quark's: 133.4 vs 132.7 tok/s (+0.5 %) over 128 fresh runs each, tokens per update
  4.64 vs 4.55 with overlapping 95 % intervals. ¹ The first 64-run sample had read 136.6 vs 130.4
  (+4.7 %); that was trajectory noise, not the drafter (see below). NVFP4 costs ~10k KV tokens (its
  fp8 → bf16 → int4 lm_head transient) and its GSM8K is within noise.
- **INT4 W4A16** runs on vLLM's `rdna_hybrid_w4a16` kernels, not on the W4A8 MXFP4 GEMM: half the
  prefill, −17 % decode. Its HF repo's `refs/main` pointed at an incomplete snapshot in our cache;
  pin `--revision` if the load reports missing weight files.
- **ParoQuant** works but is not tuned for a single card (see *ParoQuant, AutoRound and escha*);
  ggz14's numbers are from 2 × R9700.
- **Quark INT4-W4A16** has no Quark scheme in vLLM 0.29 (int4 weight-only), and its AWQ
  `algo_config` also trips `QuarkConfig.apply_vllm_mapper`; use the compressed-tensors INT4 above.

**Why a single BetterBench run cannot rank checkpoints by decode.** Every run is one deterministic
sampling trajectory (engine seed 0, a fresh nonce per run), and tokens per update of the same prompt
swing between 2.8 and 6 from one trajectory to the next; with 8 runs per category the weighted
decode carries about ±5 %. Checked on 2026-09-24 for the NVFP4 "+5 %": with the prompt lookup off and
fixed prompts the DFlash drafter accepts the same per draft position on all four checkpoints (4.04–4.13
tokens per step, greedy and T = 0.7); the prompt lookup never fires in BetterBench's short prompts
(0 of 14,416 steps: it only takes sources more than 2048 tokens back); fidelity on reference text
(NLL per token on code and prose) ranks ParoQuant 0.30 < Quark 0.349 < INT4 0.355 < NVFP4 0.393, i.e. no
link to acceptance. A fresh 128-run sample per checkpoint then put NVFP4 at +0.5 %. Rank decode on
≥ 128 runs per arm, or on the step gap, which is deterministic.

Prefill numbers carry about ±3 % between measurement sessions (the MXFP4 range above is two
sessions); within one window the same image repeats within 0.3 %. Quark and INT4 were measured in
one session, NVFP4 and ParoQuant in another, each against an MXFP4 arm of its own session.

### History: 0.2.9 (2026-09-21)

BetterBench 0.4.0, default config, all three phases (single-stream decode, prefill sweep, concurrency
1/2/4/8/16 × 48 requests; sampling at temperature 0.7 / top_p 0.95 / top_k 20, 3 warmup + 20 measured
passes per category), one run of 28 min against the production container: image 0.2.9, DFlash2 k=7
(probabilistic), fp8 KV, `--max-model-len 262144`, `--max-num-seqs 8`, `--max-num-batched-tokens 2048`,
KV pool 384,316 tokens, prompt lookup on. The server had no other traffic during the decode phase (at
most 1 running request and an empty queue in every sample of a counter sidecar, 15 s interval). Compared with the last
full vllm7 run (0.1.4, 2026-09-15, history below) and with the `full` baseline of the accept gate
(`ci/accept/baselines`, measured on 0.2.2 / 0.2.3).

**Concurrency** (48 requests per level, all ok):

| Concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| **aggregate tok/s** | **105.6** | **199.6** | **294.7** | **393.7** | **382.3** |
| Δ vs 0.1.4 (99.4 / 180.0 / 281.0 / 356.2 / 362.5) | +6.2 % | +10.9 % | +4.9 % | +10.5 % | +5.5 % |
| Δ vs `full` baseline 0.2.2/0.2.3 (109.8 / 190.0 / 305.0 / 388.0 / 385.8) | −3.8 % | +5.1 % | −3.4 % | +1.5 % | −0.9 % |
| per-request decode, median tok/s | 135 | 130 | 105 | 74 | 71 |
| TTFT p50 / p95, ms | 107 / 153 | 153 / 200 | 178 / 292 | 264 / 395 | 3258 / 5957 |
| tokens per client update (BetterBench) | 4.72 | 4.88 | 4.74 | 4.85 | 4.73 |
| scaling vs conc 1 | 1.00× | 1.89× | 2.79× | 3.73× | 3.62× |

`--max-num-seqs 8` caps the machine at conc 8; at conc 16 the surplus queues (TTFT p50 3.3 s, 7.0 s on
0.1.4). Against the 0.2.2/0.2.3 baseline the deltas go both ways and stay inside the gate's −5 %
threshold; the accept gate's own fast config (24 requests) measured conc 1 at 111.3 tok/s on this image.

**Single-stream decode** (20 passes per category, prompts of 60–112 tokens, 600-token cap; tok/s ± 95 % CI):

| Category | decode tok/s | Δ vs 0.1.4 | tokens/update | TTFT ms |
|---|---|---|---|---|
| chat | 82.1 ± 4.1 | +5.0 % | 3.09 | 111 |
| code | 130.5 ± 8.3 | +10.5 % | 4.94 | 107 |
| file_edit | 157.0 ± 6.8 | +10.8 % | 5.87 | 113 |
| json | 153.5 ± 9.1 | +12.7 % | 5.71 | 108 |
| math | 158.0 ± 8.2 | +9.0 % | 5.91 | 89 |
| prose | 77.7 ± 3.9 | +7.5 % | 2.95 | 89 |
| reasoning | 102.7 ± 13.7 | +15.3 % | 3.88 | 96 |
| summarization | 122.8 ± 6.6 | +8.4 % | 4.59 | 112 |
| **weighted** (BetterBench weights) | **122.4** (0.1.4: 110.0) | **+11.2 %** | 4.60 (4.57) | |

The gain is step time, not acceptance: the median step gap fell 42.15 → 38.23 ms (−9.3 %, worth +10.2 %),
tokens per update rose 4.57 → 4.60 (+0.8 %). What in the 0.2.x series is responsible cannot be separated
from this run — the attention tunes of 0.2.7 / 0.2.8 move the step by a few tenths of a millisecond at these context lengths.
Step gaps over all 15,581 decode updates: p50 38.2 ms, p90 38.5, p99 38.9, p99.9 40.4, max 41.8; none above
60 ms (no stalls).

**Prefill** (unique prompts, no prefix cache, 16 output tokens; there is no earlier vllm7 prefill sweep to
compare with):

| Prompt tokens | 1,514 | 5,918 | 11,793 | 23,543 | 47,055 |
|---|---|---|---|---|---|
| prefill tok/s | 2,073 | 2,053 | 2,020 | 1,913 | 1,713 |
| TTFT | 0.73 s | 2.88 s | 5.84 s | 12.3 s | 27.5 s |

**Draft acceptance** (server counters, before/after snapshot of the whole run, 42,069 draft steps): 4.20
tokens/step, acceptance rate 0.457; a draft position was accepted in 0.81 / 0.64 / 0.50 / 0.40 / 0.33 / 0.28 /
0.24 of the steps (positions 0…6). Independent of concurrency (sidecar, per level: 4.27 / 4.48 / 4.29 / 4.39 /
4.14 tokens/step at 1/2/4/8/16). 0 preemptions, peak KV usage 43.3 %, 0 failed requests (decode and
concurrency). BetterBench's "tokens/update" counts what the client sees per streamed update, so it is not the
same number as the server's tokens/step.

What this run does not show: BetterBench's prompts are tiny (longest 174 tokens, plus at most 600 generated),
so the context stays under ~800 tokens and the prompt-lookup override (which needs more than 2048 tokens of
context) cannot fire — its effect is measured in *Prompt lookup on top of the DFlash draft* above. It is a
single run, without repetition; the conc-1 figure against the old baseline (−3.8 %) is the one to repeat
before reading anything into it, and BetterBench itself flags 18 tail metrics as under-sampled (rule
n·min(p, 100−p)/100 ≥ 5), so only means and medians are quoted here. The 0.1.4 comparison spans everything
between 15 and 21 September. The INT4 reference row in the history table below was not re-measured. Raw
JSON and the HTML report: `/root/betterbench/results/vllm7-029-full-20260921.*` on the build host.

### History: 0.1.4 – 0.1.6 (2026-09-15/16)

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

## Considered and not adopted

What other R9700 stacks do that this image deliberately does not, with the reason. Every "measured" entry
is an A/B on the production arguments (BetterBench, see the sections above for the numbers); the rest are
design reasons. Revisit an entry when its reason changes.

**From ggz14/radiance-vllm-mxfp4 (upstream MXFP4 work, otherwise merged here):**

| Item | Why not |
|---|---|
| Dynamic verify width (`RADIANCE_DYNAMIC_WIDTH`, `patch_dynwidth` / `patch_async_dynwidth`) | Measured: conc 4/8/16 −3 %, conc 2 unchanged (ABAB, 300 W). Unequal per-request widths cost more than the trimmed verify rows save on this card. |
| GDN in_proj merge (`RADIANCE_GDN_MERGE_INPROJ`, `radiance_gdnmerge.py`) | Upstream's merged forward replaces `forward_hip` with the AITER GDN core; this image serves `forward_cuda` (FLA). A forward_cuda-sliced variant measured step −0.16 ms, prefill −1.5…−2.2 %, and −8.5k KV tokens (the weight copies fragment the arena). |
| Streaming decode loads (`RADIANCE_MXFP4_DECODE_NT=1`) | Measured on top of WPERM: step 37.6 vs 37.3 ms — slower. |
| libr4d decode GEMM (`RADIANCE_MXFP4_R4D_DECODE_MAX_M`) | Measured: not bit-identical, no step gain, −810 KV tokens. |
| Wide tile at M = 2048 (`RADIANCE_MXFP4_TN4_MIN_M` lowered/raised) | Measured neutral for both the folded and the A-tiled prefill GEMM. |
| DFlash2 selector top-k 24/32 (`patch_dflash_selector_topk`) | Measured: +0.8 % weighted decode, −2.5…−3 % conc 1; needs its own compile cache. |
| int2 draft head (`RADIANCE_FAST_DRAFT`) | Its own ~0.17 GiB head copy comes out of the KV pool; this image keeps the KV pool at the model maximum. |
| fp8 stream / arnq producers (`RADIANCE_FP8_STREAM`, `radiance_arnq.py`) | Same fusion as this image's `RADIANCE_FUSED_NORM_QUANT` (and its producers now emit the tiled layout too). |
| Lazy GDN snapshots (`RADIANCE_GDN_LAZY`) | Corrupts multi-turn chat (ggz14 turned it off themselves). |
| `RADIANCE_PRESHUFFLE`, `RADIANCE_VERIFY_HEAD`, `RADIANCE_DRAFT_RERANK` | Apply to FP8 block-scale checkpoints resp. the ParoQuant drafter, not to this model. |
| Top-k/top-p sampler kernels (`patch_topk_*`) | Measured the sampler's share: top-k 20 / top-p 0.95 cost 0.3 % of a step. |
| 3-rank all-reduce and 5/4-bit wire (`patch_ar_3rank`, `patch_ar_qbits`) | Written for the older two-rank `radiance_allreduce.py`; this image ships StillDeadcode's N-rank module (TP=3 rides RCCL), and the 5/4-bit kernels are not in the rebased libr4d extras. The rest of ggz14's multi-GPU work is in (0.3.5, *Several GPUs*). |

**Other images and forks:**

| Source | Why not |
|---|---|
| [testeddoughnut/vllm-openai-rocm-r9700](https://hub.docker.com/r/testeddoughnut/vllm-openai-rocm-r9700) (AITER-tuned image) | This image uses AITER only for attention. Its verify-batch decode config is tuned here (`sly/radiance_attn_decode.py`), and a 36-cell sweep of the 2D prefill config found aiter's stock cell within 1–5 % of the best (< 2 % of a 64k prefill); the GEMMs run on the MXFP4 HIP kernels, not AITER. Not evaluated in depth beyond that. |
| [hifi/vllm-radlight](https://codeberg.org/hifi/vllm-radlight) | An arrangement of ggz14 + libr4d on AMD's vLLM 0.27 image, patched at start (no original code). Its knobs are the upstream ones above; the remaining runtime flags (expandable segments, chunk 2560, fp16 SSM state, HW queues, HSA interrupts) are queued for an A/B. |
| [bkvargyas/r9700-stack](https://github.com/bkvargyas/r9700-stack) | Plugin stack for TP = 2 and NVFP4 checkpoints; its int6 embedding gather is behind this image's int4 embedding (`RADIANCE_EMBED_BITS=4`). |
| `tcclaviger/vllm`, `Dyluhn/R9V` | Separate forks with their own model mix (MoE, expert offload, TP ≥ 2); nothing single-GPU-MXFP4-specific to take over was found. |
| DFlash2-FP8 drafter (`tcclaviger/Qwen3.8-27B-DFlash2-FP8`) | Heavier than the W4A16 drafter, and stacks running it report fewer tokens per update than this image (code 4.71 vs 5.09). |
| Qwen3.8-27B-PARO-MXFP6 | Two GPUs only. |

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
| `RADIANCE_NVFP4_MXFP4` | `0` | – | 0.3.6: serve a compressed-tensors NVFP4 checkpoint by requantizing every linear to MXFP4 at load onto the W4A8 kernel (see *NVFP4 checkpoints* above). Needs `RADIANCE_MXFP4=1 RADIANCE_MXFP4_W4A8=1`. Inert for Quark checkpoints. |
| `RADIANCE_PAROQUANT` | `0` | – | 0.3.6: register the `paroquant` / `paroquant_mxfp4` configs (z-lab ParoQuant checkpoints; see *ParoQuant, AutoRound and escha*). Serve without `--quantization`. |
| `RADIANCE_AUTOROUND` | `0` | – | 0.3.6: register the `auto-round` config (int4 g128 sym on the W4A8 kernel) ahead of vLLM's INC config. |
| `RADIANCE_ESCHA` | `0` | – | 0.3.6: register the `escha` config (ExLlamaV3-derived format). |
| `RADIANCE_NVFP4_EXP` | `mse` | – | Block exponent rule of the requant: `mse` (no-clip vs one binade finer, per block by squared error), `ocp`, `noclip`. |
| `RADIANCE_NVFP4_FP8_LAYERS` | `mxfp4` | – | The checkpoint's FP8 per-channel linears: `mxfp4` requantizes them too (never lm_head); `fp8` leaves them on hipBLASLt fp8 — diagnostic only, it wedged the GPU under 8-way concurrency on ggz14's TP=2 box. |
| `RADIANCE_NVFP4_BF16_LAYERS` | `in_proj_ba` | – | Regex of unquantized linears to requantize as well. The GDN a/b gates must be on the radiance kernel for the fused add-rms quant in front of the GDN layers to fire. |
| `RADIANCE_NVFP4_LMHEAD` | `bf16` | – | The checkpoint's FP8 lm_head: `bf16` dequantizes at load (and `RADIANCE_LMHEAD_INT4=1` turns that into the int4 head), `fp8` keeps vLLM's fp8 path. All `RADIANCE_NVFP4_*` are part of the torch.compile cache key. |
| `RADIANCE_FUSED_NORM_QUANT` | `0` | `1` | Fused add+rms_norm / silu·mul / GDN gated norm + per-token fp8 quant in front of the W4A8 GEMMs (see above). Needs `RADIANCE_MXFP4_W4A8=1`, `RADIANCE_MXFP4_W4A8_MIN_M=0`, `RADIANCE_MXFP4_SANITIZE=0`; part of the torch.compile cache key. |
| `RADIANCE_FUSED_NORM_QUANT_ADD_RMS` / `_SILU` / `_GDN` | `1` | – | Per-fusion switches (only read when `RADIANCE_FUSED_NORM_QUANT=1`). |
| `GPU_MAX_HW_QUEUES` | ROCm default | `2` | ROCm HW queue count; 2 measured best for this single-process setup. |
| `RADIANCE_MXFP4_DECODE_TUNE16` | `1` | – | 0.2.1 cell table for decode M in (8, 64] (2–8 concurrent sequences): BK=128 everywhere in the band and split-K 2 for o_proj/out_proj at M=16/24, measured with `sly/mxfp4/bench_decode_cells.py`. `0` = pre-0.2.1 fill rule + `decode_bk64` (A/B control). M ≤ 8 is never touched. |
| `RADIANCE_MXFP4_DECODE_KS` | auto | – | Force the decode kernel's split-K factor (sweep/A-B knob, overrides the tables). |
| `RADIANCE_MXFP4_DECODE_BK` | auto | – | `128` pins BK=128, `64` forces the BK=64 instantiation where one exists (split 1 and 4) — sweep knobs, never a production setting. |
| `RADIANCE_MXFP4_DECODE_NT` | `0` | – | Non-temporal weight loads in the decode kernel. |
| `RADIANCE_MXFP4_TN4_MIN_M` | `2048` | – | M from which the folded kernel uses the wide TN=4 tile. |
| `RADIANCE_MXFP4_A_TILED_MIN_M` | `0` | `513` | 0.3.0: from this M on, the fused norm/quant producers write the fragment-tiled activation and the prefill GEMM takes `radiance_mxfp4_fp8_gemm_atiled` (+11 … +13 % prefill, bit-identical). Must exceed 512 and `DECODE_MAX_M`. |
| `RADIANCE_MXFP4_WPERM` | `0` | `1` | Fragment-order weights, permuted in place at load (0.3.1): step gap −1.3 ms, weighted decode +3.7 %, KV pool unchanged, bit-identical. |
| `RADIANCE_MXFP4_R4D_DECODE_MAX_M` | `0` | – | libr4d's `gemm_mxfp4a8_nt_m64` decode kernel (needs WPERM). Measured 2026-09-23: not bit-identical, no step gain, 810 KV tokens — off. |
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
| `RADIANCE_GDN_FUSED_DECODE` | `0` | – | Registers the HIP port of vLLM's fused GDN MTP decode kernel as `torch.ops._C.fused_gdn_decode_post_conv_mtp` (see *Fused GDN decode on ROCm*); use together with `VLLM_GDN_DECODE_KERNEL=cuda` (`=triton` = A/B control). |
| `RADIANCE_TP_PAD` | unset | – | 0.3.5: `3` pads the target to TP=3-divisible head counts with zero-weight dummies at load (`radiance_tp3pad.py`; see *Several GPUs*). `RADIANCE_TP_PAD_DRAFTER=0` leaves the drafter unpadded, `RADIANCE_TP_PAD_STRICT=0` demotes a coverage mismatch to a warning. |
| `RADIANCE_AR_MAX_KB` / `RADIANCE_AR_QUANT_MIN_KB` | `49152` / `128` | – | 0.3.5, TP=2 only: largest message on the P2P all-reduce kernel and smallest on its 6-bit wire (`sly/patch_ar_knobs.py`). |
| `RADIANCE_AR_QNT` / `RADIANCE_AR_QNB` | `1024` / `48` | – | 0.3.5, TP=2 only: threads per block / block cap of the 6-bit all-reduce. |
| `RADIANCE_GDN_FUSED_UPDATE` | `0` | – | 0.3.5 (libr4d extras): fused GDN decode step. Measured at TP=1: no gain. |
| `R4D_ATTN_FP8` | `0` | – | 0.3.5 (libr4d extras): 8-bit QK (`1`), PV (`2`) or both (`3`) legs of the R4D prefill attention; only with `--attention-backend R4D` and an fp8 KV cache. |

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
  -e RADIANCE_KV_GROUP_SIZE=8 \
  -e RADIANCE_EMBED_INT8=1 \
  -e RADIANCE_EMBED_BITS=4 \
  -e RADIANCE_MXFP4_A_TILED_MIN_M=513 \
  -e RADIANCE_MXFP4_WPERM=1 \
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
  --chat-template /opt/qwen-fixed.jinja \
  --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}' \
  --port 8000
```

Why these values:

| Argument | Note |
|---|---|
| `--quantization quark` | The checkpoint's quant method (fp4 + e8m0 scales); set explicitly rather than trusting auto-detect. |
| `--chat-template /opt/qwen-fixed.jinja` | froggeric's fixed Qwen template (v22.5), shipped in the image since 0.3.4: `medium` reasoning by default instead of the official 3.8 template's `xhigh`, no blank `<think></think>` injected into chat history (keeps the prefix cache), JSON-string tool arguments and `enable_thinking=false` do not crash, client effort aliases (`high`/`max` → `xhigh`, `none`/`off` → thinking off). |
| `--override-generation-config` | Server default sampling, the Qwen 3.8 thinking-mode recommendation: temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 0 (a non-zero penalty inside the chain of thought causes language mixing). Clients that switch thinking off (`enable_thinking: false` / `reasoning_effort: "none"`) should send the non-thinking values themselves: 0.7 / 0.8 / 20 / 0 / 1.5. The benchmarks in this README ran at 0.7 / 0.95 / 20. |
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

### Several GPUs (tensor parallel) — not tested on this box

This box has one R9700, so none of the multi-GPU paths below were run here. They are in the image
because ggz14 and StillDeadcode serve them on 2–3 cards; what was checked here is that they stay
out of the way at TP=1 (KV pool, throughput and GSM8K unchanged, see the 0.3.5 table above).

| TP | All-reduce | What to set |
|---|---|---|
| 2 | libr4d one-shot P2P kernel, exact bf16 up to `RADIANCE_AR_MAX_KB` (default 49152 KiB), 6-bit rotated wire above `RADIANCE_AR_QUANT_MIN_KB` (`RADIANCE_USE_R4D_AR_QUANT=1`, default) | `--tensor-parallel-size 2`. Raise `RADIANCE_AR_MAX_KB` to at least `max-num-batched-tokens × 5120 × 2 / 1024` (e.g. 98304 at 8192) or every prefill all-reduce silently falls back to RCCL. `RADIANCE_AR_QNT` / `_QNB` tune the 6-bit path (ggz14 ships 96 blocks). |
| 3 | RCCL (no 3-rank kernel in this image) | `--tensor-parallel-size 3 -e RADIANCE_TP_PAD=3 -e RADIANCE_MXFP4_WPERM=0`: `radiance_tp3pad` widens the heads (24/4/16/48 → 36/6/18/54), the MLP (17408 → 17472) and the vocab with zero-weight dummies at load, so every dimension divides by 3. Checked here on one card (`RADIANCE_TP_PAD=3` at TP=1, see below). The DFlash2 **W4A16** drafter is not padded (its packed int4 tensors are not in the padding tables), so at TP=3 either drop `--speculative-config` or use an fp8 DFlash2 drafter (ggz14's setup). |
| 4 / 8 | libr4d N-rank kernels (one-shot to 6 tokens, two-shot above, tiered-int8 wire at TP=4 with `RADIANCE_USE_R4D_AR_QUANT=1`) | `--tensor-parallel-size 4` / `8`. No padding needed (all head counts divide). |

`RADIANCE_USE_R4D_AR=0` keeps RCCL everywhere. Add `--device` access for every card and drop
`HIP_VISIBLE_DEVICES=0` (or list the cards). `--gpu-memory-utilization`, `--max-num-seqs` and the
CUDA-graph sizes in the production command were tuned for one 32 GB card and are only a starting
point.

TP=3 padding on one card (2026-09-24, `RADIANCE_TP_PAD=3`, TP=1, no speculative decoding):
the target is padded at load (1143 of 1695 tensors, coverage check OK), serves coherent text and
scores GSM8K 0.83 (100 questions, ±0.04; unpadded 0.835–0.845). The padded heads cost KV: 1.13×
instead of 1.46× of 262k on one card. Two things this run found and fixed: the coverage check
assumed a quantized MTP layer (the AMD checkpoint's `mtp.*` is bf16), and `RADIANCE_MXFP4_WPERM=1`
cannot serve the padded GDN `in_proj_ba` (N = 108 is not a multiple of 16; the fragment layout is
global to every kernel) -- **serve `RADIANCE_TP_PAD=3` with `RADIANCE_MXFP4_WPERM=0`**, which costs
the 3.7 % decode WPERM brings at TP=1.

## Build

Everything the build needs is in this directory (flat Docker context). Multi-stage: **builder**
(PyTorch, Triton, torchvision, AITER, vLLM from source for gfx1201) → **rocmprune** → **assemble**
(wheels, upstream RDNA4 patches, then the `sly/` patches, libr4d, the HIP kernel) → **venvsplit**
→ **final** (clean `ubuntu:24.04` + pruned ROCm + venv + entrypoint).

The final image has three large layers, from stable to volatile: the pruned ROCm tree
([`prune_rocm.sh`](prune_rocm.sh): gfx1201 only, no static archives, no Flang/MLIR), the **cold**
venv (the installed stack: torch, triton, aiter's metadata, the Python dependencies) and the **hot**
venv (the patched vLLM and aiter trees, the few files patched elsewhere, the radiance modules and
kernels). [`split_venv.py`](split_venv.py) makes the split and resets every mtime, so the cold layer
is byte-identical from release to release as long as the stack is: a `docker pull` of the next
release downloads the hot layer only, until the next ROCm or stack bump.

`docker inspect --format '{{json .Config.Labels}}' <image>` identifies an image: the
`org.opencontainers.image.*` labels (version, git revision, build date, source) and the
`io.slybase.radiance.*` component pins (vLLM, torch, triton, aiter, transformers, libr4d, ROCm base).

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
VERSION                       image version
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
| `ci` | PR, push to `main`, manual | `lint` (ruff E9/F63/F7/F82, shellcheck, hadolint, actionlint, `docker buildx build --check`), `patch-dryrun` (`ci/patch_dryrun.sh`: the pinned upstream sources from the Dockerfile ARGs in a venv, then the Dockerfile patch loop twice — pass 1 must apply every hunk, pass 2 must be all NOOP; `ci/patch_dryrun_skip.txt` is the documented skip allowlist), `constraints` (`ci/check_constraints.py`, see below), `consistency` (`ci/check_consistency.py`: every patch file is in the loop or in `ci/unused_patches.txt`, every `sly/patch_*.py` is documented in `sly/README.md`, image changes bump `VERSION`). The aggregate status **`ci`** is the required check on `main`. |
| `build` | push to `main` touching `VERSION`, tags `v*`, manual | Self-hosted runner (`rocm-build`, LXC 2408, CPU only — no `--device`, no GPU test, no deploy): `docker build` → import smoke test → push to `ghcr.io/slybase/vllm-sly-radiance:<VERSION>-rocm10.0` only for `v*` tags or the `push_ghcr` input. Build log is an artifact. |
| `upstream-sync` | daily 04:00 UTC, manual | Fast-forwards `upstream/*` from Codeberg (never force) and opens/updates a PR `upstream/<name>` → `main` (label `upstream-sync`) listing the new commits, the test-merge conflict status and the image-relevant files. Never merges. Needs the `SYNC_TOKEN` secret (fine-grained PAT, contents + pull-requests write) to create PRs. |
| `renovate` | every 6 h, push to `main` touching the config, manual | Self-hosted Renovate (same setup as SlyBase/helm-charts) with the upstream-sync App token (`SYNC_APP_*`). Optional secret `RENOVATE_GITHUB_COM_TOKEN` (read-only PAT) lifts the rate limit for lookups in other repositories. |

Renovate (`renovate.json`) tracks every Dockerfile ARG pin via the `# renovate:` markers above the
ARGs (ROCm base image tag + digest, ubuntu digest, torch/triton/torchvision as one group, vLLM,
AITER, transformers, rocm_bandwidth_test, libr4d commit), the CI tool versions, the actions and
`constraints.txt`. No automerge — every bump goes through `ci`, and vLLM/AITER/ROCm bumps
additionally need a build and an A/B run. `renovate/*` PRs are exempt from the VERSION-bump check;
they collect on `main` and ship with the next VERSION bump.

**Python dependencies.** Everything the image installs from PyPI is pinned in
[`constraints.txt`](constraints.txt) (`pip install -c` in the assemble stage). Without it every
rebuild that misses the build cache resolved vLLM's open ranges to whatever was newest that day.
Renovate opens one weekly PR (`renovate/python-deps`) for the minor/patch updates and one for the
majors (`renovate/major-python-deps`). The `constraints` job in `ci` replays the image's pip
resolution for the pinned vLLM ([`ci/check_constraints.py`](ci/check_constraints.py), ~30 s, no
build) and fails on a missing, stale or out-of-range pin. Two things keep those PRs green:

- **Generated limits.** Renovate looks each package up on its own and cannot know that numba pins
  llvmlite `<0.48` or that the OpenTelemetry packages pin each other exactly. The script derives
  these from the resolution and keeps them as `GENERATED` rules in `renovate.json` (`enabled: false`
  for exact pins, `allowedVersions` for upper bounds). CI fails when they are out of date.
- **Repair.** Tightly coupled pairs can still arrive half-updated (httpx2 without the httpcore2 it
  pins exactly, pydantic-core ahead of pydantic). On a `renovate/*` branch a failing check runs
  `--repair`: the changed pins become ranges between `main`'s version and the proposal, pip picks the
  newest consistent set (never newer than proposed, never older than `main`), and the job pushes it
  to the branch with the App token. Renovate leaves the branch alone after that.

After a vLLM bump (or when a pin stops resolving) re-resolve the whole file:

```bash
python3 ci/check_constraints.py --update   # needs python 3.12 on linux x86_64, like the runner
```

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
  HIP kernel; the libr4d extras (narrow-state GDN kernels, `sly/r4d/`) and the TP=3 padding
  (`radiance_tp3pad.py`).
- [vLLM](https://github.com/vllm-project/vllm), [AITER](https://github.com/ROCm/aiter),
  [DFlash](https://github.com/vllm-project/vllm/pull/52816).
- [vLLM](https://github.com/vllm-project/vllm) — `sly/gdn/radiance_gdn_decode.hip` is a HIP port of vLLM's
  fused GDN MTP decode kernel (`csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu`, v0.29.0,
  Apache-2.0), modified for gfx1201.
- [turboderp/exllamav3](https://github.com/turboderp-org/exllamav3) (MIT) — `escha/` (carried from
  ggz14; since 0.3.6 its kernel headers are built into `radiance_escha_kernel`) contains code derived
  from ExLlamaV3; its license is in `escha/EXLLAMAV3-LICENSE.txt`.
- [z-lab ParoQuant](https://huggingface.co/z-lab/Qwen3.8-27B-PARO) — the format; ggz14 wrote the W4A8
  gfx1201 reimplementation in `paroquant/`, the AutoRound kernel and the NVFP4 → MXFP4 requant (0.3.6).

License: same as upstream vllm-radiance (see `LICENSE`).
