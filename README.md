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
| `RADIANCE_MXFP4_DECODE_KS` | auto | – | Force the decode kernel's split-K factor (A/B control). |
| `RADIANCE_MXFP4_DECODE_BK` | auto (64) | – | `128` pins the old BK=128 decode tiling. |
| `RADIANCE_MXFP4_DECODE_NT` | `0` | – | Non-temporal weight loads in the decode kernel. |
| `RADIANCE_MXFP4_TN4_MIN_M` | `2048` | – | M from which the folded kernel uses the wide TN=4 tile. |
| `RADIANCE_MXFP4_A_TILED_MIN_M` | `0` | – | Tiled-A layout for very large M (must exceed 512 and `DECODE_MAX_M`). |
| `RADIANCE_MXFP4_WPERM` / `RADIANCE_MXFP4_R4D_DECODE_MAX_M` | `0` / `0` | – | Experimental: fragment-order weights + libr4d's `gemm_mxfp4a8_nt_m64` decode kernel. |
| `RADIANCE_MXFP4_MHIST` | `0` | – | Print every distinct `(N, K, M)` the plugin sees once (which M the decode path really issues). |
| `RADIANCE_MXFP4_DEBUG`, `_CHECKX`, `_CHECKALL`, `_REFLINEAR`, `_SHADOW`, `_SYNC`, `_KERNEL_N`, `_KERNEL_NK` | off | – | Correctness/diagnostic switches, see the header of `sly/mxfp4/radiance_mxfp4.py`. |

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
