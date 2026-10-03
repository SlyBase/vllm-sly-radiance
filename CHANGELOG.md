# Changelog

Every image version (`VERSION`, image tag `vllm-sly-radiance:<version>-rocm10.0`) has a section here.
The section of a version is the body of its GitHub release: `ci/release_tag.sh` publishes it when the
`v<version>` tag is cut, and `ci/check_consistency.py` fails a PR that bumps `VERSION` without one.
Measurements are on one AMD Radeon AI PRO R9700 with `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` + the DFlash2
W4A16 drafter and the production arguments; deep dives are in [docs/TECHNICAL.md](docs/TECHNICAL.md),
full benchmark tables in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions before 0.1.0 belong to the
upstream repositories (StillDeadcode/vllm-radiance, ggz14/radiance-vllm-mxfp4).

## [0.5.0] - 2026-10-03

### Added
- **Opt-in KV-cache offload** (`KVCACHE=off|ram|disk`, off by default): a second-tier prefix cache
  behind the GPU prefix cache for long-context / agent-style sessions, where every turn re-sends the
  whole conversation and the prefix outgrows the GPU pool. `KVCACHE=ram` adds a RAM tier in `/dev/shm`
  plus the 6 behavioural patches (mixed-hit, eagle-groups, mamba-stride, reconcile-reask, swa-align/
  touch, align-last-block); `KVCACHE=disk` adds a filesystem secondary tier plus the full 15-patch
  instrumented set and wants a host `kvcache-reap.sh` reaper. Off by default: with `KVCACHE` unset the
  serve command is byte-identical to the unmodified script. The offload patches, `kvwatch.py` and the
  `turnbench`/`tierbench` benches ship in `kv-cache/` and are applied at container start via the
  existing `/patches` mount (not baked).
- **MXFP6-PARO build tooling**: `paroquant/build_mxfp6.py` (the RTN builder with the per-row
  exponent-spread clamp) and `requant.sh FORMAT=mxfp6` (E2M3 grid) — the host-side tools to produce a
  MXFP6 W6A8 checkpoint from a base model. The serving path (MXFP6 kernel + loader integration,
  `QUANT=mxfp6`) lands with the kernel port in a follow-up; this release ships the tooling and the
  KV-offload feature only.

### Measured
- KV offload, one R9700 (gfx1201), TP=1, `KVCACHE_RAM_GIB=16`, three ~110k-token agent-style
  sessions served round-robin: a ~81k-token request drops from **38.8 s** (no cache) to **13.2 s**
  (RAM tier), tokens served from the tier in brackets. Numbers from `kv-cache/README.md`.
## [0.4.7] - 2026-10-04

### Changed
- **Faster image builds, same image.** radiance's HIP extensions (R4D, MXFP4 fp8, GDN decode,
  autoround/escha/paroquant) are compiled in their own `kernels` stage that copies only their
  own sources: a Python- or patch-only change no longer recompiles them (~2 min per build), and
  BuildKit builds them beside the patch chain. The three quant plugin kernels compile in
  parallel. The builder stage keeps a ccache in a BuildKit cache mount for triton, torchvision,
  aiter and vLLM, so a stack patch bump recompiles only what changed (torch not yet wrapped).

## [0.4.6] - 2026-10-03

### Fixed
- **Ship `radiance_lmhead_int2.py` into the image** (issue #90): the module was wired into
  the Dockerfile patch loop (`patch_lmhead_int2`) but omitted from the release-stage
  site-packages `COPY` allowlist, so `import radiance_lmhead_int2` raised
  `ModuleNotFoundError` in every image and the int2 lm_head feature silently fell through to
  int4/fp8/stock — `RADIANCE_LMHEAD_INT2` was a no-op because the module never reached disk.
  Added the missing allowlist entry; the module now lands in `${SP}/` like its int4/fp8 siblings.
- **README quickstart version** (issue #90): `docker pull` and `docker run` examples pointed at
  the stale `0.4.0` tag; updated to `0.4.6`.

## [0.4.5] - 2026-10-02

### Added
- **Self-service HSA/KFD diagnostics** (issue #73): when GPU enumeration fails while `rocm-smi`
  still sees the card, the startup precheck now prints the host-side facts that triage it
  without a GPU window — host kernel, `/dev/kfd` access, container pids limit — plus the
  recovery order (reboot first, raise the pids limit, then capture `strace`/`dmesg` evidence
  before rebooting). `docs` and `radiance_preamble` only; the default GPU path is unchanged.

## [0.4.4] - 2026-10-02

### Added
- **Greedy int2 lm_head** (`RADIANCE_LMHEAD_INT2=1`, `sly/mxfp4/radiance_lmhead_int2.py` +
  `sly/patch_lmhead_int2.py`): two-stage vocabulary head — a 2-bit coarse pass over all 151,936 rows
  and a bf16 re-rank of the top-16 — with roughly 7.5× less lm_head weight traffic per decode step
  than the int4 head (≈1.26 GiB saved at [151936, 5120]). Greedy-only: it declines under sampling
  and whenever `RADIANCE_LMHEAD_INT4` is active, so the default path is byte-identical.
  Pre-flight: `sly/check_lmhead_int2.py` (real-weight coverage gate) and 11 CPU unit tests.
- **Drafter bf16 pre-expansion** (`RADIANCE_DFLASH_BF16=1`, default off): the DFlash2 drafter's W4A16
  rows are dequantized once at the post-load hook into per-layer bf16 weights, and its `apply_weights`
  runs plain bf16 GEMMs; +3.22 GiB VRAM for 3.88× the hot-path weight budget. The default path is
  byte-identical (template parse + packer round-trip validated).
- **W4A8 prefill cell sweep** (`sly/mxfp4/bench_prefill_cells.py`): the prefill counterpart of
  `bench_decode_cells.py` — M = 96…8192, row-major vs fragment-tiled arms, `--check` validates every
  cell against an independent fp32 reference before any timing counts.
- `sly/audit_kernel_binary.sh`: a CPU-only VGPR/spill audit of the compiled kernel binaries
  (`--vgpr-max 213 --spill-max 0 --strict`), run against the image's .so as a per-kernel record.

### Measured
- CPU only so far: all 11 int2 unit tests pass and the packer round-trip is bit-exact; the greedy
  coverage gate (≥ 99.5 % in the coarse top-16 on real weights) runs in a 300 W GPU window, and the
  production numbers follow at the acceptance gate.

## [0.4.3] - 2026-09-30

### Added
- **Weight-only NVFP4 (NVFP4A16) checkpoints load.** `patch_nvfp4_mxfp4.py` routes compressed-tensors'
  `CompressedTensorsW4A4Fp4(use_a16=True)` branch to the radiance requant scheme under
  `RADIANCE_NVFP4_MXFP4=1`; stock forces Marlin there and aborted the load on ROCm
  (`bottlecapai/ThinkingCap-Qwen3.8-27B-NVFP4`).
- `radiance_nvfp4_diag.py` + `RADIANCE_NVFP4_DIAG=native|requant|fold` (`RADIANCE_NVFP4_DIAG_A8=1`):
  measurement-only NVFP4 schemes (bf16 dequant per forward, `--enforce-eager`) -- the exact checkpoint as
  reference, the requant with bf16 activations, and an emulated native NVFP4 kernel path ("NV fold").
- `sly/bench_fidelity.py` / `sly/check_fidelity.py`: deterministic prompt-logprob quality gate (top-20,
  14k tokens, dNLL ±0.012, KL, top-1 agreement) for numerics changes; `sly/check_nvfp4_diag.py` CPU check.

### Measured
- ThinkingCap NVFP4A16 (docs/TECHNICAL.md, *What the requantization costs*): the NVFP4 -> MXFP4 requant
  costs +0.093 NLL against the exact checkpoint (top-1 agreement 91 %); fp8 activations +0.006, the GDN
  a/b gates in MXFP4 vs bf16 ±0.000. An emulated NV fold (e2m1 x e4m3 block scale folded into the e4m3
  weight byte, as the kernel folds the MX exponent) with fp8 activations: +0.005 (~95 % of the loss
  back) -- the candidate for a native NVFP4 kernel path. int4 lm_head + int4 embedding: +0.006.
- Speed of the ThinkingCap serve path (300 W, compiled + DFlash2, short prompt): step 34.5 ms, i.e. the
  Quark step gap; `RADIANCE_NVFP4_BF16_LAYERS=` (bf16 a/b gates) is +0.6 ms/step slower for no measurable
  accuracy, so the `in_proj_ba` default stays (its comment no longer cites the rejected GDN merge).
## [0.4.2] - 2026-09-30

### Changed
- Pin the PyTorch stack to the pre-ROCm-10.1 "sanctioned trio": **torch 2.11.0** + triton 3.6.0
  + torchvision 0.24.1, down from torch 2.14.0 / triton 3.8.0 / torchvision 0.29.0. Per
  [ROCm/ROCm#6406](https://github.com/ROCm/ROCm/issues/6406), torch >= 2.12 causes abnormally
  high CPU usage after the first GPU operation; downgrading to torch 2.11 resolves it (the
  bug is expected to be fixed in the ROCm 10.1 release, after which the stack can be bumped
  again). `torchaudio` is not part of this image's stack, so no pin is added for it.

## [0.4.1] - 2026-09-27

### Added
- Split-K for the W4A16 Triton GEMM (`sly/patch_w4a16_tiles.py`, knob `RADIANCE_W4A16_SPLITK`, default on):
  the stock kernel runs the whole K loop in one workgroup, so the N = 5120 projections at decode M get
  160–320 workgroups for 64 CUs and stay latency-bound (INT4 target `down_proj`: 139 µs against 82 µs
  for the MXFP4 split-K kernel on the same shape). Per-shape entries in `_GFX12X_SPLITK`; a shape
  without an entry runs exactly as in 0.4.0. Affects compressed-tensors INT4 targets and the DFlash2
  W4A16 drafter. The same kernel carries two cheaper dequant schemes (scale applied to the fp32 tile
  result; bf16 magic-number nibble conversion with the zero point folded into the tile result) and an
  interleave-free unpack, selectable per table entry. Split-K partials stay <= 1 MiB (the CUDA-graph
  memory estimate charges them to the KV pool); `RADIANCE_W4A16_SPLITK_TABLE=<json>` loads a swept
  table without a rebuild.
- Serial split-K reduce: the program that finishes a tile last sums the partials in split order
  (per-tile counter, self-resetting) -- deterministic, and one launch less per split call (134 per
  step on the INT4 target at one request).
- Tiled W4A16 weight layout (`RADIANCE_W4A16_TILED`, default on): 16-row x 128-K blocks of 1 KB, so
  every row group a tile reads is one contiguous burst (the MXFP4 kernel's `WPERM` idea). The HIP
  skinny path for M <= 5 reads the row layout and is skipped for tiled weights.
- Fusions for W4A16 layers (`sly/patch_w4a16_fuse.py`): MLP gate_up + silu(gate) * up in one GEMM
  (`RADIANCE_W4A16_SILU`); GDN in_proj_ba re-quantized to int4 rows of in_proj_qkvz, one GEMM with two
  contiguous outputs (`RADIANCE_GDN_BA_W4`, INT4 target; frees the bf16 ba weights); the DFlash
  context-KV projection on the drafter's own int4 rows instead of a 105 MB bf16 copy read every step
  (`RADIANCE_DFLASH_KV_W4`, exact, frees ~78 MB for the KV cache); the DFlash2 grouped-conv
  kernel_projection (bf16, 10 calls per step) re-quantized to int4 (`RADIANCE_DFLASH_CONV_W4`; changes
  proposals only, the target verifies every token).
- The int4 lm_head in the tiled layout too (`RADIANCE_LMHEAD_INT4_TILED`, default on with
  `RADIANCE_W4A16_TILED`); it is the largest W4A16 GEMM of a step (2 x 656 MB).
- `sly/check_w4a16_fuse.py`: the post-load transforms end to end on stand-in layers against the unfused
  computation (fused silu, merged split, context-KV, conv projection, plain calls into transformed layers).
- `sly/check_w4a16_splitk.py` (numerics against an fp32 reference and bit-identity of the new paths,
  CPU via the Triton interpreter or GPU) and `bench_w4a16_tiles.py --splitk`.

### Measured
- One R9700, 300 W, production arguments, one GPU window (2026-09-28), step gap from streamed greedy
  requests (median chunk interval, deterministic): **RedHatAI/Qwen3.8-27B-INT4 42.70 → 37.43 ms
  (−12.4 %)**, 36.69 ms (−14.1 %) together with `RADIANCE_SKINNY_GEMM=all`; the control arm repeated
  at 42.76 ms. **Quark MXFP4 (production) 34.47 → 34.15 ms (−0.9 %, drafter GEMMs).** At unchanged
  acceptance (4.29 / 4.21 tokens per step) INT4 decode is ~125 instead of 107.8 tok/s; the gap to
  MXFP4 shrinks from −19 % to −7 % step time. At 8 concurrent requests INT4 stays far behind
  (297 vs 426 tok/s: M = 40–64, where the W4A16 kernels reach only ~250 GB/s).
- GSM8K 200 (cot zero-shot, greedy): INT4 0.825 (0.820 before), MXFP4 0.850 (baseline 0.835–0.845).
- Window E (2026-09-29, 300 W, tiled weights, serial split-K, all fusions): kernel sweep at M = 8 --
  down_proj 150.8 -> 87.7 us, out/o 70 -> 40, GDN qkvz 110.6 -> 79.2, attention qkv 98 -> 72, gate_up
  206 -> 176, drafter context-KV 194 (bf16) -> 54, drafter qkv / o / fc 47 / 48 / 207 -> 37 / 32 / 126; the
  tiled layout alone is worth 1.1-1.6x at M <= 16. Quark MXFP4 (production), all new knobs off -> on:
  step gap **34.52 -> 33.72 ms (-2.3 %)**, KV pool **384,316** (the split-K loss of 405 tokens is
  offset now that the drafter's context-KV no longer needs a bf16 copy). The INT4 target arms of that
  window were not run (stopped early).
- Window D (2026-09-29, split-K table of window B): INT4 step gap 42.68 -> 36.53 ms, BetterBench decode
  (128 runs) **INT4 127.5 tok/s against MXFP4 128.1** in the same window; GSM8K INT4 0.83.
- Window F2 (2026-09-30, image 0.4.1-rc5 with the window-E table, second starts): **INT4 all knobs off ->
  on (with `RADIANCE_SKINNY_GEMM=all`) 42.73 -> 34.61 ms (-19.0 %)**, within 3 % of MXFP4 (33.61 ms);
  KV pool 370,561 -> 375,820; GSM8K 200 0.835. Tokens per step 4.52 -> 4.24 on the four fixed prompts is
  a different greedy text, not the drafter (MXFP4 moves as much between windows, 4.41 / 4.70).
  **MXFP4 (production) all knobs on: 33.61 ms, BetterBench decode 131.0 tok/s** (128 runs; 0.4.0 in
  production: 130.4), KV pool 384,316, concurrency 1 / 8: 121.5 / 416.1 tok/s. The tiled int4 lm_head
  alone: 33.74 -> 33.61 ms (-0.4 %, bit-identical output).
- Not adopted: merging GDN in_proj_ba into in_proj_qkvz (96 extra rows add a 129th tile -- 93.9 us
  merged vs 79.2 + 3.6 separate), so `RADIANCE_GDN_BA_W4` defaults to 0; a decode-attention retune
  at long context (the shipped rule is within 1 % of the best cell, ~470 GB/s at 32k); W4A8 int8.
- Kernel level (M = 8, DRAM-cold, `bench_w4a16_tiles.py --splitk`): down_proj 150 → 95 µs,
  out/o 61 → 48, gate_up 200 → 178, GDN qkvz 106 → 82, attention qkv 89 → 76, drafter qkv 53 → 39,
  drafter fc 197 → 131.

## [0.4.0] - 2026-09-25

### Changed
- **vLLM 0.29.0 → 0.30.0**, aiter 0.1.21.post2 → 0.1.22.post1, torchvision 0.24.1 → 0.29.0, ubuntu:24.04
  digest, all Python pins re-resolved for vLLM 0.30 (`constraints.txt`: 209 pins, 34 changed, new
  `mooncake-transfer-engine-rocm` and `msgpack`; `setuptools` held at 79.0.1 because vLLM requires `<80`).
- Four patch anchors moved in vLLM 0.30 and were re-anchored for both versions (`apply_any`):
  `sly/patch_quark_mxfp4` (multi-line `_resolve_backend_kernels`), `sly/patch_lmhead_fp8` / `_int4`
  (`get_quant_method` now opens with `get_quant_method_target`), `patch_autoround` / `patch_escha`
  (`__all__` gained `resolve_quant_method`). `patch_unpad`, `patch_qwen3_toolparse` and
  `sly/patch_mamba_align_retire` are no-ops now: vLLM ships the same fixes.
- CI: ruff 0.16.8, renovate action v46.3.3, `create-github-app-token` v3.
- Renovate: no hourly or concurrent PR cap any more (updates were piling up rate-limited).
- Docs: README cut down to stack, reference numbers, quickstart, changes, options; details moved to
  `docs/`; this changelog; GitHub releases carry the changelog section.

### Measured
- Performance-neutral against 0.3.6 (prefill ±0.3 %, step gap unchanged, KV pool 384,316), GSM8K 0.855.
  Reference run: 133.2 tok/s decode, 3,166 tok/s prefill at 1.5k, 410 tok/s at conc 8 (300 W);
  124.5 / 2,527 / 362 at 210 W ([details](docs/BENCHMARKS.md#040)).
- New recommendation for shorter contexts: `--max-model-len 131072 --max-num-batched-tokens 4096`,
  +3.5 … +4.8 % prefill on 8k–64k prompts.

## [0.3.6] - 2026-09-24

### Added
- **NVFP4 checkpoints** via load-time requantization to MXFP4 (`RADIANCE_NVFP4_MXFP4=1`, ggz14's
  `radiance_nvfp4`), onto the same W4A8 kernel: `unsloth/Qwen3.8-27B-NVFP4` serves at the Quark
  decode speed, KV 374,202 tokens, GSM8K 0.825. Fixed a meta-tensor crash at load on the way.
- **ParoQuant** (`RADIANCE_PAROQUANT=1`), **AutoRound** (`RADIANCE_AUTOROUND=1`) and **escha**
  (`RADIANCE_ESCHA=1`) quantization configs and HIP kernels (ggz14), all opt-in. ParoQuant works on
  one card (GSM8K 0.84) but is not tuned for it (decode 67.7 tok/s).

## [0.3.5] - 2026-09-24

### Added
- **libr4d extras** (ggz14's rx10 patch, rebased onto our pin): GDN decode/prefill kernels for the
  bf16 SSM cache now bind instead of the FLA fallback. **Prefill +4.9 … +6.0 %**, step gap −0.7 ms,
  decode +1 %, KV pool unchanged, GSM8K 0.845.
- Multi-GPU paths for other users (untested on the maintainer's single card): TP=3 via zero-weight
  dummy heads (`RADIANCE_TP_PAD=3`, needs `RADIANCE_MXFP4_WPERM=0`), TP=2 all-reduce knobs
  (`RADIANCE_AR_MAX_KB`, …); TP=4/8 use libr4d's N-rank kernels.

## [0.3.4] - 2026-09-24

### Added
- froggeric's fixed Qwen chat template v22.5 at `/opt/qwen-fixed.jinja` (`--chat-template`).

## [0.3.3] - 2026-09-23

### Added
- HIP port of vLLM's fused GDN MTP decode kernel (`RADIANCE_GDN_FUSED_DECODE=1`, default off).

## [0.3.2] - 2026-09-24

### Changed
- Image halved (9.4 → 4.8 GB: ROCm prune to gfx1201, venv split into a stable and a volatile layer,
  so a release pull is ~35 MB); every PyPI dependency pinned in `constraints.txt`.

## [0.3.1] - 2026-09-23

### Changed
- `RADIANCE_MXFP4_WPERM=1` (fragment-order weights) permutes into the weight's own storage: step gap
  38.5 → 37.2 ms, weighted decode +3.7 %, KV pool back to 384,316.

## [0.3.0] - 2026-09-23

### Added
- Fragment-tiled activations from the fused norm/quant producers feed the A-tiled prefill GEMM
  (`RADIANCE_MXFP4_A_TILED_MIN_M=513`): **prefill +11 … +13 %**, bit-identical.

## [0.2.9] - 2026-09-21

### Added
- Prompt lookup on top of the DFlash draft (`RADIANCE_LOOKUP_DRAFT`, default on): text that repeats
  something more than 2048 tokens back (edits, quotes) decodes **+57 … +116 %** faster, lossless.

## [0.2.8] - 2026-09-20

### Added
- Split-KV launch for the DFlash2 drafter's sliding-window attention: 1.8–3.5× per call, step
  −0.2 … −1.6 ms.

## [0.2.7] - 2026-09-20

### Fixed
- The fp8 verify-batch decode attention tune never applied since aiter 0.1.21; ported to the new config
  tables. **Long-context decode +56 % at 33k, +122 % at 65k, +147 % at 99k tokens.**

## [0.2.6] - 2026-09-19

### Changed
- Ledger: unused upstream patches documented with reasons.

## [0.2.5] - 2026-09-18

### Fixed
- Load-time `max_split_size_mb` allocator scope on ROCm: KV pool +10k tokens (MXFP4), +73k (INT4).

## [0.2.4] - 2026-09-17

### Added
- int4 lm_head for compressed-tensors targets, INT4 target tile table.

## [0.2.3] - 2026-09-17

### Fixed
- Backport of vllm#55450: Mamba align-mode state retirement across null gaps (a 258k prompt exhausted
  the KV pool and preempted itself).

## [0.2.2] - 2026-09-17

### Added
- KV group size override and int4/int8 `embed_tokens`: +1.2–1.7 GiB for the KV cache.

## [0.2.1] - 2026-09-16

### Changed
- Measured decode cell table for M in (8, 64] (2–8 concurrent sequences).

## [0.2.0] - 2026-09-16

### Changed
- Merged the full upstream/ggz14 history (243 commits); the active image stayed the same.

## [0.1.6] - 2026-09-16

### Added
- Fused add+RMSNorm / SiLU·mul / GDN gated norm + fp8 quant in front of the W4A8 GEMMs: conc 1 +3 %,
  conc 8 +5.6 %.

## [0.1.5] - 2026-09-15

### Added
- int4 lm_head (W4A16, group 128): step 42.0 → 39.7 ms at conc 1, +8k KV tokens.

## [0.1.4] - 2026-09-15

### Added
- gfx1201 tile table for the DFlash2 W4A16 drafter GEMMs: step 43.4 → 42.0 ms.

## [0.1.3] - 2026-09-15

### Added
- fp8 lm_head: step 48 → 44.5 ms, +1.27 GiB KV.

## [0.1.2] - 2026-09-15

### Added
- DFlash2 with a W4A16 drafter; `DEC_MAX_N` 36864 puts `gate_up` on the decode kernel: step 67 → 48 ms.

## [0.1.1] - 2026-09-14

### Added
- gfx1201 `gemm_afp4wfp4` config for AITER (the gfx950 default crashed on RDNA4's 64 KiB LDS).

## [0.1.0] - 2026-09-13

### Added
- Fork of StillDeadcode/vllm-radiance with ggz14's MXFP4 work, ported to vLLM 0.29.0.
