# vllm-sly-radiance

[![Release](https://img.shields.io/github/v/release/SlyBase/vllm-sly-radiance?label=release)](https://github.com/SlyBase/vllm-sly-radiance/releases/latest)
[![Image](https://img.shields.io/badge/ghcr.io-1.8.1--rocm10.1-blue?logo=docker)](https://github.com/SlyBase/vllm-sly-radiance/pkgs/container/vllm-sly-radiance)
[![CI](https://github.com/SlyBase/vllm-sly-radiance/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/SlyBase/vllm-sly-radiance/actions/workflows/ci.yml)
[![Build](https://github.com/SlyBase/vllm-sly-radiance/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/SlyBase/vllm-sly-radiance/actions/workflows/build.yml)
![GPU](https://img.shields.io/badge/GPU-R9700%20gfx1201-red)
![ROCm](https://img.shields.io/badge/ROCm-10.1-green)
![vLLM](https://img.shields.io/badge/vLLM-0.32.0-orange)

**The fastest way to run Qwen3.8-27B on one AMD Radeon AI PRO R9700.** A vLLM image for gfx1201 (RDNA4)
with hand-written MXFP4 kernels, DFlash2 speculative decoding and up to 262k context on a single 32 GB card.

**1.8.0 at 300 W:** ~154 tok/s single-stream decode, ~3,900 tok/s prefill (3,370 at 47k tokens), 480 tok/s at
8 concurrent requests. Every number is measured and reproducible (see [Performance](#performance)).

```bash
docker pull ghcr.io/slybase/vllm-sly-radiance:1.8.1-rocm10.1
```

## Contents

- [Features](#features)
- [Tech stack](#tech-stack)
- [Performance](#performance) — [release history](#release-history-300-w-betterbench-040-full-run), [1.8.0](#180-2026-10-23), [vs the reference stack](#180-against-the-reference-stack)
- [Quickstart](#quickstart) — [best-performance launch](#best-performance-launch-180)
- [What this image changes, and why](#what-this-image-changes-and-why)
- [Options](#options) (configuration)
- [Repository and contributing](#repository-and-contributing)
- [Credits](#credits)
- [License](#license)

## Features

- **MXFP4 W4A8 GEMM kernels** written for RDNA4 (fp8 WMMA), with a load-time weight repack (PSW) and shape-tuned
  prefill tiles; bit-identical to the reference path.
- **DFlash2 speculative decoding** (k = 7), 5-6.5 tokens per step on code, JSON and math.
- **Fast long-context prefill**: double-buffered fp8 attention, a one-round GDN chunk scan, the drafter skips
  prefill chunks it cannot use.
- **Large KV pool**: fp8 KV cache by default, an opt-in 4-bit KV cache (TetraAttention) for 1.75x the tokens.
- **Measured, not claimed**: every default carries an A/B number, a quality check (GSM8K, KL divergence against
  bf16) and a release gate on real hardware.

## Tech stack

| Layer | Version |
|---|---|
| GPU | AMD Radeon AI PRO R9700, 32 GB, gfx1201 (RDNA4) — one card; TP 2/3/4/8 paths included but untested here |
| ROCm | 10.1 (`rocm/dev-ubuntu-24.04:10.1.0-full`, HIP 7.16, pruned to gfx1201) |
| PyTorch / Triton / torchvision | 2.13.0 / 3.8.0 (`git669b31ac`) / 0.28.0, all AMD's `+rocm10.1.0` wheels (the trio vLLM 0.31's ROCm base builds; unchanged at 0.32) |
| vLLM | 0.32.0 (V1 engine, V2 model runner), built from source |
| AITER / transformers | 0.1.24.post1 / 5.19.0 |
| Kernels | [libr4d](https://codeberg.org/StillDeadcode/libr4d) (prefill attention, gated delta net, all-reduce) + this repo's MXFP4 W4A8 GEMM, fused norm/quant, attention tunes |
| Model (benchmarks since 1.0.0) | [`slybase/Swift-1.5-Qwen3.8-27B-heretic-MXFP4-GPTQ`](https://huggingface.co/slybase/Swift-1.5-Qwen3.8-27B-heretic-MXFP4-GPTQ) (GPTQ MXFP4, gated-delta-net hybrid) |
| Also tested | [`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) (Quark MXFP4, the 0.x benchmarks) |
| Drafter | [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16), DFlash2 k = 7 + prompt lookup |
| Also serves | NVFP4, compressed-tensors INT4, ParoQuant, AutoRound, escha checkpoints ([comparison](docs/BENCHMARKS.md#other-checkpoints-on-one-r9700-036-2026-09-24)) |

## Performance

### Release history (300 W, BetterBench 0.4.0 full run)

| | **1.8.0** | 1.5.1 | 1.0.0 |
|---|---|---|---|
| Date | **2026-10-23** | 2026-10-17 | 2026-10-07 |
| **Decode, single stream** (weighted) | **154.4 tok/s** | 149.6 | 135.8 |
| Decode by category: code / json / math / file_edit / summarization / chat / reasoning / prose, tok/s | 186.5 / 210.8 / 191.4 / 173.4 / 144.0 / 131.9 / 107.5 / 90.5 | 167.5 / 174.5 / 182.1 / 183.7 / 159.0 / 116.4 / 118.5 / 101.0 | 148.7 / 168.4 / 172.0 / 179.9 / 141.3 / 99.3 / 105.7 / 84.6 |
| update gap p99 / TTFT p50 (short prompts) | 35.0 ms / 57 ms | 35.7 ms / – | 35.2 ms / – |
| **Prefill, prefill-only run**¹ 1.5k / 6k / 12k / 24k / 47k prompt tokens, tok/s | **3,927 / 3,965 / 3,890 / 3,692 / 3,370** | – | – |
| **Prefill, inside the full run**² 1.5k / 6k / 12k / 24k / 47k prompt tokens, tok/s | 3,645 / 3,668 / 3,619 / 3,505 / 3,196 | 3,024 / 3,032 / 2,994 / 2,766 / 2,430 | 3,132 / 3,136 / 3,105 / 2,949 / 2,635 |
| **Concurrency** 1 / 2 / 4 / 8 / 16, aggregate tok/s | **139 / 240 / 361 / 480 / 512** | 135 / 237 / 373 / 503 / 507 | 120 / 213 / 338 / 461 / 466 |
| GSM8K 200 (greedy) | 0.82 | 0.845 | 0.84\* |
| **KV pool**, tokens | 340,723 (65k context) | 439,031 (262k); 767,557 with the opt-in KV4 cache | 436,906 (262k) |
| Context / prefill chunk | 65,536 / 4,096 | 262,144 / 2,048 | 262,144 / 2,048 |
| Checkpoint | Swift-1.5 GPTQ MXFP4 | Swift-1.5 GPTQ MXFP4 | Swift-1.5 GPTQ MXFP4 |

All columns: one R9700 at 300 W, fp8 KV cache, DFlash2 k = 7, BetterBench 0.4.0 default config (3 warmup + 20
passes per category, 8 prefill runs per depth, 48 requests per concurrency level), T 0.7 / top_p 0.95 / top_k 20.
1.8.0 runs at a 65k context with a 4,096 prefill chunk; the older columns used the 262k production context with a
2,048 chunk, which is part of the prefill difference. GSM8K standard error is ±0.026, so all values tie.
¹ `betterbench --prefill 2000,8000,16000,32000,64000` on a freshly started server: 2 warmup + 8 runs per depth, median,
2 runs, 2026-10-22 night (the same-conditions comparison with the reference stack below). ² The prefill phase of the
full BetterBench run (phase order decode -> prefill -> concurrency, same 2 + 8 runs, same prompts), after the decode phase
on the same server; mean of the 2026-10-23 runs a and b. The older releases were only measured this way, so compare them with row ². 1.8.0 concurrency 1 is run a only (run b: 64, cold first requests). \* 1.0.0 GSM8K at 210 W. Full per-release tables: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### 1.8.0 (2026-10-23)

**Test environment (all 1.8.0 numbers below).** One AMD Radeon AI PRO R9700 (32 GB), in an LXC container on
Proxmox, ROCm 10.1 image `1.8.0-rocm10.1` (`1.6.0-rocm10.1` plus bind-mounted 1.8.0 kernels in the earlier A/B
arms). Model: Swift-1.5 Qwen3.8-27B heretic, MXFP4 (GPTQ checkpoint) with the DFlash2 drafter (k = 7). FP8 KV
cache, `--max-model-len 65536`, prefill chunk 4096 (`--max-num-batched-tokens`), prefix caching off,
`--no-async-scheduling`, GPU KV pool 340,723 tokens. Sampling in the benchmark: seed 42, temperature 0.7,
top-p 0.95, top-k 20 (BetterBench 0.4.0, full config). Host tuning: power cap 300 W and a voltage offset of
-50 mV set with LACT, core clock offset 0 (both optional, see the launch example below).

**Prefill-only run vs prefill inside the full run.** Both come from BetterBench 0.4.0 with the same protocol (2 warmup
+ 8 runs per depth, median) and the same prompts; the only difference is how BetterBench is started. The prefill-only
run (`--prefill`) hits a freshly started server; inside the full run the prefill phase follows the
single-stream decode phase on the same server. The full-run prefill is 5-7 % lower (2K: 3,645 vs 3,927), and the gap
reproduces at 210 W (-9 %). The junction temperature under load is the same in both (106-108 °C), so heat alone
does not explain it; the leading candidates are the server state after decode (allocator / KV block layout) and the
cold card at the start of a prefill-only run. Not resolved yet. Use the prefill-only numbers to compare with the
reference stack (measured the same way) and the full-run numbers to compare with older releases.

| Prefill, tok/s (300 W) | 2K | 8K | 16K | 32K | 64K |
|---|---|---|---|---|---|
| prefill-only run, 2026-10-22 night (2 runs) | **3,927** | **3,965** | **3,890** | **3,692** | **3,370** |
| prefill-only run, 2026-10-23 (2 runs) | 3,916 | 3,916 | 3,780 | 3,559 | 3,251 |
| inside the full run, 2026-10-23 (2 runs) | 3,645 | 3,668 | 3,619 | 3,505 | 3,196 |

Real prompt tokens per nominal depth: 1,515 / 5,918 / 11,794 / 23,543 / 47,056.

**Prefill, same conditions as the reference stack** (300 W, 2026-10-22 night, 2 runs per arm, prefill-only run), 2K / 8K / 16K / 32K / 64K nominal: ours **3,927 / 3,965 / 3,890 / 3,692 / 3,370** tok/s against the
reference stack 3,759 / 3,864 / 3,872 / 3,699 / 3,396, i.e. **+4.5 / +2.6 / +0.5 / −0.2 / −0.7 %**.

**Scheduling.** `--no-async-scheduling` stays the default: in the 210 W ABBA of 2026-10-22 it gave prefill
+1.2 .. +7.7 % and no measurable decode difference. **Speculative depth:** k = 7 is the best measured value
(concurrency 1: 124 tok/s at k = 7, 114 at k = 5, 92 at k = 3).

**Quality.** KL divergence against a bf16 layer-by-layer reference over 15,177 positions: **0.0527**, top-1
agreement 93.1 %, perplexity +2.2 %. 1.8.0 is bit-identical to 1.6.0 (KL divergence 0 between them); the
weights-only exact path scores 0.0403.

### 1.8.0 against the reference stack

Both stacks on the same card, the same harness and the same session conditions; the reference stack runs its own
image and its own defaults.

**Prefill, prefill-only run (same conditions, both stacks):**

| | **1.8.0** | **reference stack** | **delta** |
|---|---|---|---|
| Prefill 2K (tok/s) | **3,927** | 3,759 | **+4.5 %** |
| Prefill 8K | **3,965** | 3,864 | **+2.6 %** |
| Prefill 16K | **3,890** | 3,872 | +0.5 % |
| Prefill 32K | 3,692 | **3,699** | −0.2 % |
| Prefill 64K | 3,370 | **3,396** | −0.7 % |
| KV pool (65k context) | **340,723 tokens** | 174,634 tokens | 1.95x |

**Decode and concurrency (both stacks, one start per arm, 32-64 requests per level):**

| | **1.8.0** | **reference stack** | **delta** |
|---|---|---|---|
| Decode, single stream (weighted) | **154.4 tok/s** | 144.3 tok/s\* | +7.0 % |
| Concurrency 1 / 2 / 4, aggregate tok/s | **139 / 240 / 361** | 114 / 198 / 305\* | +22 / +21 / +18 % |
| Concurrency 8, aggregate tok/s | **480** | 422.9 (published) | +13.5 % |

Conditions: one R9700 at 300 W, −50 mV, same card and the same BetterBench 0.4.0 harness for both stacks, prefix
caching off, thinking off, 65,536 context, FP8 KV cache, DFlash2 k = 7, chunk 4096, sampling seed 42 / T 0.7 /
top-p 0.95 / top-k 20. Prefill = the prefill-only sweep (2 warmup + 8 runs per depth, median, 2 runs per arm,
2026-10-22 night; real prompt tokens 1,515 / 5,918 / 11,794 / 23,543 / 47,056). The reference stack is measured on
our own card from its published image, so stack, checkpoint and default arguments are all its own. 1.8.0 runs the
image defaults (see [Options](#options)) plus `--no-async-scheduling`.
\* Reference decode/concurrency from a single start per arm (2026-10-21); the published values of that image are
154.8 tok/s weighted decode and 422.9 tok/s at eight concurrent requests, i.e. decode is a tie and 1.8.0 is ahead
where the card is loaded.

Per-category tables, the measurement method and every earlier release (incl. the 1.5.1 KV4 arm and the old 300 W /
210 W comparison) are in [docs/BENCHMARKS.md](docs/BENCHMARKS.md). Decode differences below ~5 % between single
runs are noise; see there for why.

## Quickstart

Requirements: one R9700 (or another gfx1201 card with 32 GB), ROCm-capable kernel driver
(`/dev/kfd`, `/dev/dri`), Docker, ~20 GB of disk for the model + drafter, and enough free host RAM for the model load. Since **1.2** the
MXFP4 checkpoint is sharded by default, so vLLM loads it shard-by-shard and the peak host RAM is the largest shard (5 GiB)
instead of the whole ~19 GiB model — a 16 GiB box now loads it in minutes rather than swapping for hours. On the maintainer's
32 GB host the other VMs are paused while the model loads; on 16 GiB hosts they no longer need to be.
Windows 11 through WSL2: see [docs/WINDOWS-WSL2.md](docs/WINDOWS-WSL2.md).

```bash
# 1. model + drafter into the Hugging Face cache
hf download slybase/Swift-1.5-Qwen3.8-27B-heretic-MXFP4-GPTQ
hf download syvai/Qwen3.8-27B-DFlash2-W4A16

# 2. serve (OpenAI-compatible API on :8000)
# --group-add takes the numeric GIDs of the host's GPU groups (do not use the "video"/"render"
# names: docker resolves them against the container's group file, and the ROCm base image
# carries no "render" entry). Check yours with `getent group video render` — on Ubuntu 24.04
# these are 44 and 109; you must be a member of both (`sudo usermod -aG video,render $USER`).
docker run -d --name vllm --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 109 \
  --security-opt seccomp=unconfined --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm -v triton-cache:/root/.triton -v aiter-cache:/root/.aiter \
  -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=2 \
  -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
  -e RADIANCE_MXFP4_DECODE_MAX_M=128 -e RADIANCE_MXFP4_A_TILED_MIN_M=513 -e RADIANCE_MXFP4_WPERM=1 \
  -e RADIANCE_LMHEAD_INT4=1 -e RADIANCE_FUSED_NORM_QUANT=1 \
  -e RADIANCE_KV_GROUP_SIZE=8 -e RADIANCE_EMBED_INT8=1 -e RADIANCE_EMBED_BITS=4 \
  -e RADIANCE_MXFP4_WIDE_MAX_M=192 -e RADIANCE_ADAPTIVE_WIDTH=perseq \
  ghcr.io/slybase/vllm-sly-radiance:1.8.1-rocm10.1 \
  --model slybase/Swift-1.5-Qwen3.8-27B-heretic-MXFP4-GPTQ \
  --max-model-len 262144 --gpu-memory-utilization 0.98 \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config.method dflash \
  --speculative-config.model syvai/Qwen3.8-27B-DFlash2-W4A16 \
  --speculative-config.num_speculative_tokens 7 \
  --speculative-config.draft_sample_method greedy \
  --speculative-config.attention_backend TRITON_ATTN \
  --attention-backend R4D_HYBRID \
  --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
  --compilation-config.cudagraph_capture_sizes '[8,16,24,32,40,48,56,64]' \
  --enable-prefix-caching --skip-mm-profiling --enable-mm-embeds \
  --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --chat-template /opt/qwen-fixed.jinja \
  --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}'

# 3. test
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"slybase/Swift-1.5-Qwen3.8-27B-heretic-MXFP4-GPTQ","messages":[{"role":"user","content":"Hello!"}]}'
```

- **First start** compiles kernels and CUDA graphs (~10 min) and reports a ~2 GiB smaller KV pool;
  **restart once** — the warm start takes 4–6 min and gets the full 436k-token KV pool. Keep the cache
  volumes.
- **Power:** 210 W costs ~6.5 % decode and ~20 % prefill against 300 W ([0.4.0 measurement](docs/BENCHMARKS.md#earlier-readme-reference-runs)). The card allows
  up to 330 W: +4 … +8 % prefill and ~+1 % step time over 300 W, but the junction runs at 102 °C average
  (106 °C peak) and the fan at ~4,200 rpm — not worth it for 24/7 use.
- **iGPU:** `HIP_VISIBLE_DEVICES=0` keeps ROCm off an integrated GPU next to the R9700.
- Other checkpoints: [NVFP4, INT4, ParoQuant](docs/BENCHMARKS.md#other-checkpoints-on-one-r9700-036-2026-09-24);
  several GPUs: [docs/MULTI-GPU.md](docs/MULTI-GPU.md).

### Best-performance launch (1.8.0)

The image defaults carry the whole measured stack (the prefill knobs are Dockerfile `ENV`, the PSW weight layout
is the module default), so this launch needs no kernel `-e` beyond the quantization and memory settings. It is the
configuration behind the 2026-10-23 numbers above; the model is the Swift-1.5 Qwen3.8-27B heretic MXFP4 (GPTQ)
checkpoint in a local directory.

```bash
# GIDs of the GPU groups: getent group video render
docker run -d --name vllm --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 109 \
  --security-opt seccomp=unconfined --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v ~/models/swift-heretic-gptq:/models/swift-mxfp4:ro \
  -v vllm-cache:/root/.cache/vllm -v triton-cache:/root/.triton -v aiter-cache:/root/.aiter \
  -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=1 \
  -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
  -e RADIANCE_MXFP4_DECODE_MAX_M=128 -e RADIANCE_MXFP4_SANITIZE=0 \
  -e RADIANCE_LMHEAD_FP8=1 -e RADIANCE_LMHEAD_INT4=1 -e RADIANCE_FUSED_NORM_QUANT=1 \
  -e RADIANCE_KV_GROUP_SIZE=8 -e RADIANCE_EMBED_INT8=1 -e RADIANCE_EMBED_BITS=4 \
  -e RADIANCE_MXFP4_A_TILED_MIN_M=513 -e RADIANCE_MXFP4_WPERM=1 -e RADIANCE_MXFP4_WIDE_MAX_M=192 \
  -e RADIANCE_ADAPTIVE_WIDTH=perseq -e RADIANCE_CACTUS_DELTA=0.5 \
  ghcr.io/slybase/vllm-sly-radiance:1.8.1-rocm10.1 \
  --model /models/swift-mxfp4 --served-model-name swift-1.5 \
  --max-model-len 65536 --gpu-memory-utilization 0.98 \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config.method dflash \
  --speculative-config.model syvai/Qwen3.8-27B-DFlash2-W4A16 \
  --speculative-config.num_speculative_tokens 7 \
  --speculative-config.draft_sample_method probabilistic \
  --speculative-config.attention_backend TRITON_ATTN \
  --attention-backend R4D_HYBRID \
  --max-num-seqs 8 --max-num-batched-tokens 4096 \
  --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
  --compilation-config.cudagraph_capture_sizes '[8,16,24,32,40,48,56,64]' \
  --no-enable-prefix-caching --no-async-scheduling \
  --skip-mm-profiling --enable-mm-embeds \
  --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --chat-template /opt/qwen-fixed.jinja \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}'
```

What the flags buy: prefix caching off and `--no-async-scheduling` give the best prefill (and let the drafter
prefill skip act, which needs prefix caching off); chunk 4096 with a 65,536 context is the shape of the
measurements. For a long-context or multi-turn setup use the [quickstart](#quickstart) (prefix caching on,
262k context) instead; the two trade prefill speed for cache reuse.

**Optional host tuning** (not part of the image, measured with it): power cap 300 W (LACT), voltage offset -50 mV (LACT), core clock offset 0. The first is worth ~20 % prefill over 210 W
([docs/BENCHMARKS.md](docs/BENCHMARKS.md#earlier-readme-reference-runs)); check the junction temperature after changing the voltage offset.

## What this image changes, and why

On top of vLLM, [StillDeadcode/vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance) (the RDNA4
build, libr4d kernels, gfx1201 fixes; FP8 on two cards) and
[ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4) (MXFP4 on RDNA4, the W4A8
kernel), this fork makes one card with one MXFP4 model as fast as it gets. Each item was an A/B on the
production arguments; the numbers are the gain it measured when it went in
([details](docs/TECHNICAL.md), [per release](CHANGELOG.md)).

| Change | Why | Gain |
|---|---|---|
| MXFP4 loader + W4A8 fp8-WMMA kernel ported to vLLM 0.29/0.30, gfx1201 GEMM configs | stock vLLM emulates MXFP4 in bf16 on RDNA4, and AITER's config crashed on 64 KiB LDS | usable at all |
| `gate_up` (N = 34,816) on the decode kernel, measured decode cell table | the widest GEMM silently fell to the prefill kernel | step 67 → 48 ms |
| DFlash2 with a W4A16 drafter + gfx1201 tile table | vLLM's drafter path assumed fp8/bf16 drafters and Llama-sized tiles | ~4.5 tokens per step |
| fp8 → int4 lm_head, int4 embeddings, KV group size, allocator scope | the 2.5 GB vocabulary head was read twice per step; memory goes to the KV pool | step −8 ms, full 262k context |
| Fused RMSNorm/SiLU/GDN-norm + fp8 quant, fragment-tiled activations, fragment-order weights | one kernel instead of three per GEMM input, contiguous weight reads | prefill +11–13 %, decode +3.7 % |
| libr4d narrow-state GDN kernels (bf16 SSM cache) | the pinned libr4d fell back to Triton for a 16-bit state | prefill +5 %, step −0.7 ms |
| Verify-batch decode attention tune (aiter 0.1.21+), split-KV drafter attention | the upstream tune had silently stopped applying | decode at 99k context **2.5×** |
| Prompt lookup over the whole context | the drafter only sees the last 2,048 tokens | edits/quotes of far text **+57–116 %** |
| `--attention-backend R4D_HYBRID`: libr4d prefill attention for runs >= 512 tokens, AITER for the rest (1.0) | plain R4D prefills faster but decodes at 60.9 instead of 34.9 ms/step; the hybrid keeps AITER's decode | prefill 8k / 32k / 64k **+3 / +13 / +23 %** |
| `--gpu-memory-utilization 0.98` + `FULL_DECODE_ONLY` (1.0) | the PIECEWISE graph set cost 1.24 GiB of graphs and 1.69 GiB peak activation | KV pool 391,193 → **436,097** tokens (+11.5 %) |
| `RADIANCE_MXFP4_WIDE_MAX_M=192` (1.0) | M 129…192 ran on a 256-row tile with up to 49 % padding; now the split-K decode kernel | TTFT of a 160-token prompt **−13.8 %** |
| `RADIANCE_ADAPTIVE_WIDTH=perseq` (1.0, ported from the Radiance engine) | acceptance differs 2× between requests; per-request verify width on varlen FULL graphs ([why ggz14's version lost](docs/TECHNICAL.md#100-what-the-release-adds-and-the-measurements-behind-it)) | 8 concurrent requests **+4.7 %** |
| ROCm 10.1 + torch 2.12 (1.0) | current AMD toolchain; the CPU-spin bug that held torch back is fixed | neutral (±1–2 %) |
| vLLM 0.31.0 + torch 2.13 / triton 3.8.0 / torchvision 0.28.0 / aiter 0.1.23 (1.1) | the combination vLLM's own ROCm base builds, as AMD wheels | *pending GPU run* |
| vLLM 0.32.0 + aiter 0.1.24.post1 / transformers 5.19.0 (1.6), same torch/triton/torchvision trio | the stack vLLM 0.32.0's own ROCm base pins (aiter held back until now because 0.1.24 had dropped an alias vLLM still imported; 0.1.24.post1 restores it) | *pending GPU run* |
| Constraints for every PyPI pin, 4.8 GB image with a stable/volatile layer split | reproducible builds, ~35 MB update pulls | – |

What was tried and rejected, with reasons: [docs/NOT-ADOPTED.md](docs/NOT-ADOPTED.md).

## Options

Recommended = what the reference run above uses. All knobs are environment variables read at start;
everything not listed is off by default and documented in [docs/TECHNICAL.md](docs/TECHNICAL.md) and
[sly/README.md](sly/README.md).

### Kernels and memory (set these)

| Variable | Recommended | Why |
|---|---|---|
| `RADIANCE_MXFP4` | `1` | unlocks native MXFP4 on gfx1201; without it vLLM emulates in bf16 |
| `RADIANCE_MXFP4_W4A8` | `1` | the hand-written fp8-WMMA MXFP4 GEMM ahead of AITER's |
| `RADIANCE_MXFP4_W4A8_MIN_M` | `0` | the W4A8 kernel also for small M (decode) |
| `RADIANCE_MXFP4_DECODE_MAX_M` | `128` | split-K decode kernel up to M = 128: covers `max-num-seqs × 8` and the mixed CUDA-graph sizes |
| `RADIANCE_MXFP4_WIDE_MAX_M` | `192` (default `0` = off) | split-K decode kernel with 9–12 M fragments for M 129…192 instead of the padded 256-row tile: TTFT of 160-token prompts −13.8 %. Not 256: the folded tile wins again there (+11 % GEMM time at M 256). Needs `WPERM=1` |
| `RADIANCE_ADAPTIVE_WIDTH` | `perseq` (default `0` = off) | per-request verify width, graph-safe (varlen FULL graphs): c8 with arrivals +4.7 %, one stream unchanged. `uniform` (one width per batch) gave +1.5 %. Tuning knobs `RADIANCE_AW_*` in [sly/README.md](sly/README.md) |
| `RADIANCE_NSD` + `RADIANCE_NSD_THETA` | `0` (off) + `0.95` (opt-in, **unmeasured**) | Nucleus Speculative Decoding ([arxiv 2610.07822](https://arxiv.org/abs/2610.07822)) in the rejection sampler: a draft token inside the target nucleus (mass of strictly higher target logits <= theta) is accepted besides the usual test; residual and bonus token unchanged. Only for sampling (temperature > 0), greedy is untouched. Trades a bounded deviation from the target distribution for acceptance; theta 0.95 is the paper's default, 0.90 is the second value to test. Excludes `RADIANCE_CACTUS_DELTA` (both set = startup error). Details: [sly/README.md](sly/README.md) `patch_nsd.py` |
| `RADIANCE_R4D_PREFILL_MIN_Q` | `512` | with `--attention-backend R4D_HYBRID`: smallest per-request chunk that goes to libr4d's prefill kernel; smaller chunks (decode, TTFT of short prompts) stay on AITER |
| `RADIANCE_R4D_HYBRID_ROUTE` | `1` | `0` = the hybrid backend never routes to libr4d (= AITER, for A/Bs) |
| `R4D_ATTN_DB` (+ `R4D_ATTN_TILE`) | `1` + `16` (image ENV; `R4D_ATTN_DB=0` reverts) | 1.8.0: double-buffered K/V staging, one barrier per key tile (O_DB), only with `R4D_ATTN_FP8=3` — otherwise inert. Bit-identical output, one layer M = 4089: 27.128 → 26.310 ms at ctx 40,890 (−3.0 %), 11.763 → 11.313 ms at 16,356 (−3.8 %), 1.687 → 1.571 ms at 0 (−6.9 %). `R4D_ATTN_TILE=16/32/48` (default 48) picks the key tile; tile 16 *without* DB is 7.7 % slower than tile 48, which is why the earlier tile-16 attempt lost |
| `RADIANCE_GDN_SCAN_VARIANT` | `s3` (image ENV; empty reverts to stock) | 1.8.0: `s1` / `s2` / `s3` select build-time variants of the GDN chunk scan in the same `r4d.so` (`s1` = no `-mcumode`, `s2` = + LDS diet, `s3` = + late staging). DIGEST bit-identical at T = 64/65/880/1760/4089 and varlen; one layer T = 4089 stock 0.5720 ms, s1 0.4860 (−15.0 %), s2 0.5366 (−6.2 %), s3 0.4255 (−25.6 %); s3 189 VGPR / 43,008 B LDS vs stock 256 / 4-VGPR spill / 60,928 B. s1+s3 is the pair, s2 alone is worse than s1. Details: [docs/GDN-CHUNK-SCAN-VARIANTS.md](docs/GDN-CHUNK-SCAN-VARIANTS.md) |
| `RADIANCE_DFLASH_PREFILL_WINDOW_SKIP` (+ `..._MARGIN`) | `1` (image ENV, module default `0`; `=0` reverts) | 1.8.0: lossless (GPU identity test: identical text and per-position acceptance counters, knob on vs off, 10 requests). Skip the drafter's `fc`, context-K/V fill and draft step on intermediate prefill chunks that end more than `window(2048) + margin(512)` below the prompt length — the scheduler drops its draft tokens there anyway. Needs prefix caching **off** (with APC on it warns once and does nothing), all-sliding drafter layers, DP = 1, so it only acts with prefix caching off. CPU test `sly/tests/test_dflash_prefill_skip_cpu.py`; +2.1 / +3.6 / +3.2 % prefill at 16K / 32K / 64K |
| `RADIANCE_ATTN_GATE_VIEW` | `0` | Keeps the attention output gate a strided view of q_proj's output instead of `gate.reshape(...)`: on ROCm the eager path copies the gate (M x 6144 bf16 per attention layer) because attention splits the graph. The o_proj input is bit-identical (CPU test, eager and compiled). rocprofv3 trace of 1.8.1: the copy is 0.38 % of prefill GPU time at 1.5k-12k. Measured together with `RADIANCE_DFLASH_AUX_FROM_RESIDUAL=1` and `RADIANCE_GDN_EMPTY_OUT=1` (R9700, 300 W, A/B/A, median of 3): prefill **+0.47 / +0.59 / +0.58 / +0.60 / +0.58 %** at 1.5k / 6k / 12k / 24k / 48k, output identical. Changes the traced graph: in the compile key (`sly/patch_graph_knob_keys.py`) |
| `RADIANCE_DFLASH_AUX_FROM_RESIDUAL` | `0` | The DFlash aux hidden states are taken from the next layer's input residual (`add_rms_quant` already writes `h + r`) instead of a second residual add: same bits, 5 add kernels fewer per step. Trace: 0.11-0.13 % of prefill GPU time. In the compile key via `RADIANCE_DFLASH_*` |
| `RADIANCE_GDN_EMPTY_OUT`, `RADIANCE_GDN_STRIDED_GATES` | `0` (`patch_gdn_glue.py`, in the loop since this version) | `core_attn_out` via `torch.empty` (the R4D paths zero the pad rows themselves, the Triton fallback zero-fills) and no `b`/`a` contiguous copies on the R4D path. Measured neutral at decode (2026-09-02); prefill trace of 1.8.1: the zero-fill is 0.12 % of the GPU time (it runs at ~2.6 TB/s, the MALL absorbs it), and without it the GDN `conv_prep` kernel after it runs 14-18 % faster. Now in the compile key |
| `RADIANCE_MXFP4_AT_PSW` | `auto` (module default; `0` = off, `2` = on and refused on a conflict) | 1.8.0: PSW weight layout. At load every MXFP4 layer's nibble plane is repacked in place (preshuffled nibbles, pre-folded scale; same shapes, no extra memory) and the GEMM kernels read it directly; output bit-identical. `auto` = on unless the launch has a knob whose reader needs the checkpoint byte order (`W4A8_MIN_M>0`, no `WPERM`, `DECODE_NT`, `R4D_DECODE_MAX_M`, debug/check knobs, ParoQuant), then off and the reason is logged. The production launch qualifies. 300 W night A/B, only PSW differing: prefill **+3.1 / +2.4 / +4.2 / +2.9 / +3.5 %** at 2K/8K/16K/32K/64K; 210 W: 1,093 greedy digests identical / 0 wrong, GSM8K-200 0.835 = 0.835 |
| `RADIANCE_MXFP4_TALL` (+ `..._TALL_MIN_M`) | `1` (image ENV; default `2048`; `=0` reverts) | 1.8.0: gate_up only (N 34,816), M >= 2048, bit-identical to TN4, 1.5-1.8 % faster on that GEMM. Not for the merged GDN in_proj (measured slower) |
| `RADIANCE_ATTN_Q_FP8` | `1` (image ENV; `=0` reverts) | 1.8.0: the prefill attention kernel reads the e4m3 query directly, the 57 ms bf16 widening copy of a 47k prefill is gone; bit-identical |
| `RADIANCE_MXFP4_TN4_MIN_M`, `RADIANCE_FUSED_NQ_REG`, `RADIANCE_MXFP4_AT_SKIP_PAD`, `RADIANCE_MXFP4_TILED_OPROJ`, `R4D_ATTN_FP8`, `RADIANCE_DFLASH_FC_TILED_MIN_M` | `513`, `1`, `1`, `1`, `3`, `513` (image ENV since 1.8.0; the first four were launcher-only before) | the 1.6.0 prefill stack and the drafter `fc` tile, now defaults of the image itself, not only of `serve-mxfp4.sh` |
| `RADIANCE_NO_ASYNC_SCHED` | `1` in `serve-mxfp4.sh` (ship it; `=0` drops the flag, `ASYNC=1` forces async) | 1.8.0: append `--no-async-scheduling`. With dflash, async scheduling adds an output-drain engine step whose output is fetched through an event wait; without the flag the first token is sampled inside the prefill step. ABBA, only the flag differing: 300 W (2026-10-23) prefill 2K/8K/16K/32K/64K **+7.5 / +1.7 / +0.8 / +0.2 / 0 %**; 210 W (2026-10-22) prefill +7.7 / +2.8 / +1.8 / +3.0 / +1.2 %, decode c1/c2/c4 no measurable difference (the earlier single-start c2 −11.7 % did not reproduce). Scheduling-only: no numerics change. CPU test `sly/tests/test_launcher_async_sched_cpu.py` |
| `DET_COMPILE` | `0` (opt-in; `1` for A/B arms and reproducible restarts) | 1.10.0: add `"deterministic": true` to `inductor_compile_config`, so a fresh compile picks one config per reduction instead of re-timing them ([#189](https://github.com/SlyBase/vllm-sly-radiance/issues/189), [vllm#58899](https://github.com/vllm-project/vllm/issues/58899)). Removes start-to-start numeric drift between fresh compiles. Prefill within noise (1.5k-48k); decode and acceptance not yet measured. Changes the compile hash: one fresh compile. CPU test `sly/tests/test_launcher_deterministic_compile_cpu.py` |
| `RADIANCE_MXFP4_A_TILED_MIN_M` | `513` | fragment-tiled prefill GEMM from M = 513 (+11–13 % prefill); must stay above 512 and `DECODE_MAX_M` |
| `--build-arg RADIANCE_GROUPM` | `0` (build-time, default off) | M-band raster grouping of the W4A8 prefill GEMM (`16` = groups of 16 activation bands). Default build is unchanged; not measured on a GPU yet, so no recommendation |
| `RADIANCE_MXFP4_WPERM` | `1` | fragment-order weights (+3.7 % decode); set `0` with `RADIANCE_TP_PAD=3` |
| `RADIANCE_LMHEAD_INT4` | `1` | int4 vocabulary head (656 MB instead of 2.5 GB per call; GSM8K unchanged). `RADIANCE_LMHEAD_FP8=1` is the more exact fallback |
| `RADIANCE_LMHEAD_INT2` | `1` (with greedy decoding) | the 2-bit greedy vocabulary head: 2-bit coarse pass + bf16 top-16 re-rank, ≈7.5× less head traffic per decode step than int4; greedy-only (declines under sampling, byte-identical default otherwise) |
| `RADIANCE_DFLASH_BF16` | `0` (opt-in) | expands the drafter's W4A16 rows to bf16 at load: +3.22 GiB VRAM, 3.88× the hot-path weight budget; default path byte-identical |
| `RADIANCE_FUSED_NORM_QUANT` | `1` | fused norm/activation + fp8 quant in front of every W4A8 GEMM |
| `RADIANCE_KV_GROUP_SIZE` | `8` | groups the KV cache pages 2+6+1 instead of padding to the drafter's bucket: more KV tokens |
| `RADIANCE_EMBED_INT8` + `RADIANCE_EMBED_BITS` | `1` + `4` | int4 embedding table, 1.76 GiB more KV |
| `GPU_MAX_HW_QUEUES` | `2` | the ROCm default picks a random queue layout per start; 2 is consistently fastest |

### Defaults that are already on

`RADIANCE_ATTN_DECODE_TUNE`, `RADIANCE_ATTN_DRAFTER_TUNE`, `RADIANCE_LOOKUP_DRAFT` (prompt lookup),
`RADIANCE_USE_R4D` (libr4d kernels), `RADIANCE_MXFP4_DECODE_TUNE16`, and for the W4A16 GEMMs (INT4 targets and
the drafter): `RADIANCE_W4A16_TILES` (tile table), `RADIANCE_W4A16_SPLITK` (split-K / fast dequant, per-shape
table; `RADIANCE_W4A16_SPLITK_TABLE=<json>` replaces it), `RADIANCE_W4A16_TILED` (weights in 1 KB 16-row
blocks), `RADIANCE_LMHEAD_INT4_TILED` (the int4 lm_head too), `RADIANCE_W4A16_SILU` (gate_up + silu in one
GEMM), `RADIANCE_DFLASH_KV_W4` (drafter context-KV on its int4 rows instead of a 105 MB bf16 copy),
`RADIANCE_DFLASH_CONV_W4` (drafter conv projection int4). Set any to `0` only for an A/B.
`RADIANCE_GDN_BA_W4=1` (GDN in_proj_ba as int4 rows of in_proj_qkvz, one GEMM) stays off: measured slower.

### vLLM arguments and trade-offs

| Argument | Recommended | Trade-off |
|---|---|---|
| `--max-model-len` | `262144` | the model maximum; the KV pool (436k tokens) then holds one full-length request plus change. Lower it only together with the options below — it does not make a single request faster |
| `--attention-backend` | `R4D_HYBRID` | libr4d for long prefill runs, AITER for decode and short chunks. Plain `R4D` is not recommended (decode 60.9 vs 34.9 ms/step). `ROCM_AITER_UNIFIED_ATTN` is the fallback (slower long prefill) |
| `--gpu-memory-utilization` | `0.98` | with `FULL_DECODE_ONLY` the profiled activation peak is 0.65 GiB, so 0.98 leaves room; 0.98 with `FULL_AND_PIECEWISE` gives only 404,947 KV tokens |
| `--compilation-config.cudagraph_mode` | `FULL_DECODE_ONLY` | graphs only for pure decode batches (0.22 GiB instead of 1.24); prefill and mixed steps run eagerly, no measured TTFT or c8 cost. Keep the capture-size list |
| `--max-num-batched-tokens` | `4096` | prefill chunk size; the shipped default (`CHUNK`), and the chunk the measured prefill stack was taken at. Larger chunks prefill long prompts faster but take activation memory from the KV pool: measured at `--max-model-len 131072`, 4096 gives **+3.5 … +4.8 % prefill** over 2048 at 8k–64k prompts (short prompts unchanged) for ~24k fewer KV tokens (−7 %); 8192 is no faster than 4096 and costs ~68k tokens. 2048 keeps the full pool for 262k. At TP=1 the single-GPU profile (below) keeps `CHUNK=2560` unless you set it — that profile is the long-context shape and its KV pin is keyed on 2560; the measurement above used `CHUNK=4096` explicitly |
| `--max-num-seqs` | `8` | 8 × (7 + 1) = 64 verify tokens per step, inside `DECODE_MAX_M`; above 8 requests queue (conc 16 = conc 8 throughput) |
| `--kv-cache-dtype fp8` | on | twice the KV of bf16; accuracy unchanged |
| `--mamba-ssm-cache-dtype bfloat16` | on | halves the gated-delta-net state; with 0.3.5 also the faster libr4d kernels |
| `--speculative-config.num_speculative_tokens` | `7` | the drafter's maximum (block size 8) |
| `--speculative-config.draft_sample_method` | `greedy` | draft-side sampling: `greedy` takes the draft head's argmax, `probabilistic` samples it with vLLM's shared-Gumbel coupling (needs the full draft-logits head). Lossless either way — the verifier decides the emitted token — so this trades acceptance rate against per-step cost, not accuracy. `serve.sh` / `serve-mxfp4.sh` take it as `DRAFT_SAMPLE` (`DRAFT_SAMPLE=probabilistic ./serve-mxfp4.sh`); the like-for-like A/B against `probabilistic` is still open ([BENCHMARKS.md](docs/BENCHMARKS.md), step 2, arm G) |
| `--chat-template /opt/qwen-fixed.jinja` | on | froggeric's fixed Qwen template (medium reasoning default, no empty think blocks, tool-call fixes) |
| `--override-generation-config` | Qwen's thinking values | temperature 1.0 / top_p 0.95 / top_k 20; clients that turn thinking off should send 0.7 / 0.8 / 20 / presence 1.5 |

**Less context, more prefill:** if your requests stay well below 262k tokens, set
`--max-model-len 131072` (the prefill chunk is already `4096`): +3.5 … +4.8 % prefill on 8k–64k prompts, and the
KV pool (~326k tokens) still holds two full 131k requests. Measured with image 0.4.0 on 2026-09-25 (300 W, prefill sweep; KV pools are 0.4.0 values, 1.0.0 holds ~11 % more):

| `--max-model-len` / `--max-num-batched-tokens` | Prefill 2k / 8k / 16k / 32k / 64k tok/s | KV pool |
|---|---|---|
| 262144 / 2048 | 3,363 / 3,326 / 3,196 / 2,926 / 2,512 | 384,316 |
| 131072 / 2048 | 3,379 / 3,344 / 3,212 / 2,936 / 2,520 | ~350k |
| 131072 / **4096** | 3,375 / **3,444 / 3,351 / 3,057 / 2,613** | ~326k |
| 131072 / 8192 | 3,367 / 3,486 / 3,290 / 3,022 / 2,579 | ~282k |

Decode and concurrency are unchanged within noise. (These are prefill-only sweeps, which read ~5 % higher
than the prefill phase of the full run; compare within one table.)

### KV-cache offload (opt-in, off by default)

A second-tier prefix cache behind the GPU prefix cache for long-context / agent-style sessions, where
every turn re-sends the whole conversation and the prefix outgrows the GPU pool. Off by default: with
`KVCACHE` unset (or `off`) the serve command is byte-identical to the unmodified script — nothing is
mounted, patched or passed to the engine.

| Variable | Recommended | What it does |
|---|---|---|
| `KVCACHE` | `off` | `off` (default) = stock behaviour; `ram` = GPU → RAM tier in `/dev/shm` + the 6 behavioural patches; `disk` = GPU → RAM → disk + the full 15-patch instrumented set |
| `KVCACHE_RAM_GIB` | `16` | size of the RAM tier (GiB); needs `/dev/shm` big enough |
| `KVCACHE_DISK` | — | host path of a dedicated filesystem for the disk tier (`KVCACHE=disk` only); the disk tier also wants `kvcache-reap.sh` (see `kv-cache/README.md`) to keep the fs below 70 % |
| `STARTUP_CACHE` | `1` | **Recommended: on.** `serve-mxfp4.sh` keys its cache dir on image ID + GPU arch, persists the comgr / tvm-ffi / tilelang JIT caches across `--rm` boots, compiles `radiance_mxfp4_fp8.so` once per key and prunes stale cache trees when the disk passes 65 %. A miss is always safe (the slow path rebuilds); saves ~68 s per boot. See `startup-cache/README.md` |
| `KVCACHE_EXACT` | `0` | `1` (needs `KVCACHE=ram`/`disk`) = bit-exact cached resumes via canonical prefill chunks + checkpoint retention; six extra patches, see `kv-cache/README.md` → "Exact resumes". **Recommended with `KVCACHE=ram`/`disk` for multi-turn / long-context agents** (retention raises follow-up turns served from cache 56→77 % / 69→85 %; a cached turn becomes bit-identical to a cold one for greedy eval/A-B). **Keep `0` for short prompts** (< one 3520-token grid chunk): they are then never prefix-cached |

### 4-bit KV cache (opt-in, off by default): TetraAttention

`KVCACHE_DTYPE=kv4` stores the 16 full-attention layers' K/V as Hadamard-rotated 4-bit codes with one
e4m3 scale per token (272 bytes per head and token instead of 512) and runs their attention on
TetraAttention (backend `TETRA`), a HIP kernel that feeds the stored nibbles straight into the card's
int4 / int8 matrix instructions (`sly/kv4/README.md`). The DFlash drafter's head-128 window
attention runs on it too (its 136-byte cells are vLLM's int4 cell size, so the pages line up).

| Variable | Recommended | What it does |
|---|---|---|
| `KVCACHE_DTYPE` | `fp8` | `fp8` (default) = stock; `kv4` = TetraAttention, 4-bit K/V for the attention layers, an opt-in trade: **~1.75x the KV pool** (a full 262k context in 4.25 GiB of KV instead of 8) for slightly less precision (KL against a bf16 KV cache 0.0126 vs fp8's 0.0123, perplexity +0.54 % vs +0.06 %, top-1 agreement 95.3 vs 95.6 %; GSM8K-200 0.80 vs 0.79; measured on 1.5.1 at 262k). **1.8.1 at 65k context, 4,096 chunk, same window:** KV pool 528,722 vs 342,950 tokens (+54 %), decode 154.8 vs 158.1 tok/s, prefill about equal below 16k and **11.7 % slower at 64k** (the TetraAttention prefill kernel is 1.8x the fp8 kernel, see CHANGELOG 1.8.1). Use it when capacity matters. The attention block is 1664 tokens (fp8: 880), so a turn recomputes up to 1,663 tokens on a prefix hit. Knobs `RADIANCE_KV4H_*` (sly/kv4/README.md). An offload tier written by an fp8 serve must not be reused |

From vLLM 0.32.0 (image 1.7.0) there is also a **native** ROCm 4-bit KV cache, `--kv-cache-dtype
ultraquant_4bit` (vllm#57057): Hadamard rotation plus one power-of-two scale per 32 channels instead of
one e4m3 scale per token, with the attention kernel shipped by vLLM/AMD rather than this repo's HIP
kernel (AMD documented it on MI355X). It is incompatible with `FlashAttention >= 3` (0.32 downgrades
the KV stack's own flash-attn setting, a logged warning). Measured live on the R9700 (0.87 edt,
200k context, single request): **DFlash drafting is incompatible on this geometry** — with the
drafter on, decode degenerates into token loops from the first verify step (fp8 with the drafter and
UltraQuant without the drafter are both correct against the same probe; the difference is the
multi-token verify path inside 0.32's UltraQuant KV geometry, file upstream). So until upstream or a
later version fixes it, serve `ultraquant_4bit` **without** speculative decoding — measured live:
**698,181 KV tokens / 3.49x concurrency at 200k** vs fp8's 310,038 / 1.55x in the same window
(2.25x the pool), correct answers on both probe temperatures, and a 38k-token prefill at
~1.1 ms/token through the chunked continuation path (Triton fallback; FlyDSL is ineligible at
head_size 256). Known serve landmines this repo already patches or handles (`sly/patch_ultraquant_gqa.py`
via the loop fix and the shared-workspace hunks, documented in sly/README.md): the 0.32
continuation-prefill fallback killed the engine on any >2048-token prompt (GQA shape error, then a
746 MiB score materialization in torch's math-SDPA), and upstream also cached 512-MiB-per-layer
worst-case concat buffers (16 GiB across 16 layers at 262k).

The offload patches, `kvwatch.py` and the `turnbench`/`tierbench` benches live in `kv-cache/` and ride
the repo's existing `/patches` mount; `serve-mxfp4.sh` applies them at container start (not baked).
The disk tier additionally needs its own filesystem plus a host systemd reaper (your step, not the
image's). Measured (one R9700, TP=1, `KVCACHE_RAM_GIB=16`, 110k-token agent sessions): a ~81k-token
request drops from **38.8 s** (no cache) to **~13 s** (RAM tier) — see `kv-cache/README.md`. This
reduces re-prefill latency for long sessions; it does **not** raise `--max-model-len`.

### Cactus acceptance transform (on by default, delta 0.5)

Speculative decoding accepts a drafted token with probability min(1, q/p). Cactus
([arxiv 2604.04987](https://arxiv.org/abs/2604.04987)) raises the target probability q to
h = clamp(q + sqrt(2*delta*q*(1-q)), 0, 1) in that test. More drafted tokens are accepted per step, and the
divergence from the target distribution stays bounded by the budget `delta`. Only sampled requests
(temperature > 0) change; greedy requests, synthetic-rate and block-verification modes are untouched.
Patch: `sly/patch_cactus.py`, CPU-tested in `sly/tests/test_cactus_cpu.py`.

| Variable | Recommended | What it does |
|---|---|---|
| `RADIANCE_CACTUS_DELTA` | `0.5` (image default) | Divergence budget. `0` = bit-exact stock acceptance test (the exp/log round-trip is guarded, so the comparison is identical to upstream). Larger = more accepted tokens, more drift from the target distribution. Read once at start; it is a per-process kernel constant, so changing it needs a restart |

Measured (one R9700, 300 W, production arguments, BetterBench 0.4.0 default config, 2026-10-17): against the
1.0.0 column of the 1.5.1 reference run (see Performance), delta 0.5 gives **+3 to +20 % tokens per update** (prose +20, reasoning +14,
summarization +14, code +13, json +4, file_edit +3), **149.6 vs 135.8 t/s weighted (+10.2 %)** and +9 to +12 %
aggregate at concurrency 2 to 16. The step time is unchanged (34.6 vs 34.4 ms), so the gain is acceptance, not
speed. With the 4-bit KV cache the weighted figure is 145.2 t/s (+6.9 %; KV4 costs about 3 % decode against fp8
and gives 1.75x the KV pool). An earlier same-image A/B at 210 W (delta 0 vs 0.5, fp8) showed +0.0835 acceptance
rate and +0.585 tokens per step.

What this does not show yet: the 300 W comparison above crosses image versions (1.0.0 vs 1.5.1); a same-image
delta 0 control at 300 W and a sampled accuracy check (GSM8K runs greedy, which Cactus does not touch) are still open. **Recommendation:**
keep 0.5 for chat, coding and agent workloads, where more accepted tokens is the goal. Set
`RADIANCE_CACTUS_DELTA=0` when you need output that is statistically identical to the target model (evals,
distribution-sensitive sampling, reproducing published numbers). Do not raise it above 0.5 without measuring
quality on your own prompts.

### Nucleus Speculative Decoding (opt-in, off by default)

NSD ([arxiv 2610.07822](https://arxiv.org/abs/2610.07822)) is a second, **opt-in** acceptance rule for
sampled requests. A draft token x is accepted when the standard test passes **or** x lies in the target
nucleus: the probability mass of the tokens with a strictly higher target logit than x is <= `theta`.
The residual distribution on a rejection and the bonus token are untouched, so the rule only ever accepts
*more* — at the price of the paper's single-step bound on the deviation from the target distribution,
which is why it is not the default. Patch: `sly/patch_nsd.py`, CPU-tested in `sly/tests/test_spec_knobs_cpu.py`
(wiring) and `sly/tests/test_nsd_cpu.py` (rule, kernel in the Triton interpreter). Greedy rows, synthetic
rates and block verification are not touched.

| Variable | Recommended | What it does |
|---|---|---|
| `RADIANCE_NSD` | `0` (default) | `1` enables the nucleus rule. With it off nothing extra is launched and the NSD branch is compiled out (same arithmetic as stock). Read once at start (per-process constant) |
| `RADIANCE_NSD_THETA` | `0.95` | nucleus mass in `[0, 1)`. Smaller = a tighter nucleus = fewer extra acceptances and less drift |

NSD replaces the acceptance rule that Cactus rewrites, and the paper's bound does not cover the union of
both, so the two are mutually exclusive: in this image **`RADIANCE_NSD=1` needs `RADIANCE_CACTUS_DELTA=0`**
(otherwise the engine refuses to start with a `RuntimeError` — Cactus is on by default at 0.5, and the
check reads that default, not an unset variable). Launch:

```bash
RADIANCE_NSD=1 RADIANCE_CACTUS_DELTA=0 ./serve-mxfp4.sh
```

A/B for this branch (1 R9700, production arguments, concurrency 1; the run is not yet in
`radiance-benchmarks`, so treat the figures as the branch owner's, not as a citable window): **+26 %
weighted decode against the stock acceptance rule** and **~+5 % against the Cactus-0.5 production
config**, the latter inside the window noise. That is not enough to make it the default: a numerics-changing default needs the
quality gate, i.e. GSM8K-200 (baseline 0.855 ± 0.025) and a perplexity comparison against the Cactus-0.5
serve, plus an unchanged step time and KV pool at concurrency 1 and 8. Until that gate is run, keep
`RADIANCE_NSD=0` for chat/production and enable it only in a measured arm.

### Other checkpoints and GPUs

| Switch | For |
|---|---|
| `RADIANCE_NVFP4_MXFP4=1` + `--quantization compressed-tensors` | NVFP4 checkpoints (e.g. `unsloth/Qwen3.8-27B-NVFP4`), requantized to MXFP4 at load — same speed as Quark |
| `RADIANCE_PAROQUANT=1`, `RADIANCE_AUTOROUND=1`, `RADIANCE_ESCHA=1` (no `--quantization`) | ParoQuant, AutoRound, escha checkpoints |
| `QUANT=mxfp6 ./serve.sh` / `./setup.sh --mxfp6` | ParoQuant MXFP6 (W6A8); needs TP>=2 (upstream-validated), one R9700 is TP=1 |
| `--tensor-parallel-size 2/3/4/8`, `RADIANCE_TP_PAD=3`, `RADIANCE_AR_*` | several cards — see [docs/MULTI-GPU.md](docs/MULTI-GPU.md) |
| `RADIANCE_MINICPMV4_7=1` + `--attention-backend ROCM_AITER_UNIFIED_ATTN` | MiniCPM-V 4.7 35B-A3B with its DSpark drafter: 169.4 tok/s decode at 210 W (Qwen3.8 production 132.4) — full arguments in [docs/MINICPM-V47.md](docs/MINICPM-V47.md) |

## Repository and contributing

| Path | What |
|---|---|
| `Dockerfile`, `constraints.txt`, `VERSION` | the build (from-source stack, patch loop, kernels) and its pins |
| `sly/` | everything this fork adds: patches, kernels, attention tunes, benches ([reference](sly/README.md)) |
| `patch_*.py`, `radiance_*.py`, `paroquant/`, `escha/`, launch scripts in the root | the upstream repositories' files, synced daily; the Dockerfile loop and `ci/unused_patches.txt` say which are used |
| `ci/` | CI scripts, the acceptance gate, the release tagging |
| `docs/` | [technical deep dives](docs/TECHNICAL.md), [benchmarks](docs/BENCHMARKS.md), [not adopted](docs/NOT-ADOPTED.md), [multi-GPU](docs/MULTI-GPU.md), [Windows / WSL2](docs/WINDOWS-WSL2.md), [development & releases](docs/DEVELOPMENT.md), [automated maintenance](docs/MAINTENANCE.md) |
| `CHANGELOG.md` | per-version changes = release notes; contributor rules in [AGENTS.md](AGENTS.md) |

## Credits

This image stands on other people's work:

- **[StillDeadcode](https://codeberg.org/StillDeadcode)** — vllm-radiance and libr4d: the RDNA4 build, the
  gfx1201 kernels and fixes everything here runs on.
- **Radiance engine fork** ([StillDeadcode/radiance](https://codeberg.org/StillDeadcode/radiance), our fork
  slydlake/radiance) — the `adaptive_k` verify-width policy behind `RADIANCE_ADAPTIVE_WIDTH` and the M-band findings
  (wide decode tile) behind `RADIANCE_MXFP4_WIDE_MAX_M`, ported to vLLM's graph model.
- **[extfs](https://github.com/extfs)** (Tom Hillbrunner) — TetraAttention (`KVCACHE_DTYPE=kv4`, `sly/kv4/`): the 4-bit KV cache with the int4/int8 WMMA attention kernel, its tests and accuracy harness (PR #135, taken wholesale; the image wiring, a 1.4.0 rebase and the re-measurement are ours).
- **[EIT-NLP/Nucleus-Speculative-Decoding](https://github.com/EIT-NLP/Nucleus-Speculative-Decoding)** — the NSD acceptance rule (`RADIANCE_NSD`, arxiv 2610.07822): the rule and the reference `clean_nsd` membership test; the Triton kernel, the vLLM V2 rejection-sampler wiring and the tests are ours (code re-written, not copied).
- **[libr4d](https://codeberg.org/StillDeadcode/libr4d)** — the paged prefill attention kernel behind `R4D_HYBRID`.
- **[ggz14](https://codeberg.org/ggz14)** — radiance-vllm-mxfp4: MXFP4 on RDNA4 and the W4A8 kernel, the
  libr4d extras, TP=3 padding, and the NVFP4, ParoQuant and AutoRound paths.
- **[vLLM](https://github.com/vllm-project/vllm)**, **[AITER](https://github.com/ROCm/aiter)**, the
  **[DFlash](https://github.com/vllm-project/vllm/pull/52816)** authors; `sly/gdn/radiance_gdn_decode.hip`
  ports vLLM's fused GDN decode kernel (Apache-2.0).
- **[syvai](https://huggingface.co/syvai)** for the DFlash2 W4A16 drafter, **AMD** for the Quark MXFP4
  checkpoint, **froggeric** for the fixed Qwen chat template.
- **[turboderp/exllamav3](https://github.com/turboderp-org/exllamav3)** (MIT) — `escha/` derives from it;
  **[z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-PARO)** for the ParoQuant format.

## License

Same license as upstream vllm-radiance; the files taken from vLLM, AITER and the projects listed under
[Credits](#credits) keep their own licenses.
