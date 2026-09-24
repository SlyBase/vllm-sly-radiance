# vllm-sly-radiance

**The fastest way to run Qwen3.8-27B on one AMD Radeon AI PRO R9700.** A vLLM image for gfx1201 (RDNA4)
with hand-written MXFP4 kernels, DFlash2 speculative decoding and the full 262k context on a single
32 GB card: **~{{DECODE300}} tok/s single-stream decode, ~{{PP2K300}} tok/s prefill, {{C8_300}} tok/s at 8
concurrent requests**, with every number measured and reproducible.

```bash
docker pull ghcr.io/slybase/vllm-sly-radiance:0.4.0-rocm10.0
```

## Tech stack

| Layer | Version |
|---|---|
| GPU | AMD Radeon AI PRO R9700, 32 GB, gfx1201 (RDNA4) — one card; TP 2/3/4/8 paths included but untested here |
| ROCm | 10.0 (`rocm/dev-ubuntu-24.04:10.0.0-full`, pruned to gfx1201) |
| PyTorch / Triton / torchvision | 2.14.0 / 3.8.0 / 0.29.0, built from source for gfx1201 |
| vLLM | 0.30.0 (V1 engine, V2 model runner), built from source |
| AITER / transformers | 0.1.22.post1 / 5.17.0 |
| Kernels | [libr4d](https://codeberg.org/StillDeadcode/libr4d) (attention, gated delta net, all-reduce) + this repo's MXFP4 W4A8 GEMM, fused norm/quant, attention tunes |
| Model | [`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) (Quark MXFP4, gated-delta-net hybrid) |
| Drafter | [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16), DFlash2 k = 7 + prompt lookup |
| Also serves | NVFP4, compressed-tensors INT4, ParoQuant, AutoRound, escha checkpoints ([comparison](docs/BENCHMARKS.md#other-checkpoints-on-one-r9700-036-2026-09-24)) |

## Performance

Reference run of image **0.4.0** on {{REFDATE}}, production arguments from the [quickstart](#quickstart)
(`--max-model-len 262144`, fp8 KV, bf16 SSM state, DFlash2 k = 7, KV pool **384,316 tokens**),
BetterBench 0.4.0 default config: single-stream decode
3 warmup + 20 passes per category, prefill sweep with unique prompts, concurrency 1–16 × 48 requests,
sampling temperature 0.7 / top_p 0.95 / top_k 20. Two power settings, no other traffic:
**300 W** (firmware fan curve) and **210 W** (fan capped at 2,800 rpm, what this card runs 24/7).

| | 300 W | 210 W |
|---|---|---|
| **Decode, single stream** (weighted over 8 task categories) | **{{DECODE300}} tok/s** | {{DECODE210}} tok/s |
| step gap (one target forward + draft) | {{GAP300}} ms | {{GAP210}} ms |
| **Prefill** 1.5k / 6k / 24k / 47k prompt tokens | **{{PP300}}** tok/s | {{PP210}} tok/s |
| time to first token, 1.5k / 47k tokens | {{TTFT300}} | {{TTFT210}} |
| **Concurrency** 1 / 2 / 4 / 8 / 16, aggregate | **{{CONC300}}** tok/s | {{CONC210}} tok/s |
| GSM8K (200, cot zero-shot, greedy) | {{GSM}} | |
| board power / junction / fan (avg) | {{CARD300}} | {{CARD210}} |

What that means: code, JSON and edits decode at 150–180 tok/s (the drafter lands 5–6 tokens per step),
free prose at ~85; long contexts stay fast (99k tokens of context cost ~7 ms per step instead of 83).
Per-category tables, the measurement method and every earlier release are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). Decode differences below ~5 % between single runs are noise;
see there for why.

## Quickstart

Requirements: one R9700 (or another gfx1201 card with 32 GB), ROCm-capable kernel driver
(`/dev/kfd`, `/dev/dri`), Docker, ~20 GB of disk for the model + drafter, and enough free host RAM for the model load (on the
maintainer's 32 GB host the other VMs are paused while the model loads).

```bash
# 1. model + drafter into the Hugging Face cache
hf download amd/Qwen3.8-27B-Quark-AWQ-MXFP4
hf download syvai/Qwen3.8-27B-DFlash2-W4A16

# 2. serve (OpenAI-compatible API on :8000)
docker run -d --name vllm --restart unless-stopped \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined --ipc=host -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm -v triton-cache:/root/.triton -v aiter-cache:/root/.aiter \
  -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=2 \
  -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
  -e RADIANCE_MXFP4_DECODE_MAX_M=128 -e RADIANCE_MXFP4_A_TILED_MIN_M=513 -e RADIANCE_MXFP4_WPERM=1 \
  -e RADIANCE_LMHEAD_INT4=1 -e RADIANCE_FUSED_NORM_QUANT=1 \
  -e RADIANCE_KV_GROUP_SIZE=8 -e RADIANCE_EMBED_INT8=1 -e RADIANCE_EMBED_BITS=4 \
  ghcr.io/slybase/vllm-sly-radiance:0.4.0-rocm10.0 \
  --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --quantization quark \
  --max-model-len 262144 --gpu-memory-utilization 0.96 \
  --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config.method dflash \
  --speculative-config.model syvai/Qwen3.8-27B-DFlash2-W4A16 \
  --speculative-config.num_speculative_tokens 7 \
  --speculative-config.draft_sample_method probabilistic \
  --speculative-config.attention_backend TRITON_ATTN \
  --attention-backend ROCM_AITER_UNIFIED_ATTN \
  --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --compilation-config.cudagraph_mode FULL_AND_PIECEWISE \
  --compilation-config.cudagraph_capture_sizes '[8,16,24,32,40,48,56,64]' \
  --enable-prefix-caching --skip-mm-profiling --enable-mm-embeds \
  --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --chat-template /opt/qwen-fixed.jinja \
  --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
  --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}'

# 3. test
curl -s localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"amd/Qwen3.8-27B-Quark-AWQ-MXFP4","messages":[{"role":"user","content":"Hello!"}]}'
```

- **First start** compiles kernels and CUDA graphs (~10 min) and reports a ~2 GiB smaller KV pool;
  **restart once** — the warm start takes 4–6 min and gets the full 384k-token KV pool. Keep the cache
  volumes.
- **Power:** decode barely depends on the power cap (210 W costs ~6 % decode and ~25 % prefill against
  300 W, see the table above).
- **iGPU:** `HIP_VISIBLE_DEVICES=0` keeps ROCm off an integrated GPU next to the R9700.
- Other checkpoints: [NVFP4, INT4, ParoQuant](docs/BENCHMARKS.md#other-checkpoints-on-one-r9700-036-2026-09-24);
  several GPUs: [docs/MULTI-GPU.md](docs/MULTI-GPU.md).

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
| `RADIANCE_MXFP4_A_TILED_MIN_M` | `513` | fragment-tiled prefill GEMM from M = 513 (+11–13 % prefill); must stay above 512 and `DECODE_MAX_M` |
| `RADIANCE_MXFP4_WPERM` | `1` | fragment-order weights (+3.7 % decode); set `0` with `RADIANCE_TP_PAD=3` |
| `RADIANCE_LMHEAD_INT4` | `1` | int4 vocabulary head (656 MB instead of 2.5 GB per call; GSM8K unchanged). `RADIANCE_LMHEAD_FP8=1` is the more exact fallback |
| `RADIANCE_FUSED_NORM_QUANT` | `1` | fused norm/activation + fp8 quant in front of every W4A8 GEMM |
| `RADIANCE_KV_GROUP_SIZE` | `8` | groups the KV cache pages 2+6+1 instead of padding to the drafter's bucket: more KV tokens |
| `RADIANCE_EMBED_INT8` + `RADIANCE_EMBED_BITS` | `1` + `4` | int4 embedding table, 1.76 GiB more KV |
| `GPU_MAX_HW_QUEUES` | `2` | the ROCm default picks a random queue layout per start; 2 is consistently fastest |

### Defaults that are already on

`RADIANCE_ATTN_DECODE_TUNE`, `RADIANCE_ATTN_DRAFTER_TUNE`, `RADIANCE_LOOKUP_DRAFT` (prompt lookup),
`RADIANCE_USE_R4D` (libr4d kernels), `RADIANCE_W4A16_TILES`, `RADIANCE_MXFP4_DECODE_TUNE16`. Set any to
`0` only for an A/B.

### vLLM arguments and trade-offs

| Argument | Recommended | Trade-off |
|---|---|---|
| `--max-model-len` | `262144` | the model maximum; the KV pool (384k tokens) then holds one full-length request plus change. Lower it only together with the options below — it does not make a single request faster |
| `--max-num-batched-tokens` | `2048` | prefill chunk size. {{CHUNKNOTE}} |
| `--max-num-seqs` | `8` | 8 × (7 + 1) = 64 verify tokens per step, inside `DECODE_MAX_M`; above 8 requests queue (conc 16 = conc 8 throughput) |
| `--kv-cache-dtype fp8` | on | twice the KV of bf16; accuracy unchanged |
| `--mamba-ssm-cache-dtype bfloat16` | on | halves the gated-delta-net state; with 0.3.5 also the faster libr4d kernels |
| `--speculative-config.num_speculative_tokens` | `7` | the drafter's maximum (block size 8) |
| `--chat-template /opt/qwen-fixed.jinja` | on | froggeric's fixed Qwen template (medium reasoning default, no empty think blocks, tool-call fixes) |
| `--override-generation-config` | Qwen's thinking values | temperature 1.0 / top_p 0.95 / top_k 20; clients that turn thinking off should send 0.7 / 0.8 / 20 / presence 1.5 |

**Less context, more prefill:** {{LESSCTX}}

### Other checkpoints and GPUs

| Switch | For |
|---|---|
| `RADIANCE_NVFP4_MXFP4=1` + `--quantization compressed-tensors` | NVFP4 checkpoints (e.g. `unsloth/Qwen3.8-27B-NVFP4`), requantized to MXFP4 at load — same speed as Quark |
| `RADIANCE_PAROQUANT=1`, `RADIANCE_AUTOROUND=1`, `RADIANCE_ESCHA=1` (no `--quantization`) | ParoQuant, AutoRound, escha checkpoints |
| `--tensor-parallel-size 2/3/4/8`, `RADIANCE_TP_PAD=3`, `RADIANCE_AR_*` | several cards — see [docs/MULTI-GPU.md](docs/MULTI-GPU.md) |

## Repository

| Path | What |
|---|---|
| `Dockerfile`, `constraints.txt`, `VERSION` | the build (from-source stack, patch loop, kernels) and its pins |
| `sly/` | everything this fork adds: patches, kernels, attention tunes, benches ([reference](sly/README.md)) |
| `patch_*.py`, `radiance_*.py`, `paroquant/`, `escha/`, launch scripts in the root | the upstream repositories' files, synced daily; the Dockerfile loop and `ci/unused_patches.txt` say which are used |
| `ci/` | CI scripts, the acceptance gate, the release tagging |
| `docs/` | [technical deep dives](docs/TECHNICAL.md), [benchmarks](docs/BENCHMARKS.md), [not adopted](docs/NOT-ADOPTED.md), [multi-GPU](docs/MULTI-GPU.md), [development & releases](docs/DEVELOPMENT.md) |
| `CHANGELOG.md` | per-version changes = release notes; contributor rules in [AGENTS.md](AGENTS.md) |

## Credits

This image stands on other people's work:

- **[StillDeadcode](https://codeberg.org/StillDeadcode)** — vllm-radiance and libr4d: the RDNA4 build, the
  gfx1201 kernels and fixes everything here runs on.
- **[ggz14](https://codeberg.org/ggz14)** — radiance-vllm-mxfp4: MXFP4 on RDNA4 and the W4A8 kernel, the
  libr4d extras, TP=3 padding, and the NVFP4, ParoQuant and AutoRound paths.
- **[vLLM](https://github.com/vllm-project/vllm)**, **[AITER](https://github.com/ROCm/aiter)**, the
  **[DFlash](https://github.com/vllm-project/vllm/pull/52816)** authors; `sly/gdn/radiance_gdn_decode.hip`
  ports vLLM's fused GDN decode kernel (Apache-2.0).
- **[syvai](https://huggingface.co/syvai)** for the DFlash2 W4A16 drafter, **AMD** for the Quark MXFP4
  checkpoint, **froggeric** for the fixed Qwen chat template.
- **[turboderp/exllamav3](https://github.com/turboderp-org/exllamav3)** (MIT) — `escha/` derives from it;
  **[z-lab](https://huggingface.co/z-lab/Qwen3.8-27B-PARO)** for the ParoQuant format.

License: same as upstream vllm-radiance (see `LICENSE`).
