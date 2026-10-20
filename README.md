# vllm-sly-radiance

**The fastest way to run Qwen3.8-27B on one AMD Radeon AI PRO R9700.** A vLLM image for gfx1201 (RDNA4)
with hand-written MXFP4 kernels, DFlash2 speculative decoding and the full 262k context on a single
32 GB card: **~150 tok/s single-stream decode, ~3,000 tok/s prefill (2,430 at 47k), 503 tok/s at 8
concurrent requests, 439k-token KV pool** (1.5.1 reference run at 300 W, fp8 KV cache, see [Performance](#performance)), with every number measured and reproducible.

```bash
docker pull ghcr.io/slybase/vllm-sly-radiance:1.1.0-rocm10.1
```

## Tech stack

| Layer | Version |
|---|---|
| GPU | AMD Radeon AI PRO R9700, 32 GB, gfx1201 (RDNA4) — one card; TP 2/3/4/8 paths included but untested here |
| ROCm | 10.1 (`rocm/dev-ubuntu-24.04:10.1.0-full`, HIP 7.16, pruned to gfx1201) |
| PyTorch / Triton / torchvision | 2.13.0 / 3.8.0 (`git669b31ac`) / 0.28.0, all AMD's `+rocm10.1.0` wheels (the trio vLLM 0.31's ROCm base builds) |
| vLLM | 0.31.0 (V1 engine, V2 model runner), built from source |
| AITER / transformers | 0.1.23 / 5.17.0 |
| Kernels | [libr4d](https://codeberg.org/StillDeadcode/libr4d) (prefill attention, gated delta net, all-reduce) + this repo's MXFP4 W4A8 GEMM, fused norm/quant, attention tunes |
| Model | [`amd/Qwen3.8-27B-Quark-AWQ-MXFP4`](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4) (Quark MXFP4, gated-delta-net hybrid) |
| Drafter | [`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16), DFlash2 k = 7 + prompt lookup |
| Also serves | NVFP4, compressed-tensors INT4, ParoQuant, AutoRound, escha checkpoints ([comparison](docs/BENCHMARKS.md#other-checkpoints-on-one-r9700-036-2026-09-24)) |

## Performance

**Reference run of image 1.5.1** on 2026-10-17, 300 W (firmware fan curve), production arguments, BetterBench
0.4.0 default config, Cactus delta 0.5 (the image default). Two arms: the fp8 KV cache with `R4D_HYBRID`, and the
4-bit KV cache (`KVCACHE_DTYPE=kv4`, backend `TETRA`). The 1.0.0 column is the 1.0.0 reference run of 2026-10-07
(300 W, same method, a different GPU window and day; details in [docs/BENCHMARKS.md](docs/BENCHMARKS.md)).
Decode, prefill and concurrency 2-16 were measured on the pre-release build `1.5.1-cactus-pr153` (same kernels, delta
set by environment variable). Concurrency 1 and GSM8K come from a second window on the release image
`1.5.1-rocm10.1`, after a GSM8K warm-up.

| | **1.5.1 fp8 KV** | **1.5.1 KV4** | 1.0.0 | Δ fp8 | Δ KV4 |
|---|---|---|---|---|---|
| **Decode, single stream** (weighted) | **149.6 tok/s** | 145.2 tok/s | 135.8 tok/s | **+10.2 %** | +6.9 % |
| update gap p50 / p99 | 34.6 / 35.7 ms | 34.8 / 35.9 ms | 34.4 / 35.2 ms | ± 0 | ± 0 |
| tokens per update: code / reasoning / prose | 5.71 / 4.08 / 3.45 | 5.56 / 4.22 / 3.26 | 5.07 / 3.57 / 2.88 | +13 / +14 / +20 % | +10 / +18 / +13 % |
| tokens per update: json / file_edit / summarization | 5.83 / 6.16 / 5.45 | 5.71 / 6.07 / 5.17 | 5.60 / 5.98 / 4.79 | +4 / +3 / +14 % | +2 / +2 / +8 % |
| **Prefill** 1.5k / 6k / 12k / 24k / 47k prompt tokens | 3,024 / 3,032 / 2,994 / 2,766 / 2,430 | 3,065 / 2,977 / 2,913 / 2,757 / 2,440 | 3,132 / 3,136 / 3,105 / 2,949 / 2,635 | −3 / −3 / −4 / −6 / **−8 %** | −2 / −5 / −6 / −7 / **−7 %** |
| time to first token, 47k tokens | 19.4 s | 19.3 s | 17.9 s | +8 % | +8 % |
| **Concurrency** 1 / 2 / 4 / 8 / 16, aggregate | **135**\* / 237 / 373 / **503 / 507** tok/s | 134\* / 243 / 373 / 483 / 506 | 120 / 213 / 338 / 461 / 466 | +12 / +12 / +10 / **+9 / +9 %** | +11 / +14 / +10 / +5 / +8 % |
| KV pool (262k context) | 439,031 tokens | **767,557 tokens** | 436,906 | +0.5 % | **+76 %** |
| GSM8K 200 (greedy)\*\* | 0.845 | 0.855 | 0.84 | tie | tie |

\* Concurrency 1, warm engine, release image: fp8 136.2 and 133.7 tok/s in two runs (mean 135, TTFT p50 88 ms,
p99 121 ms); KV4 133.7 from the first window. The first fp8 run in that window gave 115 tok/s with a TTFT p99 of
2.7 s because its first requests ran cold, so it was discarded.

\*\* lm-eval `gsm8k_cot_zeroshot`, 200 questions, `temperature=0`, 16 concurrent requests, flexible-extract,
standard error 0.026 (so all three values tie). Run at 300 W on the release image; the 200 requests took 1:31
(fp8) and 1:29 (KV4) minutes. The 1.0.0 value was taken at 210 W. Cactus leaves greedy requests untouched, so this
figure checks the engine and the KV format, not delta 0.5.

What that means: the gain over 1.0.0 is acceptance (Cactus), not speed. The step time is unchanged, tokens per
update rise by 2-20 %, and decode and concurrency 2-16 follow. KV4 gives up about 3 % decode against fp8 and
gives 1.75x the KV pool. Prefill is 3-8 % lower than in the 1.0.0 window; that run was taken in another window
and this release did not change prefill, so treat it as window spread until a same-window A/B says otherwise.
The 1.5.1 numbers carry no same-image delta 0 control at 300 W (see the Cactus section).

The older reference run of image **0.4.0** (Quark checkpoint, 300 W and 210 W) is kept below for comparison.

Reference run of image **0.4.0** on 2026-09-25, production arguments from the [quickstart](#quickstart)
(`--max-model-len 262144`, fp8 KV, bf16 SSM state, DFlash2 k = 7, KV pool **384,316 tokens**),
BetterBench 0.4.0 default config: single-stream decode
3 warmup + 20 passes per category, prefill sweep with unique prompts, concurrency 1–16 × 48 requests,
sampling temperature 0.7 / top_p 0.95 / top_k 20. Two power settings, no other traffic:
**300 W** (firmware fan curve) and **210 W** (fan capped at 2,800 rpm, what this card runs 24/7).

| | 300 W | 210 W |
|---|---|---|
| **Decode, single stream** (weighted over 8 task categories) | **133.2 tok/s** | 124.5 tok/s |
| step gap (one target forward + draft) | 34.8 ms | 37.0 ms |
| **Prefill** 1.5k / 6k / 24k / 47k prompt tokens | **3,166 / 3,161 / 2,903 / 2,512** tok/s | 2,527 / 2,498 / 2,311 / 2,027 tok/s |
| time to first token, 1.5k / 47k tokens | 0.48 s / 18.7 s | 0.60 s / 23.2 s |
| **Concurrency** 1 / 2 / 4 / 8 / 16, aggregate | **121 / 214 / 332 / 410 / 406** tok/s | 111 / 197 / 312 / 362 / 369 tok/s |
| GSM8K (200, cot zero-shot, greedy) | 0.855 ± 0.025 | |
| board power / junction / fan (avg) | 296 W / 98 °C / 3,710 rpm | 210 W / 93 °C / 2,321 rpm |

What that means: code, JSON, math and edits decode at 140–175 tok/s (the drafter lands 5–6 tokens per step),
free prose at ~85; long contexts stay fast (99k tokens of context cost ~7 ms per step instead of 83).
Per-category tables, the measurement method and every earlier release are in
[docs/BENCHMARKS.md](docs/BENCHMARKS.md). Decode differences below ~5 % between single runs are noise;
see there for why.

## Quickstart

Requirements: one R9700 (or another gfx1201 card with 32 GB), ROCm-capable kernel driver
(`/dev/kfd`, `/dev/dri`), Docker, ~20 GB of disk for the model + drafter, and enough free host RAM for the model load. Since **1.2** the
MXFP4 checkpoint is sharded by default, so vLLM loads it shard-by-shard and the peak host RAM is the largest shard (5 GiB)
instead of the whole ~19 GiB model — a 16 GiB box now loads it in minutes rather than swapping for hours. On the maintainer's
32 GB host the other VMs are paused while the model loads; on 16 GiB hosts they no longer need to be.
Windows 11 through WSL2: see [docs/WINDOWS-WSL2.md](docs/WINDOWS-WSL2.md).

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
  -e RADIANCE_MXFP4_WIDE_MAX_M=192 -e RADIANCE_ADAPTIVE_WIDTH=perseq \
  ghcr.io/slybase/vllm-sly-radiance:1.1.0-rocm10.1 \
  --model amd/Qwen3.8-27B-Quark-AWQ-MXFP4 --quantization quark \
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
  -d '{"model":"amd/Qwen3.8-27B-Quark-AWQ-MXFP4","messages":[{"role":"user","content":"Hello!"}]}'
```

- **First start** compiles kernels and CUDA graphs (~10 min) and reports a ~2 GiB smaller KV pool;
  **restart once** — the warm start takes 4–6 min and gets the full 436k-token KV pool. Keep the cache
  volumes.
- **Power:** 210 W costs ~6.5 % decode and ~20 % prefill against 300 W (table above). The card allows
  up to 330 W: +4 … +8 % prefill and ~+1 % step time over 300 W, but the junction runs at 102 °C average
  (106 °C peak) and the fan at ~4,200 rpm — not worth it for 24/7 use.
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
| `--attention-backend R4D_HYBRID`: libr4d prefill attention for runs >= 512 tokens, AITER for the rest (1.0) | plain R4D prefills faster but decodes at 60.9 instead of 34.9 ms/step; the hybrid keeps AITER's decode | prefill 8k / 32k / 64k **+3 / +13 / +23 %** |
| `--gpu-memory-utilization 0.98` + `FULL_DECODE_ONLY` (1.0) | the PIECEWISE graph set cost 1.24 GiB of graphs and 1.69 GiB peak activation | KV pool 391,193 → **436,097** tokens (+11.5 %) |
| `RADIANCE_MXFP4_WIDE_MAX_M=192` (1.0) | M 129…192 ran on a 256-row tile with up to 49 % padding; now the split-K decode kernel | TTFT of a 160-token prompt **−13.8 %** |
| `RADIANCE_ADAPTIVE_WIDTH=perseq` (1.0, ported from the Radiance engine) | acceptance differs 2× between requests; per-request verify width on varlen FULL graphs ([why ggz14's version lost](docs/TECHNICAL.md#100-what-the-release-adds-and-the-measurements-behind-it)) | 8 concurrent requests **+4.7 %** |
| ROCm 10.1 + torch 2.12 (1.0) | current AMD toolchain; the CPU-spin bug that held torch back is fixed | neutral (±1–2 %) |
| vLLM 0.31.0 + torch 2.13 / triton 3.8.0 / torchvision 0.28.0 / aiter 0.1.23 (1.1) | the combination vLLM's own ROCm base builds, as AMD wheels | *pending GPU run* |
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
| `RADIANCE_R4D_PREFILL_MIN_Q` | `512` | with `--attention-backend R4D_HYBRID`: smallest per-request chunk that goes to libr4d's prefill kernel; smaller chunks (decode, TTFT of short prompts) stay on AITER |
| `RADIANCE_R4D_HYBRID_ROUTE` | `1` | `0` = the hybrid backend never routes to libr4d (= AITER, for A/Bs) |
| `RADIANCE_MXFP4_A_TILED_MIN_M` | `513` | fragment-tiled prefill GEMM from M = 513 (+11–13 % prefill); must stay above 512 and `DECODE_MAX_M` |
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
| `--max-num-batched-tokens` | `4096` | prefill chunk size; the shipped default (`CHUNK`), and the chunk the measured prefill stack was taken at. Larger chunks prefill long prompts faster but take activation memory from the KV pool: measured at `--max-model-len 131072`, 4096 gives **+3.5 … +4.8 % prefill** over 2048 at 8k–64k prompts (short prompts unchanged) for ~24k fewer KV tokens (−7 %); 8192 is no faster than 4096 and costs ~68k tokens. 2048 keeps the full pool for 262k |
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
than the prefill phase of the full run in the table above; compare within one table.)

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
| `KVCACHE_DTYPE` | `fp8` | `fp8` (default) = stock; `kv4` = TetraAttention, 4-bit K/V for the attention layers, an opt-in trade: **~1.75x the KV pool** (a full 262k context in 4.25 GiB of KV instead of 8) for slightly less precision (KL against a bf16 KV cache 0.0126 vs fp8's 0.0123, perplexity +0.54 % vs +0.06 %, top-1 agreement 95.3 vs 95.6 %; GSM8K-200 0.80 vs 0.79). Decode 4-5 % faster per step at 1 and 4 streams (same-window A/B), prefill 3-4 % faster with `CHUNK` >= 3456 (the attention block is 1664 tokens, fp8: 880, so a turn recomputes up to 1,663 tokens on a prefix hit). Knobs `RADIANCE_KV4H_*` (sly/kv4/README.md). An offload tier written by an fp8 serve must not be reused |

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

## Repository

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

License: same as upstream vllm-radiance (see `LICENSE`).
