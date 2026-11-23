# Inkling 2 Mini on one R9700

Serving recipe, knobs and measurements for `thinkingmachines/Inkling-2-Mini` with its DSpark drafter on
gfx1201 (AMD Radeon AI PRO R9700). Branch `feat/inkling2-mini` ([PR #259](https://github.com/SlyBase/vllm-sly-radiance/pull/259)),
vLLM `main@1bdc437` (the model is in no vLLM release). All numbers below are measured on the R9700 at 210 W.

## The model

Looped MoE, 26B parameters, 3.5B active per token. NVFP4 experts in layers 0-19, MXFP8 experts in layers
20-23, **bf16 dense** (attention, output, LM head, router, drafter projections). DSpark drafter, depth
k = 7 fixed (a verify step is M = 8 rows).

vLLM support is upstream PR [vllm#69028](https://github.com/vllm-project/vllm/pull/69028). No vLLM release
carries it (up to v0.34.0), so this branch pins vLLM `main@1bdc437` (`ARG VLLM_REF` in the Dockerfile,
see the CHANGELOG, *Changed*). The model's fast path is an sm120 Triton path. `sly/patch_inkling2_gfx12.py`
turns it into a ROCm path; the hand-written HIP kernels below replace the Triton kernels for decode.
All of it is a NOOP on a vLLM without `models/inkling2`.

## Arguments

```bash
-e GPU_MAX_HW_QUEUES=1 \
... --model thinkingmachines/Inkling-2-Mini \
--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}' \
--reasoning-parser inkling \
--no-async-scheduling \
--compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8]}'
```

Device, group and cache-volume flags: as in the README [Quickstart](../README.md#quickstart). Use an image
built from this branch (`build.yml`); no release image carries Inkling yet.

- `GPU_MAX_HW_QUEUES=1` is the recipe value (the Qwen recipe uses 2).
- `--no-async-scheduling` and graph sizes 1-8 match the step shape: one stream, k = 7, M <= 8.
- Startup shows `[radiance.draft] install failed ... llm_base_proposer`. It is harmless: only the
  dynamic-width draft hook (off here) is missing.
- The measurements below use `--max-model-len 65536`, prefix caching off, one stream.

## Components and knobs

| Component | Knob (default) | What it does |
|---|---|---|
| `sly/patch_inkling2_gfx12.py` | – (always, NOOP without `models/inkling2`) | the model's sm120 Triton path on ROCm: capability gates, no PDL, PTX replaced by Triton math, flashinfer-free drafter |
| HIP expert GEMV (`inkling2/radiance_inkling2_moe.hip`) | `RADIANCE_INKLING2_HIP_MOE` (on; `=0` = Triton) | NVFP4 / MXFP8 experts, decode and verify tiles (<= 8 tokens): 529-648 GB/s, 3.1-3.4x (NVFP4) and 6.0-6.8x (MXFP8) the Triton kernels; NVFP4 output bit-identical. Prefill tiles stay on Triton (no FP4 matrix unit on gfx12) |
| | `RADIANCE_INKLING2_HIP_DEBUG` (off; `=1`) | host-side launch checks of the expert kernel |
| HIP bf16 GEMV (`inkling2/radiance_inkling2_gemv.hip`) | `RADIANCE_INKLING2_HIP_GEMV` (on; `=0` = Triton) | dense projections at M 1..16, lossless; 558-596 GB/s at M = 8 on the big projections |
| Fused route without cooperative launch | – (always on ROCm) | removes the bimodal 62 / 31 ms step, see [TECHNICAL.md](TECHNICAL.md#inkling-2-mini-on-gfx1201-unreleased) |
| DSpark launch folding (`inkling2/radiance_inkling2_draft.py`) | – (always) | same math in fewer launches: 30.3 -> 29.55 ms per step |
| Drafter shortlist | `RADIANCE_INKLING2_DRAFT_SHORTLIST=<k>` (off) | Markov correction on the top-k base logits only: -1.7 ms per step, but acceptance on code 3.43 -> 2.97 tokens per step. Off |
| int8 (Q8_0) dense weights | `RADIANCE_INKLING2_W8=1` (off) | opt-in, M 1..16: 533-585 GB/s of int8 bytes at M = 8, 1.5-1.8x the bf16 GEMV |
| | `RADIANCE_INKLING2_W8_HEAD` (`1`; read only with `W8=1`) | `=0` keeps the 201,024-row LM head on bf16 |
| MXFP4 dense weights (`patch_inkling2_mxfp4dense.py`) | `RADIANCE_INKLING2_MXFP4_DENSE=1` (off) | opt-in 4-bit dense projections + LM head on the HIP W4A16 GEMV (`radiance_inkling2_mx4.hip`); dense bytes 4.77 -> 1.27 GB per step. Quality trade-off, not lossless |
| | `RADIANCE_INKLING2_MXFP4_DENSE_MODE` (`gemv`; `w4a8`; `w4a16`) | `gemv`: bf16 activations, one launch per call, only a pre-tiled code copy (M > 16 dequantizes per call); `w4a8`: the radiance fp8-WMMA kernel (slower, fp8 activation error); `w4a16` dequantizes back to bf16: isolates the weight-quantization cost |
| | `RADIANCE_INKLING2_MXFP4_DENSE_CLIP` (`rtn`; `mse`) | exponent choice per group: round-to-nearest or a small MSE search |
| | `RADIANCE_INKLING2_MXFP4_DENSE_MIN_N` (`256`) | projections with fewer output rows stay bf16 |

With every knob at its default the dense path is lossless bf16 and the experts run the exact NVFP4 /
MXFP8 arithmetic of the model.

## Measurements

Environment for all rows: one R9700, 210 W, image `ink-k3` (this branch), vLLM `main@1bdc437`, the
arguments above, `GPU_MAX_HW_QUEUES=1`.

**BetterBench single-stream decode** (65,536 context, prefix caching off, no async scheduling, CUDA graphs,
k = 7, seed 42, tok/s, two container starts):

| Category | Start 1 | Start 2 |
|---|---:|---:|
| chat | 90.3 | 90.1 |
| code | 125.6 | 125.3 |
| file_edit | 154.5 | 154.4 |
| json | 152.6 | 152.3 |
| math | 140.5 | 140.2 |
| prose | 91.3 | 91.3 |
| reasoning | 110.7 | 110.7 |
| summarization | 148.8 | 148.8 |
| **weighted** | **126.7** | **126.6** |

The verify step is about 30 ms in every category; the categories differ in accepted tokens per update
(code: 3.74). Before the cooperative-launch fix (image `ink-k2`): code 62.9, weighted 62.0.

Per-category and history tables: [BENCHMARKS.md](BENCHMARKS.md#inkling-2-mini-on-gfx1201-2026-11).

**Kernels** (rotated > 300 MB weight pool, M = 8): see the knob table. Model level, Q8 dense weights (eager,
no drafter, greedy, top-20 logprobs, 32 prompts x 256 tokens): KLD mean 0.0081, median 0.0007, p99 0.080,
top-1 97.3 %. The lossless bf16 HIP kernels against Triton measure 0.0148 / 0.0006 / 0.218 / 96.5 % on an
8-prompt set, so Q8 costs no more than switching bf16 kernels.

**MXFP4 dense weights:** weight error on the real weights 0.115 relRMS (RTN) / 0.112 (MSE), CPU check
`sly/inkling2/check_mxfp4dense_cpu.py`. Model KLD (graphs + DSpark k=7, vs the bf16 default): mean 0.0176,
median 0.00011, p99 0.179, top-1 95.7 % on 186 positions before the first divergence stops the comparison
(`w4a8` 0.026 / 0.34 / 96.6 % on 234, `w4a16` 0.022 / 0.19 / 95.6 % on 181; int8 0.008). 4-bit dense costs about
twice the int8 KLD, so it stays opt-in.

**Kernel micro-bench** (`sly/inkling2/bench_mx4.py`, rotated > 300 MB weight pool, M = 8, us per call inside ONE
replayed CUDA graph; GB/s of the arm's weight bytes: MXFP4 codes + E8M0 scales, int8 codes + fp32 scales, bf16):

| shape (N x K) | MXFP4 `gemv` | MXFP4 `w4a8` | int8 Q8 | bf16 | gemv / w4a8 | gemv / int8 |
|---|---:|---:|---:|---:|---:|---:|
| 384 x 2304 | 4.5 us (105 GB/s) | 12.1 | 5.9 | 6.7 | 2.7x | 1.3x |
| 1024 x 2304 | 6.0 (208) | 12.9 | 7.7 | 10.9 | 2.2x | 1.3x |
| 2304 x 3072 | 10.6 (356) | 17.6 | 16.4 | 26.8 | 1.7x | 1.5x |
| 3072 x 2304 | 10.3 (364) | 17.0 | 16.1 | 27.7 | 1.7x | 1.6x |
| 6552 x 2304 | 18.6 (432) | 28.0 | 31.2 | 53.5 | 1.5x | 1.7x |
| 8600 x 2304 | 22.4 (469) | 31.9 | 39.7 | 69.3 | 1.4x | 1.8x |
| 201024 x 2304 (lm_head) | 394-452 (544-624) | 1378 | 950 | 1549 | 3.0-3.5x | 2.1-2.4x |

Mid-size shapes are launch bound (~4 us fixed + bytes at ~550 GB/s); the same is true at M = 1 and 16 (within 5 %).
Two runs of the table agree within 2 % except the lm_head (452 / 394 us). The load-time proof line reports 751 MiB
of MXFP4 tiles after the draft (the unaligned k/v slices carry their own tile copies; the aligned ones are views).
The `w4a8` route's flat ~29 us per call in the old eager micro-bench (`bench_mxfp4dense.py`) was HOST time: ~28.7 us
of CPU per call (custom-op dispatch + fp8 activation-quant launch + `torch.empty` + kernel launch), the same op in a
graph costs 12.5 / 17.1 / 31.7 us at N = 384 / 3072 / 8600 (rocprofv3: quant kernel 3 us + the Qwen-tuned decode
kernel at 300-330 GB/s, 12.8 us even at N = 384). The model's decode graphs never paid the host part.

Decode speed, code category (BetterBench, graphs + DSpark k=7, image ink-k5 + overlay, two starts): 162.4 / 161.6 tok/s (weighted 157.2 / 156.8), vs int8 151.1 / 145.9, `w4a8` 143.5, default 127.3 / 124.4, prod Qwen 155.

**Default vs int8 on this image** (two starts each): default 127.3 / 124.4 code, 128.1 / 126.6 weighted; `RADIANCE_INKLING2_W8=1` 151.1 / 145.9 code, 148.5 / 145.7 weighted (BetterBench, image ink-k5 @3f5e138, 2026-11-23, two starts each; prod Qwen code 155, weighted 142).

### Why Inkling decodes code at 125.6 and Qwen at 155

The production Qwen (Swift-1.5-Qwen3.8-27B-MXFP4-GPTQ, fully 4-bit dense) decodes code at 155 tok/s;
Inkling decodes code at 125.6. Inkling's dense weights are bf16, four times the bytes of Qwen's. Of the
30 ms step, the experts take 11.3 ms and the dense projections 8.9 ms, and both already run at DRAM
bandwidth, so a faster kernel cannot close the gap. Fewer bytes can: that is what the int8 and MXFP4
dense options are for. The other 9.8 ms of the step (30 - 11.3 - 8.9, derived) are not at DRAM bandwidth.
Measured: int8 (half the dense bytes) reaches 151.1 / 145.9 code, MXFP4 dense on the HIP GEMV (~1/4 of the
bytes) 162.4 / 161.6 -- above Qwen's 155.

## Known issues

- One of four `RADIANCE_INKLING2_W8=1` starts died with `ILLEGAL_INSTRUCTION` at CUDA-graph capture. The
  faulting kernel was a foreign 512-thread kernel, not one of ours. Cause unexplained.
  W8 stays opt-in.
- The model is in no vLLM release: the branch depends on the `main@1bdc437` pin until a release carries
  PR #69028.
- `KVCACHE=ram|disk` is unsupported on this base ([kv-cache/README.md](../kv-cache/README.md)).

Not adopted for this model (Strata, Basalt, W8A8, queue and power knobs):
[NOT-ADOPTED.md](NOT-ADOPTED.md#inkling-2-mini-2026-11).

## Tests

CPU: `pytest sly/tests sly/inkling2/tests` (the GPU-only tests skip). GPU: `INKLING_VLLM_SRC=<vllm src dir>
pytest sly/inkling2/tests` on a gfx1201 host. The kernel tests include mutation runs
(`mutate_gemv_hip.py`, `mutate_moe_hip.py`) that must kill every mutant.
