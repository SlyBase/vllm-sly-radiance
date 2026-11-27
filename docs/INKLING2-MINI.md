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
--compilation-config '{"cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,16,24,32,40,48,56,64]}'   # the entrypoint adds this by default
```

Device, group and cache-volume flags: as in the README [Quickstart](../README.md#quickstart). Use an image
built from this branch (`build.yml`); no release image carries Inkling yet.

- `GPU_MAX_HW_QUEUES=1` is the recipe value (the Qwen recipe uses 2).
- `--no-async-scheduling`. Decode graph sizes `1..8,16,...,64` (entrypoint default) cover 1-8 streams of k = 7 steps (8 tokens each); sizes 1-8 alone capture one decode graph and run 2+ streams eager.
- Startup shows `[radiance.draft] install failed ... llm_base_proposer`. It is harmless: only the
  dynamic-width draft hook (off here) is missing.
- The measurements below use `--max-model-len 65536`, prefix caching off, one stream.

## Components and knobs

| Component | Knob (default) | What it does |
|---|---|---|
| `sly/patch_inkling2_gfx12.py` | – (always, NOOP without `models/inkling2`) | the model's sm120 Triton path on ROCm: capability gates, no PDL, PTX replaced by Triton math, flashinfer-free drafter |
| HIP expert GEMV (`inkling2/radiance_inkling2_moe.hip`) | `RADIANCE_INKLING2_HIP_MOE` (on; `=0` = Triton) | NVFP4 / MXFP8 experts, decode and verify tiles (<= 8 tokens): 529-648 GB/s, 3.1-3.4x (NVFP4) and 6.0-6.8x (MXFP8) the Triton kernels; NVFP4 output bit-identical. Tiles 16..64 (2-8 streams) run as tile-8 CTAs (`RADIANCE_INKLING2_HIP_MOE_SPLIT=0` = Triton); prefill tiles stay on Triton (no FP4 matrix unit on gfx12) |
| | `RADIANCE_INKLING2_HIP_DEBUG` (off; `=1`) | host-side launch checks of the expert kernel |
| HIP bf16 GEMV (`inkling2/radiance_inkling2_gemv.hip`) | `RADIANCE_INKLING2_HIP_GEMV` (on; `=0` = Triton) | dense projections at M 1..16, lossless; 558-596 GB/s at M = 8 on the big projections |
| Loop controllers on the batched HIP GEMV (`inkling2/radiance_inkling2_gemv.hip` `gemv_bf16_batched`) | `RADIANCE_INKLING2_HIP_CTRL=1` (off) | the two `torch.bmm` of the loop blend (hipBLASLt, 12 / 6-8 us per call, 0.92 ms per step) as two batched launches (stream = grid.y); bf16 weights, fp32 accumulate. Micro-bench `inkling2/bench_ctrl.py`; end-to-end not yet measured |
| Int8 fold weight (`inkling2/radiance_inkling2_draft.py`) | `RADIANCE_INKLING2_FOLD_INT8=1` (off) | lossy, drafter only: fold weight as int8 + fp32 row scale, 103 -> 51 MB per Markov step (7 per verify step). Micro-bench `inkling2/bench_fold.py`; end-to-end not yet measured |
| Fused route without cooperative launch | – (always on ROCm) | removes the bimodal 62 / 31 ms step, see [TECHNICAL.md](TECHNICAL.md#inkling-2-mini-on-gfx1201-unreleased) |
| DSpark launch folding (`inkling2/radiance_inkling2_draft.py`) | – (always) | same math in fewer launches: 30.3 -> 29.55 ms per step |
| Drafter shortlist (SLAK) | `RADIANCE_INKLING2_DRAFT_SHORTLIST=<k>` (**4096** by the image entrypoint, `0` = off) | Markov correction on the top-k base logits only: -1.7 ms per step; alone, acceptance on code fell 3.43 -> 2.97 tokens per step. Only the draft distribution changes, the rejection sampler gets it as such, so the target distribution is exact. Whether the net is positive without NSD: chain NQ. With it `FOLD_INT8` is inert (the gather path reads k fold rows only; logged) |
| Nucleus speculative decoding (NSD) | `RADIANCE_NSD` (off, **opt-in**), `RADIANCE_NSD_THETA` (0.95), `RADIANCE_CACTUS_DELTA` (image default 0.5; must be `0` with NSD, else import error) | lossy acceptance rule in the rejection kernel, see `sly/patch_nsd.py`. Degraded the greedy text in W3 (see below); do not enable for quality-sensitive use |
| Adaptive verify window | `RADIANCE_INKLING2_ADAPTIVE_K` (**3,5,7** by the image entrypoint, empty = off) | k per step from the acceptance EMA (exact for any k), `sly/patch_inkling2_adaptive.py` |
| Coda experts NVFP4 (`inkling2/radiance_inkling2_coda_fp4.py`) | `RADIANCE_INKLING2_CODA_FP4` (**1** by the image entrypoint, `0` or empty = off, or a layer list `20,22-23`) | **lossy default.** The MXFP8 coda expert layers 20-23 are requantized to NVFP4 at load: -0.44 GB per decode step. *MEASURED* 2026-11-25 vs the exact MXFP8 run: KLD mean **0.0033 nats** (CI 0.0026-0.0042, gate <= 0.010), median 0.00025, p99 0.053, top-1 agreement 98.46 %, **dPPL +0.26 %**; DFlash2 code decode 224.4 -> 233.0 tok/s (**+3.9 %**), step 23.26 -> 22.70 ms. Greedy text diverges after about 110-220 characters (expected for a lossy quantization). `RADIANCE_INKLING2_CODA_FP4=0` restores the exact experts |
| Prefill expert GEMM v2 (`inkling2/radiance_inkling2_moe.py` `_expert_gemm_v2`) | `RADIANCE_INKLING2_MOE_PREFILL_V2` (unset = **on for token tiles >= 128**, i.e. prefill chunks; `0` = the old kernel, `1` = every tile) | lossless (same products, fp32 accumulate; bf16 output rounding differs by <= 0.12 % max rel.). Activations dequantised once per row instead of once per column block, e2m1 decoded with bit tricks instead of `exp2`, even/odd nibbles as two dots, scale loaded once per 16-k group, truncating fp32->bf16 (exact for these values), N blocks of a tile run back to back (activation rows stay in L2). *MEASURED* 2026-11-26, R9700 210 W, 4096-token chunk, w13+w2 per layer visit: 40.9 ms (historical launch) -> 19.7 ms (tuning table) -> **8.5 ms (v2, 61 TF/s)**. Sweep tool `inkling2/bench_prefill_moe.py --v2` |
| Prefill launch table | `RADIANCE_INKLING2_MOE_PREFILL_CFG` (unset = built-in table for tile 128; `<path>.json` replaces it; `0` / `off` = historical launch) | the v1 kernel's BM/BN/BK/warps/stages (Inkling does not read vLLM's fused_moe configs). Arm PT: 1192/1298/1288/1245 -> 2042/2128/2109/1983 pp/s, dNLL +0.0000 |
| Breakable prefill graphs | `VLLM_USE_BREAKABLE_CUDAGRAPH` (**1** by the image entrypoint for Inkling, `0` = off) | prefill batches of the captured sizes replay a CUDA graph: cold first request 4.2 s -> 72 ms, c8 first wave 9 s -> 0.7 s, decode neutral |
| MoE plan tile | `RADIANCE_INKLING2_HIP_MOE_TILE` (**64** by the image entrypoint for Inkling, `0` = vLLM tile 16) | 17..64-token steps (2-8 streams) plan one CTA per expert: c8 +7..10 %, Mia TTFT -8 ms |
| Prefill chunk | `RADIANCE_INKLING2_CHUNK` (unset = **4096 by the image entrypoint for Inkling when an explicit `--max-model-len` <= 131072 is given**; `<N>` = that size for any context; `0` or empty = vLLM's 2048; an explicit `--max-num-batched-tokens` wins) | +14..16 % prefill pp/s at 8k..32k tokens. **Costs 0.8 GiB of KV (4640 -> 4216 blocks; the log's token count falls 94 849 -> 53 815 at max-model-len 32768) and raises the KV need of one max-length request: at `--max-model-len 202752` and 0.92 GPU utilisation the engine refuses to start (10.13 GiB needed, 8.23 available, estimated limit 140416; chunk 2048 starts with 1.09x concurrency)**, hence no default for longer or unspecified contexts |
| Prefill attention tile | – (always on ROCm, hunk 4m of `patch_inkling2_gfx12.py`) | `_rel_attention_kernel` BLOCK_N 64 -> 32 for the 128-row tile: local layer 2.2 -> 1.2 ms, global layer 11.9 -> 5.7 ms per 4096-token chunk. Changes the summation order: greedy text of 2 of 4 gate prompts diverges after 113-185 characters (coherent), dNLL -0.0001 |
| int8 (Q8_0) dense weights | `RADIANCE_INKLING2_W8=1` (off) | opt-in, M 1..16: 533-585 GB/s of int8 bytes at M = 8, 1.5-1.8x the bf16 GEMV |
| | `RADIANCE_INKLING2_W8_HEAD` (`1`; read only with `W8=1`) | `=0` keeps the 201,024-row LM head on bf16 |
| MXFP4 dense weights (`patch_inkling2_mxfp4dense.py`) | `RADIANCE_INKLING2_MXFP4_DENSE` (**1** by the image entrypoint for Inkling, `0` or empty = off) | **lossy default.** 4-bit dense projections + LM head on the HIP W4A16 GEMV (`radiance_inkling2_mx4.hip`); dense bytes 4.77 -> 1.27 GB per step. *MEASURED* 2026-11-26 (paired A/B, 1.14.0 prod config): GSM8K-150 -1.3 pp, HumanEval+ -1.2 pp (both within noise), dNLL +0.0091; see "MXFP4 dense default" |
| | `RADIANCE_INKLING2_MXFP4_DENSE_MODE` (`gemv`; `w4a8`; `w4a16`) | `gemv`: bf16 activations, one launch per call, only a pre-tiled code copy (M > 16 dequantizes per call); `w4a8`: the radiance fp8-WMMA kernel (slower, fp8 activation error); `w4a16` dequantizes back to bf16: isolates the weight-quantization cost |
| | `RADIANCE_INKLING2_MXFP4_DENSE_CLIP` (`rtn`; `mse`) | exponent choice per group: round-to-nearest or a small MSE search |
| | `RADIANCE_INKLING2_MXFP4_DENSE_MIN_N` (`256`) | projections with fewer output rows stay bf16 |

With the entrypoint defaults the dense projections and the LM head are MXFP4 (`RADIANCE_INKLING2_MXFP4_DENSE=1`, lossy, see
"MXFP4 dense default"; `=0` = lossless bf16), and the experts run the exact NVFP4 / MXFP8 arithmetic of the model except the
coda layers 20-23, which the entrypoint default requantizes to NVFP4 (`RADIANCE_INKLING2_CODA_FP4=1`, KLD 0.0033 nats, dPPL +0.26 %; `=0` = exact).

## Default decode profile (SLAK + adaptive k, lossless subset)

`radiance_entrypoint.sh` exports `RADIANCE_INKLING2_CODA_FP4=1` and `RADIANCE_INKLING2_MXFP4_DENSE=1` for every Inkling model (any drafter; both lossy, see the
table above) and two lossless knobs when the **target model** is Inkling: `--model X`, `--model=X` or a first positional whose name
contains `inkling` (case-insensitive), **or** a local dir whose `config.json` `model_type`/`architectures` contain
`inkling` (the `/model` mount; real value `inkling2_loop` / `Inkling2LoopForConditionalGeneration`), **or**
`--reasoning-parser inkling` (Inkling-only, fallback). The drafter path, `--tokenizer` and `--served-model-name` are ignored,
so a Qwen target with an `inkling` drafter gets nothing. The two lossless knobs are skipped when `--speculative-config` selects the DFlash2 drafter (`"method":"dflash"`, see the table
under "Recommended decode config"); every explicit value, an empty one included, wins, and other models are untouched. The
`--entrypoint vllm` override of a test harness bypasses it: pass the knobs with `-e`.

| Knob | Default |
|---|---|
| `RADIANCE_INKLING2_CODA_FP4` (all drafters) | `1` |
| `RADIANCE_INKLING2_MXFP4_DENSE` (all drafters; `0` = lossless bf16 dense) | `1` |
| `RADIANCE_INKLING2_DRAFT_SHORTLIST` | `4096` |
| `RADIANCE_INKLING2_ADAPTIVE_K` | `3,5,7` |

NSD is **not** a default. Commit 7060e46 had made NSD (theta 0.95) + Cactus 0 + SLAK + adaptive k the default for
+5.1 % (W2, code 170.49 vs 162.2 tok/s). **W3 (2026-11-25, greedy gate, 4 code prompts, image `ink-k9`) measured a
quality regression:** the control is 4/4 identical across three starts; the five-knob arm NS is 2/4 and the
diverging texts are visibly degraded ("In Python, binary binary in Python and Python and state its time
complexity"; "a mutual exclusion... a, a, or mutual exclusion lock"). Probable cause, from reading the code (not yet proven on the GPU): in the rejection
kernel the NSD test is OR-ed into `accepted` for greedy rows and for masked (-1) draft slots too, but the nucleus
kernel returns early for those rows and leaves its `new_empty` buffer unwritten, so garbage can pass as
"inside the nucleus" (`sly/patch_nsd.py` steps 2, 4, 5). Part of the NSD speed-up (accept 0.346 -> 0.479) may
therefore not be real. The isolating chain `/root/ink-200/nq` (arms SL, N99, N97, SLC) is queued; results go to
`docs/` when measured.

K1 (`HIP_CTRL` + `FOLD_INT8`) showed no effect in W2 because that arm's image had the new `.so` but not the
Python that reads the knobs (no K1 result yet); both now log a proof line (`[radiance.inkling2] HIP_CTRL active` /
`FOLD_INT8 active`) and say why when a knob falls back.

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
twice the int8 KLD; since 1.14.1 it is nevertheless the entrypoint default for Inkling (paired A/B below).

### MXFP4 dense default (1.14.1)

Every 1.14.0 campaign measurement ran with `RADIANCE_INKLING2_MXFP4_DENSE=1`; the image default was `0`. Prod (1.14.0, 307200 ctx, fp8 KV,
DFlash2 k=7) without the knob: Mia decode prose / code **140 / 191 tok/s**; with it **186 / 261** (tokens per update, code 5.71 -> 6.03;
prod re-run after the rollout 178 / 241, 4.14 / 5.74). The KV pool grows 387,121 -> 414,824 tokens (the bf16 dense weights are freed).

Paired quality A/B, 2026-11-26, same image and prod args, one GPU window, R9700 210 W, D0 = `DENSE` unset (bf16), D1 = `DENSE=1`:

| Metric | D0 (bf16) | D1 (MXFP4) | D1 - D0 | paired 95 % CI | McNemar p |
|---|---|---|---|---|---|
| GSM8K-150, greedy, thinking, 8192 tok | 146 (97.3 %) | 144 (96.0 %) | -1.3 pp | -3.2..+0.5 | 0.500 |
| HumanEval+ (164), thinking, 8192 tok | 153 (93.3 %) | 151 (92.1 %) | -1.2 pp | -5.0..+2.6 | 0.754 |
| NLL, 6525 code tokens | 0.7515 | 0.7606 | dNLL +0.0091 +-0.0064 | | |

Rule: no significant drop and a point delta >= -2 pp on both tasks: **PASS**. Limits: n = 150 / 164 and one run per arm, so a loss of a
few points is not excluded by the CI; the greedy gate is not byte-identical (0/4, coherent text). `RADIANCE_INKLING2_MXFP4_DENSE=0` restores
the lossless bf16 dense weights.

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

## Recommended decode config: DFlash2 drafter, k = 7

`incoai/Inkling-2-Mini-DFlash2` (0.5B, bf16, 5 sliding-window layers, candidate selector) runs on vLLM's native `dflash2`
speculator through `sly/patch_inkling2_dflash2.py`. Serve it (the draft dir name must contain `dflash2`):

Pin (the only verified artifact): repo `incoai/Inkling-2-Mini-DFlash2`, revision
`4f9088ea66c21f2b99ed657d1941c306b03e84d2`, license Apache-2.0, `model.safetensors` sha256
`217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee`. Download and verify:

```
hf download incoai/Inkling-2-Mini-DFlash2 --revision 4f9088ea66c21f2b99ed657d1941c306b03e84d2 --local-dir /models/inkling-2-mini-dflash2 && echo "217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee  /models/inkling-2-mini-dflash2/model.safetensors" | sha256sum -c -
```

```
--speculative-config '{"method":"dflash","model":"<dir named *dflash2*>","num_speculative_tokens":7,"draft_sample_method":"probabilistic","attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto"}'
```

No extra environment variable: the drafter is bf16, and the W4A16 post-load hook leaves its conv projections bf16 (it
requantizes them to int4 only for a W4A16 drafter checkpoint, or with an explicit `RADIANCE_DFLASH_CONV_W4=1`).

**MEASURED** (2026-11-25, one R9700 at 210 W, BetterBench code decode, T 0.7, seed 42, k = 7, image `ink-k9-df2-fix`):

| Arm | Drafter | code tok/s | tok/update | step ms | weighted tok/s |
|---|---|---|---|---|---|
| C1 | DSpark (previous default) | 161.96 | 3.90 | 23.26 | 157.01 |
| D1 | DFlash2, conv projections requantized to int4 (the old hook default) | 159.76 | 3.65 | 23.48 | 152.93 |
| **D4** | **DFlash2, conv projections bf16 (`RADIANCE_DFLASH_CONV_W4=0`, now the default)** | **224.16 (+38 % vs C1)** | **5.13** | 23.26 | **210.70** |

Greedy gate, 4 code prompts: D4 byte-identical to the control 4/4, no repetition. The int4 requantization is the whole
difference between D1 and D4: same step time, acceptance 5.13 -> 3.65 tokens per update (the target verifies every
token, so it only cost proposals, never correctness). Vendor card (H200, SGLang, T 1.0): acceptance length 4.80 vs 3.66
(DSpark) on HumanEval. An earlier "DFlash2 does not win here" reading came from D1 and is superseded by D4.

DSpark stays the fallback: serve `{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}`
when the DFlash2 model directory is not present.

| Config | Drafter | Image knobs (SLAK 4096 + adaptive k 3,5,7) | code tok/s | tok/update |
|---|---|---|---:|---:|
| **Recommended** | DFlash2 bf16, k = 7 | **none** (the entrypoint leaves them unset) | **224** (223.97 / 223.95, two starts) | 5.13 |
| Fallback | DSpark, k = 7 | on (entrypoint default) | 173.6 (NQ arm SL; control without them 161.8, +7.3 %, greedy gate 4/4) | – |
| Not recommended | DFlash2 bf16 + the knobs forced on | on | 221.46 | 4.86 |

*MEASURED* 2026-11-25, image `ink-200`, R9700 at 210 W, BetterBench code. The knobs help DSpark but cost the DFlash2
drafter about 1 % (its step is 22.07 vs 23.29 ms, but acceptance drops 5.13 -> 4.86), so
`radiance_entrypoint.sh` sets them only when `--speculative-config` does not contain `"method":"dflash"` (JSON with
or without spaces; no speculative config = DSpark default, knobs on). `-e RADIANCE_INKLING2_DRAFT_SHORTLIST=...` /
`RADIANCE_INKLING2_ADAPTIVE_K=...` still win.

## Tests

CPU: `pytest sly/tests sly/inkling2/tests` (the GPU-only tests skip). GPU: `INKLING_VLLM_SRC=<vllm src dir>
pytest sly/inkling2/tests` on a gfx1201 host. The kernel tests include mutation runs
(`mutate_gemv_hip.py`, `mutate_moe_hip.py`) that must kill every mutant.
