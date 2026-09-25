# Benchmarks: methodology and history

Current reference numbers are in the [README](../README.md#performance). This file keeps the full tables of earlier releases, the checkpoint comparison and why single runs mislead.

## 0.4.0

### Reference run (2026-09-25)

Image 0.4.0 (vLLM 0.30.0) with the production arguments of the README quickstart, run as the acceptance
candidate in a GPU window (second start, KV pool 384,316 tokens), BetterBench 0.4.0 default config: single-
stream decode 3 warmup + 20 passes per category, prefill sweep (unique prompts, no prefix cache, 16 output
tokens), concurrency 1/2/4/8/16 × 48 requests; sampling temperature 0.7 / top_p 0.95 / top_k 20. 300 W with
the firmware fan curve, then 210 W with the fan capped at 2,800 rpm. Card sampled every 5 s.

**Single-stream decode** (tok/s ± 95 % CI; tokens per client update):

| Category | 300 W | 210 W | tokens/update (300 W) |
|---|---|---|---|
| chat | 91.6 ± 4.0 | 89.5 ± 7.3 | 3.13 |
| code | 143.2 ± 9.2 | 130.9 ± 8.1 | 4.94 |
| file_edit | 173.8 ± 7.1 | 167.2 ± 5.8 | 5.93 |
| json | 166.3 ± 10.3 | 155.2 ± 10.5 | 5.66 |
| math | 171.1 ± 10.3 | 160.8 ± 8.4 | 5.83 |
| prose | 86.2 ± 3.6 | 79.3 ± 3.3 | 2.99 |
| reasoning | 108.2 ± 13.9 | 103.7 ± 14.0 | 3.73 |
| summarization | 133.6 ± 7.3 | 125.9 ± 8.6 | 4.55 |
| **weighted** | **133.2** | **124.5** | 4.57 |
| step gap p50 | 34.84 ms | 36.99 ms | |

**Prefill:**

| Prompt tokens | 1,514 | 5,918 | 11,794 | 23,543 | 47,056 |
|---|---|---|---|---|---|
| 300 W, tok/s | 3,166 | 3,161 | 3,108 | 2,903 | 2,512 |
| 210 W, tok/s | 2,527 | 2,498 | 2,461 | 2,311 | 2,027 |
| TTFT 300 W / 210 W | 0.48 / 0.60 s | 1.87 / 2.37 s | 3.79 / 4.79 s | 8.11 / 10.19 s | 18.7 / 23.2 s |

**Concurrency** (48 requests per level, all OK, 0 preemptions):

| Concurrency | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| 300 W, aggregate tok/s | 121.0 | 214.1 | 331.9 | 410.0 | 405.7 |
| 210 W, aggregate tok/s | 110.9 | 197.2 | 312.2 | 362.4 | 369.0 |
| TTFT p50 300 W / 210 W, ms | 90 / 99 | 135 / 148 | 155 / 172 | 198 / 223 | 192 / 204 |

**Card** (5 s samples): 300 W — 296 W avg / 400 W peak, junction 98 / 102 °C, memory 87 / 90 °C, fan
3,710 / 4,035 rpm, sclk 2,941 MHz; 210 W — 210 / 271 W, junction 93 / 97 °C, memory 90 / 94 °C, fan
2,321 / 2,428 rpm, sclk 2,298 MHz. Draft acceptance (server counters): 4.23 / 4.21 tokens per step,
acceptance rate 0.461 / 0.459. GSM8K (200, cot zero-shot, greedy): 0.855 ± 0.025.

**330 W** (the card's maximum cap; `ab.json`, 8 passes, same night, against the 300 W arm of the A/B below):
prefill +8.3 / +7.9 / +6.8 / +5.5 / +4.3 % at 2k / 8k / 16k / 32k / 64k, step gap 34.81 → 34.61 ms,
326 W avg / 422 W peak, junction 102 / 106 °C, fan 4,198 / 4,508 rpm.

### A/B against 0.3.6 (2026-09-25, 300 W, production arguments, ab.json)

| | 0.3.6 | 0.4.0 | 0.3.6 repeat |
|---|---|---|---|
| prefill 2k / 64k tok/s | 3,180 / 2,516 | 3,169 / 2,514 | 3,176 / 2,514 |
| weighted decode tok/s | 130.7 | 130.5 | 130.6 |
| step gap, 37 / 28k-token prompt | 35.04 / 36.95 ms | 35.08 / 36.93 ms | |
| KV pool (warm) | 384,316 | 384,316 | |
| GSM8K 200 | 0.845 (0.3.5) | 0.855 | |

vLLM 0.30 is performance-neutral on this stack (prefill within ±0.3 %, the same image repeats within 0.3 %).
NVFP4 loads and serves on 0.4.0 as well.

### Prefill chunk size and context length (2026-09-25, 300 W, prefill-only sweeps)

| `--max-model-len` / `--max-num-batched-tokens` | Prefill 2k / 8k / 16k / 32k / 64k tok/s | KV pool | conc 1 / 4 / 8 |
|---|---|---|---|
| 262144 / 2048 | 3,363 / 3,326 / 3,196 / 2,926 / 2,512 | 384,316 | – |
| 131072 / 2048 | 3,379 / 3,344 / 3,212 / 2,936 / 2,520 | ~350k | 116 / 350 / 413 |
| 131072 / 4096 | 3,375 / 3,444 / 3,351 / 3,057 / 2,613 | ~326k | 122 / 340 / 396 |
| 131072 / 8192 | 3,367 / 3,486 / 3,290 / 3,022 / 2,579 | ~282k | 118 / 345 / 406 |

A prefill-only sweep reads ~5 % higher than the prefill phase of a full BetterBench run (the card is cooler
when it starts); compare within one table only.

## 0.3.1 (2026-09-23)

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

## Other checkpoints on one R9700 (0.3.6, 2026-09-24)

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

## History: 0.2.9 (2026-09-21)

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

## History: 0.1.4 – 0.1.6 (2026-09-15/16)

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
