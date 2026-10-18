# Prefill 4,000 tok/s: spike analysis and GPU-window plan

Status: **measured once at 210 W** (2026-10-17, section "Result of the first A/B window" below).
The 300 W window the plan asks for is still open. Every other number marked *measured* comes from
the reference runs in [README.md](../README.md) and [BENCHMARKS.md](BENCHMARKS.md).

## Goal and baseline

Bring the 1.5k-prompt prefill from **3,024 tok/s** (1.5.1 reference, 300 W, production arguments)
to **4,000 tok/s** (+32 %).

Feasibility: Paiton's MXFP4 setup reaches 3,691 / 3,831 / 3,871 / 3,750 / 3,455 tok/s at
1.5k / 6k / 12k / 24k / 64k on the same card class (BENCHMARKS.md → Recipe step 1). The
4,000 target sits slightly above their 12k figure, so it is realistic — but it is **kernel
headroom, not config headroom** (below).

## Lever audit: the four candidates are already on

| Lever | Status in 1.5.1 production | Measured effect | Effect at 1.5k |
|---|---|---|---|
| `--max-num-batched-tokens` 2048 → 4096 | off (2048 is the 262k-context default) | +3.5 … +4.8 % at 8k–64k prompts, at `--max-model-len 131072`; 8192 no faster, costs ~68k KV tokens; 4096 costs ~24k (−7 %) | ~0 %: a 1.5k prompt fits one 2048 chunk, so chunking never triggers |
| `RADIANCE_R4D_PREFILL_MIN_Q` (default 512) | on (default) | R4D_HYBRID routes q ≥ 512 to libr4d prefill: +3 / +13 / +23 % at 8k / 32k / 64k | 0 %: 1,536 ≥ 512, the prompt already goes to the R4D kernel; lowering the threshold only helps prompts < 512 tokens |
| `RADIANCE_MXFP4_A_TILED_MIN_M` (513) | on (README quickstart) | +11–13 % prefill, bit-identical | already included in 3,024 |
| `--enable-prefix-caching` | on | only repeated prefixes; the reference prefill uses unique prompts | 0 % for the target metric |

Conclusion: the measured config headroom stacks to at most ~+8 %
(chunk 4096 +3.5–4.8 % on long prompts, TETRA KV4 +3–4 % at chunk ≥ 3456),
i.e. **~3,300 tok/s — not 4,000**. The remaining ~24 % is in the prefill kernels.

## Structural gaps found in the code

1. **Fragment-tiled GEMM covers 64 % of the prefill GEMM FLOPs.**
   `sly/mxfp4/bench_prefill_cells.py` (per-layer shapes, TP=1):

   | GEMM | N×K | share of GEMM FLOPs | tiled producer |
   |---|---|---|---|
   | gate_up | 34,816×5,120 | 47 % | `add_rms_quant` (from M=513) |
   | qkvz / qkv | 16,384/14,336×5,120 | 17 % | `add_rms_quant` (from M=513) |
   | down | 5,120×17,408 | 23 % | `silu_mul_quant` (from M=513): its N is the 17,408 half of the 34,816-wide gate_up row, inside the N ≤ 18,432 limit — no gap (corrected; verify with `bench_prefill_cells.py --check`) |
   | o_proj / out_proj | 5,120×6,144 | 16 % | tiled `gdn_norm_quant` behind `RADIANCE_MXFP4_TILED_OPROJ=1` (default 0, GPU-window pending) |

   A tiled `down` (a MAXG-16 silu producer or a second-stage producer) and a tiled
   `gdn_norm_quant` would extend the +11–13 % tile gain to the remaining 36 % of GEMM FLOPs —
   order +4 % prefill if the tile gain transfers, more if the row-major path is worse than that.
2. **Attention and GDN prefill kernels.** The BENCHMARKS Paiton gap "grows with depth
   (≈ +12 % at 2k, ≈ +27 % at 64k)" after the chunk correction — attributed to their
   attention/GDN prefill kernels. Ours: libr4d paged prefill attention (`R4D_HYBRID`) and the
   narrow-state bf16 GDN kernels (`sly/gdn/`). This is the largest unmeasured block.
3. **Power.** 210 W (the 24/7 setting) costs ~20 % prefill against 300 W (measured, 0.4.0).
   At the production 210 W cap the 1.5k prefill is ≈ 2,400 tok/s, so **4,000 tok/s is only
   reachable in a 300 W window** — and at 210 W the target is +67 %, not +32 %.

## GPU-window measurement plan

Script: [`ci/prefill_ab.sh`](../ci/prefill_ab.sh) (production arguments, two starts per arm,
full BetterBench run per arm, control first and last, power-cap check read-only).

Prerequisites before the first arm: owner approves the window; 300 W + firmware fan curve set;
restore 210 W / 2,800 rpm after the window. Second start is the measured one (fresh compile on
the first). The KV pool must be read from the log per arm (`*.kvlog`); for the control arm it
must be 436,906 tokens (439,031 on 1.5.1).

| # | Arm | Question it answers |
|---|---|---|
| 1 | C: production 1.5.1 (262k, chunk 2048) | window baseline; must land at ~3,024 ± 3 % at 1.5k |
| 2 | A: chunk 4096 @ 262k | does the +3.5–4.8 % hold at 262k (it was measured at 131k)? KV pool cost |
| 3 | B: chunk 4096 @ 131k | the README "less context" option as a production candidate |
| 4 | M: `RADIANCE_R4D_PREFILL_MIN_Q=256` | is there TTFT headroom below 512 (short-prompt path) |
| 5 | T0: `RADIANCE_MXFP4_A_TILED_MIN_M=0` | re-verify the +11–13 % tile gain on the 1.5.1 stack (bit-identical check via `sly/mxfp4/bench_prefill_cells.py --check`) |
| 6 | K: `KVCACHE_DTYPE=kv4` + chunk 4096 | TETRA +3–4 % prefill and the 1.75× KV pool in the same window |
| 7 | C2: control again | window drift; discard the window if C2 deviates > 3 % from C1 |

Success criteria: any arm ≥ 3,600 tok/s at 1.5k (within 8 % of Paiton's 12k figure) → follow-up
with production conditions (KV pool, GSM8K 200 against `ci/accept/baselines/`), then README /
CHANGELOG per AGENTS.md. If no arm gets above ~3,300, the gain is in the kernels:

- `sly/mxfp4/bench_prefill_cells.py` — cell-level GEMM sweep; the `down` and `o_proj`/`out_proj`
  row-major gap is the first kernel work item (a tiled producer each).
- `sly/bench_decode_attn.py` (2D prefill/extend mode) — attention cells at the 1,536-q_len shape.
- GDN prefill: the `sly/gdn/` HIP kernels against the Paiton gap at 2k.

## Result of the first A/B window (2026-10-17, 210 W)

Conditions: image `vllm-sly-radiance:1.5.1-rocm10.1`, production unit `docker-vllm7.service`
(profile `swift-mxfp4`, `/root/swift-heretic-gptq`) with exactly one delta per arm, **210 W cap**
(the owner kept the production cap; the plan asks for 300 W), BetterBench prefill-only sweep
(2 warmup + 8 runs per depth, unique prompts, no prefix cache). Arms ran C A B M T0 K C2 in one
window, 15:15–16:39 UTC. BetterBench depths are targets at ~4 chars per token; the server reports
the real prompt tokens, which are ~25 % lower. Columns are real prompt tokens, median tok/s:

| Arm | 1.5k | 11.8k | 15.8k | 47k | KV pool (warm start) |
|---|--:|--:|--:|--:|--:|
| C   control (262k, chunk 2048)        | 2,605 | 2,483 | 2,379 | 1,990 | 439,031 |
| A   chunk 4096 @ 262k                 | 2,585 | 2,627 | 2,457 | 2,084 | 423,463 |
| B   chunk 4096 @ 131k                 | 2,580 | 2,630 | 2,455 | 2,072 | 384,597 (131k context) |
| M   `R4D_PREFILL_MIN_Q=256`           | 2,562 | 2,360 | 2,285 | 1,940 | 439,031 |
| T0  `A_TILED_MIN_M=0`                 | 2,283 | 2,176 | 2,094 | 1,775 | 439,031 |
| K   KV4 (TETRA) + chunk 4096          | 2,576 | 2,611 | 2,556 | 2,115 | 739,967 |
| C2  control again                     | 2,525 | 2,407 | 2,319 | 1,942 | 439,031 |

Change against C (same window; the conclusions below were also checked against the mean of C and C2):

| Arm | 1.5k | 11.8k | 15.8k | 47k |
|---|--:|--:|--:|--:|
| A  | −0.8 % | +5.8 % | +3.3 % | +4.7 % |
| B  | −1.0 % | +5.9 % | +3.2 % | +4.1 % |
| M  | −1.7 % | −5.0 % | −4.0 % | −2.5 % |
| T0 | −12.4 % | −12.4 % | −12.0 % | −10.8 % |
| K  | −1.1 % | +5.2 % | +7.4 % | +6.3 % |

What the numbers say:

- **4,000 tok/s is not reached.** The best 1.5k value is the control, 2,605 tok/s at 210 W. No arm
  moved the 1.5k prefill outside the window drift; the lever audit above (chunking never triggers
  for a 1.5k prompt) holds. The gap to 4,000 is +54 % at 210 W; at 300 W the plan expects ~3,000
  for C, i.e. the gap there is the +32 % named at the top. That 300 W window is not measured.
- **Window drift: C2 is 3.1 % below C at 1.5k (2.4–3.1 % at the other depths).** The plan's own
  rule discards a window above 3 %. Read the table as one-directional drift (the card warms up;
  junction peaked at 90 °C, power stayed at the 210 W cap): effects below ~3 % (A, B and K at 1.5k,
  M at 1.5k) are noise. Effects of +4 … +8 % at 11.8k–47k (A, B, K) and −10 … −12 % (T0) are larger
  than the drift, and they hold against the C/C2 mean as well. One window, one sample per arm:
  repeat before changing any recommendation.
- **A and B (chunk 4096) help long prompts only**: +4 … +6 % at 11.8k–47k, nothing at 1.5k.
  The KV pool drops 439,031 → 423,463 tokens (−3.5 %) at 262k context, and B gives 384,597
  tokens at 131k context. The plan's estimate of −7 % came from a 131k comparison; at 262k it is −3.5 %.
- **K (KV4 + chunk 4096)** gives the same +5 … +7 % on long prompts and a 739,967-token pool
  (1.69× C). It is not faster than A at 1.5k and 11.8k; at 15.8k and 47k it is +4 % / +1 % over A.
- **M (`R4D_PREFILL_MIN_Q=256`) does not help prefill**: −2.5 … −5 % against C, mostly inside the
  drift for 1.5k; its target (TTFT of prompts below 512 tokens) was not measured here.
- **T0 (`A_TILED_MIN_M=0`) is 11–12 % slower than 513**, consistent with the +11–13 % that the tiled
  path brings at M ≥ 513 (0.3.0). With 0 the tiled path also serves small M, which is slower.
  Keep 513.

Open: 300 W window; repeat C/A/K in an ABBA order; TTFT for prompts below 512 tokens (arm M).

Raw files: `summary.tsv`, per-arm BetterBench JSON, launch-line diff and KV log; they live in the GPU
window directory `/root/prefill-ab-20261017` on the 2408 host (copy in the agent scratch space).

## What this spike deliberately did not do

- No image change, no `VERSION` bump: every config lever is already on in production. The first
  window (above) shows chunk 4096 trading 3.5 % KV pool for +4 … +6 % on long prompts only, and
  none of it reaches the 4,000 target.
- No kernel edits: the tiled `down` / `gdn_norm_quant` producers are HIP work that needs the
  `bench_prefill_cells.py --check` gate and the GPU window before it can be claimed.

## Result of the night windows (2026-10-18, 210 W and 300 W)

Same harness as above (`ci/prefill_ab.sh`, SHORT mode: prefill-only sweep, 1 warmup + 3 runs, one start per
arm, median tok/s at real prompt tokens). Arms are built from this branch and bind-mounted over the 1.5.1
image (`O_DIR`): O = tiled `gdn_norm_quant` (`RADIANCE_MXFP4_TILED_OPROJ=1`), T4 = `RADIANCE_MXFP4_TN4_MIN_M=513`,
OT = O + T4, OTA = OT + chunk 4096, OTR = OT + `R4D_ATTN_FP8=3` (changes numerics, report only). Control C and C2
bracket each window; compare only inside one window.

**300 W (fan released, junction <= 72 C, power held at 300-302 W), window 1:**

| Arm | 1.5k | 15.8k | 47k | KV pool |
|---|--:|--:|--:|--:|
| C   control | 3,318 | 3,201 | 2,639 | 439,031 |
| OT  | 3,426 (+3.3 %) | 3,303 (+3.2 %) | 2,710 (+2.7 %) | 437,836 |
| OTA | 3,427 (+3.3 %) | 3,405 (+6.4 %) | 2,823 (+7.0 %) | 423,463 |
| OTR | 3,444 (+3.8 %) | 3,417 (+6.7 %) | 2,960 (+12.2 %) | 439,031 |
| C2  | 3,305 (-0.4 %) | 3,197 | 2,614 (-0.9 %) | 437,836 |

Window 2 repeats OTA (3,421 / 3,405 / 2,814) and OT (3,423 / 3,177 / 2,635; the 15.8k and 47k OT runs contain one
slow run each, min 1,998 and 2,626, so read them as noise), C2 3,300 / 3,195 / 2,543.

**210 W (5 windows, same arms):** OT +4.5 % at 1.5k and +6 % at 47k; OTA +9 % at 47k; OTR +15 % at 47k.
Decode sample (4 x 256 tokens) is unchanged for every arm (63.4 tok/s). GSM8K-200 (C / OTA / OTR) is 0.845 for all
three (flexible-extract; strict-match is 0 because of the answer format).

What the numbers say:

- **4,000 tok/s is not reached.** Best 1.5k value is OTR at 3,444 tok/s at 300 W (86 % of the goal);
  without the numerics-changing arm 3,426. The measured config and the one GEMM change give +3-4 % at 1.5k.
- The tiled producer for `o_proj`/`out_proj` is correct (0 WRONG, tiled == row-major byte-exact in the cell bench) and
  its cell gain is +10-16 % on those two shapes (16 % of GEMM FLOPs), end to end it is small (about +2 %).
- `down` already uses the tiled producer (`silu_mul_quant` gets N = 17,408, below the 18,432 limit); the earlier
  "36 % of FLOPs without tiled producer" statement is wrong, the real gap is 16 %.
- `TN4_MIN_M=513` gives +3-5 % GEMM at M 513..1536 and -1..-2 % at >= 2048 (cell bench); the end-to-end gain is
  about +2-3 % at 1.5k.
- Chunk 4096 (OTA) adds +3 % at 15.8k and +4 % at 47k and costs 3.5 % KV pool, nothing at 1.5k.
- Cell-bench GEMM efficiency is 158 TF/s at M = 1,536 and 171 TF/s at M = 4,096. Ablations show the floor of
  WMMA + LDS reads + barrier is 259-281 TF/s; the W stream (global load -> dequant -> LDS store -> barrier) overlaps
  poorly with the WMMA. A cheaper dequant alone is worth at most about 5 %. The lever is more overlap
  (producer/consumer waves, double-buffered LDS), which is real kernel work.
- Smaller row tiles (TM = 2/3, `RADIANCE_MXFP4_AT_TMS`) are slower at M >= 768; TM = 3 helps only M ~ 513..576
  (+17..28 %, padding effect). Default off.
- Not comparable with the README 3,024: that is a full-run prefill phase; the sweep here reads about 5 % higher
  and the 300 W control in the same window is 3,318.

New knobs (all default off, default path unchanged): `RADIANCE_MXFP4_TILED_OPROJ`, `RADIANCE_MXFP4_AT_TMS`,
`RADIANCE_MXFP4_AT_TMS_MAX_M`. Existing: `RADIANCE_MXFP4_TN4_MIN_M`.

## Verified floor knobs: SKIPPAD and the PSW weight layout (cell bench, 210 W)

Two further A-tiled GEMM changes, both **default off**, both **bit-identical** to the current kernel
(sha256 DIGEST of the bf16 output), and both measured only in the cell bench so far. Full report,
raw logs and the 300 W night script: `research/paiton-vs-sly/prefill-night/floor/` (REPORT.md,
night300.sh). Gate: `ci/bench_at_psw.py`, layout self-check `ci/test_at_psw_layout.py`.

- **SKIPPAD** (`RADIANCE_MXFP4_AT_SKIP_PAD=1`, commit 48c5fa1): a wave whose M-fragments all lie past
  M is pure tile padding; it still stages W and meets every barrier but skips its A loads, LDS reads
  and WMMAs. Only dispatched for WPERM e4m3 with a non-zero padding remainder.
- **PSW weight layout** (`set_at_psw`, no env var, commit c843649): preshuffled nibbles + pre-folded
  per-row scale, moved out of the K loop into the load. Same 32 bits, so bit-identical. Removes
  20-25 of the 225 loop instructions (TN2; 190 -> 165/170 for TN4).

| cell (210 W, ABBA) | SKIPPAD | PSW (mode 2) | both |
|---|--:|--:|--:|
| gate_up M=880 | 0.902-0.906 | 0.990-0.991 | 0.893 |
| down M=880 | 0.903-0.909 | 0.986 | 0.886-0.888 |
| gate_up M=1514 | -- | 0.989-0.990 | 0.987 |
| down M=1514 | -- | 0.980-0.982 | 0.990-0.992 |
| gate_up M=1760 | 0.996-0.997 | 0.982-0.985 | 0.987 |
| down M=1760 | 1.002-1.007 | 0.978-0.981 | 0.988-0.991 |
| gate_up M=4096 | 1.00 (no dispatch) | 0.986-0.987 | -- |
| down M=4096 | 1.00 (no dispatch) | 0.978-0.981 | -- |

Ratio = variant time / control time, median of 2-4 ABBA pairs per cell; board power 209.5 W in every
leg. 384 DIGEST/PDIGEST comparisons (gate_up/down/qkvz, M 513-4096, TN2 and TN4 paths, merged-linear
p1/p2 split): 0 WRONG. Resources: <= 207 VGPR and 0 spills on every prefill instantiation (+4 VGPR
for SKIPPAD, +/-0 for PSW); the 12 default instantiations compile to byte-identical bodies
(`__hip_cuid` normalised).

SKIPPAD across M follows the dead-wave fraction of the last 256-row block (4 waves of 64 rows):
M=700 -> 0.958-0.963 (1 wave dead), 880 -> 0.902-0.909 (2 of 4), 1100 -> 0.920-0.922, 1760 ->
0.996-1.007 (no dead wave), 2000 -> 0.998-1.003, 4096 -> 1.00 (not dispatched). So the knob pays
exactly where a wave is pure padding and costs at most 0.7 % elsewhere (branch only).

Reading: with GEMM at 74.3 % of the 1.5k prefill, SKIPPAD/PSW/both are worth **+0.7-1.0 %** at
M >= 1514 (where the last M-block is only partly padding) and **+7-8 %** for chunks whose M lands at
880. PSW's clock drops 1.3-1.9 % at the fixed cap while time drops 1-2 % -- at 210 W the kernel is
energy-bound, so the win is the 20-25 fewer issued instructions, not the clock.

**PSW is not loader-ready.** The weight bytes are global to every consumer: the decode kernel reads
the same `layer.weight` and needs the same PSW nibble unpack (plus the same `set_at_psw` gate) before
a production start is possible; the pack itself would be in place at load (same shape/dtype, no extra
memory) and `weight_scale` becomes `8 * clamp(Wref - Ws, 0, 15)`. See REPORT.md section 6.

Not measured: 300 W, any server A/B, TTFT, GSM8K. The night script for the 300 W server A/B exists
and has not been run.
