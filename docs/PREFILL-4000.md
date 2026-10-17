# Prefill 4,000 tok/s: spike analysis and GPU-window plan

Status: **unmeasured** — no GPU was reachable while this was written (2026-10-17).
Every number marked *measured* comes from the reference runs in [README.md](../README.md) and
[BENCHMARKS.md](BENCHMARKS.md); the arms below are the open measurement steps.

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
   | down | 5,120×17,408 | 23 % | **none** — `silu_mul_quant`'s N ≤ 18,432 limit excludes the 34,816-wide gate_up strip |
   | o_proj / out_proj | 5,120×6,144 | 16 % | **none** — `gdn_norm_quant` has no tiled variant |

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

## What this spike deliberately did not do

- No image change, no `VERSION` bump: every config lever is already on in production, and
  changing the recommended launch without a same-window A/B would trade the KV pool (−7 %)
  for a gain that does not reach the 4,000 target.
- No kernel edits: the tiled `down` / `gdn_norm_quant` producers are HIP work that needs the
  `bench_prefill_cells.py --check` gate and the GPU window before it can be claimed.
