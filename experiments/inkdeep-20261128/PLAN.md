# Inkling-2-Mini deep-context decode: attribution + hill-climb (2026-11-28)

Card t_11ca1edc. Branch `perf/inkling2-deep-decode` off origin/main c50c3ba (1.15.1).
Steering 2026-11-28 (Timon): brief point 8 (upstream radiance-2 comparison) is STRUCK —
separate card after the prod switch; no GPU/build budget for it here. Reddit values are
literature reference only.
Host: LXC 2408 (.51), R9700 gfx1201, 210 W, prod unit docker-vllm7 (image 1.15.1-rocm10.1).
Raw data + ledger: host `/root/inkdeep-20261128/`, mirrored here (committed).

## Baseline (MESSUNG, ab-1151 window 2026-11-28, 210 W, same scripts)

| metric | 1.14.2 | 1.15.1 | delta |
|---|---|---|---|
| MIA prose / code tok/s | 177.9 / 227.2 | 192.9 / 262.5 | +8.4 % / +15.5 % |
| ctx 8k / 64k / 150k tok/s | 122.6 / 106.7 / 81.9 | 126.5 / 98.2 / 71.0 | +3.2 / -7.9 / -13.3 % |
| steps/s 8k / 64k / 150k | 41.8 / 37.5 / 31.8 | 45.4 / 36.6 / 27.9 | +8.6 / -2.4 / -12.3 % |
| AL 8k / 64k / 150k | 2.94 / 2.88 / 2.59 | 2.79 / 2.72 / 2.56 | -5.1 / -5.6 / -1.2 % |

Decomposition (RECHNUNG, tok/s = steps/s x AL): the 64k loss is mostly **AL** (drafter),
the 150k loss is mostly **steps/s** (step time, grows with depth -> per-key attention cost).
Two distinct causes, not one.

## Suspects (code diff cd29507..c50c3ba, no vLLM bump in Dockerfile)

1. Global decode tile: 1.14.2 BLOCK_N=64/warps=4 -> 1.15.1 BLOCK_N=32/warps=8
   (`RADIANCE_INKLING2_DECODE_BLOCK_N/_WARPS`). Depth-growing cost -> prime suspect for 150k.
2. Local SWA plan+tile: 1.14.2 stock BLOCK_N=128/warps=4, splits=cdiv(1023,128)+1=9 ->
   1.15.1 BLOCK_N=32/warps=4, splits=min(33, cdiv(1023,32)+1)=33
   (`RADIANCE_INKLING2_LOCAL_BLOCK_N/_LOCAL_WARPS`). Depth-independent, candidate for 8k gain.
3. MXFP4 drafter default on (`RADIANCE_DFLASH_MXFP4=1` with MXFP4_DENSE=1, RTN clip):
   AL -5 % at 8k/64k. bf16 drafter in 1.14.2.
4. VEC_KV b128 loads: measured NEUTRAL end-to-end (ab-1151 B arm). Note: VEC_KV=0 does NOT
   revert tiles/plan -> that arm did not test suspects 1/2. Do not reopen VEC_KV alone.
5. ADAPTIVE_K: entrypoint default "3,5,7" UNCHANGED since 1.14.2; 1.15.0 only added the
   RADIANCE_ADAPTIVE_K alias + a proof line. Low prior, one bounding arm.
6. DECODE_SPLITS=0 collapses long context (ab-1151 C arm) -> split plan carries, NOT the cause.
   Do not reopen.

## Ceilings (RECHNUNG, estimates; measured BW 563 GB/s, peak 640)

Config: 24 layers, loop core 6..18 x3 passes = 48 layer visits/step; MoE 128 experts top-8,
expert = 3x2304x1152 MXFP4 ~ 4.23 MB; routed union ~20 (w=4) .. 29.6 (w=8) experts/visit;
8 global attn calls/step (layers 5,23 x1; 11,17 x3), KV fp8 8 heads x 128 x 2 = 2048 B/token/layer;
40 local visits x window 1024; lm_head int4 0.23 GB; drafter bf16 ~0.5 GB vs MXFP4 ~0.15 GB (est).

| ctx | bytes/step (est) | floor @563 GB/s | measured step 1.15.1 | eff |
|---|---|---|---|---|
| 8k | ~5.4-7.4 GB | ~10-13 ms | 22.0 ms | ~50 % |
| 64k | ~6.5-8.5 GB | ~12-15 ms | 27.3 ms | ~50 % |
| 150k | ~7.9-9.9 GB | ~14-18 ms | 35.8 ms | ~45 % |

Not at the bandwidth wall; kernel/launch efficiency headroom exists, largest at depth.
Acceptance needs step <= ~27.7 ms @64k (AL 2.88) and <= ~32.4 ms @150k (AL 2.59).

## Hill-climb literature (steering 2026-11-28, ONLY after attribution+fix; clean-room, no forced arms)

Notes: /opt/data/analysis/inkling2-mini/references-external.md
- radiance PR #586 split-KV decode: live-bound merge (8 tiles/split), 4-K-word prefetch then
  score, butterfly wave-sum over 8 keys, int8 K+V in one load; 250k kernel 1.17->0.92 ms,
  512k 4.06->1.90 ms. Check whether our split-KV path tolerates similar prefetch/reduction/load
  improvements (technique description only, no code/identifier takeover).
- saber-engine (Rust/R9700): resolved plan instead of env namespace; W4A8-iu8 on int8 matrix
  cores (per-row int4, per-token int8) — technique idea only.
- Reddit Tangenius: long-context draft-verify fastpath gated >=4 conversations, switch opens
  it for c1 (+12-21 % @46K-95K) — check whether our tree has an equivalent concurrency gate.
- losingle/basalt fork: Blackwell/CUDA only, ignore.

## Window 1 arms (one knob each; C first and last; ~6 min/arm)

| arm | image | env delta | question |
|---|---|---|---|
| M0 | 1.15.1 | microbench --sweep, no server | which (BLOCK_N,warps) wins per depth, global+local |
| C1 | 1.15.1 | none | control |
| OLD | 1.14.2 | none (old compile cache) | same-window reference |
| T | 1.15.1 | DECODE_BLOCK_N=64 DECODE_WARPS=4 LOCAL_BLOCK_N=128 LOCAL_WARPS=4 | old tiles/plan recover 150k? |
| D | 1.15.1 | DFLASH_MXFP4=0 | bf16 drafter recover AL? |
| A | 1.15.1 | ADAPTIVE_K="" INKLING2_ADAPTIVE_K="" | bound adaptive-k share |
| C2 | 1.15.1 | none | drift control |

Suites per arm: mia (400 tok, 3 reps) + ctx (8k/64k/150k). Validity per arm: result files
non-empty, `docker exec env` proof, hooks.txt, MemAvailable >= 14 GB before start (A2 died
at 11.5 GB available during weight load). Drift floor ~3 %.

## Acceptance (hard, vs 1.14.2 control in the SAME window)

- ctx 64k >= ~104 tok/s, 150k >= ~80 tok/s; MIA prose >= ~185, code >= ~245; ctx 8k >= ~122.
- Quality: GSM8K paired vs 1.15.1 (baseline twice, noise floor), greedy gate, needle 64k/150k
  exact-match (lc.py). No loss above noise.
- Deliverables: open PR (CI green, no merge), provisional ghcr image (pull verified), ledger.

## Rejected / do not reopen

- VEC_KV=0 alone: neutral (ab-1151 B).
- DECODE_SPLITS=0: collapses long context (ab-1151 C).
- (append after each arm)
