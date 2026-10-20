# Prefill 4,000 tok/s: the non-GEMM producers (Hebel #4)

Status: **measured, bit-identical, small** (2026-10-18, 210 W, production launch line).
Parent document: [PREFILL-4000.md](PREFILL-4000.md). Full report with all logs: the research tree
`paiton-vs-sly/prefill-night/nongemm/REPORT.md` (not part of this repo).

## Result

After the prefill GEMMs, the three fused norm/act-quant producers (`silu_mul_quant`,
`add_rms_quant`, `gdn_norm_quant`) were the largest remaining block: **46.9 ms of 570 ms** at a
1,514-token prompt. They ran at 326-418 GB/s instead of ~570. `RADIANCE_FUSED_NQ_REG=1` makes them
register-resident and takes **13.4 ms of 550 ms (-2.4 %)** off the same request, bit-identically.

| request (production args, 210 W) | stock | `RADIANCE_FUSED_NQ_REG=1` | delta |
|---|--:|--:|--:|
| 1,514 tok (2 steps) | 550.2 ms | 536.9 ms | **-2.43 %** |
| 2,048 tok = 880+1168 | 823.7 ms | 806.1 ms | -2.13 % |
| 3,520 tok = 2x1760 | 1,397.2 ms | 1,365.3 ms | -2.28 % |
| 15,800 tok = 9x1760 | 6,540 ms | 6,394 ms | -2.2 % |
| producer class at 1,514 tok (profiled) | 46.9 ms | 29.1 ms | -17.7 ms |

Standalone on real in-situ activations (ABBA, 50 reps, hipEvent):

| kernel | M=880 | M=1514 | M=1760 |
|---|--:|--:|--:|
| `silu_mul_quant` | 1.63x (326 → 531 GB/s) | 1.69x (332 → 561) | 1.71x (334 → 571) |
| `add_rms_quant` | 1.34x (408 → 548) | 1.40x (417 → 586) | 1.38x (418 → 576) |
| `gdn_norm_quant` | 1.56x (340 → 530) | 1.58x (357 → 563) | 1.59x (359 → 571) |

## Cause

All three producers kept the per-thread row slice in a *runtime-indexed* scratch buffer:
`hipcc -Rpass-analysis` reports 304 B/lane of scratch in `silu_mul_quant` and 176 B/lane in
`add_rms_quant` / `gdn_norm_quant` — every element is stored and reloaded. The REG variant uses a
constant trip count plus a guard, so the slice stays in VGPRs: scratch 0 B, VGPR 91-94, occupancy
unchanged (16 waves/SIMD). Bandwidth, not occupancy, was the limit.

## Knob

`RADIANCE_FUSED_NQ_REG` (kernel default **0** = stock, `serve-mxfp4.sh` ships `1` since 1.5.2 -- it is
part of the measured stack; `=0` restores stock, `set_nq_reg()` overrides it at runtime for in-process
A/B). Decode is unaffected (M<=64 was already fine: 1.05-1.4x, bit-identical).

`RADIANCE_FUSED_NQ_ART` (128/512/1024, default 0 = launcher's choice, only with NQ_REG) sets the
producer workgroup size. Measured at 512: `silu_mul_quant` +2.2 … +4.3 %, `gdn_norm_quant` -0.4 …
-0.6 % (noise), all bit-identical (both kernels only reduce a max, whose result is order-exact).
**Not worth a default change**; left off.

`RADIANCE_MXFP4_AT_MSPLIT` (M-split of the long prefill GEMMs) removes all 164 post-GEMM stalls
(20.1 → 0 ms) but the GEMMs get slower: +1.5 % (512 rows) / +2.1 % (256 rows) wall at 1,514 tok.
Not a candidate.

## Numerics

Bit-identical throughout: every eager producer call with M > 64 was relaunched on its real inputs
into private buffers with the knob off and on, and the q / scale / residual bytes compared
(`ci/prefill_nq_ab.py CHECK=1`) — `diff=0` over 12 (kernel, M) cells and 1,434 calls, reproduced in
three independent processes; greedy 128-token decode identical as well.

## Reproduce

```bash
# build the .so from this branch (CPU-only, no GPU device, ~3 min)
OUT=/root/nongemm/so-night ci/build_so_nq.sh

# in-process A/B on the production launch line (one process, arms per request, ABBA)
TAG=fin SO=/root/nongemm/so-night LENS=1514,2048,3520 ORDER=01100110 CHECK=1 ci/run_nq_ab.sh

# standalone producer bench on the dumps that CHECK=1 wrote
bash bench.sh real1 --ms 880,1514,1760 --reps 50

# server A/B in a window (arms G0 = control with the same .so, G = knob on)
SHORT=1 G_DIR=/root/nongemm/so-night ARMS="G0 G G0 G" OUT=/root/prefill-ab-nq ci/prefill_ab.sh
```

## Not verified

- 300 W: the whole table above is 210 W. A 300 W night script exists
  (`paiton-vs-sly/prefill-night/nongemm/night300.sh`, arms G0/G, ~1.5k / 16k / 47k); it has not run.
- 47k prompt: only 15.8k was measured in situ. The producers are per-token bandwidth work, so the
  relative gain should hold, but it is extrapolation.
- Concurrency > 1 (batching mixes M per call), prefix caching on, and other model checkpoints.
- Decode at > 128 generated tokens and with speculative decoding.
