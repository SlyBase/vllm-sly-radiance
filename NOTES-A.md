# NOTES-A: MXFP4 W4A8 GEMM band selection for M 16..512 (lever 2)

Status: implemented behind a new knob, default OFF. Compiled for gfx1201 without a GPU (0 spills in the new
kernels); NOT run on a GPU. All performance statements below are expectations, not measurements.

## 1. What serves every M today (production env, `sly/mxfp4/band_map.py` reproduces it)

Env: `DECODE_MAX_M=128`, `WPERM=1`, `A_TILED_MIN_M=513`, `TN4_MIN_M` unset (= 2048), `TUNE16` default on,
`DECODE_NT` off. Split-K scratch 72 MiB (4 x 128 x 36864 floats); a split that does not fit falls to the folded kernel.

| M | gate_up 34816x5120 | qkvz 16384x5120, qkv 14336x5120 | o_proj/out_proj 5120x6144 | down 5120x17408 | in_proj_ba N=96 |
|---|---|---|---|---|---|
| 1-8 | decode d1/b128 | d1/b128 | d4/b128 | d4/b128 | d4/b128 |
| 9-15 | d1/b128 | d1/b128 | d4/b128 (M<=15) | d4/b128 | d4/b128 |
| 16-24 | d1/b128 | d1/b128 | d2/b128 | d4/b128 | d4/b128 |
| 25-64 | d1/b128 | d1/b128 | d2/b128 | d2/b128 | d4/b128 |
| 65-80 | d1/b64 | d1/b64 | d2/b128 | d2/b128 | d4/b128 |
| 81-128 | d1/b64 | d1/b64 | d4/b64 | d4/b64 | d4/b128 |
| **129-512** | **folded TN2, 256-row tile** | same | same | same | same |
| 513-2047 | atiled TN2 (producer tiled) | atiled TN2 | folded TN2 (no tiled producer) | folded TN2 | atiled TN2 |
| 2048 | atiled TN4 | atiled TN4 | folded TN4 | folded TN4 | TN2 |

(d = split-K, b = BK, tm = ceil(M/16) fragments, always the smallest that covers M.) Full per-M table: `python3 sly/mxfp4/band_map.py`.
M never exceeds 2048 (`--max-num-batched-tokens 2048`); "above 2048" only exists as chunk tails, i.e. a prompt of
2048k+r tokens ends with a chunk of r rows, which falls into the table above (r<=128 decode, 129..512 folded).

## 2. Comparison with Radiance's final band map

Tune file read-only from 192.168.178.54 (79 lines). Radiance bands: <=64 decode kernel (libr4d, bk/ks/nt per cell),
128 `dec=1` (wide decode, ks 1 for N>=14336, ks 4 + nt for down/out/N=1024), 256/512/1024/2048 `tn=4` tiled.
The Radiance "cliff" (M 65-128 on a 256-row tile, ~89 ms vs ~35 ms per prefill step) does not exist in vLLM any
more: the decode kernel has served M<=128 since 2026-08-29 (`DEC_MAX_TM=8`, tier7 table in `launch()`), and the
16-64 band got its cell table on 2026-09-16 (`TUNE16`). What I compared cell by cell:

| Radiance cell | vLLM equivalent | verdict |
|---|---|---|
| M=16 N=14336/16384 ks1 | d1/b128 | same |
| M=32 down ks4, gate_up ks1, N=16384 bk128 | d2..d4/b128, d1, b128 | same except down: vLLM d2 at M 25-64 (tier7 re-measurement), Radiance ks4; `bench_decode_cells.py` already covers that cell |
| M=48 down ks4 **nt=1** | d2/b128, plain loads | difference: NT on split-K shapes. vLLM measured NT only in the DECODE_NT global form (-5..-8 % at M<=48 on gate_up/n8192/n7168/down, ~0 at 64) and left it OFF; not in the prod launch. The `nt` bench arm re-measures it per cell. |
| M=128 `dec=1` | decode TM8 d1/b64 etc. | same idea, already there |
| M=256/512/1024/2048 `tn=4` (-6/-12/-8/-2.5 %) | folded TN2 below 2048 | libr4d's tiled kernel. On vLLM's folded kernel TN4 measured **worse** below 2048 (-1.6 % at M=1536, -4.2 % at 1024, -8.8 % at 512, see the comment above `radiance_mxfp4_fp8_gemm_folded`). Different kernels: `tn4` bench arm re-checks M 129..512; `RADIANCE_MXFP4_TN4_MIN_M` already exists. No code change. |

Real gaps left in vLLM:
1. **M 129-512 on the folded kernel** (the only band with no purpose-built kernel): the 256-row tile computes
   up to 255 padding rows: M=129 pays for 256 rows, M=272 for 512. Arises in production in c8 arrival steps
   (prompt of 65-192 tokens + up to 64 verify rows of the other sequences) and for 129-512-token prompts/chunk tails.
2. 129..2047 for down/o_proj/out_proj never gets the tiled path (no tiled producer), unchanged here.

## 3. Change

`sly/mxfp4/radiance_mxfp4_fp8.hip`, `launch_impl()`: new block between the decode band and the folded fallback.
For M in (128, `RADIANCE_MXFP4_WIDE_MAX_M`] it launches `radiance_mxfp4_fp8_gemm_decode<DWN=8, BK=64, KS, TM=9..16,
WPERM, DIRECT, NT, E6=false>`, i.e. the existing split-K decode kernel with 9..16 M-fragments (a template
parameter that was already generic; only the dispatch was missing). Resource usage (hipcc -Rpass-analysis): TM16
= 225 VGPRs, 0 spills, 27.6 KB LDS; TM9 = 150 VGPRs, 19.6 KB LDS. The 11 spilling instantiations in the file are
the pre-existing BK=128/TM8 ones, unchanged.

Split table (`wks`, first cut, to be validated by the bench; `RADIANCE_MXFP4_WIDE_KS` forces 1/2/4):
nblk >= 48 -> 1 (gate_up, qkvz, qkv; Radiance dec=1 also ks1 there), nblk <= 8 -> 4 (in_proj_ba),
K >= 6144 (down/o/out, nblk 40) -> 4 for M <= 192, else 2 (partials 4 x M x N x 4 B would be 21 MB at M=256),
else 1. A split that does not fit the existing 72 MiB scratch falls through to the folded kernel, so no
VRAM is added and the KV pool (384,316 tokens) is untouched. Eligibility: fragment-order weight
(`WPERM=1`), plain MXFP4 (no MXFP6 `wh` plane), N <= 36864; everything else keeps the folded path.
`radiance_mxfp4.py`: reads/validates the same env var, logs `wide decode band M in (128, N] ON|INACTIVE: reason`,
and `MHIST` reports `decode_kernel=wide`.

### Knobs

| Knob | Default | Recommended | Why |
|---|---|---|---|
| `RADIANCE_MXFP4_WIDE_MAX_M` | `0` (off) | 256, or a lower edge if the bench shows the folded tile wins near 256 | wide decode band for M 129..N; clamps to 256 |
| `RADIANCE_MXFP4_WIDE_KS` | `0` (table) | unset | bench sweep only |

Existing knobs the bench also sweeps: `RADIANCE_MXFP4_DECODE_NT` (arm `nt`), `RADIANCE_MXFP4_TN4_MIN_M` (arm `tn4`).
README options table row to add (main session): `RADIANCE_MXFP4_WIDE_MAX_M | 0 | 256 if tests-lessons/A passes | ...`.

## 4. Expected effect (reasoning, not measured here)

Radiance: GEMM sum per prefill step at band 128 -88.7 -> 54.1 ms (-39 %), TTFT 72-256 tokens -40..-10 %, c8 arrivals +2.8 %.
vLLM already has that for <= 128, so the transferable part is smaller: M 129-~192 should lose most of the padding
(up to 49 % of tile rows at M=129); near 256 the folded tile is full and likely equal or better than 16 fragments, which
is why the edge is a knob and why the bench prints the per-step weighted GEMM sum per M. Plausible: TTFT of 129-192-token
prompts and c8 arrival steps a few percent, c8 +0..1 %. If the bench shows no gain at M=160, the answer is "no-go, vLLM
already had the Radiance fix" and belongs in `docs/NOT-ADOPTED.md`.

Numerics: the wide kernel is the decode kernel (same fp8 WMMA, same fold, same epilogue); with split 1 it accumulates
K in the same slab order as the folded kernel, with split 2/4 the fp32 partials sum in a different order (bf16 output may
differ by 1 ulp). Not bit-identical in general; the bench gates on fp32-reference rel < 0.02 and rel-to-cur <= 3e-3.

## 5. Tests (`/root/lessons/A/test.sh`, copy in `tests-lessons/A/`)

`test.sh all`: (1) kernel bench ~4-5 min: arms cur/wide/w1/w2/w4/nt/tn4, 7 shapes, M in {16,24,32,48,64,72,96,128,160,192,256,384,512},
DRAM-fed, exactness gate; prints per-cell deltas, per-step weighted GEMM sums, PASS/FAIL. (2) only if the gate passes (>= 3 % at M=160):
server A/B/A (control, `WIDE_MAX_M=256`, control) with greedy-equality probe, TTFT 72/110/160/256/904, c8 arrivals; decision criteria are
in the script header. Estimated total 15-20 min with three server starts; `ARMS="A B"` cuts one start. Build helper: `build.sh`.

## 6. Risks / unverified
- Never executed on GPU: correctness of TM9..16 at tile edges (M not a multiple of 16, last rows clamped like the existing TM<=8 code) is
  only covered by the bench's reference check.
- 225 VGPRs at TM16 limits occupancy; the table's ks choice for M 193-256 is a guess.
- `MHIST=1` is on in all server arms (small Python set lookup per eager call) to prove the path ran.
- `ci/patch_dryrun.sh` not run: no patch changed. `check_consistency.py` passes (no VERSION/CHANGELOG bump by instruction).
