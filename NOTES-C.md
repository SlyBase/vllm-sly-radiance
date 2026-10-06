# NOTES-C: int4 lm_head at verify batch sizes (lever 3)

## Findings (no GPU used; kernels compiled for gfx1201 without a GPU)

- Which kernel: `RADIANCE_LMHEAD_INT4=1` runs `torch.ops.vllm.rdna_hybrid_w4a16_apply`. M <= 5 would be the HIP
  `wvSplitK_int4_g`, but the head is stored **tiled** (LAYOUT 1, `RADIANCE_LMHEAD_INT4_TILED=1`), which skips the HIP path,
  so M = 8..64 always run the Triton `_triton_w4a16_skinny_fmt_kernel` with the (g128, K 5120, N 248320) rows of
  `_GFX12X_DRAFT_OVERRIDES` (`sly/patch_w4a16_tiles.py`), (BM, BN, BK, warps, stages):
  M<=8 and M<=16: (16,64,128,4,1); M<=32: (32,128,64,8,-); M<=40: (64,128,64,8,-); M<=64: (64,64,64,4,1).
  No split-K entry exists for this shape.
- Spills: **none**. gfx1201 compile of exactly these configs (also confirmed from the production Triton cache on
  2408, 654 skinny/split-K kernels): 139 / 139 / 156 / 196 / 197 VGPR, 0 spills, 0 B scratch. The Radiance problem
  (744 B scratch from MT >= 2) does not exist in the Triton kernel; its loop is a runtime `range` loop (rolled) already.
  Spills do appear in the split-K kernel at some configs (e.g. (64,64,128,4,None) deq 1: 102 spills / 372 B) -- so lean
  configs must be compile-checked; `compile_cfgs.py --cands` does that (all candidates: 0 spills).
- Table timings already in the repo (`patch_w4a16_tiles.py` comment, DRAM-cold, 656 MB/call):
  M 8/16/32/40/64 = 1305 / 1333 / 1585 / 2099 / 2185 us (stock tiles: 2623 / 2709 / 5327 / 2715 / 2824).
  Radiance's LEAN numbers (M32 2177, M48 2928 us) are worse than what vLLM already has. Floor = 656 MB at ~560 GB/s
  = 1.17 ms: M <= 16 is at the floor, M 40/64 are 2x above it (VALU-bound: scale multiply per dequantised weight).
- Drafter head: DFlash2 has no own head. `_maybe_share_lm_head` hands the **target's lm_head** (the int4 one) to the
  drafter; `compute_candidates` -> `get_top_k_tokens(self.lm_head, ...)`. So per step the head runs twice:
  target verify (seqs x (k+1) = 64 rows at c8) and drafter candidates (seqs x k = 56 rows, bucket 64). At c8 both are
  bucket 64 = ~2 x 2.19 = 4.4 ms of a ~45-55 ms step (8-10 %); at c1 M = 8 -> bucket 8 (1.3 ms x 2).
  Radiance's c8 head share was 4.52 ms: same order, which is why lever 3 looked worth checking.

## Change

`sly/mxfp4/radiance_lmhead_int4.py`: knob **`RADIANCE_LMHEAD_INT4_LEAN=0|auto|1`** (default 0 = stock tiles = control),
`RADIANCE_LMHEAD_INT4_LEAN_CFG="32=bm,bn,bk,warps,stages,split_k,mode,deq,unpack[,kstep];40=...;64=..."`.
When on, `install_lean()` (called from `process_weights_after_loading`) writes split-K-table entries for
(128, 5120, 248320, bucket); `triton_w4a16_skinny_fmt_gemm` consults that table before the tile table, so no new custom op,
no graph/compile change (the config is chosen from M in Python at trace/capture time, as for the other W4A16 shapes).
`auto` = buckets 32/40/64 (M <= 16 keeps the stock tiles); `1` = also 8/16 (A/B only).
Defaults (spill-free, to be replaced by the microbench winners): 32: (32,128,128,8,-,1,0,DEQ1,UNPACK1),
40/64: (64,128,64,8,-,1,0,DEQ1). DEQ 1 multiplies the fp32 tile result by the scale instead of every bf16 weight
(fewer VALU ops per byte); numerics differ from the stock kernel only by fp32-vs-bf16 rounding of that multiply,
so outputs are not bit-equal; the microbench checks argmax + top-20 set equality and max |diff| / logit std.
No patch file, nothing new in the Dockerfile (the module is already installed); `sly/README.md` row added.

## Expected effect (estimate, unmeasured)

Bucket 64 from 2.19 ms to somewhere between 1.6 ms (M32-like) and 1.3 ms: 0.6-0.9 ms x 2 calls = 1.2-1.8 ms per c8
step = 2.5-3.5 % of the step, c8 +2-3 %; c1/c2 unchanged (buckets 8/16 untouched). That is below Radiance's -5 %
because the Triton head has no spill to remove. If the microbench finds nothing >= 5 % faster than the stock tile,
the knob stays 0 and the lever is closed (test.sh then skips the server arms).

## Test

`tests-lessons/C/test.sh` (copy in `/root/lessons/C/test.sh` on 2408): phase 0 compile report, phase 1 microbench
(`microbench.py`, M 8..64, control vs `cands.py`, exactness vs control + fp32 reference, prints
`RECOMMENDED_LEAN_CFG=` for configs >= 5 % faster), phase 2 server A (LEAN=0) / B (LEAN=auto + winners) [/ A2 with
`ABA=1`], each with accprobe3 greedy (ms/step, tok/upd) and the c8 arrivals probe. ~10 min for A+B.
Accept when: bucket-64 winner >= 5 % faster and argmax/top-20 exact on all rows; B tok/upd == A within noise; c8
arrivals B > A (A2 ~ A). Ship as default-on only after a GSM8K/accept-length check as for the int4 head itself.

## Risks / not verified

- Nothing was run on a GPU: timings above are the repo's existing table numbers; the lean speedup is a hypothesis
  (DEQ 1/2 won at M 40/64 for qkv/gate_up shapes in the window-A2/D sweeps, `_GFX12X_SPLITK` comments).
- The split-K table is process-global: `RADIANCE_W4A16_SPLITK_TABLE` (JSON) replaces it, install_lean then adds to the
  loaded one; `RADIANCE_W4A16_SPLITK=0` disables every split-K entry, including these.
- `triton_w4a16_splitk_gemm` at split 1 / DEQ>0 runs `_radiance_w4a16_splitk_kernel` MODE 3 (direct epilogue); its VGPR use
  is high at some configs (up to 245/256) -- the shipped defaults are 206 and 238.
- An alternative the stock M=64 case may need beyond tile tuning: 2 launches of M=32 re-read the weight (no gain), a
  weight layout with the scale pre-multiplied is not possible in int4 -- DEQ 1/UNPACK 1 is the realistic lever.
