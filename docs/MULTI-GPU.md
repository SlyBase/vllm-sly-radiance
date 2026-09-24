# Several GPUs (tensor parallel)

Not tested on the maintainer's single-card box.

This box has one R9700, so none of the multi-GPU paths below were run here. They are in the image
because ggz14 and StillDeadcode serve them on 2–3 cards; what was checked here is that they stay
out of the way at TP=1 (KV pool, throughput and GSM8K unchanged, see the 0.3.5 table above).

| TP | All-reduce | What to set |
|---|---|---|
| 2 | libr4d one-shot P2P kernel, exact bf16 up to `RADIANCE_AR_MAX_KB` (default 49152 KiB), 6-bit rotated wire above `RADIANCE_AR_QUANT_MIN_KB` (`RADIANCE_USE_R4D_AR_QUANT=1`, default) | `--tensor-parallel-size 2`. Raise `RADIANCE_AR_MAX_KB` to at least `max-num-batched-tokens × 5120 × 2 / 1024` (e.g. 98304 at 8192) or every prefill all-reduce silently falls back to RCCL. `RADIANCE_AR_QNT` / `_QNB` tune the 6-bit path (ggz14 ships 96 blocks). |
| 3 | RCCL (no 3-rank kernel in this image) | `--tensor-parallel-size 3 -e RADIANCE_TP_PAD=3 -e RADIANCE_MXFP4_WPERM=0`: `radiance_tp3pad` widens the heads (24/4/16/48 → 36/6/18/54), the MLP (17408 → 17472) and the vocab with zero-weight dummies at load, so every dimension divides by 3. Checked here on one card (`RADIANCE_TP_PAD=3` at TP=1, see below). The DFlash2 **W4A16** drafter is not padded (its packed int4 tensors are not in the padding tables), so at TP=3 either drop `--speculative-config` or use an fp8 DFlash2 drafter (ggz14's setup). |
| 4 / 8 | libr4d N-rank kernels (one-shot to 6 tokens, two-shot above, tiered-int8 wire at TP=4 with `RADIANCE_USE_R4D_AR_QUANT=1`) | `--tensor-parallel-size 4` / `8`. No padding needed (all head counts divide). |

`RADIANCE_USE_R4D_AR=0` keeps RCCL everywhere. Add `--device` access for every card and drop
`HIP_VISIBLE_DEVICES=0` (or list the cards). `--gpu-memory-utilization`, `--max-num-seqs` and the
CUDA-graph sizes in the production command were tuned for one 32 GB card and are only a starting
point.

TP=3 padding on one card (2026-09-24, `RADIANCE_TP_PAD=3`, TP=1, no speculative decoding):
the target is padded at load (1143 of 1695 tensors, coverage check OK), serves coherent text and
scores GSM8K 0.83 (100 questions, ±0.04; unpadded 0.835–0.845). The padded heads cost KV: 1.13×
instead of 1.46× of 262k on one card. Two things this run found and fixed: the coverage check
assumed a quantized MTP layer (the AMD checkpoint's `mtp.*` is bf16), and `RADIANCE_MXFP4_WPERM=1`
cannot serve the padded GDN `in_proj_ba` (N = 108 is not a multiple of 16; the fragment layout is
global to every kernel) -- **serve `RADIANCE_TP_PAD=3` with `RADIANCE_MXFP4_WPERM=0`**, which costs
the 3.7 % decode WPERM brings at TP=1.
