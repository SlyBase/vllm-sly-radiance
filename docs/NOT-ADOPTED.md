# Considered and not adopted

What other R9700 stacks do that this image deliberately does not, with the reason. Every "measured" entry
is an A/B on the production arguments (BetterBench, see the sections above for the numbers); the rest are
design reasons. Revisit an entry when its reason changes.

**From ggz14/radiance-vllm-mxfp4 (upstream MXFP4 work, otherwise merged here):**

| Item | Why not |
|---|---|
| Dynamic verify width as ggz14 wrote it (`RADIANCE_DYNAMIC_WIDTH`, `patch_dynwidth` / `patch_async_dynwidth`) | Measured: conc 4/8/16 −3 %, conc 2 unchanged (ABAB, 300 W). Unequal per-request widths made every batch non-uniform, so the step fell from the FULL graph to the ~65-piece PIECEWISE route. **Superseded in 1.0.0 by `RADIANCE_ADAPTIVE_WIDTH=perseq`** (Radiance's adaptive_k with varlen FULL graphs, c8 arrivals +4.7 %), see [TECHNICAL.md](TECHNICAL.md#100-what-the-release-adds-and-the-measurements-behind-it). |
| GDN in_proj merge (`RADIANCE_GDN_MERGE_INPROJ`, `radiance_gdnmerge.py`) | Upstream's merged forward replaces `forward_hip` with the AITER GDN core; this image serves `forward_cuda` (FLA). A forward_cuda-sliced variant measured step −0.16 ms, prefill −1.5…−2.2 %, and −8.5k KV tokens (the weight copies fragment the arena). |
| Streaming decode loads (`RADIANCE_MXFP4_DECODE_NT=1`) | Measured on top of WPERM: step 37.6 vs 37.3 ms — slower. |
| libr4d decode GEMM (`RADIANCE_MXFP4_R4D_DECODE_MAX_M`) | Measured: not bit-identical, no step gain, −810 KV tokens. |
| Wide tile at M = 2048 (`RADIANCE_MXFP4_TN4_MIN_M` lowered/raised) | Measured neutral for both the folded and the A-tiled prefill GEMM. |
| DFlash2 selector top-k 24/32 (`patch_dflash_selector_topk`) | Measured: +0.8 % weighted decode, −2.5…−3 % conc 1; needs its own compile cache. |
| int2 draft head (`RADIANCE_FAST_DRAFT`) | Its own ~0.17 GiB head copy comes out of the KV pool; this image keeps the KV pool at the model maximum. |
| fp8 stream / arnq producers (`RADIANCE_FP8_STREAM`, `radiance_arnq.py`) | Same fusion as this image's `RADIANCE_FUSED_NORM_QUANT` (and its producers now emit the tiled layout too). |
| Lazy GDN snapshots (`RADIANCE_GDN_LAZY`) | Corrupts multi-turn chat (ggz14 turned it off themselves). |
| `RADIANCE_PRESHUFFLE`, `RADIANCE_VERIFY_HEAD`, `RADIANCE_DRAFT_RERANK` | Apply to FP8 block-scale checkpoints resp. the ParoQuant drafter, not to this model. |
| Top-k/top-p sampler kernels (`patch_topk_*`) | Measured the sampler's share: top-k 20 / top-p 0.95 cost 0.3 % of a step. |
| 3-rank all-reduce and 5/4-bit wire (`patch_ar_3rank`, `patch_ar_qbits`) | Written for the older two-rank `radiance_allreduce.py`; this image ships StillDeadcode's N-rank module (TP=3 rides RCCL), and the 5/4-bit kernels are not in the rebased libr4d extras. The rest of ggz14's multi-GPU work is in (0.3.5, *Several GPUs*). |

**Own measurements (W4A16 work, 0.4.1):**

| Item | Why not |
|---|---|
| W4A8 int8 for the W4A16 GEMMs at M >= 32 (activations quantized per row and K tile in-kernel, int8 WMMA) | Measured (window D, 2026-09-29): never among the five fastest configs of any shape; removed. |
| GDN in_proj_ba as int4 rows of in_proj_qkvz, one GEMM (`RADIANCE_GDN_BA_W4`, kept, default 0) | Measured (window E): the 96 extra rows add a 129th tile, a nearly empty third wave on 64 CUs -- 93.9 us merged against 79.2 + 3.6 us for the two GEMMs. |
| Decode-attention retune at long context | Measured (window E, `bench_decode_attn.py`): the shipped rule is within 1 % of the best of 100+ cells at 32k/96k (136 us per layer call at 32k, ~470 GB/s). |
| Removing the GDN output `torch.zeros` / the spec-decode copy chains (~100 launches per step) | vLLM needs the zeroed rows for padded tokens (vllm#28182); the copies are the speculative-decoding bookkeeping. ~0.1-0.2 ms, not worth the risk. |

**Own measurements, 1.0 (2026-10-07/08, 210 W, A/B/A, second start):**

| Item | Why not |
|---|---|
| Gated gate_up + SwiGLU fold (W4A8 decode band; bytes exact) | Isolated −0.16…−0.23 ms per step at M 8–24, but server greedy 35.00 / 35.12 / 35.07 ms: no end-to-end gain. down_proj needs a per-token activation scale, so its quantization stays a separate launch and the fold removes nothing there. |
| int4 lm_head LEAN configs (split-K DEQ1/UNPACK1 for M > 16) | vLLM's Triton kernel has no spills to remove; no candidate is ≥ 5 % faster in any M bucket. |
| fp16 SSM state (`--mamba-ssm-cache-dtype float16`, A4) | KV −1.6k tokens, prefill +0.3 %, c8 401.3 vs 398.8 / 403.2: neutral. |
| Draft refill after a prefix-cache hit | Not needed: vLLM keeps the drafter's KV in the paged cache, so a hit restores it. Multi-turn turn 2 with a prefix hit: 4.20 / 3.82 tokens per step, 3.3 / 3.5 s against 7.1 s cold. |
| `--attention-backend R4D` (plain), `R4D_ATTN_FP8` with R4D | See row A5 below: decode 60.9 vs 34.9 ms per step, c8 302 vs ~402; fp8 legs +2 % at 64k. `R4D_HYBRID` is adopted instead. |

**Other images and forks:**

| Source | Why not |
|---|---|
| [testeddoughnut/vllm-openai-rocm-r9700](https://hub.docker.com/r/testeddoughnut/vllm-openai-rocm-r9700) (AITER-tuned image) | This image uses AITER only for attention. Its verify-batch decode config is tuned here (`sly/radiance_attn_decode.py`), and a 36-cell sweep of the 2D prefill config found aiter's stock cell within 1–5 % of the best (< 2 % of a 64k prefill); the GEMMs run on the MXFP4 HIP kernels, not AITER. Not evaluated in depth beyond that. |
| [hifi/vllm-radlight](https://codeberg.org/hifi/vllm-radlight) | An arrangement of ggz14 + libr4d on AMD's vLLM 0.27 image, patched at start (no original code). Its knobs are the upstream ones above; the remaining runtime flags (expandable segments, chunk 2560, fp16 SSM state, HW queues, HSA interrupts) are queued for an A/B. |
| [bkvargyas/r9700-stack](https://github.com/bkvargyas/r9700-stack) | Plugin stack for TP = 2 and NVFP4 checkpoints; its int6 embedding gather is behind this image's int4 embedding (`RADIANCE_EMBED_BITS=4`). |
| `tcclaviger/vllm`, `Dyluhn/R9V` | Separate forks with their own model mix (MoE, expert offload, TP ≥ 2); nothing single-GPU-MXFP4-specific to take over was found. |
| DFlash2-FP8 drafter (`tcclaviger/Qwen3.8-27B-DFlash2-FP8`) | Heavier than the W4A16 drafter, and stacks running it report fewer tokens per update than this image (code 4.71 vs 5.09). |
| Qwen3.8-27B-PARO-MXFP6 | Two GPUs only. |

**4-bit KV cache (KV-cache roadmap, 2026-10-13, steps M0 / S2 / S1):**

| Item | Why not |
|---|---|
| 4-bit K/V cache (MXFP4 OCP/OAS block 32/16, token-wise INT4 group 256/64, each with and without Hadamard32) instead of the fp8 cache | Measured, gate G1 failed. Captured post-RoPE Q/K/V of all 16 attention layers of Swift-1.5 (5 prompts, 0.6k-131k context, `sly/kvq/reference.py` as oracle): the best token-local candidate (INT4, group 64, Hadamard32) gives an attention-output error of **2.8x the fp8 cache** (median over layers, max 4.0x); MXFP4-OAS+H32 3.5x (max 5.0x). The gate asked for <= 2x on every layer. The ratio is the format's resolution, not an outlier problem: on Gaussian data fp8 has 2.7 % and every 4-bit format 11 % relative noise, so no scale rule closes it. Keeping the last 896 positions in fp8 lowers the median to 1.6x but the worst layer stays at 3.2x and needs a second pool. The upside was real: decode attention is 33 / 39 % of the step at 2 / 3 x 128k, 3 x (128k + 16k output) preempts at the 439k pool, and a 4-bit spec would give 1.73-1.75x pool tokens (S2). Revisit with a format that is not token-local or with a much larger model-side quality budget (end-to-end GSM8K / needle, which this study did not run). Update 2026-10-16: end-to-end numbers now exist for one 4-bit format that is not in this table (TetraAttention, Walsh-Hadamard rotation plus one e4m3 scale per token and an int4 WMMA kernel, shipped opt-in in 1.5.0, PR #151). Against fp8 on the same card and image: KV pool 747,492 vs 439,031 tokens (+70 %), GSM8K-200 0.855 vs 0.845, KL vs a bf16 KV cache 0.0208 vs 0.0245, but the true-token NLL is +0.0057 vs +0.0004 (about +0.5 % perplexity). The G1 result above still stands for the formats it measured and for fp8 as the default; the 4-bit format ships only as an opt-in. Data: `radiance-benchmarks` windows `m0-210`, `m0b-210`, `s1-210`. |

**Contextual sparsity / Gather-GEMV ([MartinsMessias/radeon-sparse](https://github.com/MartinsMessias/radeon-sparse), reviewed 2026-10-18, source read, nothing run):**

| Item | Why not |
|---|---|
| Gather-GEMV: read only the ~15-25 % active SwiGLU rows of `up`/`down` per token | Verify batches here are M = 8...64 rows. The active set is the union over all rows; at 20 % per token and independent tokens that is ~36 % at M = 2, ~59 % at M = 4, ~83 % at M = 8 (**estimate, not measured**). Little is left to skip. Prefill (M >= 513) is compute-bound. A predictor is also missing: `gate` must be read in full to know which rows are active. |
| Their numbers (1.4x...2.9x `tg128`, up to 9.6x kernel speedup, 5,700 GB/s) | Not end-to-end: `radeon_sparse_bench.c` multiplies one layer's time by the layer count, and the "+ N-Gram Spec" row is a hard-coded `x 1.55`. The kernel loop re-reads one matrix 150 times without rotation, so a 20-30 MiB working set sits in the 64 MB Infinity Cache. The 75-85 % sparsity figure is from random Gaussian weights, not model activations. No perplexity, GSM8K or other quality check. |
| Their kernels (OpenCL, INT8, `vload16`, LDS tree reduction) | Dense GEMV on INT8; no WMMA, no MXFP4/W4A8. Our HIP kernels (`sly/mxfp4/`) are already specific to gfx1201. |

Revisit when: measured per-token FFN activation sparsity on Qwen3.8-27B is >= 70 % **and** the union at M = 8 stays below ~40 % **and** GSM8K holds the `ci/accept/baselines/` band. Related published work (TEAL, CATS) reports ~40-50 % with small loss, not 80 %.

## Paiton (Eliovp-BV/paiton-vllm-plugin, `models/Qwen3.8-MXFP4-DFlash2`)

Reviewed 2026-09-27 against their 26 September 65K image. Paiton is a vLLM plugin whose speed comes
from **native HIP kernels built by a private compiler** (GEMM, attention, GDN prefill/replay, fused GDN
spec-verify, target/draft heads, fused SiLU/RMS + fp8). The repo ships only the Python adapter
(Apache-2.0) and downloads the `.so` files as a closed bundle (`licenses="NOASSERTION"`, "compiler and
implementation source stay private"). Nothing of that can be taken over as code.

**Their numbers vs ours are not one A/B.** Their table (MXFP4 arm): weighted decode 156.1, C8 428.0,
prefill 3,691 @ 1.5k / 3,455 @ 47k; ours (0.4.0 reference): 133.2 / 410.0 / 3,166 / 2,512. Different
BetterBench (0.6.0 quick vs 0.4.0 default), thinking off, 65,536 context, different drafter. Decoded
into step time and tokens per update:

| | Paiton (MXFP4) | 0.4.0 | Where the gap comes from |
|---|---|---|---|
| decode update p50 | 33.3 ms (fwd 28.0 ms) | 34.84 ms | ~4 %: their native GEMM/GDN-verify kernels |
| tokens / update, weighted | ≈ 5.2 | 4.57 | ~14 %: benchmark version, thinking off, FP8 drafter + greedy draft sampling |
| prefill @ 47k | 3,455 | 2,512 (2,613 at 131k/4096) | native long-prefill attention (16-key tiles), GDN chunk scan in one GPU round (~30 % faster GDN core), 4096 chunk |

So most of the decode gap is acceptance (i.e. drafter + benchmark), not kernels; the prefill gap is kernels.
The measurement plan (BetterBench 0.6.0, thinking off first, then drafter/greedy arms) is the
[like-for-like recipe](BENCHMARKS.md#recipe-like-for-like-against-paiton-queued-not-yet-run).

**Transferable = runtime flags only. Queued for an A/B** (production args, 300 W, same window, control
first and last, second start, ≥ 128 BetterBench runs per arm or the step gap; KV pool must stay 436,906 at 1.0.0 and later):

| Arm | Paiton setting | Ours today | Why it might help |
|---|---|---|---|
| A1 | `GPU_MAX_HW_QUEUES=1` | `2` | Paiton: "removes a slower decode mode some fresh processes start in"; ours picked 2 against the random default, 1 was not in that sweep |
| A2 | `draft_sample_method: greedy` | `probabilistic` | lossless either way; greedy drafts skip the draft-side sampling, and the lookup draft already writes point masses |
| A3 | `tcclaviger/Qwen3.8-27B-DFlash2-FP8` drafter | `syvai/...-W4A16` | Paiton's tokens/update (json ≈ 7.2, code ≈ 6.0) are well above ours (5.66 / 4.94); the old "fewer tokens per update" row above came from other stacks, re-check on this image. Costs KV (heavier drafter) |
| A4 | `--mamba-ssm-cache-dtype float16` | `bfloat16` | fp16 state: more mantissa, same bytes; the libr4d extras have fp16 kernels (also on the radlight list) | **Measured 1.0: neutral, not adopted** (KV −1.6k tokens, prefill +0.3 %, c8 401.3 vs 398.8 / 403.2).
| A5 | `--attention-backend R4D` | `R4D_HYBRID` (1.0.0) | **Measured, plain R4D not adopted:** prefill 64k +19 % but decode 60.9 vs 34.9 ms per step and c8 302 vs ~402 tok/s. The hybrid (libr4d only for prefill runs >= 512 tokens) keeps the prefill gain and the AITER decode: adopted in 1.0.0. `R4D_ATTN_FP8` on plain R4D: 64k 1918 tok/s (+2 % over the f16 legs), not adopted. |
| A6 | `cudagraph_capture_sizes` incl. `1, 2, 4` | `[8 … 64]` | only matters if a step ever runs below 8 tokens (non-spec paths) — expect neutral |

`--max-num-batched-tokens 4096` at shorter context is already the README's "less context, more prefill"
recommendation. Paiton's opt-in n-gram co-drafting (`PAITON_NGRAM_CODRAFT`) is the same idea as this image's
`RADIANCE_LOOKUP_DRAFT` (default on since 0.2.9); nothing to add. Their 3-bit W3A4 weights
(`EliovpAI/Qwen3.8-27B-W3Rot-INT3-Paiton-RDNA4`, +19.9 % decode, MMLU-Pro −2.9 pts) need their closed
kernels — Transformers, stock vLLM and this image cannot load them.

**Kernel leads worth reimplementing ourselves** (ideas from their release notes, no code available):
16-key tiles for long-prefill attention (+6.6 % attention), GDN gate read in place instead of copied, a GDN
chunk-scan launch that fills the card in one round, and a fused GDN kernel for the speculative verify
step (+1.5 % weighted decode, bit-exact to the old path).

## MiniCPM-V 4.7 (2026-10)

Measured with the arguments in [MINICPM-V47.md](MINICPM-V47.md), 210 W, BetterBench `--decode` or probe step time.

| Item | Why not |
|---|---|
| Drafter attention on AITER (`"attention_backend":"ROCM_AITER_UNIFIED_ATTN"` in `--speculative-config`) | Does not start: `ValueError ... not valid for this configuration ... non-causal attention not supported` (DSpark drafts with non-causal attention). |
| k = 10 / 12 after the AITER switch | Probe step 26.2 / 27.5 ms vs 24.7 ms at k = 8; BB k = 12 158.2 vs 169.4 (tokens per step 4.65 vs 4.24, only math gains). |
| `GPU_MAX_HW_QUEUES=2` | Step 55.4 vs 30.2 ms. |
| `--no-async-scheduling`, `HIP_FORCE_DEV_KERNARG=1` | No effect (30.2 / 30.3 vs 30.2 ms). |
| `--attention-backend TRITON_ATTN` | 26.5 ms, AITER 24.7 ms. |
| `enable_adaptive_verification` (DSpark confidence head) | Refuses the GDN attention backend at startup. |
| CUDA-graph sizes aligned to k+1 (`[9,18,...]`) | Slower than `[8,16,32,...]`. |
| Same AITER flag for Qwen3.8 production | Production already runs `R4D_HYBRID`; AITER 34.96 vs 35.00 ms at equal acceptance. |

## Prefill kernel levers that failed their gate (2026-10-20)

Both were written, compiled and measured on the R9700 at 210 W against the shipped kernels; neither
beats its control, so both were dropped from the release. The work is archived on a branch
(`exp/gdn-scan2`, `exp/attn-a1`) rather than carried here.

| Item | Why not |
|---|---|
| GDN `chunk_scan` v2, staging prefetch issued in the chunk tail (`RADIANCE_GDN_SCAN_V2`) | Does what it set out to do and still loses: scratch spills 4 -> 0, VGPR 256 -> 205, LDS footprint, barriers, grid and instruction mix unchanged, **bit-identical on all 28 digest lines** (7 shapes x 2 layouts). Time at T = 4089, H = 48 (three runs, ABBA): **0.521 -> 0.541 ms = 1.042 x slower** (1.024 / 1.037 on the other two), against a GO of <= 0.92 x. Together with w3 (LDS diet, +8.9 %) the scan is therefore neither LDS- nor register-bound -- what is left is latency and the four chunk barriers. |
| Attention A1 VALU diet, `O_CFOLD` + `O_MAX3` (`R4D_ATTN_OPT2`) | Cuts the arithmetic it targets -- softmax **53 -> 46 VALU** per steady m-tile block, 218 VGPR / 0 spills, outputs identical to the shipped arm -- and the time does not move: **1.008 x at 16k, 0.990 x at 40k**, i.e. noise around parity against a gate that asked for >= 4 %. The softmax VALU is not what limits the prefill attention kernel; WMMA and the LDS traffic are. |

## TetraAttention: query quantization in the attention kernel's prologue (2026-10-10)

Tried: the HIP kernel takes the bf16 query, rotates it (Walsh-Hadamard as a butterfly over the
lane pair that holds a row's 256 dims), quantizes to int8 and parks the int4 planes -- one launch
less per layer (the Triton `quantize_query`, 16 us graph-timed). Rejected: the prologue's 128 live
floats push the kernel from 237 to 249 VGPRs, occupancy 6 -> 5 waves per SIMD, and the tile loop
pays for that on every tile: +15-20 us per call at 19k context (1 sequence), +50 us at 4 sequences
-- as much as or more than the launch it saves (graph-timed full path 150 vs 150 us at 1 sequence,
561 vs 484 at 4). Scheduling barriers and staged loads removed the spills but not the occupancy
loss. The un-rotation folded into the segment merge (`reduce_unrotate`) is kept: it costs nothing
extra there. `sly/kv4/radiance_kv4h.hip` keeps the quantized query input (int4 by default, int8 as two planes).

## Prefill GEMM: overlap, big tile, library paths (2026-10, 210 W cell bench, tried for the 4,000 tok/s goal)

All variants below were bit-identical to the A-tiled kernel (sha256 of the bf16 output equal for every cell) and
measured against it ABBA in the same invocation. None gave a reproducible gain, so none is in the image. The
sources of the two kernel variants are kept on the branch `exp/prefill-gemm-attempts` (commits b8249f1, 1e8fc9d).

| Item | Why not |
|---|---|
| Software-pipelined W stream, double-buffered `sW`, one barrier per slab (`AT_OVL`) | Neutral to 0-4 % slower (gate_up / down, M 1536 and 4096). The compiler keeps dequant and the LDS store after the WMMA group (`sched_barrier`), only the global-load stream overlaps. |
| Dequant interleaved between the WMMA groups | +1-3 % slower. |
| Producer/consumer wave split (2 or 4 extra producer waves) | +2-8 % slower, reproducible in both ABBA pairs; VGPR 143-193, no spills. |
| 16-wave big tile 256x128 / 256x256 (`AT_BIG`) | 256x128: +1-10 % slower. 256x256: +4-5 % slower at M >= 2048, about 2-3 % faster at M 1024-1536 for gate_up/qkvz (two pairs only, not enough for a server arm). |
| Smaller row tile TM = 2/3 (`AT_TMS`, kept, default off) | Slower at M >= 768; only M 513-576 gains (+17-28 %, padding effect). |
| `RADIANCE_GROUPM` raster grouping (K2) | Gain below the window drift (+0.1 % / +0.3 %). |
| Dequant to a fp8 scratch + hipBLASLt (`torch._scaled_mm`), per matrix | 0.47-0.75x of the A-tiled kernel. hipBLASLt row-wise fp8 reaches only 96-129 TF/s on gfx1201 (TunableOp: <= 166), the dequant adds 108-812 us per matrix. Not bit-identical. |
| bf16 dequant + `torch.matmul` | GEMM alone 70-113 TF/s. |
| Triton fp8 GEMM (autotuned `tl.dot`) | 150-210 TF/s without any dequant, i.e. about the A-tiled kernel; a fused W4A8 Triton kernel cannot exceed that. |
| libr4d GEMMs (`gemm_mxfp4a8_nt_m64`, `gemm_w4a8_nt_m64`, ...) | Constraint M <= 64; no large-M entry. |

Where the time goes (ablation, `AT_ABL` flag, timing only): dequant ALU about 7 %, LDS store about 10 %, W global
loads about 8-12 %, A stream 5-13 %; WMMA + LDS reads + barrier alone reach 259-281 TF/s against 160-230 TF/s
for the kernel. A cheaper dequant alone is worth at most about 5 %. At 210 W the same cell varies up to 41 %
between runs (M = 513), so gains below about 10 % need ABBA with many repeats.

## Prefill levers measured on 2026-10-19 (R9700, 210 W, image 1.5.1, chunk 4096 unless noted)

All four were checked after the drafter `fc` tile (#165) to see what is left of the 5 % prefill gap
at 47k tokens against the reference stack on the same card. None of them is adopted.

### GDN `chunk_scan` with a smaller LDS footprint ("w3")

Tried: `r4d_gdn_chunk_scan_k128_v128_c64_bf16` has 256 VGPRs, 4 spilled VGPRs (20 B/lane scratch) and 60,928 B LDS,
so it runs one workgroup per CU and the 96 workgroups of a 4089-token chunk need two rounds (64 + 32). A variant
(S and W+sA share one LDS union, one extra barrier per chunk, row scale and gate kept in registers, built without
`-mcumode`) needs 43,520 B LDS, 240 VGPRs, 1 spill, 6 waves per SIMD, i.e. three workgroups per WGP and one round.
Rejected: output is bit-identical to the stock kernel (28 of 28 digests over T = 64..4089, N = 1 and 2; error against
an fp64 reference 4.4e-3 / 3.4e-3 for output / state) but the kernel is **9 % slower** (ABBA, median of 20, T = 4089,
H = 48: 0.522 ms stock, 0.569 ms variant). The extra barrier, the spill and the lower rate per workgroup cost more
than the second round saves. `chunk_scan` is 380 ms of 18.8 s at 47k (2 %), so even the planned -30 % would have been
0.6 % end to end.

### Attention without the LDS path: ablation only, no rewrite started

Measured: one attention layer (`r4d_attn_prefill_h256_gqa6`, `R4D_ATTN_FP8=3`, M = 4089, ctx 0 / 16,356 / 40,890),
timing-only variants that each remove one cost item (results are wrong by design; the patched base kernel equals the
shipped one within 1 %): softmax replaced by `p = score` -9.5..-10.6 %; K/V fragments from a register instead of LDS
-20.8..-22.0 %; K/V staging and barriers only for the first tile -14.9..-29.3 %; QK WMMA removed -25.6..-30.9 %; PV WMMA
removed -28.6..-31.4 %. The items overlap (the sum is far above 100 %), so each number is an upper bound. The only
removable item above 15 % is the LDS path; a kernel without LDS staging would have to load the fragments from L0/L2
instead. Attention is 18.6 % of the 47k prefill, so a 10 % faster kernel is 1.9 % end to end, 15 % is 2.8 %. Not started:
weeks of work for an unproven 0..3 %. The shipped f16-leg path was not ablated.

### Launcher idle / prefill CUDA graph

From the in-process profile (one request per depth): GPU idle is 7.5 % at 1.5k tokens and 2.5 % (471.7 ms) at 47k, a
constant ~39 ms per chunk. 64 % of it (2,663 gaps, 302.7 ms) are gaps of 112..116 us right after `gemm_atiled<4>`
before `add_rms_quant`, `silu_mul_quant` or the next GEMM: the length is the same whatever the number of launches, which
looks like a stall at the end of a long kernel, not launch latency. A graph is not expected to remove it (at most
~0.5 % end to end at 47k, an estimate), and fusing the silu quant into the GEMM epilogue already measured 0 +- 0.5 %.
The reference stack has more idle than ours at 47k (497 vs 472 ms). Not built.

### Negative voltage offset beyond -50 mV

The card already runs `voltage_offset: -50` (LACT, since September). Tried -75 mV at the 210 W cap with the GTOA stack:
the GEMM cell check stayed bit-identical (24 digests, 0 WRONG) but the clock under load did not rise (median 1,895 vs
1,916 MHz at 209 W) and prefill read 2,606 vs 2,721 tok/s at 16k and 2,200 vs 2,232 at 47k (window drift is up to 3 %,
so this is "no gain", not a measured loss). -100 mV and GSM8K at -75 mV were not run. The assumption that the power cap
turns less voltage into more clock (about +2 % per 25 mV from P ~ f V^2) did not hold on this card.

## TokenGate (token-gate-llms/TokenGate, arXiv 2610.26754, reviewed 2026-10-21)

A learned token-pruning gate for long-context prefill: a gate at an early layer L scores every token of a marked
"state" span, keeps the top-K, and the layers above L process and KV-cache only those K tokens ("skimming"; up to
11.3x TTFT and 3.4x peak memory, batch 1, H100). The non-differentiable selection is trained end-to-end through a
per-key attention bias built from the scores ("token-gated attention"), jointly with LoRA on q/v projections. Apache-2.0.

| Item | Why not |
|---|---|
| Gate checkpoints | Released only for Qwen3-1.7B and Qwen3.5-2B. A gate is task-specific (NIAH, RepoBench, BioDEX, 2Wiki, MultiNews), so Qwen3.8-27B needs its own trained gate plus adapter. Training is DDP torchrun, CUDA-first, wandb — no path on the 32 GB R9700 (production card, power capped). |
| Fast attention path | The key-bias kernel is a fork of FlashAttention-2 2.8.3 compiled with nvcc (sm_80+PTX, bf16, head_dim 128 only); the no-build fallback is FlexAttention (torch.compile). Our prefill attention is the libr4d HIP kernel inside vLLM V1 — no CUDA and no FlexAttention path here. Taking the idea means writing the key-bias prefill kernel ourselves. |
| vLLM integration | None in the repo: inference is a custom `generate` loop over HF transformers with its own cache handling. In vLLM this is an engine feature, not a plugin — per-request variable KV above layer L, key-biased attention in every full-attention layer, scheduler and KV-manager changes. |
| Regime of the headline numbers | Batch-1 TTFT on H100 over document-plus-question prompts (NIAH shape) where the prunable state dominates. Our production traffic is coding-agent workloads where the context is broadly relevant, and prefill is already 2,430-3,024 tok/s and compute-bound. The prefix cache already skips prefilled context; TokenGate prunes inside new context, which our traffic does not put there. |
| Quality evidence | Task metrics only (NIAH acc, RepoBench ES, BioDEX F1), and the numbers mix the gate+LoRA fine-tune gain with the pruning gain (base RepoBench ES 0.285 -> 0.587 unpruned). No GSM8K band, no perplexity, nothing comparable to our acceptance gate. |

Revisit when: (a) a workload with a dominant prunable state (document-plus-question) appears in production traffic;
(b) a gate trained for Qwen3.8-27B exists (needs a training GPU, not the R9700); (c) a HIP key-bias prefill kernel
is ported to libr4d. The bar then: >= 10x TTFT at >= 32k context, batch 1, with GSM8K-200 inside the 0.855 +/- 0.025 band.

Process lesson kept: their benchmark is an honest template for any prefill-pruning idea — batch 1, frozen held-out
test split, an unpruned reference arm, and random-selection baselines.


## Prefill levers measured on 2026-10-21/22 (R9700, image 1.6.0 base + the 1.8.0 kernels)

### Drafter prefill-window skip with prefix caching on: FAIL, removed

A sub-knob (`RADIANCE_DFLASH_PREFILL_WINDOW_SKIP_APC=1`) let the skip act with prefix caching on. 210 W GPU check,
greedy, same prompt, only the switch differing: the ON arm's output was no longer identical (text ended `...-bbc52c2`
instead of `...-bbc`), the accepted-draft count on the 39,622-token prompt went 82 -> 179 (17 -> 33 drafts), the
prefix-cache query counter 39,622 -> 86,807 and the request took 2.1x as long, which reads as a recompute after
masking. Over two 10-round sessions TTFT was no better and 0.8-1.8 % slower (the skip does not even fire in 2K
rounds, gate 3,808). The commit was reverted; the skip stays gated on prefix caching off.

### Tall tile for the merged GDN in_proj: slower in the cell bench, not enabled

`RADIANCE_MXFP4_TALL` is gate_up only (N 34,816): the same tile on in_proj (N 16,480) was up to 2 % slower than
TN2/TN4 in the cell bench. At serve level the in_proj variant (night arm V8, 300 W) against the gate_up-only stack
(V6) read -0.1 / 0.0 / +2.1 / +0.9 / +1.7 % at 2K-64K, but the two V6 starts differ from each other by up to 3.7 %,
so this is noise, not a gain; it is not enabled.

### NVFP4 checkpoint instead of MXFP4: not faster

Same image, same launcher, 210 W, prefill tok/s at 2K/8K/16K/32K/64K: MXFP4 3,051 / 3,006 / 2,922 / 2,746 / 2,488,
NVFP4 2,968 / 2,901 / 2,833 / 2,687 / 2,445 (-2.7 / -3.5 / -3.0 / -2.1 / -1.7 %). The weight format is not the cause of
the remaining GEMM gap to the reference stack.

### GEMM / W-stream overlap: already in the kernel

The overlap the reference stack is credited with (the next slab's weight loads issued while the current slab computes)
is what the A-tiled kernel already does; the explicit software-pipelined variant (`AT_OVL`, section above) was neutral
to 0-4 % slower. Nothing to port.

## Row-granular drafter prefill skip (PR #187, reviewed 2026-10-25): correct, but 0 % in production

`RADIANCE_DFLASH_PREFILL_PARTIAL_SKIP=1` runs the DFlash2 drafter's `fc` and context-K/V fill only over the last
`window + margin` rows of the prompt (3,808 rows at a 6,000-token prompt instead of 6,000). The code is correct: the CI
gates and the patch dry run are green, the kept rows stay bit-identical (never fewer than 513 per chunk), greedy text
and all `spec_decode` counters are identical, and the CPU test catches deliberate breakages. The PR measured at 300 W
with prefix caching off: +0.80 % prefill at 6k, +0.27 % at 24k, 0 % at 1.5k / 12k / 48k.

Not adopted, because it uses the same gate as the whole skip (section above): it turns itself off when prefix caching
is on, and production runs with prefix caching on. Turning prefix caching off to get the knob is a bad trade: one
cache hit on a 20k-token prefix saves about 5 s of prefill, and the knob saves about 0.8 % of a cold prefill. If we
merged it, it would act only in the benchmarks (prefix caching off for determinism) and make the published prefill
numbers higher than what production sees. Open it again only if a version of the skip works with prefix caching on.


## 2:4 structured sparsity on the weights via FP8 SWMMAC (RDNA4), reviewed 2026-10-21

The idea: prune every layer to 2:4 (two of each four weights along K are zero) in one shot with SparseGPT or Wanda,
store the weights as compressed FP8 plus a sparsity index, and let the card's sparse matrix instruction do the MACs.
Carrot: the prefill GEMM is 66 % of the 47k prefill span, so halving its compute would be about +22 % prefill.

The hardware is there, the payoff is not.

| Item | Finding |
|---|---|
| Sparse FP8 instruction on gfx1201 | Exists and is reachable from the toolchain we ship. `clang++ -x hip --offload-arch=gfx1201 -S` emits `v_swmmac_f32_16x16x32_fp8_fp8 v[0:7], v[12:13], v[8:11], v15`; the builtin is `__builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32` (ROCm 10.1, image 1.6.0). |
| Rate gain per instruction | Not documented by AMD. The page states the gain as *"halves the storage and bandwidth required for A"* and says nothing about throughput. Arithmetic: 16x16x32 with 16 non-zeros holds the same 4,096 useful MACs as the dense 16x16x16 FP8 instruction, so a gain can only come from issuing 2x the B operand per cycle. Unmeasured. |
| Dense FP8 reference width | `v_wmma_f32_16x16x32_fp8_fp8` does not exist on gfx1201 ("invalid instruction"), so the sparse form cannot simply be compared to a wider dense form. |
| Headroom left in the kernel | Small. The A-tiled FP8 GEMM already runs at 270 TF/s effective at 47k (2.54 PFLOP over 9,425 ms) against 325 TF/s measured register-resident peak = 83 % of the card. WMMA alone is 55 % of the GEMM time; the other 45 % is unpack, LDS and barriers and stays. |
| Best case end to end | +22 % prefill at 47k and +24 % at 2k **if** the sparse instruction doubles the rate and the operand path is free. If the rate is at parity the gain is 0, because prefill is compute-bound and A-bandwidth buys nothing. |
| Serving path | None. vLLM 0.31 refuses a sparse checkpoint outright: `compressed_tensors/compressed_tensors.py:311` raises *"Sparsity support has been removed from compressed-tensors. Please use a model without sparsity configuration."* A sparse model here means our own quant config, loader, checkpoint converter and kernel. |
| Pruning tool | Present but not the bottleneck: `llmcompressor/modifiers/pruning/sparsegpt/base.py` takes `sparsity: 0.5` and `mask_structure: "2:4"`. |
| Dense base checkpoint | Not on the host. What is served is the MXFP4 checkpoint (19 GB); one-shot pruning needs the dense BF16 model plus a calibration set. |
| Operand slot | The sparse operand is SWMMAC's A slot, and our GEMM keeps the weights in B (`C[M,N] = A[M,K] . W[N,K]^T`). Pruning the weights means transposing the formulation to `C^T[N,M] = W . A^T`, which changes the epilogue, the store layout and the scale axis. That is a new kernel, not a change to the existing one. |

Accuracy is the deciding item, and it is not close. 2:4 in one shot was validated on 2023-scale models; on current
models it costs a factor of three.

| Model | Dense | SparseGPT 2:4 | Change |
|---|---:|---:|---:|
| OPT-13B (WikiText2) | 10.13 | 12.96 | +28 % |
| OPT-175B (WikiText2) | 8.35 | 8.74 | +5 % |
| LLaMA-7B (WikiText2) | 5.68 | 11.00 | +94 % |
| Llama-3.1-8B Instruct (Wiki) | 7.27 | 26.19 | 3.6x |
| Llama-3.1-70B (Wiki) | 3.34 | 10.93 | 3.3x |

Sources: Frantar et al., SparseGPT (ICML 2023) Table 1; PGZ (ICLR 2024) Table 1; arXiv 2501.18015 Table 1.
Our model is already 4-bit group-quantized and an abliterated variant, so 2:4 would be the third degradation stacked
on it. A 22 % prefill gain does not buy a 3x perplexity loss.

Revisit when: (a) a measured register-resident microbenchmark shows the sparse FP8 instruction above 1.6x the useful
MAC rate of the dense one (the only cheap decisive test, ~30 min in a GPU window, and the reason to run it first);
(b) a consumer for sparse weights exists in vLLM again, or we are willing to carry our own loader; (c) pruning with
recovery training is on the table, because one-shot 2:4 is what fails here, not sparsity as such.

## EAGLE-4 ([eagle-hq/EAGLE-4](https://github.com/eagle-hq/EAGLE-4), reviewed 2026-10-21, source read, nothing run)

`eagle-harness`: a speculative-decoding plugin that runs a trained 1-layer EAGLE-4 draft inside pinned SGLang /
vLLM / TokenSpeed builds. Measured: 4x B300 (3.40x SGLang / 2.99x vLLM, conc 1, Qwen3.8-27B BF16) and 1x MI355X
SGLang (2.07x conc 8, 1.56x conc 32). Their HF card compares EAGLE-4 3.59x to DFlash2 2.80x at T = 0 on 4x B300.

| Item | Why not |
|---|---|
| gfx1201 support | None. The kernel tiers are NVIDIA sm_80+ and AMD CDNA2/3/4 (gfx90a, gfx942, gfx950); `kernels/arch.py` refuses `EAGLE_HARNESS` at startup on any other GPU. No serving measurement on any RDNA4. |
| License | "Eagle 4 Source-Available License", not OSI open source. Commercial Use of a Competing Offering requires a separate written commercial license; other commercial use is gated at a US $20M aggregate-revenue threshold. The Radiance image on DockerHub cannot ship it without a commercial license from licensing@eagle.ai. |
| vLLM verify path | Tree verification needs an unquantized KV cache (`known-limits.md`); this image serves fp8 KV. The GDN tree-verify Gluon kernel is B300-only; the Triton fallback tier does not cover gfx1201 (row above). |
| Engine coupling | The plugin pins specific engine builds (sglang `0.5.23.dev252+eagle.5`, a vLLM with `suspend_drafting` / `verify_budget` hooks and the `CUSTOM` attention-backend slot) and registers through a generic hook list that refuses to load on any name mismatch. This image's vLLM is patched (MXFP4, DFlash2, R4D_HYBRID); the port surface is the whole hook list, not a flag. |
| Draft model | 1 layer, hidden 5120, 40 heads, low-rank candidate head (rank 128, 128 candidates), reads 3 target hidden states (layers 3/31/59), sliding window 4096, BF16, ~1.5 GB. Smaller than this image's DFlash2 W4A16 full-model drafter; quantization is not an algorithmic requirement (their runs are BF16) but a VRAM-budget question. |
| Numbers transfer | Every speedup is B300 / MI355X at a BF16 target, conc 1-32; on SGLang above conc 64 they fall to 0.61x-1.35x. Nothing on gfx1201, nothing MXFP4, nothing fp8 KV. This image's DFlash2 does 4.5-5.09 tokens per step on this card; there is no A/B to compare against. |

Ideas to take (no code, the license blocks reuse): a low-rank candidate head for the drafter's LM-head cost
(rank 128 / 128 candidates instead of the full-vocab head), and cost-model-driven verify widths that narrow at
high batch (their conc-64 rows: width 16 = 0.76x, width 4 = 1.16x) -- the same direction as `RADIANCE_ADAPTIVE_WIDTH`.

Revisit when: (a) a gfx1201 (or other RDNA4) tier ships and compiles; (b) the license becomes OSI open source, or
a commercial license is granted; (c) the vLLM tree verify works with a quantized KV cache.

**Own measurements, 1.8 (2026-10-23/24, KV4 prefill):**

| Item | Why not |
|---|---|
| libr4d int4-KV prefill route (`R4D_KV4I`, variants A/B/B2) to close the KV4 prefill gap | KV4 (Tetra) prefill is 11.7-11.8 % slower than fp8 KV at 64k (210 W and 300 W); the kernel is 1.80x (rocprof). B2 is more accurate than Tetra against an fp8-KV reference (KL 0.01557 vs 0.01702, GSM8K-200 0.855 vs 0.845) but not faster (-1.3...+3.4 %, noise). KV4 itself stays supported (528,722 vs 342,950 KV tokens, prefill -11.7 % at 64k); only this route is dropped. Experiment branch `exp/kv4i-dump` (parked, not merged). |

## W4A4 prefill (int4 activations) from the hipfire engine (2026-10-26)

Cross-engine test on one R9700, 210 W, same window, BetterBench `--quick`: hipfire 0.4.3.1 (Qwen3.8-27B,
MQ4-XTS + its DFlash draft) against 1.8.1 (Swift-1.5 MXFP4 + DFlash2 W4A16). hipfire prefills 18-24 % faster
(3273/3487/3337/3282/2932 against 2696/2818/2779/2689/2484 tok/s at 1515/5920/11796/23546/47058 prompt
tokens) and decodes 6.6 % slower (143.5 against 153.7 weighted), so only its prefill path was examined.

| Item | Why not |
|---|---|
| **IU4 prefill route: int4 weights x int4 activations** (`v_wmma_i32_16x16x32_iu4`, FWHT-rotated activations) | The gain is the activation width, not the matrix rate: their own CHANGELOG puts the A4 route at 4.8-4.9k tok/s and their native-FP8 route (W4A8-class) at 3.8-3.9k at pp8192 on other cards, which is where this image already sits. Their measured quality cost on gfx1201 for A4 against their own A8: WT2 KLD 0.0419 -> 0.0696 (+66 %) and code24 0.0299 -> 0.0521 (+74 %); on gfx1151 MoE it is +0.0205 nats. This image's budget for a prefill-route change is dKLD <= 0.010 nats, so A4 fails by the competitor's own evidence. Their "Quality-first" preset turns it off on gfx1201. |
| Porting cost, even if the quality bar moved | MXFP4 values x2 reach +-12 and do not fit int4, so the route needs a rotated int4 trunk (ParoQuant int4 exists on HF and this image serves it as W4A8) plus an int4 activation quantiser and an int4-WMMA GEMM. That is a new prefill kernel, not a flag. |
| Their prompt-processing chunk size | Not a lever: their gfx1201 default is 8192 rows (the `prefill.chunk_rows` 512 in their config schema is the fallback), and the 60-68 us/token gap is flat over depth, i.e. a per-token GEMM cost. |
| Their DFlash "hidden rows stay on device" prefill gain (+29 % in their CHANGELOG) | Fixed a host copy in their own path; this image has no equivalent host round-trip. |
| Their `fa_prefill_pm` attention kernel | H16/KV2 geometry only, not the 27B's H24/Hkv4/D256. |

Revisit when: (a) a rotated-int4 trunk with a measured dKLD <= 0.010 nats against our W4A8 exists; (b) gfx1201
gains an FP4 matrix instruction (NVFP4 W4A4 checkpoints exist on HF, but the ISA has no such GEMM today).
