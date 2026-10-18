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
