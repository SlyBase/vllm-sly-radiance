# Considered and not adopted

What other R9700 stacks do that this image deliberately does not, with the reason. Every "measured" entry
is an A/B on the production arguments (BetterBench, see the sections above for the numbers); the rest are
design reasons. Revisit an entry when its reason changes.

**From ggz14/radiance-vllm-mxfp4 (upstream MXFP4 work, otherwise merged here):**

| Item | Why not |
|---|---|
| Dynamic verify width (`RADIANCE_DYNAMIC_WIDTH`, `patch_dynwidth` / `patch_async_dynwidth`) | Measured: conc 4/8/16 −3 %, conc 2 unchanged (ABAB, 300 W). Unequal per-request widths cost more than the trimmed verify rows save on this card. |
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

**Other images and forks:**

| Source | Why not |
|---|---|
| [testeddoughnut/vllm-openai-rocm-r9700](https://hub.docker.com/r/testeddoughnut/vllm-openai-rocm-r9700) (AITER-tuned image) | This image uses AITER only for attention. Its verify-batch decode config is tuned here (`sly/radiance_attn_decode.py`), and a 36-cell sweep of the 2D prefill config found aiter's stock cell within 1–5 % of the best (< 2 % of a 64k prefill); the GEMMs run on the MXFP4 HIP kernels, not AITER. Not evaluated in depth beyond that. |
| [hifi/vllm-radlight](https://codeberg.org/hifi/vllm-radlight) | An arrangement of ggz14 + libr4d on AMD's vLLM 0.27 image, patched at start (no original code). Its knobs are the upstream ones above; the remaining runtime flags (expandable segments, chunk 2560, fp16 SSM state, HW queues, HSA interrupts) are queued for an A/B. |
| [bkvargyas/r9700-stack](https://github.com/bkvargyas/r9700-stack) | Plugin stack for TP = 2 and NVFP4 checkpoints; its int6 embedding gather is behind this image's int4 embedding (`RADIANCE_EMBED_BITS=4`). |
| `tcclaviger/vllm`, `Dyluhn/R9V` | Separate forks with their own model mix (MoE, expert offload, TP ≥ 2); nothing single-GPU-MXFP4-specific to take over was found. |
| DFlash2-FP8 drafter (`tcclaviger/Qwen3.8-27B-DFlash2-FP8`) | Heavier than the W4A16 drafter, and stacks running it report fewer tokens per update than this image (code 4.71 vs 5.09). |
| Qwen3.8-27B-PARO-MXFP6 | Two GPUs only. |

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
first and last, second start, ≥ 128 BetterBench runs per arm or the step gap; KV pool must stay 384,316):

| Arm | Paiton setting | Ours today | Why it might help |
|---|---|---|---|
| A1 | `GPU_MAX_HW_QUEUES=1` | `2` | Paiton: "removes a slower decode mode some fresh processes start in"; ours picked 2 against the random default, 1 was not in that sweep |
| A2 | `draft_sample_method: greedy` | `probabilistic` | lossless either way; greedy drafts skip the draft-side sampling, and the lookup draft already writes point masses |
| A3 | `tcclaviger/Qwen3.8-27B-DFlash2-FP8` drafter | `syvai/...-W4A16` | Paiton's tokens/update (json ≈ 7.2, code ≈ 6.0) are well above ours (5.66 / 4.94); the old "fewer tokens per update" row above came from other stacks, re-check on this image. Costs KV (heavier drafter) |
| A4 | `--mamba-ssm-cache-dtype float16` | `bfloat16` | fp16 state: more mantissa, same bytes; the libr4d extras have fp16 kernels (also on the radlight list) |
| A5 | `--attention-backend R4D` | `ROCM_AITER_UNIFIED_ATTN` (quickstart) | Paiton and `serve-mxfp4.sh` both use R4D for the target |
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
