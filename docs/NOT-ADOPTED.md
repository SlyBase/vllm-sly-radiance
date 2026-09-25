# Considered and not adopted

What other R9700 stacks do that this image deliberately does not, with the reason.

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

**Other images and forks:**

| Source | Why not |
|---|---|
| [testeddoughnut/vllm-openai-rocm-r9700](https://hub.docker.com/r/testeddoughnut/vllm-openai-rocm-r9700) (AITER-tuned image) | This image uses AITER only for attention. Its verify-batch decode config is tuned here (`sly/radiance_attn_decode.py`), and a 36-cell sweep of the 2D prefill config found aiter's stock cell within 1–5 % of the best (< 2 % of a 64k prefill); the GEMMs run on the MXFP4 HIP kernels, not AITER. Not evaluated in depth beyond that. |
| [hifi/vllm-radlight](https://codeberg.org/hifi/vllm-radlight) | An arrangement of ggz14 + libr4d on AMD's vLLM 0.27 image, patched at start (no original code). Its knobs are the upstream ones above; the remaining runtime flags (expandable segments, chunk 2560, fp16 SSM state, HW queues, HSA interrupts) are queued for an A/B. |
| [bkvargyas/r9700-stack](https://github.com/bkvargyas/r9700-stack) | Plugin stack for TP = 2 and NVFP4 checkpoints; its int6 embedding gather is behind this image's int4 embedding (`RADIANCE_EMBED_BITS=4`). |
| `tcclaviger/vllm`, `Dyluhn/R9V` | Separate forks with their own model mix (MoE, expert offload, TP ≥ 2); nothing single-GPU-MXFP4-specific to take over was found. |
| DFlash2-FP8 drafter (`tcclaviger/Qwen3.8-27B-DFlash2-FP8`) | Heavier than the W4A16 drafter, and stacks running it report fewer tokens per update than this image (code 4.71 vs 5.09). |
| Qwen3.8-27B-PARO-MXFP6 | Two GPUs only. |
