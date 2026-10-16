# MiniCPM-V 4.7 35B-A3B on one R9700

Serving recipe and measurements for `openbmb/MiniCPM-V-4.7-35B-A3B` (GPTQ W4A16 compressed-tensors checkpoint)
with its DSpark drafter (`openbmb/MiniCPM-V-4.7-35B-A3B-DSpark`). Needs `RADIANCE_MINICPMV4_7=1` (model backport)
and `transformers>=5.18` for the processor (see the CHANGELOG of 1.4.0). All numbers: BetterBench `--decode`,
210 W power cap, 64k context, one stream, 2026-10-14/16.

## Arguments

```bash
-e RADIANCE_MINICPMV4_7=1 \
-e RADIANCE_USE_R4D=1 -e RADIANCE_LMHEAD_INT4=1 \
-e RADIANCE_NVFP4_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
-e RADIANCE_MXFP4_DECODE_MAX_M=128 -e RADIANCE_MXFP4_A_TILED_MIN_M=513 -e RADIANCE_MXFP4_WIDE_MAX_M=192 \
-e RADIANCE_MXFP4_SANITIZE=0 -e RADIANCE_FUSED_NORM_QUANT=1 -e RADIANCE_MXFP4_WPERM=1 \
-e RADIANCE_NVFP4_BF16_LAYERS='(language_model.+(linear_attn.(in_proj_qkvz|out_proj)|self_attn.(qkv_proj|o_proj)|shared_expert.(gate_up_proj|down_proj)))|(^model.(layers.[0-9]+.(self_attn.(qkv_proj|o_proj)|mlp.(gate_up_proj|down_proj))|fc)$)' \
-e GPU_MAX_HW_QUEUES=1 \
... --speculative-config '{"method":"dspark","model":"<drafter dir>","num_speculative_tokens":8,"quantization":"compressed-tensors"}' \
--attention-backend ROCM_AITER_UNIFIED_ATTN \
--compilation-config.cudagraph_mode FULL_DECODE_ONLY --compilation-config.cudagraph_capture_sizes [8,16,32,48,64,96,128] \
--max-model-len 65536 --gpu-memory-utilization 0.98 --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
--max-num-seqs 8 --max-num-batched-tokens 2048 --skip-mm-profiling \
--default-chat-template-kwargs '{"enable_thinking": false}'
```

- `RADIANCE_NVFP4_BF16_LAYERS`: the GPTQ checkpoint keeps GDN, attention, shared expert (and the drafter) in bf16;
  this converts them to MXFP4 at load. The drafter part needs a copy of the drafter dir whose `config.json` carries
  `"quantization_config": {"quant_method":"compressed-tensors","format":"pack-quantized","config_groups":{},"ignore":[]}`
  (DSpark gets `quant_config=None` otherwise, and no regex reaches it). Marker: `linear layers: 181/181`.
- `RADIANCE_MXFP4_W4A8_MIN_M=0` is required: with the default 256 every decode GEMM goes to aiter (74 ms eager step,
  hang in CUDA-graph capture).
- The tuned int4 MoE config `moe-configs/E=256,N=512,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json` ships in
  the image (marker: `Using configuration from ... for MoE layer`).

## Measured steps (BetterBench --decode, weighted tok/s; Qwen3.8 + DFlash2 production = 132.4 in the same window)

| Step | Decode | Step gap |
|---|---:|---:|
| GPTQ, graphs, tuned MoE, k=8 | 112.5 | 38.0 ms |
| + bf16 linears to MXFP4 at load | 140.9 | 34.2 ms |
| + drafter linears to MXFP4 | 146.8 | 32.8 ms |
| **+ `--attention-backend ROCM_AITER_UNIFIED_ATTN`** | **169.4** (reproduced 169.4, 169.3) | 26.2 ms |

GSM8K 200 (flexible extract): 0.835 with MXFP4 + AITER vs 0.85 for the bf16 base.
MoE config alone: 1.2-1.7x per MoE call, ~+8 % end-to-end.

Why the attention backend matters: the default `ROCM_ATTN` runs the k+1 verify rows through the prefill
`_fwd_kernel` (3.9 ms per step); AITER's unified kernel needs ~0.16 ms.

Not adopted for this model: see [NOT-ADOPTED.md](NOT-ADOPTED.md#minicpm-v-47-2026-10).
