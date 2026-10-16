#!/usr/bin/env python3
"""vLLM's int4_per_token_head kernel on ROCm: bf16 dot operands and a matmul Hadamard for D > 128.

Used by the DFlash drafter's five sliding-window layers, which stay on TRITON_ATTN with vLLM's own
int4 mode when the target runs KV4 (the --kv-cache-dtype is one string for the whole model). As
shipped (vLLM 0.31) that kernel computes its dots in fp32 and, for head sizes above its Triton
Hadamard's 128-row limit, rotates with a torch butterfly; on the R9700 that measured 10x the fp8
Triton kernel. The two edits below bring it to 2-3x (kv4_check, 2026-10-09):

  1. the four tl.dot calls take bf16 operands: the nibbles 0..15 and the bf16 query are exact in
     bf16, P loses what the fp8 kernel's P cast loses anyway; accumulation stays fp32.
  2. D in (128, 1024]: one bf16 matmul against the cached Hadamard matrix instead of the butterfly.
  3. non-causal attention: the DFlash drafter attends non-causally inside its draft block
     (causal=False, keys allowed when < seq_len and inside the window); the kernel only ever
     computed the causal mask and unified_attention asserted on anything else. `causal` is now
     threaded through to the mask helper (which already knows USE_CAUSAL) and the window pre-mask
     on V mirrors the core kernel's two-sided window for the non-causal case.

Idempotent; anchors on the 0.31 source. Upstream candidate.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/attention/ops/int4_per_token_head.py"

apply(F,
      "        K_s0 = K_s0_u.to(tl.float32)\n        K_s1 = K_s1_u.to(tl.float32)\n",
      "        K_s0 = K_s0_u.to(tl.bfloat16)\n        K_s1 = K_s1_u.to(tl.bfloat16)\n",
      "K_s0 = K_s0_u.to(tl.bfloat16)", "int4-pth: K operands bf16")
apply(F,
      "        V_s0 = V_s0_u.to(tl.float32)\n        V_s1 = V_s1_u.to(tl.float32)\n",
      "        V_s0 = V_s0_u.to(tl.bfloat16)\n        V_s1 = V_s1_u.to(tl.bfloat16)\n",
      "V_s0 = V_s0_u.to(tl.bfloat16)", "int4-pth: V operands bf16")
apply(F,
      "    Q_sum = tl.sum(Q_s0, axis=1) + tl.sum(Q_s1, axis=1)\n",
      "    Q_sum = tl.sum(Q_s0, axis=1) + tl.sum(Q_s1, axis=1)\n"
      "    Q_s0_b = Q_s0.to(tl.bfloat16)\n    Q_s1_b = Q_s1.to(tl.bfloat16)\n",
      "Q_s0_b = Q_s0.to(tl.bfloat16)", "int4-pth: Q operands bf16")
apply(F,
      "        raw_dot = tl.dot(Q_s0, K_s0) + tl.dot(Q_s1, K_s1)\n",
      "        raw_dot = tl.dot(Q_s0_b, K_s0) + tl.dot(Q_s1_b, K_s1)\n",
      "tl.dot(Q_s0_b, K_s0)", "int4-pth: QK dot bf16")
apply(F,
      "        acc_s0 += tl.dot(P_v, V_s0) - Pv_zp_sum[:, None]\n"
      "        acc_s1 += tl.dot(P_v, V_s1) - Pv_zp_sum[:, None]\n",
      "        P_v_b = P_v.to(tl.bfloat16)\n"
      "        acc_s0 += tl.dot(P_v_b, V_s0) - Pv_zp_sum[:, None]\n"
      "        acc_s1 += tl.dot(P_v_b, V_s1) - Pv_zp_sum[:, None]\n",
      "P_v_b = P_v.to(tl.bfloat16)", "int4-pth: PV dot bf16")
apply(F,
      "    # Tier 3 — PyTorch butterfly (CPU / unsupported dtype / D < 16).\n",
      "    # Tier 2b (radiance) — plain bf16 matmul against the cached Hadamard matrix for D above\n"
      "    # the Triton kernel's LDS budget (head_size 256): fp32 accumulate, one GEMM per call.\n"
      "    if x.is_cuda and _TRITON_HADAMARD_MAX_D < d <= 1024 and x.dtype in (torch.float16, torch.bfloat16, torch.float32):\n"
      "        work = torch.bfloat16 if x.dtype == torch.float32 else x.dtype\n"
      "        H_mat = _get_hadamard_matrix(d, work, x.device)\n"
      "        return (x.to(work).reshape(-1, d) @ H_mat).reshape(x.shape).to(x.dtype)\n\n"
      "    # Tier 3 — PyTorch butterfly (CPU / unsupported dtype / D < 16).\n",
      "Tier 2b (radiance)", "int4-pth: matmul Hadamard for D > 128")

# ---- 3. causal=False for the drafter ----------------------------------------------------------
apply(F,
      "    SLIDING_WINDOW: tl.constexpr,\n    USE_MM_PREFIX: tl.constexpr,\n    MAX_MM_RANGES: tl.constexpr,\n    mm_prefix_range_ptr,\n",
      "    SLIDING_WINDOW: tl.constexpr,\n    USE_MM_PREFIX: tl.constexpr,\n    MAX_MM_RANGES: tl.constexpr,\n    mm_prefix_range_ptr,\n"
      "    USE_CAUSAL: tl.constexpr,\n",
      "    USE_CAUSAL: tl.constexpr,\n", "int4-pth: USE_CAUSAL kernel arg")
apply(F,
      "            SLIDING_WINDOW,\n            USE_MM_PREFIX,\n            MAX_MM_RANGES,\n        )\n",
      "            SLIDING_WINDOW,\n            USE_MM_PREFIX,\n            MAX_MM_RANGES,\n            USE_CAUSAL=USE_CAUSAL,\n        )\n",
      "            USE_CAUSAL=USE_CAUSAL,\n", "int4-pth: mask honours USE_CAUSAL")
apply(F,
      "            sw_mask = (context_len + qpos_lo - seq_offset) < SLIDING_WINDOW\n",
      "            dist = context_len + qpos_lo - seq_offset\n"
      "            if USE_CAUSAL:\n"
      "                sw_mask = dist < SLIDING_WINDOW\n"
      "            else:\n"
      "                sw_mask = (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW)\n",
      "sw_mask = (dist < SLIDING_WINDOW) & (dist > -SLIDING_WINDOW)", "int4-pth: two-sided window pre-mask when non-causal")
apply(F,
      "    softmax_segm_expsum,\n    packing_factor: int,\n):\n",
      "    softmax_segm_expsum,\n    packing_factor: int,\n    causal: bool = True,\n):\n",
      "    causal: bool = True,\n):\n", "int4-pth: launcher takes causal")
apply(F,
      "        SLIDING_WINDOW=(1 + window_size[0]),\n        USE_MM_PREFIX=use_mm_prefix,\n",
      "        SLIDING_WINDOW=(1 + window_size[0]),\n        USE_CAUSAL=causal,\n        USE_MM_PREFIX=use_mm_prefix,\n",
      "        USE_CAUSAL=causal,\n", "int4-pth: launch passes USE_CAUSAL")
apply(F,
      "    mm_prefix_range: torch.Tensor | None,\n    k_scale_cache: torch.Tensor,\n    v_scale_cache: torch.Tensor,\n    seq_threshold_3D: int | None = None,\n",
      "    mm_prefix_range: torch.Tensor | None,\n    k_scale_cache: torch.Tensor,\n    v_scale_cache: torch.Tensor,\n    causal: bool = True,\n    seq_threshold_3D: int | None = None,\n",
      "    causal: bool = True,\n    seq_threshold_3D", "int4-pth: unified_attention_int4 takes causal")
apply(F,
      "        softmax_segm_expsum=softmax_segm_expsum,\n        packing_factor=_INT4_PACKING_FACTOR,\n",
      "        softmax_segm_expsum=softmax_segm_expsum,\n        packing_factor=_INT4_PACKING_FACTOR,\n        causal=causal,\n",
      "        causal=causal,\n", "int4-pth: entry point forwards causal")
U = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/attention/ops/triton_unified_attention.py"
apply(U,
      '        assert use_causal and not use_per_seq_causal, (\n            "INT4_PER_TOKEN_HEAD only supports causal attention"\n        )\n',
      '        assert not use_per_seq_causal, (\n            "INT4_PER_TOKEN_HEAD does not support per-sequence causal"\n        )\n',
      '"INT4_PER_TOKEN_HEAD does not support per-sequence causal"', "int4-pth: unified_attention accepts causal=False")
apply(U,
      "            mm_prefix_range=mm_prefix_range,\n            k_scale_cache=k_scale_cache,\n            v_scale_cache=v_scale_cache,\n            seq_threshold_3D=seq_threshold_3D,\n",
      "            mm_prefix_range=mm_prefix_range,\n            k_scale_cache=k_scale_cache,\n            v_scale_cache=v_scale_cache,\n            causal=use_causal,\n            seq_threshold_3D=seq_threshold_3D,\n",
      "            causal=use_causal,\n            seq_threshold_3D", "int4-pth: unified_attention forwards causal")
