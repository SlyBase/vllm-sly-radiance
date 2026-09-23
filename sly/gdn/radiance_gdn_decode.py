"""torch.ops._C.fused_gdn_decode_post_conv_mtp on gfx1201 (RADIANCE_GDN_FUSED_DECODE=1, default off).

vLLM 0.29 ships a fused CUDA kernel for the post-conv half of the gated-delta-net decode of a
speculative verify batch and switches its whole GDN forward to it (VLLM_GDN_DECODE_KERNEL=cuda, the
default) -- but only when `torch.ops._C.fused_gdn_decode_post_conv_mtp` exists, and vLLM's CMake builds
it for NVIDIA archs only. On ROCm the layer logged "Falling back to the Triton GDN decode path: ... is
not built" and ran the FLA update kernel plus its glue.

This registers the op under the name vLLM probes, backed by radiance_gdn_decode_ext (the HIP port in
radiance_gdn_decode.hip). Imported at the top of qwen_gdn_linear_attn.py (sly/patch_gdn_fused_decode.py)
so it exists before any layer runs its `_fused_gdn_decode_unsupported_reason` check.

Off by default (image-neutral): without RADIANCE_GDN_FUSED_DECODE=1 the op is not registered and vLLM
logs its usual Triton fallback. Serve it with RADIANCE_GDN_FUSED_DECODE=1 VLLM_GDN_DECODE_KERNEL=cuda --
the explicit kernel choice also keys vLLM's torch.compile cache (VLLM_* env vars are hashed), which the
default value would not, and makes vLLM fail loudly instead of falling back if the op is missing.
A/B inside one image: VLLM_GDN_DECODE_KERNEL=triton keeps vLLM on the Triton path.
"""
import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_GDN_FUSED_DECODE", "0") == "1"
_K = _V = 128
_MAX_TOKENS = 8
_STATE = {"ext": None, "logged": False}


def _ext():
    if _STATE["ext"] is None:
        import radiance_gdn_decode_ext
        _STATE["ext"] = radiance_gdn_decode_ext
    return _STATE["ext"]


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(f"radiance fused_gdn_decode_post_conv_mtp: {msg}")


def _impl(mixed_qkv, a, b, A_log, dt_bias, state_indices, cu_seqlens, num_accepted_tokens, state,
          output_gate, norm_weight, out, scale, norm_eps, output_gate_activation):
    bf16, f32 = torch.bfloat16, torch.float32
    _check(mixed_qkv.dtype == bf16 and a.dtype == bf16 and b.dtype == bf16, "mixed_qkv/a/b must be bf16")
    _check(A_log.dtype == f32, "A_log must be float32")
    _check(dt_bias.dtype in (f32, bf16, torch.float16), "dt_bias must be f32/bf16/f16")
    _check(state_indices.dtype == torch.int32 and cu_seqlens.dtype == torch.int32
           and num_accepted_tokens.dtype == torch.int32, "index tensors must be int32")
    _check(state.dtype in (f32, bf16), "state must be f32 or bf16")
    _check(output_gate.dtype == bf16 and out.dtype == bf16, "output_gate/out must be bf16")
    _check(norm_weight.dtype in (f32, bf16), "norm_weight must be f32 or bf16")
    _check(output_gate_activation in ("silu", "sigmoid"), "activation must be silu or sigmoid")
    _check(mixed_qkv.dim() == 2, "mixed_qkv must be [L, 2*H*128 + HV*128]")
    L = mixed_qkv.size(0)
    _check(L > 0, "needs at least one token")
    _check(state.dim() == 4 and state.size(2) == _V and state.size(3) == _K, "state must be [slots, HV, 128, 128]")
    HV = state.size(1)
    key_width = mixed_qkv.size(1) - HV * _V
    _check(key_width > 0 and key_width % (2 * _K) == 0, "mixed_qkv width inconsistent with state")
    H = key_width // (2 * _K)
    _check(HV % H == 0 and HV // H in (1, 2, 3, 4, 8), "HV/H must be in {1, 2, 3, 4, 8}")
    _check(state_indices.dim() == 2 and state_indices.size(0) > 0
           and 0 < state_indices.size(1) <= _MAX_TOKENS, "state_indices must be [N, S], 1 <= S <= 8")
    N = state_indices.size(0)
    _check(cu_seqlens.dim() == 1 and cu_seqlens.numel() == N + 1, "cu_seqlens must have N + 1 elements")
    _check(num_accepted_tokens.dim() == 1 and num_accepted_tokens.numel() == N, "num_accepted_tokens must have N")
    _check(a.dim() == 2 and a.size(0) == L and a.size(1) == HV, "a must be [L, HV]")
    _check(b.dim() == 2 and b.size(0) == L and b.size(1) == HV, "b must be [L, HV]")
    _check(A_log.is_contiguous() and A_log.numel() == HV, "A_log must be contiguous [HV]")
    _check(dt_bias.is_contiguous() and dt_bias.numel() == HV, "dt_bias must be contiguous [HV]")
    _check(state_indices.is_contiguous() and cu_seqlens.is_contiguous()
           and num_accepted_tokens.is_contiguous(), "index tensors must be contiguous")
    _check(output_gate.dim() == 3 and output_gate.size(0) == L and output_gate.size(1) == HV
           and output_gate.size(2) == _V, "output_gate must be [L, HV, 128]")
    _check(norm_weight.is_contiguous() and norm_weight.numel() == _V, "norm_weight must be contiguous [128]")
    _check(out.dim() == 3 and out.size(0) == L and out.size(1) == HV and out.size(2) == _V
           and out.is_contiguous(), "out must be contiguous [L, HV, 128]")
    _check(mixed_qkv.stride(1) == 1, "mixed_qkv channels must be contiguous")
    _check(a.stride(1) == 1 and b.stride(1) == 1, "a and b heads must be contiguous")
    _check(state.stride(0) >= HV * _V * _K and state.stride(1) == _V * _K and state.stride(2) == _K
           and state.stride(3) == 1, "state slots must be contiguous [HV, 128, 128]")
    per_copy = 4 if state.dtype == f32 else 8
    _check(state.data_ptr() % 16 == 0 and state.stride(0) % per_copy == 0, "state slots must be 16-byte aligned")
    _check(output_gate.stride(2) == 1 and output_gate.stride(1) == _V, "output_gate head rows must be contiguous")
    _check(norm_eps >= 0.0, "norm_eps must be non-negative")
    dtb = 0 if dt_bias.dtype == f32 else (1 if dt_bias.dtype == bf16 else 2)
    if not _STATE["logged"]:
        _STATE["logged"] = True
        sys.stderr.write(f"[radiance.gdn_decode] fused GDN MTP decode active: N={N} L={L} H={H} HV={HV} "
                         f"S={state_indices.size(1)} state={str(state.dtype).split('.')[-1]} "
                         f"gate={output_gate_activation}\n")
        sys.stderr.flush()
    _ext().launch(mixed_qkv.data_ptr(), a.data_ptr(), b.data_ptr(), A_log.data_ptr(), dt_bias.data_ptr(),
                  state_indices.data_ptr(), cu_seqlens.data_ptr(), num_accepted_tokens.data_ptr(),
                  state.data_ptr(), output_gate.data_ptr(), norm_weight.data_ptr(), out.data_ptr(),
                  N, H, HV, state_indices.size(1), dtb, norm_weight.dtype == bf16, float(scale),
                  float(norm_eps), mixed_qkv.stride(0), a.stride(0), b.stride(0), output_gate.stride(0),
                  state.stride(0), state.dtype == f32, output_gate_activation == "sigmoid",
                  torch.cuda.current_stream().cuda_stream)


_SCHEMA = ("fused_gdn_decode_post_conv_mtp(Tensor mixed_qkv, Tensor a, Tensor b, Tensor A_log, "
           "Tensor dt_bias, Tensor state_indices, Tensor cu_seqlens, Tensor num_accepted_tokens, "
           "Tensor(a!) state, Tensor output_gate, Tensor norm_weight, Tensor(b!) out, float scale, "
           "float norm_eps, str output_gate_activation) -> ()")
_LIB = None


def install() -> bool:
    """Register the op once. True when torch.ops._C.fused_gdn_decode_post_conv_mtp exists afterwards."""
    global _LIB
    if hasattr(torch.ops._C, "fused_gdn_decode_post_conv_mtp"):
        return True
    if not ENABLED:
        return False
    try:
        _ext()
    except Exception as e:  # noqa: BLE001 -- never block model import on our extension
        sys.stderr.write(f"[radiance.gdn_decode] extension unavailable, vLLM stays on Triton: {e!r}\n")
        return False
    _LIB = torch.library.Library("_C", "FRAGMENT")
    _LIB.define(_SCHEMA)
    _LIB.impl("fused_gdn_decode_post_conv_mtp", _impl, "CUDA")
    torch.library.register_fake("_C::fused_gdn_decode_post_conv_mtp", lib=_LIB)(lambda *args: None)
    sys.stderr.write("[radiance.gdn_decode] torch.ops._C.fused_gdn_decode_post_conv_mtp registered (HIP port)\n")
    return True
