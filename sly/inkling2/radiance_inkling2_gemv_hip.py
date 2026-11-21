"""int8 (Q8_0) weight HIP GEMV dispatch for Inkling 2's low-M bf16 projections on gfx12 (RDNA4).

The patch hook in ops/gemv.py (sly/patch_inkling2_gfx12.py) calls hip_gemv() before vLLM's Triton
_gemv_kernel; False means "not handled", Triton on the original bf16 weight stays the fallback.
Only the quantized arm lives here: vLLM's Triton bf16 GEMV already streams at DRAM bandwidth, so a
bf16 HIP kernel cannot win; fewer bytes can.

Opt-in RADIANCE_INKLING2_W8=1 (default 0): big projection weights (>= Q8_MIN_NUMEL = 3M elements; NOT
the MoE router, NOT tiny tensors) are converted once, lazily, on the first eager call to int8
with one fp32 scale per 32 consecutive K values (symmetric, scale = amax / 127, round to
nearest) = ggml Q8_0 with an fp32 scale; activations stay bf16 -> fp32. Cost: 1.125 B/weight
on top of the bf16 original, which stays alive for M > Q8_MAX_M (prefill, where the kernel
loses to Triton). Conversion allocates, so it never runs under graph capture: vLLM's eager
warmup of each capture size converts first; a weight first seen during capture runs through
Triton instead.

The module name radiance_inkling2_gemv belongs to the .so (CPython prefers extensions over .py of
the same name), hence this file's _hip suffix. torch is imported first: a bare HIP extension
cannot resolve libamdhip64 without torch's runtime in the process.
"""
import os

import torch

W8 = os.environ.get("RADIANCE_INKLING2_W8", "0") == "1"
Q8_MIN_NUMEL = 3 << 20  # k/v 1024 x 2304 (2.4M) gains <= 5 %: stays on Triton; router 0.3M too
Q8_BLOCK = 32  # K values per scale
MAX_M = 16  # kernel range
Q8_MAX_M = 4  # routed up to here: 1.25-1.88x over Triton at M 1-4; at M 5+ the sweep shows <= 1.2x and losses (commit message)
_CONV_ROWS = 16384  # conversion chunk: bounds the fp32 temporaries (unembed: 5.5 GB at once)

try:
    import radiance_inkling2_gemv as _ext
except ImportError:
    _ext = None

# (data_ptr, K) -> (weight_ref, wq int8 [N, K], scale fp32 [N, K/32]). The ref keeps the address
# from being reused by a different tensor; a row-0 prefix slice (global layers' q-side of the
# fused qkvr) reuses the entry instead of quantizing the same rows twice.
_Q8_CACHE: dict = {}
_SCRATCH = None  # split-K fp32 partials, 4 MB


def to_q8(weight: torch.Tensor):
    """bf16 [N, K] -> (int8 [N, K], fp32 [N, K/32] scale = block amax / 127), K % 32 == 0."""
    n, k = weight.shape
    wq = torch.empty((n, k), dtype=torch.int8, device=weight.device)
    scale = torch.empty((n, k // Q8_BLOCK), dtype=torch.float32, device=weight.device)
    for i in range(0, n, _CONV_ROWS):
        wf = weight[i:i + _CONV_ROWS].float().reshape(-1, k // Q8_BLOCK, Q8_BLOCK)
        s = wf.abs().amax(dim=2) / 127.0
        q = torch.where(s[..., None] > 0, wf / s[..., None], 0.0).round().clamp(-127, 127)
        wq[i:i + _CONV_ROWS] = q.to(torch.int8).reshape(-1, k)
        scale[i:i + _CONV_ROWS] = s
    return wq, scale


def _q8_for(weight: torch.Tensor):
    key = (weight.data_ptr(), weight.shape[1])
    ent = _Q8_CACHE.get(key)
    n = weight.shape[0]
    if ent is not None and ent[1].shape[0] >= n:
        return ent[1][:n], ent[2][:n]
    wq, scale = to_q8(weight)
    _Q8_CACHE[key] = (weight, wq, scale)
    return wq, scale


def _want_q8(weight: torch.Tensor) -> bool:
    return W8 and weight.dtype == torch.bfloat16 and weight.numel() >= Q8_MIN_NUMEL


def _scratch():
    """(ptr, bytes) of the split-K scratch buffer, or (0, 0) under graph capture."""
    global _SCRATCH
    if _SCRATCH is None:
        if torch.cuda.is_current_stream_capturing():
            return 0, 0
        _SCRATCH = torch.empty(1 << 20, dtype=torch.float32, device="cuda")
    return _SCRATCH.data_ptr(), _SCRATCH.numel() * 4


def hip_gemv(x2: torch.Tensor, weight: torch.Tensor, bias, out: torch.Tensor) -> bool:
    """out = x2 @ weight.T (+ bias) via the int8 copy of `weight`; True if handled."""
    if _ext is None or not _want_q8(weight):
        return False
    m, k = x2.shape
    n = weight.shape[0]
    if not (
        0 < m <= Q8_MAX_M
        and weight.ndim == 2
        and weight.shape[1] == k
        and x2.dtype == torch.bfloat16
        and out.dtype in (torch.bfloat16, torch.float32)
        and x2.stride(-1) == 1
        and k % 32 == 0  # one scale per 32 K values, 16 B vector weight loads
        and x2.stride(0) % 8 == 0  # 16 B vector x staging
        and x2.data_ptr() % 16 == 0
        and (bias is None or (bias.dtype == torch.bfloat16 and bias.stride(0) == 1))
    ):
        return False
    ent = _Q8_CACHE.get((weight.data_ptr(), k))
    if ent is not None and ent[1].shape[0] >= n:
        wq, scale = ent[1][:n], ent[2][:n]
    elif not torch.cuda.is_current_stream_capturing():
        wq, scale = _q8_for(weight)
    else:
        return False
    scratch, scratch_bytes = _scratch()
    _ext.gemv_q8(x2.data_ptr(), wq.data_ptr(), scale.data_ptr(),
                 bias.data_ptr() if bias is not None else 0, out.data_ptr(),
                 m, n, k, x2.stride(0), wq.stride(0), out.stride(0),
                 int(out.dtype == torch.float32), scratch, scratch_bytes,
                 torch.cuda.current_stream().cuda_stream, -1)
    return True
