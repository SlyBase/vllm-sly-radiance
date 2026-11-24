"""HIP GEMV dispatch for Inkling 2's low-M projections on gfx12 (RDNA4): bf16 (default) and int8 (opt-in).

The patch hook in ops/gemv.py (sly/patch_inkling2_gfx12.py) calls hip_gemv() before vLLM's Triton
_gemv_kernel; False means "not handled", the Triton kernel on the original bf16 weight stays the
fallback.

bf16 (default on, RADIANCE_INKLING2_HIP_GEMV=0 = Triton): the model's own bf16 weight, M 1..16, fp32
accumulation (lossless). Triton pads M to 16 rows and its tl.dot streams a spec-verify width (M=8)
slower than the card can; the HIP kernel reads each weight row once with 16 B non-temporal loads and
accumulates all M rows per load. No copy, no scratch: it routes under graph capture too.

loop controllers: opt-in RADIANCE_INKLING2_HIP_CTRL=1 (default 0): the two torch.bmm of the loop blend
(hipBLASLt, 0.92 ms per verify step) run as two launches of gemv_bf16_batched (stream = grid.y), see
controller_gemms. Same bf16 weights, fp32 accumulation; only the summation order differs.

int8: opt-in RADIANCE_INKLING2_W8=1 (default 0): big projection weights (>= Q8_MIN_NUMEL = 3M elements; NOT
the MoE router, NOT tiny tensors) are converted once, lazily, on the first eager call to int8
with one fp32 scale per 32 consecutive K values (symmetric, scale = amax / 127, round to
nearest) = ggml Q8_0 with an fp32 scale; activations stay bf16 -> fp32. The kernel reads a tiled copy
(pack_q8) and serves M 1..16 (a DSpark verify step is k + 1 rows). RADIANCE_INKLING2_W8_HEAD=0 keeps
the vocab-sized LM head (>= Q8_HEAD_MIN_N rows) on bf16 (default 1: head quantized too). Cost:
1.125 B/weight on top of the bf16 original, which stays alive for M > 16 (prefill). Conversion
allocates, so it never runs under graph capture: vLLM's eager warmup of each capture size converts
first; a weight first seen during capture runs through Triton instead. No split-K scratch.

The module name radiance_inkling2_gemv belongs to the .so (CPython prefers extensions over .py of
the same name), hence this file's _hip suffix. torch is imported first: a bare HIP extension
cannot resolve libamdhip64 without torch's runtime in the process.
"""
import os

import torch

W8 = os.environ.get("RADIANCE_INKLING2_W8", "0") == "1"
W8_HEAD = os.environ.get("RADIANCE_INKLING2_W8_HEAD", "1") != "0"  # only read when W8 is on
HIP_BF16 = os.environ.get("RADIANCE_INKLING2_HIP_GEMV", "1") != "0"
HIP_CTRL = os.environ.get("RADIANCE_INKLING2_HIP_CTRL", "0") == "1"  # loop controllers on the batched GEMV
Q8_MIN_NUMEL = 3 << 20  # k/v 1024 x 2304 (2.4M, launch bound, 1.4x) and the router stay bf16 by choice
Q8_HEAD_MIN_N = 1 << 17  # rows: the 201024-row unembed is the only projection this tall
Q8_BLOCK = 32  # K values per scale
Q8_TILE = 16  # weight rows per kernel tile (one wmma operand)
Q8_GROUP = 64  # K values per tile group = 2 scale blocks; the kernel reads whole groups
MAX_M = 16  # kernel range
_CONV_ROWS = 16384  # conversion chunk: bounds the fp32 temporaries (unembed: 5.5 GB at once); multiple of Q8_TILE

try:
    import radiance_inkling2_gemv as _ext
except ImportError:
    _ext = None

# (data_ptr, K) -> (weight_ref, tiled int8 [NT, K/64, 1024], tiled fp32 scale [NT, K/64, 32]). The ref
# keeps the address from being reused by a different tensor; a row-0 prefix slice (global layers'
# q-side of the fused qkvr) reuses the entry instead of quantizing the same rows twice.
_Q8_CACHE: dict = {}


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


def pack_q8(wq: torch.Tensor, scale: torch.Tensor):
    """(int8 [N, K], fp32 [N, K/32]) -> the kernel's tiles: int8 [NT, K/64, 1024], fp32 [NT, K/64, 32].

    Per (tile of 16 rows, group of 64 K) the int8 bytes are ordered [block b][half h][row c][16 B] (the
    order lane 16 h + c of a wave loads them, 512 contiguous bytes per wave load) and the fp32 scales
    [row c][block b]. Rows past N (last tile) are zero: code 0, scale 0."""
    n, k = wq.shape
    assert n > 0 and k > 0 and k % Q8_GROUP == 0 and tuple(scale.shape) == (n, k // Q8_BLOCK)
    nt, kg = -(-n // Q8_TILE), k // Q8_GROUP
    if nt * Q8_TILE != n:
        wq = torch.cat([wq, wq.new_zeros(nt * Q8_TILE - n, k)])
        scale = torch.cat([scale, scale.new_zeros(nt * Q8_TILE - n, k // Q8_BLOCK)])
    qt = wq.reshape(nt, Q8_TILE, kg, 2, 2, 16).permute(0, 2, 3, 4, 1, 5).reshape(nt, kg, -1)
    st = scale.reshape(nt, Q8_TILE, kg, 2).permute(0, 2, 1, 3).reshape(nt, kg, -1)
    return qt.contiguous(), st.contiguous()


def _q8_for(weight: torch.Tensor):
    """Tiled int8 copy of `weight` (cached): (qt, st) with at least ceil(N / 16) tiles."""
    key = (weight.data_ptr(), weight.shape[1])
    ent = _Q8_CACHE.get(key)
    n, k = weight.shape
    nt = -(-n // Q8_TILE)
    if ent is not None and ent[1].shape[0] >= nt:
        return ent[1][:nt], ent[2][:nt]
    qt = torch.empty((nt, k // Q8_GROUP, Q8_GROUP * Q8_TILE), dtype=torch.int8, device=weight.device)
    st = torch.empty((nt, k // Q8_GROUP, Q8_GROUP // Q8_BLOCK * Q8_TILE), dtype=torch.float32, device=weight.device)
    step = _CONV_ROWS // Q8_TILE
    for i in range(0, n, _CONV_ROWS):
        qt[i // Q8_TILE:i // Q8_TILE + step], st[i // Q8_TILE:i // Q8_TILE + step] = pack_q8(*to_q8(weight[i:i + _CONV_ROWS]))
    _Q8_CACHE[key] = (weight, qt, st)
    print(f"radiance_inkling2: int8 copy of the {n}x{k} bf16 weight ({nt} tiles)", flush=True)
    return qt, st


def _want_q8(weight: torch.Tensor) -> bool:
    return (W8 and weight.dtype == torch.bfloat16 and weight.numel() >= Q8_MIN_NUMEL
            and (W8_HEAD or weight.shape[0] < Q8_HEAD_MIN_N))


def _aligned(t: torch.Tensor) -> bool:
    return t.data_ptr() % 16 == 0


def _hip_bf16(x2: torch.Tensor, weight: torch.Tensor, bias, out: torch.Tensor) -> bool:
    """out = x2 @ weight.T (+ bias) reading `weight` as it is; True if handled."""
    m, k = x2.shape
    n = weight.shape[0]
    if not (
        0 < m <= MAX_M
        and weight.ndim == 2
        and weight.shape[1] == k
        and x2.dtype == weight.dtype == torch.bfloat16
        and out.dtype in (torch.bfloat16, torch.float32)
        and tuple(out.shape) == (m, n)
        and k % 8 == 0  # 16 B vector weight loads
        and x2.stride(-1) == 1
        and weight.stride(-1) == 1
        and out.stride(-1) == 1
        and x2.stride(0) % 8 == 0  # 16 B vector x staging
        and weight.stride(0) % 8 == 0
        and _aligned(x2)
        and _aligned(weight)
        and (bias is None or (bias.dtype == torch.bfloat16 and bias.stride(0) == 1 and bias.shape == (n,)))
    ):
        return False
    _ext.gemv_bf16(x2.data_ptr(), weight.data_ptr(), bias.data_ptr() if bias is not None else 0,
                   out.data_ptr(), m, n, k, x2.stride(0), weight.stride(0), out.stride(0),
                   int(out.dtype == torch.float32), torch.cuda.current_stream().cuda_stream, -1)
    return True


def hip_gemv(x2: torch.Tensor, weight: torch.Tensor, bias, out: torch.Tensor) -> bool:
    """out = x2 @ weight.T (+ bias) on the HIP kernels; True if handled, False = Triton."""
    if _ext is None:
        return False
    if _want_q8(weight) and _hip_q8(x2, weight, bias, out):
        return True
    return HIP_BF16 and _hip_bf16(x2, weight, bias, out)


def _hip_q8(x2: torch.Tensor, weight: torch.Tensor, bias, out: torch.Tensor) -> bool:
    """out = x2 @ weight.T (+ bias) via the int8 copy of `weight`; True if handled."""
    m, k = x2.shape
    n = weight.shape[0]
    if not (
        1 <= m <= MAX_M  # the kernel's M range: a spec-verify step (k + 1 rows) is routed whole
        and weight.ndim == 2
        and weight.shape[1] == k
        and x2.dtype == torch.bfloat16
        and out.dtype in (torch.bfloat16, torch.float32)
        and x2.stride(-1) == 1
        and k % 64 == 0  # the kernel reads whole 64 K tile groups
        and x2.stride(0) % 8 == 0  # 16 B vector x staging
        and x2.data_ptr() % 16 == 0
        and (bias is None or (bias.dtype == torch.bfloat16 and bias.stride(0) == 1))
    ):
        return False
    ent = _Q8_CACHE.get((weight.data_ptr(), k))
    if ent is not None and ent[1].shape[0] >= -(-n // Q8_TILE):
        qt, st = ent[1], ent[2]
    elif not torch.cuda.is_current_stream_capturing():
        qt, st = _q8_for(weight)
    else:
        return False
    _ext.gemv_q8(x2.data_ptr(), qt.data_ptr(), st.data_ptr(),
                 bias.data_ptr() if bias is not None else 0, out.data_ptr(),
                 m, n, k, x2.stride(0), out.stride(0), int(out.dtype == torch.float32),
                 -(-n // Q8_TILE), torch.cuda.current_stream().cuda_stream, -1)
    return True


def controller_gemms(pair: torch.Tensor, down: torch.Tensor, gain_forget: torch.Tensor):
    """The loop controllers' two batched products on gemv_bf16_batched: fp32 [S, T, 2D] =
    bf16(pair [S, T, 2D] @ down [S, r, 2D]^T) @ gain_forget [S, 2D, r]^T, or None = caller runs torch.bmm
    (knob off, no batched kernel in the build, T > 16, anything not dense bf16, r % 8)."""
    if not HIP_CTRL or _ext is None or not hasattr(_ext, "gemv_bf16_batched"):
        return None
    s, t, d2 = pair.shape
    r = down.shape[1]
    if not (
        0 < t <= MAX_M
        and r % 8 == 0
        and pair.dtype == down.dtype == gain_forget.dtype == torch.bfloat16
        and tuple(down.shape) == (s, r, d2)
        and tuple(gain_forget.shape) == (s, d2, r)
        and pair.is_contiguous()
        and down.is_contiguous()
        and gain_forget.is_contiguous()
        and d2 % 8 == 0
        and all(_aligned(a) for a in (pair, down, gain_forget))
    ):
        return None
    stream = torch.cuda.current_stream().cuda_stream
    hidden = torch.empty((s, t, r), dtype=torch.bfloat16, device=pair.device)
    out = torch.empty((s, t, d2), dtype=torch.float32, device=pair.device)
    _ext.gemv_bf16_batched(pair.data_ptr(), down.data_ptr(), hidden.data_ptr(), s, t, r, d2,
                           d2, d2, r, t * d2, r * d2, t * r, 0, stream, -1)
    _ext.gemv_bf16_batched(hidden.data_ptr(), gain_forget.data_ptr(), out.data_ptr(), s, t, d2, r,
                           r, r, d2, t * r, d2 * r, t * d2, 1, stream, -1)
    return out
