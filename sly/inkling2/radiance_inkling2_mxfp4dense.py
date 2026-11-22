"""Opt-in MXFP4 (4-bit) quantization of Inkling-2-Mini's DENSE bf16 projections + lm_head.

Default stays lossless bf16. RADIANCE_INKLING2_MXFP4_DENSE=1 quantizes the dense weights to MXFP4
(group-32 E2M1 codes + E8M0 scales, RTN or a small MSE exponent search) at LOAD time (lazily, on the
first eager call) and routes the matmul through the EXISTING radiance MXFP4 W4A8 fp8-WMMA kernel
(radiance_mxfp4_fp8.so, the same one prod Qwen runs). The bf16 original stays alive: the DSpark
drafter reads lm_head.weight in bf16 (fold + its own head linear), and any weight first seen under
CUDA-graph capture falls back to bf16. This is a quality trade-off like Qwen's own 4-bit, NOT lossless.

Why: Inkling keeps its dense projections + lm_head in bf16 (only the MoE experts are 4-bit), so it
streams ~4x the dense weight bytes of the fully-4-bit prod Qwen. MXFP4 dense cuts those bytes ~3.8x.

Modes (RADIANCE_INKLING2_MXFP4_DENSE_MODE):
  w4a8  (default) fp8 activations through the W4A8 kernel -- the speed point.
  w4a16           dequantize the MXFP4 codes back to bf16 and run the bf16 matmul -- isolates the
                  weight-quantization quality cost (no fp8 activation error, no speed gain).

The MoE router (mlp.gate, N=128) and other tiny projections (N < MIN_N) stay bf16: quantizing the
router would move expert routing. The loop message controllers run as a batched torch.bmm over the
4 streams (different weights per stream); they are ~4% of the dense bytes and stay bf16 for now
(ponytail: a grouped MXFP4 launch for 4% of bytes is not worth a new kernel path).

The quantizer matches the compressed-tensors mxfp4-pack-quantized grid the kernel reads: packed
[N, K/2] uint8 (low nibble = even k, bit3 = sign), weight_scale [K/32, N] uint8 E8M0 (2^(e-127)),
per-row reference exponent wref [N] = max block exponent (the folded kernel's epilogue scale).
"""
import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_INKLING2_MXFP4_DENSE", "0") == "1"
MODE = os.environ.get("RADIANCE_INKLING2_MXFP4_DENSE_MODE", "w4a8")
CLIP = os.environ.get("RADIANCE_INKLING2_MXFP4_DENSE_CLIP", "rtn")
MIN_N = int(os.environ.get("RADIANCE_INKLING2_MXFP4_DENSE_MIN_N", "256"))
DECODE_MAX_M = int(os.environ.get("RADIANCE_MXFP4_DECODE_MAX_M", "0"))
CHUNK_ROWS = 8192  # bounds the fp32 transient (lm_head 201024 x 2304 fp32 = 1.85 GB at once)

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# RTN nearest-magnitude boundaries: midpoint of consecutive E2M1 magnitudes. searchsorted(left)
# ties to the lower magnitude, matching argmin-first nearest.
_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

# (data_ptr, K, N) -> entry. The weight ref keeps the address from being reused by another tensor.
_CACHE: dict = {}
_linear_op = None  # resolved lazily; overridable in CPU tests
_ext = None
_scratch_ready = False
_SCRATCH_KEEP = []  # holds the decode scratch tensors alive for the process


def _resolve_op():
    """Import radiance_mxfp4 (registers torch.ops.radiance.mxfp4_linear + loads the .so)."""
    global _linear_op, _ext
    if _linear_op is not None:
        return _linear_op
    try:
        import radiance_mxfp4  # noqa: F401  (registers the custom op, imports the .so)
        import radiance_mxfp4_fp8 as ext
        _ext = ext
        _linear_op = torch.ops.radiance.mxfp4_linear
    except Exception as e:  # noqa: BLE001  # .so missing: stay on bf16
        sys.stderr.write(f"[radiance.inkling2.mxfp4dense] W4A8 op unavailable, bf16 fallback: {e!r}\n")
        _linear_op = False
    return _linear_op or None


def _setup_scratch(device):
    """Allocate the decode kernel's split-K scratch once, outside capture (same sizing as
    radiance_mxfp4.py). No-op when the decode band is dark (DECODE_MAX_M=0 -> folded GEMM)."""
    global _scratch_ready
    if _scratch_ready or DECODE_MAX_M <= 0 or _ext is None:
        return
    if torch.cuda.is_current_stream_capturing():
        return
    partial = torch.empty(4 * max(64, DECODE_MAX_M) * 36864, dtype=torch.float32, device=device)
    cnt = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=device)
    _ext.set_decode_scratch(partial.data_ptr(), partial.numel() * 4, cnt.data_ptr())
    _scratch_ready = True
    # keep alive for the process
    _SCRATCH_KEEP.extend((partial, cnt))
    sys.stderr.write(f"[radiance.inkling2.mxfp4dense] decode scratch {partial.numel() * 4 >> 20} MiB "
                     f"(M<={DECODE_MAX_M})\n")


def _e8m0(amax):
    """OCP shared exponent per 32-block: e = floor(log2 amax) - 2 (amax>0), else -127.
    Returns biased uint8 e8m0 = e + 127 and the fp32 scale 2^e."""
    e = torch.where(amax > 0, torch.floor(torch.log2(amax.clamp(min=1e-30))) - 2,
                    torch.full_like(amax, -127.0))
    return e


def _codes(x, s):
    """RTN magnitude index + sign for |x|/s on the E2M1 grid. x [.., 32] abs, s [.., 1]."""
    v = x / s
    idx = torch.searchsorted(_BOUNDS.to(v.device), v.clamp(min=0.0), right=False).clamp(max=7)
    return idx


def quantize_mxfp4(w: torch.Tensor, clip: str | None = None):
    """bf16/fp32 [N, K] -> (packed [N, K/2] u8, scale [K/32, N] u8 E8M0, wref [N] u8).

    K % 32 == 0. RTN on the OCP E8M0 grid; clip='mse' searches e in {e-1, e, e+1} per block for the
    least squared error (the E8M0 analog of the lmhead_int4 clip-ratio search; the scale stays a
    power of two so the kernel reads it unchanged).
    """
    clip = CLIP if clip is None else clip
    n, k = w.shape
    assert k % 32 == 0, f"K={k} not divisible by 32"
    ng = k // 32
    packed = torch.empty((n, k // 2), dtype=torch.uint8, device=w.device)
    e8m0 = torch.empty((n, ng), dtype=torch.uint8, device=w.device)
    bounds = _BOUNDS.to(w.device)
    for a in range(0, n, CHUNK_ROWS):
        b = min(a + CHUNK_ROWS, n)
        wb = w[a:b].float().reshape(-1, ng, 32)
        x, neg = wb.abs(), wb < 0
        amax = x.amax(-1)
        e = _e8m0(amax)
        if clip == "mse":
            best_err = None
            best_e = e
            best_idx = None
            for de in (-1, 0, 1):
                ec = (e + de).clamp(-127, 128)
                s = torch.exp2(ec)[..., None]
                idx = torch.searchsorted(bounds, (x / s).clamp(min=0.0), right=False).clamp(max=7)
                dq = torch.tensor(_E2M1, device=w.device)[idx] * s
                err = (dq - x).square().sum(-1)
                if best_err is None:
                    best_err, best_e, best_idx = err, ec, idx
                else:
                    take = err < best_err
                    best_err = torch.where(take, err, best_err)
                    best_e = torch.where(take, ec, best_e)
                    best_idx = torch.where(take[..., None], idx, best_idx)
            e, idx = best_e, best_idx
        else:
            s = torch.exp2(e)[..., None]
            idx = torch.searchsorted(bounds, (x / s).clamp(min=0.0), right=False).clamp(max=7)
        code = (idx.to(torch.uint8) | (neg.to(torch.uint8) << 3)).reshape(b - a, k)
        packed[a:b] = code[:, 0::2] | (code[:, 1::2] << 4)
        e8m0[a:b] = (e + 127).to(torch.uint8)
    scale = e8m0.T.contiguous()          # [K/32, N]
    wref = e8m0.max(dim=1).values.contiguous()  # [N]
    return packed, scale, wref


def dequant_mxfp4(packed: torch.Tensor, scale_kn: torch.Tensor) -> torch.Tensor:
    """packed [N, K/2] + scale [K/32, N] -> bf16 [N, K]. The format-spec dequant (kernel grid)."""
    n, kh = packed.shape
    k = kh * 2
    c = torch.stack([packed & 0x0F, (packed >> 4) & 0x0F], -1).reshape(n, k).long()
    table = torch.tensor(_E2M1, device=packed.device)
    mag = table[c & 7]
    val = torch.where(c >= 8, -mag, mag)
    sc = torch.exp2(scale_kn.float() - 127.0).T.repeat_interleave(32, dim=1)
    return (val * sc).to(torch.bfloat16)


def _entry(weight: torch.Tensor):
    key = (weight.data_ptr(), weight.shape[1], weight.shape[0])
    ent = _CACHE.get(key)
    if ent is not None:
        return ent
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return None  # first seen under capture: fall back to bf16
    packed, scale, wref = quantize_mxfp4(weight)
    ent = {"w": weight, "packed": packed, "scale": scale, "wref": wref, "dq": None}
    _CACHE[key] = ent
    _setup_scratch(weight.device)
    return ent


def _finish(y, x, n, out, bias):
    if bias is not None:
        y = y + bias
    y = y.reshape(x.shape[:-1] + (n,))
    if out is not None:
        out.copy_(y)
        return out
    return y


def mxfp4_dense_linear(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None,
                       bias: torch.Tensor | None = None):
    """x @ weight.T via the MXFP4 copy of `weight`; None = not handled (caller runs bf16)."""
    if not ENABLED or weight.dtype != torch.bfloat16 or weight.ndim != 2:
        return None
    n, k = weight.shape
    if n < MIN_N or k % 64 or x.dtype != torch.bfloat16 or x.shape[-1] != k:
        return None
    x2 = x.reshape(-1, k)
    if x2.shape[0] == 0:
        return None
    ent = _entry(weight)
    if ent is None:
        return None
    if MODE == "w4a16":
        if ent["dq"] is None:
            ent["dq"] = dequant_mxfp4(ent["packed"], ent["scale"])
        y = torch.nn.functional.linear(x2, ent["dq"])
        return _finish(y, x, n, out, bias)
    op = _resolve_op()
    if op is None:
        return None
    y = op(x2, ent["packed"], ent["scale"], ent["wref"])
    return _finish(y, x, n, out, bias)


def mxfp4_dense_head(weight: torch.Tensor, hidden_states: torch.Tensor, head_dtype):
    """The lm_head/unembed matmul through the MXFP4 copy, cast to head_dtype; None = bf16 fallback."""
    y = mxfp4_dense_linear(hidden_states, weight)
    if y is None:
        return None
    return y.to(head_dtype)