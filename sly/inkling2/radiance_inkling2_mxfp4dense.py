"""Opt-in MXFP4 (4-bit) quantization of Inkling-2-Mini's DENSE bf16 projections + lm_head.

Default stays lossless bf16. RADIANCE_INKLING2_MXFP4_DENSE=1 quantizes the dense weights to MXFP4
(group-32 E2M1 codes + E8M0 scales, RTN or a small MSE exponent search) at LOAD time and routes the
matmul through the EXISTING radiance MXFP4 W4A8 fp8-WMMA kernel (radiance_mxfp4_fp8.so, the same one
prod Qwen runs). This is a quality trade-off like Qwen's own 4-bit, NOT lossless.

Memory: the bf16 originals are RELEASED right after quantization (the Parameter keeps its shape and
dtype over a one-value-per-row stub, see _stub), so vLLM's memory profiling sees the freed bytes before
it sizes the KV cache. Nothing is quantized lazily: every weight the model hands to linear() is
registered at load (process_target / process_draft, called from the models' process_weights_after_loading),
so the dispatch never needs a bf16 fallback and nothing is allocated under CUDA-graph capture. What
still reads bf16: only the lm_head, until the DSpark draft has folded it (Inkling2MarkovHead.fold reads
lm_head.weight row by row in fp64 at attach_target); release_head() frees it right after. Without a
draft the head stays bf16. Weight reload / RL weight transfer is not supported with the knob on.

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

# (data_ptr, K, N) of the weight (or of a row slice of it) the model passes to linear() -> entry.
_CACHE: dict = {}
_linear_op = None  # resolved lazily; overridable in CPU tests
_ext = None
_scratch_ready = False
_SCRATCH_KEEP = []  # holds the decode scratch tensors alive for the process


def _resolve_op():
    """Import radiance_mxfp4 (registers torch.ops.radiance.mxfp4_linear + loads the .so)."""
    global _linear_op, _ext
    if _linear_op is not None:
        return _linear_op or None
    try:
        import radiance_mxfp4  # registers the custom op, imports the .so
        import radiance_mxfp4_fp8 as ext
        # M <= MIN_M goes to aiter's W4A4 GEMM, which is wrong on this weight layout (decode is M <= 8 * 8)
        assert radiance_mxfp4.MIN_M <= 0, "RADIANCE_MXFP4_W4A8_MIN_M must be 0 (the op sends M <= MIN_M to aiter)"
        _ext = ext
        _linear_op = torch.ops.radiance.mxfp4_linear
    except Exception as e:  # noqa: BLE001  # .so missing: stay on bf16
        sys.stderr.write(f"[radiance.inkling2.mxfp4dense] W4A8 op unavailable, bf16 fallback: {e!r}\n")
        _linear_op = False
    return _linear_op or None


def _setup_scratch(device):
    """Allocate the decode kernel's split-K scratch once, at load (same sizing as radiance_mxfp4.py).
    No-op when the decode band is dark (DECODE_MAX_M=0 -> folded GEMM)."""
    global _scratch_ready
    if _scratch_ready or DECODE_MAX_M <= 0 or _ext is None:
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


def _finish(y, x, n, out, bias):
    if bias is not None:
        y = y + bias
    y = y.reshape(x.shape[:-1] + (n,))
    if out is not None:
        out.copy_(y)
        return out
    return y


def _eligible(w: torch.Tensor) -> bool:
    return w.dtype == torch.bfloat16 and w.ndim == 2 and w.shape[0] >= MIN_N and w.shape[1] % 64 == 0


def _key(w: torch.Tensor):
    return (w.data_ptr(), w.shape[1], w.shape[0])


def _stub(w: torch.Tensor) -> torch.Tensor:
    """Same shape/dtype/device as `w` over ONE NaN per row (strides (1, 0)): ~2 bytes per row instead of
    2 * K, a distinct address per row (so the row slices the model takes still key into _CACHE), and
    NaN logits instead of plausible numbers if anything ever reads it as a real weight."""
    return torch.full((w.shape[0],), float("nan"), dtype=w.dtype, device=w.device).as_strided(w.shape, (1, 0))


def _is_stub(w: torch.Tensor) -> bool:
    return w.ndim == 2 and w.stride(-1) == 0


def _entry(w: torch.Tensor) -> dict:
    packed, scale, wref = quantize_mxfp4(w)
    # w4a16 (quality arm) runs the bf16 matmul on the dequantized codes: build it now, not on first call
    return {"packed": packed, "scale": scale, "wref": wref,
            "dq": dequant_mxfp4(packed, scale) if MODE == "w4a16" else None}


def _rows(ent: dict, r0: int, n: int) -> dict:
    """Entry of rows [r0, r0+n): quantization is per row, so this is bit-identical to quantizing the slice.
    The kernel reads scale [K/32, n] and wref [n] densely, hence the copies (small: K/32 + 1 bytes a row)."""
    return {"packed": ent["packed"][r0:r0 + n], "scale": ent["scale"][:, r0:r0 + n].contiguous(),
            "wref": ent["wref"][r0:r0 + n].clone(),
            "dq": None if ent["dq"] is None else ent["dq"][r0:r0 + n]}


@torch.no_grad()
def register(p: torch.nn.Parameter, rows=(), free: bool = True) -> None:
    """Quantize the bf16 weight `p` now and route linear() calls on it (and on the row slices `rows`
    = [(first row, count)] the model takes of it) to the MXFP4 copy; free=True releases the bf16."""
    w = p.data
    if _key(w) in _CACHE or _is_stub(w) or not _eligible(w):  # already done (a second finalize pass)
        return
    assert all(_eligible(w[r0:r0 + n]) for r0, n in rows), "row slice not routable (N < MIN_N)"
    view = _stub(w) if free else w
    ent = _entry(w)
    _CACHE[_key(view)] = ent
    for r0, n in rows:
        _CACHE[_key(view[r0:r0 + n])] = _rows(ent, r0, n)
    if free:
        p.data = view


def _ready(model) -> bool:
    """Knob on and the kernel usable (checked once, at load); allocates the decode scratch."""
    if not ENABLED:
        return False
    if MODE != "w4a16":
        if _resolve_op() is None:
            return False
        _setup_scratch(next(model.parameters()).device)
    return True


def _qkvr_rows(attn):
    """The row slices Inkling2Attention._project() takes of qkvr.weight when q, k, v have separate inputs
    (full layers, and local loop-core layers after the first pass): q side (+ alignment pad), k, v."""
    q_side, kv = sum(attn.split_sizes[:4]), attn.split_sizes[4]
    return [(0, sum(attn.q_side_splits)), (q_side, kv), (q_side + kv, kv)]


def process_target(model) -> None:
    """Load-time hook of Inkling2ForConditionalGeneration: quantize + release every attention qkvr / wo_ud
    (the weights ops.linear.linear() gets), quantize the lm_head and KEEP it (see release_head)."""
    if not _ready(model):
        return
    for m in model.modules():
        if hasattr(m, "qkvr") and hasattr(m, "wo_ud"):
            register(m.qkvr.weight, _qkvr_rows(m))
            register(m.wo_ud.weight)
    # LoRA swaps in the stock LogitsProcessor, which reads the bf16 head itself: leave that one alone
    if "_apply_head" in vars(type(model.logits_processor)):
        register(model.lm_head.weight, free=False)


# the DSpark draft's own projections, all passed whole to linear() (its lm_head is the target's)
_DRAFT_LINEARS = (".qkv_proj.weight", ".kv_proj_ctx.weight", ".o_proj.weight", ".gate_up_proj.weight",
                  ".down_proj.weight")


def process_draft(model) -> None:
    """Load-time hook of Inkling2DSparkModel. hidden_out / rnn_proj / markov_w1 are read directly
    (fold, addmm, the markov kernel): they stay bf16 by name, not by shape."""
    if not _ready(model):
        return
    for name, p in model.named_parameters():
        if name.endswith(_DRAFT_LINEARS):
            register(p)


@torch.no_grad()
def release_head(p: torch.nn.Parameter) -> None:
    """Free the bf16 lm_head once the draft has folded it (no-op if it was not registered)."""
    ent = _CACHE.pop(_key(p.data), None)
    if ent is None:
        return
    p.data = _stub(p.data)
    _CACHE[_key(p.data)] = ent
    torch.cuda.empty_cache()  # attach_target runs after the loader's own empty_cache


def mxfp4_dense_linear(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None,
                       bias: torch.Tensor | None = None):
    """x @ weight.T via the MXFP4 copy registered for `weight`; None = not ours (caller runs bf16)."""
    if not ENABLED or weight.ndim != 2:
        return None
    n, k = weight.shape
    ent = _CACHE.get(_key(weight))
    if ent is None:
        if _is_stub(weight):  # its bf16 is gone: running bf16 on it would be garbage
            raise RuntimeError(f"[radiance.inkling2.mxfp4dense] released {n}x{k} weight has no MXFP4 entry")
        return None
    assert x.dtype == torch.bfloat16 and x.shape[-1] == k, (x.dtype, tuple(x.shape), k)
    x2 = x.reshape(-1, k)
    if not x2.shape[0]:  # the kernel has no empty launch
        return _finish(x2.new_empty((0, n)), x, n, out, bias)
    if MODE == "w4a16":
        y = torch.nn.functional.linear(x2, ent["dq"])
    else:
        y = _resolve_op()(x2, ent["packed"], ent["scale"], ent["wref"])
    return _finish(y, x, n, out, bias)


def mxfp4_dense_head(weight: torch.Tensor, hidden_states: torch.Tensor, head_dtype):
    """The lm_head/unembed matmul through the MXFP4 copy, cast to head_dtype; None = bf16 fallback."""
    y = mxfp4_dense_linear(hidden_states, weight)
    if y is None:
        return None
    return y.to(head_dtype)
