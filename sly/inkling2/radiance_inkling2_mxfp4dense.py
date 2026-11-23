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
MODE = os.environ.get("RADIANCE_INKLING2_MXFP4_DENSE_MODE", "gemv")
CLIP = os.environ.get("RADIANCE_INKLING2_MXFP4_DENSE_CLIP", "rtn")
MIN_N = int(os.environ.get("RADIANCE_INKLING2_MXFP4_DENSE_MIN_N", "256"))
DECODE_MAX_M = int(os.environ.get("RADIANCE_MXFP4_DECODE_MAX_M", "0"))
MX_MAX_M = 16  # rows per gemv_mx4 launch (the kernel's range; a DSpark verify step is k + 1 <= 16)
HEAD_MIN_N = int(os.environ.get("RADIANCE_INKLING2_MXFP4_DENSE_HEAD_MIN_N", str(1 << 17)))  # the 201024-row unembed
CHUNK_ROWS = 8192  # bounds the fp32 transient (lm_head 201024 x 2304 fp32 = 1.85 GB at once)

_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# RTN nearest-magnitude boundaries: midpoint of consecutive E2M1 magnitudes. searchsorted(left)
# ties to the lower magnitude, matching argmin-first nearest.
_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])

# (data_ptr, K, N) of the weight (or of a row slice of it) the model passes to linear() -> entry.
_CACHE: dict = {}
_linear_op = None  # resolved lazily; overridable in CPU tests
_ext = None
_mx = None  # radiance_inkling2_mx4 (gemv mode), resolved once at load; False = not importable
_scratch_ready = False
_STATS = [0, 0]  # weights registered, bytes of their MXFP4 copies (the proof line printed at load)
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


def _resolve_mx():
    """Import the HIP W4A16 GEMV extension (radiance_inkling2_mx4.so); None if it is not there."""
    global _mx
    if _mx is None:
        try:
            import torch  # noqa: F401  # a bare HIP extension needs torch's runtime loaded first
            import radiance_inkling2_mx4 as ext
            _mx = ext
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[radiance.inkling2.mxfp4dense] radiance_inkling2_mx4 unavailable: {e!r}\n")
            _mx = False
    return _mx or None


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


MX_TILE = 16   # weight rows per kernel tile (one wmma operand)
MX_GROUP = 64  # K values per tile group = 2 scale blocks; the kernel reads whole groups
# position p of a code dword's 8 nibbles holds the wmma operand slot _SLOT_OF_POS[p] (the decode chain of
# radiance_inkling2_mx4.hip yields the slots in the order 0 2 4 6 1 3 5 7 of the nibbles)
_SLOT_OF_POS = [0, 4, 1, 5, 2, 6, 3, 7]


def pack_mx4(packed: torch.Tensor, scale_kn: torch.Tensor):
    """(packed [N, K/2] u8, scale [K/32, N] u8) -> the kernel's tiles: codes u8 [NT, K/64, 512], scales u8 [NT, K/64, 32].

    Per (tile of 16 rows, group of 64 K) the codes are ordered [half h][row c][block b][step j][4 B]: lane
    16 h + c of a wave reads its 16 B at lane * 16, one wave load = 512 contiguous bytes. Its dword 2 b + j
    holds the 8 codes of K 64 g + 32 b + 16 h + 8 j + slot, nibble _SLOT_OF_POS^-1 order. The scale bytes are
    [row c][block b]. Rows past N (last tile) are zero: code 0, scale byte 0 (= factor 0)."""
    n, kh = packed.shape
    k = kh * 2
    assert n > 0 and k % MX_GROUP == 0 and tuple(scale_kn.shape) == (k // 32, n)
    nt, kg = -(-n // MX_TILE), k // MX_GROUP
    sc = scale_kn.T
    if nt * MX_TILE != n:
        packed = torch.cat([packed, packed.new_zeros(nt * MX_TILE - n, kh)])
        sc = torch.cat([sc, sc.new_zeros(nt * MX_TILE - n, k // 32)])
    codes = torch.stack([packed & 0x0F, packed >> 4], -1).reshape(nt, MX_TILE, kg, 2, 2, 2, 8)  # [t,c,g,b,h,j,slot]
    codes = codes[..., _SLOT_OF_POS]  # nibble position p <- slot _SLOT_OF_POS[p]
    byts = codes[..., 0::2] | (codes[..., 1::2] << 4)  # [t,c,g,b,h,j,4]
    qt = byts.permute(0, 2, 4, 1, 3, 5, 6).reshape(nt, kg, 512)  # [t,g,h,c,b,j,4]
    st = sc.reshape(nt, MX_TILE, kg, 2).permute(0, 2, 1, 3).reshape(nt, kg, 32)
    return qt.contiguous(), st.contiguous()


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


def _gemv_entry(packed: torch.Tensor, scale: torch.Tensor) -> dict:
    qt, st = pack_mx4(packed, scale)
    return {"qt": qt, "st": st}


def _tiles(ent: dict, r0: int, n: int) -> dict:
    """Entry of the tile-aligned rows [r0, r0+n): a view of the tiles that hold them (the last tile may carry
    rows past n, the kernel never stores them)."""
    t0 = r0 // MX_TILE
    return {"qt": ent["qt"][t0:t0 + -(-n // MX_TILE)], "st": ent["st"][t0:t0 + -(-n // MX_TILE)]}


def _rows(ent: dict, r0: int, n: int) -> dict:
    """Entry of rows [r0, r0+n): quantization is per row, so this is bit-identical to quantizing the slice.
    The kernel reads scale [K/32, n] and wref [n] densely, hence the copies (small: K/32 + 1 bytes a row)."""
    return {"packed": ent["packed"][r0:r0 + n], "scale": ent["scale"][:, r0:r0 + n].contiguous(),
            "wref": ent["wref"][r0:r0 + n].clone(),
            "dq": None if ent["dq"] is None else ent["dq"][r0:r0 + n]}


def _entries(w: torch.Tensor, rows) -> tuple:
    """Quantize `w` -> (entry, [entry per row slice in `rows`]) in the layout MODE's kernel reads."""
    packed, scale, wref = quantize_mxfp4(w)
    if MODE == "gemv":  # tiles only: the [N, K/2] copy is not kept
        ent = _gemv_entry(packed, scale)
        return ent, [_tiles(ent, r0, n) if r0 % MX_TILE == 0 else _gemv_entry(packed[r0:r0 + n], scale[:, r0:r0 + n])
                     for r0, n in rows]
    # w4a16 (quality arm) runs the bf16 matmul on the dequantized codes: build it now, not on first call
    ent = {"packed": packed, "scale": scale, "wref": wref,
           "dq": dequant_mxfp4(packed, scale) if MODE == "w4a16" else None}
    return ent, [_rows(ent, r0, n) for r0, n in rows]


@torch.no_grad()
def register(p: torch.nn.Parameter, rows=(), free: bool = True) -> None:
    """Quantize the bf16 weight `p` now and route linear() calls on it (and on the row slices `rows`
    = [(first row, count)] the model takes of it) to the MXFP4 copy; free=True releases the bf16."""
    w = p.data
    if _key(w) in _CACHE or _is_stub(w) or not _eligible(w):  # already done (a second finalize pass)
        return
    assert all(_eligible(w[r0:r0 + n]) for r0, n in rows), "row slice not routable (N < MIN_N)"
    view = _stub(w) if free else w
    ent, sub = _entries(w, rows)
    _CACHE[_key(view)] = ent
    _STATS[0] += 1
    _STATS[1] += sum(t.numel() * t.element_size() for t in ent.values() if t is not None)
    for (r0, n), e in zip(rows, sub):
        _CACHE[_key(view[r0:r0 + n])] = e
    if free:
        p.data = view


def _ready(model) -> bool:
    """Knob on and the kernel usable (checked once, at load); allocates the decode scratch."""
    global MODE
    if not ENABLED:
        return False
    if MODE not in ("gemv", "w4a8", "w4a16"):
        raise ValueError(f"RADIANCE_INKLING2_MXFP4_DENSE_MODE must be gemv, w4a8 or w4a16, got {MODE!r}")
    if MODE == "gemv" and _resolve_mx() is None:  # no HIP GEMV in this image: the W4A8 kernel serves
        sys.stderr.write("[radiance.inkling2.mxfp4dense] MODE=gemv needs radiance_inkling2_mx4, using w4a8\n")
        MODE = "w4a8"
    if MODE == "w4a8":
        if _resolve_op() is None:
            return False
        _setup_scratch(next(model.parameters()).device)
    return True


def _report(what: str) -> None:
    sys.stderr.write(f"[radiance.inkling2.mxfp4dense] MODE={MODE}: {_STATS[0]} weights registered after {what}, "
                     f"{_STATS[1] >> 20} MiB of MXFP4\n")


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
    _report("the target")


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
    _report("the draft")


@torch.no_grad()
def release_head(p: torch.nn.Parameter) -> None:
    """Free the bf16 lm_head once the draft has folded it (no-op if it was not registered)."""
    ent = _CACHE.pop(_key(p.data), None)
    if ent is None:
        return
    p.data = _stub(p.data)
    _CACHE[_key(p.data)] = ent
    torch.cuda.empty_cache()  # attach_target runs after the loader's own empty_cache


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream().cuda_stream if t.is_cuda else 0


def _gemv(x2: torch.Tensor, ent: dict, n: int, dt: torch.dtype) -> torch.Tensor:
    """y [M, n] = x2 @ W^T, W4A16 on the HIP GEMV, 16 rows per launch (more rows only for the unembed)."""
    m, k = x2.shape
    if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:  # the kernel's 16 B vector loads
        x2 = x2.clone(memory_format=torch.contiguous_format)
    y = torch.empty((m, n), dtype=dt, device=x2.device)
    qt, st, stream = ent["qt"], ent["st"], _stream(x2)
    for r in range(0, m, MX_MAX_M):
        _mx.gemv_mx4(x2[r:].data_ptr(), qt.data_ptr(), st.data_ptr(), y[r:].data_ptr(), min(MX_MAX_M, m - r), n, k,
                     x2.stride(0), y.stride(0), int(dt == torch.float32), -(-n // MX_TILE), stream, -1)
    return y


def mxfp4_dense_linear(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None,
                       bias: torch.Tensor | None = None, dtype: torch.dtype | None = None):
    """x @ weight.T via the MXFP4 copy registered for `weight`; None = not ours (caller runs bf16).
    `dtype` (default bf16): result dtype, fp32 straight from the kernel's accumulator in gemv mode."""
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
    if "qt" in ent:  # gemv mode
        if x2.shape[0] <= MX_MAX_M or n >= HEAD_MIN_N:
            y = _gemv(x2, ent, n, torch.float32 if dtype == torch.float32 else torch.bfloat16)
        else:  # prefill: dequantize the tiles to bf16 once per call and run the bf16 GEMM
            dq = torch.empty((n, k), dtype=torch.bfloat16, device=x2.device)
            _mx.dequant_mx4(ent["qt"].data_ptr(), ent["st"].data_ptr(), dq.data_ptr(), n, k, ent["qt"].shape[0],
                            _stream(x2))
            y = torch.nn.functional.linear(x2, dq)
    elif MODE == "w4a16":
        y = torch.nn.functional.linear(x2, ent["dq"])
    else:
        y = _resolve_op()(x2, ent["packed"], ent["scale"], ent["wref"])
    return _finish(y if dtype is None else y.to(dtype), x, n, out, bias)


def mxfp4_dense_head(weight: torch.Tensor, hidden_states: torch.Tensor, head_dtype):
    """The lm_head/unembed matmul through the MXFP4 copy, cast to head_dtype; None = bf16 fallback."""
    return mxfp4_dense_linear(hidden_states, weight, dtype=head_dtype)
