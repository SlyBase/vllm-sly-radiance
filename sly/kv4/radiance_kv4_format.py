"""KV4: the 4-bit KV cache format, in plain torch.

(Every constant here is built on the CPU explicitly: vLLM sets torch's default device to the GPU
before the backend is imported, and a CPU generator with a GPU default device is an error.)

This module is the oracle. The Triton store kernel (radiance_kv4_store.py) must produce the same
bytes as `encode_side`, and the attention kernel (radiance_kv4_kernel.py) must see the same values
as `decode_side`. Everything the kernels hard-code -- group size, codebooks, the Hadamard sign
vector, the byte layout -- is defined here once and imported by them.

Format, per token and head, K and V alike:

  256 dims -> Walsh-Hadamard rotation with a fixed +-1 sign vector (unnormalized H, H @ H = 256 I)
  -> K: 2 groups of 128 (the two halves), V: 1 group of 256 -> per group one e4m3 scale.

  side (K or V), 136 bytes:   [ NGROUP x e4m3 scale, spare up to 8 ][ 128 x packed nibbles ]
  packed byte p: dim p in the low nibble, dim p + 128 in the high nibble. So the low nibbles are
  group 0 and the high nibbles group 1: each of the attention kernel's two streams carries one
  group, and its scale multiplies that stream's score tile instead of every element.
  content per (head, slot), 272 bytes:   [ K side ][ V side ]

Why halves and not 32-dim groups: a scale per 32 dims is 0.3 bit better on Gaussian data, but
the kernel then needs either a per-element scale (a gather or an LDS broadcast per tile: 9-25x
the fp8 kernel on the R9700) or eight K=32 dots per stream (2.3x before any arithmetic). Two
groups keep the fp8 kernel's two K=128 dots and put the scale on the (BLOCK_M, TILE) score tile.
Why V has one scale per token: the V scale multiplies P before the P.V dot, and P is staged
through LDS once per distinct scaling -- two V scales meant two stagings per tile, one more
than the fp8 kernel. K pays nothing for its second scale (it lands on the score tile).

Codebook: 16 values. The kernel runs `uniform` (i - 7.5) for K and `geo` for V (see CODEBOOK_K /
CODEBOOK_V below); `lloyd` (Lloyd-Max for a Gaussian group divided by its own amax, scaled to
c_max = 8) is a reference format for the accuracy comparison in format_report. The scale is amax_group / c_max rounded UP to e4m3 so the
group maximum never clips; the dequantized value is codebook[nibble] * scale, rounded once to fp8
by the attention kernel -- the same single rounding today's fp8 KV cache has.

Rounding is floor(v + 0.5) everywhere (not torch.round's half-to-even), so the kernel can do the
same thing with one instruction and the test can demand bit equality.
"""
import torch

HEAD = 256
GROUP = 128
NGROUP = HEAD // GROUP          # K: 2, one scale per half, i.e. per packed-nibble stream
NGROUP_K = NGROUP
NGROUP_V = 1                    # V: one scale per token (see below)
PACKED = HEAD // 2              # 128 bytes of nibbles per side
SCALE_BYTES = 8                 # the scales, padded to 8 so the cell stays 16-byte aligned
SIDE = SCALE_BYTES + PACKED     # 136
CONTENT = 2 * SIDE              # 272: K side then V side
K_SCALE_OFF, K_PACK_OFF = 0, SCALE_BYTES
V_SCALE_OFF, V_PACK_OFF = SIDE, SIDE + SCALE_BYTES

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
FP8_MIN_SUBNORMAL = 2.0 ** -9
FP8_MIN_NORMAL = 2.0 ** -6

CODEBOOKS = ("uniform", "lloyd", "geo")
UNIFORM = tuple(float(i) - 7.5 for i in range(16))
# "geo": nibble n = s t2 t1 t0 -> (-1)^s * {2, 3, 4, 6, 8, 12, 16, 24}[t]. Its e4m3 bit pattern is
# 0x40 | s << 7 | t << 2, so the attention kernel turns a nibble into an fp8 operand with two
# shifts, two ands and two ors on the packed byte -- no compare, no select, no conversion. That
# is what makes it the kernel's codebook: on the R9700 a single select per tile element costs
# 60 % of the whole fp8 attention kernel, a conversion 5x (see radiance_kv4_kernel).
GEO_MAG = (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0)
GEO = tuple(GEO_MAG) + tuple(-m for m in GEO_MAG)
# What the attention kernel runs per side. K: uniform, because the score product runs on the
# int8 WMMA path (the nibble is the int8 operand as is, the -7.5 offset is a row-sum correction),
# which costs nothing per element beyond the nibble mask. V: geo, because P.V stays on fp8 (P
# needs fp8's exponent range) and geo is the codebook whose fp8 operand is a bit shuffle.
CODEBOOK_K = "uniform"
CODEBOOK_V = "geo"

# Same construction as vLLM's int4_per_token_head sign vector: a fixed seed, so every process
# (store kernel, attention kernel, offloaded bytes read back a week later) agrees on the rotation.
_SIGN_SEED = 0x9E3779B9


def _lloyd_max_groupnorm(levels: int = 16, iters: int = 60, groups: int = 1 << 18) -> tuple:
    """Lloyd-Max codebook for the statistic the kernel actually quantizes: a N(0, 1) group of
    GROUP values divided by the group's own amax, so c_max = 1 and the group maximum lands on the
    outermost level exactly. Deterministic (seeded samples, float64)."""
    g = torch.Generator(device="cpu").manual_seed(_SIGN_SEED)
    x = torch.randn(groups, GROUP, generator=g, dtype=torch.float64, device="cpu")
    x = (x / x.abs().amax(-1, keepdim=True)).reshape(-1)
    c = torch.linspace(-1.0, 1.0, levels, dtype=torch.float64, device="cpu")
    for _ in range(iters):
        thr = 0.5 * (c[1:] + c[:-1])
        idx = torch.bucketize(x, thr)
        w = torch.zeros(levels, dtype=torch.float64, device="cpu").index_add_(0, idx, torch.ones_like(x))
        m = torch.zeros(levels, dtype=torch.float64, device="cpu").index_add_(0, idx, x)
        c = m / w.clamp_min(1.0)
        c = 0.5 * (c - c.flip(0))  # enforce symmetry; kills float drift
    c[0], c[-1] = -1.0, 1.0  # the group amax is a codeword
    # scaled to c_max = 8 (the uniform codebook's range): a pure re-parametrization that keeps
    # scale = amax / 8 well inside e4m3 (max 448) for any amax up to 3584. The values are then
    # rounded to e4m3 once, because the attention kernel keeps the table in fp8.
    return tuple((8.0 * c).to(torch.float32).to(FP8).to(torch.float64).tolist())


_LLOYD: tuple | None = None


def lloyd() -> tuple:
    """The Lloyd codebook, computed on first use (~20 s of CPU: no import-time cost for the
    paths that never use it, i.e. every KV4H process)."""
    global _LLOYD
    if _LLOYD is None:
        _LLOYD = _lloyd_max_groupnorm()
    return _LLOYD


_DEV_CACHE: dict = {}


def _on(t: torch.Tensor, device) -> torch.Tensor:
    """A small constant on `device`, cached (vLLM captures CUDA graphs around these calls: a
    fresh CPU -> GPU copy inside the capture is an error)."""
    key = (id(t), str(device))
    if key not in _DEV_CACHE:
        _DEV_CACHE[key] = t.to(device)
    return _DEV_CACHE[key]


def codebook(name: str) -> torch.Tensor:
    """The 16 values indexed by nibble (uniform and lloyd ascending; geo sign-magnitude)."""
    if name in _CB:
        return _CB[name]
    if name == "uniform":
        t = torch.tensor(UNIFORM, dtype=torch.float32, device="cpu")
    elif name == "lloyd":
        t = torch.tensor(lloyd(), dtype=torch.float32, device="cpu")
    elif name == "geo":
        t = torch.tensor(GEO, dtype=torch.float32, device="cpu")
    else:
        raise ValueError(f"unknown KV4 codebook {name!r}, expected one of {CODEBOOKS}")
    _CB[name] = t
    return t


_CB: dict = {}


def c_max(name: str) -> float:
    return float(codebook(name).max())


def thresholds(name: str) -> torch.Tensor:
    """Decision thresholds (midpoints) for nearest-codeword lookup. lloyd: 15, over the sorted
    values, idx = sum(v > thr). geo: 7, over the magnitudes, t = sum(|v| > thr), idx = 8 s + t."""
    if name in _THR:
        return _THR[name]
    if name == "geo":
        m = torch.tensor(GEO_MAG, dtype=torch.float32, device="cpu")
        t = 0.5 * (m[1:] + m[:-1])
    else:
        cb = codebook(name)
        t = 0.5 * (cb[1:] + cb[:-1])
    _THR[name] = t
    return t


_THR: dict = {}


# ---------------------------------------------------------------------------------------------
# Hadamard rotation
# ---------------------------------------------------------------------------------------------

_SIGNS: dict = {}
_HMAT: dict = {}


def signs(device=None, size: int = HEAD) -> torch.Tensor:
    """The fixed +-1 vector D (float32, length `size`)."""
    key = (str(device), size)
    if key not in _SIGNS:
        g = torch.Generator(device="cpu").manual_seed(_SIGN_SEED)
        d = torch.randint(0, 2, (size,), generator=g, dtype=torch.int64, device="cpu") * 2 - 1
        _SIGNS[key] = d.to(torch.float32).to(device)
    return _SIGNS[key]


def hadamard(device=None, dtype=torch.float32, size: int = HEAD) -> torch.Tensor:
    """Unnormalized Sylvester Hadamard matrix H (size x size), H @ H = size * I."""
    key = (str(device), dtype, size)
    if key not in _HMAT:
        h = torch.ones(1, 1, dtype=torch.float64, device="cpu")
        while h.shape[0] < size:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        _HMAT[key] = h.to(dtype).to(device)
    return _HMAT[key]


def rotation_matrix(device=None, dtype=torch.float32, size: int = HEAD) -> torch.Tensor:
    """R = diag(D) @ H so that rotate(x) = x @ R. R @ R.T = size * I."""
    return signs(device, size).to(dtype)[:, None] * hadamard(device, dtype, size)


def rotate(x: torch.Tensor) -> torch.Tensor:
    """x (..., size) -> (x * D) @ H, computed in the dtype of x (size = 256 or 128)."""
    return x @ rotation_matrix(x.device, x.dtype, x.shape[-1])


def unrotate(y: torch.Tensor) -> torch.Tensor:
    """Inverse of rotate: (y @ H) * D / size."""
    return (y @ rotation_matrix(y.device, y.dtype, y.shape[-1]).T) * (1.0 / y.shape[-1])


# ---------------------------------------------------------------------------------------------
# Quantization (reference)
# ---------------------------------------------------------------------------------------------

def round_half_up(v: torch.Tensor) -> torch.Tensor:
    return torch.floor(v + 0.5)


def round_up_e4m3(v: torch.Tensor) -> torch.Tensor:
    """Smallest e4m3 value >= v (v >= 0), as float32. Integer bit math on the fp32 pattern, not a
    dtype conversion: the kernel does the same ops, so both land on the same byte without relying
    on the two conversions' rounding modes agreeing."""
    v = v.to(torch.float32).clamp_min(0.0)
    big = v.clamp(FP8_MIN_NORMAL, FP8_MAX)
    b = big.view(torch.int32)
    low = (1 << 20) - 1                     # fp32 keeps 23 mantissa bits, e4m3 the top 3
    t = b & ~low
    t = torch.where((b & low) != 0, t + (1 << 20), t)  # ceil; a carry rolls into the exponent
    normal = t.view(torch.float32).clamp_max(FP8_MAX)
    small = torch.ceil(v * 512.0).clamp_min(1.0) * (1.0 / 512.0)  # subnormal grid, 2^-9 steps
    return torch.where(v < FP8_MIN_NORMAL, small, normal)


def group_scales(x_rot: torch.Tensor, name: str, ngroup: int = NGROUP) -> torch.Tensor:
    """x_rot (..., HEAD) rotated values -> (..., ngroup) e4m3-representable float32 scales."""
    amax = x_rot.reshape(*x_rot.shape[:-1], ngroup, HEAD // ngroup).abs().amax(-1)
    return round_up_e4m3(amax / c_max(name))


def quantize_indices(x_rot: torch.Tensor, scales: torch.Tensor, name: str) -> torch.Tensor:
    """x_rot (..., HEAD), scales (..., ngroup) -> nibble indices (..., HEAD) as uint8 in [0, 15]."""
    ngroup = scales.shape[-1]
    v = x_rot.reshape(*x_rot.shape[:-1], ngroup, HEAD // ngroup) / scales[..., None]
    if name == "uniform":
        idx = round_half_up(v + 7.5).clamp(0, 15)
    elif name == "geo":
        thr = _on(thresholds(name), v.device)
        idx = (v.abs()[..., None] > thr).sum(-1) + 8 * (v < 0)
    else:
        thr = _on(thresholds(name), v.device)
        idx = (v[..., None] > thr).sum(-1)
    return idx.reshape(*x_rot.shape[:-1], HEAD).to(torch.uint8)


def dequantize(idx: torch.Tensor, scales: torch.Tensor, name: str) -> torch.Tensor:
    """Indices (..., HEAD) + scales (..., NGROUP) -> float32 values (..., HEAD), before the fp8
    rounding the attention kernel applies (use `.to(FP8).float()` on the result for that)."""
    cb = _on(codebook(name), idx.device)
    ngroup = scales.shape[-1]
    v = cb[idx.long()].reshape(*idx.shape[:-1], ngroup, HEAD // ngroup) * scales[..., None]
    return v.reshape(*idx.shape[:-1], HEAD)


def pack(idx: torch.Tensor) -> torch.Tensor:
    """(..., HEAD) nibbles -> (..., PACKED) bytes; dim p low, dim p + PACKED high."""
    lo = idx[..., :PACKED] & 0xF
    hi = idx[..., PACKED:] & 0xF
    return (lo | (hi << 4)).to(torch.uint8)


def unpack(packed: torch.Tensor) -> torch.Tensor:
    """(..., PACKED) bytes -> (..., HEAD) nibbles as uint8."""
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    return torch.cat([lo, hi], -1)


def encode_side(x: torch.Tensor, name: str, ngroup: int = NGROUP) -> torch.Tensor:
    """x (..., HEAD) UNROTATED float -> (..., SIDE) bytes: the K (ngroup 2) or V (ngroup 1) side."""
    x_rot = rotate(x.to(torch.float32))
    scales = group_scales(x_rot, name, ngroup)
    idx = quantize_indices(x_rot, scales, name)
    scale_bytes = scales.to(FP8).view(torch.uint8)
    pad = torch.zeros(*scales.shape[:-1], SCALE_BYTES - ngroup, dtype=torch.uint8, device=x.device)
    return torch.cat([scale_bytes, pad, pack(idx)], -1)


def decode_side(side: torch.Tensor, name: str, fp8_rounded: bool = True,
                ngroup: int = NGROUP) -> torch.Tensor:
    """(..., SIDE) bytes -> (..., HEAD) ROTATED float32 values (fp8_rounded=True: after one
    rounding to e4m3, i.e. what a per-element fp8 dequant would see; the kernel's dots see the
    exact values, so compare with fp8_rounded=False)."""
    scales = side[..., :ngroup].contiguous().view(FP8).to(torch.float32)
    idx = unpack(side[..., SCALE_BYTES:])
    v = dequantize(idx, scales, name)
    return v.to(FP8).to(torch.float32) if fp8_rounded else v


def encode_content(k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """k, v (..., HEAD) -> (..., CONTENT) bytes, the whole (head, slot) cell as the kernel reads it:
    K uniform with a scale per half, V geo with a scale per token."""
    return torch.cat([encode_side(k, CODEBOOK_K, NGROUP_K), encode_side(v, CODEBOOK_V, NGROUP_V)], -1)


def decode_content(content: torch.Tensor, fp8_rounded: bool = False):
    k = decode_side(content[..., :SIDE], CODEBOOK_K, fp8_rounded, NGROUP_K)
    v = decode_side(content[..., SIDE:], CODEBOOK_V, fp8_rounded, NGROUP_V)
    return k, v


def reconstruct(x: torch.Tensor, name: str, fp8_rounded: bool = True,
                ngroup: int = NGROUP) -> torch.Tensor:
    """Round trip in the ORIGINAL basis: what the model effectively attends over."""
    return unrotate(decode_side(encode_side(x, name, ngroup), name, fp8_rounded, ngroup))


def asym_reconstruct(x: torch.Tensor, group: int) -> torch.Tensor:
    """Asymmetric int4 (scale + zero point per `group` rotated dims), round trip in the original
    basis. Not a kernel format (a per-group scale inside the dot is what the Triton kernel cannot
    afford); the accuracy reference for a group-scaled HIP kernel."""
    xr = rotate(x.to(torch.float32))
    xg = xr.reshape(*xr.shape[:-1], HEAD // group, group)
    lo, hi = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
    s = ((hi - lo) / 15).clamp_min(1e-8)
    q = ((xg - lo) / s).round().clamp(0, 15)
    return unrotate((q * s + lo).reshape(xr.shape))


def fp8_static_roundtrip(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Today's path for comparison: per-tensor static scale, one e4m3 rounding."""
    return (x.to(torch.float32) / scale).to(FP8).to(torch.float32) * scale


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.to(torch.float32) - b.to(torch.float32)) ** 2).mean())


def format_report(x: torch.Tensor, fp8_scale: float = 1.0) -> dict:
    """Relative MSE (to the signal power) of every variant on x (..., HEAD)."""
    x = x.to(torch.float32)
    p = float((x * x).mean())
    out = {"fp8_static": mse(x, fp8_static_roundtrip(x, fp8_scale)) / p}
    for name in CODEBOOKS:
        out[f"kv4_{name}_k"] = mse(x, reconstruct(x, name, False, NGROUP_K)) / p
        out[f"kv4_{name}_v"] = mse(x, reconstruct(x, name, False, NGROUP_V)) / p
    return out


__all__ = [n for n in dir() if not n.startswith("_") and n not in ("math", "torch")]
