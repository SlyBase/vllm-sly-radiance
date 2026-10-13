"""4-bit KV-cache reference quantizers (KV-cache roadmap PR-1). numpy only, CPU.

Oracle for the S1 error study and the PR-3 write kernel. Formats follow OCP MX v1.0: E2M1 elements,
one E8M0 scale per block, byte 0xFF (NaN) never emitted.
"""
import numpy as np

E8M0_BIAS = 127
E8M0_MAX = 254                     # 0xFF is NaN in E8M0
E2M1_EMAX = 2                      # largest E2M1 magnitude 6 = 1.5 * 2^2


def _amax(x):
    """Block absmax over the last axis (float64)."""
    return np.max(np.abs(np.asarray(x, np.float64)), -1)


def _out(s):
    """Scalar int for one block, uint8 array for [..., block]."""
    return int(s) if np.ndim(s) == 0 else s.astype(np.uint8)


def _ocp_bytes(amax):
    # frexp: amax = m * 2^e, m in [0.5, 1) -> floor(log2 amax) = e - 1, exact (no log2 rounding)
    e = np.frexp(amax)[1].astype(np.int64) - 1 - E2M1_EMAX
    s = np.clip(e + E8M0_BIAS, 0, E8M0_MAX)
    s = np.where(amax == 0.0, 0, s)
    return np.where(np.isfinite(amax), s, E8M0_MAX)


def e8m0_ocp(x):
    """OCP shared scale byte per block (last axis): floor(log2 amax) - emax(E2M1), biased, clamped to 0..254."""
    return _out(_ocp_bytes(_amax(x)))


E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], np.float32)   # magnitude of codes 0..7; bit 3 = sign
E2M1_MID = (E2M1[:-1] + E2M1[1:]) / 2                                     # 0.25 0.75 1.25 1.75 2.5 3.5 5


class MX:
    """Packed MXFP4: payload uint8 [..., n/2] (low nibble = even index), scales uint8 [..., n/block]."""

    def __init__(self, payload, scales, block):
        self.payload, self.scales, self.block = payload, scales, block


def _e2m1_codes(v):
    """Round |v| to the nearest E2M1 magnitude, ties to the even code, saturate at 6; sign into bit 3."""
    a = np.minimum(np.abs(v), 6.0)
    # E2M1_MID[j] is the tie between codes j and j+1: left/right search differ only at an exact tie,
    # then take the even one of j, j+1
    lo = np.searchsorted(E2M1_MID, a, "left")
    hi = np.searchsorted(E2M1_MID, a, "right")
    code = np.where(lo == hi, lo, lo + (lo % 2))
    return (code | np.where(np.signbit(v) & (code > 0), 8, 0)).astype(np.uint8)


def e8m0_oas(x):
    """Overflow-Aware Scaling (arXiv 2603.08713 s4.2): OCP maps amax to [4, 8); OAS maps it to (3.5, 7],
    i.e. takes the next larger exponent when amax / 2^e_ocp > 7 (saturation error larger than rounding error)."""
    amax = _amax(x)
    s = _ocp_bytes(amax)
    with np.errstate(invalid="ignore", over="ignore"):
        up = np.isfinite(amax) & (amax > 0.0) & (amax / np.exp2(s - E8M0_BIAS) > 7.0)
    return _out(np.minimum(s + up, E8M0_MAX))


SCALE_RULES = {"ocp": e8m0_ocp, "oas": e8m0_oas}


def quant_mxfp4(x, rule="ocp", block=32):
    x = np.asarray(x, np.float32)
    if rule not in SCALE_RULES:
        raise ValueError(f"unknown kv format scale rule {rule!r}")
    g = x.reshape(*x.shape[:-1], -1, block)
    scales = np.asarray(SCALE_RULES[rule](g), np.uint8)
    inv = np.exp2(E8M0_BIAS - scales.astype(np.float64))[..., None]
    codes = _e2m1_codes(g * inv).reshape(*x.shape[:-1], -1)
    payload = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    return MX(payload, scales, block)


def dequant_mxfp4(q):
    codes = np.stack([q.payload & 0x0F, q.payload >> 4], -1).reshape(*q.payload.shape[:-1], -1)
    v = E2M1[codes & 7] * np.where(codes & 8, -1.0, 1.0).astype(np.float32)
    g = v.reshape(*v.shape[:-1], -1, q.block)
    s = np.exp2(q.scales.astype(np.float64) - E8M0_BIAS)[..., None]
    return (g * s).astype(np.float32).reshape(v.shape)


H32 = np.array([[(-1) ** bin(i & j).count("1") for j in range(32)] for i in range(32)], np.float64) / np.sqrt(32)


def hadamard32(x):
    """Normalized Sylvester Walsh-Hadamard over the last axis (length 32). Symmetric and self-inverse."""
    return (np.asarray(x, np.float64) @ H32).astype(np.float32)


def hadamard32_blocks(x):
    """blockdiag(H32) over the last axis (head_dim a multiple of 32): rotate each 32-value group."""
    x = np.asarray(x, np.float32)
    return hadamard32(x.reshape(*x.shape[:-1], -1, 32)).reshape(x.shape)


class INT4:
    """Token-wise symmetric INT4: codes int8 in -7..7 [..., n], scales fp16 [..., n/group] = amax / 7."""

    def __init__(self, codes, scales, group):
        self.codes, self.scales, self.group = codes, scales, group


def quant_int4(x, group=256):
    x = np.asarray(x, np.float32)
    g = x.reshape(*x.shape[:-1], -1, group)
    scales = (np.abs(g).max(-1) / 7.0).astype(np.float16)
    s = scales.astype(np.float32)[..., None]
    codes = np.clip(np.rint(np.divide(g, s, out=np.zeros_like(g), where=s > 0)), -7, 7).astype(np.int8)
    return INT4(codes.reshape(x.shape), scales, group)


def dequant_int4(q):
    g = q.codes.reshape(*q.codes.shape[:-1], -1, q.group).astype(np.float32)
    return (g * q.scales.astype(np.float32)[..., None]).reshape(q.codes.shape)


def _formats():
    f = {}
    for rule in ("ocp", "oas"):
        for blk, tag in ((32, ""), (16, "-16")):
            f[f"mxfp4{tag}-{rule}"] = (lambda x, r=rule, b=blk: dequant_mxfp4(quant_mxfp4(x, r, b)),
                                       lambda n, b=blk: n // 2 + n // b)
    for grp in (256, 64):
        f[f"int4-token-{grp}"] = (lambda x, g=grp: dequant_int4(quant_int4(x, g)), lambda n, g=grp: n // 2 + 2 * (n // g))
    f["fp8"] = (None, lambda n: n)
    return f


FORMATS = _formats()


def _check(fmt):
    if fmt not in FORMATS:
        raise ValueError(f"unknown kv format {fmt!r}; known: {sorted(FORMATS)}")


def bytes_per_vector(fmt, n=256):
    _check(fmt)
    return FORMATS[fmt][1](n)


def roundtrip(x, fmt, rotate=False):
    """dequant(quant(x)) per token vector (last axis). rotate=True quantizes H x and returns H dequant(...),
    i.e. the value the attention sees once Q (scores) or the output (P V) is rotated back (H = H^T = H^-1)."""
    _check(fmt)
    if FORMATS[fmt][0] is None:
        raise ValueError(f"kv format {fmt!r} has no numpy round trip here")
    x = np.asarray(x, np.float32)
    if not rotate:
        return FORMATS[fmt][0](x)
    return hadamard32_blocks(FORMATS[fmt][0](hadamard32_blocks(x)))


def attention_error(q, k, v, kq, vq):
    """||O_q - O_ref|| / ||O_ref||, O = softmax(Q K^T / sqrt(d)) V in float64, one head, no mask."""
    def attn(kk, vv):
        s = np.asarray(q, np.float64) @ np.asarray(kk, np.float64).T / np.sqrt(q.shape[-1])
        p = np.exp(s - s.max(-1, keepdims=True))
        return (p / p.sum(-1, keepdims=True)) @ np.asarray(vv, np.float64)

    ref = attn(k, v)
    return float(np.linalg.norm(attn(kq, vq) - ref) / np.linalg.norm(ref))
