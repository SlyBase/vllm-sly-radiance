"""KV4H: the 4-bit KV cache layout the HIP iu4-WMMA kernel reads, in plain torch (the oracle).

Same bytes per token as KV4 (272 per head of 256: 136 K, 136 V; head 128: 68 + 68, vLLM's own
int4 cell size for the DFlash drafter), same rotation (radiance_kv4_format's Walsh-Hadamard with
the fixed sign vector), but laid out for the matrix instructions instead of for Triton's tile
loads, per (block, kv head) region of block_size x 2 x TPT bytes (head 256 shown):

  K region (block_size * 136 bytes), per group of 16 tokens 2176 bytes:
      [ 2 dim-groups x 16 tokens: e4m3 scale ]             32 B   byte g*16 + t
      [ spare ]                                            96 B
      [ 16 tokens x 128 B nibbles ]                      2048 B   byte p of token t = (dim 2p, dim 2p+1)
  V region (block_size * 136 bytes, after the K region), per group of 16 tokens 2176 bytes:
      [ 16 tokens: e4m3 scale ]                            16 B   byte t
      [ 128 dim-pairs x 16 tokens ]                      2048 B   byte d*16 + t = (V[t][d], V[t][d+128])
      [ spare ]                                           112 B

Nibbles are two's-complement int4 (-8..7; the quantizer uses -8..7 by default, -7..7 with
RADIANCE_KV4H_QMIN=-7), so the iu4 WMMA consumes them with its signed-operand flag and no zero
point. K has one scale per GROUP dims (RADIANCE_KV4H_K_GROUP, default 256 = one per token; 128 and
64 are the other builds): the 16x16x32 WMMAs of a group chain into one accumulator, which is
scaled per token on the result (the kernel is instruction bound, so fewer scale groups are
cheaper; the sweep and the GSM8K gate picked one per token, see sly/kv4/README.md). V has one
scale per token, applied to P before the P.V product. V is stored
dim-major within a 16-token group so that the B operand of P.V (8 or 16 consecutive tokens of
one dim) is one contiguous 8- or 16-byte load; the token stays the sole owner of its bytes
(dims d and d+128 share a byte, never two tokens), so the store needs no read-modify-write.

Scales are amax / 7 rounded UP to e4m3 (radiance_kv4_format.round_up_e4m3), so nothing clips;
with RADIANCE_KV4H_SCALE_STEPS > 1 the store tries that many e4m3 scales downward from there and
keeps the one with the smallest squared error (clipping the rare extreme value), with
RADIANCE_KV4H_QMIN=-8 the codes use the 16th level too.
"""
import os

import torch

import radiance_kv4_format as F

HEAD = F.HEAD                 # 256: the target's heads (module-level constants are this layout)
GROUP = int(os.environ.get("RADIANCE_KV4H_K_GROUP", "") or 256)   # K dims per scale (default knob): one scale per token
TG = 16                       # tokens per layout group
QMAX = 7.0
FP8 = F.FP8
# Store-side knobs (the kernel reads any int4 code and any e4m3 scale, so both are free there):
# SCALE_STEPS = scale candidates per (token, group): the absmax scale (amax / 7 rounded up to e4m3,
# nothing clips) and the SCALE_STEPS - 1 e4m3 values below it; the one with the smallest squared
# error wins (ties: the larger). 1 = absmax only. QMIN = lowest code, -7 (symmetric) or -8.
SCALE_STEPS = int(os.environ.get("RADIANCE_KV4H_SCALE_STEPS", "") or 1)
QMIN = float(os.environ.get("RADIANCE_KV4H_QMIN", "") or -8)
# SCALE_CLIP: the first candidate is amax * SCALE_CLIP / 7 rounded up (1.0: nothing clips; 0.8, the
# default with QMIN -8, clips the rare extreme value of a near-Gaussian rotated row: SQNR on the
# model's K/V 17.6 -> 19.2 dB, the same as a 9-step search, at no store cost)
SCALE_CLIP = float(os.environ.get("RADIANCE_KV4H_SCALE_CLIP", "") or 0.8)


class Layout:
    """The constants of one head size. Head 256 (the target): 136 bytes per token and side;
    head 128 (the DFlash drafter): 68, which is vLLM's own int4 cell for that head size, so the
    drafter's KV pages keep lining up with the target's (8 x 136 = 4 x 272 bytes per token)."""

    def __init__(self, head: int, group: int = GROUP):
        group = min(group, head)                # a group wider than the head is one scale per token
        assert head in (128, 256) and head % group == 0 and group % 32 == 0
        self.HEAD = head
        self.GROUP = group
        self.NG = head // group                 # K scale groups (1, 2 or 4)
        self.TG = TG
        self.PACKED = head // 2                 # nibble bytes per token and side
        self.TPT = self.PACKED + (8 if head == 256 else 4)
        self.REG = TG * self.TPT                # bytes per 16-token group, one side
        self.K_SCALES = self.NG * TG            # scale bytes used per 16-token group
        self.K_AREA = TG * (self.TPT - self.PACKED)   # bytes before the K nibbles
        self.V_SCALES = TG
        self.QMAX = QMAX
        assert self.K_SCALES <= self.K_AREA and self.V_SCALES + self.PACKED * TG <= self.REG

    def k_region_bytes(self, block_size: int) -> int:
        assert block_size % TG == 0
        return block_size * self.TPT

    def k_scale_off(self, t, g):
        return (t // TG) * self.REG + g * TG + (t % TG)

    def k_nib_off(self, t, p):
        return (t // TG) * self.REG + self.K_AREA + (t % TG) * self.PACKED + p

    def v_scale_off(self, t, block_size):
        return self.k_region_bytes(block_size) + (t // TG) * self.REG + (t % TG)

    def v_nib_off(self, t, d, block_size):
        """d in 0..PACKED-1: the byte holding dims d (low nibble) and d + PACKED (high) of token t."""
        return self.k_region_bytes(block_size) + (t // TG) * self.REG + self.V_SCALES + d * TG + (t % TG)


LAYOUTS = {256: Layout(256), 128: Layout(128)}
L256 = LAYOUTS[256]


def layout(head: int, group: int | None = None) -> Layout:
    """The layout of a head size with `group` K dims per scale (None = the RADIANCE_KV4H_K_GROUP
    default, 256). The store and the kernel must agree; the kernel is built per group count."""
    if group is None:
        return LAYOUTS[head]
    key = (head, min(group, head))
    if key not in LAYOUTS:
        LAYOUTS[key] = Layout(head, group)
    return LAYOUTS[key]


NG, PACKED, TPT, REG, K_SCALES, K_AREA, V_SCALES = (L256.NG, L256.PACKED, L256.TPT, L256.REG,
                                                    L256.K_SCALES, L256.K_AREA, L256.V_SCALES)


def k_region_bytes(block_size: int) -> int:
    return L256.k_region_bytes(block_size)


def k_scale_off(t, g):
    return L256.k_scale_off(t, g)


def k_nib_off(t, p):
    return L256.k_nib_off(t, p)


def v_scale_off(t, block_size):
    return L256.v_scale_off(t, block_size)


def v_nib_off(t, d, block_size):
    return L256.v_nib_off(t, d, block_size)


# ---------------------------------------------------------------------------------------------
# quantization (any head size: the last dim decides)
# ---------------------------------------------------------------------------------------------

def e4m3_step_down(s: torch.Tensor) -> torch.Tensor:
    """The next e4m3 value below s (s an e4m3 value > 0; floor 2^-9), integer bit math like
    round_up_e4m3 so the store kernel lands on the same value."""
    b = s.to(torch.float32).view(torch.int32)
    normal = (b - (1 << 20)).view(torch.float32)
    sub = (s - 1.0 / 512.0).clamp_min(1.0 / 512.0)
    return torch.where(s > F.FP8_MIN_NORMAL, normal, sub)


def _scales(x_rot: torch.Tensor, group: int, steps: int | None = None, qmin: float | None = None,
            clip: float | None = None) -> torch.Tensor:
    steps = SCALE_STEPS if steps is None else steps
    qmin = QMIN if qmin is None else qmin
    clip = SCALE_CLIP if clip is None else clip
    head = x_rot.shape[-1]
    xg = x_rot.reshape(*x_rot.shape[:-1], head // group, group)
    s = F.round_up_e4m3(xg.abs().amax(-1) * clip / QMAX)
    best, best_err = s, None
    for i in range(steps):
        if i:
            s = e4m3_step_down(s)
        err = ((F.round_half_up(xg / s[..., None]).clamp(qmin, QMAX) * s[..., None] - xg) ** 2).sum(-1)
        if best_err is None:
            best_err = err
        else:
            better = err < best_err
            best, best_err = torch.where(better, s, best), torch.where(better, err, best_err)
    return best


def _quant(x_rot: torch.Tensor, scales: torch.Tensor, qmin: float | None = None) -> torch.Tensor:
    """-> int8 codes in [QMIN, 7], (..., head)."""
    qmin = QMIN if qmin is None else qmin
    head = x_rot.shape[-1]
    group = head // scales.shape[-1]
    v = x_rot.reshape(*x_rot.shape[:-1], head // group, group) / scales[..., None]
    return F.round_half_up(v).clamp(qmin, QMAX).reshape(*x_rot.shape[:-1], head).to(torch.int8)


def _dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    head = codes.shape[-1]
    group = head // scales.shape[-1]
    v = codes.to(torch.float32).reshape(*codes.shape[:-1], head // group, group) * scales[..., None]
    return v.reshape(*codes.shape[:-1], head)


def quantize_k(k: torch.Tensor, group: int = GROUP, steps: int | None = None, qmin: float | None = None,
               clip: float | None = None):
    """k (T, head) unrotated -> (codes int8 (T, head), scales fp32 (T, head // group))."""
    kr = F.rotate(k.to(torch.float32))
    s = _scales(kr, min(group, k.shape[-1]), steps, qmin, clip)
    return _quant(kr, s, qmin), s


def quantize_v(v: torch.Tensor, steps: int | None = None, qmin: float | None = None, clip: float | None = None):
    """v (T, head) unrotated -> (codes int8 (T, head), scales fp32 (T, 1))."""
    vr = F.rotate(v.to(torch.float32))
    s = _scales(vr, vr.shape[-1], steps, qmin, clip)
    return _quant(vr, s, qmin), s


def reconstruct_k(k: torch.Tensor, group: int = GROUP, steps: int | None = None, qmin: float | None = None,
                  clip: float | None = None) -> torch.Tensor:
    c, s = quantize_k(k, group, steps, qmin, clip)
    return F.unrotate(_dequant(c, s))


def reconstruct_v(v: torch.Tensor, steps: int | None = None, qmin: float | None = None,
                  clip: float | None = None) -> torch.Tensor:
    c, s = quantize_v(v, steps, qmin, clip)
    return F.unrotate(_dequant(c, s))


# ---------------------------------------------------------------------------------------------
# block encode / decode (one kv head)
# ---------------------------------------------------------------------------------------------

def encode_block(k: torch.Tensor, v: torch.Tensor, block_size: int, group: int = GROUP,
                 steps: int | None = None, qmin: float | None = None, clip: float | None = None) -> torch.Tensor:
    """k, v (T, head) with T <= block_size (the rest of the block stays zero) -> uint8 (block_size * 2 * TPT,)."""
    L = layout(k.shape[-1], group)
    T = k.shape[0]
    assert T <= block_size and block_size % TG == 0
    out = torch.zeros(block_size * 2 * L.TPT, dtype=torch.uint8, device=k.device)
    kc, ks = quantize_k(k, group, steps, qmin, clip)
    vc, vs = quantize_v(v, steps, qmin, clip)
    t = torch.arange(T, device=k.device)
    g = torch.arange(L.NG, device=k.device)
    out[L.k_scale_off(t[:, None], g[None, :])] = ks.to(FP8).view(torch.uint8)
    knib = ((kc[:, 0::2] & 0xF) | ((kc[:, 1::2] & 0xF) << 4)).to(torch.uint8)
    p = torch.arange(L.PACKED, device=k.device)
    out[L.k_nib_off(t[:, None], p[None, :])] = knib
    out[L.v_scale_off(t, block_size)] = vs[:, 0].to(FP8).view(torch.uint8)
    vnib = ((vc[:, :L.PACKED] & 0xF) | ((vc[:, L.PACKED:] & 0xF) << 4)).to(torch.uint8)
    out[L.v_nib_off(t[:, None], p[None, :], block_size)] = vnib
    return out


def _signed(nib: torch.Tensor) -> torch.Tensor:
    n = nib.to(torch.int16)
    return torch.where(n >= 8, n - 16, n).to(torch.int8)


def decode_block(region: torch.Tensor, block_size: int, T: int | None = None, head: int = HEAD, group: int = GROUP):
    """-> (k_rot, v_rot) float32 (T, head): the rotated values the kernel's dots see (no fp8
    rounding anywhere: int4 codes times e4m3 scales)."""
    L = layout(head, group)
    T = block_size if T is None else T
    t = torch.arange(T, device=region.device)
    g = torch.arange(L.NG, device=region.device)
    p = torch.arange(L.PACKED, device=region.device)
    ks = region[L.k_scale_off(t[:, None], g[None, :])].view(FP8).to(torch.float32)
    kb = region[L.k_nib_off(t[:, None], p[None, :])]
    kc = torch.empty(T, head, dtype=torch.int8, device=region.device)
    kc[:, 0::2] = _signed(kb & 0xF)
    kc[:, 1::2] = _signed((kb >> 4) & 0xF)
    vs = region[L.v_scale_off(t, block_size)].view(FP8).to(torch.float32)[:, None]
    vb = region[L.v_nib_off(t[:, None], p[None, :], block_size)]
    vc = torch.cat([_signed(vb & 0xF), _signed((vb >> 4) & 0xF)], -1)
    return _dequant(kc, ks), _dequant(vc, vs)
