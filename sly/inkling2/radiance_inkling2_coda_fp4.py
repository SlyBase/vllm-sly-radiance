"""Coda experts MXFP8 -> NVFP4 at load (Inkling-2-Mini layers 20-23), lossy, opt-in.

The checkpoint keeps its last four MoE layers in MXFP8 (1 B/weight + E8M0 per 32): 8.21 MB per expert
visit against 4.48 MB for the NVFP4 layers 0-19.  Decode is weight-byte bound, so serving them as NVFP4
cuts ~0.44 GB per step.  Nothing new on the kernel side: after the conversion the layer IS an NVFP4 expert
layer (same params, same HIP / Triton GEMMs, NVFP4 activations), only the numbers come from a requantization.

  RADIANCE_INKLING2_CODA_FP4   unset / 0 = off (default) | 1 = every MXFP8 expert layer |
                               layer list "20,22-23" (a layer that is not MXFP8 is never touched)

Layout (radiance_inkling2_moe.py): packed uint8 [E, N, K/2] (even column low nibble, sign = bit 3), e4m3
block scales [E, N, K/16], per-expert outer scale amax / 2688 (2688 = 448 * 6, the checkpoint's own value).
Encoder: amax = max|W| of the expert, block scale = e4m3(block_max / 6 * 2688 / amax), codes = RTN to e2m1.
"""
import os

import torch

ENV = "RADIANCE_INKLING2_CODA_FP4"
DENOMINATOR = 2688.0                      # 448 (e4m3 max) * 6 (e2m1 max)
_MID = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)   # RTN boundaries between the e2m1 magnitudes
_PROOF_N = 0


def _parse():
    """None = knob off, True = all layers, frozenset = those layers.  Garbage raises (lossy knob)."""
    v = os.environ.get(ENV, "").strip()
    if v in ("", "0"):
        return None
    if v == "1":
        return True
    layers = set()
    for part in v.split(","):
        lo, _, hi = part.strip().partition("-")
        if not lo.isdigit() or (_ and not hi.isdigit()):
            raise ValueError(f"{ENV}={v!r}: want 0, 1 or a layer list like 20,22-23")
        a, b = int(lo), int(hi or lo)
        if b < a:
            raise ValueError(f"{ENV}={v!r}: empty range {part!r}")
        layers.update(range(a, b + 1))
    return frozenset(layers)


def enabled() -> bool:
    return _parse() is not None


def wants(layer_id: int) -> bool:
    """True when this (MXFP8) layer is to be served as NVFP4."""
    sel = _parse()
    return sel is True or (sel is not None and layer_id in sel)


def _convert(e, n, k, get, chunk, device):
    """Shared encoder: `get(a, b)` yields the fp32 weights of experts [a, b)."""
    assert k % 16 == 0
    packed = torch.empty((e, n, k // 2), dtype=torch.uint8, device=device)
    scale = torch.empty((e, n, k // 16), dtype=torch.float8_e4m3fn, device=device)
    amax = torch.empty((e,), dtype=torch.float32, device=device)
    mid = torch.tensor(_MID, device=device)
    for a in range(0, e, chunk):
        x = get(a, min(a + chunk, e))
        c = x.shape[0]
        am = x.abs().amax((1, 2))
        gs = (am / DENOMINATOR).clamp_min(1e-30)[:, None, None]           # outer scale
        blk = x.reshape(c, n, k // 16, 16)
        sq = (blk.abs().amax(-1) / 6.0 / gs).clamp(max=448.0).to(torch.float8_e4m3fn)
        real = sq.float() * gs                                              # block scale in weight units
        y = torch.where(real[..., None] > 0, blk / real[..., None].clamp_min(1e-30), torch.zeros_like(blk))
        mag = torch.bucketize(y.abs(), mid).to(torch.uint8)
        code = mag | (((y < 0) & (mag > 0)).to(torch.uint8) << 3)
        code = code.reshape(c, n, k)
        packed[a:a + c] = code[..., 0::2] | (code[..., 1::2] << 4)
        scale[a:a + c], amax[a:a + c] = sq, am
    return packed, scale, amax


def fp32_to_nvfp4(w, chunk=4):
    """fp32/bf16 [E, N, K] -> (packed u8 [E, N, K/2], e4m3 [E, N, K/16], amax f32 [E])."""
    return _convert(*w.shape, lambda a, b: w[a:b].float(), chunk, w.device)


def mxfp8_to_nvfp4(w, s, chunk=4):
    """MXFP8 e4m3 [E, N, K] + E8M0 u8 [E, N, K/32] -> the same triple as `fp32_to_nvfp4`."""
    def get(a, b):
        return w[a:b].float() * torch.exp2(s[a:b].float() - 127).repeat_interleave(32, -1)
    return _convert(*w.shape, get, chunk, w.device)


def requant_experts(experts):
    """Turn a loaded MXFP8 `Inkling2Experts` into an NVFP4 one (call from process_weights_after_loading)."""
    global _PROOF_N
    if experts.nvfp4:
        return
    before = after = 0
    for name in ("w13", "w2"):
        w, s = getattr(experts, f"{name}_weight"), getattr(experts, f"{name}_weight_scale")
        before += w.numel() * w.element_size() + s.numel() * s.element_size()
        packed, scale, amax = mxfp8_to_nvfp4(w.data, s.data)
        w.data, s.data = packed, scale                                       # frees the fp8 copy
        after += packed.numel() + scale.numel()
        experts.register_parameter(f"{name}_weight_amax", torch.nn.Parameter(amax, requires_grad=False))
        experts.register_parameter(
            f"{name}_weight_outer_scale_denominator",
            torch.nn.Parameter(torch.full((1,), DENOMINATOR, device=amax.device), requires_grad=False))
    experts.nvfp4 = True
    _PROOF_N += 1
    print(f"RADIANCE_CODA_FP4_PROOF n={_PROOF_N} mxfp8->nvfp4 experts={experts.w13_weight.shape[0]} "
          f"bytes {before / 1e6:.0f} MB -> {after / 1e6:.0f} MB", flush=True)
