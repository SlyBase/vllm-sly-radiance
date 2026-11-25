"""Coda experts MXFP8 -> NVFP4 at load (RADIANCE_INKLING2_CODA_FP4): CPU-only tests.

Layout truth = what the NVFP4 expert kernels read (radiance_inkling2_moe.py): packed uint8 [E, N, K/2]
(even column in the low nibble, sign = bit 3), e4m3 block scales [E, N, K/16], and the per-expert outer
scale amax / denominator (denominator 2688 = 448 * 6 in the checkpoint's own NVFP4 layers).
Run: pytest sly/inkling2/tests/test_coda_fp4.py   (no vLLM tree needed)
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import radiance_inkling2_coda_fp4 as cf  # noqa: E402  (RED until the module exists)

ENV = "RADIANCE_INKLING2_CODA_FP4"
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
DEN = 2688.0


# ------------------------------------------------------------------ helpers (independent of the module)

def dequant_nvfp4(packed, scale, amax):
    """[E,N,K/2] u8 + [E,N,K/16] e4m3 + [E] amax -> [E,N,K] fp32 (the kernel's contract)."""
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    codes = torch.stack([lo, hi], -1).reshape(*packed.shape[:-1], -1)
    vals = E2M1[codes & 7] * torch.where(codes >> 3 == 1, -1.0, 1.0)
    return vals * scale.float().repeat_interleave(16, -1) * (amax / DEN)[:, None, None]


def quant_mxfp8(w):
    """fp32 [E,N,K] -> (e4m3 [E,N,K], E8M0 u8 [E,N,K/32]); block max lands near the top of e4m3."""
    e, n, k = w.shape
    blk = w.reshape(e, n, k // 32, 32)
    amax = blk.abs().amax(-1).clamp_min(1e-30)
    exp = torch.floor(torch.log2(amax)) - 8          # 448 ~ 2^8.8
    scale = torch.exp2(exp)
    q = (blk / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape(e, n, k), (exp + 127).to(torch.uint8)


def dequant_mxfp8(q, s):
    return q.float() * torch.exp2(s.float() - 127).repeat_interleave(32, -1)


def rel_rms(a, b):
    return ((a - b).pow(2).mean() / b.pow(2).mean()).sqrt().item()


# ------------------------------------------------------------------ knob

@pytest.fixture(autouse=True)
def _fresh_counter():
    cf._PROOF_N = 0
    yield


def test_off_by_default(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert cf.enabled() is False
    assert cf.wants(20) is False


@pytest.mark.parametrize("value", ["", "0"])
def test_zero_or_empty_is_off(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    assert cf.enabled() is False and cf.wants(21) is False


def test_one_selects_every_asked_layer(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    assert cf.enabled() is True
    assert [cf.wants(i) for i in (20, 21, 22, 23)] == [True] * 4


def test_layer_list_and_ranges(monkeypatch):
    monkeypatch.setenv(ENV, "20,22-23")
    assert [cf.wants(i) for i in (19, 20, 21, 22, 23)] == [False, True, False, True, True]


@pytest.mark.parametrize("value", ["abc", "20-", "23-20", "-1", "1,2x"])
def test_garbage_fails_loud(monkeypatch, value):
    monkeypatch.setenv(ENV, value)           # a lossy knob must not silently turn off or on
    with pytest.raises(ValueError):
        cf.enabled()


# ------------------------------------------------------------------ converter

def test_output_layout_and_dtypes():
    w = torch.randn(3, 8, 64) * 0.02
    packed, scale, amax = cf.mxfp8_to_nvfp4(*quant_mxfp8(w))
    assert packed.shape == (3, 8, 32) and packed.dtype == torch.uint8
    assert scale.shape == (3, 8, 4) and scale.dtype == torch.float8_e4m3fn
    assert amax.shape == (3,) and amax.dtype == torch.float32


def test_round_trip_error_is_nvfp4_class_on_gaussian():
    torch.manual_seed(0)
    q, s = quant_mxfp8(torch.randn(4, 64, 256) * 0.02)
    ref = dequant_mxfp8(q, s)
    err = rel_rms(dequant_nvfp4(*cf.mxfp8_to_nvfp4(q, s)), ref)
    assert 0.05 < err < 0.12, err              # NVFP4 RTN on gaussian blocks of 16 sits near 0.09


def test_round_trip_error_bounded_on_heavy_tails():
    torch.manual_seed(1)
    w = torch.distributions.StudentT(3.0).sample((2, 64, 256)) * 0.02
    q, s = quant_mxfp8(w)
    ref = dequant_mxfp8(q, s)
    assert rel_rms(dequant_nvfp4(*cf.mxfp8_to_nvfp4(q, s)), ref) < 0.16


def test_no_systematic_bias():
    torch.manual_seed(2)
    q, s = quant_mxfp8(torch.randn(2, 64, 512) * 0.02)
    ref = dequant_mxfp8(q, s)
    out = dequant_nvfp4(*cf.mxfp8_to_nvfp4(q, s))
    assert abs((out - ref).mean().item()) < 0.01 * ref.abs().mean().item()


def test_amax_is_the_per_expert_max_and_top_scale_is_448():
    torch.manual_seed(3)
    q, s = quant_mxfp8(torch.randn(3, 16, 128) * torch.tensor([0.01, 0.02, 0.05])[:, None, None])
    ref = dequant_mxfp8(q, s)
    _, scale, amax = cf.mxfp8_to_nvfp4(q, s)
    assert torch.allclose(amax, ref.abs().amax((1, 2)))
    assert scale.float().amax((1, 2)).tolist() == [448.0] * 3


def test_fp32_core_is_exact_on_nvfp4_grid():
    torch.manual_seed(4)
    e, n, k = 2, 16, 128
    codes = torch.randint(0, 16, (e, n, k), dtype=torch.uint8)
    codes[codes == 8] = 0                       # -0 has no encoder output
    codes.view(e, n, k // 16, 16)[..., 0] = 7
    sc = torch.randint(8, 440, (e, n, k // 16)).float()
    sc[:, 0, 0] = 448.0
    scale = sc.to(torch.float8_e4m3fn)
    amax = torch.tensor([0.3, 0.07])
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    w = dequant_nvfp4(packed, scale, amax)
    p2, s2, a2 = cf.fp32_to_nvfp4(w)
    assert torch.equal(p2, packed) and torch.equal(s2.float(), scale.float())
    assert torch.allclose(a2, amax, rtol=1e-6)


def test_zero_blocks_and_zero_expert_do_not_make_nan():
    w = torch.randn(2, 8, 64) * 0.02
    w[0] = 0.0                                  # a dead expert
    w[1, :, :16] = 0.0                          # dead blocks inside a live one
    packed, scale, amax = cf.mxfp8_to_nvfp4(*quant_mxfp8(w))
    out = dequant_nvfp4(packed, scale, amax)
    assert torch.isfinite(out).all() and out[0].abs().max() == 0 and out[1, :, :16].abs().max() == 0
    assert amax[0] == 0


def test_chunking_does_not_change_the_result():
    torch.manual_seed(5)
    q, s = quant_mxfp8(torch.randn(5, 32, 128) * 0.02)
    a = cf.mxfp8_to_nvfp4(q, s, chunk=1)
    b = cf.mxfp8_to_nvfp4(q, s, chunk=64)
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_sign_and_nibble_order():
    w = torch.zeros(1, 1, 32)
    w[0, 0, 0], w[0, 0, 1], w[0, 0, 2] = 6.0, -3.0, 0.5   # even column low nibble, sign bit 3
    packed, scale, amax = cf.mxfp8_to_nvfp4(*quant_mxfp8(w))
    assert packed[0, 0, 0].item() == (0x7 | (0xD << 4))     # +6 -> 0b0111, -3 -> 0b1101
    assert packed[0, 0, 1].item() & 0xF == 0x1             # +0.5


# ------------------------------------------------------------------ module surgery

class _Experts(torch.nn.Module):
    """Duck-typed Inkling2Experts as built for the MXFP8 format."""
    def __init__(self, e=2, inter=32, hidden=64):
        super().__init__()
        self.nvfp4 = False
        for name, (n, k) in zip(("w13", "w2"), ((2 * inter, hidden), (hidden, inter))):
            q, s = quant_mxfp8(torch.randn(e, n, k) * 0.02)
            setattr(self, f"{name}_weight", torch.nn.Parameter(q, requires_grad=False))
            setattr(self, f"{name}_weight_scale", torch.nn.Parameter(s, requires_grad=False))


def test_requant_experts_flips_format_in_place():
    m = _Experts()
    before = {n: dequant_mxfp8(getattr(m, f"{n}_weight").data, getattr(m, f"{n}_weight_scale").data) for n in ("w13", "w2")}
    cf.requant_experts(m)
    assert m.nvfp4 is True
    for n in ("w13", "w2"):
        w, sc = getattr(m, f"{n}_weight"), getattr(m, f"{n}_weight_scale")
        amax, den = getattr(m, f"{n}_weight_amax"), getattr(m, f"{n}_weight_outer_scale_denominator")
        assert w.dtype == torch.uint8 and sc.dtype == torch.float8_e4m3fn
        assert amax.dtype == torch.float32 and amax.shape == (2,)
        assert den.shape == (1,) and den.item() == DEN
        assert rel_rms(dequant_nvfp4(w.data, sc.data, amax.data), before[n]) < 0.12


def test_requant_experts_twice_is_a_noop():
    m = _Experts()
    cf.requant_experts(m)
    snap = m.w13_weight.data.clone()
    cf.requant_experts(m)
    assert torch.equal(m.w13_weight.data, snap)


def test_requant_prints_one_counted_proof_line_per_layer(capsys):
    cf.requant_experts(_Experts())
    cf.requant_experts(_Experts())
    lines = [l for l in capsys.readouterr().out.splitlines() if "RADIANCE_CODA_FP4_PROOF" in l]
    assert len(lines) == 2 and "n=1 " in lines[0] and "n=2 " in lines[1] and "mxfp8->nvfp4" in lines[0]
