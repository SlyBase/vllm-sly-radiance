"""CPU tests for the opt-in MXFP4 dense quantizer + routing (radiance_inkling2_mxfp4dense).

No GPU: the quantizer round-trips against an independent E2M1/E8M0 dequant (the same grid the
radiance_mxfp4_fp8 kernel reads), the routing gates are exercised, and the W4A16 fake-quant path
runs on plain torch. The W4A8 kernel itself is verified on the R9700 against _exact_ref.
"""
import importlib
import os
import sys

import pytest
import torch

# Import the module under test fresh per test so env knobs (ENABLED/MODE) take effect.
MODNAME = "radiance_inkling2_mxfp4dense"


def load(**env):
    for k in ("RADIANCE_INKLING2_MXFP4_DENSE", "RADIANCE_INKLING2_MXFP4_DENSE_MODE",
              "RADIANCE_INKLING2_MXFP4_DENSE_CLIP"):
        os.environ.pop(k, None)
    os.environ.update(env)
    sys.modules.pop(MODNAME, None)
    return importlib.import_module(MODNAME)


# Independent ground-truth dequant: packed [N,K/2] (low nibble = even k, bit3 = sign) +
# e8m0 [K/32,N] -> fp32 [N,K]. Written from the format spec, NOT from the module under test.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def ref_dequant(packed, scale_kn):
    n, kh = packed.shape
    k = kh * 2
    c = torch.stack([packed & 0x0F, (packed >> 4) & 0x0F], -1).reshape(n, k).long()
    mag = torch.tensor(E2M1)[c & 7]
    val = torch.where(c >= 8, -mag, mag)
    sc = torch.exp2(scale_kn.float() - 127.0).T.repeat_interleave(32, dim=1)  # [N,K]
    return val * sc


def test_dequant_matches_independent_reference():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16)
    packed, scale, wref = m.quantize_mxfp4(w)
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == (64, 128)
    assert scale.dtype == torch.uint8 and tuple(scale.shape) == (256 // 32, 64)
    assert tuple(wref.shape) == (64,)
    got = m.dequant_mxfp4(packed, scale).float()
    exp = ref_dequant(packed, scale)
    assert torch.equal(got, exp), "module dequant must equal the format-spec dequant bit-exact"


def test_roundtrip_error_in_mxfp4_range():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    torch.manual_seed(1)
    w = torch.randn(128, 512, dtype=torch.bfloat16)
    packed, scale, _ = m.quantize_mxfp4(w)
    dq = m.dequant_mxfp4(packed, scale).float()
    rel = (dq - w.float()).norm() / w.float().norm()
    assert 0.05 < rel < 0.25, f"MXFP4 relRMS out of expected range: {rel}"


def test_low_nibble_is_even_k():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    # A single group whose first two weights quantise to distinct known codes.
    w = torch.zeros(1, 32, dtype=torch.bfloat16)
    w[0, 0] = 1.0   # even k -> low nibble
    w[0, 1] = -2.0  # odd k  -> high nibble
    packed, scale, _ = m.quantize_mxfp4(w)
    b = int(packed[0, 0])
    lo, hi = b & 0x0F, (b >> 4) & 0x0F
    # scale for this block: amax=2 -> e=floor(log2 2)-2=-1 -> s=0.5; 1/0.5=2 -> idx4; -2/0.5=-4 -> idx6 sign
    assert lo == 4, f"even-k code {lo} != 4 (value 2.0)"
    assert hi == (6 | 8), f"odd-k code {hi} != 14 (value -4.0)"


def test_e8m0_scale_rule():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    w = torch.zeros(1, 32, dtype=torch.bfloat16)
    w[0, 0] = 5.0  # amax 5 -> floor(log2 5)=2 -> e=0 -> e8m0=127
    _, scale, wref = m.quantize_mxfp4(w)
    assert int(scale[0, 0]) == 127
    assert int(wref[0]) == 127


def test_zero_weight_is_finite():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    w = torch.zeros(8, 64, dtype=torch.bfloat16)
    packed, scale, wref = m.quantize_mxfp4(w)
    dq = m.dequant_mxfp4(packed, scale)
    assert torch.isfinite(dq).all()
    assert (dq == 0).all()
    assert (packed == 0).all()


def test_mse_clip_not_worse_than_rtn():
    m_rtn = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_CLIP="rtn")
    m_mse = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_CLIP="mse")
    torch.manual_seed(2)
    w = torch.randn(64, 256, dtype=torch.bfloat16)
    p1, s1, _ = m_rtn.quantize_mxfp4(w)
    p2, s2, _ = m_mse.quantize_mxfp4(w)
    e1 = (m_rtn.dequant_mxfp4(p1, s1).float() - w.float()).square().sum()
    e2 = (m_mse.dequant_mxfp4(p2, s2).float() - w.float()).square().sum()
    assert e2 <= e1 * 1.001, f"mse {e2} should not be worse than rtn {e1}"


def test_routing_disabled_returns_none():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="0")
    x = torch.randn(2, 256, dtype=torch.bfloat16)
    w = torch.randn(512, 256, dtype=torch.bfloat16)
    assert m.mxfp4_dense_linear(x, w) is None


def test_routing_gates():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    w = torch.randn(512, 256, dtype=torch.bfloat16)
    # non-bf16 weight
    assert m.mxfp4_dense_linear(torch.randn(2, 256), w.float()) is None
    # K not a multiple of 64
    assert m.mxfp4_dense_linear(torch.randn(2, 200, dtype=torch.bfloat16),
                                torch.randn(512, 200, dtype=torch.bfloat16)) is None
    # tiny / router-class N excluded
    assert m.mxfp4_dense_linear(torch.randn(2, 256, dtype=torch.bfloat16),
                                torch.randn(128, 256, dtype=torch.bfloat16)) is None


def test_w4a16_fakequant_matches_quantized_matmul():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MODE="w4a16")
    torch.manual_seed(3)
    x = torch.randn(3, 256, dtype=torch.bfloat16)
    w = torch.randn(512, 256, dtype=torch.bfloat16)
    y = m.mxfp4_dense_linear(x, w)
    assert y is not None
    packed, scale, _ = m.quantize_mxfp4(w)
    dq = m.dequant_mxfp4(packed, scale).to(torch.bfloat16)
    exp = torch.nn.functional.linear(x, dq)
    assert torch.allclose(y.float(), exp.float(), atol=1e-2, rtol=1e-2)


def test_w4a8_dispatch_uses_op_and_bias():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MODE="w4a8")
    calls = []

    def fake_op(x2, packed, scale, wref):
        calls.append((tuple(x2.shape), tuple(packed.shape), tuple(scale.shape)))
        # reference: dequant matmul in fp32
        dq = ref_dequant(packed, scale)
        return (x2.float() @ dq.T).to(torch.bfloat16)

    m._linear_op = fake_op
    torch.manual_seed(4)
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    w = torch.randn(512, 256, dtype=torch.bfloat16)
    bias = torch.randn(512, dtype=torch.bfloat16)
    y = m.mxfp4_dense_linear(x, w, bias=bias)
    assert y is not None and tuple(y.shape) == (4, 512)
    assert len(calls) == 1
    packed, scale, _ = m.quantize_mxfp4(w)
    exp = (x.float() @ ref_dequant(packed, scale).T + bias.float()).to(torch.bfloat16)
    assert torch.allclose(y.float(), exp.float(), atol=1e-2, rtol=1e-2)
    # second call reuses the cache (no re-quant): op called again, quant once
    y2 = m.mxfp4_dense_linear(x, w, bias=bias)
    assert len(calls) == 2
    assert torch.equal(y, y2)


def test_head_routes_and_casts_dtype():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MODE="w4a16")
    torch.manual_seed(5)
    x = torch.randn(2, 256, dtype=torch.bfloat16)
    w = torch.randn(2048, 256, dtype=torch.bfloat16)  # large-N head class
    y = m.mxfp4_dense_head(w, x, torch.float32)
    assert y is not None and y.dtype == torch.float32 and tuple(y.shape) == (2, 2048)