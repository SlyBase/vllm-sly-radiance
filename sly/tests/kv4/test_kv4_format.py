#!/usr/bin/env python3
"""CPU tests for the KV4 format reference (sly/kv4/radiance_kv4_format.py).

Plain torch, no GPU, no vllm. The GPU tests (test_kv4_kernels.py) hold the Triton kernels to
this reference bit for bit.

Run:  python3 sly/tests/kv4/test_kv4_format.py     (exit 0 = pass)
pytest picks up the test_* functions unchanged.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kv4"))

import radiance_kv4_format as F  # noqa: E402

torch.manual_seed(0)


def _gauss(n=4096):
    return torch.randn(n, F.HEAD)


def _outlier(n=4096):
    """Keys the way the model makes them: a few channels 10-30x the rest, per-token magnitude."""
    x = torch.randn(n, F.HEAD)
    x[:, :6] *= 20.0
    x[:, 100] *= 12.0
    return x * torch.empty(n, 1).uniform_(0.3, 3.0)


def test_layout_constants():
    assert F.HEAD == 256 and F.GROUP == 128 and F.NGROUP_K == 2 and F.NGROUP_V == 1 and F.SCALE_BYTES == 8
    assert F.SIDE == 136 and F.CONTENT == 272
    assert F.K_PACK_OFF == 8 and F.V_SCALE_OFF == 136 and F.V_PACK_OFF == 144


def test_codebooks():
    u = F.codebook("uniform")
    assert u.tolist() == [i - 7.5 for i in range(16)]
    ll = F.codebook("lloyd")
    assert torch.all(ll[1:] > ll[:-1]) and abs(float(ll.sum())) < 1e-5
    # designed on amax-normalized groups: the group maximum is a codeword
    assert float(ll[-1]) == 8.0 and float(ll[0]) == -8.0 and 0.25 < float(ll[8]) < 1.0
    assert F.c_max("uniform") == 7.5 and F.c_max("lloyd") == 8.0 and F.c_max("geo") == 24.0
    assert len(F.thresholds("lloyd")) == 15 and len(F.thresholds("geo")) == 7
    g = F.codebook("geo")
    assert g[:8].tolist() == list(F.GEO_MAG) and torch.equal(g[8:], -g[:8])
    # the kernel's bit identity: e4m3 bits of value(n) == 0x40 | (n >> 3) << 7 | (n & 7) << 2
    bits = g.to(F.FP8).view(torch.uint8).tolist()
    assert bits == [0x40 | ((n >> 3) << 7) | ((n & 7) << 2) for n in range(16)]


def test_rotation_is_orthogonal_and_invertible():
    x = _outlier(64)
    y = F.rotate(x)
    assert torch.allclose(F.unrotate(y), x, atol=1e-3, rtol=1e-4)
    # energy preserved up to the HEAD factor
    assert torch.allclose((y * y).sum(-1), (x * x).sum(-1) * F.HEAD, rtol=1e-4)
    # the rotation spreads the outlier channels: the max/rms ratio drops by a lot
    ratio = lambda t: float((t.abs().amax(-1) / t.pow(2).mean(-1).sqrt()).mean())  # noqa: E731
    assert ratio(y) < ratio(x) / 2


def test_round_up_e4m3_never_below_and_tight():
    v = torch.rand(100000) * 400 + 2 ** -6  # normal range
    r = F.round_up_e4m3(v)
    assert torch.all(r >= v)
    assert torch.all(r.to(F.FP8).to(torch.float32) == r)  # representable
    assert torch.all(r <= v * (1 + 2 ** -3 + 1e-6))  # at most one e4m3 ulp (1/8 at a binade start)
    sub = torch.rand(1000) * 2 ** -6  # subnormals: absolute ulp 2^-9
    rs = F.round_up_e4m3(sub)
    assert torch.all(rs >= sub) and torch.all(rs - sub <= 2 ** -9 + 1e-9)
    assert float(F.round_up_e4m3(torch.tensor([0.0]))) == F.FP8_MIN_SUBNORMAL
    assert float(F.round_up_e4m3(torch.tensor([1e9]))) == F.FP8_MAX


def test_pack_unpack_roundtrip():
    idx = torch.randint(0, 16, (3, 5, F.HEAD), dtype=torch.uint8)
    p = F.pack(idx)
    assert p.shape == (3, 5, F.PACKED) and p.dtype == torch.uint8
    assert torch.equal(F.unpack(p), idx)
    # dim p low nibble, dim p + 128 high nibble
    assert int(p[0, 0, 5]) == int(idx[0, 0, 5]) | (int(idx[0, 0, 5 + F.PACKED]) << 4)


def test_quantize_never_clips_and_is_bounded():
    for name in F.CODEBOOKS:
        x = _outlier(512)
        y = F.rotate(x)
        s = F.group_scales(y, name)
        idx = F.quantize_indices(y, s, name)
        assert idx.min() >= 0 and idx.max() <= 15
        d = F.dequantize(idx, s, name)
        step = (s[..., None].expand(-1, -1, F.GROUP).reshape(-1, F.HEAD))
        if name == "uniform":
            # nearest half-integer times the scale: error <= scale / 2, never clipped
            assert torch.all((y - d).abs() <= step * 0.5 + 1e-4)
        else:
            # nearest centroid: no element lands outside [c_min, c_max] * scale, and the sign
            # is never wrong where it matters
            assert torch.all((y / step).abs() <= F.c_max(name) + 1e-6)
            assert torch.all((d * y >= 0) | (y.abs() < step))


def test_encode_decode_side_layout():
    x = _gauss(7)
    for name in F.CODEBOOKS:
        side = F.encode_side(x, name)
        assert side.shape == (7, F.SIDE) and side.dtype == torch.uint8
        y = F.rotate(x)
        s = F.group_scales(y, name)
        assert torch.equal(side[:, :F.NGROUP].contiguous().view(F.FP8).to(torch.float32), s)
        assert int(side[:, F.NGROUP:F.SCALE_BYTES].abs().sum()) == 0
        assert torch.equal(side[:, 8:], F.pack(F.quantize_indices(y, s, name)))
        d = F.decode_side(side, name, fp8_rounded=False)
        assert torch.equal(d, F.dequantize(F.quantize_indices(y, s, name), s, name))
        d8 = F.decode_side(side, name)
        assert torch.all(d8 == d8.to(F.FP8).to(torch.float32))


def test_content_is_k_then_v():
    k, v = _gauss(4), _gauss(4)
    c = F.encode_content(k, v)
    assert c.shape == (4, F.CONTENT)
    assert torch.equal(c[:, : F.SIDE], F.encode_side(k, F.CODEBOOK_K, F.NGROUP_K))
    assert torch.equal(c[:, F.SIDE:], F.encode_side(v, F.CODEBOOK_V, F.NGROUP_V))
    assert int(c[:, F.SIDE + 1:F.SIDE + 8].abs().sum()) == 0  # one V scale, the rest spare
    kk, vv = F.decode_content(c)
    assert torch.equal(kk, F.decode_side(c[:, : F.SIDE], F.CODEBOOK_K, False, F.NGROUP_K))
    assert torch.equal(vv, F.decode_side(c[:, F.SIDE:], F.CODEBOOK_V, False, F.NGROUP_V))


def test_per_token_scale_costs_little_over_per_half():
    x = _outlier(2048)
    r = F.format_report(x)
    for name in F.CODEBOOKS:
        assert r[f"kv4_{name}_k"] <= r[f"kv4_{name}_v"] < 2.0 * r[f"kv4_{name}_k"]


def test_lloyd_beats_uniform_on_gaussian_and_both_are_sane():
    x = _gauss(8192)
    r = F.format_report(x)
    assert r["kv4_lloyd_k"] < r["kv4_uniform_k"]
    # 4-bit on N(0,1) with per-half scales: relative MSE in the low percent range
    assert 0.002 < r["kv4_uniform_k"] < 0.03 and 0.002 < r["kv4_lloyd_k"] < 0.03
    assert r["fp8_static"] < r["kv4_lloyd_k"]  # fp8 stays better per element; the point is 2x bytes


def test_report_on_outlier_keys():
    x = _outlier(4096)
    r = F.format_report(x)
    print("relative MSE, outlier keys:", {k: f"{v:.2e}" for k, v in r.items()})
    assert r["kv4_uniform_k"] < 0.05 and r["kv4_uniform_v"] < 0.05
    # the kernel's codebook pays for its bit-cheap dequant with a coarser grid: under 3x uniform
    assert r["kv4_geo_k"] < 3.0 * r["kv4_uniform_k"]


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok  ", name)
            except Exception as e:  # noqa: BLE001
                fails += 1
                print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
