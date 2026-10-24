"""KV4H_PVF8: P.V on fp8 (e4m3) WMMA with one f32 accumulator across tiles.

CPU: a numpy mirror of the kernel's P.V math (bare V nibbles as exact e4m3 bytes, P packed to e4m3,
per-tile power-of-two lift folded into the alpha rescale) against an fp64 reference. Compile gate: the
device assembly of the shipped prefill instance; skipped without hipcc.
"""
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ci"))
HIP = ROOT / "sly/kv4/radiance_kv4h.hip"


def e4m3_decode(b):
    b = np.asarray(b, np.uint8).astype(np.int64)
    s = np.where(b & 0x80, -1.0, 1.0)
    e, m = (b >> 3) & 15, b & 7
    return s * np.where(e == 0, m * 2.0**-9, (1 + m / 8) * 2.0 ** (e.astype(float) - 7))


def e4m3_round(x):
    """Round-to-nearest-even to the e4m3 grid, saturating at 448 (the value, not the byte)."""
    x = np.asarray(x, np.float64)
    a = np.minimum(np.abs(x), 448.0)
    e = np.maximum(np.floor(np.log2(np.where(a > 0, a, 1.0))), -6)   # subnormals share the 2^-6 step
    step = 2.0 ** (e - 3)
    return np.sign(x) * np.minimum(np.round(a / step) * step, 448.0)   # np.round is ties-to-even


def v_byte(nibble):
    """The kernel's V operand byte: (nibble ^ 8), i.e. the offset-binary code c + 8 in 0..15."""
    return (np.asarray(nibble) & 15) ^ 8


def test_macro_present_and_default_off():
    src = HIP.read_text()
    assert re.search(r"#ifndef KV4H_PVF8\s*\n#define KV4H_PVF8 0", src)


def test_offset_code_bytes_are_exact_e4m3():
    """Raw byte u (0..15) read as e4m3 is u * 2^-9 for every u (subnormals 0..7, e=1 normals 8..15)."""
    u = np.arange(16)
    assert np.array_equal(e4m3_decode(u), u * 2.0**-9)
    nib = np.arange(16)                              # two's complement nibble
    c = np.where(nib >= 8, nib - 16, nib)
    assert np.array_equal(e4m3_decode(v_byte(nib)) * 2.0**9 - 8, c)
    assert e4m3_decode(0x08) == 8 * 2.0**-9          # the correction operand: 8 * 2^-9 per token


def test_e4m3_round_helper_pins():
    assert e4m3_round(448.0) == 448.0 and e4m3_round(449.0) == 448.0
    assert e4m3_round(1.0625) == 1.0 and e4m3_round(1.1875) == 1.25   # ties to even
    assert e4m3_round(2.0**-10) == 0.0 and e4m3_round(2.0**-9) == 2.0**-9


def pv_mirror(P, nib, sv):
    """P [tiles][16 q][T], nib [tiles][T][D] two's complement nibbles, sv [tiles][T] e4m3 scales.
    Mirrors the kernel: lift = 2^floor(log2(448 / vmax)) per tile, P' = e4m3(P * sv * lift), one f32
    accumulator (units of lift_run) with the lift ratio folded into the alpha rescale, the offset
    correction accumulated with the same P' (B = 0x08 bytes), unlift at the end."""
    f32 = np.float32
    acc = np.zeros((P.shape[1], nib.shape[2]), f32)
    corr = np.zeros(P.shape[1], f32)
    lift_run = f32(1)
    for p, n, s in zip(P, nib, sv):
        vmax = s.max()
        lift = f32(2.0 ** np.floor(np.log2(448.0 / vmax))) if vmax > 0 else f32(1)
        ratio = lift / lift_run
        acc *= ratio
        corr *= ratio
        lift_run = lift
        pp = e4m3_round(p * s[None, :] * lift).astype(f32)
        b = (e4m3_decode(v_byte(n))).astype(f32)               # (c + 8) * 2^-9
        acc += pp @ b
        corr += pp.sum(1) * f32(8 * 2.0**-9)
    return (acc - corr[:, None]) * f32(2.0**9) / lift_run


@pytest.mark.parametrize("seed", range(4))
def test_pv_mirror_matches_fp64(seed):
    rng = np.random.default_rng(seed)
    tiles, T, D, Q = 24, 64, 32, 16
    P = np.exp2(-rng.exponential(3.0, (tiles, Q, T)))        # softmax numerators in (0, 1]
    nib = rng.integers(0, 16, (tiles, T, D))
    sv = e4m3_round(np.exp(rng.normal(-1, 1.5, (tiles, T))))  # per-token V scales, varying per tile
    sv = np.maximum(sv, 2.0**-9)
    c = np.where(nib >= 8, nib - 16, nib).astype(np.float64)
    ref = np.einsum("tqk,tkd->qd", P * sv[:, None, :], c)
    mag = np.einsum("tqk,tkd->qd", P * sv[:, None, :], np.abs(c))
    got = pv_mirror(P, nib, sv).astype(np.float64)
    err = np.abs(got - ref)
    assert (err <= mag * 2.0**-4 + 1e-6).all()               # worst case: every P' off by half an e4m3 step
    assert np.sqrt((err**2).mean()) <= 0.05 * np.sqrt((ref**2).mean() + 1e-12)   # e4m3: ~3.6 % rms per P element, random signs


def test_pv_mirror_exact_when_p_on_grid():
    """P' exactly representable -> the only error left is f32 accumulation."""
    rng = np.random.default_rng(7)
    P = e4m3_round(rng.uniform(0, 1, (3, 16, 64)))
    nib = rng.integers(0, 16, (3, 64, 16))
    sv = np.ones((3, 64))                                    # lift = 256: P' = P * 256, still on the grid
    c = np.where(nib >= 8, nib - 16, nib).astype(np.float64)
    ref = np.einsum("tqk,tkd->qd", P, c)
    assert np.allclose(pv_mirror(P, nib, sv), ref, rtol=1e-5, atol=1e-4)


# ---- compile gate ---------------------------------------------------------------------------------

def _asm(*flags):
    hipcc = shutil.which("hipcc")
    if not hipcc:
        pytest.skip("needs hipcc (serving image)")
    inc = subprocess.run([sys.executable, "-m", "pybind11", "--includes"], capture_output=True, text=True)
    if inc.returncode:
        pytest.skip("needs pybind11")
    out = Path(tempfile.mkdtemp()) / "kv4h.s"
    subprocess.run([hipcc, "-O3", "-fPIC", "-std=c++20", "--offload-arch=gfx1201", *inc.stdout.split(),
                    "-DKV4H_NRT=4", "-DKV4H_QP=1", "-DKV4H_NS=1", "-DKV4H_PREF=1", "-DKV4H_MODNAME=m",
                    *flags, "--offload-device-only", "-S", str(HIP), "-o", str(out)],
                   check=True, capture_output=True)
    return out


def _shipped(path):
    from attn_sched_lint import parse
    return _shipped_from(parse(str(path)))


def _shipped_from(kernels):
    ks = [k for n, k in kernels.items() if "kv4h_attentionILi256ELi4ELi4ELi1ELi1E" in n]
    assert len(ks) == 1
    return ks[0]


def test_pvf8_instance_uses_fp8_wmma_only():
    k = _shipped(_asm("-DKV4H_PVF8=1"))
    mn = [m for m, _ in k.ins]
    assert any("wmma_f32_16x16x16_fp8_fp8" in m for m in mn)
    assert not any("wmma_i32_16x16x16_iu8" in m for m in mn)
    assert k.vgpr <= 240 and not k.spill, (k.vgpr, k.spill)


def test_pvf8_off_is_identical_to_unset():
    """Every kernel's instruction stream, labels and register counts (the listing's cuid / path
    lines differ per compile and are not code)."""
    from attn_sched_lint import parse
    a, b = parse(str(_asm())), parse(str(_asm("-DKV4H_PVF8=0")))
    assert a.keys() == b.keys()
    for n in a:
        assert (a[n].ins, a[n].labels, a[n].vgpr, a[n].spill) == (b[n].ins, b[n].labels, b[n].vgpr, b[n].spill), n
    assert any("wmma_i32_16x16x16_iu8" in m for m, _ in _shipped_from(a).ins)
