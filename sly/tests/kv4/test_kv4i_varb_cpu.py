"""CPU pin of Variant B of the O_KV4I route (docs/R4D-INT4-PREFILL.md): bare int4 codes are staged as
exact e4m3 bytes, the K token scale multiplies the fp32 score, the V token scale multiplies P before P is
packed to e4m3 (per-row power-of-two lift into [224,448]), the denominator sums the packed P.

Pinned: (1) the staged byte of a code is exactly torch's e4m3 of the code, all 16 codes (C++ header, host
compiler); (2) the offline model kv4i_offline.attend_b equals a from-the-formulas reference; (3) its
kernel-faithful lazy-reference arm stays close to the per-tile-lift arm; (4) the HIP patch carries the
variant B tables and no longer scales at unpack. Skips without torch / host compiler.
"""
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on this runner")
KV4 = Path(__file__).resolve().parents[2] / "kv4"
sys.path.insert(0, str(KV4))
import kv4i_offline as O  # noqa: E402

PATCH = (KV4.parent / "r4d" / "r4d_kv4i.patch").read_text()


def _e4m3(x):
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)


def test_staged_byte_of_a_bare_code_is_torch_e4m3(tmp_path):
    cxx = next((c for c in (os.environ.get("CXX"), "g++", "clang++", "c++") if c and shutil.which(c)), None)
    if cxx is None:
        pytest.skip("no host C++ compiler")
    src = tmp_path / "d.cpp"
    src.write_text('#include "kv4i_unpack.h"\n#include <cstdio>\n'
                   'int main() { for (int c = -8; c <= 7; ++c) printf("%d %u %u\\n", c, kv4i::code_bare(c), kv4i::code_e4m3(c)); }\n')
    subprocess.run([cxx, "-std=c++17", "-I", str(KV4), str(src), "-o", str(tmp_path / "d")], check=True, capture_output=True)
    rows = [list(map(int, ln.split())) for ln in subprocess.run([str(tmp_path / "d")], check=True, capture_output=True, text=True).stdout.splitlines()]
    assert len(rows) == 16
    for c, bare, lut in rows:
        assert bare == lut == int(torch.tensor([float(c)]).to(torch.float8_e4m3fn).view(torch.uint8)[0]), c


def _case(ctx=96, R=4, seed=3):
    g = torch.Generator().manual_seed(seed)
    cK, cV = (torch.randint(-8, 8, (ctx, 256), generator=g).float() for _ in range(2))
    sK, sV = (torch.rand(ctx, 1, generator=g) * 0.05 + 0.01 for _ in range(2))
    q = torch.randn(R, 256, generator=g) * 0.3
    return q, cK, sK, cV, sV, torch.arange(ctx - R, ctx), 1.0 / 256 * O.LOG2E * 16


def _lift(x):
    a = x.abs().amax(-1, keepdim=True)
    return torch.where(a > 0, torch.exp2(torch.floor(torch.log2(448.0 / a.clamp_min(1e-30)))), torch.ones_like(a))


def test_attend_b_equals_the_formulas():
    q, cK, sK, cV, sV, qpos, sc = _case()
    Lq = _lift(q)
    qe = _e4m3(q * Lq) / Lq                                        # e4m3 query, no scale folded in
    s = (qe @ cK.T) * sK[:, 0][None] * sc                          # fp32 score: QK on bare codes, then the K token scale
    s = s.masked_fill(torch.arange(cK.shape[0])[None] > qpos[:, None], -math.inf)
    m = s[:, :16].amax(-1, keepdim=True)                           # reference max: set by tile 0, never moves (spread < 8 octaves)
    assert (s.amax(-1, keepdim=True) - m < 8).all()
    p = torch.exp2(s - m)
    out = torch.zeros(q.shape[0], 256)
    lq = torch.zeros(q.shape[0], 1)
    for k0 in range(0, cK.shape[0], 16):
        pv = p[:, k0:k0 + 16] * sV[k0:k0 + 16, 0][None]            # V token scale folded into P
        L = _lift(pv)                                              # per row, per tile, power of two, amax in [224,448]
        assert ((pv * L).amax(-1) <= 448).all()
        p8 = _e4m3(pv * L)
        out += (p8 @ cV[k0:k0 + 16]) / L
        lq += (p8 / L / sV[k0:k0 + 16, 0][None]).sum(-1, keepdim=True)
    _, got, P = O.attend_b(q, cK, sK, cV, sV, qpos, sc, probs=True)
    assert (got - out / lq).norm() / (out / lq).norm() < 1e-3
    assert (P.sum(-1) - 1).abs().max() < 1e-4


def test_lazy_reference_arm_stays_close_to_the_per_tile_lift():
    q, cK, sK, cV, sV, qpos, sc = _case(ctx=256)
    _, a = O.attend_b(q, cK, sK, cV, sV, qpos, sc)
    _, b = O.attend_b(q, cK, sK, cV, sV, qpos, sc, lazy=True)
    assert (a - b).norm() / a.norm() < 0.03
    _, c = O.attend_b(q, cK, sK, cV, sV, qpos, sc, lazy=True, rnd=False)       # no rounding: exact softmax
    ref = O.attend(q * sc, cK * sK, cV * sV, qpos)
    assert (c - ref).abs().max() < 1e-4


def test_patch_stages_bare_codes_and_carries_the_scale_tables():
    assert "kv4_ksc" in PATCH and "kv4_lsv" in PATCH and "kv4_rsv" in PATCH
    assert "ks, ks, ks, ks" not in PATCH and "s01.x, s01.y" not in PATCH   # variant A's scale-at-unpack is gone


def test_kernel_denominator_is_a_two_term_fp8_dot():
    # lazy arm = libr4d: l = sum over keys of P8 * (hi + lo) / RLIFT, hi = e4m3(RLIFT/sV), lo = e4m3(RLIFT/sV - hi)
    # (one v_dot4_f32_fp8_fp8 per 4 keys per term instead of cvt + fma per key); P8 itself is unchanged.
    q, cK, sK, cV, sV, qpos, sc = _case(ctx=64)
    sV = sV * 400.0                                  # scales in the range of the real caches (3..16)
    _, got, P = O.attend_b(q, cK, sK, cV, sV, qpos, sc, lazy=True, probs=True)
    r = O.RLIFT / sV[:, 0]
    hi = _e4m3(r)
    rq = (hi + _e4m3(r - hi)) / O.RLIFT
    assert ((rq - 1 / sV[:, 0]).abs() * sV[:, 0]).max() < 2 ** -7        # two terms: ~2^-8 relative, not 2^-4
    _, exact = O.attend_b(q, cK, sK, cV, sV, qpos, sc, lazy=True, split=False)
    assert (got - exact).norm() / exact.norm() > 1e-6                    # the model really uses the quantized weights
    assert (P.sum(-1) - 1).abs().max() < 1e-4


def test_patch_denominator_is_fp8_dots_and_the_table_fill_is_lane_parallel():
    assert PATCH.count("__builtin_amdgcn_dot4_f32_fp8_fp8") == 4                  # 2 terms x 2 packed dwords per m-tile
    assert "cvt_pk_f32_fp8" not in PATCH and "float kv4_rsv" not in PATCH   # no cvt + fma, no float rsv table
    assert f"KV4I_RLIFT = {O.RLIFT}f" in PATCH and "vdesc * (KV4I ? KV4I_RLIFT" in PATCH
    assert "if (vd < 8)" in PATCH and "if (vd == 0)" not in PATCH                  # 8 lanes, one token each, not 1 lane x 8
