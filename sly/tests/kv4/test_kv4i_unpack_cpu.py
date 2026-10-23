"""CPU test for sly/kv4/kv4i_unpack.h: every formula of the int4 read against the torch oracle.

The header is plain C++ so it can be checked without a GPU and without hipcc: the test compiles a
tiny driver with the host compiler, runs it over the 16 codes and a grid of scales, and compares
each value against torch -- the exact-code LUT against `torch.float8_e4m3fn` (must be equal for all
16 codes, which is the claim that makes the LUT free), the scaled path against the same cast (must
be equal bit for bit, round-to-nearest-even), and the nibble helpers against the oracle's layout
(low nibble = dim 2p, high nibble = dim 2p+1, two's complement).

Skips when there is no host compiler or no torch, so it stays green on the bare CI runner.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
HEADER = HERE.parents[1] / "kv4" / "kv4i_unpack.h"

torch = pytest.importorskip("torch", reason="torch is not installed on this runner")


def _compiler():
    for c in (os.environ.get("CXX"), "g++", "clang++", "c++"):
        if c and shutil.which(c):
            return shutil.which(c)
    return None


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    cxx = _compiler()
    if cxx is None:
        pytest.skip("no host C++ compiler")
    src = tmp_path_factory.mktemp("kv4i") / "drive.cpp"
    src.write_text(
        '#include "kv4i_unpack.h"\n'
        "#include <cstdio>\n"
        "int main() {\n"
        "  for (int c = -8; c <= 7; ++c) printf(\"code %d %u\\n\", c, kv4i::code_e4m3(c));\n"
        "  for (int c = -8; c <= 7; ++c) printf(\"nib %d %d %d\\n\", c, kv4i::low(uint8_t(c) & 0xF),\n"
        "                                              kv4i::high(uint8_t(c) << 4));\n"
        "  double scales[] = {1.0, 0.5, 0.25, 0.8/7, 1.0/7, 0.125, 2.0, 0.0625, 0.001953125, 0.0};\n"
        "  for (double s : scales) for (int c = -8; c <= 7; ++c)\n"
        "    printf(\"scaled %d %.9g %u\\n\", c, s, kv4i::code_e4m3_scaled(c, float(s)));\n"
        "  double vals[] = {0.0, 1.0, -1.0, 0.5, 1.5, 448.0, -448.0, 449.0, 512.0, 0.001953125,\n"
        "                   0.0009765625, 0.0029296875, 3.0, 7.0, -8.0, 0.8/7, 1e9, -1e9, 1.0009765625};\n"
        "  for (double v : vals) printf(\"rne %.9g %u\\n\", v, kv4i::rne(float(v)));\n"
        "  return 0;\n"
        "}\n"
    )
    exe = src.with_suffix("")
    subprocess.run([cxx, "-std=c++17", "-O2", "-I", str(HEADER.parent), str(src), "-o", str(exe)],
                   check=True, capture_output=True)
    out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    rows = [ln.split() for ln in out.strip().splitlines()]
    return rows


def _e4m3_bits(values):
    t = torch.tensor([float(v) for v in values], dtype=torch.float32).to(torch.float8_e4m3fn)
    return [int(b) for b in t.view(torch.uint8).tolist()]


def test_code_lut_is_exact(driver):
    rows = [r for r in driver if r[0] == "code"]
    assert len(rows) == 16
    got = [int(r[2]) for r in rows]
    want = _e4m3_bits([int(r[1]) for r in rows])
    assert got == want, "the int4 -> e4m3 LUT must equal torch's cast for all 16 codes"


def test_nibbles_are_signed_and_split_low_high(driver):
    rows = [r for r in driver if r[0] == "nib"]
    assert len(rows) == 16
    for _, code, lo, hi in rows:
        c = int(code)
        # the packed byte holds the code twice (low and high nibble); both must decode to c
        assert int(lo) == c and int(hi) == c, f"nibble {c} decoded as {lo}/{hi}"


def test_scaled_path_matches_torch_rne(driver):
    rows = [r for r in driver if r[0] == "scaled"]
    assert rows
    codes = [int(r[1]) for r in rows]
    scales = [float(r[2]) for r in rows]
    got = [int(r[3]) for r in rows]
    want = _e4m3_bits([c * s for c, s in zip(codes, scales)])
    bad = [(c, s, g, w) for c, s, g, w in zip(codes, scales, got, want) if g != w]
    assert not bad, f"scaled path differs from torch in {len(bad)} of {len(rows)} cells: {bad[:4]}"


def test_float_path_matches_torch(driver):
    rows = [r for r in driver if r[0] == "rne"]
    assert rows
    vals = [float(r[1]) for r in rows]
    got = [int(r[2]) for r in rows]
    # torch has no e4m3 infinity: an overflow saturates, which the reference does for us
    want = _e4m3_bits(vals)
    bad = [(v, g, w) for v, g, w in zip(vals, got, want) if g != w]
    assert not bad, f"rne differs from torch: {bad}"


def test_zero_scale_collapses_the_row(driver):
    rows = [r for r in driver if r[0] == "scaled" and float(r[2]) == 0.0]
    # a negative code times zero is -0.0, and e4m3 keeps the sign bit (0x80), as torch does: check
    # the magnitude, not the sign
    assert rows and all(int(r[3]) & 0x7F == 0 for r in rows), "a zero scale must give zero magnitude"
