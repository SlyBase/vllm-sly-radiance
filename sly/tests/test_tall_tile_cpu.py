"""CPU check of the opt-in tall GEMM tile dispatch rule (band_map replica + .hip source agreement)."""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mxfp4"))
import band_map as B  # noqa: E402

HIP = os.path.join(os.path.dirname(__file__), "..", "mxfp4", "radiance_mxfp4_fp8.hip")
GATE_UP, IN_PROJ = 34816, 16480


def lab(M, N, **kw):
    args = dict(tiled_prod=True, decode_max=128, wide_max=0, tn4_min=513, a_tiled_min=513,
                scratch_bytes=1 << 30)
    args.update(kw)
    return B.label(M, N, 5120, **args)[0]


def test_default_off():
    for M in (2048, 4089, 4096):
        assert "TALL" not in lab(M, GATE_UP)
        assert "TALL" not in lab(M, IN_PROJ)


def test_on_only_for_gate_up_and_large_m():
    assert lab(2048, GATE_UP, tall=1) == "atiled TALL TMT8/TN2 (BMF512)"
    assert lab(4089, GATE_UP, tall=1) == lab(2048, GATE_UP, tall=1)
    assert "TALL" not in lab(2047, GATE_UP, tall=1)      # below the default min M
    assert "TALL" not in lab(1514, GATE_UP, tall=1)


def test_in_proj_keeps_tn2_tn4():
    # The tall tile measured up to 2 % slower on the merged GDN in_proj, so it is gate_up only:
    # the knob must leave that shape on the TN2/TN4 tile at every M.
    for M in (1514, 2048, 4089, 4096):
        assert "TALL" not in lab(M, IN_PROJ, tall=1)
        assert "TALL" not in lab(M, IN_PROJ, tall=1, tall_min=0)
    for N in (16384, 14336, 5120, 96):                   # qkvz, qkv, down/out_proj, ba
        assert "TALL" not in lab(4096, N, tall=1)


def test_min_m_configurable():
    assert "TALL" in lab(880, GATE_UP, tall=1, tall_min=800)
    assert "TALL" not in lab(799, GATE_UP, tall=1, tall_min=800)


def test_hip_knob_and_defaults_match():
    src = open(HIP).read()
    assert re.search(r'getenv\("RADIANCE_MXFP4_TALL"\);\s*return e \? atoi\(e\) : 0;', src)
    assert re.search(r'getenv\("RADIANCE_MXFP4_TALL_MIN_M"\);\s*return e \? atoi\(e\) : 2048;', src)
    assert re.search(r"g_at_tall && wperm && !wh && N == 34816 && tn_m >= g_at_tall_min_m", src)
