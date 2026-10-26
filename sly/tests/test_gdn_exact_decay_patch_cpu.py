"""CPU: sly/r4d/r4d_gdn_exact_decay.patch (libr4d issue #4) -- chunk_scan takes the exact decay form for a chunk
whose gate span g_0 - g_last exceeds 160, where the split form's clamp at e^80 attenuates the chunk.

No GPU is needed for:
  * the patch carries the pieces the fp64 reference models (sly/gdn/chunk_scan_ref.py, split="hybrid"):
    the 160 threshold, EXACT on by default, the masked exact score, K scaled by e^{g_last-g_j} in place,
    the undecayed state update -- and leaves the LDS layout (43,008 B) and the patch stack order alone;
  * resource budget of every unit with the patch applied (fixtures in fixtures/gdn_scan_exact, made by
    `DEST=sly/tests/fixtures/gdn_scan_exact sly/gdn/scan_resources.sh <libr4d + all three patches>`).
The maths itself is pinned against the naive recurrence in sly/gdn/tests/test_chunk_scan_ref.py.
"""
import pathlib
import re
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_gdn_scan_variants_cpu import report as variants_report  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = pathlib.Path(__file__).parent / "fixtures" / "gdn_scan_exact"
PATCH = (ROOT / "sly/r4d/r4d_gdn_exact_decay.patch").read_text()
UNITS = ("stock", "s1", "s2", "s3")


def added():
    return "\n".join(ln[1:] for ln in PATCH.splitlines() if ln.startswith("+") and not ln.startswith("+++"))


def test_patch_touches_the_scan_source_only():
    assert re.findall(r"^diff --git a/(\S+)", PATCH, re.M) == ["r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip"]


def test_exact_is_on_by_default():
    assert re.search(r"#ifndef EXACT\n#define EXACT 1\n#endif", added())


def test_threshold_matches_the_reference():
    sys.path.insert(0, str(ROOT / "sly/gdn"))
    from chunk_scan_ref import SPLIT_MAX_SPAN
    assert re.search(r"split\s*=\s*g0 - gl <= %.1ff;" % SPLIT_MAX_SPAN, added())


def test_exact_score_is_masked_before_the_exponent_is_used():
    # j > i would be e^{+span}: inf * 0 = NaN if it were multiplied and masked afterwards
    assert re.search(r"\(j0 \+ e <= ii\) \? p\[i\]\[e\] \* __expf\(gis - VEC_RS\(j0 \+ e\)\) : 0\.f", added())


def test_exact_state_update_has_no_split_factor():
    a = added()
    assert re.search(r"ec = split \? __expf\(cref\) : __expf\(gl\)", a)
    assert re.search(r"lam = split \? __expf\(gl - cref\) : 1\.f", a)
    assert re.search(r"#if EXACT\n\s*if \(!split\)", a)            # K <- e^{g_last-g_j} K only off the split path


def test_lds_layout_is_unchanged():
    assert "static_assert(sizeof(Smem)" not in "\n".join(
        ln for ln in PATCH.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---")))


def test_dockerfile_applies_it_after_the_variants():
    df = (ROOT / "Dockerfile").read_text()
    order = [df.index(f"git apply /opt/kernels/{p}.patch") for p in
             ("r4d_extras_rx10", "r4d_gdn_scan_variants", "r4d_gdn_exact_decay")]
    assert order == sorted(order)
    assert re.search(r"^COPY .*sly/r4d/r4d_gdn_exact_decay\.patch .*/opt/kernels/", df, re.M)


# ---- resources -----------------------------------------------------------------------------------
def report(name):
    text = (FIX / f"{name}.remark.txt").read_text()

    def num(label):
        m = re.search(label + r"\s*(?:\[[^\]]*\])?:\s*(\d+)", text)
        assert m, f"{label} missing in {name}"
        return int(m.group(1))
    return dict(vgpr=num("VGPRs"), spills=num("VGPRs Spill"), scratch=num("ScratchSize"),
                lds=num("LDS Size"), occ=num("Occupancy"))


@pytest.mark.parametrize("unit", UNITS)
def test_no_unit_spills_and_none_loses_occupancy(unit):
    r, base = report(unit), variants_report(unit)
    assert r["spills"] == 0 and r["scratch"] == 0
    assert r["lds"] == base["lds"] and r["occ"] >= base["occ"]
    assert r["vgpr"] <= 256


def test_s3_keeps_three_workgroups_per_wgp():
    r = report("s3")
    assert r["lds"] == 43008 and r["occ"] >= 3
