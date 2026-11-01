"""CPU: the opt-in chunk_scan build variants s1 / s2 / s3 (docs/GDN-CHUNK-SCAN-VARIANTS.md).

Four things no GPU is needed for:
  * resource budget of each variant from compiler reports (fixtures, made by sly/gdn/scan_resources.sh);
  * the LDS hazard model of the s2 / s3 schedule (S aliased onto sA + W, one barrier added, SWRITE moved);
  * the patch really carries the pieces that model assumes (barrier (1b), SWRITE behind barrier (4),
    the 43,008 B static_assert, no -mcumode on the variant units, stock unit untouched);
  * the RADIANCE_GDN_SCAN_VARIANT knob: default off, unknown or missing variant keeps the stock kernel.
"""
import importlib
import pathlib
import re
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = pathlib.Path(__file__).parent / "fixtures" / "gdn_scan"
PATCH = (ROOT / "sly/r4d/r4d_gdn_scan_variants.patch").read_text()
STOCK, VARIANTS = "gdn_chunk_scan_k128_v128_c64_bf16", ("s1", "s2", "s3")


# ---- resources -----------------------------------------------------------------------------------
def report(name):
    text = (FIX / f"{name}.remark.txt").read_text()

    def num(label):
        m = re.search(label + r"\s*(?:\[[^\]]*\])?:\s*(\d+)", text)
        assert m, f"{label} missing in {name}"
        return int(m.group(1))
    return dict(vgpr=num("VGPRs"), spills=num("VGPRs Spill"), scratch=num("ScratchSize"),
                lds=num("LDS Size"), occ=num("Occupancy"))


def test_stock_is_the_known_baseline():
    r = report("stock")
    assert (r["vgpr"], r["spills"], r["scratch"], r["lds"], r["occ"]) == (256, 4, 20, 60928, 4)


def test_s1_changes_the_mode_only():
    assert report("s1") == report("stock")        # -mcumode does not show in the resource report


# Occupancy is reported in wave64: a workgroup is 256 threads = 4 waves, so three workgroups per WGP need
# 3 waves/SIMD -- and 128 KB of LDS / 43,008 B allows exactly 3. The diet alone is therefore enough; the
# register side already fits at stock's 256 VGPR (5 waves/SIMD). No amdgpu_waves_per_eu hint is used:
# the VGPR drop in s3 comes from the moved prefetch alone, which the fixtures prove.
def test_s2_fits_three_workgroups_per_wgp_without_spilling():
    r = report("s2")
    assert r["lds"] == 43008 and 3 * r["lds"] <= 128 * 1024
    assert r["spills"] == 0 and r["scratch"] == 0
    assert r["occ"] >= 3                            # three workgroups = 3 waves/SIMD
    assert r["vgpr"] <= 256                         # the diet must not cost registers


def test_s3_buys_the_register_budget_from_the_late_prefetch():
    r2, r3 = report("s2"), report("s3")
    assert r3["lds"] == r2["lds"] and r3["spills"] == 0 and r3["occ"] >= 3
    assert r3["vgpr"] < r2["vgpr"]                   # 189 < 256: no forced waves_per_eu anywhere


# ---- LDS hazard model ----------------------------------------------------------------------------
BAR = ("BAR",)
# (access, region) per phase of one chunk; regions that share bytes share a name in `alias`.
# The vectors ge/gv/rs/bt sit in the row padding of k / D (own bytes, never aliased), so they are modelled
# as their own regions.
STOCK_SCHED = dict(alias={}, chunk=[
    ("W", "k"), ("W", "D"), ("W", "vec"), BAR,                                       # staging, (1)
    ("R", "S"), ("R", "k"), ("R", "vec"), ("R", "D"), ("W", "W"), ("W", "A"), BAR,   # main loop, W / sA, (2)
    ("R", "W"), ("R", "vec"), ("W", "D"), BAR,                                       # V', (3)
    ("R", "A"), ("R", "D"), ("R", "k"), ("W", "S"), BAR,                             # O, state, SWRITE, (4)
])
ALIAS_ALL = {"S": {"u0", "u1"}, "W": {"u0"}, "A": {"u1"}}
ALIAS_SCHED = dict(alias=ALIAS_ALL, chunk=[
    ("W", "S"), ("W", "k"), ("W", "D"), ("W", "vec"), BAR,                           # SWRITE, staging, (1)
    ("R", "S"), ("R", "k"), BAR,                                                     # main loop, (1b)
    ("R", "vec"), ("R", "D"), ("W", "W"), ("W", "A"), BAR,                           # W / sA, (2)
    ("R", "W"), ("R", "vec"), ("W", "D"), BAR,                                       # V', (3)
    ("R", "A"), ("R", "D"), ("R", "k"), BAR,                                         # O, state, (4)
])


def hazards(sched, iters=3):
    """Conflicting access pairs to one region with no barrier between them (loop unrolled `iters` x)."""
    seq = sched["chunk"] * iters
    region = lambda b: sched["alias"].get(b, {b})  # noqa: E731
    bad = set()
    for i, a in enumerate(seq):
        if a is BAR:
            continue
        for b in seq[i + 1:]:
            if b is BAR:
                break
            if region(a[1]) & region(b[1]) and "W" in (a[0], b[0]):
                bad.add((a, b))
    return sorted(bad)


def test_stock_schedule_is_hazard_free():
    assert hazards(STOCK_SCHED) == []


def test_alias_schedule_is_hazard_free():
    assert hazards(ALIAS_SCHED) == []


def test_alias_without_barrier_1b_races():
    sched = dict(alias=ALIAS_ALL, chunk=[e for i, e in enumerate(ALIAS_SCHED["chunk"]) if i != 7])
    assert ("R", "S") in {a for a, _ in hazards(sched)}


def test_alias_with_swrite_before_barrier_4_races():
    body = [e for e in ALIAS_SCHED["chunk"] if e != ("W", "S")]
    sched = dict(alias=ALIAS_ALL, chunk=body[:-1] + [("W", "S"), BAR])
    assert hazards(sched)


# ---- the patch carries what the model assumes ---------------------------------------------------
def added(path_re):
    """The '+' lines of the patch hunks of one file."""
    out, on = [], False
    for ln in PATCH.splitlines():
        if ln.startswith("diff --git"):
            on = re.search(path_re, ln) is not None
        elif on and ln.startswith("+") and not ln.startswith("+++"):
            out.append(ln[1:])
    return "\n".join(out)


def hunks(path_re):
    """Every line of one file's hunks, prefix stripped, so context lines are visible too."""
    out, on = [], False
    for ln in PATCH.splitlines():
        if ln.startswith("diff --git"):
            on = re.search(path_re, ln) is not None
        elif on and ln[:1] in ("+", "-", " ") and not ln.startswith(("+++", "---")):
            out.append(ln[1:])
    return "\n".join(out)


def test_patch_kernel_source_matches_the_model():
    src = added(r"r4d_gdn_chunk_scan_k128_v128_c64_bf16\.hip$")
    assert "static_assert(sizeof(Smem) == 43008" in src
    assert re.search(r"BAR\(\);\s*// \(1b\)", src)
    assert re.search(r"#if ALIAS\n\s*SWRITE\(\);", src)           # SWRITE at the top of the chunk
    assert "#if PLATE == 3" in src
    # the stock SWRITE stays exactly where it was, behind the #if (context lines, hence `hunks`)
    assert re.search(r"#if !ALIAS\n(?:.*\n)*?\s*SWRITE\(\);\n#endif\n\s*BAR\(\);\s*// \(4\)",
                     hunks(r"r4d_gdn_chunk_scan_k128_v128_c64_bf16\.hip$"))
    # and no amdgpu_waves_per_eu hint anywhere: the register drop is the moved prefetch's doing
    assert "waves_per_eu" not in hunks(r"r4d_gdn_chunk_scan_k128_v128_c64_bf16\.hip$")


def test_variants_are_built_without_cumode_and_stock_keeps_it():
    build = added(r"build\.sh$")
    for v in VARIANTS:
        assert f'"r4d_gdn_chunk_scan_k128_v128_c64_bf16_{v}:"' in build
    assert "mcumode" not in build                    # every added unit line carries no flag


def test_patch_does_not_touch_the_stock_unit_flag():
    assert not [ln for ln in PATCH.splitlines() if ln.startswith("-") and "mcumode" in ln]


# ---- the knob ------------------------------------------------------------------------------------
def fake_r4d(has_variants):
    m = types.ModuleType("r4d")
    m.GDN_HEAD_K, m.GDN_HEAD_V, m.GDN_CHUNK = 128, 128, 64
    asked = []

    def select(op, **kw):
        asked.append((op, kw))
        if op != "gdn_chunk_scan":
            return f"{op}_stub"
        v = kw.get("variant")
        return f"{STOCK}_{v}" if has_variants and v in VARIANTS else STOCK
    m.select, m.asked = select, asked
    for name in (STOCK, *(f"{STOCK}_{v}" for v in VARIANTS), "gdn_conv_prep_stub", "gdn_kkt_solve_stub",
                 "gdn_conv_update_stub", "gdn_recurrent_update_stub", "gdn_fused_update_stub",
                 "gdn_gated_rmsnorm_stub"):
        setattr(m, name, (lambda n: (lambda *a: n))(name))
        getattr(m, name).__name__ = name
    return m


def load(monkeypatch, has_variants, knob):
    pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "r4d", fake_r4d(has_variants))
    for k in ("RADIANCE_GDN_SCAN_VARIANT", "RADIANCE_GDN_LAZY", "RADIANCE_GDN_FUSED_UPDATE"):
        monkeypatch.delenv(k, raising=False)
    if knob is not None:
        monkeypatch.setenv("RADIANCE_GDN_SCAN_VARIANT", knob)
    monkeypatch.setenv("RADIANCE_USE_R4D", "1")
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("radiance_gdn", None)
    return importlib.import_module("radiance_gdn")


def test_knob_unset_is_stock_and_never_asks_for_a_variant(monkeypatch):
    g = load(monkeypatch, True, None)
    assert g._CHUNK_SCAN.__name__ == STOCK
    assert all("variant" not in kw for _, kw in sys.modules["r4d"].asked)


@pytest.mark.parametrize("v", VARIANTS)
def test_knob_picks_the_variant(monkeypatch, v):
    assert load(monkeypatch, True, v)._CHUNK_SCAN.__name__ == f"{STOCK}_{v}"


def test_knob_unknown_variant_keeps_stock_and_says_so(monkeypatch, capsys):
    assert load(monkeypatch, True, "s9")._CHUNK_SCAN.__name__ == STOCK
    assert capsys.readouterr().err.count("RADIANCE_GDN_SCAN_VARIANT=s9") == 1


def test_knob_on_a_build_without_variants_keeps_stock(monkeypatch, capsys):
    assert load(monkeypatch, False, "s2")._CHUNK_SCAN.__name__ == STOCK
    assert capsys.readouterr().err.count("RADIANCE_GDN_SCAN_VARIANT=s2") == 1


if __name__ == "__main__":  # same entry point as the other CPU tests: `python3 sly/tests/<this file>`
    sys.exit(pytest.main([__file__, "-q"]))
