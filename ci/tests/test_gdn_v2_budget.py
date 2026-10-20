"""Resource budget of the chunk_scan v2 variant.

Parses a `hipcc -Rpass-analysis=kernel-resource-usage` report (ci/gdn_scan_resources.sh, CPU only).
The variant changes no LDS buffer, no barrier, no grid and no arithmetic step: the slice rests on
per-workgroup efficiency, because the w3 measurement showed that more waves per SIMD do not pay
(docs/GDN-CHUNK-SCAN-V2.md). So the gates are the stock geometry -- LDS exactly 60,928 B, 4
waves/SIMD, VGPR <= 256 -- plus the point of the slice: ZERO spills, where the stock kernel spills
four of them (20 B/lane of scratch traffic inside the chunk loop).
"""
import pathlib
import re

FIX = pathlib.Path(__file__).parent / "fixtures"
STOCK_VGPR, STOCK_SPILLS, STOCK_SCRATCH = 256, 4, 20      # libr4d a3e4833 + extras, CPU compile
LDS_STOCK = 60928
OCC_STOCK = 4


def parse(text):
    def num(label):
        m = re.search(label + r"\s*(?:\[[^\]]*\])?:\s*(\d+)", text)
        assert m, f"{label} missing in report"
        return int(m.group(1))
    return dict(vgpr=num("VGPRs"), spills=num("VGPRs Spill"), scratch=num("ScratchSize"),
                lds=num("LDS Size"), occ=num("Occupancy"))


def check_v2(r):
    """Return the list of violated gates (empty = pass)."""
    bad = []
    if r["vgpr"] > STOCK_VGPR: bad.append(f"VGPR {r['vgpr']} > {STOCK_VGPR}")
    if r["spills"] != 0: bad.append(f"spills {r['spills']} != 0")
    if r["scratch"] != 0: bad.append(f"scratch {r['scratch']} != 0")
    if r["lds"] != LDS_STOCK: bad.append(f"LDS {r['lds']} != {LDS_STOCK}")
    if r["occ"] < OCC_STOCK: bad.append(f"occupancy {r['occ']} < {OCC_STOCK}")
    return bad


def test_stock_is_the_baseline_this_slice_measures_against():
    r = parse((FIX / "gdn_scan_stock.remark.txt").read_text())
    assert (r["vgpr"], r["spills"], r["scratch"], r["lds"]) == (STOCK_VGPR, STOCK_SPILLS,
                                                               STOCK_SCRATCH, LDS_STOCK)


def test_stock_fails_the_no_spill_gate():
    # the red that this slice exists to turn green
    assert check_v2(parse((FIX / "gdn_scan_stock.remark.txt").read_text())) == ["spills 4 != 0",
                                                                              "scratch 20 != 0"]


def test_v2_within_budget():
    r = parse((FIX / "gdn_scan_v2.remark.txt").read_text())
    assert check_v2(r) == [], r
