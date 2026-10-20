"""CPU: sly/r4d/r4d_gdn_scan_v2.patch -- the new chunk_scan translation unit and its wiring.

Two layers, so the test is useful on a machine without the libr4d sources:

* always: the patch is a well-formed unified diff, adds the variant row to the registry (with
  `variant="v2"`, so the stock row keeps answering a query without that key), binds the entry point
  in r4d_module.hip and adds the unit to build.sh, and touches nothing else;
* when R4D_SRC holds a pin checkout (a3e4833 + sly/r4d/r4d_extras_rx10.patch): the patch applies
  cleanly to a copy, and build.sh then compiles the new unit and the registry names it.
"""
import os
import pathlib
import re
import shutil
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATCH = ROOT / "sly/r4d/r4d_gdn_scan_v2.patch"
UNIT = ROOT / "sly/r4d/r4d_gdn_chunk_scan_k128_v128_c64_bf16_v2.hip"
SRC = pathlib.Path(os.environ.get("R4D_SRC", "/opt/data/cache/scratch/arch-r4d/pin"))
PIN_FILES = ("build.sh", "r4d.h", "r4d_module.hip", "r4d_registry.hip")


def text():
    return PATCH.read_text()


def test_unit_and_patch_exist():
    assert UNIT.exists() and PATCH.exists()


def test_patch_is_a_unified_diff_of_the_four_wiring_files_only():
    files = re.findall(r"^diff --git a/(\S+) b/\S+$", text(), re.M)
    assert files == ["build.sh", "r4d.h", "r4d_module.hip", "r4d_registry.hip"]


def test_registry_row_carries_the_variant_key_and_the_stock_row_does_not():
    t = text()
    assert re.search(r'^\+.*C_IN\("variant", "v2"\)', t, re.M)
    # every added constraint line that mentions variant must be inside the v2 table
    assert t.count('C_IN("variant", "v2")') == 1


def test_module_binds_the_entry_point_and_build_compiles_the_unit():
    t = text()
    assert '+  m.def("gdn_chunk_scan_k128_v128_c64_bf16_v2"' in t
    assert '+  "r4d_gdn_chunk_scan_k128_v128_c64_bf16_v2:"' in t
    # the stock unit keeps its -mcumode; v2 is a separate TU with no mode flag
    assert "-mcumode" not in "".join(l for l in t.splitlines(True)
                                     if l.startswith("+") and "r4d_gdn_chunk_scan" in l)


def test_version_is_not_bumped():
    assert "VERSION" not in re.findall(r"^diff --git a/(\S+)", text(), re.M)


@pytest.mark.skipif(not all((SRC / f).exists() for f in PIN_FILES),
                    reason="libr4d pin checkout absent (R4D_SRC)")
def test_patch_applies_to_a_pin_copy(tmp_path):
    src = tmp_path / "pin"
    shutil.copytree(SRC, src)
    shutil.copy(UNIT, src / UNIT.name)          # the unit is a repo file, not a hunk
    r = subprocess.run(["git", "apply", "-v", str(PATCH)], cwd=src, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    build = (src / "build.sh").read_text()
    assert '"r4d_gdn_chunk_scan_k128_v128_c64_bf16_v2:"' in build
    assert '"r4d_gdn_chunk_scan_k128_v128_c64_bf16:-mcumode"' in build
    reg = (src / "r4d_registry.hip").read_text()
    assert 'C_IN("variant", "v2")' in reg and '"gdn_chunk_scan_k128_v128_c64_bf16_v2"' in reg
    # select() answers the first match and the stock row ignores an unknown `variant` key, so the
    # variant row has to come first or a variant="v2" query would never reach it
    assert reg.index('"gdn_chunk_scan_k128_v128_c64_bf16_v2"') < \
        reg.index('{"gdn_chunk_scan_k128_v128_c64_bf16", "gdn"')


@pytest.mark.skipif(not all((SRC / f).exists() for f in PIN_FILES),
                    reason="libr4d pin checkout absent (R4D_SRC)")
def test_patch_is_idempotent_in_the_sense_that_a_second_apply_fails(tmp_path):
    src = tmp_path / "pin"
    shutil.copytree(SRC, src)
    shutil.copy(UNIT, src / UNIT.name)
    subprocess.run(["git", "apply", str(PATCH)], cwd=src, check=True)
    again = subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=src,
                           capture_output=True, text=True)
    assert again.returncode != 0
