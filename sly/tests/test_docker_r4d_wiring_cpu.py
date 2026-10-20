#!/usr/bin/env python3
"""Wiring test: the image build carries BOTH r4d source changes, not just the extras patch.

Why this exists: `sly/r4d/r4d_extras_rx10.patch` is the only r4d input the kernels stage knew
about. The v2 chunk_scan arrived as a second, separate patch (`r4d_gdn_scan_v2.patch`, which adds
the unit to libr4d's `build.sh` and the `variant="v2"` registry row) plus a new translation unit
(`r4d_gdn_chunk_scan_k128_v128_c64_bf16_v2.hip`) that is a repo file rather than a hunk. Neither was
COPYed, so `RADIANCE_GDN_SCAN_V2=1` in a built image asked the registry for a variant no kernel
provides and silently ran the stock scan -- a knob that measures nothing. This pins the two lines
that make the variant exist in the image.

The checks read the Dockerfile as text (no docker, no ROCm, no GPU): the copy, the apply order, and
the fact that the unit lands in the libr4d source tree under the name the patch's build.sh compiles.

Run:  python3 sly/tests/test_docker_r4d_wiring_cpu.py
      (exit 0 = pass; also collectable by pytest)
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "Dockerfile"
R4D = ROOT / "sly" / "r4d"
EXTRAS = "r4d_extras_rx10.patch"
V2_PATCH = "r4d_gdn_scan_v2.patch"
V2_UNIT = "r4d_gdn_chunk_scan_k128_v128_c64_bf16_v2.hip"


def dockerfile():
    return DOCKERFILE.read_text()


def r4d_run_block():
    """The single RUN that clones, patches and builds libr4d (the kernels stage)."""
    text = dockerfile()
    m = re.search(r"^COPY sly/r4d/" + re.escape(EXTRAS) + r".*?^ && cd / && rm -rf /src/libr4d$",
                  text, re.S | re.M)
    assert m, "the libr4d RUN block (COPY r4d_extras_rx10.patch .. rm -rf /src/libr4d) is gone"
    return m.group(0)


def test_both_r4d_inputs_are_copied_into_the_kernels_stage():
    text = dockerfile()
    for name in (EXTRAS, V2_PATCH, V2_UNIT):
        assert re.search(r"^COPY sly/r4d/" + re.escape(name) + r" /opt/kernels/$", text, re.M), \
            f"sly/r4d/{name} is not COPYed into /opt/kernels by the kernels stage"
        assert (R4D / name).is_file(), f"sly/r4d/{name} does not exist"


def test_v2_patch_is_applied_after_the_extras_patch_and_before_build_sh():
    block = r4d_run_block()
    extras = block.find(f"git apply /opt/kernels/{EXTRAS}")
    v2 = block.find(f"git apply /opt/kernels/{V2_PATCH}")
    build = block.find("./build.sh")
    assert extras >= 0, "the extras patch is no longer applied in the libr4d RUN block"
    assert v2 >= 0, ("r4d_gdn_scan_v2.patch is not applied in the libr4d RUN block -- its hunks "
                     "touch the files the extras patch already rewrote (build.sh, r4d.h, "
                     "r4d_module.hip, r4d_registry.hip), so it has to come second")
    assert extras < v2, "r4d_gdn_scan_v2.patch must be applied AFTER r4d_extras_rx10.patch"
    assert build >= 0, "./build.sh is gone from the libr4d RUN block"
    assert v2 < build, "the v2 patch has to be applied before ./build.sh runs"


def test_the_unit_is_staged_under_the_name_the_patch_builds():
    """build.sh compiles `<unit>.hip` out of the source root, so the COPY must land there."""
    patch = (R4D / V2_PATCH).read_text()
    # the row is "<unit>[:<flags>]"; v2 builds with the stock unit's flags, so -mcumode is expected
    m = re.search(r'^\+\s*"' + re.escape(V2_UNIT.removesuffix(".hip")) + r'(?::-mcumode)?"', patch,
                  re.M)
    assert m, f"r4d_gdn_scan_v2.patch does not add the {V2_UNIT.removesuffix('.hip')} unit to build.sh"
    block = r4d_run_block()
    staged = re.search(r"cp /opt/kernels/" + re.escape(V2_UNIT) + r" \.?/?\s*\\", block)
    assert staged, f"the unit is never copied into /src/libr4d (cp /opt/kernels/{V2_UNIT} .)"
    assert block.find(staged.group(0)) < block.find("./build.sh"), \
        "the unit has to be in the source tree before ./build.sh runs"


def test_the_hip_does_not_need_a_second_hipcc_invocation():
    """One OUT=/out/r4d.so build.sh call: build.sh owns the unit list, the Dockerfile does not."""
    block = r4d_run_block()
    assert block.count("./build.sh") == 1, "the libr4d RUN block builds more than once"
    assert block.count("OUT=/out/r4d.so") == 1, "OUT must be set on the single build.sh call"
    assert "hipcc" not in block, "hipcc in the libr4d RUN block means the unit list is duplicated"


def test_both_knobs_stay_opt_in() -> object:
    """The image ships the code; the env decides whether it runs (default 0 = stock)."""
    for path, knob in ((R4D / V2_PATCH, "variant"), (R4D / V2_UNIT, "PLATE")):
        assert path.is_file()
    gdn = (ROOT / "radiance_gdn.py").read_text()
    assert re.search(r'"RADIANCE_GDN_SCAN_V2",\s*"0"', gdn), \
        "RADIANCE_GDN_SCAN_V2 must default to 0 (stock) in radiance_gdn.py"


if __name__ == "__main__":
    test_both_r4d_inputs_are_copied_into_the_kernels_stage()
    test_v2_patch_is_applied_after_the_extras_patch_and_before_build_sh()
    test_the_unit_is_staged_under_the_name_the_patch_builds()
    test_the_hip_does_not_need_a_second_hipcc_invocation()
    test_both_knobs_stay_opt_in()
    print("docker r4d wiring tests OK")
    sys.exit(0)
