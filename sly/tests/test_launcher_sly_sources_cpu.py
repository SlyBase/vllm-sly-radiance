#!/usr/bin/env python3
"""CPU test: `serve-mxfp4.sh` builds and copies the kernel sources the image ships (`sly/mxfp4/`).

The repo-root `radiance_mxfp4_fp8.hip` / `radiance_mxfp4.py` are older twins. The root `.hip` has a
10-argument `launch_gdn_norm_quant`; `sly/mxfp4/radiance_fused_norm.py` (baked into the image) calls
it with 11. A launcher that compiles the root file breaks that call (issue #223).

Run:  python3 sly/tests/test_launcher_sly_sources_cpu.py
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (ROOT / "serve-mxfp4.sh").read_text()
HIP = "sly/mxfp4/radiance_mxfp4_fp8.hip"


def test_launcher_compiles_the_sly_hip():
    """Every compile of the kernel names the sly/ source; no bare root `.hip` is left."""
    assert f"hip-so-cache.sh {HIP}" in LAUNCHER and f"{HIP} -o" in LAUNCHER
    code = "\n".join(l for l in LAUNCHER.splitlines() if not l.lstrip().startswith("#")).replace(HIP, "")
    assert not re.search(r"(?<![/\w])radiance_mxfp4_fp8\.hip\b", code), \
        "serve-mxfp4.sh still compiles the repo-root radiance_mxfp4_fp8.hip"


def test_launcher_copies_the_sly_python():
    """The two modules that share the `launch_gdn_norm_quant` contract come from sly/mxfp4/."""
    for mod in ("radiance_mxfp4.py", "radiance_fused_norm.py"):
        assert f"sly/mxfp4/{mod}" in LAUNCHER, f"launcher does not copy sly/mxfp4/{mod}"
    cp_line = next(l for l in LAUNCHER.splitlines() if "cp radiance_nvfp4.py" in l)
    assert "radiance_mxfp4.py" not in cp_line, "root radiance_mxfp4.py is still in the cp list"


def test_hip_arity_matches_the_python_call():
    """The launched `.hip` takes `tiled`, which radiance_fused_norm.py always passes."""
    hip = (ROOT / HIP).read_text()
    sig = re.search(r"static void launch_gdn_norm_quant\((.*?)\)\s*\{", hip, re.S).group(1)
    assert sig.count(",") + 1 == 11
    assert "1 if tiled else 0" in (ROOT / "sly/mxfp4/radiance_fused_norm.py").read_text()


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
