#!/usr/bin/env python3
"""Wiring test: NSD and `draft_sample_method` are switchable per env, and the default that ships is
the default that is documented. No torch, no triton, no GPU.

Why this exists: both knobs were only reachable by editing a launcher or by mounting a patch into the
image. NSD's patch was not in the Dockerfile loop (so `RADIANCE_NSD=1` did nothing in a built image),
`serve.sh` hard-coded `draft_sample_method: greedy` while README/NOT-ADOPTED called the production
value `probabilistic`, and NSD's Cactus exclusivity check read `RADIANCE_CACTUS_DELTA` with its own
default 0, so it passed while the image's Cactus patch used 0.5 -> the union of both rules, which the
paper's bound does not cover. The checks below pin all four down.

Run:  python3 sly/tests/test_spec_knobs_cpu.py [path-to-pinned-rejection_sampler_utils.py]
      (exit 0 = pass; the apply checks skip without a pinned vLLM source)
"""
import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NSD = ROOT / "sly" / "patch_nsd.py"
CACTUS = ROOT / "sly" / "patch_cactus.py"
README = ROOT / "README.md"
SERVES = (ROOT / "serve.sh", ROOT / "serve-mxfp4.sh")

sys.path.insert(0, str(ROOT / "ci"))
from patch_list import patch_names  # noqa: E402

_ARGV_SOURCE = None


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print("SKIP", msg)


def _pinned_source():
    if _ARGV_SOURCE is not None:
        return _ARGV_SOURCE
    cand = (ROOT / ".dryrun" / "src" / "vllm" / "vllm" / "v1" / "worker" / "gpu"
            / "spec_decode" / "rejection_sampler_utils.py")
    return cand if cand.exists() else None


def test_nsd_and_cactus_are_baked_into_the_image():
    """A patch that is not in the Dockerfile loop is not in the image: the env knob would be inert."""
    loop = patch_names()
    assert "sly/patch_nsd" in loop, "sly/patch_nsd is not in the Dockerfile patch loop"
    assert "sly/patch_cactus" in loop, "sly/patch_cactus left the loop"
    assert NSD.is_file(), "sly/patch_nsd.py missing"
    for name in ("sly/patch_nsd", "sly/patch_cactus"):
        assert (ROOT / f"{name}.py").is_file(), f"{name}.py missing"


def test_nsd_defaults_to_off_and_cactus_default_to_half():
    """Default off (no numerics change without an explicit env), and the exclusivity check must use
    the Cactus knob's *effective* default (0.5, sly/patch_cactus.py), not 0."""
    src = NSD.read_text()
    flat = src.replace('\\"', '"')          # the hunks are escaped strings inside the patch source
    assert '"RADIANCE_NSD", "0"' in flat, "RADIANCE_NSD must default to 0"
    assert '"RADIANCE_NSD_THETA", "0.95"' in flat, "RADIANCE_NSD_THETA must default to 0.95"
    # the name stays split (patch_cactus.py uses the whole literal as its "already applied" sentinel)
    m = re.search(r'RADIANCE_CACTUS" \+ "_DELTA", "([0-9.]+)"', flat)
    assert m, "the NSD exclusivity check no longer reads the Cactus knob"
    assert m.group(1) == "0.5", (
        "the NSD exclusivity check must default the Cactus knob to 0.5 (patch_cactus.py's default); "
        f"with {m.group(1)} NSD=1 silently runs Cactus 0.5 OR NSD")
    # and the message has to say how to get a legal combination (with the name split, like the read:
    # a literal RADIANCE_CACTUS_DELTA in the patched file would trip patch_cactus.py's sentinel and
    # make it skip its own env read when NSD ran first -- caught by the order test below)
    assert 'set RADIANCE_CACTUS" + "_DELTA=0' in flat, "the exclusivity error must name the way out"


def _scratch_site(src: Path):
    td = tempfile.TemporaryDirectory()
    site = Path(td.name) / "site-packages"
    pkg = site / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"
    pkg.mkdir(parents=True)
    target = pkg / "rejection_sampler_utils.py"
    shutil.copy(src, target)
    # patch_cactus.py also patches the caller next to it (prompt lengths for the length fade)
    shutil.copy(Path(src).with_name("rejection_sampler.py"), target.with_name("rejection_sampler.py"))
    return td, site, target


def _run(patch: Path, site: Path):
    env = dict(os.environ, PYTHONPATH=str(ROOT), RADIANCE_PATCH_TARGET_DIR=str(site))
    return subprocess.run([sys.executable, str(patch)], env=env, capture_output=True, text=True)


def test_applies_to_the_pinned_source_in_either_order():
    """NSD and Cactus hook the same function; the Dockerfile order may not matter."""
    src = _pinned_source()
    if src is None:
        return _skip("pinned vLLM source not available (ci/patch_dryrun.sh covers apply + idempotence)")
    seen = []
    for order in ((CACTUS, NSD), (NSD, CACTUS)):
        td, site, target = _scratch_site(src)
        with td:
            for p in order:
                r = _run(p, site)
                assert r.returncode == 0, f"{p.name} failed:\n{r.stdout}\n{r.stderr}"
            patched = target.read_text()
            ast.parse(patched)
            # second pass: everything already applied, nothing re-applied
            r2 = _run(NSD, site)
            assert r2.returncode == 0 and "OK" not in r2.stdout and "NOOP" in r2.stdout, r2.stdout
            assert patched.count('"RADIANCE_NSD"') == 1, "env read must be module-level, once"
            assert '"RADIANCE_CACTUS" + "_DELTA", "0.5"' in patched, "exclusivity default must be 0.5"
            assert "USE_NSD: tl.constexpr" in patched and "if USE_NSD:" in patched, \
                "the knob-off path must be a constexpr branch (compiles as stock)"
            assert "cactus_delta" in patched, "the Cactus hunks must survive the NSD patch"
            seen.append(sorted(l for l in patched.splitlines() if not l.strip().startswith("#")))
    assert seen[0] == seen[1], "patch order changes the resulting file"


def test_draft_sample_method_is_switchable_and_documented():
    """One env (`DRAFT_SAMPLE`) in both launchers, and README names the same default the code ships."""
    defaults = set()
    for serve in SERVES:
        txt = serve.read_text()
        m = re.search(r"DRAFT_SAMPLE=\$\{DRAFT_SAMPLE:-(\w+)\}", txt)
        assert m, f"{serve.name} does not default DRAFT_SAMPLE"
        defaults.add(m.group(1))
        assert 'draft_sample_method\\":\\"$DRAFT_SAMPLE' in txt, \
            f"{serve.name} does not pass $DRAFT_SAMPLE into the speculative config"
    assert len(defaults) == 1, f"the launchers disagree on the default: {defaults}"
    default = defaults.pop()

    readme = README.read_text()
    row = re.search(r"\|\s*`--speculative-config\.draft_sample_method`\s*\|\s*`([a-z]+)`\s*\|", readme)
    assert row, "README has no draft_sample_method row in the vLLM arguments table"
    assert row.group(1) == default, f"README documents {row.group(1)!r}, the launchers ship {default!r}"
    other = "probabilistic" if default == "greedy" else "greedy"
    assert f"DRAFT_SAMPLE={other}" in readme, f"README must show how to switch to {other}"


def test_nsd_is_documented_as_opt_in():
    """The default the README documents has to be the default the patch code carries -- text pinned
    against text would not notice the code moving off 0."""
    code = NSD.read_text().replace('\\"', '"')
    m = re.search(r'"RADIANCE_NSD", "([^"]+)"', code)
    assert m, "sly/patch_nsd.py no longer reads RADIANCE_NSD with a literal default"
    default = m.group(1)
    readme = README.read_text()
    assert re.search(rf"\|\s*`RADIANCE_NSD`\s*\|\s*`{re.escape(default)}`", readme), \
        f"README must document RADIANCE_NSD with its default {default} (sly/patch_nsd.py)"
    assert "RADIANCE_NSD=1" in readme, "README must show how to switch NSD on"
    assert "RADIANCE_CACTUS_DELTA=0" in readme, \
        "README must say that NSD=1 needs the Cactus knob at 0"
    sly_readme = (ROOT / "sly" / "README.md").read_text()
    assert "patch_nsd.py" in sly_readme, "sly/README.md must document patch_nsd.py"


if __name__ == "__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        _ARGV_SOURCE = Path(sys.argv[1])
    test_nsd_and_cactus_are_baked_into_the_image()
    test_nsd_defaults_to_off_and_cactus_default_to_half()
    test_applies_to_the_pinned_source_in_either_order()
    test_draft_sample_method_is_switchable_and_documented()
    test_nsd_is_documented_as_opt_in()
    print("spec knobs wiring tests OK")
