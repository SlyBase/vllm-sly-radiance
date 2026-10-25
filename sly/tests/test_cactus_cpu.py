#!/usr/bin/env python3
"""CPU tests for sly/patch_cactus.py: the Cactus acceptance transform in vLLM 0.31's
spec-decode rejection kernel (vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py).

Cactus (arxiv 2604.04987) raises the acceptance threshold from the target probability
q_n to h_n = clamp(q_n + sqrt(2*delta*q_n*(1-q_n)), 0, 1), trading a bounded KL budget
for a higher acceptance rate. The patch threads `delta` (env RADIANCE_CACTUS_DELTA)
into the Leviathan-branch acceptance test; delta=0 must stay bit-exact.

Plain torch + subprocess only, no vllm, no GPU. The GPU half (the Triton kernel) is
proven by the accept gate.

Run:  python3 sly/tests/test_cactus_cpu.py [path-to-pinned-rejection_sampler_utils.py]
      (exit 0 = pass)
"""
import ast
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "sly" / "patch_cactus.py"


def h_n(q: float, delta: float) -> float:
    """Reference Cactus transform (the formula the kernel hunk must implement)."""
    return min(max(q + math.sqrt(2.0 * delta * q * (1.0 - q)), 0.0), 1.0)


def test_math_properties():
    for delta in (0.0, 0.2, 0.5, 1.0):
        assert h_n(0.0, delta) == 0.0 and h_n(1.0, delta) == 1.0, "endpoints must stay fixed"
        for q in (0.1, 0.35, 0.5, 0.8, 0.99):
            h = h_n(q, delta)
            assert 0.0 <= h <= 1.0, "h must stay in [0, 1]"
            if delta > 0:
                assert h > q, f"h must exceed q in the interior (q={q}, delta={delta})"
    # delta=0 is the exact identity (the bit-exact guard in the kernel relies on it)
    for q in (0.0, 0.1, 0.35, 0.5, 0.8, 0.99, 1.0):
        assert h_n(q, 0.0) == q, "delta=0 must be the identity"


def test_acceptance_rate_increases():
    """Monte Carlo: under the same (p, q), the Cactus rule (u*p < h) accepts at least as
    much as the standard rule (u*p < q), strictly more in the interior."""
    random.seed(1234)
    p, q = 0.9, 0.35  # p > h for all delta below, so the rates stay unsaturated
    rates = []
    for delta in (0.0, 0.5, 1.0):
        h = h_n(q, delta)
        rates.append(sum(random.random() * p < h for _ in range(100_000)) / 100_000)
    assert rates[0] < rates[1] < rates[2], f"rates must rise with delta: {rates}"
    assert 0.35 < rates[0] < 0.45, f"delta=0 must match the standard rate q/p={q / p}: {rates[0]}"


def _run_patch_on(src: Path) -> str:
    """Apply sly/patch_cactus.py to a scratch copy of the pinned source file; return the text."""
    with tempfile.TemporaryDirectory() as td:
        site = Path(td) / "site-packages"
        pkg = site / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"
        pkg.mkdir(parents=True)
        target = pkg / "rejection_sampler_utils.py"
        shutil.copy(src, target)
        # patch_cactus.py also patches the caller next to it (prompt lengths for the length fade)
        shutil.copy(Path(src).with_name("rejection_sampler.py"), target.with_name("rejection_sampler.py"))
        env = dict(os.environ, PYTHONPATH=str(ROOT), RADIANCE_PATCH_TARGET_DIR=str(site))
        r1 = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)
        assert r1.returncode == 0, f"pass 1 failed:\n{r1.stdout}\n{r1.stderr}"
        assert "OK" in r1.stdout, f"pass 1 did not apply: {r1.stdout}"
        r2 = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)
        assert r2.returncode == 0, f"pass 2 failed:\n{r2.stdout}\n{r2.stderr}"
        assert "NOOP" in r2.stdout, f"pass 2 not idempotent: {r2.stdout}"
        return target.read_text()


def test_default_delta_is_half():
    # Release default: delta=0.5 when the env is unset; an explicit RADIANCE_CACTUS_DELTA=0
    # must still select the bit-exact stock path (kernel guard `cactus_delta == 0.0`).
    src = PATCH.read_text()
    m = re.search(r'environ\.get\(\\?"RADIANCE_CACTUS_DELTA\\?",\s*\\?"([0-9.]+)\\?"\)', src)
    assert m, "env read with default not found"
    assert m.group(1) == "0.5", f"default must be 0.5, got {m.group(1)}"


def h_capped(q: float, r: float, delta: float = 0.5) -> float:
    """Fade-free cap formula: h' = min(h_cactus, 1 - r*(1-q))."""
    return min(h_n(q, delta), 1.0 - r * (1.0 - q))


def escape(q: float, p: float, rule) -> float:
    """Chance that a drafted token with target prob q and draft prob p is rejected."""
    return 1.0 - min(1.0, rule(q) / p)


def test_cap_never_stricter_than_stock():
    # The cap must only ever give away Cactus gain, never accept less than the stock rule.
    for r in (0.0, 0.25, 0.5, 0.75, 1.0):
        for q in (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0):
            assert h_capped(q, r) >= q - 1e-12, f"h' < q at q={q}, r={r}"
    assert all(abs(h_capped(q, 1.0) - q) < 1e-12 for q in (0.0, 0.3, 0.6, 0.9, 1.0)), "r=1 must equal stock"
    assert all(abs(h_capped(q, 0.0) - h_n(q, 0.5)) < 1e-12 for q in (0.1, 0.5, 0.9)), "r=0 must equal plain Cactus"


def test_cap_keeps_a_positive_escape():
    # In the loop regime the drafter is certain (p -> 1), and there the cap keeps exactly r of the stock
    # escape at every q < 1, while Cactus loses it completely above 1/(1+2*delta) = 0.5.
    for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
        for r in (0.25, 0.5, 0.75):
            assert abs(escape(q, 1.0, lambda x: h_capped(x, r)) - r * escape(q, 1.0, lambda x: x)) < 1e-12
            assert escape(q, 1.0, lambda x: h_capped(x, r)) > 0.0, f"escape fell to 0 at q={q}, r={r}"
    # Cactus itself has no escape there: h = 1 exactly.
    for q in (0.5, 0.8, 0.99):
        assert escape(q, 1.0, lambda x: h_n(x, 0.5)) == 0.0
    # monotone in r
    esc = [escape(0.9, 1.0, lambda x, r=r: h_capped(x, r)) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert esc == sorted(esc) and esc[0] == 0.0
    # For p < 1 the guarantee is weaker (h'/p can reach 1): the ratio is below r, never above.
    for q in (0.5, 0.9):
        for r in (0.25, 0.5, 0.75):
            got = escape(q, 0.98, lambda x: h_capped(x, r))
            want = r * escape(q, 0.98, lambda x: x)
            assert 0 < got <= want + 1e-12, f"cap escape exceeds r*stock at q={q}, r={r}: {got} > {want}"


def test_cap_default_is_half():
    src = PATCH.read_text()
    m = re.search(r'RADIANCE_CACTUS_CAP_R\\?",\s*\\?"([0-9.]+)', src)
    assert m and m.group(1) == "0.5", "RADIANCE_CACTUS_CAP_R default must be 0.5"


def test_fade_defaults():
    # Length fade (issue #164): delta falls from 0.5 at 4096 output tokens to 0 at 16384.
    src = PATCH.read_text()
    for knob, want in (("RADIANCE_CACTUS_FADE_START", "4096"), ("RADIANCE_CACTUS_FADE_END", "16384")):
        m = re.search(knob + r'\\?",\s*\\?"([0-9]+)', src)
        assert m and m.group(1) == want, f"{knob} default must be {want}"


def _pinned_source():
    # pytest passes its own argv (e.g. "sly/tests"); only a real file counts as the pinned source.
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        return Path(sys.argv[1])
    cand = ROOT / ".dryrun" / "src" / "vllm" / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "rejection_sampler_utils.py"
    return cand if cand.exists() else None


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print("SKIP", msg)


def test_fade_kernel_in_triton_interpreter():
    """The patched kernel in the Triton interpreter: fade vs a float64 reference, fade off, delta 0 == stock
    (sly/tests/cactus_fade_kernel_check.py)."""
    try:
        import triton  # noqa: F401
    except ImportError:
        return _skip("triton not installed (the interpreter check runs where it is)")
    src = _pinned_source()
    if src is None:
        return _skip("pinned vLLM source not available")
    with tempfile.TemporaryDirectory() as td:
        site = Path(td) / "site-packages"
        pkg = site / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"
        pkg.mkdir(parents=True)
        target = pkg / "rejection_sampler_utils.py"
        shutil.copy(src, target)
        # patch_cactus.py also patches the caller next to it (prompt lengths for the length fade)
        shutil.copy(Path(src).with_name("rejection_sampler.py"), target.with_name("rejection_sampler.py"))
        env = dict(os.environ, PYTHONPATH=str(ROOT), RADIANCE_PATCH_TARGET_DIR=str(site))
        assert subprocess.run([sys.executable, str(PATCH)], env=env).returncode == 0
        env["TRITON_INTERPRET"] = "1"
        r = subprocess.run([sys.executable, str(ROOT / "sly" / "tests" / "cactus_fade_kernel_check.py"),
                            str(src), str(target)], env=env, capture_output=True, text=True, timeout=1200)
        assert r.returncode == 0 and "cactus fade kernel check OK" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


def test_patch_structure():
    # pytest passes its own argv (e.g. "sly/tests"); only a real file counts as the pinned source.
    src = sys.argv[1] if len(sys.argv) > 1 and Path(sys.argv[1]).is_file() else None
    if src is None:
        cand = ROOT / ".dryrun" / "src" / "vllm" / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "rejection_sampler_utils.py"
        if not cand.exists():
            # The CI cpu-tests job has no pinned tree; ci/patch_dryrun.sh covers apply + idempotence.
            try:
                import pytest
                pytest.skip("pinned vLLM source not available (ci/patch_dryrun.sh covers it)")
            except ImportError:
                print("SKIP cactus patch structure: no pinned source (pass the file as argv[1])")
            return
        src = cand
    s = _run_patch_on(Path(src))
    ast.parse(s)  # the patched file must parse
    assert "RADIANCE_CACTUS_DELTA" in s, "env knob missing"
    assert "delta == 0.0" in s, "bit-exact delta=0 guard missing"
    # the Leviathan branch must compare against the Cactus threshold, not the raw target logprob
    assert "h_n" in s and "cactus_delta" in s, "transform / delta threading missing"
    assert s.count("RADIANCE_CACTUS_DELTA") == 1, "env read must happen once, module-level"
    assert '"RADIANCE_CACTUS_DELTA", "0.5"' in s, "patched file must default to 0.5"
    assert "cactus_prompt_len_ptr" in s and "req_delta" in s, "length fade missing"
    assert "cactus_cap_r" in s and "1.0 - cactus_cap_r * (1.0 - q_n)" in s, "acceptance cap missing"


if __name__ == "__main__":
    test_math_properties()
    test_acceptance_rate_increases()
    test_default_delta_is_half()
    test_cap_never_stricter_than_stock()
    test_cap_keeps_a_positive_escape()
    test_cap_default_is_half()
    test_fade_defaults()
    test_patch_structure()
    test_fade_kernel_in_triton_interpreter()
    print("cactus CPU tests OK")
