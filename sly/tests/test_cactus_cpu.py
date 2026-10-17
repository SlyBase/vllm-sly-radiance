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


if __name__ == "__main__":
    test_math_properties()
    test_acceptance_rate_increases()
    test_default_delta_is_half()
    test_patch_structure()
    print("cactus CPU tests OK")
