#!/usr/bin/env python3
"""Tests for the Cactus escape mixture (sly/patch_cactus.py, issue #164). CPU only, no GPU/vLLM needed
for the math and patch-wiring parts; the Triton-interpreter check runs when `triton` and a pinned
vLLM source are available.

    python3 sly/tests/test_cactus_escape_cpu.py          # direct
    pytest sly/tests/test_cactus_escape_cpu.py -v        # or via pytest

Why the mixture: threshold-only Cactus (any delta > 0) accepts a confident draft of a confident
target token with probability 1 (h >= 1 for q >= 1/(1+2*delta)), so a repetition loop becomes
absorbing; the acceptance cap only restores escape while the drafter is unsure (p < 1). Mixing the
exact stock test in with probability eps makes acceptance (1-eps)*min(1,h/p) + eps*min(1,q/p), hence
out = (1-eps)*out_cactus + eps*q and escape >= eps*stock at EVERY (q,p).
"""
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "sly" / "patch_cactus.py"
CHECK = Path(__file__).resolve().parent / "cactus_escape_kernel_check.py"
PINNED_REL = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
CALLER_REL = "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"

# The fade-era (1.9.0-1.11.0) acceptance tail: shape 3 of the threshold hunk must upgrade it.
_FADE_TAIL = (
    "                    log_h_n = tl.where(req_delta == 0.0, target_logprob, tl.log(h_n))\n"
    "                    accepted = log_h_n > tl.log(u) + draft_logprob\n"
)


def pinned_source():
    base = os.environ.get("RADIANCE_TEST_VLLM_SRC")
    if base:
        p = Path(base) / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "rejection_sampler_utils.py"
        return p if p.exists() else None
    for site in sys.path:
        p = Path(site or ".") / PINNED_REL
        if p.exists():
            return p
    return None


class _Skip(Exception):
    pass


def _skip(reason):
    try:
        import pytest
    except ImportError:
        raise _Skip(reason)
    pytest.skip(reason, allow_module_level=True)


# --------------------------------------------------------------------------- math (pure python)
def h_of(q, delta, cap_r=0.0):
    """The shipped threshold: gamma*(delta) clipped, then the cap (fade off)."""
    h = min(max(q + math.sqrt(2.0 * delta * q * (1.0 - q)), 0.0), 1.0)
    return min(h, 1.0 - cap_r * (1.0 - q))


def acceptance(q, p, delta=0.5, cap_r=0.0, eps=0.0):
    h = h_of(q, delta, cap_r)
    a_h, a_q = min(1.0, h / p), min(1.0, q / p)
    return (1.0 - eps) * a_h + eps * a_q


def stock_residual(q, p):
    return [max(q[i] - p[i], 0.0) for i in range(len(q))]


def output_dist(q, p, delta=0.5, cap_r=0.0, eps=0.0):
    """One-step output law of the kernel: accept -> token, reject -> stock residual (renormalised)."""
    out = [0.0] * len(q)
    r = stock_residual(q, p)
    s = sum(r)
    for n in range(len(q)):
        h = h_of(q[n], delta, cap_r)
        a = (1.0 - eps) * min(1.0, h / p[n]) + eps * min(1.0, q[n] / p[n])
        out[n] += p[n] * a
        rej = p[n] * (1.0 - a)
        if rej > 0 and s > 1e-12:
            for i in range(len(q)):
                out[i] += rej * r[i] / s
    total = sum(out)
    return [x / total for x in out]


def tv(a, b):
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b))


def rand_dist(rng, v):
    w = [rng.random() + 1e-6 for _ in range(v)]
    s = sum(w)
    return [x / s for x in w]


def test_acceptance_is_convex_mix():
    for q in (0.05, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99, 0.999):
        for p in (0.2, 0.5, 0.8, 0.95, 1.0):
            for eps in (0.0, 0.25, 0.5, 1.0):
                a_h = min(1.0, h_of(q, 0.5) / p)
                a_q = min(1.0, q / p)
                a = acceptance(q, p, eps=eps)
                assert abs(a - ((1 - eps) * a_h + eps * a_q)) < 1e-15
                assert a_q - 1e-15 <= a <= a_h + 1e-15


def test_escape_floor_everywhere():
    """escape = P(reject the loop token) = 1 - a_mix >= eps * stock escape, at every (q, p)."""
    for q in (0.2, 0.5, 0.7, 0.9, 0.99, 0.999):
        for p in (0.3, 0.6, 0.9, 0.99, 1.0):
            esc_stock = 1.0 - min(1.0, q / p)
            for eps in (0.1, 0.25, 0.5):
                esc_mix = 1.0 - acceptance(q, p, eps=eps)
                assert esc_mix >= eps * esc_stock - 1e-12, (q, p, eps)
                if q < p:
                    assert esc_mix > 0.0  # wherever stock can escape, the mixture can too
                else:
                    assert esc_mix == 0.0  # q >= p: stock itself always accepts, nothing to escape
    # the corner the cap cannot reach: confident drafter, confident target, h still above p
    assert acceptance(0.9, 1.0, eps=0.0) == 1.0          # cactus: absorbing
    assert acceptance(0.9, 1.0, cap_r=0.5, eps=0.0) < 1.0  # cap helps here (p=1) ...
    assert acceptance(0.98, 0.99, cap_r=0.5, eps=0.0) == 1.0  # ... but not here: h=0.99 >= p=0.99
    assert acceptance(0.98, 0.99, cap_r=0.5, eps=0.25) < 1.0  # the mixture does


def test_output_distribution_identity():
    """out_mix == (1-eps)*out_cactus + eps*q exactly (V=4 enumeration, random pairs)."""
    rng = random.Random(164)
    v = 4
    worst = 0.0
    for _ in range(300):
        q, p = rand_dist(rng, v), rand_dist(rng, v)
        q[v - 1] += 0.5
        s = sum(q)
        q = [x / s for x in q]
        base = output_dist(q, p, eps=0.0)
        for eps in (0.25, 0.5, 1.0):
            mix = output_dist(q, p, eps=eps)
            want = [(1 - eps) * b + eps * qi for b, qi in zip(base, q)]
            worst = max(worst, tv(mix, want))
        assert tv(output_dist(q, p, delta=0.0), q) < 1e-12  # delta=0 stays exact
    assert worst < 1e-12, worst


def test_zero_escape_threshold():
    """q >= 1/(1+2*delta) => h >= 1 => zero escape at p=1 without the mixture."""
    for delta in (0.1, 0.25, 0.5):
        qstar = 1.0 / (1.0 + 2.0 * delta)
        assert abs(h_of(qstar, delta) - 1.0) < 1e-12
        assert acceptance(qstar + 1e-6, 1.0, delta=delta) == 1.0
        assert acceptance(0.999, 1.0, delta=delta, eps=0.25) <= 1.0 - 0.25 * (1.0 - 0.999) + 1e-12
        assert acceptance(0.999, 1.0, delta=delta, eps=0.25) < 1.0


# --------------------------------------------------------------------------- patch wiring
def _run_patch_on(tmp):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(tmp), PYTHONPATH=str(ROOT))
    r = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def _patched_tree():
    src = pinned_source()
    if src is None:
        return None, None
    tmp = Path(tempfile.mkdtemp(prefix="cactus_escape_test_"))
    dst = tmp / PINNED_REL
    dst.parent.mkdir(parents=True)
    dst.write_text(src.read_text())
    caller = tmp / CALLER_REL
    caller.write_text((src.parent / "rejection_sampler.py").read_text())
    return tmp, dst


def test_patch_wiring():
    tmp, dst = _patched_tree()
    if tmp is None:
        _skip("no pinned vLLM source (install vllm or set RADIANCE_TEST_VLLM_SRC)")
    _run_patch_on(tmp)
    t = dst.read_text()
    assert t.count('CACTUS_ESCAPE = float(os.environ.get("RADIANCE_CACTUS_ESCAPE", "0"))') == 1
    assert 'RADIANCE_CACTUS_ESCAPE must be in [0, 1]' in t
    assert "    cactus_escape,  # radiance cactus escape mixture: kernel arg" in t
    assert "        CACTUS_ESCAPE,  # radiance cactus escape mixture: launch arg" in t
    assert "cactus_escape: float = CACTUS_ESCAPE," in t
    assert "acc_mix = u < (1.0 - cactus_escape) * a_h + cactus_escape * a_q" in t
    assert "acc_stock = target_logprob > tl.log(u) + draft_logprob" in t
    assert "tl.where(cactus_escape == 1.0, acc_stock," in t
    # the delta=0 bit-exact guard survives
    assert "log_h_n = tl.where(req_delta == 0.0, target_logprob, tl.log(h_n))" in t
    assert "tl.where((cactus_escape > 0.0) & (req_delta > 0.0)," in t


def test_patch_idempotent():
    tmp, dst = _patched_tree()
    if tmp is None:
        _skip("no pinned vLLM source")
    _run_patch_on(tmp)
    once = dst.read_text()
    out = _run_patch_on(tmp)
    assert dst.read_text() == once
    assert "NOOP" in out


def test_upgrade_from_fade_era():
    """Shape 3: a 1.9.0-1.11.0 file (fade tail, no mixture) upgrades to the same bytes as a fresh apply."""
    tmp, dst = _patched_tree()
    if tmp is None:
        _skip("no pinned vLLM source")
    _run_patch_on(tmp)
    fresh = dst.read_text()
    # revert ONLY the acceptance tail to the fade-era form (the rest of the wiring stays)
    start = fresh.index("                    log_h_n = tl.where(req_delta == 0.0")
    end = fresh.index("\n", fresh.index("acc_mix, acc_cactus))", start)) + 1
    old = fresh[:start] + _FADE_TAIL + fresh[end:]
    assert "# radiance cactus escape mixture: acceptance" not in old
    dst.write_text(old)
    out = _run_patch_on(tmp)
    assert "shape 3" in out, out
    assert dst.read_text() == fresh


# --------------------------------------------------------------------------- interpreter launcher
def test_patched_kernel_escape_interpreter():
    src = pinned_source()
    if src is None:
        _skip("no pinned vLLM source (install vllm or set RADIANCE_TEST_VLLM_SRC)")
    try:
        import triton  # noqa: F401
    except ImportError:
        _skip("triton not installed (CPU interpreter mode needs it)")
    with tempfile.TemporaryDirectory(prefix="cactus_escape_") as tmp:
        tmp = Path(tmp)
        tree = tmp / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"
        tree.mkdir(parents=True)
        patched = tree / "rejection_sampler_utils.py"
        patched.write_text(src.read_text())
        caller = tree / "rejection_sampler.py"
        caller.write_text((src.parent / "rejection_sampler.py").read_text())
        env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(tmp), PYTHONPATH=str(ROOT))
        r = subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        env = dict(os.environ, TRITON_INTERPRET="1", PYTHONUNBUFFERED="1")
        r = subprocess.run([sys.executable, str(CHECK), str(src), str(patched)],
                           env=env, capture_output=True, text=True, timeout=1800)
        print(r.stdout)
        assert r.returncode == 0, r.stdout + r.stderr
        assert "cactus escape kernel check OK" in r.stdout


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except _Skip as e:
                print(f"SKIP {name}: {e}")
            except Exception as e:  # noqa: BLE001 - report and continue
                fails += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if fails else 0)
