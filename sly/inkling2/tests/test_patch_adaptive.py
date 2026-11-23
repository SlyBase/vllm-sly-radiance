"""sly/patch_inkling2_adaptive.py: adaptive DSpark verify window (Inkling-2-Mini), by source patch.

Runs the real patch script against a COPY of the vLLM source tree (INKLING_VLLM_SRC) through
RADIANCE_PATCH_TARGET_DIR, so the anchors are checked against the real upstream text. Checks the
scheduler hooks land, the files still parse, and a second apply is a NOOP (idempotency).
"""
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
PATCH = REPO / "sly" / "patch_inkling2_adaptive.py"
SCHED = "vllm/v1/core/sched/scheduler.py"


@pytest.fixture()
def tree(tmp_path):
    src = os.environ.get("INKLING_VLLM_SRC")
    if not src or not Path(src, "v1/core/sched/scheduler.py").exists():
        pytest.skip("INKLING_VLLM_SRC not set")
    shutil.copytree(src, tmp_path / "vllm", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def run(tree):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(tree), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(PATCH)], capture_output=True, text=True, env=env)


def sched(tree):
    return (tree / SCHED).read_text()


def test_hooks_land_and_parse(tree):
    assert run(tree).returncode == 0
    s = sched(tree)
    ast.parse(s)
    # observation of (drafted, accepted) per spec step
    assert "self._rad_ik_observe(num_draft_tokens, num_accepted)" in s
    # the per-step k override sits after the native dynamic_sd_lookup block
    assert "num_spec_tokens_to_schedule = self._rad_ik_k(" in s
    # the policy module is imported lazily (so the file parses with no module present)
    assert "radiance_inkling2_adaptive_k" in s


def test_k_override_is_after_dynamic_lookup(tree):
    run(tree)
    s = sched(tree)
    # the override must run after the native lookup so it can win
    assert s.index("num_spec_tokens_to_schedule = self._rad_ik_k(") > s.index(
        "self.dynamic_sd_lookup"
    )


def test_observe_is_inside_the_spec_block(tree):
    """The observe call reads num_draft_tokens / num_accepted, which only exist in the spec
    (drafted-token) branch; it must land there, not at module scope."""
    run(tree)
    s = sched(tree)
    obs = s.index("self._rad_ik_observe(num_draft_tokens, num_accepted)")
    # after num_accepted is computed in the same branch
    assert s.index("num_accepted = max(len(generated_token_ids)") < obs


def test_idempotent(tree):
    run(tree)
    first = sched(tree)
    r2 = run(tree)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert sched(tree) == first, "second apply changed the file"
    assert "NOOP" in r2.stdout


def test_off_by_default_is_a_noop_on_k(tree):
    """With the knob unset, the k override returns its argument unchanged (policy is inert)."""
    run(tree)
    # import the policy straight from the module to prove the inert contract
    policy = REPO / "sly" / "inkling2" / "radiance_inkling2_adaptive_k.py"
    code = policy.read_text()
    ns = {}
    exec(compile(code, str(policy), "exec"), ns)
    assert ns["state_from_env"]() is None  # unset env -> off -> no k change
