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


MR = "vllm/v1/worker/gpu/model_runner.py"


def _query_lens_fn(tree):
    """Extract the appended _rad_ik_query_lens helper from the patched model_runner."""
    src = (tree / MR).read_text()
    start = src.index("def _rad_ik_query_lens")
    ns = {}
    exec(compile(src[start:], "mr_tail", "exec"), ns)
    return ns["_rad_ik_query_lens"]


class _Cfg:
    scheduler_config = type("sc", (), {"max_num_seqs": 8})()

    def __init__(self, lens):
        self._lens = lens

    def get_decode_cudagraph_query_lens(self, q):
        return dict(self._lens)


def test_multiwidth_hook_lands_and_parses(tree):
    assert run(tree).returncode == 0
    m = (tree / MR).read_text()
    ast.parse(m)
    assert "decode_query_lens=_rad_ik_query_lens(" in m
    assert "def _rad_ik_query_lens" in m


def test_query_lens_adds_adaptive_widths(tree, monkeypatch):
    run(tree)
    fn = _query_lens_fn(tree)
    monkeypatch.setenv("RADIANCE_INKLING2_ADAPTIVE_K", "3,5,7")
    lens = fn(_Cfg({8: 8}), 8)
    assert set(lens) == {4, 6, 8}, lens  # k=3,5,7 -> q=4,6,8 FULL graphs
    # fixed single k adds only its width
    monkeypatch.setenv("RADIANCE_INKLING2_ADAPTIVE_K", "3")
    assert set(fn(_Cfg({8: 8}), 8)) == {4, 8}


def test_query_lens_inert_when_off(tree, monkeypatch):
    run(tree)
    fn = _query_lens_fn(tree)
    monkeypatch.delenv("RADIANCE_INKLING2_ADAPTIVE_K", raising=False)
    assert fn(_Cfg({8: 8}), 8) == {8: 8}  # unchanged -> only the config width captured


def test_idempotent_includes_model_runner(tree):
    run(tree)
    first = (tree / MR).read_text()
    r2 = run(tree)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert (tree / MR).read_text() == first, "second apply changed model_runner"
