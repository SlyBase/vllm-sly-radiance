"""sly/patch_inkling2_decode_attn.py: long-context decode attention of Inkling 2 on gfx12, by source patch.

Runs the real patch script (after patch_inkling2_gfx12, as in the Dockerfile) against a COPY of a vLLM source
tree (INKLING_VLLM_SRC) through RADIANCE_PATCH_TARGET_DIR: anchors are checked against the real upstream text.
The numerics and the speed are checked on the GPU by `sly/inkling2/bench_decode_attn.py`.
"""
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
OPS = "vllm/models/inkling2/nvidia/ops"


@pytest.fixture()
def tree(tmp_path):
    src = os.environ.get("INKLING_VLLM_SRC")
    if not src or not Path(src, "models/inkling2").exists():
        pytest.skip("INKLING_VLLM_SRC not set")
    shutil.copytree(src, tmp_path / "vllm", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def run(target, name):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(target), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(REPO / "sly" / name)], capture_output=True, text=True, env=env)


def patched(tree):
    assert run(tree, "patch_inkling2_gfx12.py").returncode == 0
    r = run(tree, "patch_inkling2_decode_attn.py")
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_hunks_land_and_parse(tree):
    patched(tree)
    plan = (tree / OPS / "attention.py").read_text()
    kern = (tree / OPS / "attention_sm90.py").read_text()
    ast.parse(plan)
    ast.parse(kern)
    assert "_ROCM_CTAS_PER_SM * num_sms // work" in plan
    assert "BULK_ON: tl.constexpr" in kern
    assert "for i in range(num_bulk, num_iters):" in kern


def test_prefill_launch_keeps_the_stock_tile(tree):
    patched(tree)
    kern = (tree / OPS / "attention_sm90.py").read_text()
    # the new tile and the bulk flag are decode-only: prefill (MODE_OUT_LSE) must keep BLOCK_N and no bulk loop
    assert "mode != MODE_OUT_LSE and window == 0 and block_m <= 32" in kern
    assert "BULK=_BULK and mode != MODE_OUT_LSE" in kern


def test_second_run_is_a_noop(tree):
    patched(tree)
    before = {p: p.read_text() for p in (tree / OPS).glob("attention*.py")}
    r = run(tree, "patch_inkling2_decode_attn.py")
    assert r.returncode == 0 and "OK " not in r.stdout.replace("NOOP", "")
    assert before == {p: p.read_text() for p in (tree / OPS).glob("attention*.py")}


def test_absent_model_is_a_noop(tmp_path):
    r = run(tmp_path, "patch_inkling2_decode_attn.py")
    assert r.returncode == 0 and "NOOP" in r.stdout


# ---- prefill launch (sly/patch_inkling2_prefill_attn.py, stacked on the decode patch) ----------------------------


def prefill_patched(tree):
    patched(tree)
    r = run(tree, "patch_inkling2_prefill_attn.py")
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_prefill_hunks_land_and_parse(tree):
    prefill_patched(tree)
    kern = (tree / OPS / "attention_sm90.py").read_text()
    ast.parse(kern)
    assert "RADIANCE_INKLING2_PREFILL_BULK" in kern
    assert "RADIANCE_INKLING2_PREFILL_BLOCK_N" in kern
    assert "RADIANCE_INKLING2_PREFILL_LAZY" in kern and "            if LAZY:" in kern
    # prefill (MODE_OUT_LSE) now runs the unmasked bulk loop, gated by both knobs; decode gating is unchanged
    assert "BULK=_BULK and (_PREFILL_BULK or mode != MODE_OUT_LSE)" in kern
    assert "mode != MODE_OUT_LSE and window == 0 and block_m <= 32" in kern


def test_prefill_second_run_is_a_noop(tree):
    prefill_patched(tree)
    before = {p: p.read_text() for p in (tree / OPS).glob("attention*.py")}
    r = run(tree, "patch_inkling2_prefill_attn.py")
    assert r.returncode == 0 and "OK " not in r.stdout.replace("NOOP", "")
    assert before == {p: p.read_text() for p in (tree / OPS).glob("attention*.py")}


def test_prefill_absent_model_is_a_noop(tmp_path):
    r = run(tmp_path, "patch_inkling2_prefill_attn.py")
    assert r.returncode == 0 and "NOOP" in r.stdout
