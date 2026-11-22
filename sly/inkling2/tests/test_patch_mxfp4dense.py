"""sly/patch_inkling2_mxfp4dense.py: opt-in MXFP4 dense routing hooks, by source patch.

Applies the gfx12 patch first (the image ships gfx12-patched model files; this patch anchors on that
base), then the MXFP4-dense patch, against a COPY of the vLLM source tree. Checks the hooks land,
the files still parse, and a second apply is a NOOP (idempotency).
"""
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GFX12 = REPO / "sly" / "patch_inkling2_gfx12.py"
PATCH = REPO / "sly" / "patch_inkling2_mxfp4dense.py"


@pytest.fixture()
def tree(tmp_path):
    src = os.environ.get("INKLING_VLLM_SRC")
    if not src or not Path(src, "models/inkling2").exists():
        pytest.skip("INKLING_VLLM_SRC not set")
    shutil.copytree(src, tmp_path / "vllm", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def run(script, target):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(target), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env)


def test_hooks_land_and_parse(tree):
    assert run(GFX12, tree).returncode == 0
    r = run(PATCH, tree)
    assert r.returncode == 0, r.stdout + r.stderr
    linear = (tree / "vllm/models/inkling2/nvidia/ops/linear.py").read_text()
    model = (tree / "vllm/models/inkling2/nvidia/model.py").read_text()
    assert "mxfp4_dense_linear" in linear
    assert "mxfp4_dense_head" in model
    ast.parse(linear)
    ast.parse(model)
    # the hook runs before the bf16 dispatch in linear()
    assert linear.index("mxfp4_dense_linear") < linear.index("if out is not None:")


def test_idempotent(tree):
    run(GFX12, tree)
    assert run(PATCH, tree).returncode == 0
    linear1 = (tree / "vllm/models/inkling2/nvidia/ops/linear.py").read_text()
    model1 = (tree / "vllm/models/inkling2/nvidia/model.py").read_text()
    r2 = run(PATCH, tree)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert "NOOP" in r2.stdout
    assert (tree / "vllm/models/inkling2/nvidia/ops/linear.py").read_text() == linear1
    assert (tree / "vllm/models/inkling2/nvidia/model.py").read_text() == model1


def test_noop_without_inkling2(tmp_path):
    (tmp_path / "vllm").mkdir()
    r = run(PATCH, tmp_path)
    assert r.returncode == 0
    assert "NOOP" in r.stdout