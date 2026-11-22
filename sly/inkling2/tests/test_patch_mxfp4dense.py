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


def method(source, cls, name):
    """Source of `cls.name` in `source` (None if absent)."""
    for c in ast.walk(ast.parse(source)):
        if isinstance(c, ast.ClassDef) and c.name == cls:
            for f in c.body:
                if isinstance(f, ast.FunctionDef) and f.name == name:
                    return ast.get_source_segment(source, f)
    return None


def test_load_time_hooks_land_in_the_right_classes(tree):
    run(GFX12, tree)
    assert run(PATCH, tree).returncode == 0
    ink = tree / "vllm/models/inkling2/nvidia"
    target = method((ink / "model.py").read_text(), "Inkling2ForConditionalGeneration",
                    "process_weights_after_loading")
    assert target and "process_target(self)" in target and "torch.version.hip" in target
    dspark = (ink / "dspark.py").read_text()
    draft = method(dspark, "Inkling2DSparkModel", "process_weights_after_loading")
    assert draft and "process_draft(self)" in draft and "torch.version.hip" in draft
    attach = method(dspark, "Inkling2DSparkModel", "attach_target")
    # the head is released only after the fold has read it in bf16
    assert attach.index("self.refresh_derived_weights()") < attach.index("release_head(self.lm_head.weight)")


def test_idempotent(tree):
    run(GFX12, tree)
    assert run(PATCH, tree).returncode == 0
    files = ("ops/linear.py", "model.py", "dspark.py")
    first = {f: (tree / "vllm/models/inkling2/nvidia" / f).read_text() for f in files}
    r2 = run(PATCH, tree)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert "NOOP" in r2.stdout and "  OK  " not in r2.stdout
    for f in files:
        assert (tree / "vllm/models/inkling2/nvidia" / f).read_text() == first[f]


def test_noop_without_inkling2(tmp_path):
    (tmp_path / "vllm").mkdir()
    r = run(PATCH, tmp_path)
    assert r.returncode == 0
    assert "NOOP" in r.stdout