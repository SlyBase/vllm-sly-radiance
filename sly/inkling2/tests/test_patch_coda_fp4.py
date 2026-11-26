"""patch_inkling2_coda_fp4.py: lands, parses, idempotent, inert; the hooks do what they say (CPU)."""
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
PATCH = REPO / "sly" / "patch_inkling2_coda_fp4.py"
sys.path.insert(0, str(REPO / "sly" / "inkling2"))
SRC = Path(os.environ.get("INKLING_VLLM_SRC",
                          "/opt/data/profiles/daidalos/cache/scratch/ink-k9-src/vllm"))
NV = SRC / "models/inkling2/nvidia"
pytestmark = pytest.mark.skipif(not (NV / "moe.py").exists(), reason="no vLLM source tree")


@pytest.fixture()
def tree(tmp_path):
    d = tmp_path / "vllm/models/inkling2/nvidia"
    d.mkdir(parents=True)
    for f in ("moe.py", "model.py"):
        shutil.copyfile(NV / f, d / f)
    return tmp_path


def run(target):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(target), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(PATCH)], capture_output=True, text=True, env=env, cwd=REPO)


def test_lands_and_parses(tree):
    r = run(tree)
    assert r.returncode == 0, r.stdout + r.stderr
    for f in ("moe.py", "model.py"):
        ast.parse((tree / "vllm/models/inkling2/nvidia" / f).read_text())
    assert "radiance_inkling2_coda_fp4" in (tree / "vllm/models/inkling2/nvidia/moe.py").read_text()
    assert "radiance_inkling2_coda_fp4" in (tree / "vllm/models/inkling2/nvidia/model.py").read_text()


def test_idempotent(tree):
    run(tree)
    before = {f: (tree / "vllm/models/inkling2/nvidia" / f).read_bytes() for f in ("moe.py", "model.py")}
    r = run(tree)
    assert r.returncode == 0 and "NOOP" in r.stdout
    assert before == {f: (tree / "vllm/models/inkling2/nvidia" / f).read_bytes() for f in ("moe.py", "model.py")}


def test_noop_without_inkling2(tmp_path):
    r = run(tmp_path)
    assert r.returncode == 0 and "NOOP" in r.stdout


def _hook(tree):
    """The patched Inkling2Experts.process_weights_after_loading, exec'd against a stub self."""
    run(tree)
    mod = ast.parse((tree / "vllm/models/inkling2/nvidia/moe.py").read_text())
    fn = next(n for c in mod.body if isinstance(c, ast.ClassDef) and c.name == "Inkling2Experts"
              for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "process_weights_after_loading")
    ns = {}
    exec(compile(ast.Module([fn], []), "patched", "exec"), ns)
    return ns["process_weights_after_loading"]


class _Stub(torch.nn.Module):
    def __init__(self, flag):
        super().__init__()
        self.nvfp4, self.sm100, self.sm120, self.humming_nvfp4 = False, False, True, False
        if flag is not None:
            self.rad_requant = flag
        for name, (n, k) in (("w13", (64, 64)), ("w2", (64, 32))):
            setattr(self, f"{name}_weight", torch.nn.Parameter(torch.zeros(2, n, k, dtype=torch.float8_e4m3fn), requires_grad=False))
            setattr(self, f"{name}_weight_scale", torch.nn.Parameter(torch.full((2, n, k // 32), 120, dtype=torch.uint8), requires_grad=False))


def test_hook_converts_only_flagged_experts(tree):
    hook = _hook(tree)
    on, off, plain = _Stub(True), _Stub(False), _Stub(None)
    for m in (on, off, plain):
        hook(m)
    assert on.nvfp4 and on.w13_weight.dtype == torch.uint8
    assert not off.nvfp4 and off.w13_weight.dtype == torch.float8_e4m3fn
    assert not plain.nvfp4


def test_model_hook_marks_mxfp8_layers_only(tree):
    run(tree)
    s = (tree / "vllm/models/inkling2/nvidia/model.py").read_text()
    i = s.index("rad_requant")
    ctx = s[i - 400:i + 200]
    assert 'expert_format == "mxfp8"' in ctx and "wants(layer_id)" in ctx
