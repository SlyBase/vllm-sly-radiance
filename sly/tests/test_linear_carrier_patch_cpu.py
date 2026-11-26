"""sly/patch_linear_carrier.py: the fused-norm (q, scale) carrier survives vLLM main's apply_flattened.

Found by the acceptance gate of PR #259 (run 45225663768, Qwen prod profile on ink-k7): vLLM main@1bdc437
routes every linear layer through `apply_flattened(fn, x)`, which reads `x.ndim` before it calls the
quant method. The RADIANCE_FUSED_NORM_QUANT hooks hand the W4A8 GEMMs a `(q, scale)` tuple (the
activation already quantized), so profile_run died in `Qwen3MoeMLP.forward -> down_proj` with
`AttributeError: 'tuple' object has no attribute 'ndim'`. vLLM 0.34 called `quant_method.apply`
directly, so no CPU dry run (anchors only) could see it.

The patch runs on a verbatim fixture of the upstream function (sly/tests/fixtures), no vllm import.
Plain python, no GPU. Run: pytest sly/tests/test_linear_carrier_patch_cpu.py -q
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PATCH = REPO / "sly" / "patch_linear_carrier.py"
FIXTURE = REPO / "sly/tests/fixtures/linear_apply_flattened_main1bdc437.py.txt"
LINEAR = "vllm/model_executor/layers/linear.py"
QA_IMPORT = "from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation\n"


class _QA:  # QuantizedActivation stand-in: only isinstance() and orig_shape are read
    def __init__(self, orig_shape):
        self.orig_shape = orig_shape


@pytest.fixture()
def tree(tmp_path):
    f = tmp_path / LINEAR
    f.parent.mkdir(parents=True)
    f.write_text(FIXTURE.read_text())
    return tmp_path


def run(tree):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(tree), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(PATCH)], capture_output=True, text=True, env=env)


def load_apply_flattened(path):
    ns = {"QuantizedActivation": _QA}
    exec(compile(path.read_text().replace(QA_IMPORT, ""), str(path), "exec"), ns)
    return ns["apply_flattened"]


def test_stock_apply_flattened_rejects_the_carrier(tree):
    """The reported failure, on the unpatched upstream function."""
    stock = load_apply_flattened(tree / LINEAR)
    with pytest.raises(AttributeError, match="'tuple' object has no attribute 'ndim'"):
        stock(lambda x: x, (torch.zeros(4, 8), torch.ones(4, 1)))


def test_patched_apply_flattened_hands_the_carrier_to_the_kernel(tree):
    assert run(tree).returncode == 0
    patched = load_apply_flattened(tree / LINEAR)
    carrier = (torch.zeros(4, 8), torch.ones(4, 1))
    seen = []
    assert patched(lambda x: seen.append(x) or "out", carrier) == "out"
    assert seen[0] is carrier  # the very tuple, untouched


def test_tensor_and_quantized_activation_paths_are_unchanged(tree):
    stock = load_apply_flattened(tree / LINEAR)
    assert run(tree).returncode == 0
    patched = load_apply_flattened(tree / LINEAR)
    x3 = torch.arange(24.0).reshape(2, 3, 4)
    for fn_in in (x3, x3.transpose(1, 2), torch.zeros(5, 4), torch.zeros(4), _QA((2, 3, 4)), _QA((5, 4))):
        calls = {}
        for name, f in (("stock", stock), ("patched", patched)):
            seen = []

            def kernel(x, seen=seen):
                seen.append(tuple(x.shape) if isinstance(x, torch.Tensor) else "qa")
                return torch.zeros(*(x.shape[:-1] if isinstance(x, torch.Tensor) else (7,)), 3)

            try:
                out = f(kernel, fn_in)
            except Exception as e:  # the stand-in QuantizedActivation has no flatten(): both must agree
                out = type(e).__name__
            calls[name] = (seen, out if isinstance(out, str) else tuple(out.shape))
        assert calls["stock"] == calls["patched"]


def test_second_apply_is_a_noop(tree):
    assert run(tree).returncode == 0
    once = (tree / LINEAR).read_text()
    second = run(tree)
    assert second.returncode == 0 and "NOOP" in second.stdout
    assert (tree / LINEAR).read_text() == once


def test_tree_without_apply_flattened_is_skipped(tmp_path):
    """vLLM 0.34 calls quant_method.apply directly: nothing to patch, not a failure."""
    f = tmp_path / LINEAR
    f.parent.mkdir(parents=True)
    f.write_text("class RowParallelLinear:\n    pass\n")
    r = run(tmp_path)
    assert r.returncode == 0 and "SKIP" in r.stdout
    assert f.read_text() == "class RowParallelLinear:\n    pass\n"


def test_fixture_is_the_upstream_function():
    src = os.environ.get("INKLING_VLLM_SRC")
    real = Path(src, "model_executor/layers/linear.py") if src else None
    if not real or not real.exists():
        pytest.skip("INKLING_VLLM_SRC not set")
    text = real.read_text()
    fn = text[text.index("def apply_flattened("):text.index("\n\nclass LinearBase")].rstrip() + "\n"
    assert fn in FIXTURE.read_text()
