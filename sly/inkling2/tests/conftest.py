"""Test harness for the Inkling 2 gfx12 kernels: CPU only, Triton interpreter.

The kernels import vLLM's own `_scale_offsets` (the layout truth).  vLLM's package `__init__` pulls
the whole engine, so the fixture below builds bare namespace packages that point at a vLLM *source
tree* (env INKLING_VLLM_SRC, the `vllm/` directory of a checkout that has `models/inkling2`) and a
`vllm.triton_utils` shim over the real `triton`.  Everything is removed again after the session.
Tests are skipped when the tree or triton is missing.
"""
import importlib
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")  # before triton is imported anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the module under test


@pytest.fixture(scope="session", autouse=True)
def vllm_shim():
    src = os.environ.get("INKLING_VLLM_SRC")
    triton = pytest.importorskip("triton")
    if not src or not Path(src, "models/inkling2/nvidia/ops/block_scales.py").exists():
        pytest.skip("INKLING_VLLM_SRC (vllm/ source dir with models/inkling2) not set")
    import triton.language as tl

    made = []

    def pkg(name, path):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m
        made.append(name)

    root = Path(src)
    pkg("vllm", root)
    pkg("vllm.models", root / "models")
    pkg("vllm.models.inkling2", root / "models/inkling2")
    pkg("vllm.models.inkling2.nvidia", root / "models/inkling2/nvidia")
    pkg("vllm.models.inkling2.nvidia.ops", root / "models/inkling2/nvidia/ops")
    shim = types.ModuleType("vllm.triton_utils")
    shim.tl, shim.triton = tl, triton
    sys.modules["vllm.triton_utils"] = shim
    made.append("vllm.triton_utils")
    yield
    for n in made:
        sys.modules.pop(n, None)
    for n in [k for k in sys.modules if k.startswith("radiance_inkling2")]:
        sys.modules.pop(n, None)
