"""Test harness for the Inkling 2 gfx12 kernels: CPU only, Triton interpreter.

The kernels import vLLM's own `_scale_offsets` (the layout truth).  vLLM's package `__init__` pulls
the whole engine, so the fixture below builds bare namespace packages that point at a vLLM *source
tree* (env INKLING_VLLM_SRC, the `vllm/` directory of a checkout that has `models/inkling2`) and a
`vllm.triton_utils` shim over the real `triton`.  Everything is removed again after the session.
Tests are skipped when the tree or triton is missing.
"""
import importlib
import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("TRITON_INTERPRET", "1")  # before triton is imported anywhere
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the module under test


# Test modules that import triton (directly, or through the bench module they test) at module level:
# without triton they cannot even be collected. The CPU job (ci.yml `inkling2-tests`) has no triton, so
# they are left out of the run and named in the header; with triton (GPU host / Triton interpreter) they run.
NEEDS_TRITON = {"test_bench_prefill_moe.py", "test_nvfp4_expert_gemm_v2.py"}
_HAS_TRITON = importlib.util.find_spec("triton") is not None


# test_patch_loopcd.py reads vllm/models/inkling2/nvidia/loop.py from a developer scratch tree (LoopCD is not
# in the pinned vLLM); on any other machine it cannot run, so it is left out there.
_LOOP_SRC = Path("/opt/data/profiles/daidalos/cache/scratch/ink-k9-src/vllm/models/inkling2/nvidia/loop.py")
NEEDS_LOCAL_TREE = {"test_patch_loopcd.py"}


def _left_out():
    out = set() if _HAS_TRITON else set(NEEDS_TRITON)
    return out if _LOOP_SRC.exists() else out | NEEDS_LOCAL_TREE


def pytest_ignore_collect(collection_path, config):
    return collection_path.name in _left_out()


def pytest_report_header(config):
    left = sorted(_left_out())
    return f"not collected here (triton / dev tree missing): {', '.join(left)}" if left else None


@pytest.fixture(scope="session", autouse=True)
def vllm_shim():
    src = os.environ.get("INKLING_VLLM_SRC")
    if src and Path(src, "models/inkling2/nvidia/ops/block_scales.py").exists():
        triton = pytest.importorskip("triton")
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
    else:
        made = []
    yield
    for n in made:
        sys.modules.pop(n, None)
    for n in [k for k in sys.modules if k.startswith("radiance_inkling2")]:
        sys.modules.pop(n, None)
