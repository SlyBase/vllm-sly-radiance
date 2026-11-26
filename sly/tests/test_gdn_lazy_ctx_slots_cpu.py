"""patch_gdn_lazy.py: the lazy GDN glue can store its state on vLLM main's slotted MambaSpecDecodeGPUContext.

Found by the acceptance gate of PR #259 (run 45237859572, Qwen prod profile on ink-k8): vLLM main@1bdc437
declares `@dataclasses.dataclass(slots=True) class MambaSpecDecodeGPUContext`, so the hunk that does
`self._radiance_kv_cfg = kv_cache_config` in `initialize_from_forward_context` died at KV-cache init with
`AttributeError: 'MambaSpecDecodeGPUContext' object has no attribute '_radiance_kv_cfg'` (the hunk runs for
every start, RADIANCE_GDN_LAZY or not). The fix declares the five attributes as dataclass fields.

The patch runs on verbatim copies of the three upstream files it edits (sly/tests/fixtures/gdn_lazy_main1bdc437),
no vllm import. Plain python + torch, no GPU. Run: pytest sly/tests/test_gdn_lazy_ctx_slots_cpu.py -q
"""
import ast
import dataclasses
import os
import subprocess
import sys
import typing
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
PATCH = REPO / "patch_gdn_lazy.py"
FIX = REPO / "sly/tests/fixtures/gdn_lazy_main1bdc437"
FILES = ("v1/worker/mamba_utils.py", "model_executor/layers/mamba/abstract.py", "v1/attention/backends/gdn_attn.py")
MAMBA = FILES[0]
ATTRS = ("_radiance_kv_cfg", "_radiance_fwd_ctx", "_radiance_copy_funcs", "_radiance_lazy_temporal", "_radiance_lazy_tables")


@pytest.fixture()
def tree(tmp_path):
    for rel in FILES:
        f = tmp_path / "vllm" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text((FIX / (rel + ".txt")).read_text())
    return tmp_path


def run(tree):
    code = ("import runpy, sys, sysconfig; p = sysconfig.get_paths(); p['purelib'] = sys.argv[1];"
            "sysconfig.get_paths = lambda *a, **k: p; runpy.run_path(sys.argv[2], run_name='__main__')")
    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, "-c", code, str(tree), str(PATCH)], capture_output=True, text=True, env=env)


def load_ctx_class(path):
    """The MambaSpecDecodeGPUContext class statement of `path`, executed on its own."""
    src = Path(path).read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "MambaSpecDecodeGPUContext")
    lines = src.splitlines()[node.decorator_list[0].lineno - 1:node.end_lineno]
    stub = type("Stub", (), {})
    ns = {"dataclasses": dataclasses, "torch": torch, "Any": typing.Any, "ReplaySSMModelContext": stub,
          "KVCacheConfig": stub, "MambaStateCopyFuncsByType": stub}
    exec(compile("\n".join(lines), str(path), "exec"), ns)
    return ns["MambaSpecDecodeGPUContext"]


def make(cls):
    required = {f.name: None for f in dataclasses.fields(cls) if f.default is dataclasses.MISSING}
    return cls(**required)


def test_stock_context_rejects_the_lazy_attribute(tree):
    """The reported failure, on the unpatched upstream class."""
    ctx = make(load_ctx_class(tree / "vllm" / MAMBA))
    with pytest.raises(AttributeError, match="_radiance_kv_cfg"):
        ctx._radiance_kv_cfg = object()


def test_patched_context_holds_every_lazy_attribute(tree):
    r = run(tree)
    assert r.returncode == 0, r.stdout + r.stderr
    ctx = make(load_ctx_class(tree / "vllm" / MAMBA))
    assert ctx._radiance_lazy_temporal is False  # read by precopy/postprocess even before the hunk ran
    assert getattr(ctx, "_radiance_lazy_tables", "unset") is None  # radiance_gdn_lazy builds it on first use
    marks = {a: object() for a in ATTRS}
    for a, v in marks.items():
        setattr(ctx, a, v)
    assert all(getattr(ctx, a) is v for a, v in marks.items())


def test_second_apply_is_a_noop(tree):
    assert run(tree).returncode == 0
    once = (tree / "vllm" / MAMBA).read_text()
    second = run(tree)
    assert second.returncode == 0 and "FAIL" not in second.stdout
    assert (tree / "vllm" / MAMBA).read_text() == once


@pytest.mark.parametrize("rel", FILES)
def test_fixture_is_the_upstream_file(rel):
    src = os.environ.get("INKLING_VLLM_SRC")
    real = Path(src, rel) if src else None
    if not real or not real.exists():
        pytest.skip("INKLING_VLLM_SRC not set")
    assert real.read_text() == (FIX / (rel + ".txt")).read_text()
