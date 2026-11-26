"""sly/patch_qwen35_legacy.py: Qwen3.5 on ROCm keeps the legacy torch.compile model (QFIX).

Found by the 1.14.0 release check (PR #262 on base #259): the Qwen prod config on the CI image was -4..-6 % on
four decode categories and -6 % on 8k/16k prefill against 1.13.4. Cause: vLLM main@1bdc437 routes Qwen3.5 on ROCm
to the new eager model `vllm/models/qwen3_5/amd` (own decoder `forward`, own `Qwen3_5RMSNorm`) and, because
`default_breakable_cudagraph_architectures()` lists Qwen3.5 on ROCm, switches torch.compile OFF (mode NONE, breakable
CUDA graphs). Every radiance hook lives in the legacy `model_executor/models/qwen3_next.py` layer, so
`radiance::add_rms_quant` (input / post-attention norm) never ran and the A-tiled prefill GEMMs lost their tiled
activations. The patch makes the legacy model the ROCm default again; RADIANCE_QWEN35_AMD=1 opts into the upstream model.

The patch runs on verbatim fixtures of the two upstream files (sly/tests/fixtures); no vllm import.
Run: pytest sly/tests/test_qwen35_legacy_patch_cpu.py -q
"""
import os
import subprocess
import sys
import textwrap
from functools import lru_cache
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PATCH = REPO / "sly" / "patch_qwen35_legacy.py"
FIX = REPO / "sly" / "tests" / "fixtures"
INIT = "vllm/models/qwen3_5/__init__.py"
CONFIG = "vllm/config/vllm.py"
QWEN = {"Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration", "Qwen3_5MTP",
        "Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeMTP"}
GATED_NON_QWEN = {"DeepseekV41ForCausalLM", "HYV4ForCausalLM", "HYV4VLForConditionalGeneration", "HYV4MTPModel"}


@pytest.fixture()
def tree(tmp_path):
    for rel, fx in ((INIT, "vllm_models_qwen3_5_init_main1bdc437.py.txt"),
                    (CONFIG, "vllm_breakable_archs_main1bdc437.py.txt")):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text((FIX / fx).read_text())
    return tmp_path


def run(tree):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(tree), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(PATCH)], capture_output=True, text=True, env=env)


def rocm_archs(tree, platform="rocm", amd_env=None):
    """default_breakable_cudagraph_architectures() of the (patched) config excerpt, on a stub platform."""
    class P:
        is_rocm = staticmethod(lambda: platform == "rocm")
        is_cuda = staticmethod(lambda: platform == "cuda")
        is_xpu = staticmethod(lambda: platform == "xpu")

    old = os.environ.pop("RADIANCE_QWEN35_AMD", None)
    if amd_env is not None:
        os.environ["RADIANCE_QWEN35_AMD"] = amd_env
    try:
        ns = {"os": os, "lru_cache": lru_cache, "current_platform": P, "frozenset": frozenset}
        exec(compile((tree / CONFIG).read_text(), CONFIG, "exec"), ns)
        return set(ns["default_breakable_cudagraph_architectures"]())
    finally:
        os.environ.pop("RADIANCE_QWEN35_AMD", None)
        if old is not None:
            os.environ["RADIANCE_QWEN35_AMD"] = old


def which_model(tree, platform, amd_env=None, tmp=None):
    """Import the patched vllm.models.qwen3_5 package against stub submodules; return the origin tag."""
    # one dir per call: the stub platform files of two calls differ only in True/False (same size), and a
    # shared dir with a same-second rewrite is served from the stale .pyc
    stub = Path(tmp) / f"stub-{platform}-{amd_env}"
    names = ["Qwen3_5ForCausalLM", "Qwen3_5ForCausalLMBase", "Qwen3_5ForConditionalGeneration", "Qwen3_5Model",
             "Qwen3_5MoeForCausalLM", "Qwen3_5MoeForConditionalGeneration"]
    mtp = ["Qwen3_5MoeMTP", "Qwen3_5MTP"]

    def mod(rel, tag, ns):
        p = stub / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(f"{n} = type({n!r}, (), {{'origin': {tag!r}}})\n" for n in ns))

    for d in ("vllm", "vllm/models", "vllm/model_executor", "vllm/model_executor/models", "vllm/models/qwen3_5"):
        (stub / d).mkdir(parents=True, exist_ok=True)
        (stub / d / "__init__.py").touch()
    (stub / "vllm/models/qwen3_5/__init__.py").write_text((tree / INIT).read_text())
    for sub in ("amd", "nvidia", "xpu"):
        (stub / "vllm/models/qwen3_5" / sub).mkdir(exist_ok=True)
        (stub / "vllm/models/qwen3_5" / sub / "__init__.py").touch()
        mod(f"vllm/models/qwen3_5/{sub}/model.py", sub, names)
        mod(f"vllm/models/qwen3_5/{sub}/mtp.py", sub, mtp)
    mod("vllm/model_executor/models/qwen3_5.py", "legacy", names)
    mod("vllm/model_executor/models/qwen3_5_mtp.py", "legacy", mtp)
    (stub / "vllm/platforms.py").write_text(textwrap.dedent(f"""
        class current_platform:
            is_cuda = staticmethod(lambda: {platform == 'cuda'})
            is_xpu = staticmethod(lambda: {platform == 'xpu'})
            is_rocm = staticmethod(lambda: {platform == 'rocm'})
    """))
    env = {k: v for k, v in os.environ.items() if k != "RADIANCE_QWEN35_AMD"}
    env["PYTHONPATH"] = str(stub)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if amd_env is not None:
        env["RADIANCE_QWEN35_AMD"] = amd_env
    r = subprocess.run([sys.executable, "-c",
                        "import vllm.models.qwen3_5 as m; print(m.Qwen3_5ForConditionalGeneration.origin, "
                        "m.Qwen3_5MTP.origin)"], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr
    return r.stdout.split()


def test_stock_rocm_selects_the_amd_model_and_breakable_graphs(tree, tmp_path):
    """The reported behaviour, on the unpatched upstream files."""
    assert QWEN <= rocm_archs(tree)
    assert which_model(tree, "rocm", tmp=tmp_path) == ["amd", "amd"]


def test_rocm_default_is_the_legacy_model_with_torch_compile(tree, tmp_path):
    assert run(tree).returncode == 0
    archs = rocm_archs(tree)
    assert not (QWEN & archs), "Qwen3.5 must not default to breakable graphs (mode NONE) any more"
    assert GATED_NON_QWEN <= archs, "models that cannot torch.compile keep their default"
    assert which_model(tree, "rocm", tmp=tmp_path) == ["legacy", "legacy"]


def test_opt_in_keeps_the_upstream_amd_model(tree, tmp_path):
    assert run(tree).returncode == 0
    assert QWEN <= rocm_archs(tree, amd_env="1")
    assert which_model(tree, "rocm", amd_env="1", tmp=tmp_path) == ["amd", "amd"]
    assert not (QWEN & rocm_archs(tree, amd_env="0"))


def test_other_platforms_are_unchanged(tree, tmp_path):
    stock = {p: rocm_archs(tree, p) for p in ("cuda", "xpu")}
    assert run(tree).returncode == 0
    assert {p: rocm_archs(tree, p) for p in ("cuda", "xpu")} == stock
    assert which_model(tree, "cuda", tmp=tmp_path) == ["nvidia", "nvidia"]
    assert which_model(tree, "xpu", tmp=tmp_path) == ["xpu", "xpu"]


def test_second_apply_is_a_noop(tree):
    assert run(tree).returncode == 0
    once = {rel: (tree / rel).read_text() for rel in (INIT, CONFIG)}
    second = run(tree)
    assert second.returncode == 0 and second.stdout.count("NOOP") == 2
    assert {rel: (tree / rel).read_text() for rel in (INIT, CONFIG)} == once


def test_tree_without_the_amd_model_is_skipped(tmp_path):
    """vLLM 0.34 has no vllm/models/qwen3_5: nothing to patch, not a failure."""
    f = tmp_path / CONFIG
    f.parent.mkdir(parents=True)
    f.write_text("x = 1\n")
    r = run(tmp_path)
    assert r.returncode == 0 and "SKIP" in r.stdout
    assert f.read_text() == "x = 1\n"


def test_fixtures_are_the_upstream_text():
    src = os.environ.get("INKLING_VLLM_SRC")
    if not src or not Path(src, "config/vllm.py").exists():
        pytest.skip("INKLING_VLLM_SRC not set")
    assert (FIX / "vllm_breakable_archs_main1bdc437.py.txt").read_text() in Path(src, "config/vllm.py").read_text()
    assert (FIX / "vllm_models_qwen3_5_init_main1bdc437.py.txt").read_text() == \
        Path(src, "models/qwen3_5/__init__.py").read_text()
