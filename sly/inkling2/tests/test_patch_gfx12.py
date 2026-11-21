"""sly/patch_inkling2_gfx12.py: Inkling 2's sm120 (Triton) path on gfx12, by source patch.

Runs the real patch script against a COPY of a vLLM source tree (INKLING_VLLM_SRC's parent) through
RADIANCE_PATCH_TARGET_DIR, so the anchors are checked against the real upstream text.
"""
import ast
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
PATCH = REPO / "sly" / "patch_inkling2_gfx12.py"


@pytest.fixture()
def tree(tmp_path):
    src = os.environ.get("INKLING_VLLM_SRC")
    if not src or not Path(src, "models/inkling2").exists():
        pytest.skip("INKLING_VLLM_SRC not set")
    shutil.copytree(src, tmp_path / "vllm", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def run(target):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(target), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(PATCH)], capture_output=True, text=True, env=env)


def py_files(target):
    return list((target / "vllm/models/inkling2").rglob("*.py"))


def test_removes_pdl_and_cuda_extras(tree):
    r = run(tree)
    assert r.returncode == 0, r.stdout + r.stderr
    text = "\n".join(p.read_text() for p in py_files(tree))
    assert "tl.extra.cuda" not in text
    assert "launch_pdl" not in text


def test_every_sm120_gate_also_accepts_rocm(tree):
    run(tree)
    for p in py_files(tree):
        for line in p.read_text().splitlines():
            if "is_device_capability_family(120)" in line and p.name != "nvfp4.py":
                assert "is_rocm()" in line, f"{p.name}: {line.strip()}"


def test_native_e2m1_gate_stays_false_on_rocm(tree):
    """ops/nvfp4.py: the gate selects the PTX e2m1 conversion; gfx12 must take the software path."""
    run(tree)
    s = (tree / "vllm/models/inkling2/nvidia/ops/nvfp4.py").read_text()
    assert "is_rocm" not in s


def test_only_the_deliberate_ptx_is_left(tree):
    run(tree)
    left = set()
    for p in py_files(tree):
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr == "inline_asm_elementwise" and n.func.value.id == "tl":
                left.add(p.name)
    # fp8-KV unpack (only compiled with --kv-cache-dtype fp8) and the dead e2m1 hardware branches
    assert left <= {"attention_sm90.py", "nvfp4.py"}, left


@pytest.mark.parametrize("asm,args,expect", [
    ("ex2.approx.f32 $0, $1;", "x", "tl.exp2(x)"),
    ("lg2.approx.ftz.f32 $0, $1;", "x", "tl.log2(x)"),
    ("rcp.approx.f32 $0, $1;", "exp + 1.0", "(1.0 / (exp + 1.0))"),
    ("rsqrt.approx.f32 $0, $1;", "v", "tl.rsqrt(v)"),
    ("cvt.rn.bf16.f32 $0, $1;", "x", "(x).to(tl.bfloat16)"),
    ("mov.b32 $0, $1;", "partial", "(partial)"),
])
def test_ptx_rewrite_table(asm, args, expect):
    sys.path.insert(0, str(REPO / "sly"))
    sys.path.insert(0, str(REPO))
    try:
        import patch_inkling2_gfx12 as p
    finally:
        sys.path.pop(0)
        sys.path.pop(0)
    src = f"def f(x):\n    return tl.inline_asm_elementwise(\n        {asm!r},\n        '=f,f',\n        [{args}],\n        dtype=tl.float32,\n    )\n"
    new, n = p.replace_ptx(src)
    assert n == 1
    assert expect in new  # the rcp row pins operator precedence: 1/(e+1), not 1/e+1


def test_nvfp4_experts_use_the_dequant_kernel_on_rocm(tree):
    run(tree)
    s = (tree / "vllm/models/inkling2/nvidia/ops/moe_sm120.py").read_text()
    assert "from radiance_inkling2_moe import nvfp4_expert_gemm" in s
    assert "torch.version.hip" in s


def test_result_still_parses_and_second_run_changes_nothing(tree):
    run(tree)
    for p in py_files(tree):
        ast.parse(p.read_text())
    before = {p: p.read_text() for p in py_files(tree)}
    r = run(tree)
    assert r.returncode == 0
    assert "OK" not in re.findall(r"^  (OK|NOOP)", r.stdout, re.M)
    assert before == {p: p.read_text() for p in py_files(tree)}


def test_vllm_without_inkling2_is_a_noop(tmp_path):
    (tmp_path / "vllm/models").mkdir(parents=True)
    r = run(tmp_path)
    assert r.returncode == 0 and "NOOP" in r.stdout


def test_math_templates_compute_the_ptx_function():
    """Each rewrite must equal the function its PTX computed (tanh/sigmoid are the easy ones to get
    wrong: the exp2 argument needs 2/ln2 resp. 1/ln2)."""
    import types

    sys.path[:0] = [str(REPO / "sly"), str(REPO)]
    try:
        import patch_inkling2_gfx12 as p
    finally:
        del sys.path[:2]
    tl = types.SimpleNamespace(exp2=torch.exp2, log2=torch.log2, rsqrt=torch.rsqrt,
                               bfloat16=torch.bfloat16)
    x = torch.linspace(-30.0, 30.0, 241)
    pos = x.abs() + 0.5
    want = {
        r"tanh": (x, torch.tanh(x)),
        "mul": (x, torch.sigmoid(x)),  # activation._sigmoid_approx
        r"^rsqrt": (pos, torch.rsqrt(pos)),
        r"^lg2": (pos, torch.log2(pos)),
        r"^ex2": (x / 4, torch.exp2(x / 4)),
        r"^rcp": (pos, 1.0 / pos),
    }
    seen = set()
    for pat, tmpl in p._ASM:
        for key, (arg, expected) in want.items():
            if key in pat or pat.startswith(key):
                got = eval(tmpl.format("arg"), {"tl": tl, "arg": arg})
                torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-6)
                seen.add(key)
    assert seen == set(want), set(want) - seen


def test_rocm_platform_lists_inkling2_quantization(tree):
    """Found in the first GPU window: ModelConfig raised 'inkling2 quantization is currently not
    supported in rocm' (platforms/rocm.py supported_quantization)."""
    run(tree)
    s = (tree / "vllm/platforms/rocm.py").read_text()
    block = s.split("supported_quantization: list[str] = [")[1].split("\n    ]")[0]
    assert '"inkling2"' in block


def test_libdevice_functions_missing_on_hip_are_replaced(tree):
    """Second GPU-window finding: hip libdevice has no rcp_rn / div_rn (AttributeError at compile).
    tl.div_rn is core and exact on both."""
    run(tree)
    text = "\n".join(p.read_text() for p in py_files(tree))
    assert "tldevice.rcp_rn" not in text and "tldevice.div_rn" not in text
    assert "tl.div_rn(1.0, 1536.0 * tl.load(WDenominator))" in text


def test_launch_kwarg_maxnreg_is_dropped(tree):
    """Third finding: AMD's Triton rejects the NVIDIA-only launch kwarg `maxnreg`."""
    run(tree)
    assert "maxnreg=maxnreg" not in "\n".join(p.read_text() for p in py_files(tree))


def test_hardware_e2m1_branch_is_removed_not_just_gated(tree):
    """Fourth finding: Triton compiles the statements after `if not _HW_E2M1: return ...` anyway, so
    the PTX there reached LLVM ('could not allocate output register for constraint f')."""
    run(tree)
    s = (tree / "vllm/models/inkling2/nvidia/ops/nvfp4.py").read_text()
    assert "inline_asm_elementwise" not in s
    ast.parse(s)
    assert "_e2m1(a)" in s and "_e2m1(b)" in s  # the software path survives
