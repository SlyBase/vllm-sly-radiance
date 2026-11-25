"""sly/patch_inkling2_gfx12.py: Inkling 2's sm120 (Triton) path on gfx12, by source patch.

Runs the real patch script against a COPY of a vLLM source tree (INKLING_VLLM_SRC's parent) through
RADIANCE_PATCH_TARGET_DIR, so the anchors are checked against the real upstream text.
"""
import ast
import os
import re
import shutil
import subprocess
import types
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
    # only the dead e2m1 hardware branches (the fp8-KV unpack is rewritten, see below)
    assert left <= {"nvfp4.py"}, left


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


def test_mxfp8_branch_routes_to_hip_on_rocm(tree):
    run(tree)
    s = (tree / "vllm/models/inkling2/nvidia/ops/moe_sm120.py").read_text()
    assert "from radiance_inkling2_moe import mxfp8_expert_gemm" in s
    assert "torch.version.hip" in s
    assert "mxfp8_expert_gemm(act, act_scale, weight, weight_scale, ones," in s


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


def test_mxfp8_gemm_tiles_fit_the_64k_lds_of_gfx12(tree):
    """Fifth finding: the sm120 MXFP8 tiles (BN 256, BT 128, 3 stages) need 99 KB of shared memory,
    gfx12 has 64 KB."""
    run(tree)
    s = (tree / "vllm/models/inkling2/nvidia/ops/moe_sm120.py").read_text()
    assert "if torch.version.hip:  # 64 KB LDS" in s


def test_rel_attention_uses_one_stage_on_rocm(tree):
    """Sixth finding: rel_attention with 2 stages needs 69 KB LDS (limit 64 KB)."""
    run(tree)
    s = (tree / "vllm/models/inkling2/nvidia/ops/attention_sm90.py").read_text()
    assert "num_stages=1 if torch.version.hip else 2," in s


def test_flashinfer_is_never_touched_on_rocm(tree):
    """Seventh finding: `is_flashinfer_autotuning()` imports flashinfer (absent on ROCm) before the
    `not use_sm100_kernels()` short-circuit gets a chance."""
    run(tree)
    s = (tree / "vllm/models/inkling2/nvidia/ops/linear.py").read_text()
    cond = s.split("# Autotuning would profile")[1].split("):")[0]
    assert cond.index("use_sm100_kernels") < cond.index("is_flashinfer_autotuning")


def _patched_rmsnorm(tree):
    """Exec the patched `_RMSNorm` of the DSpark draft on CPU (no vLLM import needed)."""
    import torch.nn as nn

    src = (tree / "vllm/models/inkling2/nvidia/dspark.py").read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "_RMSNorm")
    ns = {"torch": torch, "nn": nn}
    exec(ast.get_source_segment(src, node), ns)
    return ns["_RMSNorm"]


def test_draft_rmsnorm_needs_no_flashinfer(tree):
    run(tree)
    src = (tree / "vllm/models/inkling2/nvidia/dspark.py").read_text()
    seg = src.split("class _RMSNorm")[1].split("def all_reduce_add_rmsnorm")[0]
    assert "flashinfer" not in seg


@pytest.mark.parametrize("with_residual", [False, True])
def test_draft_rmsnorm_matches_the_flashinfer_contract(tree, with_residual):
    """rmsnorm: bf16 out of an fp32 norm. fused_add: residual += x in place (rounded to bf16), x =
    norm(unrounded fp32 sum) in place."""
    run(tree)
    N = _patched_rmsnorm(tree)
    g = torch.Generator().manual_seed(1)
    x = torch.randn(5, 64, generator=g).bfloat16()
    r = torch.randn(5, 64, generator=g).bfloat16()
    norm = N(64, 1e-6)
    norm.weight.data = torch.rand(64, generator=g).bfloat16() + 0.5
    w = norm.weight.float()

    def ref(v):
        return (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6) * w).bfloat16()

    if not with_residual:
        out = norm(x.clone())
        torch.testing.assert_close(out, ref(x.float()))
        return
    xs, rs = x.clone(), r.clone()
    out = norm(xs, rs)
    total = x.float() + r.float()
    torch.testing.assert_close(rs, total.bfloat16())      # residual rounded once
    torch.testing.assert_close(out, ref(total))           # norm from the UNROUNDED sum
    assert out.data_ptr() == xs.data_ptr()                # in place, as flashinfer does


def test_draft_kv_write_does_not_need_the_cuda_cache_op(tree):
    """Fourth GPU-window finding (DSpark): FlashAttentionImpl.do_kv_cache_update calls
    reshape_and_cache_flash, which is not imported on ROCm (NameError)."""
    run(tree)
    src = (tree / "vllm/models/inkling2/nvidia/dspark.py").read_text()
    assert "do_kv_cache_update" not in src


def test_draft_kv_write_is_cuda_graph_safe(tree):
    """No data-dependent shapes (boolean indexing / nonzero) in the draft KV write: it runs inside
    the captured draft graph."""
    run(tree)
    src = (tree / "vllm/models/inkling2/nvidia/dspark.py").read_text()
    fn = src.split("def _write_kv")[1].split("    def forward")[0]
    assert "[ok]" not in fn and "nonzero" not in fn


def test_draft_kv_write_calls_the_triton_scatter(tree):
    run(tree)
    src = (tree / "vllm/models/inkling2/nvidia/dspark.py").read_text()
    fn = src.split("def _write_kv")[1].split("    def forward")[0]
    assert "kv_scatter(self.attn.kv_cache, k, v, slot_mapping)" in fn


def test_fused_route_is_not_a_cooperative_launch_on_rocm(tree):
    """A cooperative launch gets its own HIP hardware queue: the route kernel then alternates with
    queue 1 48x per step and every small kernel of the step hit a ~17 us floor on ~2/3 of the starts
    (62 vs 31 ms per verify step). The grid (routers + <= 16 tokens = 28 programs) is co-resident on
    any idle gfx12 anyway, which is all the spin-wait needs -- asserted before the launch."""
    assert run(tree).returncode == 0
    moe = (tree / "vllm/models/inkling2/nvidia/ops/moe.py").read_text()
    assert "launch_cooperative_grid=True," not in moe
    assert "launch_cooperative_grid=not torch.version.hip," in moe
    assert "multi_processor_count" in moe  # co-residency guard on ROCm


# ---------------------------------------------------------------- launch folding (draft, dynamic taps)


class _Torch:
    """`torch` for exec'd model methods: the real one with `torch.version.hip` set and an `addmm` that
    honours out_dtype on CPU."""

    def __init__(self, hip):
        self.version = types.SimpleNamespace(hip=hip)

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def addmm(bias, a, b, out_dtype=None):
        return (bias.float() + a.float() @ b.float()).to(out_dtype or a.dtype)


def _method(tree, cls, name):
    src = (tree / "vllm/models/inkling2/nvidia/dspark.py").read_text()
    c = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == cls)
    f = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(src, f)


def test_dynamic_taps_use_the_low_m_gemv_on_rocm(tree):
    run(tree)
    s = (tree / "vllm/models/inkling2/nvidia/sconv.py").read_text()
    taps = s.split("def dynamic_taps")[1].split("    def forward")[0]
    assert "from .ops.linear import linear" in taps
    assert taps.index("torch.version.hip") < taps.index("nn.functional.linear(x, self.weight_proj)")
    assert "linear_k_groups" in taps  # the 2560+ wide branch is untouched


def test_draft_rmsnorm_is_one_kernel_on_rocm(tree):
    run(tree)
    src = (tree / "vllm/models/inkling2/nvidia/dspark.py").read_text()
    seg = src.split("class _RMSNorm")[1].split("def all_reduce_add_rmsnorm")[0]
    assert "from radiance_inkling2_draft import rmsnorm" in seg
    assert "flashinfer" not in seg
    assert "residual.copy_(v)" in seg  # the torch body stays for what the kernel declines


def test_markov_step_and_logits_use_the_folded_kernels_on_rocm(tree):
    run(tree)
    step = _method(tree, "Inkling2MarkovHead", "step")
    assert "from radiance_inkling2_draft import markov_step" in step
    logits = _method(tree, "Inkling2DSparkModel", "markov_logits")
    assert "from radiance_inkling2_draft import fold_softcap" in logits


def _ref_gate(raw, state):
    r = state.shape[1]
    b = lambda t: t.bfloat16().float()
    gate, cand, out = b(raw[:, :r]), b(raw[:, r:2 * r]), b(raw[:, 2 * r:])
    gate = b(torch.sigmoid(gate))
    cand = b(torch.tanh(cand))
    keep = b(gate * state.float())
    update = b(b(1.0 - gate) * cand)
    return (keep + update).bfloat16(), torch.tanh(out).bfloat16()


def _ref_softcap(a, bias, cap):
    x = (a.float() + bias.float()).bfloat16().float()
    return (cap * torch.tanh(x / cap)).bfloat16()


def test_folded_markov_path_equals_the_original_ops(tree):
    """Patched Inkling2MarkovHead.step + Inkling2DSparkModel.markov_logits, run once with the original
    ops (hip off) and once folded (hip on), on the same inputs: the draft logits agree."""
    run(tree)
    r, rows, V, H = 32, 3, 200, 24
    g = torch.Generator().manual_seed(7)
    bf = lambda *shape, s=1.0: (torch.randn(*shape, generator=g) * s).bfloat16()
    head = types.SimpleNamespace(
        markov_rank=r,
        # fp32 holding bf16 values: the interpreter's bf16 tl.dot is wrong, the kernels follow the dtype
        fold_weight=bf(V, r, s=0.3).float(), fold_bias=bf(V).float(),
        markov_w1=types.SimpleNamespace(weight=bf(50, r).float()),
        rnn_proj=types.SimpleNamespace(weight=bf(3 * r, 2 * r + 2 * H, s=0.2).float()),
    )
    head.markov_w1.__call__ = None
    model = types.SimpleNamespace(
        model=types.SimpleNamespace(markov_head=head),
        config=types.SimpleNamespace(final_logit_softcapping=30.0, vocab_size=V - 3))
    state, prev = torch.tanh(bf(rows, r).float()).bfloat16(), torch.randint(0, 50, (rows,), generator=g)
    context = torch.randn(rows, 3 * r, generator=g)
    base = (bf(rows, 7, V, s=10))[:, 4]
    results = []
    sys.modules.pop("radiance_inkling2_draft", None)
    for hip in (None, "6.4"):
        ns = {"torch": _Torch(hip), "markov_gate": _ref_gate, "add_softcap": _ref_softcap,
              "tensor_model_parallel_all_gather": lambda x: x}
        exec(_method(tree, "Inkling2MarkovHead", "step"), ns)
        exec(_method(tree, "Inkling2DSparkModel", "markov_logits"), ns)
        h = types.SimpleNamespace(**vars(head))
        h.markov_w1 = lambda ids, w=head.markov_w1.weight: w[ids]   # nn.Embedding stand-in
        h.markov_w1.weight = head.markov_w1.weight
        m = types.SimpleNamespace(model=types.SimpleNamespace(markov_head=h), config=model.config)
        new_state, bias = ns["step"](h, state, prev, context)
        logits = ns["markov_logits"](m, base, bias)
        results.append((new_state, logits))
    assert "radiance_inkling2_draft" in sys.modules  # the hip run really took the folded path
    (s0, l0), (s1, l1) = results
    assert l1.shape == l0.shape == (rows, V - 3)
    torch.testing.assert_close(s1.float(), s0.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(l1.float(), l0.float(), atol=0.25, rtol=2e-2)


def test_shortlist_is_plumbed_from_the_speculator_to_the_fold(tree):
    run(tree)
    spec = (tree / "vllm/models/inkling2/speculator.py").read_text()
    assert "top = self.model.shortlist(base_logits)" in spec
    assert "None if top is None else (top[0][:, i], top[1][:, i])" in spec
    assert "markov_argmax(base_logits[:, i], bias)" in spec  # the greedy-only draft keeps the dense path
    assert "short: tuple[torch.Tensor, torch.Tensor] | None = None" in \
        _method(tree, "Inkling2DSparkModel", "markov_logits")
    assert "from radiance_inkling2_draft import SHORTLIST_K, shortlist" in \
        _method(tree, "Inkling2DSparkModel", "shortlist")


def test_shortlist_path_matches_dense_on_members_and_masks_the_rest(tree, monkeypatch):
    """Same harness as the folded-path test, shortlist on (k=20 of 200): members carry the dense
    logits, the rest is -inf, and k=0 is the dense path again."""
    run(tree)
    r, rows, V, H, n_spec = 32, 3, 200, 24, 7
    g = torch.Generator().manual_seed(11)
    bf = lambda *shape, s=1.0: (torch.randn(*shape, generator=g) * s).bfloat16()
    head = types.SimpleNamespace(
        markov_rank=r, fold_weight=bf(V, r, s=0.3).float(), fold_bias=bf(V).float(),
        markov_w1=types.SimpleNamespace(weight=bf(50, r).float()),
        rnn_proj=types.SimpleNamespace(weight=bf(3 * r, 2 * r + 2 * H, s=0.2).float()))
    w1 = head.markov_w1.weight
    head.markov_w1 = lambda ids, w=w1: w[ids]   # nn.Embedding stand-in
    head.markov_w1.weight = w1
    cfg = types.SimpleNamespace(final_logit_softcapping=30.0, vocab_size=V - 3)
    model = types.SimpleNamespace(model=types.SimpleNamespace(markov_head=head), config=cfg)
    state = torch.tanh(bf(rows, r).float()).bfloat16()
    prev = torch.randint(0, 50, (rows,), generator=g)
    context = torch.randn(rows, 3 * r, generator=g)
    base_all = bf(rows, n_spec, V, s=10)

    def run_path(hip, k):
        monkeypatch.setenv("RADIANCE_INKLING2_DRAFT_SHORTLIST", str(k))
        sys.modules.pop("radiance_inkling2_draft", None)
        ns = {"torch": _Torch(hip), "markov_gate": _ref_gate, "add_softcap": _ref_softcap,
              "tensor_model_parallel_all_gather": lambda x: x}
        exec(_method(tree, "Inkling2MarkovHead", "step"), ns)
        exec(_method(tree, "Inkling2DSparkModel", "markov_logits"), ns)
        exec(_method(tree, "Inkling2DSparkModel", "shortlist"), ns)
        _, bias = ns["step"](head, state, prev, context)
        top = ns["shortlist"](model, base_all)
        short = None if top is None else (top[0][:, 3], top[1][:, 3])
        return ns["markov_logits"](model, base_all[:, 3], bias, short), top

    dense, _ = run_path(None, 20)
    folded, top0 = run_path("6.4", 0)
    assert top0 is None
    torch.testing.assert_close(folded.float(), dense.float(), atol=0.25, rtol=2e-2)
    short, top = run_path("6.4", 20)
    assert top[1].shape == (rows, n_spec, 20)
    member = torch.zeros(rows, V, dtype=torch.bool).scatter_(1, top[1][:, 3], True)[:, : V - 3]
    assert torch.isneginf(short[~member].float()).all()
    torch.testing.assert_close(short[member].float(), dense[member].float(), atol=0.25, rtol=2e-2)
    assert int(member.sum()) >= rows * 20 - rows * 3   # at most 3 members per row are sliced off


# ---------------------------------------------------------------- loop controllers (RADIANCE_INKLING2_HIP_CTRL)


def _blend(tree, hip, controller_gemms, calls):
    """The patched Inkling2LoopDecoderLayer.blend, exec'd with stub kernels: pair_rms_norm / blend_rms_norm
    record, torch.bmm is the real one (counted), radiance_inkling2_gemv_hip is `controller_gemms`."""
    src = (tree / "vllm/models/inkling2/nvidia/loop.py").read_text()
    cls = next(n for n in ast.parse(src).body if isinstance(n, ast.ClassDef) and n.name == "Inkling2LoopDecoderLayer")
    fn = ast.get_source_segment(src, next(n for n in cls.body if getattr(n, "name", "") == "blend"))
    mod = types.ModuleType("radiance_inkling2_gemv_hip")
    mod.controller_gemms = controller_gemms
    sys.modules["radiance_inkling2_gemv_hip"] = mod

    class T(_Torch):
        def bmm(self, a, b, out_dtype=None):
            calls.append("bmm")
            return (a.float() @ b.float()).to(out_dtype or a.dtype)

    ns = {"torch": T(hip), "pair_rms_norm": lambda d, m: m.bfloat16().repeat(1, 1, 2),
          "blend_rms_norm": lambda d, m, gf, w, e: gf}
    exec("from __future__ import annotations\n" + fn.replace("    def blend", "def blend", 1).replace("\n    ", "\n"), ns)
    return ns["blend"]


def _layer():
    return types.SimpleNamespace(controller_down=torch.randn(4, 64, 8).bfloat16(),
                                 controller_gain_forget=torch.randn(4, 8, 64).bfloat16())


@pytest.mark.parametrize("hip", [True, False])
def test_loop_blend_uses_the_batched_controller_gemms_on_rocm_only(tree, hip):
    assert run(tree).returncode == 0
    calls, seen = [], []
    sentinel = torch.full((3, 2, 8), 7.0)

    def cg(pair, down, gf):
        seen.append((tuple(pair.shape), tuple(down.shape), tuple(gf.shape)))
        return sentinel

    blend = _blend(tree, hip, cg, calls)
    out = blend(_layer(), slice(0, 3), torch.randn(3, 2, 4), torch.randn(3, 2, 4), torch.ones(3, 4), 1e-6)
    if hip:  # the kernels' result goes straight into blend_rms_norm, torch.bmm never runs
        assert out is sentinel and calls == [] and seen == [((3, 2, 8), (3, 64, 8), (3, 8, 64))]
    else:
        assert seen == [] and calls == ["bmm", "bmm"]


def test_loop_blend_falls_back_to_bmm_when_the_kernels_decline(tree):
    assert run(tree).returncode == 0
    calls = []
    blend = _blend(tree, True, lambda p, d, g: None, calls)
    blend(_layer(), slice(3, 4), torch.randn(1, 2, 4), torch.randn(1, 2, 4), torch.ones(1, 4), 1e-6)
    assert calls == ["bmm", "bmm"]


def test_loop_blend_hook_is_idempotent(tree):
    run(tree)
    first = (tree / "vllm/models/inkling2/nvidia/loop.py").read_text()
    assert first.count("controller_gemms") >= 2
    run(tree)
    assert (tree / "vllm/models/inkling2/nvidia/loop.py").read_text() == first


def test_fp8_kv_unpack_has_no_ptx_on_rocm(tree):
    """--kv-cache-dtype fp8 compiled `cvt.rn.f16x2.e4m3x2` (NVIDIA PTX) into the attention kernel."""
    run(tree)
    src = (tree / "vllm/models/inkling2/nvidia/ops/attention_sm90.py").read_text()
    assert "e4m3x2" not in src and "inline_asm_elementwise" not in src
    assert "_load_fp8_kv(K, kv_ptrs" in src and "_load_fp8_kv(V, kv_ptrs" in src  # callers untouched


def test_fp8_kv_unpack_reads_the_e4m3_values(tree):
    """The patched helper, run in the Triton interpreter, returns each e4m3 byte as its value."""
    import importlib.util

    import triton
    import triton.language as tl

    run(tree)
    path = tree / "vllm/models/inkling2/nvidia/ops/attention_sm90.py"
    spec = importlib.util.spec_from_file_location("attn_sm90_patched", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    globals()["_patched_sm90"] = mod                              # a jit function resolves module globals

    @triton.jit
    def probe(Src, Offsets, Out, N: tl.constexpr, D: tl.constexpr):
        i = tl.arange(0, N)
        off = tl.load(Offsets + i)
        x = _patched_sm90._load_fp8_kv(Src, off, i < N - 1, tl.float32, D)   # the last row is masked out
        tl.store(Out + i[:, None] * D + tl.arange(0, D)[None, :], x)

    D, N = 16, 4
    src = (torch.arange(0, 256, dtype=torch.float32)[:8 * D].view(8, D) / 8 - 8).to(torch.float8_e4m3fn)
    offsets = torch.tensor([3 * D, 0, 7 * D, 5 * D], dtype=torch.int64)
    out = torch.full((N, D), -99.0)
    probe[(1,)](src, offsets, out, N=N, D=D)
    want = src.view(-1, D)[offsets // D].float()
    want[N - 1] = 0.0                                             # masked row: other=0
    assert torch.equal(out, want)
