#!/usr/bin/env python3
"""CPU tests for the gate view (sly/patch_attn_gate_view.py) and the aux-from-residual capture
(sly/patch_dflash_aux_residual.py), plus the compile-key and loop wiring of both and of patch_gdn_glue.py.

Both patches claim the same bits as the stock path, so the tests run the patched vLLM 0.32.0 code
(fixtures/qwen3_next_v0.32.0.py.txt, verbatim) with small CPU modules, knob on against knob off,
eager and under torch.compile, and compare with torch.equal:

  * gate view: the attention output is bit-identical, and with a graph break where vLLM splits at
    attention, the graph before the break no longer writes a gate copy (Inductor's byte count);
  * aux residual: final hidden states and every aux state are bit-identical, aux states keep their
    order, and the bf16 residual adds for aux indices that feed a layer are gone (dispatch count),
    while an aux index after the last layer and a layer with its own forward keep the stock add.

Needs torch (CPU); no vLLM, no GPU.

Run:  python3 sly/tests/test_inductor_waste_cpu.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci"))
from patch_list import patch_names  # noqa: E402

FIXTURE = ROOT / "sly" / "tests" / "fixtures" / "qwen3_next_v0.32.0.py.txt"
REL = Path("vllm/model_executor/models/qwen3_next.py")
GATE = ROOT / "sly" / "patch_attn_gate_view.py"
AUX = ROOT / "sly" / "patch_dflash_aux_residual.py"
KEYS = ROOT / "sly" / "patch_graph_knob_keys.py"
IMPORT = "from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate\n"
BF = torch.bfloat16


def _run_patch(patch, site):
    env = dict(os.environ, PYTHONPATH=str(ROOT), RADIANCE_PATCH_TARGET_DIR=str(site))
    return subprocess.run([sys.executable, str(patch)], env=env, capture_output=True, text=True,
                          cwd=str(ROOT / "sly"))


_SRC = {}


def _patched_source():
    if "src" not in _SRC:
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / REL
            f.parent.mkdir(parents=True)
            f.write_text(FIXTURE.read_text())
            for p in (GATE, AUX):
                r = _run_patch(p, td)
                assert r.returncode == 0, r.stdout + r.stderr
            _SRC["src"] = f.read_text()
    return _SRC["src"]


_PP = NS(is_first_rank=True, is_last_rank=True)


class _EagleMixin:  # vllm/model_executor/models/interfaces.py, v0.32.0
    def _maybe_add_hidden_state(self, aux_hidden_states, layer_idx, hidden_states, residual):
        if layer_idx in self.aux_hidden_state_layers:
            value = hidden_states + residual if residual is not None else hidden_states
            aux_hidden_states.append(value)
        return aux_hidden_states


def _module(gate_view="0", aux="0"):
    """The patched fixture executed with both knobs as given (they are read at import)."""
    old = {k: os.environ.get(k) for k in ("RADIANCE_ATTN_GATE_VIEW", "RADIANCE_DFLASH_AUX_FROM_RESIDUAL")}
    os.environ.update(RADIANCE_ATTN_GATE_VIEW=gate_view, RADIANCE_DFLASH_AUX_FROM_RESIDUAL=aux)
    try:
        ns = dict(torch=torch, nn=torch.nn, islice=__import__("itertools").islice, _rfn=None,
                  support_torch_compile=lambda c: c, EagleModelMixin=_EagleMixin,
                  get_pp_group=lambda: _PP,
                  IntermediateTensors=dict, __name__="qwen3_next_fixture")
        src = _patched_source().replace(IMPORT, "fused_qk_rmsnorm_rope_gate = None\n")
        exec(compile("from __future__ import annotations\n" + src, "qwen3_next.py", "exec"), ns)
        return ns
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# --- gate view ---------------------------------------------------------------------------------

H, KVH, D, HID = 4, 2, 32, 64


@torch.library.custom_op("radiance_test::add_rms", mutates_args=())
def _add_rms(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque add + RMSNorm, as radiance.add_rms_quant is to Inductor: residual out = bf16(f32 + f32)."""
    s = (x.float() + residual.float()).to(x.dtype)
    sf = s.float()
    y = sf * torch.rsqrt(sf.pow(2).mean(-1, keepdim=True) + eps)
    return (y * weight.float()).to(x.dtype), s


@_add_rms.register_fake
def _(x, residual, weight, eps):
    return torch.empty_like(x), torch.empty_like(x)


class _RMS(torch.nn.Module):
    def __init__(self, n, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter((torch.rand(n, generator=g) + 0.5).to(BF), requires_grad=False)
        self.variance_epsilon = 1e-6

    def forward(self, x, residual=None):
        if residual is not None:
            return torch.ops.radiance_test.add_rms(x, residual, self.weight, self.variance_epsilon)
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return (y * self.weight.float()).to(x.dtype)


def _attention(ns, graph_break=False):
    cls = ns["Qwen3NextAttention"]

    class Attn(torch.nn.Module):
        _project_qkv_gate = cls._project_qkv_gate
        forward = cls.forward

        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(0)
            self.num_heads, self.num_kv_heads, self.head_dim = H, KVH, D
            self.q_size, self.kv_size = H * D, KVH * D
            self.attn_output_gate, self.use_fused_qk_norm_rope_gate = True, False
            self.q_norm, self.k_norm = _RMS(D, 1), _RMS(D, 2)
            self.wqkv = torch.nn.Parameter(torch.randn(HID, 2 * H * D + 2 * KVH * D, generator=g).to(BF) / 8,
                                           requires_grad=False)
            self.wo = torch.nn.Parameter(torch.randn(H * D, HID, generator=g).to(BF) / 8, requires_grad=False)
            self.graph_break = graph_break

        def qkv_proj(self, x):
            return x @ self.wqkv, None

        def rotary_emb(self, positions, q, k):
            c = torch.cos(positions.float())[:, None].to(q.dtype)
            return q * c, k * c

        def attn(self, q, k, v):  # stands in for the attention op: new, contiguous [M, heads*head_dim]
            if self.graph_break:
                torch._dynamo.graph_break()   # vLLM splits the graph at attention
            vv = v.view(-1, KVH, D).repeat_interleave(H // KVH, dim=1)
            return (q.view(-1, H, D) * 0.5 + vv).reshape(-1, H * D)

        def o_proj(self, x):
            return x @ self.wo, None

    return Attn()


def _inputs(m=37):
    g = torch.Generator().manual_seed(7)
    return torch.arange(m), torch.randn(m, HID, generator=g).to(BF)


def test_gate_view_is_bit_identical_eager_and_compiled():
    pos, x = _inputs()
    off, on = _attention(_module("0")), _attention(_module("1"))
    with torch.no_grad():
        assert torch.equal(off(pos, x), on(pos, x))
        torch._dynamo.reset()
        c_off = torch.compile(off, fullgraph=True)(pos, x)
        torch._dynamo.reset()
        c_on = torch.compile(on, fullgraph=True)(pos, x)
    assert torch.equal(c_off, c_on)


def test_gate_view_drops_the_gate_copy_before_the_split():
    """Inductor's code for the graph before the split: stock allocates an [M, heads, head_dim] buffer
    for the gate copy and returns it; with the view it returns a reinterpret of the projection."""
    import re
    from torch._inductor.utils import run_and_get_code
    pos, x = _inputs(256)
    allocs, returns = {}, {}
    for knob in ("0", "1"):
        torch._dynamo.reset()
        with torch.no_grad():
            _, codes = run_and_get_code(torch.compile(_attention(_module(knob), graph_break=True)), pos, x)
        first = codes[0]
        allocs[knob] = re.findall(r"empty_strided_\w+\(\(([^)]*)\)", first)
        returns[knob] = next(l for l in first.splitlines() if l.strip().startswith("return ("))
    gate_buf = f"256, {H}, {D}"
    assert gate_buf in allocs["0"] and gate_buf not in allocs["1"], allocs
    assert len(allocs["0"]) == len(allocs["1"]) + 1, allocs
    # the gate leaves the graph as a view of the q/k/v projection (row stride = its full width)
    width = 2 * H * D + 2 * KVH * D
    assert f"({width}, {2 * D}, 1)" in returns["1"], returns["1"]   # per head: [q | gate]


def test_project_qkv_gate_default_keeps_the_2d_gate():
    # radiance_paroquant's attention forward calls _project_qkv_gate(qkv, positions) and wants [M, h*d]
    attn = _attention(_module("1"))
    pos, x = _inputs()
    qkv, _ = attn.qkv_proj(x)
    with torch.no_grad():
        assert attn._project_qkv_gate(qkv, pos)[3].shape == (37, H * D)
        assert attn._project_qkv_gate(qkv, pos, gate_view=True)[3].shape == (37, H, D)


# --- aux states from the input residual -----------------------------------------------------------

def _model(ns, n_layers=6, aux_layers=(0, 2, 3, 6), other_forward_at=None):
    layer_cls, model_cls = ns["Qwen3NextDecoderLayer"], ns["Qwen3NextModel"]

    class Layer(torch.nn.Module):
        forward = layer_cls.forward

        def __init__(self, i):
            super().__init__()
            g = torch.Generator().manual_seed(100 + i)
            self.layer_type, self.layer_scale, self.use_attn_reduce_scatter_for_moe = "linear_attention", False, False
            self.input_layernorm, self.post_attention_layernorm = _RMS(HID, 10 + i), _RMS(HID, 50 + i)
            self.w1 = torch.nn.Parameter(torch.randn(HID, HID, generator=g).to(BF) / 8, requires_grad=False)
            self.w2 = torch.nn.Parameter(torch.randn(HID, HID, generator=g).to(BF) / 8, requires_grad=False)

        def linear_attn(self, hidden_states):
            return torch.tanh(hidden_states @ self.w1)

        def mlp(self, x):
            return torch.nn.functional.silu(x @ self.w2)

    class OwnForward(Layer):
        def forward(self, hidden_states, residual, positions, **kw):
            return Layer.forward(self, hidden_states, residual, positions)

    class Model(model_cls):
        def __init__(self):
            torch.nn.Module.__init__(self)
            g = torch.Generator().manual_seed(3)
            self.layers = torch.nn.ModuleList(
                (OwnForward if i == other_forward_at else Layer)(i) for i in range(n_layers))
            self.start_layer, self.end_layer = 0, n_layers
            self.emb = torch.nn.Parameter(torch.randn(1000, HID, generator=g).to(BF), requires_grad=False)
            self.norm = _RMS(HID, 99)
            self.aux_hidden_state_layers = tuple(aux_layers)

        def embed_input_ids(self, input_ids):
            return self.emb[input_ids]

    return Model()


class _CountBf16Adds(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.n = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func is torch.ops.aten.add.Tensor and args[0].dtype == BF:
            self.n += 1
        return func(*args, **(kwargs or {}))


def _run(model, compiled=False):
    ids = torch.arange(29) * 7 % 1000
    pos = torch.arange(29)
    if not compiled:
        with torch.no_grad():
            return model(ids, pos)
    # round fused bf16 intermediates like eager, so the toy layers around the patched code compare
    # bit for bit (the model's own norms are opaque custom ops, as on the GPU path)
    torch._dynamo.reset()
    with torch._inductor.config.patch(emulate_precision_casts=True), torch.no_grad():
        return torch.compile(model, fullgraph=True)(ids, pos)


def _same(a, b):
    (ha, auxa), (hb, auxb) = a, b
    return torch.equal(ha, hb) and len(auxa) == len(auxb) and all(torch.equal(x, y) for x, y in zip(auxa, auxb))


def test_aux_from_residual_is_bit_identical_eager_and_compiled():
    off, on = _model(_module(aux="0")), _model(_module(aux="1"))
    on.load_state_dict(off.state_dict())
    assert _same(_run(off), _run(on))
    assert _same(_run(off, compiled=True), _run(on, compiled=True))
    assert len(_run(on)[1]) == 4


def test_aux_from_residual_drops_the_adds_that_feed_a_layer():
    counts = {}
    for knob in ("0", "1"):
        m = _model(_module(aux=knob))
        with _CountBf16Adds() as c, torch.no_grad():
            m(torch.arange(29) * 7 % 1000, torch.arange(29))
        counts[knob] = c.n
    # aux (0, 2, 3, 6) of 6 layers: 0 is the embedding (no add), 6 follows the last layer (stock add),
    # 2 and 3 are the inputs of layers 2 and 3 -> two adds fewer
    assert counts["0"] - counts["1"] == 2, counts


def test_aux_from_residual_keeps_the_stock_path_for_other_layer_forwards():
    off = _model(_module(aux="0"), other_forward_at=4)
    on = _model(_module(aux="1"), other_forward_at=4)
    on.load_state_dict(off.state_dict())
    assert _same(_run(off), _run(on))
    with _CountBf16Adds() as c, torch.no_grad():
        on(torch.arange(29) * 7 % 1000, torch.arange(29))
    with _CountBf16Adds() as c0, torch.no_grad():
        off(torch.arange(29) * 7 % 1000, torch.arange(29))
    assert c.n == c0.n, (c.n, c0.n)   # one foreign layer: the whole capture stays stock


# --- patch mechanics and wiring -----------------------------------------------------------------

def test_patches_apply_after_fused_norm_quant_and_are_idempotent():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / REL
        f.parent.mkdir(parents=True)
        f.write_text(FIXTURE.read_text())
        for p in (AUX, GATE):   # either order: both keep the shared import anchor
            assert _run_patch(p, td).returncode == 0
        out = f.read_text()
        compile(out, "qwen3_next.py", "exec")
        for p in (GATE, AUX):
            r = _run_patch(p, td)
            assert r.returncode == 0 and "OK" not in r.stdout, r.stdout
        assert f.read_text() == out


def test_compile_key_patch():
    line = '                                   "RADIANCE_LMHEAD_INT4_TILED")):\n'
    stub = ("def compile_factors():\n    factors = {}\n    import os\n    for _radiance_k in sorted(os.environ):\n"
            "        if _radiance_k.startswith((\"RADIANCE_FUSED_NORM_QUANT\",\n" + line +
            "            factors[_radiance_k] = os.environ[_radiance_k]\n    return factors\n")
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "vllm" / "envs.py"
        f.parent.mkdir(parents=True)
        f.write_text(stub)
        assert _run_patch(KEYS, td).returncode == 0
        ns = {}
        exec(f.read_text(), ns)
        os.environ["RADIANCE_ATTN_GATE_VIEW"] = "1"
        try:
            got = ns["compile_factors"]()
        finally:
            os.environ.pop("RADIANCE_ATTN_GATE_VIEW")
        assert got.get("RADIANCE_ATTN_GATE_VIEW") == "1"
        assert "NOOP" in _run_patch(KEYS, td).stdout


def test_wiring_loop_and_docs():
    names = patch_names()
    for p in ("sly/patch_attn_gate_view", "sly/patch_dflash_aux_residual", "patch_gdn_glue",
              "sly/patch_graph_knob_keys"):
        assert p in names, f"{p} is not in the Dockerfile loop"
    i = names.index
    assert i("sly/patch_fused_norm_quant") < i("sly/patch_dflash_aux_residual")
    assert i("patch_r4d") < i("patch_gdn_glue")
    assert i("sly/patch_w4a16_fuse") < i("sly/patch_graph_knob_keys")
    unused = (ROOT / "ci" / "unused_patches.txt").read_text()
    assert "patch_gdn_glue.py" not in unused
    readme = (ROOT / "README.md").read_text()
    for knob in ("RADIANCE_ATTN_GATE_VIEW", "RADIANCE_DFLASH_AUX_FROM_RESIDUAL", "RADIANCE_GDN_EMPTY_OUT"):
        assert knob in readme, knob
    sly_readme = (ROOT / "sly" / "README.md").read_text()
    for p in ("patch_attn_gate_view.py", "patch_dflash_aux_residual.py", "patch_graph_knob_keys.py"):
        assert p in sly_readme, p


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("inductor waste tests OK")
