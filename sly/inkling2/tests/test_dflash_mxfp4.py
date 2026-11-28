"""CPU tests for the MXFP4 post-load quantization of an unquantized bf16 DFlash/DFlash2 drafter
(process_dflash_draft) and the release of the dead shared bf16 lm_head (maybe_release_shared_head).

The drafter is a small stand-in with the module layout of DFlashQwen3Model (layers[i].self_attn.{qkv_proj,
o_proj}, layers[i].mlp.{gate_up_proj, down_proj}, fc, _fused_kv_weight). The matmul runs in w4a16 mode
(dequantized codes on plain torch), the same grid the kernels read.
"""
import importlib
import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

MODNAME = "radiance_inkling2_mxfp4dense"
ENV = ("RADIANCE_INKLING2_MXFP4_DENSE", "RADIANCE_INKLING2_MXFP4_DENSE_MODE", "RADIANCE_INKLING2_MXFP4_DENSE_MIN_N",
       "RADIANCE_DFLASH_MXFP4", "RADIANCE_INKLING2_FREE_DFLASH_HEAD")
H, LAYERS = 128, 5


def load(**env):
    for k in ENV:
        os.environ.pop(k, None)
    os.environ.update({"RADIANCE_INKLING2_MXFP4_DENSE_MODE": "w4a16", "RADIANCE_INKLING2_MXFP4_DENSE_MIN_N": "64", **env})
    sys.modules.pop(MODNAME, None)
    return importlib.import_module(MODNAME)


@pytest.fixture(autouse=True)
def _clean_env():
    yield
    for k in ENV:
        os.environ.pop(k, None)
    sys.modules.pop(MODNAME, None)


class UnquantizedLinearMethod:
    def apply(self, layer, x, bias=None):
        return F.linear(x, layer.weight, bias)


class CompressedTensorsLinearMethod(UnquantizedLinearMethod):
    pass


class Lin(torch.nn.Module):
    def __init__(self, k, n, qm=UnquantizedLinearMethod):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(n, k, dtype=torch.bfloat16), requires_grad=False)
        self.bias = None
        self.quant_method = qm()

    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)


def ns(**kw):
    m = torch.nn.Module()
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def make_drafter(qm=UnquantizedLinearMethod):
    torch.manual_seed(0)
    layers = torch.nn.ModuleList(
        ns(self_attn=ns(qkv_proj=Lin(H, 3 * H, qm), o_proj=Lin(2 * H, H, qm)),
           mlp=ns(gate_up_proj=Lin(H, 4 * H, qm), down_proj=Lin(2 * H, H, qm)))
        for _ in range(LAYERS))
    inner = ns(layers=layers, fc=Lin(4 * H, H, qm),
               pre_attn_conv=ns(kernel_projection=Lin(H, 2 * H)),
               selector=ns(hidden_projection=Lin(H, 2 * H)))
    inner._fused_kv_weight = torch.randn(LAYERS * 2 * 32, H, dtype=torch.bfloat16)
    inner._fused_kv_bias = None
    inner._normalize_context_states = lambda x: x

    def project(context_states, num_ctx, num_layers, num_kv_heads, head_dim):
        y = F.linear(inner._normalize_context_states(context_states), inner._fused_kv_weight, inner._fused_kv_bias)
        return y.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)

    inner._project_context_kv_token_major = project
    return ns(model=inner)


def cos(a, b):
    return F.cosine_similarity(a.float().flatten(), b.float().flatten(), dim=0).item()


def projections(d):
    ls = d.model.layers
    return [m for l in ls for m in (l.self_attn.qkv_proj, l.self_attn.o_proj, l.mlp.gate_up_proj, l.mlp.down_proj)] + [d.model.fc]


def freed(w):  # the bf16 is replaced by a one-value-per-row stub
    return w.stride(-1) == 0 and w.untyped_storage().nbytes() == w.shape[0] * w.element_size()


# ---------------------------------------------------------------- env knobs


def test_knob_defaults_follow_the_dense_knob():
    assert load(RADIANCE_INKLING2_MXFP4_DENSE="1").DFLASH_MXFP4 is True
    assert load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_DFLASH_MXFP4="0").DFLASH_MXFP4 is False
    assert load().DFLASH_MXFP4 is False
    assert load(RADIANCE_DFLASH_MXFP4="1").DFLASH_MXFP4 is True  # standalone
    assert load().FREE_DFLASH_HEAD is True
    assert load(RADIANCE_INKLING2_FREE_DFLASH_HEAD="0").FREE_DFLASH_HEAD is False


# ---------------------------------------------------------------- process_dflash_draft


def test_quantizes_all_21_linears_and_fused_kv(capfd):
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    d = make_drafter()
    x3 = torch.randn(2, 3, H, dtype=torch.bfloat16)
    ctx = torch.randn(4, H, dtype=torch.bfloat16)
    ref_kv = d.model._project_context_kv_token_major(ctx, 4, LAYERS, 2, 16)
    ins = {id(p): torch.randn(5, p.weight.shape[1], dtype=torch.bfloat16) for p in projections(d)}
    ref = {id(p): p(ins[id(p)]) for p in projections(d)}
    ref3 = d.model.layers[0].self_attn.qkv_proj(x3)

    assert m.process_dflash_draft(d) == 21
    assert "dflash_mxfp4: quantized 21 linear tensors (+fused_kv=True)" in capfd.readouterr().err
    assert len(projections(d)) == 21
    for p in projections(d):
        assert freed(p.weight)
        assert type(p.quant_method).__name__ == "_MXFP4DraftLinearMethod"
        assert cos(p(ins[id(p)]), ref[id(p)]) > 0.99
    out3 = d.model.layers[0].self_attn.qkv_proj(x3)
    assert out3.shape == ref3.shape == (2, 3, 3 * H) and cos(out3, ref3) > 0.99
    assert freed(d.model._fused_kv_weight)
    kv = d.model._project_context_kv_token_major(ctx, 4, LAYERS, 2, 16)
    assert kv.shape == ref_kv.shape and cos(kv, ref_kv) > 0.99


def test_second_call_is_a_noop():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    d = make_drafter()
    assert m.process_dflash_draft(d) == 21
    assert m.process_dflash_draft(d) == 0
    d.model._project_context_kv_token_major(torch.randn(2, H, dtype=torch.bfloat16), 2, LAYERS, 2, 16)


def test_conv_selector_and_prequantized_layers_stay_bf16():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    d = make_drafter()
    kp, hp = d.model.pre_attn_conv.kernel_projection, d.model.selector.hidden_projection
    m.process_dflash_draft(d)
    for p in (kp, hp):
        assert p.weight.stride(-1) == 1 and p.weight.untyped_storage().nbytes() == p.weight.numel() * 2
        assert type(p.quant_method).__name__ == "UnquantizedLinearMethod"
    q = make_drafter(CompressedTensorsLinearMethod)
    q.model._fused_kv_weight = None  # a quantized drafter builds no fused bf16 kv weight
    assert m.process_dflash_draft(q) == 0
    for p in projections(q):
        assert p.weight.stride(-1) == 1 and type(p.quant_method).__name__ == "CompressedTensorsLinearMethod"


def test_k_not_multiple_of_64_stays_bf16():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    d = make_drafter()
    d.model.fc = Lin(96, H)
    assert m.process_dflash_draft(d) == 20
    assert d.model.fc.weight.stride(-1) == 1


@pytest.mark.parametrize("env", [dict(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_DFLASH_MXFP4="0"), dict()])
def test_knob_off_leaves_everything(env):
    m = load(**env)
    d = make_drafter()
    assert m.process_dflash_draft(d) == 0
    for p in projections(d):
        assert p.weight.stride(-1) == 1
    assert d.model._fused_kv_weight.stride(-1) == 1


def test_standalone_knob_works_without_the_dense_knob():
    m = load(RADIANCE_DFLASH_MXFP4="1")
    d = make_drafter()
    ctx = torch.randn(3, H, dtype=torch.bfloat16)
    ref = d.model._project_context_kv_token_major(ctx, 3, LAYERS, 2, 16)
    assert m.process_dflash_draft(d) == 21
    assert cos(d.model._project_context_kv_token_major(ctx, 3, LAYERS, 2, 16), ref) > 0.99
    # the target-side hooks stay off: the dense knob is what gates them
    assert m._ready(d) is False


# ---------------------------------------------------------------- maybe_release_shared_head


class Stock:  # stands in for vLLM's LogitsProcessor
    pass


class Inkling(Stock):
    def _apply_head(self, lm_head, h, bias):
        return h


def head_and_draft(m, logits=Inkling, cand=Inkling, with_cand=True, free=False):
    head = ns(weight=torch.nn.Parameter(torch.randn(512, H, dtype=torch.bfloat16), requires_grad=False))
    m.register(head.weight, free=free)
    d = ns(logits_processor=logits(), lm_head=head)
    if with_cand:
        d.candidate_logits_processor = cand()
    return head, d


def test_release_frees_head_when_both_processors_use_the_mxfp4_head():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    head, d = head_and_draft(m)
    assert m.maybe_release_shared_head(d, head) is True
    assert freed(head.weight)
    h = torch.randn(3, H, dtype=torch.bfloat16)
    assert m.mxfp4_dense_head(head.weight, h, torch.float32).shape == (3, 512)  # still served from the tiles


@pytest.mark.parametrize("kw", [dict(logits=Stock), dict(cand=Stock)])
def test_release_refuses_a_stock_processor(kw):
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    head, d = head_and_draft(m, **kw)
    assert m.maybe_release_shared_head(d, head) is False
    assert head.weight.stride(-1) == 1 and head.weight.untyped_storage().nbytes() == 512 * H * 2


def test_release_without_candidate_processor_and_knob_off_and_unregistered():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    head, d = head_and_draft(m, with_cand=False)
    assert m.maybe_release_shared_head(d, head) is True
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_FREE_DFLASH_HEAD="0")
    head, d = head_and_draft(m)
    assert m.maybe_release_shared_head(d, head) is False and head.weight.stride(-1) == 1
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    head = ns(weight=torch.nn.Parameter(torch.randn(512, H, dtype=torch.bfloat16)))  # never registered
    assert m.maybe_release_shared_head(ns(logits_processor=Inkling(), lm_head=head), head) is False
    assert m.maybe_release_shared_head(ns(logits_processor=Inkling()), None) is False
