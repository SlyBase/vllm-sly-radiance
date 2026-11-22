"""CPU tests for the opt-in MXFP4 dense quantizer + routing (radiance_inkling2_mxfp4dense).

No GPU: the quantizer round-trips against an independent E2M1/E8M0 dequant (the same grid the
radiance_mxfp4_fp8 kernel reads), the routing gates are exercised, and the W4A16 fake-quant path
runs on plain torch. The W4A8 kernel itself is verified on the R9700 against _exact_ref.
"""
import importlib
import os
import sys
import types

import pytest
import torch

# Import the module under test fresh per test so env knobs (ENABLED/MODE) take effect.
MODNAME = "radiance_inkling2_mxfp4dense"


def load(**env):
    for k in ("RADIANCE_INKLING2_MXFP4_DENSE", "RADIANCE_INKLING2_MXFP4_DENSE_MODE",
              "RADIANCE_INKLING2_MXFP4_DENSE_CLIP", "RADIANCE_INKLING2_MXFP4_DENSE_MIN_N"):
        os.environ.pop(k, None)
    os.environ.update(env)
    sys.modules.pop(MODNAME, None)
    return importlib.import_module(MODNAME)


# Independent ground-truth dequant: packed [N,K/2] (low nibble = even k, bit3 = sign) +
# e8m0 [K/32,N] -> fp32 [N,K]. Written from the format spec, NOT from the module under test.
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def ref_dequant(packed, scale_kn):
    n, kh = packed.shape
    k = kh * 2
    c = torch.stack([packed & 0x0F, (packed >> 4) & 0x0F], -1).reshape(n, k).long()
    mag = torch.tensor(E2M1)[c & 7]
    val = torch.where(c >= 8, -mag, mag)
    sc = torch.exp2(scale_kn.float() - 127.0).T.repeat_interleave(32, dim=1)  # [N,K]
    return val * sc


def test_dequant_matches_independent_reference():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16)
    packed, scale, wref = m.quantize_mxfp4(w)
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == (64, 128)
    assert scale.dtype == torch.uint8 and tuple(scale.shape) == (256 // 32, 64)
    assert tuple(wref.shape) == (64,)
    got = m.dequant_mxfp4(packed, scale).float()
    exp = ref_dequant(packed, scale)
    assert torch.equal(got, exp), "module dequant must equal the format-spec dequant bit-exact"


def test_roundtrip_error_in_mxfp4_range():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    torch.manual_seed(1)
    w = torch.randn(128, 512, dtype=torch.bfloat16)
    packed, scale, _ = m.quantize_mxfp4(w)
    dq = m.dequant_mxfp4(packed, scale).float()
    rel = (dq - w.float()).norm() / w.float().norm()
    assert 0.05 < rel < 0.25, f"MXFP4 relRMS out of expected range: {rel}"


def test_low_nibble_is_even_k():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    # A single group whose first two weights quantise to distinct known codes.
    w = torch.zeros(1, 32, dtype=torch.bfloat16)
    w[0, 0] = 1.0   # even k -> low nibble
    w[0, 1] = -2.0  # odd k  -> high nibble
    packed, scale, _ = m.quantize_mxfp4(w)
    b = int(packed[0, 0])
    lo, hi = b & 0x0F, (b >> 4) & 0x0F
    # scale for this block: amax=2 -> e=floor(log2 2)-2=-1 -> s=0.5; 1/0.5=2 -> idx4; -2/0.5=-4 -> idx6 sign
    assert lo == 4, f"even-k code {lo} != 4 (value 2.0)"
    assert hi == (6 | 8), f"odd-k code {hi} != 14 (value -4.0)"


def test_e8m0_scale_rule():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    w = torch.zeros(1, 32, dtype=torch.bfloat16)
    w[0, 0] = 5.0  # amax 5 -> floor(log2 5)=2 -> e=0 -> e8m0=127
    _, scale, wref = m.quantize_mxfp4(w)
    assert int(scale[0, 0]) == 127
    assert int(wref[0]) == 127


def test_zero_weight_is_finite():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    w = torch.zeros(8, 64, dtype=torch.bfloat16)
    packed, scale, wref = m.quantize_mxfp4(w)
    dq = m.dequant_mxfp4(packed, scale)
    assert torch.isfinite(dq).all()
    assert (dq == 0).all()
    assert (packed == 0).all()


def test_mse_clip_not_worse_than_rtn():
    m_rtn = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_CLIP="rtn")
    m_mse = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_CLIP="mse")
    torch.manual_seed(2)
    w = torch.randn(64, 256, dtype=torch.bfloat16)
    p1, s1, _ = m_rtn.quantize_mxfp4(w)
    p2, s2, _ = m_mse.quantize_mxfp4(w)
    e1 = (m_rtn.dequant_mxfp4(p1, s1).float() - w.float()).square().sum()
    e2 = (m_mse.dequant_mxfp4(p2, s2).float() - w.float()).square().sum()
    assert e2 <= e1 * 1.001, f"mse {e2} should not be worse than rtn {e1}"


def test_routing_disabled_returns_none():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="0")
    x = torch.randn(2, 256, dtype=torch.bfloat16)
    w = torch.randn(512, 256, dtype=torch.bfloat16)
    assert m.mxfp4_dense_linear(x, w) is None


def test_routing_gates():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1")
    w = torch.randn(512, 256, dtype=torch.bfloat16)
    # non-bf16 weight
    assert m.mxfp4_dense_linear(torch.randn(2, 256), w.float()) is None
    # K not a multiple of 64
    assert m.mxfp4_dense_linear(torch.randn(2, 200, dtype=torch.bfloat16),
                                torch.randn(512, 200, dtype=torch.bfloat16)) is None
    # tiny / router-class N excluded
    assert m.mxfp4_dense_linear(torch.randn(2, 256, dtype=torch.bfloat16),
                                torch.randn(128, 256, dtype=torch.bfloat16)) is None


def test_w4a16_fakequant_matches_quantized_matmul():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MODE="w4a16")
    torch.manual_seed(3)
    x = torch.randn(3, 256, dtype=torch.bfloat16)
    w = torch.randn(512, 256, dtype=torch.bfloat16)
    p = torch.nn.Parameter(w.clone(), requires_grad=False)
    m.register(p)
    y = m.mxfp4_dense_linear(x, p)
    assert y is not None
    packed, scale, _ = m.quantize_mxfp4(w)
    dq = m.dequant_mxfp4(packed, scale).to(torch.bfloat16)
    exp = torch.nn.functional.linear(x, dq)
    assert torch.allclose(y.float(), exp.float(), atol=1e-2, rtol=1e-2)


def test_w4a8_dispatch_uses_op_and_bias():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MODE="w4a8")
    calls = []

    def fake_op(x2, packed, scale, wref):
        calls.append((tuple(x2.shape), tuple(packed.shape), tuple(scale.shape)))
        # reference: dequant matmul in fp32
        dq = ref_dequant(packed, scale)
        return (x2.float() @ dq.T).to(torch.bfloat16)

    m._linear_op = fake_op
    torch.manual_seed(4)
    x = torch.randn(4, 256, dtype=torch.bfloat16)
    w = torch.randn(512, 256, dtype=torch.bfloat16)
    bias = torch.randn(512, dtype=torch.bfloat16)
    p = torch.nn.Parameter(w.clone(), requires_grad=False)
    m.register(p)
    y = m.mxfp4_dense_linear(x, p, bias=bias)
    assert y is not None and tuple(y.shape) == (4, 512)
    assert len(calls) == 1
    packed, scale, _ = m.quantize_mxfp4(w)
    exp = (x.float() @ ref_dequant(packed, scale).T + bias.float()).to(torch.bfloat16)
    assert torch.allclose(y.float(), exp.float(), atol=1e-2, rtol=1e-2)
    # second call reuses the registered copy: op called again, nothing re-quantized
    y2 = m.mxfp4_dense_linear(x, p, bias=bias)
    assert len(calls) == 2
    assert torch.equal(y, y2)


def test_head_routes_and_casts_dtype():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MODE="w4a16")
    torch.manual_seed(5)
    x = torch.randn(2, 256, dtype=torch.bfloat16)
    w = torch.nn.Parameter(torch.randn(2048, 256, dtype=torch.bfloat16), requires_grad=False)  # head class
    m.register(w, free=False)
    y = m.mxfp4_dense_head(w, x, torch.float32)
    assert y is not None and y.dtype == torch.float32 and tuple(y.shape) == (2, 2048)


# ---- load-time quantization + release of the bf16 originals --------------------------------------
# A fake module set shaped like the real one (Inkling2Attention: fused qkvr + wo_ud, the DSpark
# draft's five linears, the shared lm_head). Small sizes, MIN_N lowered to match.
# q-side 149 rows (+3 alignment pad, taken from the k rows), k 149:181, v 181:213, 3 pad rows at the end
K, SPLITS, QSIDE, QROWS = 128, [64, 16, 64, 5, 32, 32, 3], 149, 152


class FakeAttn(torch.nn.Module):
    def __init__(self, is_local):
        super().__init__()
        self.is_local = is_local
        self.split_sizes = SPLITS
        self.q_side_splits = [*SPLITS[:4], -QSIDE % 8]
        assert sum(self.q_side_splits) == QROWS
        self.qkvr = torch.nn.Linear(K, sum(SPLITS), bias=False, dtype=torch.bfloat16)
        self.wo_ud = torch.nn.Linear(192, 128, bias=False, dtype=torch.bfloat16)


class Processor:  # stands in for Inkling2LogitsProcessor: it defines _apply_head itself
    def _apply_head(self):
        pass


class FakeTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([FakeAttn(True), FakeAttn(False)])
        # bf16 2D and big enough to look quantizable by shape, but read by an embedding lookup
        self.embed_tokens = torch.nn.Embedding(512, K, dtype=torch.bfloat16)
        self.lm_head = torch.nn.Linear(K, 512, bias=False, dtype=torch.bfloat16)
        self.logits_processor = Processor()


def storage(p):
    return p.untyped_storage().nbytes()


def serving(**env):
    """Module with the W4A8 op replaced by the format-spec matmul (the kernel itself is GPU-only)."""
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MIN_N="32", **env)
    def op(x2, packed, scale, wref):
        # the kernel's operand contract: dense, scale [K/32, N], wref [N]
        assert packed.is_contiguous() and scale.is_contiguous() and wref.is_contiguous()
        assert packed.data_ptr() % 16 == 0 and scale.data_ptr() % 16 == 0 and wref.data_ptr() % 16 == 0
        assert tuple(scale.shape) == (packed.shape[1] * 2 // 32, packed.shape[0]) and wref.shape == (packed.shape[0],)
        return (x2.float() @ ref_dequant(packed, scale).T).to(torch.bfloat16)

    m._linear_op = op
    return m


def expected(m, x, w):
    """x @ (MXFP4 round trip of w).T -- the pre-change routing, computed from the ORIGINAL bf16."""
    packed, scale, _ = m.quantize_mxfp4(w.clone())
    return (x.float() @ ref_dequant(packed, scale).T).to(torch.bfloat16)


def test_target_release_frees_bf16_and_routes_like_before():
    m = serving()
    torch.manual_seed(10)
    model = FakeTarget()
    orig = {n: p.data.clone() for n, p in model.named_parameters()}
    before = {n: storage(p) for n, p in model.named_parameters()}
    m.process_target(model)
    x = torch.randn(3, K, dtype=torch.bfloat16)
    for i, layer in enumerate(model.layers):
        w = layer.qkvr.weight
        assert w.dtype == torch.bfloat16 and tuple(w.shape) == (216, K)          # shape kept
        assert storage(w) <= before[f"layers.{i}.qkvr.weight"] // 32              # bf16 released
        assert torch.isnan(w.float()).all()                                       # a stray bf16 read is loud
        assert torch.equal(m.mxfp4_dense_linear(x, w), expected(m, x, orig[f"layers.{i}.qkvr.weight"]))
        # the row slices the model's _project() takes (loop passes / full layers)
        o = orig[f"layers.{i}.qkvr.weight"]
        for r0, n in ((0, QROWS), (QSIDE, 32), (QSIDE + 32, 32)):
            assert torch.equal(m.mxfp4_dense_linear(x, w[r0:r0 + n]), expected(m, x, o[r0:r0 + n]))
        xo = torch.randn(40, 192, dtype=torch.bfloat16)                          # M > 32: `out` form
        out = torch.empty(40, 128, dtype=torch.bfloat16)
        got = m.mxfp4_dense_linear(xo, layer.wo_ud.weight, out)
        assert got is out
        assert torch.equal(out, expected(m, xo, orig[f"layers.{i}.wo_ud.weight"]))
    # not read through linear(): untouched
    assert torch.equal(model.embed_tokens.weight.data, orig["embed_tokens.weight"])
    assert storage(model.embed_tokens.weight) == before["embed_tokens.weight"]


def test_head_stays_bf16_for_the_drafter_fold_then_release_head_frees_it():
    m = serving()
    torch.manual_seed(11)
    model = FakeTarget()
    head = model.lm_head.weight
    w0, nbytes = head.data.clone(), storage(head)
    m.process_target(model)
    assert storage(head) == nbytes and torch.equal(head.data, w0)     # fold (bf16 fp64 reads) still works
    x = torch.randn(2, K, dtype=torch.bfloat16)
    y = m.mxfp4_dense_head(head, x, torch.float32)                    # target _apply_head + draft base logits
    assert torch.equal(y, expected(m, x, w0).float())
    m.release_head(head)
    assert storage(head) <= nbytes // 32 and tuple(head.shape) == (512, K)
    assert torch.equal(m.mxfp4_dense_head(head, x, torch.float32), y)  # same routing after the release


def test_draft_release_only_the_linears_the_draft_passes_to_linear():
    m = serving()
    torch.manual_seed(12)
    draft = torch.nn.Module()
    draft.layer = layer = torch.nn.Module()
    layer.qkv_proj = torch.nn.Linear(K, 256, bias=False, dtype=torch.bfloat16)
    layer.kv_proj_ctx = torch.nn.Linear(K, 64, bias=False, dtype=torch.bfloat16)
    layer.o_proj = torch.nn.Linear(192, 128, bias=False, dtype=torch.bfloat16)
    layer.gate_up_proj = torch.nn.Linear(K, 512, bias=False, dtype=torch.bfloat16)
    layer.down_proj = torch.nn.Linear(256, 128, bias=False, dtype=torch.bfloat16)
    # eligible by shape (N>=MIN_N, K%64==0) but read directly: fold() does hidden_out.weight.double()
    draft.hidden_out = torch.nn.Linear(64, 128, bias=True, dtype=torch.bfloat16)
    orig = {n: p.data.clone() for n, p in draft.named_parameters()}
    m.process_draft(draft)
    for name in ("qkv_proj", "kv_proj_ctx", "o_proj", "gate_up_proj", "down_proj"):
        w, o = getattr(layer, name).weight, orig[f"layer.{name}.weight"]
        x = torch.randn(4, o.shape[1], dtype=torch.bfloat16)
        assert storage(w) <= o.numel() * 2 // 32
        assert torch.equal(m.mxfp4_dense_linear(x, w), expected(m, x, o))
    assert torch.equal(draft.hidden_out.weight.data, orig["hidden_out.weight"])
    assert storage(draft.hidden_out.weight) == orig["hidden_out.weight"].numel() * 2


def test_released_weight_without_entry_fails_loudly_unknown_weight_falls_through():
    m = serving()
    p = torch.nn.Parameter(torch.randn(128, K, dtype=torch.bfloat16), requires_grad=False)
    x = torch.randn(2, K, dtype=torch.bfloat16)
    assert m.mxfp4_dense_linear(x, p) is None                    # never registered, bf16 intact: bf16 path
    m.register(p)
    with pytest.raises(RuntimeError, match="no MXFP4 entry"):
        m.mxfp4_dense_linear(x, p.data[:32])                     # a slice the load hook did not register


def test_zero_rows_do_not_reach_the_kernel():
    m = serving()
    p = torch.nn.Parameter(torch.randn(128, K, dtype=torch.bfloat16), requires_grad=False)
    m.register(p)
    m._linear_op = lambda *a: pytest.fail("kernel launched with M=0")
    y = m.mxfp4_dense_linear(torch.empty(0, K, dtype=torch.bfloat16), p)
    assert tuple(y.shape) == (0, 128)
    out = torch.empty(0, 128, dtype=torch.bfloat16)
    assert m.mxfp4_dense_linear(torch.empty(0, K, dtype=torch.bfloat16), p, out) is out


def test_knob_off_touches_nothing():
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="0", RADIANCE_INKLING2_MXFP4_DENSE_MIN_N="32")
    model = FakeTarget()
    before = {n: (p.data_ptr(), storage(p)) for n, p in model.named_parameters()}
    m.process_target(model)
    m.release_head(model.lm_head.weight)
    assert {n: (p.data_ptr(), storage(p)) for n, p in model.named_parameters()} == before
    assert not m._CACHE


def test_kernel_unavailable_keeps_every_bf16_weight():
    m = serving()
    m._linear_op = False                                         # what _resolve_op leaves on a failed import
    model = FakeTarget()
    before = {n: p.data_ptr() for n, p in model.named_parameters()}
    m.process_target(model)
    assert {n: p.data_ptr() for n, p in model.named_parameters()} == before and not m._CACHE


def test_min_m_above_zero_means_aiter_would_get_decode_so_nothing_is_quantized(monkeypatch, capsys):
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MIN_N="32")
    r = types.ModuleType("radiance_mxfp4")
    r.MIN_M = 256                                                # the module's default: M <= 256 goes to aiter
    monkeypatch.setitem(sys.modules, "radiance_mxfp4", r)
    monkeypatch.setitem(sys.modules, "radiance_mxfp4_fp8", types.ModuleType("radiance_mxfp4_fp8"))
    assert m._resolve_op() is None
    assert "MIN_M" in capsys.readouterr().err


def test_lora_logits_processor_keeps_the_bf16_head():
    m = serving()
    model = FakeTarget()
    model.logits_processor = object()                            # stock LogitsProcessor: no hooked _apply_head
    head = model.lm_head.weight
    m.process_target(model)
    assert m._key(head) not in m._CACHE and storage(head) == 512 * K * 2
    assert m.mxfp4_dense_head(head, torch.randn(2, K, dtype=torch.bfloat16), torch.float32) is None
    m.release_head(head)                                         # not registered: stays bf16
    assert storage(head) == 512 * K * 2


def test_load_hook_is_idempotent():
    m = serving()
    model = FakeTarget()
    m.process_target(model)
    entries = dict(m._CACHE)
    m.process_target(model)                                      # second finalize pass: stubs are skipped
    assert m._CACHE.keys() == entries.keys() and all(m._CACHE[k] is e for k, e in entries.items())


def test_decode_scratch_is_allocated_at_load_not_on_first_call():
    m = serving()
    m.DECODE_MAX_M = 16
    calls = []
    m._ext = types.SimpleNamespace(set_decode_scratch=lambda *a: calls.append(a))
    model = FakeTarget()
    m.process_target(model)
    assert len(calls) == 1
    m.process_target(model)
    assert len(calls) == 1                                       # once per process


def test_w4a16_builds_the_dequantized_copy_at_load():
    m = serving(RADIANCE_INKLING2_MXFP4_DENSE_MODE="w4a16")
    model = FakeTarget()
    m.process_target(model)
    assert all(e["dq"] is not None for e in m._CACHE.values())
