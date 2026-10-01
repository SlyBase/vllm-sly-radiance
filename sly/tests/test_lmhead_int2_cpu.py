#!/usr/bin/env python3
"""CPU tests for the int2 two-stage greedy lm_head (sly/mxfp4/radiance_lmhead_int2.py).

Plain torch only: no vllm, no GPU, no checkpoint -- the module's guarded imports degrade
to a logging fallback outside the serving image, which is exactly what these tests use.
The GPU window reruns the same file on the image (plus sly/check_lmhead_int2.py on the
real weight).

Run:  python3 sly/tests/test_lmhead_int2_cpu.py     (exit 0 = pass)
pytest also picks up the test_* functions unchanged.
"""

import importlib
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mxfp4"))

# knob hygiene: the module reads RADIANCE_LMHEAD_INT2* at import time; the tests below
# reload it under a controlled environment
for _k in ("RADIANCE_LMHEAD_INT2", "RADIANCE_LMHEAD_INT2_GS", "RADIANCE_LMHEAD_INT2_TOPK",
           "RADIANCE_LMHEAD_INT2_LAYOUT"):
    os.environ.pop(_k, None)

import radiance_lmhead_int2 as I2  # noqa: E402


class FakeLayer(torch.nn.Module):
    """Just enough ParallelLMHead surface for the quant-method class path."""

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)


def test_pack_unpack_roundtrip():
    for layout in ("row", "quarter"):
        for rows, k in ((17, 5120), (64, 1024), (1, 128)):
            codes = torch.randint(0, 4, (rows, k), dtype=torch.uint8)
            packed = I2.pack_int2_rows(codes, layout)
            assert packed.dtype == torch.uint8 and packed.shape == (rows, k // 4)
            back = I2.unpack_int2_rows(packed, layout)
            assert back.dtype == torch.uint8 and back.shape == codes.shape
            assert torch.equal(back, codes), f"{layout} {rows}x{k} roundtrip failed"
    # non-multiple-of-4 hidden must be rejected loudly, not silently mis-packed
    try:
        I2.pack_int2_rows(torch.randint(0, 4, (2, 254), dtype=torch.uint8), "row")
    except ValueError:
        pass
    else:
        raise AssertionError("k=254 must raise")
    try:
        I2.pack_int2_rows(torch.zeros((2, 8), dtype=torch.uint8), "bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown layout must raise")


def test_group_scale_bounds():
    torch.manual_seed(1)
    w = (torch.randn(128, 5120) * 0.02).to(torch.bfloat16)
    packed, scales = I2.quantize_int2_rows(w, 128)
    assert packed.shape == (128, 1280) and packed.dtype == torch.uint8
    assert scales.shape == (128, 40) and scales.dtype == torch.bfloat16
    sf = scales.float()
    # symmetric absmax: the bf16-rounded scale must sit within one bf16 ulp of the group
    # amax it was computed from
    amax = w.float().reshape(128, 40, 128).abs().amax(dim=-1)
    lo, hi = amax * (1 - 2**-8), amax * (1 + 2**-8)
    assert bool((sf >= lo).all()) and bool((sf <= hi).all()), "scale drifted from amax"
    # dequantisation error: |w - deq| <= s/2 (rounding) + amax*2^-8 (bf16 scale slack)
    deq = I2.dequant_int2_rows(packed, scales, 128)
    s_exp = sf.repeat_interleave(128, dim=1)
    err = (w.float() - deq).abs()
    assert bool((err <= s_exp * 0.51 + 1e-12).all()), "dequant error exceeded the bound"
    # representable range: |q| <= 2 with s ~= amax bounds the dequantised magnitude
    assert bool((deq.abs() <= amax.repeat_interleave(128, dim=1) * 2.01).all())
    # all-zero (padded vocab) rows: scale floored, codes quantise to exactly zero
    w0 = torch.zeros(3, 5120, dtype=torch.bfloat16)
    p0, s0 = I2.quantize_int2_rows(w0, 128)
    assert bool((s0.float() <= 2e-8).all())
    assert bool(torch.equal(I2.unpack_int2_rows(p0, "row"),
                            torch.full((3, 5120), 2, dtype=torch.uint8)))
    assert bool((I2.dequant_int2_rows(p0, s0, 128) == 0).all())


def test_quantize_is_row_separable():
    """Chunked quantisation (what process_weights_after_loading does) must be identical
    to per-row quantisation -- no state may leak across the chunk boundary."""
    torch.manual_seed(2)
    w = (torch.randn(300, 1024) * 0.02).to(torch.bfloat16)
    packed_all, scales_all = I2.quantize_int2_rows(w, 128)
    packed_chunks, scales_chunks = torch.empty_like(packed_all), torch.empty_like(scales_all)
    for i in range(0, 300, 97):
        j = min(i + 97, 300)
        packed_chunks[i:j], scales_chunks[i:j] = I2.quantize_int2_rows(w[i:j], 128)
    assert torch.equal(packed_all, packed_chunks)
    assert torch.equal(scales_all, scales_chunks)


def test_coarse_matches_dequant_matmul():
    torch.manual_seed(3)
    w = (torch.randn(700, 2048) * 0.02).to(torch.bfloat16)
    packed, scales = I2.quantize_int2_rows(w, 128)
    x = torch.randn(16, 2048).to(torch.bfloat16)
    coarse = I2.coarse_logits(x, packed, scales, 128, chunk_rows=256)
    deq = I2.dequant_int2_rows(packed, scales, 128)
    ref = x.float() @ deq.t()
    assert coarse.shape == (16, 700) and coarse.dtype == torch.float32
    assert torch.allclose(coarse, ref, atol=1e-4, rtol=1e-5), \
        "chunked stage 1 drifted from the plain dequant-matmul"
    # a mismatched hidden size must be rejected, not silently mis-read
    try:
        I2.coarse_logits(torch.zeros(1, 2044), packed, scales, 128)
    except ValueError:
        pass
    else:
        raise AssertionError("K mismatch must raise")


def _dominant_case(n, k, seed=4):
    """A row-aligned x whose exact winner is r*: E[r*] ~ 3*||w||^2 towers over the
    ~N(0, 3*sigma^2*sqrt(k)) cross terms of the other rows, so the exact fp32 argmax --
    and hence the coarse top-16 membership -- is guaranteed by construction, not luck."""
    torch.manual_seed(seed)
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16)
    r_star = 7
    x = (3.0 * w[r_star].float()).to(torch.bfloat16).unsqueeze(0)
    return w, x, r_star


def test_two_stage_oracle():
    for k in (5120, 1024):
        w, x, r_star = _dominant_case(512, k)
        packed, scales = I2.quantize_int2_rows(w, 128)
        ref = x.float() @ w.float().t()
        assert ref.argmax(dim=1).item() == r_star, "construction sanity: exact winner"
        mixture, cand_idx, cand = I2.greedy_two_stage(x, packed, scales, 128, 16, (w, None))
        assert cand_idx.shape == (1, 16)
        assert bool((cand_idx == r_star).any()), "guaranteed member was not in the pool"
        # the rerank-oracle property: argmax over the exactly-scored candidates
        # (cand.argmax is a position in the pool -- map it back to the token id)
        two_am = cand_idx.gather(1, cand.argmax(dim=1, keepdim=True)).squeeze(1)
        assert two_am.item() == r_star, "two-stage argmax != exact argmax"
        assert mixture.argmax(dim=1).item() == r_star, "mixture argmax != decision"
    # conditional oracle on random inputs: every row whose exact top-1 IS covered by the
    # coarse pool must come out of the two-stage head as the exact winner
    torch.manual_seed(5)
    w = (torch.randn(4096, 2048) * 0.02).to(torch.bfloat16)
    packed, scales = I2.quantize_int2_rows(w, 128)
    x = torch.randn(64, 2048).to(torch.bfloat16)
    ref = x.float() @ w.float().t()
    exact_am = ref.argmax(dim=1)
    _, cand_idx, cand = I2.greedy_two_stage(x, packed, scales, 128, 16, (w, None))
    covered = (cand_idx == exact_am.unsqueeze(1)).any(dim=1)
    assert covered.sum().item() > 0, "degenerate: nothing covered"
    two_am = cand_idx.gather(1, cand.argmax(dim=1, keepdim=True)).squeeze(1)
    assert bool((two_am[covered] == exact_am[covered]).all()), \
        "a covered row disagreed with the exact argmax"


def test_two_stage_oracle_with_bias():
    w, x, r_star = _dominant_case(512, 2048)
    packed, scales = I2.quantize_int2_rows(w, 128)
    torch.manual_seed(6)
    bias = (torch.randn(512) * 0.01).float()
    ref = x.float() @ w.float().t() + bias
    assert ref.argmax(dim=1).item() == r_star, "construction sanity with bias"
    mixture, cand_idx, cand = I2.greedy_two_stage(x, packed, scales, 128, 16, (w, None),
                                                  bias=bias)
    two_am = cand_idx.gather(1, cand.argmax(dim=1, keepdim=True)).squeeze(1)
    assert two_am.item() == r_star
    assert mixture.argmax(dim=1).item() == r_star
    # the bias must be folded in, not double-added: rerank scores equal exact+bias
    assert torch.allclose(cand[0], ref[0, cand_idx[0]], atol=1e-5, rtol=1e-5)


def test_mixture_encoding():
    torch.manual_seed(7)
    w = (torch.randn(2048, 1024) * 0.02).to(torch.bfloat16)
    packed, scales = I2.quantize_int2_rows(w, 128)
    x = torch.randn(8, 1024).to(torch.bfloat16)
    mixture, cand_idx, cand = I2.greedy_two_stage(x, packed, scales, 128, 16, (w, None))
    # decision preserved: the clamped row's argmax is exactly the rerank decision
    two_am = cand_idx.gather(1, cand.argmax(dim=1, keepdim=True)).squeeze(1)
    assert torch.equal(mixture.argmax(dim=1), two_am)
    # candidate positions hold the exact scores, everything else sits at/below the
    # candidate minimum, so no unscored row can ever be promoted
    assert torch.equal(mixture.gather(1, cand_idx), cand)
    floor = cand.min(dim=1, keepdim=True).values
    mask = torch.zeros_like(mixture, dtype=torch.bool)
    mask.scatter_(1, cand_idx, True)
    assert bool((mixture.masked_fill(mask, float("-inf")) <= floor).all())
    # topk larger than the vocab clamps instead of crashing. A full pool scores
    # every row exactly, so its decision IS the exact fp32 argmax; the 16-pool
    # decision must agree on every row it covered (and only claims those rows).
    mixture2, cand_idx2, cand2 = I2.greedy_two_stage(x, packed, scales, 128, 99999,
                                                     (w, None))
    assert cand_idx2.shape == (8, 2048)
    two_full = cand_idx2.gather(1, cand2.argmax(dim=1, keepdim=True)).squeeze(1)
    exact_am = (x.float() @ w.float().t()).argmax(dim=1)
    assert torch.equal(two_full, exact_am), "full pool must decide the exact argmax"
    assert torch.equal(mixture2.argmax(dim=1), two_full)
    covered = (cand_idx == exact_am.unsqueeze(1)).any(dim=1)
    assert bool((two_am[covered] == two_full[covered]).all()), \
        "a covered row must make the same decision at any pool size"


def test_class_process_and_apply():
    torch.manual_seed(8)
    n, k = 9000, 5120  # n > CHUNK_ROWS: exercises the chunk-boundary in process
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16)
    layer = FakeLayer(w.clone())
    method = I2.RadianceLMHeadInt2()
    method.process_weights_after_loading(layer)
    # the exact rerank source stays loaded and untouched
    assert layer.weight.dtype == torch.bfloat16 and torch.equal(layer.weight, w)
    assert layer.weight_int2.shape == (n, k // 4)
    assert layer.weight_int2.dtype == torch.uint8
    assert layer.weight_int2_scale.shape == (n, k // 128)
    assert layer.weight_int2_scale.dtype == torch.bfloat16
    # process is idempotent (a second pass over shared modules must not re-pack)
    packed_ref = layer.weight_int2.clone()
    method.process_weights_after_loading(layer)
    assert torch.equal(layer.weight_int2, packed_ref)
    # packed rows are identical to a direct quantisation of the same rows
    p_direct, s_direct = I2.quantize_int2_rows(w[8191:8193], 128)
    assert torch.equal(layer.weight_int2[8191:8193], p_direct)
    assert torch.equal(layer.weight_int2_scale[8191:8193], s_direct)
    # apply(): same mixture as the function path, right dtype/shape on a 3-D input
    x = torch.randn(2, 3, k).to(torch.bfloat16)
    out = method.apply(layer, x)
    assert out.shape == (2, 3, n) and out.dtype == torch.bfloat16
    fn_mix, _, _ = I2.greedy_two_stage(x.reshape(-1, k), layer.weight_int2,
                                       layer.weight_int2_scale, 128, 16,
                                       (layer.weight, None))
    assert torch.equal(out.reshape(-1, n), fn_mix.to(torch.bfloat16))
    # the dominant-row oracle holds end to end through the class path
    wd, xd, r_star = _dominant_case(512, 5120)
    layer_d = FakeLayer(wd.clone())
    method.process_weights_after_loading(layer_d)
    out = method.apply(layer_d, xd)
    assert out.argmax(dim=-1).item() == r_star


def test_sampling_guard():
    """Under sampling the head must hard-disable: apply() serves the exact full-width
    row (a valid distribution) instead of the clamped two-stage row."""
    torch.manual_seed(9)
    w = (torch.randn(512, 2048) * 0.02).to(torch.bfloat16)
    layer = FakeLayer(w.clone())
    method = I2.RadianceLMHeadInt2()
    method.process_weights_after_loading(layer)
    x = torch.randn(8, 2048).to(torch.bfloat16)
    I2.note_sampling()
    try:
        out = method.apply(layer, x)
        stock = torch.matmul(x, layer.weight.t())
        assert out.dtype == torch.bfloat16 and torch.equal(out, stock), \
            "sampling fallback must be the stock exact row, bit for bit"
        # the full row is valid to sample from: every entry is an approximately exact logit
        ref = x.float() @ w.float().t()
        assert torch.allclose(out.float(), ref, atol=0.05, rtol=0.05)
    finally:
        I2.reset_sampling()
    # and with the latch cleared the two-stage path is back
    out = method.apply(layer, x)
    assert not torch.equal(out, torch.matmul(x, layer.weight.t()))


def test_fp8_source():
    """The fp8-composed head: the coarse copy is built from the dequantised fp8 rows and
    the rerank scores (w8, per-row scale) exactly -- the same source the fp8 head loads."""
    try:
        probe = torch.zeros(4, 4).to(torch.float8_e4m3fn).float()
        del probe
    except Exception as exc:  # torch builds without CPU fp8 casts
        print(f"  [skip] fp8 CPU cast unavailable ({exc!r}); covered on the GPU image")
        return
    torch.manual_seed(10)
    w = (torch.randn(512, 5120) * 0.02)
    scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    w8 = w.div(scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    deq = w8.float() * scale  # what the fp8 head actually serves
    layer = FakeLayer(w8.clone())
    layer.weight_scale = torch.nn.Parameter(scale.view(1, -1), requires_grad=False)
    method = I2.RadianceLMHeadInt2()
    method.process_weights_after_loading(layer)
    r_star = 7
    x = (3.0 * deq[r_star]).to(torch.bfloat16).unsqueeze(0)
    ref = x.float() @ deq.t()
    assert ref.argmax(dim=1).item() == r_star
    src = I2._src_rows(layer)
    assert src is not None and src[1] is not None and torch.allclose(src[1], scale.view(-1))
    mixture, cand_idx, cand = I2.greedy_two_stage(x, layer.weight_int2,
                                                  layer.weight_int2_scale, 128, 16, src)
    assert bool((cand_idx == r_star).any())
    two_am = cand_idx.gather(1, cand.argmax(dim=1, keepdim=True)).squeeze(1)
    assert two_am.item() == r_star
    assert mixture.argmax(dim=1).item() == r_star


def test_knob_defaults_and_cpu_hook():
    assert I2.ENABLED is False, "must be disabled unless RADIANCE_LMHEAD_INT2=1"
    assert I2.GROUP_SIZE == 128 and I2.TOPK == 16 and I2.LAYOUT == "row"
    os.environ["RADIANCE_LMHEAD_INT2"] = "1"
    os.environ["RADIANCE_LMHEAD_INT2_TOPK"] = "32"
    mod = importlib.reload(I2)
    assert mod.ENABLED is True and mod.TOPK == 32
    os.environ.pop("RADIANCE_LMHEAD_INT2", None)
    os.environ.pop("RADIANCE_LMHEAD_INT2_TOPK", None)
    mod = importlib.reload(I2)
    assert mod.ENABLED is False and mod.TOPK == 16
    # outside the image (no vllm) the hook must decline: nothing to wire into
    assert I2.quant_method_for(FakeLayer(torch.zeros(4, 4)), "lm_head") is None
    assert I2._HAVE_VLLM is False, "this box unexpectedly has vllm; flip this test"


def main():
    tests = [
        test_pack_unpack_roundtrip,
        test_group_scale_bounds,
        test_quantize_is_row_separable,
        test_coarse_matches_dequant_matmul,
        test_two_stage_oracle,
        test_two_stage_oracle_with_bias,
        test_mixture_encoding,
        test_class_process_and_apply,
        test_sampling_guard,
        test_fp8_source,
        test_knob_defaults_and_cpu_hook,
    ]
    for t in tests:
        print(f"[run] {t.__name__}")
        t()
        I2.reset_sampling()
    print("DONE: all int2 lm_head CPU tests passed")


if __name__ == "__main__":
    main()
