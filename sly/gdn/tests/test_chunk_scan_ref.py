"""CPU: the chunk_scan math (exact and the kernel's split-decay form) against the naive token-by-token
recurrence, all fp64 (tolerance relRMSE <= 1e-10).  Runs anywhere with torch-cpu.

The v2 variant must not change a single arithmetic step of the kernel, so this reference is the
contract both entry points share; the GPU digest test (sly/gdn/tests/gpu_chunk_scan_variants.py) uses the
same module as its oracle.
"""
import pathlib, sys
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from chunk_scan_ref import make_inputs, naive, chunked, chunk_cumsum, rel_rmse, BT, SPLIT_MAX_SPAN

CASES = [([1], 3.0), ([63], 3.0), ([64], 3.0), ([65], 3.0), ([200], 3.0), ([65, 130], 3.0), ([200], 120.0)]


@pytest.mark.parametrize("T_list,span", CASES)
def test_chunked_exact_equals_naive(T_list, span):
    inp = make_inputs(T_list, span=span)
    o0, s0 = naive(inp); o1, s1 = chunked(inp)
    assert rel_rmse(o1, o0) <= 1e-10 and rel_rmse(s1, s0) <= 1e-10


@pytest.mark.parametrize("T_list,span", CASES)
def test_split_decay_equals_naive_while_clamp_idle(T_list, span):
    # per-chunk span 120 -> each half <= e^60 < e^80: the clamp is idle, the factorisation is exact
    inp = make_inputs(T_list, span=span)
    o0, s0 = naive(inp); o1, s1 = chunked(inp, split=True)
    assert rel_rmse(o1, o0) <= 1e-10 and rel_rmse(s1, s0) <= 1e-10


def test_split_decay_clamp_path_stays_finite():
    # span 200: halves reach e^100 > e^80, the clamp attenuates (documented, scan:470-490); no NaN/Inf
    inp = make_inputs([200], span=200.0)
    o1, s1 = chunked(inp, split=True)
    assert torch.isfinite(o1).all() and torch.isfinite(s1).all()


# r4d_gdn_exact_decay.patch (libr4d issue #4): a chunk whose gate span g_0 - g_last exceeds 160 takes the exact
# form; every other chunk keeps the split form. span 160 mixes both kinds of chunk in one sequence.
LARGE = [([200], 160.0), ([200], 200.0), ([65, 130], 400.0), ([200], 1000.0)]


def _spans(inp):
    g, cu = chunk_cumsum(inp["glog"], inp["cu"]), inp["cu"]
    return [float(g[c0, h] - g[min(c0 + BT, cu[n + 1]) - 1, h])
            for n in range(len(cu) - 1) for c0 in range(cu[n], cu[n + 1], BT) for h in range(g.shape[1])]


@pytest.mark.parametrize("T_list,span", CASES + LARGE)
def test_hybrid_equals_naive(T_list, span):
    inp = make_inputs(T_list, span=span)
    o0, s0 = naive(inp); o1, s1 = chunked(inp, split="hybrid")
    assert rel_rmse(o1, o0) <= 1e-10 and rel_rmse(s1, s0) <= 1e-10


@pytest.mark.parametrize("T_list,span", CASES)
def test_hybrid_is_the_split_form_below_the_threshold(T_list, span):
    # every chunk <= 160: the patched kernel runs the unpatched arithmetic, so the reference must be identical
    inp = make_inputs(T_list, span=span)
    assert max(_spans(inp)) <= SPLIT_MAX_SPAN
    o0, s0 = chunked(inp, split=True); o1, s1 = chunked(inp, split="hybrid")
    assert torch.equal(o1, o0) and torch.equal(s1, s0)


def test_mixed_case_has_both_kinds_of_chunk():
    sp = _spans(make_inputs([200], span=160.0))
    assert min(sp) <= SPLIT_MAX_SPAN < max(sp)


@pytest.mark.parametrize("T_list,span", LARGE)
def test_split_decay_is_wrong_above_the_threshold(T_list, span):
    # what the patch fixes: the clamped split form attenuates these outputs by 7-87 % relRMSE
    inp = make_inputs(T_list, span=span)
    o0, _ = naive(inp); o1, _ = chunked(inp, split=True)
    assert rel_rmse(o1, o0) > 0.05
