"""CPU: the chunk_scan math (exact and the kernel's split-decay form) against the naive token-by-token
recurrence, all fp64 (tolerance relRMSE <= 1e-10).  Runs anywhere with torch-cpu.

The v2 variant must not change a single arithmetic step of the kernel, so this reference is the
contract both entry points share; the GPU digest test (sly/gdn/tests/gpu_chunk_scan_v2.py) uses the
same module as its oracle.
"""
import pathlib, sys
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import torch
from chunk_scan_ref import make_inputs, naive, chunked, rel_rmse

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
