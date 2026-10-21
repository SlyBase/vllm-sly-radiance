#!/usr/bin/env python3
"""CPU tests for RADIANCE_GDN_GATE_INPLACE: the chunk scan's initial state without the extra copies.

vLLM hands the prefill step a paged mamba state. The stock glue materializes the scan's h0 with two
kernels plus a fill on every (GDN layer, prefill chunk):

    h0 = ssm_state[indices].float()          # gather (bf16 -> bf16 copy) + widen (second copy)
    h0[~has_initial_state, ...] = 0          # masked_fill over the whole tensor
    48 layers x 12 chunks at 47k tokens = 3 launches x 576 per prefill

`RADIANCE_GDN_GATE_INPLACE=1` takes the two shortcuts that produce the same bytes with fewer
kernels: a row that carries no state is zero by definition (a shared scratch buffer instead of a
gather), and a batch where every row carries a state needs no fill at all. Values are unchanged by
construction -- the tests below compare against the stock expression bit for bit.

Plain torch, no vllm, no GPU: radiance_gdn imports without libr4d and stays inert (ENABLED=False).

Run:  python3 sly/tests/test_gdn_gate_inplace_cpu.py     (exit 0 = pass)
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import radiance_gdn as G  # noqa: E402


def _stock(ssm_state, indices, mask):
    """The expression the knob replaces, verbatim."""
    h0 = ssm_state[indices].float()
    h0[~mask, ...] = 0
    return h0


def _state(blocks=3, h=2, v=4, k=4, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(blocks, h, v, k, generator=g) * 3).to(dtype)


CASES = {
    "all_missing": torch.tensor([False, False]),
    "all_present": torch.tensor([True, True]),
    "mixed": torch.tensor([False, True]),
}


def _with_knob(on):
    old = G.GATE_INPLACE
    G.GATE_INPLACE = on
    G._H0_ZEROS.clear()
    return old


def test_states_are_bit_identical_to_the_stock_expression():
    for dtype in (torch.bfloat16, torch.float32):
        st = _state(dtype=dtype)
        idx = torch.tensor([0, 2])
        for name, mask in CASES.items():
            want = _stock(st, idx, mask)
            old = _with_knob(True)
            try:
                got = G._prefill_h0(st, idx, mask, G._h0_mode(SimpleNamespace(prefill_has_initial_state=mask)))
            finally:
                G.GATE_INPLACE = old
                G._H0_ZEROS.clear()
            assert got.dtype == torch.float32, (name, dtype, got.dtype)
            assert torch.equal(got, want), f"{name} / {dtype}: knob output differs from stock"


def test_knob_picks_the_cheapest_path_that_is_equivalent():
    for name, mask in CASES.items():
        md = SimpleNamespace(prefill_has_initial_state=mask)   # fresh per step, as in a serve
        old = _with_knob(False)
        try:
            off = G._h0_mode(md)
        finally:
            G.GATE_INPLACE = old
        assert off == "gather", name
        old = _with_knob(True)
        try:
            on = G._h0_mode(md)
        finally:
            G.GATE_INPLACE = old
        want = "zeros" if not bool(mask.any()) else ("nomask" if bool(mask.all()) else "gather")
        assert on == want, (name, on, want)


def test_zero_state_is_one_shared_buffer_and_read_only():
    """Every layer of a step must get the same zero buffer: no gather, no fill, no allocation."""
    st = _state()
    idx = torch.tensor([0, 1])
    old = _with_knob(True)
    try:
        a = G._prefill_h0(st, idx, CASES["all_missing"], "zeros")
        b = G._prefill_h0(st, idx, CASES["all_missing"], "zeros")
    finally:
        G.GATE_INPLACE = old
    assert a.data_ptr() == b.data_ptr(), "the zero h0 must be reused across layers"
    assert not a.any(), "the shared h0 buffer must stay zero"


def test_all_present_skips_the_fill_but_still_gathers():
    st = _state()
    idx = torch.tensor([1, 2])
    old = _with_knob(True)
    try:
        got = G._prefill_h0(st, idx, CASES["all_present"], "nomask")
    finally:
        G.GATE_INPLACE = old
    assert torch.equal(got, st[idx].float())


def test_zero_buffer_does_not_alias_the_state_cache():
    st = _state()
    old = _with_knob(True)
    try:
        buf = G._prefill_h0(st, torch.tensor([0, 1]), CASES["all_missing"], "zeros")
    finally:
        G.GATE_INPLACE = old
    assert buf.data_ptr() != st.data_ptr()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
