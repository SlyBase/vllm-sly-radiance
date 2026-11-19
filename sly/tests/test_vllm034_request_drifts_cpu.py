#!/usr/bin/env python3
"""CPU tests for the two vLLM 0.34 request-time drifts (release gate blockers).

1. Drafter attn tune (`sly/radiance_attn_drafter.py`): 0.34 hands the DFlash drafter's TRITON_ATTN an
   fp8 query (q/k/v all e4m3); the gate only accepted bf16 q and declined, so the tune never engaged
   and the required `drafter attn plan:` log never appeared. The kernel is vLLM's own
   `kernel_unified_attention` (only the launch shape is ours); its fp8-q path is Q_IS_FP8=True +
   q_scale=<q_descale tensor> (USE_FP8_Q_DESCALE auto-derives from KV_QUANT_MODE==1). TILE=32 already
   satisfies the fp8 P@V dot minimum.

2. Lookup draft (`sly/radiance_lookup_draft.py`): 0.34's `speculator.draft_logits` is a
   DraftLogitsCache object (`.values` [rows, steps, vocab], head_dtype=bf16, `.row_mapping` None for
   replicated TP=1), not the float32 tensor 0.32 exposed. `dl.dtype` raised AttributeError and the
   lookup lever switched itself off (`off after an error`, a forbidden gate pattern). The override
   must resolve `.values`, accept bf16, and decline sharded caches (row != slot).

Plain torch, no GPU, no vllm import. Run: pytest sly/tests/test_vllm034_request_drifts_cpu.py -q
"""
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sly"))

import radiance_attn_drafter as D
import radiance_lookup_draft as L

# CI's cpu-tests env has torch but no triton; _prepare only needs next_power_of_2 (pure math, not the
# behavior under test). Stub it so the cache-resolution logic runs there too. Only this file imports L.
if L.triton is None:
    L.triton = types.SimpleNamespace(next_power_of_2=lambda x: 1 << (int(x) - 1).bit_length())


class _Launch:
    """A triton-kernel stand-in: kernel[grid](*a, **kw) captures the launch kwargs."""
    def __init__(self, rec, slot):
        self.rec, self.slot = rec, slot

    def __getitem__(self, grid):
        def launch(*a, **kw):
            setattr(self.rec, self.slot, kw)
        return launch


class _Recorder:
    """Captures the launch kwargs of vLLM's kernel (the tune's job is to build them)."""
    def __init__(self):
        self.main = None
        self.reduce = None
        self.kernel_unified_attention = _Launch(self, "main")
        self.reduce_segments = _Launch(self, "reduce")


def _drafter_kw(qdtype, *, q_descale="present"):
    fp8 = torch.float8_e4m3fn
    nseq, nq, kvh, head = 1, 16, 4, 128
    ntok = nq  # one query block
    q = torch.zeros(ntok, nq, head, dtype=qdtype)
    k = torch.zeros(2, 16, kvh, head, dtype=fp8)
    v = torch.zeros(2, 16, kvh, head, dtype=fp8)
    out = torch.zeros(ntok, nq, head, dtype=torch.bfloat16)
    sentinel = object()
    D._state.update(bf16=torch.bfloat16, fp8=fp8, fp8_per_tensor=sentinel,
                    torch=torch, logged=True, failed=False, declined=False)
    kw = dict(q=q, k=k, v=v, out=out, kv_quant_mode=sentinel,
              causal=True, window_size=(8192, 0), max_seqlen_q=8,
              seqused_k=torch.tensor([16], dtype=torch.int32),
              block_table=torch.zeros(nseq, 4, dtype=torch.int32),
              cu_seqlens_q=torch.tensor([0, ntok], dtype=torch.int32),
              softmax_scale=0.088, k_descale=torch.tensor([1.0]), v_descale=torch.tensor([1.0]),
              softcap=0.0, use_td=False, use_alibi_sqrt=False, chunk_lookback=-1)
    if q_descale == "present":
        kw["q_descale"] = torch.tensor([1.0])
    elif q_descale is None:
        kw["q_descale"] = None
    # _MUST_BE_NONE members absent
    return kw


def test_drafter_gate_accepts_fp8_query():
    kw = _drafter_kw(torch.float8_e4m3fn)
    plan, why = D._gate(kw)
    assert why is None, f"fp8 query declined: {why}"
    assert plan is not None and plan[1] == 8


def test_drafter_gate_still_accepts_bf16_query():
    kw = _drafter_kw(torch.bfloat16, q_descale=None)
    plan, why = D._gate(kw)
    assert why is None, f"bf16 query declined: {why}"
    assert plan is not None


def test_drafter_gate_declines_fp8_query_without_descale():
    kw = _drafter_kw(torch.float8_e4m3fn, q_descale=None)
    plan, why = D._gate(kw)
    assert plan is None and why and "q_descale" in why


def test_drafter_run_wires_q_is_fp8_and_q_scale():
    kw = _drafter_kw(torch.float8_e4m3fn)
    rec = _Recorder()
    D._state["TU"] = rec
    D._slice_window = _Launch(rec, "slice")  # window-slice kernel: capture only
    plan, why = D._gate(kw)
    assert why is None
    D._run(kw, *plan)
    assert rec.main is not None, "main kernel not launched"
    assert rec.main["Q_IS_FP8"] is True
    assert rec.main["q_scale"] is kw["q_descale"]
    assert rec.main["KV_QUANT_MODE"] is D._state["fp8_per_tensor"]


def test_drafter_run_bf16_query_keeps_q_scale_none():
    kw = _drafter_kw(torch.bfloat16, q_descale=None)
    rec = _Recorder()
    D._state["TU"] = rec
    D._slice_window = _Launch(rec, "slice")
    plan, why = D._gate(kw)
    assert why is None
    D._run(kw, *plan)
    assert rec.main["Q_IS_FP8"] is False
    assert rec.main["q_scale"] is None


class _Cache:
    """Mimics vLLM 0.34 DraftLogitsCache: .values tensor + .row_mapping (None = replicated)."""
    def __init__(self, values, row_mapping=None):
        self.values = values
        self.row_mapping = row_mapping


def _lookup_sp(dl, *, steps=7, vocab=64, n=4, top_k=3):
    sp = types.SimpleNamespace()
    sp.draft_logits = dl
    sp.num_speculative_steps = steps
    sp.top_k = top_k
    sp.selector_top_k = top_k
    sp.vllm_config = types.SimpleNamespace(
        model_config=types.SimpleNamespace(get_vocab_size=lambda: vocab))
    sp.candidate_sampler = types.SimpleNamespace(
        cached_candidate_ids=torch.zeros(n, steps, top_k, dtype=torch.int64))
    sp.dflash_config = {"swa_window_size": 8192}
    rs = types.SimpleNamespace(
        all_token_ids=types.SimpleNamespace(gpu=torch.zeros(n, 4096, dtype=torch.int32)),
        total_len=types.SimpleNamespace(gpu=torch.full((n,), 100, dtype=torch.int32)))
    return sp, rs


def test_lookup_prepare_accepts_draft_logits_cache():
    values = torch.zeros(4, 7, 64, dtype=torch.bfloat16)  # head_dtype bf16, [rows, steps, vocab]
    sp, rs = _lookup_sp(_Cache(values))
    L._state.update(failed=False, ready=False)
    L._prepare(sp, rs)  # 0.32 code raises AttributeError: 'DraftLogitsCache' has no 'dtype'
    assert L._state["ready"] is True


def test_lookup_prepare_accepts_plain_tensor():
    tensor = torch.zeros(4, 7, 64, dtype=torch.float32)  # 0.32 style
    sp, rs = _lookup_sp(tensor)
    L._state.update(failed=False, ready=False)
    L._prepare(sp, rs)
    assert L._state["ready"] is True


def test_lookup_prepare_declines_sharded_cache():
    values = torch.zeros(4, 7, 64, dtype=torch.bfloat16)
    sharded = _Cache(values, row_mapping=torch.zeros(4, dtype=torch.int32))
    sp, rs = _lookup_sp(sharded)
    L._state.update(failed=False, ready=False)
    try:
        L._prepare(sp, rs)
    except RuntimeError as e:
        assert "shard" in str(e).lower()
    else:
        raise AssertionError("sharded cache must decline (row != slot indexing)")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))