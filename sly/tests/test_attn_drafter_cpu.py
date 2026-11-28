#!/usr/bin/env python3
"""CPU tests for the drafter split-KV launch with a bf16/fp16 KV cache and general GQA ratios.

vLLM's kernels run with KV_QUANT_MODE 0 (no quantisation) for a bf16 cache, so the gate must accept it, and
the launch must pass the mode and the head ratio through instead of fixed fp8 / 4 values. The kernels are
stubbed: the test checks what the launch hands them. Plain torch, no vllm, no GPU.
"""
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sly"))

import radiance_attn_drafter as D

FP8 = torch.float8_e4m3fn
PER_TENSOR, INT8_PTH = object(), object()   # stand-ins for the KVQuantMode members


class _Launch:
    def __init__(self, rec, slot):
        self.rec, self.slot = rec, slot

    def __getitem__(self, grid):
        def launch(*a, **kw):
            setattr(self.rec, self.slot, (grid, kw))
        return launch


class _Rec:
    def __init__(self):
        self.main = self.reduce = self.slice = None
        self.kernel_unified_attention = _Launch(self, "main")
        self.reduce_segments = _Launch(self, "reduce")


@pytest.fixture()
def rec(monkeypatch):
    r = _Rec()
    monkeypatch.setattr(D, "ENABLED", True)
    monkeypatch.setattr(D, "_slice_window", _Launch(r, "slice"))
    monkeypatch.setattr(D, "_state", {
        "orig": lambda *a, **kw: "orig", "torch": torch, "TU": r, "bf16": torch.bfloat16, "fp16": torch.float16,
        "fp8": FP8, "fp8_per_tensor": PER_TENSOR, "kv_none": 0, "logged": True, "failed": False,
        "declined": True, "calls": 0, "seen": set()})
    return r


def _kw(*, dtype=torch.bfloat16, mode=0, nqpkv=4, kvh=8, ntok=16, **over):
    kv_dtype = FP8 if mode is PER_TENSOR else dtype
    nq = kvh * nqpkv
    kw = dict(q=torch.zeros(ntok, nq, 128, dtype=dtype), k=torch.zeros(2, 16, kvh, 128, dtype=kv_dtype),
              v=torch.zeros(2, 16, kvh, 128, dtype=kv_dtype), out=torch.zeros(ntok, nq, 128, dtype=dtype),
              kv_quant_mode=mode, causal=False, window_size=(2047, 0), max_seqlen_q=8,
              seqused_k=torch.tensor([16, 16], dtype=torch.int32)[:ntok // 8 or 1],
              block_table=torch.zeros(2, 4, dtype=torch.int32),
              cu_seqlens_q=torch.tensor([0, 8, 16], dtype=torch.int32),
              softmax_scale=0.088, k_descale=None, v_descale=None, softcap=0.0, use_td=False,
              use_alibi_sqrt=False, chunk_lookback=-1)
    kw.update(over)
    return kw


@pytest.mark.parametrize("mode", [0, None])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("nqpkv,block_m", [(4, 32), (3, 16), (2, 16), (8, 16), (16, 16)])
def test_unquantised_kv_is_taken(rec, mode, dtype, nqpkv, block_m):
    kw = _kw(dtype=dtype, mode=mode, nqpkv=nqpkv)
    plan, why = D._gate(kw)
    assert why is None, why
    D._run(kw, *plan)
    grid, m = rec.main
    bq = block_m // nqpkv
    assert (m["KV_QUANT_MODE"], m["num_queries_per_kv"], m["BLOCK_M"], m["BLOCK_Q"]) == (0, nqpkv, block_m, bq)
    assert m["Q_IS_FP8"] is False and m["USE_TD"] is False and m["USE_TD_QO"] is False
    assert grid == (16 // bq + 2, 8, D.segments_for(2))
    rgrid, r = rec.reduce
    assert r["BLOCK_Q"] == bq and rgrid == (16, 8 * nqpkv)


def test_fp8_per_tensor_still_taken(rec):
    kw = _kw(mode=PER_TENSOR, k_descale=torch.tensor([1.0]), v_descale=torch.tensor([1.0]))
    plan, why = D._gate(kw)
    assert why is None, why
    D._run(kw, *plan)
    assert rec.main[1]["KV_QUANT_MODE"] is PER_TENSOR and rec.main[1]["BLOCK_M"] == 32


@pytest.mark.parametrize("over", [
    dict(mode=INT8_PTH),                                 # per-token-head scales
    dict(max_seqlen_q=17),                               # prefill chunk
    dict(causal=torch.tensor([True, False])),            # per-sequence causal
    dict(nqpkv=5), dict(nqpkv=12), dict(nqpkv=32),       # ratios without a BLOCK_Q
    dict(output_scale=torch.tensor([1.0])),              # quantised output
])
def test_declined_cases_use_stock_launch(rec, over):
    kw = _kw(**{k: v for k, v in over.items() if k in ("mode", "nqpkv")})
    kw.update({k: v for k, v in over.items() if k not in ("mode", "nqpkv")})
    assert D._gate(kw)[0] is None
    assert D.wrapper(**kw) == "orig" and rec.main is None


def test_mismatched_dtypes_declined(rec):
    kw = _kw()
    kw["k"] = kw["k"].to(torch.float16)
    assert D._gate(kw)[0] is None
    assert D._gate(_kw(dtype=torch.float32))[0] is None


def test_env_off_bypasses(rec, monkeypatch):
    monkeypatch.setattr(D, "ENABLED", False)
    assert D.wrapper(**_kw()) == "orig" and rec.main is None


def test_wrapper_launches_and_logs_once_per_pair(rec, capsys):
    D.wrapper(**_kw(nqpkv=4))
    D.wrapper(**_kw(nqpkv=4))
    D.wrapper(**_kw(nqpkv=3))
    assert rec.main is not None and D._state["calls"] == 3
    lines = [l for l in capsys.readouterr().err.splitlines() if "split-KV active" in l]
    assert len(lines) == 2 and "nqpkv=4" in lines[0] and "nqpkv=3" in lines[1], lines
