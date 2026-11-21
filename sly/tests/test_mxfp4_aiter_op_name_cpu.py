"""The aiter MXFP4 dynamic-quant op was renamed upstream at vLLM main@1bdc437.

radiance::mxfp4_linear owns the whole dispatch: every GEMM the W4A8 kernel cannot serve
(weight_ref numel 1), and with the module default RADIANCE_MXFP4_W4A8_MIN_M=256 every
decode-sized M, goes to vLLM's aiter op. Up to 0.34 that op was
torch.ops.vllm.gemm_with_dynamic_quant, registered by kernels/linear/mxfp4/aiter.py;
main moved it into _aiter_ops.py as rocm_aiter_mxfp4_gemm_dynamic_quant (same positional
signature). The dispatch must resolve whichever name the installed vLLM registers --
a missing op is an AttributeError inside the custom op, i.e. a dead EngineCore.

CPU only: the op namespace is a recorder, no aiter / GPU import.
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mxfp4"))
import radiance_mxfp4 as R  # noqa: E402

M, N, K = 8, 64, 128
NEW_NAME = "rocm_aiter_mxfp4_gemm_dynamic_quant"
OLD_NAME = "gemm_with_dynamic_quant"


class _OnlyOps:
    """A torch.ops.vllm stand-in that registers exactly one op name."""

    def __init__(self, name):
        self.name = name
        self.calls = []

    def __getattr__(self, item):
        if item != self.name:
            raise AttributeError(item)

        def rec(*a, **k):
            self.calls.append((a, k))
            return torch.zeros(a[0].shape[0], a[1].shape[0], dtype=torch.bfloat16)

        return rec


@pytest.fixture
def dispatch(monkeypatch):
    monkeypatch.setattr(R, "_stats_reported", [True])
    monkeypatch.setattr(R, "MIN_M", 256)          # the module default: decode M takes the aiter arm
    monkeypatch.setattr(R, "SHADOW_NK", None)
    monkeypatch.setattr(R, "ACT_SPLIT", False)

    def run(name):
        ops = _OnlyOps(name)
        monkeypatch.setattr(torch.ops, "vllm", ops)
        x = torch.zeros(M, K, dtype=torch.bfloat16)
        w = torch.zeros(N, K // 2, dtype=torch.uint8)
        ws = torch.zeros(K // 32, N, dtype=torch.uint8)
        wref = torch.zeros(1)                    # numel 1: the W4A8 kernel cannot serve this layer
        out = R.mxfp4_linear(x, w, ws, wref)
        return ops, out, (x, w, ws)

    return run


def test_main_op_name_is_called_with_the_stock_arguments(dispatch):
    ops, out, args = dispatch(NEW_NAME)
    assert len(ops.calls) == 1
    a, k = ops.calls[0]
    assert a == (args[0], args[1], args[2], False, torch.bfloat16)
    assert k == {}
    assert out.shape == (M, N)


def test_release_op_name_still_works(dispatch):
    ops, out, args = dispatch(OLD_NAME)
    assert len(ops.calls) == 1
    assert ops.calls[0][0] == (args[0], args[1], args[2], False, torch.bfloat16)
    assert out.shape == (M, N)
