"""Issue #223: a row-major (HOIST_QUANT) activation must never take the A-tiled GEMM path.

`a_tiled_take` trusts data_ptr + (M, K). With TILED_OPROJ=1 the GDN producer registers
(M, 6144); a later row-major o_proj activation (hoisted quant, same K) can land on that address
and be read as tiled -> garbage scales in the partial 16-row fragment -> NaN logits.
The hoisted branch therefore passes allow_tiled=False; the tuple (producer) branch keeps True.
CPU only: the GEMM launchers are replaced by recorders.
"""
import os
import re
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mxfp4"))
import radiance_mxfp4 as R  # noqa: E402

M, N, K = 600, 64, 6144


class _Ext:
    def __init__(self):
        self.calls = []

    def launch(self, *a):
        self.calls.append("rowmajor")

    def launch_at(self, *a):
        self.calls.append("tiled")


@pytest.fixture
def ext(monkeypatch):
    e = _Ext()
    monkeypatch.setattr(R, "_ext", e)
    monkeypatch.setattr(R, "A_TILED_MIN_M", 513)
    monkeypatch.setattr(R, "_stats_reported", [True])
    monkeypatch.setattr(R, "ACT_SPLIT", False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: type("S", (), {"cuda_stream": 0})())
    R._A_TILED.clear()
    yield e
    R._A_TILED.clear()


def _run(x, **kw):
    weight = torch.zeros(N, K // 2, dtype=torch.uint8)
    wscale = torch.zeros(K // 32, N, dtype=torch.uint8)
    return torch.ops.radiance.mxfp4_linear_pq(
        x, torch.zeros(M), weight, wscale, weight, **kw)


def test_registered_producer_output_takes_the_tiled_path(ext):
    x = torch.zeros(M, K, dtype=torch.uint8)
    R.a_tiled_register(x, M, K)
    _run(x)
    assert ext.calls == ["tiled"]


def test_row_major_input_at_a_registered_address_stays_row_major(ext):
    x = torch.zeros(M, K, dtype=torch.uint8)
    R.a_tiled_register(x, M, K)           # stale entry of an earlier tiled producer
    _run(x, allow_tiled=False)
    assert ext.calls == ["rowmajor"]


def test_hoisted_branch_passes_allow_tiled_false():
    src = open(R.__file__).read()
    hoist = src.split("if HOIST_QUANT:", 1)[1].split("else:\n                y = torch.ops.radiance.mxfp4_linear(", 1)[0]
    assert re.search(r"allow_tiled=False", hoist)
