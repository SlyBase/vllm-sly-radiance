#!/usr/bin/env python3
"""CPU test: `_head_is_empty` in radiance_drafthead.py keeps its semantics after the chunked
rewrite (upstream ggz14 2ec647c, mirror PR #231).

The rewrite replaces the full-table `abs().max()` temporary with a chunked loop to bound startup
memory (opt-in path, RADIANCE_FAST_DRAFT=1). Behaviour must be unchanged:
  * bf16/fp32: empty iff every value is zero, signed zeros (-0.0) count as zero;
  * fp8 (scale given): byte check -- any non-zero BYTE means not empty, including the fp8
    signed-zero byte 0x80 (fp8 has no abs());
  * the chunk loop must see non-zeros past the first 8192-row chunk.

CPU tensors only, no GPU. Run:  pytest sly/tests/test_drafthead_head_is_empty_cpu.py -v
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch-cpu needed for the drafthead tensor checks")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import radiance_drafthead as d  # noqa: E402


def _bf16(n, k=8):
    return torch.zeros(n, k, dtype=torch.bfloat16)


def _fp8(n, k=8):
    return torch.zeros(n, k, dtype=torch.uint8).view(torch.float8_e4m3fn)


def test_bf16_all_zero_is_empty():
    assert d._head_is_empty(_bf16(16), None) is True


def test_bf16_one_nonzero_is_not_empty():
    t = _bf16(16)
    t[3, 5] = 1.0
    assert d._head_is_empty(t, None) is False


def test_bf16_signed_zeros_are_empty():
    t = _bf16(16)
    t[:] = -0.0
    assert d._head_is_empty(t, None) is True


def test_bf16_nan_is_not_empty():
    t = _bf16(16)
    t[0, 0] = float("nan")
    assert d._head_is_empty(t, None) is False


def test_bf16_nonzero_past_first_chunk_is_not_empty():
    t = _bf16(10000)
    t[9000, 0] = 1.0
    assert d._head_is_empty(t, None) is False


def test_fp8_all_zero_bytes_is_empty():
    assert d._head_is_empty(_fp8(16), torch.ones(16)) is True


def test_fp8_one_nonzero_byte_is_not_empty():
    t = _fp8(16)
    t.view(torch.uint8)[3, 5] = 0x3C  # 1.0 in e4m3
    assert d._head_is_empty(t, torch.ones(16)) is False


def test_fp8_signed_zero_byte_is_not_empty():
    # Byte check preserved: 0x80 is -0.0 in e4m3, but a non-zero byte means "not empty".
    t = _fp8(16)
    t.view(torch.uint8)[0, 0] = 0x80
    assert d._head_is_empty(t, torch.ones(16)) is False


def test_fp8_nonzero_past_first_chunk_is_not_empty():
    t = _fp8(10000)
    t.view(torch.uint8)[9000, 0] = 0x3C
    assert d._head_is_empty(t, torch.ones(10000)) is False


def test_zero_rows_is_empty():
    assert d._head_is_empty(_bf16(0), None) is True
    assert d._head_is_empty(_fp8(0), torch.ones(0)) is True
