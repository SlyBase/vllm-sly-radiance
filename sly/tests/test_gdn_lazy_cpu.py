#!/usr/bin/env python3
"""CPU tests for the lazy GDN glue (radiance_gdn_lazy.py): when the lazy temporal path runs.

Lazy temporal states need the stash block, and only a drafter allocates it
(num_speculative_blocks > 0). Without one, mode-0 materialize zeroed the stash headers at
bt[dst_col + 1] -- a column past the request's blocks that can hold a stale id of another
block, so the zeroing hit live state (1.3.1: lazy no-spec runs not deterministic, eager are).

Plain torch only, no vllm, no GPU. The GPU half (the kernel guard) is
sly/tests/gpu_gdn_lazy_materialize.py.

Run:  python3 sly/tests/test_gdn_lazy_cpu.py     (exit 0 = pass)
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

os.environ["RADIANCE_GDN_LAZY"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import radiance_gdn_lazy as L  # noqa: E402


def _kv_cfg(num_spec_blocks):
    spec = SimpleNamespace(num_speculative_blocks=num_spec_blocks)
    return SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)] * 3)


def test_temporal_needs_a_stash_block():
    assert L.temporal_enabled(_kv_cfg(1), [0, 1, 2]) is True
    assert L.temporal_enabled(_kv_cfg(0), [0, 1, 2]) is False


def test_temporal_off_without_the_env_knob():
    try:
        L.ENABLED = False
        assert L.temporal_enabled(_kv_cfg(1), [0, 1, 2]) is False
    finally:
        L.ENABLED = True


def _run_materialize(has_stash, mode):
    calls = []
    t = torch.zeros(1, dtype=torch.int32)
    tb = SimpleNamespace(fn=lambda *a: calls.append(a), has_stash=has_stash, n=1, H=48, Hg=16,
                         K=128, V=128, st_head=16384, state_ptrs=t, slot_strides=t,
                         group_idx=t, alog_ptrs=t, dtb_ptrs=t)
    ctx = SimpleNamespace(_radiance_lazy_tables=tb, block_table_ptrs=t,
                          block_table_stride_req=64, block_size=2048)
    real = torch.cuda.current_stream
    torch.cuda.current_stream = lambda: SimpleNamespace(cuda_stream=0)   # CPU: no stream
    try:
        L.materialize(ctx, mode, 1, t, t, t, None)
    finally:
        torch.cuda.current_stream = real
    return calls[0][8]                       # the kernel's mode argument


def test_kernel_gets_the_stash_flag_only_with_a_stash_block():
    assert _run_materialize(True, 0) == 0 | L.MODE_STASH
    assert _run_materialize(True, 1) == 1 | L.MODE_STASH
    assert _run_materialize(False, 0) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
