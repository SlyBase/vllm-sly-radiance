"""CPU gate for RADIANCE_KV4H_PIPE: compile radiance_kv4h.hip for gfx1201 (device assembly only, no
GPU) and check what each schedule bit claims. Skips without a compiler: `hipcc` on PATH (the image),
or RADIANCE_KV4H_TEST_ROCM=<rocm root with lib/llvm/bin/clang++> (e.g. an unpacked TheRock
rocm-sdk-core wheel), plus pybind11 for the module's headers. Numerics are the GPU test's job
(test_kv4h.py::test_attn_pipe_identity); this proves the schedule shape and the register budget.
"""
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "kv4" / "radiance_kv4h.hip"
FLAGS = ["-DKV4H_NRT=4", "-DKV4H_QP=1", "-DKV4H_NS=1", "-DKV4H_PREF=1"]   # the shipped prefill build


def _compiler():
    root = os.environ.get("RADIANCE_KV4H_TEST_ROCM", "")
    if root:
        return [f"{root}/lib/llvm/bin/clang++", "-x", "hip", f"--rocm-path={root}",
                f"--rocm-device-lib-path={root}/lib/llvm/amdgcn/bitcode"]
    return ["hipcc"] if shutil.which("hipcc") else None


@lru_cache(maxsize=None)
def _asm(pipe):
    cc = _compiler()
    if cc is None or importlib.util.find_spec("pybind11") is None:
        pytest.skip("needs hipcc (or RADIANCE_KV4H_TEST_ROCM) and pybind11")
    inc = subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True).split()
    out = Path(os.environ.get("TMPDIR", "/tmp")) / f"kv4h_pipe{pipe}_{os.getpid()}.s"
    subprocess.run([*cc, "--offload-arch=gfx1201", "-O3", "-fPIC", "-std=c++20", "--offload-device-only", "-S", *inc,
                    *FLAGS, f"-DKV4H_PIPE={pipe}", str(SRC), "-o", str(out)], check=True)
    t = out.read_text()
    out.unlink()
    return t


def _inst(pipe, tile, head=256):
    """(body, vgpr, spill, lds) of one kernel instance."""
    t = _asm(pipe)
    name = f"_Z14kv4h_attentionILi{head}ELi{tile // 16}E"
    i = t.index("\n" + name) + 1
    body = t[i:t.index("s_endpgm", i)]
    meta = t[t.index(".name:           " + name):]
    g = lambda k: int(re.search(rf"\.{k}:\s+(\d+)", meta).group(1))  # noqa: E731
    return body, g("vgpr_count"), g("vgpr_spill_count"), g("group_segment_fixed_size")


def test_pipe_global_loads():
    """bit 1: no FLAT staging loads (they also count on the LDS counter), the table entry is a scalar load"""
    for tile in (16, 32, 64):
        assert "flat_load" in _inst(0, tile)[0], "stock staging is expected to be FLAT (premise of bit 1)"
        body = _inst(1, tile)[0]
        assert "flat_load" not in body and "global_load_b128" in body and "s_load_b32" in body, tile


def test_pipe_one_barrier_per_tile():
    """bit 2: one barrier fewer in the kernel (the tile loop's second one), the LDS tile doubled"""
    for tile in (16, 32):
        b0, _, _, lds0 = _inst(0, tile)
        b2, _, _, lds2 = _inst(2, tile)
        assert b2.count("s_barrier_wait") == b0.count("s_barrier_wait") - 1, tile
        assert lds2 > lds0


def test_pipe_no_spill_where_recommended():
    """the instances the knob's comment recommends compile without spill (head 256)"""
    for pipe, tile in ((1, 16), (1, 32), (3, 16), (3, 32), (5, 16)):
        _, vgpr, spill, lds = _inst(pipe, tile)
        assert spill == 0 and vgpr <= 256 and lds <= 64 * 1024, (pipe, tile, vgpr, spill, lds)
