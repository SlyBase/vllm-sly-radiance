"""KV4H V nibble unpack (P.V B operand): the dim-half dh is wave-uniform and must not cost a
v_cndmask per dword. CPU: the unpack math is pinned for both halves. Compile gate: the device
assembly of the shipped prefill instance (head 256, tile 64) is counted; skipped without hipcc."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "ci"))

M = np.uint32(0xF0F0F0F0)


def unpack(w, dh):
    """Mirror of the kernel: V's nibble as the high half of its byte (16 * v as int8)."""
    w = w.astype(np.uint32)
    return (w & M) if dh else ((w << np.uint32(4)) & M)


def test_unpack_math_pinned():
    w = np.random.default_rng(0).integers(0, 2**32, 4096, dtype=np.uint64).astype(np.uint32)
    for dh, byte_nib in ((1, lambda b: b >> 4), (0, lambda b: b & 15)):
        out = unpack(w, dh)
        for i in range(4):  # byte i of the output = chosen nibble of byte i, in the high half
            src = (w >> np.uint32(8 * i)) & np.uint32(0xFF)
            assert np.array_equal((out >> np.uint32(8 * i)) & np.uint32(0xFF), byte_nib(src) << np.uint32(4))


def _asm():
    hipcc = shutil.which("hipcc")
    if not hipcc:
        pytest.skip("needs hipcc (serving image)")
    inc = subprocess.run([sys.executable, "-m", "pybind11", "--includes"], capture_output=True, text=True)
    if inc.returncode:
        pytest.skip("needs pybind11")
    out = Path(tempfile.mkdtemp()) / "kv4h.s"
    subprocess.run([hipcc, "-O3", "-fPIC", "-std=c++20", "--offload-arch=gfx1201", *inc.stdout.split(),
                    "-DKV4H_NRT=4", "-DKV4H_QP=1", "-DKV4H_NS=1", "-DKV4H_PREF=1", "-DKV4H_MODNAME=m",
                    "--offload-device-only", "-S", str(ROOT / "sly/kv4/radiance_kv4h.hip"), "-o", str(out)],
                   check=True, capture_output=True)
    return out


def test_prefill_instance_has_no_per_dword_dh_select():
    from attn_sched_lint import parse
    ks = [k for n, k in parse(str(_asm())).items() if "kv4h_attentionILi256ELi4ELi4ELi1ELi1E" in n]
    assert len(ks) == 1
    k = ks[0]
    cnd = sum(1 for m, _ in k.ins if m.startswith("v_cndmask"))
    assert cnd < 320, f"{cnd} v_cndmask (352 before: dh lane-selected per V dword, 64 per tile)"
    assert k.vgpr <= 240 and not k.spill, (k.vgpr, k.spill)
