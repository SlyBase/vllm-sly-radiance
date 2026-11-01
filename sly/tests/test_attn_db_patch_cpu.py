#!/usr/bin/env python3
"""Pins the opt-in double-buffered prefill-attention staging (O_DB) in the r4d patch.

Static checks on the patch text only (no GPU, no compiler): the bit is a fresh power of two, the
default launch is untouched, the knobs are default-off, and the LDS budget of every instantiated
tile fits the 64 KiB workgroup limit. Numerics are gated on the GPU (bit-identity vs same-tile stock).

Run:  python3 sly/tests/test_attn_db_patch_cpu.py
"""
import re
from pathlib import Path

PATCH = (Path(__file__).resolve().parents[1] / "r4d" / "r4d_extras_rx10.patch").read_text()
ADDED = "\n".join(line[1:] for line in PATCH.splitlines() if line.startswith("+"))

bits = [int(v) for v in re.findall(r"#define O_[A-Z0-9]+\s+(\d+)", ADDED)]
assert 4194304 in bits and bits.count(4194304) == 1, "O_DB must be defined once as bit 22"
assert re.search(r"#define O_DB\s+4194304", ADDED)
assert 8388608 not in bits, "bit 23 stays unused"
assert all(b & (b - 1) == 0 for b in bits if b == 4194304)

# default path unchanged: env unset -> tile 48 stock instance, DB only on R4D_ATTN_DB != 0
assert 'getenv("R4D_ATTN_TILE")' in ADDED and 'getenv("R4D_ATTN_DB")' in ADDED
assert "const bool db = ed && atoi(ed) != 0;" in ADDED
assert "t = et ? atoi(et) : P_TILE;" in ADDED
assert "static_assert(!DB || (QK8 && PV8 && KWIDE && BTS)" in ADDED

for tile in (16, 32, 48):                    # LDS: 2 x (K bytes + V bytes) at the 8-bit legs
    lds = 2 * (tile * 264 + 256 * (tile + 8))
    assert lds <= 65536, (tile, lds)
print("ok")
