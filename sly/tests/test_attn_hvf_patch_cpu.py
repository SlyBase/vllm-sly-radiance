#!/usr/bin/env python3
"""Pins the opt-in heaviest-first dispatch of the prefill attention (O_HVF) in the r4d patch.

Static checks on the patch text only (no GPU, no compiler): the bit is a fresh power of two that collides
with neither the shipped bits nor O_KV4I (1 << 25, exp/kv4i-dump), the knob is default-off, only the
tile-16 DB launch can pick the HVF instance, the swapped grid is used only by the HVF instance, and the
kernel maps the swapped grid back to the same (q-block, kv head) pair. Numerics are gated on the GPU
(bit-identity vs the same tile with R4D_ATTN_HVF=0).

Run:  python3 sly/tests/test_attn_hvf_patch_cpu.py
"""
import re
from pathlib import Path

R4D = Path(__file__).resolve().parents[1] / "r4d"
PATCH = (R4D / "r4d_attn_heavy_first.patch").read_text()
ADDED = "\n".join(line[1:] for line in PATCH.splitlines() if line.startswith("+"))
SHIPPED = "\n".join(line[1:] for p in ("r4d_extras_rx10.patch", "r4d_gdn_scan_variants.patch")
                    for line in (R4D / p).read_text().splitlines() if line.startswith("+"))

O_HVF = 1 << 26
assert re.search(r"#define O_HVF\s+67108864", ADDED), "O_HVF must be bit 26"
shipped = {int(v) for v in re.findall(r"#define O_[A-Z0-9]+\s+(\d+)", SHIPPED)}
assert O_HVF not in shipped and (1 << 25) != O_HVF, "O_HVF collides with a shipped bit or O_KV4I"

# default off; the env is read once
assert 'getenv("R4D_ATTN_HVF")' in ADDED and "return e && atoi(e) != 0;" in ADDED
assert 'getenv("R4D_ATTN_HVF_MAX_CTX")' in ADDED and "return e ? atoi(e) : 0;" in ADDED
assert "const bool hvf_on = hvf && (hvf_max == 0 || a->max_ctx <= hvf_max);" in ADDED

# only the tile-16 DB launch takes it, the tile-32/48 lines are untouched
assert "if (db && hvf_on) P_LAUNCHTQ(16, B | O_DB | O_HVF);" in ADDED
assert len([l for l in ADDED.splitlines() if "P_LAUNCHTQ(" in l and "O_HVF" in l]) == 1
assert "((OPTX) & O_HVF) ? dim3(grid.y, grid.x, grid.z) : grid" in ADDED

# the kernel undoes the swap: x = kv head, y = q-block walked from the last (heaviest) one
assert "const int qb  = HVF ? (int)(gridDim.y - 1 - blockIdx.y) : (int)blockIdx.x;" in ADDED
assert "const int kvh = HVF ? (int)blockIdx.x : (int)blockIdx.y;" in ADDED
for nq, nkv in ((1, 4), (64, 4), (65, 4)):   # (q-blocks, kv heads): HVF visits every pair exactly once
    stock = {(x, y) for x in range(nq) for y in range(nkv)}
    hvf = {(nq - 1 - y, x) for x in range(nkv) for y in range(nq)}
    assert hvf == stock, (nq, nkv)
print("ok")
