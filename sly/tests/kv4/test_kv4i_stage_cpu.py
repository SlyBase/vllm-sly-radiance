"""CPU proof of the O_KV4I staging (sly/r4d/r4d_kv4i.patch): the byte stream the kernel puts in LDS.

The kernel (fetchKV / storeKV under KV4I) maps each of the 16 * TILE staging threads to 16 K elements
of one key and to 8 V tokens of one byte row d, reads raw bytes of the int4 cache with fixed offsets,
and writes e4m3 bytes rne(code * scale) into the K tile sK8[key][dim] and the V tile sV8[dim][key].
This test replays that thread mapping and those byte offsets in pure torch over a cache written by the
layout oracle (radiance_kv4h_format.encode_block), and compares the two tiles with the oracle's own
dequantization (decode_block: code * e4m3 scale) rounded once to float8_e4m3fn -- the indexing proof
(low/high nibble, token/dim order, block table of 16-token group ids, 1664 blocks straddled by tiles,
ragged tail with garbage NaN scales past ctx). What it cannot prove: that v_cvt_pk_fp8_f32 rounds like
torch (it is the instruction the shipped PV8 path already relies on) -- that is a GPU question.

Skips when torch is missing.
"""
import functools
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch is not installed on this runner")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kv4"))
import radiance_kv4h_format as F4  # noqa: E402

BS, GPB, GBYTES, VOFF, KVHEADS = 1664, 104, 2176, 226304, 1
REGION = BS * 2 * 136


def _e4m3(x):
    return x.to(torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)


@functools.lru_cache(maxsize=None)
def _q(code, sbyte):  # rne_e4m3(code * scale(sbyte)): the byte the kernel stages
    s = torch.tensor([sbyte], dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)
    return int(_e4m3(torch.tensor([float(code)]) * s)[0])


def _same(a, b):  # as values: a negative code times a zero scale stages 0x80 (-0.0), which the MMA reads as 0
    return torch.equal(a.view(torch.float8_e4m3fn).to(torch.float32), b.view(torch.float8_e4m3fn).to(torch.float32))


def _signed(n):
    return n - 16 if n >= 8 else n


def stage(cache, table, ctx, k0, TILE):
    """The kernel's fetchKV + storeKV for one key tile -> (sK8 [TILE,256], sV8 [256,TILE]) uint8."""
    def base(G):  # grp(): block G / 104, group G % 104 (kv head 0, block stride = REGION)
        b, g = divmod(G, GPB)
        return b * REGION + g * GBYTES

    def G_of(k):
        return table[(k - k0) // 16]

    sK = torch.zeros(TILE, 256, dtype=torch.uint8)
    sV = torch.zeros(256, TILE, dtype=torch.uint8)
    for tid in range(16 * TILE):
        key, ch = 2 * (tid >> 5) + (tid & 1), (tid & 31) >> 1
        kk = min(k0 + key, ctx - 1)
        gk, kt = base(G_of(kk)), kk & 15
        ksc = int(cache[gk + kt])
        kn = [int(x) for x in cache[gk + 128 + kt * 128 + ch * 8: gk + 128 + kt * 128 + ch * 8 + 8]]
        for e in range(16):
            nib = (kn[e // 2] >> (4 * (e & 1))) & 0xF
            sK[key, ch * 16 + e] = _q(_signed(nib), ksc)
        vd, hg = tid & 127, tid >> 7
        kb = k0 + hg * 8
        kl = min(kb, (ctx - 1) & ~7)
        gv, vt = base(G_of(kl)) + VOFF, kl & 15
        vn = [int(x) for x in cache[gv + 16 + vd * 16 + vt: gv + 16 + vd * 16 + vt + 8]]
        vs = [int(x) for x in cache[gv + vt: gv + vt + 8]]
        nv = min(max(ctx - kb, 0), 8)
        vs = [b if j < nv else 0 for j, b in enumerate(vs)]       # the SGPR mask: tokens >= ctx -> scale 0
        for j in range(8):
            lo, hi = _signed(vn[j] & 0xF), _signed(vn[j] >> 4)
            sV[vd, hg * 8 + j] = _q(lo, vs[j])
            sV[vd + 128, hg * 8 + j] = _q(hi, vs[j])
    return sK, sV


def test_layout_constants_match_oracle():
    L = F4.L256
    assert (BS // 16, L.REG, L.k_region_bytes(BS), L.TPT * 2 * BS) == (GPB, GBYTES, VOFF, REGION)
    assert L.k_nib_off(5, 7) == 128 + 5 * 128 + 7 and L.k_scale_off(5, 0) == 5
    assert L.v_nib_off(5, 9, BS) == VOFF + 16 + 9 * 16 + 5 and L.v_scale_off(5, BS) == VOFF + 5
    assert F4.GROUP == 256


@pytest.mark.parametrize("TILE", [16, 32, 48])
@pytest.mark.parametrize("ctx", [1664 + 203])
def test_stage_matches_dequant_rounded_to_e4m3(TILE, ctx):
    g = torch.Generator().manual_seed(7)
    nblk = -(-ctx // BS)
    phys = [2, 0, 1][:nblk]                                         # non-identity block table
    cache = torch.randint(0, 256, (3 * REGION,), generator=g, dtype=torch.uint8)   # garbage everywhere
    kv, expk, expv = [], [], []
    for i, p in enumerate(phys):
        T = min(BS, ctx - i * BS)
        k = torch.randn(T, 256, generator=g) * (1 + 3 * torch.rand(T, 1, generator=g))
        v = torch.randn(T, 256, generator=g) * (0.5 + torch.rand(T, 1, generator=g))
        reg = F4.encode_block(k, v, BS)
        # unwritten tokens: garbage nibbles, and 0xFF (e4m3 NaN) scales -- must never reach the tile
        gl = torch.arange(T, BS)
        for t in gl.tolist():
            reg[F4.L256.k_scale_off(t, 0)] = 0xFF
            reg[F4.L256.v_scale_off(t, BS)] = 0xFF
        cache[p * REGION:(p + 1) * REGION] = reg
        kr, vr = F4.decode_block(reg, BS, T)
        expk.append(kr)
        expv.append(vr)
    expk, expv = torch.cat(expk), torch.cat(expv)
    table = [p * GPB + g_ for p in phys for g_ in range(GPB)]       # 16-token group ids
    for k0 in range(0, ctx, TILE):
        sK, sV = stage(cache, table[k0 // 16:], ctx, k0, TILE)
        for key in range(TILE):
            kk = min(k0 + key, ctx - 1)
            assert _same(sK[key], _e4m3(expk[kk])), (k0, key)
            tok = k0 + key
            want = _e4m3(expv[tok]) if tok < ctx else torch.zeros(256, dtype=torch.uint8)
            assert _same(sV[:, key], want), (k0, key)
