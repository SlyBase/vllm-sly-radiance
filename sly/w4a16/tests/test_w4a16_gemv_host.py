"""Host side of the HIP W4A16 GEMV (radiance_w4a16_gemv.hip): CPU only, no GPU, no built module.

What is proven here, each with plain loops / numpy / torch so an indexing slip shows up before a GPU window:
  * the C++ device math (nibble decode, bf16 rounding, scale widening, launch guards) is COMPILED with g++ from
    the very text of the .hip file and run on the host: bit-exact against a bit-for-bit Python mirror;
  * the decode is exact for all 16 nibbles at all 8 nibble positions: n -> bf16(n - 8);
  * ExLlama [N, K/8] int32 / [N, K/2] int8 / tiled 4-D weights -> sequential nibbles -> the kernel's tiles, checked
    against an independent per-element address formula;
  * a vectorised emulation of gemv_u4_kernel (tile / group / WMMA lane layout, group scales G 32/64/128, bias,
    odd N, M 1..16, any wave split) equals dequantize-then-dot within one bf16 ulp;
  * the dispatcher guards, the cache rules (no allocation under stream capture, memory budget), and the vLLM patch
    (two passes OK -> NOOP, parses, runs before the skinny branch, compatible with patch_w4a16_tiles).
"""
import ast
import ctypes
import os
import re
import shutil
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
W4 = HERE.parent  # sly/w4a16
REPO = W4.parents[1]
HIP = W4 / "radiance_w4a16_gemv.hip"
PATCH = REPO / "sly" / "patch_w4a16_hip_gemv.py"
TILES_PATCH = REPO / "sly" / "patch_w4a16_tiles.py"
sys.path.insert(0, str(W4))

# the pinned upstream file the patches rewrite (read-only reference tree); tests that need it skip without it
UPSTREAM = Path(os.environ.get(
    "W4A16_UPSTREAM_SRC",
    "/opt/data/repos/vllm-sly-radiance/.dryrun/src/vllm/vllm/model_executor/kernels/linear/mixed_precision"
    "/rdna_hybrid_w4a16.py"))
needs_upstream = pytest.mark.skipif(not UPSTREAM.exists(), reason="pinned rdna_hybrid_w4a16.py not available")

import radiance_w4a16_gemv_hip as W  # noqa: E402

SLOT_OF_POS = [0, 4, 1, 5, 2, 6, 3, 7]  # nibble position p of a code dword holds wmma slot SLOT_OF_POS[p]
POS_OF_SLOT = [0, 2, 4, 6, 1, 3, 5, 7]  # slot s <- nibble position (the order the decode emits them)
BF16_OF_Q = [0xC100, 0xC0E0, 0xC0C0, 0xC0A0, 0xC080, 0xC040, 0xC000, 0xBF80,  # q = -8 .. -1
             0x0000, 0x3F80, 0x4000, 0x4040, 0x4080, 0x40A0, 0x40C0, 0x40E0]  # q = 0 .. 7  (index = nibble n)


# ======================================================================== Python mirror of the C++ bit math

def perm(s0, s1, sel):
    """v_perm_b32: result byte j = byte (sel >> 8 j) & 0xFF of the 8 bytes {s1 (0..3), s0 (4..7)}."""
    s0, s1 = np.asarray(s0, np.uint64), np.asarray(s1, np.uint64)
    out = np.zeros(np.broadcast(s0, s1).shape, np.uint64)
    for j in range(4):
        i = (sel >> (8 * j)) & 0xFF
        src = (s1 >> np.uint64(8 * i)) if i < 4 else (s0 >> np.uint64(8 * (i - 4)))
        out |= (src & np.uint64(0xFF)) << np.uint64(8 * j)
    return out.astype(np.uint32)


def u4_val(w, p):
    """nibble p of the low 16 bits of w -> float32 n - 8: magic 2^23 | n 16^p, then ONE fma (exact)."""
    inv = 1.0 / 16 ** p
    f = ((w & np.uint32(0xF << (4 * p))) | np.uint32(0x4B000000)).view(np.float32)
    return np.float32(f.astype(np.float64) * inv - (8388608.0 * inv + 8.0))


def bf16x2(lo, hi):
    return perm(hi.view(np.uint32), lo.view(np.uint32), 0x07060302)


def u4_decode(w):
    """uint32 words [...] -> [..., 4] uint32: bf16 pairs, slot s <- nibble POS_OF_SLOT[s]."""
    w = np.asarray(w, np.uint32)
    h = w >> np.uint32(16)
    return np.stack([bf16x2(u4_val(w, 0), u4_val(w, 2)), bf16x2(u4_val(h, 0), u4_val(h, 2)),
                     bf16x2(u4_val(w, 1), u4_val(w, 3)), bf16x2(u4_val(h, 1), u4_val(h, 3))], -1)


def pairs_to_f32(d):
    """[..., 4] bf16-pair dwords -> [..., 8] float32 (slot order)."""
    lo, hi = (d & np.uint32(0xFFFF)) << np.uint32(16), d & np.uint32(0xFFFF0000)
    return np.stack([lo, hi], -1).reshape(d.shape[:-1] + (8,)).view(np.float32)


def f2bf(v):
    u = np.asarray(v, np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bits(t):  # bf16 tensor -> uint16 numpy
    return t.contiguous().view(torch.int16).numpy().view(np.uint16)


# ======================================================================== (a) decode bit math

def test_bf16_table_is_n_minus_8():
    for n in range(16):
        assert struct.unpack("<f", struct.pack("<I", BF16_OF_Q[n] << 16))[0] == float(n - 8)


@pytest.mark.parametrize("p", range(8))
def test_decode_is_exact_for_every_nibble_at_every_position(p):
    for n in range(16):
        w = np.array([0x88888888 & ~(0xF << 4 * p) | (n << 4 * p)], np.uint32)  # all other nibbles: q = 0
        d = u4_decode(w)[0]
        got = [int(v) for pair in d for v in (pair & 0xFFFF, pair >> 16)]
        want = [BF16_OF_Q[n] if POS_OF_SLOT[s] == p else 0x0000 for s in range(8)]
        assert got == want, (p, n, [hex(g) for g in got])


def test_decode_random_words_match_the_table():
    w = np.random.default_rng(0).integers(0, 2 ** 32, 20000, dtype=np.uint64).astype(np.uint32)
    got = u4_decode(w)
    for s in range(8):
        n = (w >> np.uint32(4 * POS_OF_SLOT[s])) & np.uint32(0xF)
        have = (got[:, s // 2] >> np.uint32(16 * (s % 2))) & np.uint32(0xFFFF)
        assert (have == np.array(BF16_OF_Q, np.uint32)[n]).all(), s


# ---- the SAME bit math, compiled from the .hip text with g++ and run on the host
@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    if not shutil.which("g++"):
        pytest.skip("no g++")
    src = HIP.read_text()
    a, b = src.index("// BEGIN host-testable device math"), src.index("// END host-testable device math")
    d = tmp_path_factory.mktemp("w4a16_host")
    (d / "h.cpp").write_text(r'''
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#define __device__
#define __forceinline__ inline __attribute__((always_inline))
typedef unsigned int u32x4 __attribute__((vector_size(16)));
static inline uint32_t host_perm(uint32_t s0, uint32_t s1, uint32_t sel) {
  uint8_t src[8];
  for (int i = 0; i < 4; i++) { src[i] = (s1 >> (8 * i)) & 0xFF; src[4 + i] = (s0 >> (8 * i)) & 0xFF; }
  uint32_t r = 0;
  for (int j = 0; j < 4; j++) r |= (uint32_t)src[(sel >> (8 * j)) & 7] << (8 * j);
  return r;
}
#define __builtin_amdgcn_perm host_perm
static inline float __uint_as_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t __float_as_uint(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
''' + src[a:b] + r'''
int main() {
  uint32_t mode, n;
  if (fread(&mode, 4, 1, stdin) != 1 || fread(&n, 4, 1, stdin) != 1) return 2;
  if (mode == 0) {  // decode
    std::vector<uint32_t> in(n);
    fread(in.data(), 4, n, stdin);
    for (uint32_t i = 0; i < n; i++) { u32x4 r = u4_decode(in[i]); fwrite(&r, 16, 1, stdout); }
  } else if (mode == 1) {  // f2bf
    std::vector<float> in(n);
    fread(in.data(), 4, n, stdin);
    for (uint32_t i = 0; i < n; i++) { uint16_t r = f2bf(in[i]); fwrite(&r, 2, 1, stdout); }
  } else if (mode == 2) {  // scale pair -> two floats
    std::vector<uint32_t> in(n);
    fread(in.data(), 4, n, stdin);
    for (uint32_t i = 0; i < n; i++) { float r[2] = {sc_lo(in[i]), sc_hi(in[i])}; fwrite(r, 4, 2, stdout); }
  } else {  // guards: 12 int64 per case
    std::vector<int64_t> in(12 * (size_t)n);
    fread(in.data(), 8, in.size(), stdin);
    for (uint32_t i = 0; i < n; i++) {
      const int64_t* a = &in[12 * (size_t)i];
      const char* e = u4_check(a[0], a[1], a[2], a[3], a[4], (uintptr_t)a[5], (uintptr_t)a[6], (uintptr_t)a[7],
                               (uintptr_t)a[8], (uintptr_t)a[9], a[10], a[11]);
      puts(e ? e : "OK");
    }
  }
  return 0;
}
''')
    exe = d / "h"
    r = subprocess.run(["g++", "-std=c++17", "-O1", "-Wall", "-Werror", "-o", str(exe), str(d / "h.cpp")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return exe


def run_h(exe, mode, arr, dtype):
    arr = np.ascontiguousarray(arr, dtype)
    inp = struct.pack("<II", mode, arr.shape[0]) + arr.tobytes()
    r = subprocess.run([str(exe)], input=inp, capture_output=True)
    assert r.returncode == 0
    return r.stdout


def test_cpp_decode_equals_the_python_mirror_bit_for_bit(harness):
    rng = np.random.default_rng(1)
    w = np.concatenate([rng.integers(0, 2 ** 32, 20000, dtype=np.uint64).astype(np.uint32),
                        np.array([0, 0xFFFFFFFF, 0x88888888, 0x01234567, 0x89ABCDEF, 0x76543210], np.uint32)])
    got = np.frombuffer(run_h(harness, 0, w, np.uint32), np.uint32).reshape(-1, 4)
    assert (got == u4_decode(w)).all()
    for s in range(8):  # and the table, independent of the mirror
        n = (w >> np.uint32(4 * POS_OF_SLOT[s])) & np.uint32(0xF)
        have = (got[:, s // 2] >> np.uint32(16 * (s % 2))) & np.uint32(0xFFFF)
        assert (have == np.array(BF16_OF_Q, np.uint32)[n]).all()


def test_cpp_f2bf_rounds_like_torch(harness):
    v = torch.randn(50000) * torch.exp2(torch.randint(-20, 20, (50000,)).float())
    v[:4] = torch.tensor([0.0, 1.0, 1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8])  # ties
    got = np.frombuffer(run_h(harness, 1, v.numpy(), np.float32), np.uint16)
    assert (got == bits(v.to(torch.bfloat16))).all()
    assert (f2bf(v.numpy()) == got).all()  # the Python mirror used by the emulation


def test_cpp_scale_pair_widens_two_bf16(harness):
    s = torch.randn(2000).to(torch.bfloat16)
    pair = bits(s).astype(np.uint32).reshape(-1, 2)
    word = pair[:, 0] | (pair[:, 1] << 16)
    got = np.frombuffer(run_h(harness, 2, word, np.uint32), np.float32).reshape(-1, 2)
    assert (got == s.float().numpy().reshape(-1, 2)).all()


GOOD = dict(M=8, N=100, K=128, SX=128, SY=100, x=64, q=64, s=64, y=64, bias=0, ntiles=7, cfg=-1)
GUARDS = [  # (override, must fail)
    ({}, False), (dict(M=1), False), (dict(M=16), False), (dict(M=0), True), (dict(M=17), True),
    (dict(K=64, SX=64), False), (dict(K=0, SX=0), True), (dict(K=96, SX=96), True), (dict(K=32, SX=32), True),
    (dict(SX=132), True), (dict(SX=136), False), (dict(SX=64), True), (dict(SY=99), True), (dict(SY=104), False),
    (dict(N=0, SY=0, ntiles=0), True), (dict(x=8), True), (dict(q=8), True), (dict(s=8), True), (dict(y=8), True),
    (dict(bias=66), False), (dict(bias=65), True), (dict(ntiles=6), True), (dict(cfg=-2), True),
    (dict(cfg=11), False), (dict(cfg=12), True),
]


def test_cpp_launch_guards(harness):
    cases = [{**GOOD, **o} for o, _ in GUARDS]
    keys = ["M", "N", "K", "SX", "SY", "x", "q", "s", "y", "bias", "ntiles", "cfg"]
    arr = np.array([[c[k] for k in keys] for c in cases], np.int64)
    out = run_h(harness, 3, arr, np.int64).decode().split("\n")[:-1]
    assert len(out) == len(cases)
    for (o, bad), msg in zip(GUARDS, out):
        assert (msg != "OK") == bad, (o, msg)


# ======================================================================== (b) packing + kernel emulation

def exllama_pack(u):
    """The ExLlama shuffle, written from its documented formula: [N, K] nibbles -> [N, K/8] int32."""
    n, k = u.shape
    g = u.to(torch.int64).view(n, k // 8, 8)
    p = (g[..., 0] | (g[..., 2] << 4) | (g[..., 4] << 8) | (g[..., 6] << 12) | (g[..., 1] << 16) | (g[..., 3] << 20)
         | (g[..., 5] << 24) | (g[..., 7] << 28))
    return ((p + 2 ** 31) % 2 ** 32 - 2 ** 31).to(torch.int32)  # two's complement wrap


def tile_like_stock(w32):
    """radiance_w4a16_tile: [N, K/8] int32 -> [N/16, K/128, 16, 64] int8 (the layout the tiles patch stores)."""
    n, k8 = w32.shape
    return w32.view(n // 16, 16, k8 // 16, 16).permute(0, 2, 1, 3).contiguous().view(torch.int8)


def rand_nibbles(n, k, seed=0):
    return torch.randint(0, 16, (n, k), generator=torch.Generator().manual_seed(seed), dtype=torch.uint8)


@needs_upstream
def test_exllama_pack_matches_upstream_function():
    src = UPSTREAM.read_text()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "pack_int4_exllama_shuffle")
    ns = {"torch": torch}
    exec(ast.get_source_segment(src, fn), ns)
    u = rand_nibbles(48, 256, 3)
    assert torch.equal(ns["pack_int4_exllama_shuffle"](u), exllama_pack(u))


@pytest.mark.parametrize("form", ["int32", "int8", "int8_padded_rows", "tiled"])
def test_unpack_exllama_nibbles_roundtrip(form):
    u = rand_nibbles(48, 256, 4)
    w32 = exllama_pack(u)
    if form == "int32":
        w = w32
    elif form == "int8":
        w = w32.view(torch.int8)
    elif form == "int8_padded_rows":  # gfx1151 row padding: the int8 view keeps a longer row stride
        buf = torch.zeros(48, 32 + 8, dtype=torch.int32)
        buf[:, :32] = w32
        w = buf.view(torch.int8)[:, :128]
    else:
        w = tile_like_stock(w32)
    out = W.unpack_exllama_nibbles(w)
    assert out.dtype == torch.uint8 and tuple(out.shape) == (48, 256) and torch.equal(out, u)


def gold_tiles(u, scales, g):
    """Per-element address formula of the .hip header, independent of pack_u4."""
    n, k = u.shape
    nt, kg = -(-n // 16), k // 64
    qt = np.zeros((nt, kg, 512), np.uint8)
    for r in range(n):
        for c_ in range(k):
            t, c = r // 16, r % 16
            g64, rem = c_ // 64, c_ % 64
            b, h, j, slot = rem // 32, (rem % 32) // 16, (rem % 16) // 8, rem % 8
            lane, d, pos = 16 * h + c, 2 * b + j, POS_OF_SLOT[slot]
            qt[t, g64, lane * 16 + d * 4 + pos // 2] |= int(u[r, c_]) << (4 * (pos % 2))
    st = np.zeros((nt, kg, 32), np.uint16)
    sb = bits(scales)
    for r in range(n):
        for g64 in range(kg):
            for b in range(2):
                st[r // 16, g64, (r % 16) * 2 + b] = sb[r, (64 * g64 + 32 * b) // g]
    return qt, st


@pytest.mark.parametrize("n,k,g", [(16, 64, 32), (37, 128, 64), (50, 256, 128), (5, 192, 64), (33, 128, 32)])
def test_pack_u4_matches_the_address_formula(n, k, g):
    u = rand_nibbles(n, k, 5)
    s = (torch.rand(n, k // g, generator=torch.Generator().manual_seed(6)) + 0.1).to(torch.bfloat16)
    qt, st = W.pack_u4(exllama_pack(u).view(torch.int8), s, g)
    gq, gs = gold_tiles(u, s, g)
    assert qt.dtype == torch.uint8 and st.dtype == torch.uint16
    assert tuple(qt.shape) == gq.shape and tuple(st.shape) == gs.shape
    assert (qt.numpy() == gq).all() and (st.view(torch.int16).numpy().view(np.uint16) == gs).all()


def test_pack_u4_chunking_is_invisible(monkeypatch):
    u = rand_nibbles(100, 128, 7)
    s = torch.rand(100, 1).to(torch.bfloat16)
    w = exllama_pack(u)
    a = W.pack_u4(w, s, 128)
    monkeypatch.setattr(W, "CHUNK", 32)
    b = W.pack_u4(w, s, 128)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1].view(torch.int16), b[1].view(torch.int16))


def split_groups(ngroups, w):  # the kernel's g0 / g1 per wave
    return [(ngroups * wv // w, ngroups * (wv + 1) // w) for wv in range(w)]


def test_wave_split_partitions_the_groups():
    for ng in (1, 2, 3, 7, 40, 400):
        for w in (1, 2, 4, 8):
            spans = split_groups(ng, w)
            assert [g for a, b in spans for g in range(a, b)] == list(range(ng))


def emulate(x, qt, st, bias, n, w_waves=4):
    """gemv_u4_kernel on tensors: x [M, K] bf16, tiles -> y [M, n] bf16 bits as float32. Follows the kernel:
    the decode of every code dword, the wmma 16x16x16 sums (A = x, B = weights, lane 16 h + c), the per-block
    scale on the exact K = 32 partial sum, the group split over waves, the cross-wave sum, bias, f2bf."""
    m, k = x.shape
    nt, kg = qt.shape[:2]
    words = qt.contiguous().numpy().view(np.uint32).reshape(nt, kg, 2, 16, 4)  # [t, g, h, c, dword 2 b + j]
    wv = pairs_to_f32(u4_decode(words))  # [t, g, h, c, dword, slot]
    wv = wv.reshape(nt, kg, 2, 16, 2, 2, 8)  # [t, g, h, c, b, j, slot]
    wmat = torch.from_numpy(wv).permute(0, 1, 4, 2, 5, 6, 3)  # [t, g, b, h, j, slot, c]
    xm = torch.zeros(16, kg, 2, 2, 2, 8)
    xm[:m] = x.float().reshape(m, kg, 2, 2, 2, 8)  # K = 64 g + 32 b + 16 h + 8 j + slot; rows >= M are zero
    tmp = torch.einsum("mgbhjs,tgbhjsc->tgbmc", xm, wmat)  # [t, g, b, m, c]: exact K = 32 partial sums
    sw = st.contiguous().view(torch.int16).numpy().view(np.uint32).reshape(nt, kg, 16)  # lane row c: (b0 | b1 << 16)
    scale = np.stack([(sw << np.uint32(16)), sw & np.uint32(0xFFFF0000)], 2).view(np.float32)  # [t, g, b, c]
    scale = torch.from_numpy(np.ascontiguousarray(scale))
    red = torch.zeros(nt, 16, 16)
    for g0, g1 in split_groups(kg, w_waves):
        acc = torch.zeros(nt, 16, 16)
        for g in range(g0, g1):
            for b in range(2):
                acc = acc + tmp[:, g, b] * scale[:, g, b, None, :]
        red = red + acc
    y = red.permute(1, 0, 2).reshape(16, nt * 16)[:m, :n]
    if bias is not None:
        y = y + bias.float()
    return torch.from_numpy(f2bf(y.contiguous().numpy()).astype(np.int16)).view(torch.bfloat16)


def reference(x, u, s, g, bias):
    dq = (u.float() - 8) * s.float().repeat_interleave(g, dim=1)
    y = F.linear(x.float(), dq, None if bias is None else bias.float())
    return y, F.linear(x.float().abs(), dq.abs())


def ulp_ok(got, ref, absref):
    ulp = torch.exp2(torch.floor(torch.log2(ref.abs().clamp(min=2 ** -100))) - 7)
    return ((got.float() - ref).abs() <= ulp + 1e-5 * absref).all().item()


def make_case(n, k, g, m, bias, seed=0):
    gen = torch.Generator().manual_seed(seed)
    u = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.uint8)
    s = (torch.rand(n, k // g, generator=gen) * 0.05 + 0.005).to(torch.bfloat16)
    x = torch.randn(m, k, generator=gen).to(torch.bfloat16)
    b = torch.randn(n, generator=gen).to(torch.bfloat16) if bias else None
    return u, s, x, b


@pytest.mark.parametrize("n,k,g,m,bias,waves", [
    (5000, 256, 128, 8, False, 4), (5000, 256, 128, 8, True, 4), (17, 192, 64, 1, True, 2),
    (1, 64, 32, 16, False, 1), (33, 384, 32, 5, True, 8), (130, 640, 128, 16, True, 8), (64, 128, 64, 1, False, 4),
    (248320, 128, 128, 8, True, 4), (16, 64, 64, 3, False, 8),
])
def test_emulated_kernel_equals_dequantize_then_dot(n, k, g, m, bias, waves):
    u, s, x, b = make_case(n, k, g, m, bias, seed=n + k)
    qt, st = W.pack_u4(exllama_pack(u), s, g)
    got = emulate(x, qt, st, b, n, waves)
    ref, absref = reference(x, u, s, g, b)
    assert tuple(got.shape) == (m, n)
    assert ulp_ok(got, ref, absref)


def test_every_nibble_value_at_every_k_position_is_wired_to_its_own_weight():
    """x = one-hot over K, one row each: y[m, n] must be exactly (u[n, k_m] - 8) * scale -- catches any K mix-up."""
    n, k, g = 20, 128, 64
    u, s, _, _ = make_case(n, k, g, 1, False, seed=9)
    qt, st = W.pack_u4(exllama_pack(u), s, g)
    for k0 in range(0, k, 16):
        x = torch.zeros(16, k, dtype=torch.bfloat16)
        for m in range(16):
            x[m, k0 + m] = 1
        got = emulate(x, qt, st, None, n, 2).float()
        want = ((u.float() - 8) * s.float().repeat_interleave(g, dim=1))[:, k0:k0 + 16].T
        assert torch.equal(got, want.to(torch.bfloat16).float()), k0


# ======================================================================== (c) dispatcher

class FakeExt:
    """Stands in for the compiled module: rebuilds the tensors from the pointers and runs the emulation."""

    NCFG = 12

    def __init__(self):
        self.calls = []

    @staticmethod
    def _t(ptr, shape, dtype, stride_rows=None):
        nb = int(np.prod(shape)) * torch.empty(0, dtype=dtype).element_size()
        return torch.frombuffer((ctypes.c_char * nb).from_address(ptr), dtype=dtype).reshape(shape)

    def gemv_u4(self, x, q, s, y, bias, M, N, K, SX, SY, out32, ntiles, stream, cfg):
        self.calls.append(dict(x=x, q=q, s=s, y=y, bias=bias, M=M, N=N, K=K, SX=SX, SY=SY, out32=out32,
                               ntiles=ntiles, stream=stream, cfg=cfg))
        kg = K // 64
        xt = torch.stack([self._t(x + r * SX * 2, (K,), torch.bfloat16) for r in range(M)])
        qt = self._t(q, (ntiles, kg, 512), torch.uint8)
        st = self._t(s, (ntiles, kg, 32), torch.int16).view(torch.uint16)
        b = self._t(bias, (N,), torch.bfloat16) if bias else None
        out = emulate(xt, qt, st, b, N)
        for r in range(M):
            self._t(y + r * SY * 2, (N,), torch.bfloat16).copy_(out[r])


@pytest.fixture()
def disp(monkeypatch):
    fake = FakeExt()
    monkeypatch.setattr(W, "_ext", fake)
    monkeypatch.setattr(W, "ENABLED", True)
    monkeypatch.setattr(W, "_is_gpu", lambda t: True)
    monkeypatch.setattr(W, "_stream", lambda t: 7)
    monkeypatch.setattr(W, "_U4_CACHE", {})
    monkeypatch.setattr(W, "_U4_BYTES", 0)
    monkeypatch.setattr(W, "_SEEN", set())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)  # the CPU build has no CUDA runtime
    return W, fake


def layer(n=48, k=256, g=64, seed=0, form="int8"):
    u, s, x, b = make_case(n, k, g, 8, True, seed)
    w32 = exllama_pack(u)
    w = {"int8": w32.view(torch.int8), "tiled": tile_like_stock(w32)}[form]
    return u, w, s, x, b


@pytest.mark.parametrize("form", ["int8", "tiled"])
@pytest.mark.parametrize("m,bias,shape", [(8, True, None), (1, False, None), (16, True, None), (6, False, (2, 3))])
def test_dispatch_runs_the_kernel_and_matches_the_reference(disp, form, m, bias, shape):
    mod, fake = disp
    u, w, s, x, b = layer(form=form)
    x = x[:1].expand(m, -1).clone() * torch.linspace(0.5, 1.5, m)[:, None].to(torch.bfloat16)
    x_in = x.reshape(shape + (x.shape[-1],)) if shape else x
    y = mod.try_w4a16_hip_gemv(x_in, w, s, None, 64, b if bias else None)
    ref, absref = reference(x, u, s, 64, b if bias else None)
    assert tuple(y.shape) == tuple(x_in.shape[:-1]) + (48,) and y.dtype == torch.bfloat16
    assert ulp_ok(y.reshape(m, 48), ref, absref)
    (c,) = fake.calls
    assert (c["M"], c["N"], c["K"], c["out32"], c["cfg"], c["stream"]) == (m, 48, 256, 0, -1, 7)
    assert c["ntiles"] == 3 and (c["bias"] != 0) == bias and c["x"] % 16 == 0


def test_dispatch_copies_an_unaligned_activation(disp):
    mod, fake = disp
    u, w, s, x, b = layer()
    base = torch.zeros(8 * 256 + 8, dtype=torch.bfloat16)
    xo = base[1:1 + 8 * 256].view(8, 256)  # data_ptr % 16 == 2
    xo.copy_(x)
    y = mod.try_w4a16_hip_gemv(xo, w, s, None, 64, None)
    assert y is not None and fake.calls[0]["x"] % 16 == 0


@pytest.mark.parametrize("what", ["disabled", "no_ext", "act_fn", "zp", "x_fp16", "scale_fp16", "group_48",
                                  "m_17", "m_0", "k_not_64", "not_gpu", "scale_shape", "bias_fp32", "bias_len"])
def test_dispatch_returns_none_for_everything_it_does_not_own(disp, monkeypatch, what):
    mod, fake = disp
    u, w, s, x, b = layer()
    kw = dict(x=x, w_q=w, w_s=s, w_zp=None, group_size=64, bias=b, act_fn=None)
    if what == "disabled":
        monkeypatch.setattr(mod, "ENABLED", False)
    elif what == "no_ext":
        monkeypatch.setattr(mod, "_ext", None)
    elif what == "act_fn":
        kw["act_fn"] = "silu_and_mul"
    elif what == "zp":
        kw["w_zp"] = torch.zeros(6, 4, dtype=torch.int32)
    elif what == "x_fp16":
        kw["x"] = x.half()
    elif what == "scale_fp16":
        kw["w_s"] = s.half()
    elif what == "group_48":
        kw["group_size"] = 48
    elif what == "m_17":
        kw["x"] = torch.zeros(17, 256, dtype=torch.bfloat16)
    elif what == "m_0":
        kw["x"] = torch.zeros(0, 256, dtype=torch.bfloat16)
    elif what == "k_not_64":
        kw["x"] = torch.zeros(8, 224, dtype=torch.bfloat16)
    elif what == "not_gpu":
        monkeypatch.setattr(mod, "_is_gpu", lambda t: False)
    elif what == "scale_shape":
        kw["w_s"] = s[:, :3].contiguous()
    elif what == "bias_fp32":
        kw["bias"] = b.float()
    elif what == "bias_len":
        kw["bias"] = b[:10].contiguous()
    assert mod.try_w4a16_hip_gemv(**kw) is None
    assert not fake.calls and not mod._U4_CACHE


def test_env_gate_defaults_off_and_one_turns_it_on():
    code = "import radiance_w4a16_gemv_hip as W; print(W.ENABLED)"
    for val, want in ((None, "False"), ("0", "False"), ("1", "True")):
        env = {k: v for k, v in os.environ.items() if k != "RADIANCE_W4A16_HIP_GEMV"}
        if val is not None:
            env["RADIANCE_W4A16_HIP_GEMV"] = val
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=W4, env=env)
        assert r.stdout.strip().endswith(want), (val, r.stdout, r.stderr)


def test_cache_is_built_once_and_prints_one_proof_line_per_shape(disp, capsys):
    mod, fake = disp
    u, w, s, x, b = layer()
    for _ in range(3):
        assert mod.try_w4a16_hip_gemv(x, w, s, None, 64, b) is not None
    assert len(mod._U4_CACHE) == 1 and len(fake.calls) == 3
    err = capsys.readouterr().err
    assert err.count("[radiance.w4a16] gemv_u4 tiled 48x256 g64") == 1
    u2, w2, s2, _, _ = layer(seed=1)  # a second weight of the same shape: a second cache entry, no second line
    assert mod.try_w4a16_hip_gemv(x, w2, s2, None, 64, b) is not None
    assert len(mod._U4_CACHE) == 2 and "[radiance.w4a16]" not in capsys.readouterr().err


def test_cache_key_is_pointers_and_shape(disp):
    mod, _ = disp
    u, w, s, x, b = layer()
    mod.try_w4a16_hip_gemv(x, w, s, None, 64, b)
    assert list(mod._U4_CACHE) == [(w.data_ptr(), s.data_ptr(), 48, 256, 64)]


def test_no_allocation_while_the_stream_is_capturing(disp, monkeypatch):
    mod, fake = disp
    u, w, s, x, b = layer()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert mod.try_w4a16_hip_gemv(x, w, s, None, 64, b) is None  # not cached yet: stock path runs
    assert not mod._U4_CACHE and not fake.calls
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert mod.try_w4a16_hip_gemv(x, w, s, None, 64, b) is not None  # eager warm-up populates the cache
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert mod.try_w4a16_hip_gemv(x, w, s, None, 64, b) is not None  # a captured replay of a cached weight runs


def test_memory_budget_refuses_and_remembers(disp, monkeypatch, capsys):
    mod, fake = disp
    u, w, s, x, b = layer()
    monkeypatch.setattr(mod, "MAX_CACHE_MB", 0)
    assert mod.try_w4a16_hip_gemv(x, w, s, None, 64, b) is None
    assert mod.try_w4a16_hip_gemv(x, w, s, None, 64, b) is None
    assert not fake.calls and "budget" in capsys.readouterr().err
    assert mod._U4_BYTES == 0


def test_cache_bytes_are_accounted(disp):
    mod, _ = disp
    u, w, s, x, b = layer()
    mod.try_w4a16_hip_gemv(x, w, s, None, 64, b)
    assert mod._U4_BYTES == 3 * 4 * (512 + 64)  # tiles x groups x (code bytes + scale bytes)


# ======================================================================== (d) the vLLM patch

TARGET = "vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py"


@pytest.fixture()
def tree(tmp_path):
    if not UPSTREAM.exists():
        pytest.skip("pinned rdna_hybrid_w4a16.py not available")
    f = tmp_path / TARGET
    f.parent.mkdir(parents=True)
    shutil.copy(UPSTREAM, f)
    return tmp_path


def run_patch(script, tree):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(tree), PYTHONPATH=str(REPO))
    if script.name != PATCH.name:  # the older patches take site-packages from sysconfig: point it at the tree
        code = ("import sys, sysconfig, runpy; g = sysconfig.get_paths; "
                "sysconfig.get_paths = lambda *a, **k: {**g(*a, **k), 'purelib': sys.argv[1]}; "
                "sys.path.insert(0, sys.argv[3]); runpy.run_path(sys.argv[2], run_name='__main__')")
        return subprocess.run([sys.executable, "-c", code, str(tree), str(script), str(REPO)],
                              capture_output=True, text=True, env=env)
    return subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env)


def impl_source(path):
    src = path.read_text()
    fn = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "_rdna_hybrid_w4a16_apply_impl")
    return ast.get_source_segment(src, fn)


def test_patch_two_passes_ok_then_noop_and_parses(tree):
    r1 = run_patch(PATCH, tree)
    assert r1.returncode == 0 and "OK" in r1.stdout and "NOOP" not in r1.stdout, r1.stdout + r1.stderr
    first = (tree / TARGET).read_text()
    ast.parse(first)
    r2 = run_patch(PATCH, tree)
    assert r2.returncode == 0 and "NOOP" in r2.stdout and "OK " not in r2.stdout.replace("NOOP", ""), r2.stdout
    assert (tree / TARGET).read_text() == first


@pytest.mark.parametrize("order", ["tiles_then_hip", "hip_then_tiles"])
def test_patch_composes_with_the_tile_patch_in_either_order(tree, order):
    seq = [TILES_PATCH, PATCH] if order == "tiles_then_hip" else [PATCH, TILES_PATCH]
    for script in seq:
        r = run_patch(script, tree)
        assert r.returncode == 0 and "FAIL" not in r.stdout, (script.name, r.stdout, r.stderr)
    ast.parse((tree / TARGET).read_text())
    for script in seq:  # and both are idempotent after the other ran
        r = run_patch(script, tree)
        assert r.returncode == 0 and "OK " not in r.stdout.replace("NOOP", ""), (script.name, r.stdout)
    assert "try_w4a16_hip_gemv" in (tree / TARGET).read_text()


def exec_impl(tree, hip, **kw):
    """Run the patched _rdna_hybrid_w4a16_apply_impl with stubbed collaborators; returns what it returned."""
    import types
    ops = types.SimpleNamespace(wvSplitK_int4_g=lambda *a: "skinny")
    vllm = types.ModuleType("vllm")
    vllm._custom_ops = ops
    saved = {k: sys.modules.get(k) for k in ("vllm", "vllm._custom_ops")}
    sys.modules.update({"vllm": vllm, "vllm._custom_ops": ops})
    try:
        ns = dict(torch=torch, MAX_SKINNY_BATCH_SIZE=5, MEDIUM_SKINNY_LIMIT_ELEMENTS=10 ** 9, LDS_CAPACITY_ELEMENTS=10 ** 9,
                  _on_gfx115x=lambda: False, _max_skinny_batch_size=lambda: 5, _profile_scope=lambda s: __import__("contextlib").nullcontext(),
                  rocm_unquantized_gemm_impl=lambda *a: "dense", triton_w4a16_skinny_fmt_gemm=lambda **k: torch.zeros(1),
                  _pad_activation_rows=lambda x: x, MIN_DENSE_M=256, _radiance_hip_gemv=hip)
        exec(impl_source(tree / TARGET), ns)
        return ns["_rdna_hybrid_w4a16_apply_impl"](**kw)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class Hip:
    def __init__(self, out):
        self.out, self.args = out, None

    def try_w4a16_hip_gemv(self, *a, **k):
        self.args = (a, k)
        return self.out


@pytest.mark.parametrize("m", [1, 5, 8, 16])
def test_patched_impl_asks_the_hip_kernel_before_the_skinny_branch(tree, m):
    assert run_patch(PATCH, tree).returncode == 0
    x = torch.zeros(m, 256, dtype=torch.bfloat16)
    w = torch.zeros(48, 128, dtype=torch.int8)
    s = torch.zeros(48, 4, dtype=torch.bfloat16)
    bias = torch.zeros(48, dtype=torch.bfloat16)
    hip = Hip("hip")
    out = exec_impl(tree, hip, x_2d=x, w_q=w, w_s=s, w_zp=None, bias=bias, cu_count=64, group_size=64)
    assert out == "hip"
    (a, k) = hip.args
    assert k == {} and len(a) == 6 and a[0] is x and a[1] is w and a[2] is s and a[3] is None and a[4] == 64
    assert a[5] is bias


@pytest.mark.parametrize("m,want", [(1, "skinny"), (8, None)])
def test_patched_impl_falls_through_to_the_stock_path_when_the_kernel_declines(tree, m, want):
    assert run_patch(PATCH, tree).returncode == 0
    x = torch.zeros(m, 256, dtype=torch.bfloat16)
    w = torch.zeros(48, 128, dtype=torch.int8)
    s = torch.zeros(48, 4, dtype=torch.bfloat16)
    for hip in (Hip(None), None):  # declined, and the module missing from the image
        got = exec_impl(tree, hip, x_2d=x, w_q=w, w_s=s, w_zp=None, bias=None, cu_count=64, group_size=64)
        if want:
            assert got == want
        else:
            assert isinstance(got, torch.Tensor)  # the triton stub


def test_patch_leaves_the_fake_impl_and_the_registration_untouched(tree):
    before = (tree / TARGET).read_text()
    assert run_patch(PATCH, tree).returncode == 0
    after = (tree / TARGET).read_text()
    for name in ("_rdna_hybrid_w4a16_apply_fake", "direct_register_custom_op"):
        seg = lambda s: s[s.index(name):s.index(name) + 900]  # noqa: E731
        assert seg(before) == seg(after)
    assert after.count("try_w4a16_hip_gemv") == 1


def test_patch_fails_loud_when_the_target_has_drifted(tmp_path):
    f = tmp_path / TARGET
    f.parent.mkdir(parents=True)
    f.write_text("def something_else():\n    pass\n")
    r = run_patch(PATCH, tmp_path)
    assert r.returncode != 0 and "FAIL" in (r.stdout + r.stderr)


# ======================================================================== (e) the whole .hip against the compiler front end

def test_whole_hip_source_passes_a_host_syntax_check(tmp_path):
    """g++ -fsyntax-only over the unmodified kernel with stub headers: types, templates, lambdas, pybind glue.
    (The stubs only stand in for hip_runtime.h and the AMD builtins; vector typedefs are rewritten to GCC's form.)"""
    if not shutil.which("g++"):
        pytest.skip("no g++")
    inc_py = sysconfig.get_paths()["include"]
    inc_torch = Path(torch.__file__).parent / "include"
    if not (inc_torch / "pybind11" / "pybind11.h").exists() or not (Path(inc_py) / "Python.h").exists():
        pytest.skip("pybind11 / Python headers not available")
    (tmp_path / "hip").mkdir()
    (tmp_path / "hip" / "hip_runtime.h").write_text(r'''
#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <algorithm>
#define __global__
#define __device__
#define __shared__
#define __forceinline__ inline __attribute__((always_inline))
#define __launch_bounds__(x)
using std::min;
struct dim3 { unsigned x, y, z; dim3(unsigned a = 1, unsigned b = 1, unsigned c = 1) : x(a), y(b), z(c) {} };
static dim3 threadIdx, blockIdx, blockDim;
typedef void* hipStream_t;
static inline void __syncthreads() {}
static inline float __uint_as_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t __float_as_uint(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static inline uint32_t host_perm(uint32_t, uint32_t, uint32_t) { return 0; }
#define __builtin_amdgcn_perm host_perm
template <class A, class B, class C> static inline C host_wmma(A, B, C c) { return c; }
#define __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12 host_wmma
#define __builtin_nontemporal_load(p) (*(p))
#define hipLaunchKernelGGL(k, g, b, s, st, ...) ((void)(k)(__VA_ARGS__))
''')
    src = HIP.read_text()
    src = re.sub(r"typedef (__bf16|float|unsigned int) (\w+) __attribute__\(\(ext_vector_type\((\d+)\)\)\);",
                 lambda m: f"typedef {m[1]} {m[2]} __attribute__((vector_size({m[3]} * sizeof({m[1]}))));", src)
    assert "ext_vector_type" not in src
    (tmp_path / "k.cpp").write_text(src)
    r = subprocess.run(["g++", "-std=c++17", "-fsyntax-only", "-Wall", "-Wno-unused-function", "-Wno-ignored-attributes",
                        "-I", str(tmp_path), "-I", str(inc_torch), "-I", inc_py, str(tmp_path / "k.cpp")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]


def test_bench_script_parses_and_covers_the_qwen_shapes():
    src = (W4 / "bench_w4a16_gemv.py").read_text()
    ast.parse(src)
    for shape in ("6144x5120", "5120x4096", "5120x17408", "5120x25600", "248320x5120"):
        assert shape in src


KERNEL_PINS = [  # the lines the emulation above mirrors: an edit to one of them must be a deliberate, re-tested change
    "qp[t] = q + tile * ngroups * U4_GROUP_BYTES + lane * 16;",
    "sp[t] = s + tile * ngroups * U4_GROUP_SCALE_BYTES + c * 4;",
    "(const u32x4*)(qp[t] + (int64_t)g * U4_GROUP_BYTES)",
    "*(const uint32_t*)(sp[t] + (int64_t)g * U4_GROUP_SCALE_BYTES)",
    "const u32x4 w0 = u4_decode(k ? b.w[t][2] : b.w[t][0]);",
    "const u32x4 w1 = u4_decode(k ? b.w[t][3] : b.w[t][1]);",
    "const float sc = k ? sc_hi(b.sc[t]) : sc_lo(b.sc[t]);",
    "xr + g * U4_GROUP + k * 32 + j * 8",
    "(int64_t)(xlive ? c : 0) * SX + h * 16;",
    "const int g0 = (int)((int64_t)ngroups * wv / W);",
    "const int g1 = (int)((int64_t)ngroups * (wv + 1) / W);",
    "const int m = 8 * (l >> 4) + e, n = (tile0 + t) * 16 + (l & 15);",
    "if (bias) v += __uint_as_float((uint32_t)bias[n] << 16);",
    "tmp = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(bf16x8, xb.x[k][0]),",
    "tmp = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(bf16x8, xb.x[k][1]),",
    "__builtin_bit_cast(bf16x8, w0), tmp);",
    "__builtin_bit_cast(bf16x8, w1), tmp);",
    "dim3((ntiles + T - 1) / T), dim3(W * 32), (size_t)W * T * 1024,",
]


@pytest.mark.parametrize("pin", KERNEL_PINS)
def test_kernel_addressing_lines_are_pinned(pin):
    assert HIP.read_text().count(pin) == 1


def test_kernel_layout_constants_agree_with_the_python_packer():
    src = HIP.read_text()
    c = {m[1]: int(m[2]) for m in re.finditer(r"#define (U4_\w+|MAX_M) (\d+)", src)}
    assert (c["U4_TILE"], c["U4_GROUP"], c["U4_GROUP_BYTES"], c["MAX_M"]) == (W.TILE, W.GROUP, 512, W.MAX_M)
    assert c["U4_GROUP_BYTES"] + c["U4_GROUP_SCALE_BYTES"] == W.TILE_BYTES
    assert c["U4_NCFG"] == len(re.findall(r"^    case \d+: run_u4<", src, re.M)) == 12
