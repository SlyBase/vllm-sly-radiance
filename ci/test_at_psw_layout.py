"""Self-check of the A-tiled PSW byte layout (radiance_mxfp4.psw_pack_w / psw_fold_ws).
`pytest ci/` or run directly. CPU only -- both functions are pure tensor bit shuffles.

The kernel (sly/mxfp4/radiance_mxfp4_fp8.hip, PSW clause) reads one lane word of 8 nibbles in
fragment order (byte b = e[2b] | e[2b+1] << 4) and expects, per byte b of the packed word:
  PSW=1: byte b = e[b] | e[b+4] << 4
  PSW=2: byte b bits 0-2 mag(e[b]), bit 7 sign(e[b]), bits 4-6 mag(e[b+4]),
         bit 3 sign(e[(b+3)%4 + 4])   (so alignbit(w, w, 4) carries e[b+4], sign at bit 7)
Both modes are permutations of the same 32 bits -- nothing is requantized, which is why the GEMM
output stays bit-identical (DIGEST) and why the pack costs no memory (same shape, done in place).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sly", "mxfp4"))
import radiance_mxfp4 as R  # noqa: E402


def elems(packed):
    """[N, K/2] packed bytes -> the eight nibbles per lane word, e[0..7]."""
    w = packed.reshape(-1, 4).to(torch.int64)
    return torch.stack([w & 15, w >> 4], -1).reshape(-1, 8)


def test_mode1_is_the_documented_half_word_interleave():
    w = torch.randint(0, 256, (64, 3), dtype=torch.uint8)
    e = elems(w)
    ref = (e[:, 0:4] | (e[:, 4:8] << 4)).to(torch.uint8)
    assert torch.equal(R.psw_pack_w(w, 1).reshape(-1, 4), ref)
    # round trip: byte b is e[b] | e[b+4] << 4
    p = R.psw_pack_w(w, 1).reshape(-1, 4).to(torch.int64)
    assert torch.equal(torch.cat([p & 15, p >> 4], 1), e)


def test_mode2_is_a_bit_permutation():
    """All 32 single-bit words must map to 32 distinct single-bit words: nothing is merged."""
    seen = set()
    for bit in range(32):
        v = 1 << bit
        word = torch.tensor([[v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >> 24) & 255]],
                            dtype=torch.uint8)
        out = int(R.psw_pack_w(word, 2).reshape(-1).to(torch.int64)
                  @ torch.tensor([1, 1 << 8, 1 << 16, 1 << 24]))
        assert bin(out).count("1") == 1, f"bit {bit} -> {out:#x} is not a single bit"
        seen.add(out)
    assert len(seen) == 32


def test_mode2_places_sign_and_magnitude_as_documented():
    w = torch.randint(0, 256, (128, 3), dtype=torch.uint8)
    e = elems(w)
    lo, hi = e[:, 0:4], e[:, 4:8]
    ref = (lo & 7) | (((hi.roll(1, 1) >> 3) & 1) << 3) | ((hi & 7) << 4) | ((lo >> 3) << 7)
    assert torch.equal(R.psw_pack_w(w, 2).reshape(-1, 4), ref.to(torch.uint8))


def test_mode2_round_trip_recovers_every_nibble():
    w = torch.randint(0, 256, (128, 3), dtype=torch.uint8)
    e = elems(w)
    p = R.psw_pack_w(w, 2).reshape(-1, 4).to(torch.int64)
    lo = (p & 7) | (((p >> 7) & 1) << 3)                   # e[0..3]: mag bits 0-2, sign bit 7
    hi = ((p >> 4) & 7) | (((p.roll(-1, 1) >> 3) & 1) << 3)  # e[4..7]: sign of e[b+4] in byte b+1
    assert torch.equal(torch.cat([lo, hi], 1), e)


def test_fold_ws_is_the_kmag_row_offset():
    ws = torch.randint(0, 32, (4, 96), dtype=torch.uint8)
    ref = torch.randint(0, 32, (96,), dtype=torch.uint8)
    out = R.psw_fold_ws(ws, ref)
    d = (ref.to(torch.int64)[None, :] - ws.to(torch.int64)).clamp(0, 15) * 8
    assert out.shape == ws.shape and out.dtype == torch.uint8
    assert torch.equal(out.to(torch.int64), d)
    assert int(out.max()) <= 120 and int(out.min()) >= 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
