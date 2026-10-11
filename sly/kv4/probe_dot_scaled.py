#!/usr/bin/env python3
"""Does Triton's own MXFP4 emulation (tl.dot_scaled with an e2m1 operand) beat a hand-unpacked
4-bit operand on gfx1201? Micro-benchmark of the inner product only: a (64, 128) query tile
against a (128, TILE) key tile, looped over N tiles from a packed cache, for

  fp8   : tl.dot(q_fp8, k_fp8)                                 (what the fp8 kernel runs)
  i8    : tl.dot(q_i8, (k_u8 & 0xF).to(i8))                    (KV4's K path)
  mxfp4 : tl.dot_scaled(q_bf16, None, "bf16", k_e2m1, scale_e8m0, "e2m1")

  python3 sly/kv4/probe_dot_scaled.py [--tile 16] [--ntiles 2048]
"""
import argparse
import sys

import torch
import triton
import triton.language as tl

dev = "cuda"


@triton.jit
def _probe(q_ptr, k_ptr, s_ptr, out_ptr, ntiles,
           TILE: tl.constexpr, MODE: tl.constexpr, num_warps: tl.constexpr):
    om = tl.arange(0, 64)
    if MODE == 2:
        Q = tl.load(q_ptr + om[:, None] * 128 + tl.arange(0, 128)[None, :])          # bf16 (64, 128)
        acc = tl.zeros([64, TILE], dtype=tl.float32)
        pk = tl.arange(0, 64)                                                        # packed K rows (128 // 2)
        for j in range(0, ntiles):
            K = tl.load(k_ptr + j * (64 * TILE) + pk[:, None] * TILE + tl.arange(0, TILE)[None, :])   # u8 (64, TILE)
            S = tl.load(s_ptr + j * (TILE * 4) + tl.arange(0, TILE)[:, None] * 4 + tl.arange(0, 4)[None, :])  # e8m0 (TILE, 4)
            acc += tl.dot_scaled(Q, None, "bf16", K, S, "e2m1")
    else:
        Q = tl.load(q_ptr + om[:, None] * 128 + tl.arange(0, 128)[None, :])          # fp8 or i8 (64, 128)
        acc = tl.zeros([64, TILE], dtype=tl.float32)
        pk = tl.arange(0, 128)
        for j in range(0, ntiles):
            K = tl.load(k_ptr + j * (128 * TILE) + pk[:, None] * TILE + tl.arange(0, TILE)[None, :])
            if MODE == 0:
                acc += tl.dot(Q, K.to(tl.float8e4nv, bitcast=True))
            else:
                acc += tl.dot(Q, (K & 0xF).to(tl.int8)).to(tl.float32)
    tl.store(out_ptr + om[:, None] * TILE + tl.arange(0, TILE)[None, :], acc)


def run(mode, tile, ntiles, warps, reps=20):
    out = torch.empty(64, tile, dtype=torch.float32, device=dev)
    if mode == 2:
        q = torch.randn(64, 128, device=dev).to(torch.bfloat16)
        k = torch.randint(0, 256, (ntiles, 64, tile), dtype=torch.uint8, device=dev)
        s = torch.full((ntiles, tile, 4), 127, dtype=torch.uint8, device=dev)
    else:
        q = torch.randn(64, 128, device=dev).to(torch.float8_e4m3fn) if mode == 0 else \
            torch.randint(-127, 128, (64, 128), dtype=torch.int8, device=dev)
        k = torch.randint(0, 256, (ntiles, 128, tile), dtype=torch.uint8, device=dev)
        s = out
    grid = (128,)   # 128 workgroups, each walks all tiles: like one head x 32 splits x 4
    fn = lambda: _probe[grid](q, k, s, out, ntiles, TILE=tile, MODE=mode, num_warps=warps)  # noqa: E731
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        return f"FAILED: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()   # noqa: E702
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return f"{ts[len(ts) // 2]:8.1f} us"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--ntiles", type=int, default=2048)
    ap.add_argument("--warps", type=int, default=8)
    a = ap.parse_args()
    print(f"tile {a.tile}, {a.ntiles} tiles per workgroup, {a.warps} warps:")
    for mode, name in ((0, "fp8 dot"), (1, "i8 dot (masked nibble)"), (2, "dot_scaled e2m1 (Triton MXFP4 emulation)")):
        print(f"  {name:45s} {run(mode, a.tile, a.ntiles, a.warps)}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
