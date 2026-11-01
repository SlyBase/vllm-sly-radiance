#!/usr/bin/env python3
"""CPU-only checks of radiance_r4d_hybrid_attn (no GPU, no kernel launch). Run inside the image:

  docker run --rm -e HIP_VISIBLE_DEVICES=-1 -v /root/lessons/H:/h --entrypoint bash \
      vllm-sly-radiance:0.7.0-rocm10.0 -c 'cd /h && python selftest_cpu.py'

1. _plan: run cutting, thresholds, padding requests, short runs around long ones.
2. The sub-block addressing trick: libr4d reads `blk * block_stride + head * head_stride + slot * C`
   with 16-slot blocks; the hybrid hands it block_stride = 16*C and ids b*mult + j for manager block b,
   sub-block j. Checked against the plain (block, head, slot, c) index of a cache with block 64 slots,
   for the LBHNC strides and for an LHBNC-style permutation.
3. _block_table16 shape / values / sharing via the metadata cache.
"""
import sys
import types

import torch

import radiance_r4d_hybrid_attn as H

fails = 0


def check(name, ok):
    global fails
    print(("PASS " if ok else "FAIL ") + name)
    fails += 0 if ok else 1


# 1. plan ---------------------------------------------------------------------------------------
def qsl(lens):
    out = [0]
    for n in lens:
        out.append(out[-1] + n)
    return torch.tensor(out, dtype=torch.int32)


check("pure decode -> None", H._plan(qsl([8] * 8), 8, 512) is None)
check("short prefill (400) -> None", H._plan(qsl([400]), 1, 512) is None)
check("one long chunk", H._plan(qsl([2048]), 1, 512) == (("r4d", 0, 1, 2048, 0),))
check(
    "decodes then a long chunk",
    H._plan(qsl([8, 8, 2048]), 3, 512) == (("aiter", 0, 2, 0, 16, 8), ("r4d", 2, 1, 2048, 16)),
)
check(
    "long, decode, tail",
    H._plan(qsl([1024, 8, 300]), 3, 512) == (("r4d", 0, 1, 1024, 0), ("aiter", 1, 2, 1024, 308, 300)),
)
check(
    "two equal long requests batch into one run",
    H._plan(qsl([512, 512, 8]), 3, 512) == (("r4d", 0, 2, 512, 0), ("aiter", 2, 1, 1024, 8, 8)),
)
check(
    "two different long requests -> two launches",
    H._plan(qsl([512, 1024]), 2, 512) == (("r4d", 0, 1, 512, 0), ("r4d", 1, 1, 1024, 512)),
)
p = H._plan(qsl([2048, 0, 0]), 3, 512)  # graph-capture style padding after the real request
check("trailing empty requests dropped", p == (("r4d", 0, 1, 2048, 0),))
p = H._plan(qsl([8, 0, 2048]), 3, 512)
check("empty request inside a short run kept in it", p == (("aiter", 0, 2, 0, 8, 8), ("r4d", 2, 1, 2048, 8)))
check("threshold is >=", H._plan(qsl([512]), 1, 512) is not None and H._plan(qsl([511]), 1, 512) is None)

# 2. addressing -----------------------------------------------------------------------------------
HEADS, N, C = 4, 64, 8  # tiny cache: 64-slot manager blocks, 4 kv heads, content 8
BLOCKS = 7
ratio = N // 16


def emulate(kv, shape, strides, bt, mult, block_stride, head_stride, req_len):
    """What libr4d computes: element offset of key position kk of head h, via the 16-slot table."""
    flat = kv.flatten()
    out = []
    for h in range(HEADS):
        for kk in range(req_len):
            blk = int(bt[kk // 16])
            off = blk * block_stride + h * head_stride + (kk % 16) * C
            out.append(flat[off])
    return torch.stack(out)


def reference(kv_view, bt_mgr, req_len):
    out = []
    for h in range(HEADS):
        for kk in range(req_len):
            out.append(kv_view[int(bt_mgr[kk // N]), h, kk % N, 0])
    return torch.stack(out)


for name, perm in (("LBHNC", (0, 1, 2, 3)), ("LHBNC", (1, 0, 2, 3))):
    # physical storage (dims in `perm` order of [B, H, N, C]); the per-layer view is permuted back
    base_shape = [BLOCKS, HEADS, N, C]
    phys_shape = [base_shape[i] for i in perm]
    storage = torch.arange(int(torch.tensor(phys_shape).prod()), dtype=torch.int64).reshape(phys_shape)
    inv = [perm.index(i) for i in range(4)]
    view = storage.permute(*inv)  # (B, H, N, C)
    s0, s1, s2, s3 = view.stride()
    unit = 16 * C
    ok = s2 == C and s3 == 1 and s0 % unit == 0
    check(f"{name}: strides admit the sub-block trick {view.stride()}", ok)
    mult = s0 // unit
    bt_mgr = torch.tensor([5, 2, 6, 0, 3], dtype=torch.int32)
    meta = types.SimpleNamespace(
        block_table=bt_mgr.unsqueeze(0), max_seq_len=N * 3, hyb_cache=None, hyb_sub={}
    )
    table, lo = H.R4DHybridAttentionImpl._block_table16(meta, (("r4d", 0, 1, 100, 0),), mult, ratio)
    check(f"{name}: table shape", tuple(table.shape) == (1, bt_mgr.numel() * ratio) and lo == 0)
    t2, _ = H.R4DHybridAttentionImpl._block_table16(meta, (("r4d", 0, 1, 100, 0),), mult, ratio)
    check(f"{name}: table shared across layers", t2 is table)
    req_len = N * 3 + 20
    got = emulate(storage, None, None, table[0], mult, unit, s1, req_len)
    want = reference(view, bt_mgr, req_len)
    check(f"{name}: addresses == plain (block, head, slot) indexing over {req_len} keys", bool(torch.equal(got, want)))

print("FAILED" if fails else "ALL OK")
sys.exit(1 if fails else 0)
