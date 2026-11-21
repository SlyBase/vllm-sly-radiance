"""GPU micro-benchmark of the Inkling 2 expert GEMMs at the real decode shapes (run in the image).

Reports time per call and achieved weight bandwidth (GB/s) for a config scan, so the kernel tuning is
judged against the card's memory bandwidth, not against another kernel.
    python bench_experts.py nvfp4|mxfp8 [rows_per_expert]
Shapes (Inkling-2-Mini): hidden 2304, expert intermediate 1152 -> w13 (N 2304, K 2304), w2 (N 2304, K 1152);
decode = 8 active experts, one token each (tile = rows per expert).
"""
import itertools, sys, types
import torch
import triton


def timeit(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1000  # us


def nvfp4(rows):
    from radiance_inkling2_moe import nvfp4_expert_gemm
    dev, E = "cuda", 8
    for N, K in ((2304, 2304), (2304, 1152)):
        w = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=dev)
        ws = (torch.rand(E, N, K // 16, device=dev) + 0.5).to(torch.float8_e4m3fn)
        tile = max(rows, 8)
        R = tile * E
        act = torch.randint(0, 256, (R, K // 2), dtype=torch.uint8, device=dev)
        asc = (torch.rand(R * K // 16 + 4096, device=dev) + 0.5).to(torch.float8_e4m3fn)
        rs, es = torch.ones(R, device=dev), torch.ones(E, device=dev)
        tiles = torch.cat([torch.arange(E, dtype=torch.int32), torch.full((E,), rows, dtype=torch.int32) + torch.arange(E, dtype=torch.int32) * tile, torch.tensor([E], dtype=torch.int32)]).to(dev)
        plan = types.SimpleNamespace(tile=tile, max_ctas=E, rows=R)
        nbytes = E * N * (K // 2 + K // 16)
        res = []
        for bm, bn, bk, nw, ns in itertools.product((16, 32), (32, 64, 128), (64, 128, 256), (2, 4, 8), (1, 2)):
            if bm > triton.next_power_of_2(tile) * 4:
                continue
            try:
                us = timeit(lambda: nvfp4_expert_gemm(act, asc, rs, w, ws, es, tiles, plan, bm, bn, bk, num_warps=nw, num_stages=ns))
            except Exception as e:
                continue
            res.append((us, bm, bn, bk, nw, ns))
        res.sort()
        print(f"NVFP4 N={N} K={K} rows/expert={rows}: bytes {nbytes/1e6:.1f} MB")
        for us, bm, bn, bk, nw, ns in res[:3] + res[-1:]:
            print(f"  {us:8.1f} us  {nbytes/us/1e3:6.1f} GB/s  bm{bm} bn{bn} bk{bk} warps{nw} stages{ns}")
        d = [r for r in res if (r[1], r[2], r[3], r[4], r[5]) == (min(32, triton.next_power_of_2(tile)), 64, min(128, K), 4, 1)]
        print("  shipped-default-like:", d[:1])


def mxfp8(rows):
    from vllm.models.inkling2.nvidia.ops.moe_sm120 import _mxfp8_gemm
    dev, E = "cuda", 8
    for N, K in ((2304, 2304), (2304, 1152)):
        w = torch.randint(0, 120, (E, N, K), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        ws = torch.full((E, N, K // 32), 127, dtype=torch.uint8, device=dev)
        tile = max(rows, 8)
        R = tile * E
        act = torch.randint(0, 120, (R, K), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        asc = torch.full((R * K // 32 + 4096,), 127, dtype=torch.uint8, device=dev)
        ones = torch.ones(E, device=dev)
        tiles = torch.cat([torch.arange(E, dtype=torch.int32), torch.full((E,), rows, dtype=torch.int32) + torch.arange(E, dtype=torch.int32) * tile, torch.tensor([E], dtype=torch.int32)]).to(dev)
        out = torch.empty(R, N, dtype=torch.bfloat16, device=dev)
        nbytes = E * N * (K + K // 32)
        res = []
        for bn, bt, nw, ns in itertools.product((32, 64, 128), (8, 16, 32), (2, 4, 8), (1, 2)):
            try:
                us = timeit(lambda: _mxfp8_gemm[(triton.cdiv(N, bn), E * triton.cdiv(tile, bt))](
                    act, asc, w, ws, ones, out, tiles, MAX_CTAS=E, N=N, K=K, TILE=tile, BN=bn, BT=bt, BK=128, num_warps=nw, num_stages=ns))
            except Exception:
                continue
            res.append((us, bn, bt, nw, ns))
        res.sort()
        print(f"MXFP8 N={N} K={K} rows/expert={rows}: bytes {nbytes/1e6:.1f} MB")
        for us, bn, bt, nw, ns in res[:3] + res[-1:]:
            print(f"  {us:8.1f} us  {nbytes/us/1e3:6.1f} GB/s  bn{bn} bt{bt} warps{nw} stages{ns}")


if __name__ == "__main__":
    {"nvfp4": nvfp4, "mxfp8": mxfp8}[sys.argv[1]](int(sys.argv[2]) if len(sys.argv) > 2 else 1)
