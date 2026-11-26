"""GPU micro-benchmark of the Inkling 2 expert GEMMs at the real decode shapes (run in the image).

Reports time per call and achieved weight bandwidth (GB/s) so the kernels are judged against the
card's memory bandwidth, not against each other.
    python bench_experts.py compare [fmt ...]   HIP vs Triton, NVFP4 + MXFP8, w13 + w2, decode and spec
    python bench_experts.py nvfp4|mxfp8 [rows]  Triton config scan (8 experts x `rows` rows)
Shapes (Inkling-2-Mini): hidden 2304, expert intermediate 1152 -> w13 (N 2304, K 2304), w2 (N 2304, K 1152).
`compare` steps: 1 token = 8 CTAs x 1 row; 2 / 4 / 8 tokens use the MEASURED routing union of
13 / 20 / 29 distinct experts (0.79 / 0.69 / 0.62 / 0.46 x 8 w).  Weights are a pool > 300 MB and every
call reads a different window of it, so the 64 MB Infinity Cache never serves a weight twice in a row.
GB/s counts the distinct expert bytes (weights + block scales) of the call.  Each (fmt, projection)
runs in its own process with a time limit; per-call time comes from a replayed graph of one call per
pool window (the way vLLM runs decode), so Python launch overhead is not part of it.
"""
import itertools, json, os, subprocess, sys, types
import torch
import triton

SHAPES = {"w13": (2304, 2304), "w2": (2304, 1152)}
UNION = {1: 8, 2: 13, 4: 20, 8: 29}      # tokens -> distinct experts of the step (decode: 8)
POOL_BYTES = 320 << 20


def rows_per_cta(tokens):
    """tokens * 8 routed rows spread over the union of CTAs, no CTA above 8 rows."""
    ctas = UNION[tokens]
    base, extra = divmod(8 * tokens, ctas)
    return [base + (i < extra) for i in range(ctas)]


def graph_us(fn, slots, reps=8):
    """GPU us per call: one graph holding `slots` calls (a different pool window each)."""
    for i in range(slots):
        fn(i)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(slots):
            fn(i)
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / (reps * slots) * 1000


def make_pool(fmt, n, k, dev):
    nvfp4 = fmt == "nvfp4"
    wb, sb = (k // 2, k // 16) if nvfp4 else (k, k // 32)
    experts = max(16, -(-POOL_BYTES // (n * (wb + sb))))
    if nvfp4:
        w = torch.randint(0, 256, (experts, n, wb), dtype=torch.uint8, device=dev)
        ws = (torch.rand(experts, n, sb, device=dev) * 3.75 + 0.25).to(torch.float8_e4m3fn)
    else:
        w = torch.randint(0, 120, (experts, n, wb), dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn)
        ws = torch.randint(120, 135, (experts, n, sb), dtype=torch.uint8, device=dev)
    return w, ws, experts, n * (wb + sb)


def run_case(fmt, proj):
    import radiance_inkling2_moe as m
    n, k = SHAPES[proj]
    nvfp4 = fmt == "nvfp4"
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(0)
    w, ws, experts, bytes_per_expert = make_pool(fmt, n, k, dev)
    real_hip = m._hip
    assert real_hip is not None, "radiance_inkling2_moe_hip is not importable"
    for tokens in (1, 2, 4, 8):
        rows_per = rows_per_cta(tokens)
        count, max_ctas = len(rows_per), 8 * tokens
        rows = 8 * max_ctas
        plan = types.SimpleNamespace(tile=8, max_ctas=max_ctas, rows=rows)
        ab, sg = (k // 2, k // 16) if nvfp4 else (k, k // 32)
        act = torch.randint(0, 120, (rows, ab), dtype=torch.uint8, device=dev)
        if nvfp4:
            act = torch.randint(0, 256, (rows, ab), dtype=torch.uint8, device=dev)
            asc = (torch.rand(rows * sg, device=dev) * 3.75 + 0.25).to(torch.float8_e4m3fn)
            rs, es = torch.rand(rows, device=dev) + 0.5, torch.rand(experts, device=dev) + 0.5
        else:
            act = act.view(torch.float8_e4m3fn)
            asc = torch.randint(120, 135, (rows * sg,), dtype=torch.uint8, device=dev)
            es = torch.rand(experts, device=dev) + 0.5
        slots = max(2, experts // count)               # window s = experts [s*count, (s+1)*count)
        tl = []
        for s in range(slots):
            ids = [(s * count + t) % experts for t in range(count)] + [0] * (max_ctas - count)
            ends = [t * 8 + (rows_per[t] if t < count else 0) for t in range(max_ctas)]
            tl.append(torch.tensor(ids + ends + [count, sum(rows_per)], dtype=torch.int32, device=dev))
        distinct = count * bytes_per_expert

        def call(impl, i):
            t = tl[i % slots]
            if nvfp4:
                kw = dict(block_m=16, block_n=64, block_k=64, num_warps=4, num_stages=2) \
                    if impl == "triton_tuned" else {}
                return m.nvfp4_expert_gemm(act, asc, rs, w, ws, es, t, plan, **kw)
            return m.mxfp8_expert_gemm(act, asc, w, ws, es, t, plan)

        impls = ["hip", "triton"] + (["triton_tuned"] if nvfp4 else [])
        impls = [i for i in impls if i in os.environ.get("BENCH_IMPLS", ",".join(impls)).split(",")]
        ref = None
        for impl in impls:
            m._hip = real_hip if impl == "hip" else None
            try:
                out = call(impl, 0).float()
                torch.cuda.synchronize()
                valid = torch.zeros(rows, dtype=torch.bool, device=dev)
                for t_, r_ in enumerate(rows_per):
                    valid[t_ * 8:t_ * 8 + r_] = True
                out = out[valid]
                if ref is None:
                    ref = out                              # hip is first: the others are checked against it
                    diff, nbad = 0.0, 0
                else:
                    diff = ((out - ref).abs().max() / ref.abs().max()).item()
                    nbad = int((out != ref).sum())         # bf16 results that differ at all
                us = graph_us(lambda i: call(impl, i), min(slots, 16))
                print(json.dumps(dict(fmt=fmt, proj=proj, tokens=tokens, ctas=count, impl=impl,
                                      us=round(us, 1), gbs=round(distinct / us / 1e3, 1),
                                      bytes_mb=round(distinct / 1e6, 1), vs_hip_maxrel=float(f"{diff:.2g}"),
                                      vs_hip_differing=f"{nbad}/{out.numel()}")),
                      flush=True)
            except Exception as e:
                print(json.dumps(dict(fmt=fmt, proj=proj, tokens=tokens, impl=impl,
                                      error=f"{type(e).__name__}: {str(e)[:200]}")), flush=True)
        m._hip = real_hip


def compare(fmts, limit=240):
    rows = []
    for fmt in fmts:
        for proj in SHAPES:
            try:
                r = subprocess.run([sys.executable, __file__, "case", fmt, proj], capture_output=True,
                                   text=True, timeout=limit)
                out = r.stdout
                if r.returncode:
                    print(f"# {fmt} {proj}: rc {r.returncode}: {r.stderr[-300:]}", flush=True)
            except subprocess.TimeoutExpired as e:
                out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
                print(f"# {fmt} {proj}: time limit {limit}s", flush=True)
            for line in out.splitlines():
                if line.startswith("{"):
                    print(line, flush=True)
                    rows.append(json.loads(line))
    by = {}
    for r in rows:
        by.setdefault((r["fmt"], r["proj"], r["tokens"]), {})[r["impl"]] = r
    print("\n| format | proj | tokens | CTAs | MB | HIP us | HIP GB/s | Triton us | Triton GB/s | tuned Triton us | speedup | Triton vs HIP (bf16 elements that differ / max rel) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (fmt, proj, tok), d in sorted(by.items()):
        h, t, tt = d.get("hip", {}), d.get("triton", {}), d.get("triton_tuned", {})
        best = min([x["us"] for x in (t, tt) if "us" in x], default=None)
        sp = f"{best / h['us']:.1f}x" if best and "us" in h else "-"
        print(f"| {fmt} | {proj} | {tok} | {h.get('ctas', '-')} | {h.get('bytes_mb', '-')} | {h.get('us', '-')} | "
              f"{h.get('gbs', '-')} | {t.get('us', '-')} | {t.get('gbs', '-')} | {tt.get('us', '-')} | {sp} | "
              f"{t.get('vs_hip_differing', '-')} / {t.get('vs_hip_maxrel', '-')} |")




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
    import radiance_inkling2_moe
    radiance_inkling2_moe._hip = None                  # the scan is of the Triton kernel
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
    if sys.argv[1] == "compare":
        compare(sys.argv[2:] or ["nvfp4", "mxfp8"])
    elif sys.argv[1] == "case":
        run_case(sys.argv[2], sys.argv[3])
    else:
        {"nvfp4": nvfp4, "mxfp8": mxfp8}[sys.argv[1]](int(sys.argv[2]) if len(sys.argv) > 2 else 1)
