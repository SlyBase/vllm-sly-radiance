"""Standalone A/B of the norm/act-quant producers (stock vs RADIANCE_FUSED_NQ_REG) on REAL activations.

Inputs: dump_<kernel>_<M>.pt written by ci/prefill_nq_ab.py CHECK=1 (one real forward pass). M values not
dumped are built by stacking real rows (M=4096 from the largest dump). Inputs are rotated over copies
totalling >= 256 MB so no call is served from the 64 MB last-level cache. Arms run in ABBA order per
cell, hipEvent timing, median of REPS launches per arm block. --check compares q / scale / residual bytes
of both arms and prints a sha256 DIGEST of the stock bytes.

  bench_nq_producers.py DUMP_DIR [--ms 880,1514,1760,4096] [--reps 50] [--tiled 1]
"""
import argparse, glob, hashlib, os, re, statistics

import torch
import radiance_mxfp4_fp8 as ext

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--ms", default="880,1514,1760,4096")
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--tiled", type=int, default=1)
ap.add_argument("--random", action="store_true", help="uniform random bf16 instead of the dump")
a = ap.parse_args()
dev = "cuda"
st = torch.cuda.current_stream().cuda_stream
U8 = torch.uint8


def load(kind):
    fs = sorted(glob.glob(f"{a.dump}/dump_{kind}_*.pt"), key=lambda f: int(re.findall(r"_(\d+)\.pt", f)[0]))
    return torch.load(fs[-1]) if fs else None


def rows(t, M, rowbytes):
    """Real rows, repeated to M rows (bytes view)."""
    r = t.view(-1, rowbytes)
    reps = (M + r.shape[0] - 1) // r.shape[0]
    return r.repeat(reps, 1)[:M].contiguous().view(-1)


def copies(nbytes):
    return max(2, (256 << 20) // max(nbytes, 1) + 1)


def cell(kind, M):
    d = load(kind)
    if d is None:
        return None
    if kind == "silu_mul_quant":
        N = 17408
        gu = d["gu"]
        if a.random:
            gu = (torch.rand(gu.numel() // 2) * 8 - 4).to(torch.bfloat16).view(U8)
        gu = rows(gu, M, 2 * N * 2)
        nb = gu.numel()
        ins = [gu.to(dev) for _ in range(copies(nb))]
        qn = ((M + 15) // 16 * 16 if a.tiled else M) * N
        outs = [torch.empty(qn, dtype=U8, device=dev), torch.empty(4 * M, dtype=U8, device=dev)]

        def run(i, o):
            ext.launch_silu_mul_quant(gu=ins[i % len(ins)].data_ptr(), q=o[0].data_ptr(), scale=o[1].data_ptr(),
                                      M=M, N=N, stream=st, tiled=a.tiled)
        return run, outs, nb + qn + 4 * M
    if kind == "add_rms_quant":
        K = d["w"].numel() // 2
        y, r = rows(d["y"], M, 2 * K), rows(d["res"], M, 2 * K)
        if a.random:
            y = (torch.randn(M * K)).to(torch.bfloat16).view(U8)
            r = (torch.randn(M * K)).to(torch.bfloat16).view(U8)
        nb = 2 * y.numel()
        ins = [(y.to(dev), r.to(dev)) for _ in range(copies(nb))]
        w = d["w"].to(dev)
        qn = ((M + 15) // 16 * 16 if a.tiled else M) * K
        outs = [torch.empty(qn, dtype=U8, device=dev), torch.empty(4 * M, dtype=U8, device=dev),
                torch.empty(2 * M * K, dtype=U8, device=dev)]

        def run(i, o):
            yy, rr = ins[i % len(ins)]
            ext.launch_add_rms_quant(yy.data_ptr(), rr.data_ptr(), w.data_ptr(), o[0].data_ptr(), o[1].data_ptr(),
                                     o[2].data_ptr(), M, K, 1e-6, st, a.tiled)
        return run, outs, nb + qn + 4 * M + 2 * M * K
    if kind == "gdn_norm_quant":
        N = 6144
        x = rows(d["x"], M, 2 * N)
        # z is a strided column view in production; the dump holds the strided span: take N per row.
        zs = (d["z"].numel() // 2 - N) // max(1, (d["x"].numel() // (2 * N)) - 1)
        zr = torch.cat([d["z"].view(-1), torch.zeros(2 * (zs - N), dtype=U8)]).view(-1, 2 * zs)[:, :2 * N]
        z = rows(zr.contiguous().view(-1), M, 2 * N)
        if a.random:
            x = torch.randn(M * N).to(torch.bfloat16).view(U8)
            z = torch.randn(M * N).to(torch.bfloat16).view(U8)
        nb = x.numel() + z.numel()
        ins = [(x.to(dev), z.to(dev)) for _ in range(copies(nb))]
        w = d["w"].to(dev)
        tiled = 0  # production default (TILED_OPROJ off)
        outs = [torch.empty(M * N, dtype=U8, device=dev), torch.empty(4 * M, dtype=U8, device=dev)]

        def run(i, o):
            xx, zz = ins[i % len(ins)]
            ext.launch_gdn_norm_quant(xx.data_ptr(), zz.data_ptr(), N, w.data_ptr(), o[0].data_ptr(),
                                      o[1].data_ptr(), M, N, 1e-6, st, tiled)
        return run, outs, nb + M * N + 4 * M
    return None


def timed(run, outs, reg):
    ext.set_nq_reg(reg)
    for i in range(3):
        run(i, outs)
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(a.reps)]
    for i, (s, e) in enumerate(ev):
        s.record()
        run(i, outs)
        e.record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1e3 for s, e in ev)


print(f"# data={'random' if a.random else 'real dump'} tiled={a.tiled} reps={a.reps}")
print("kernel M | stock_us REG_us (ABBA medians) | ratio | stock GB/s REG GB/s | DIGEST equal")
for kind in ("silu_mul_quant", "add_rms_quant", "gdn_norm_quant"):
    for M in [int(x) for x in a.ms.split(",")]:
        c = cell(kind, M)
        if c is None:
            print(f"{kind} M={M}: no dump")
            continue
        run, outs, nbytes = c
        res = {0: [], 1: []}
        for reg in (0, 1, 1, 0):
            res[reg].append(timed(run, outs, reg))
        dig = []
        for reg in (0, 1):
            o2 = [torch.zeros_like(t) for t in outs]  # tiled pad rows stay 0
            ext.set_nq_reg(reg)
            run(0, o2)
            torch.cuda.synchronize()
            h = hashlib.sha256()
            for t in o2:
                h.update(t.cpu().numpy().tobytes())
            dig.append(h.hexdigest()[:16])
        t0, t1 = statistics.mean(res[0]), statistics.mean(res[1])
        print(f"{kind} M={M} | {t0:.1f} {t1:.1f} ({'/'.join(f'{x:.1f}' for x in res[0])} | "
              f"{'/'.join(f'{x:.1f}' for x in res[1])}) | {t0 / t1:.3f}x | {nbytes / t0 / 1e3:.0f} "
              f"{nbytes / t1 / 1e3:.0f} | DIGEST {dig[0]} {dig[1]} {'EQUAL' if dig[0] == dig[1] else 'WRONG'}",
              flush=True)
ext.set_nq_reg(0)
