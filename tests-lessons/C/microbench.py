"""Isolated microbench of the int4 lm_head GEMM (N=248320, K=5120, g128, tiled) at verify M = 8..64.

Control = the production path (stock tile table via triton_w4a16_skinny_fmt_gemm with the split-K
table untouched). Candidates = triton_w4a16_splitk_gemm configs (split 1, direct epilogue) that
compile spill-free (compile_cfgs.py). DRAM-cold timing (a 256 MB write between runs flushes the
64 MB infinity cache). Exactness: the control is bit-compared with itself across runs; every
candidate is compared with the control (argmax over the full row, top-20 set, max |diff| relative
to the logit RMS) and with an fp32 reference on 2048 rows.

Prints per M the control, the best candidate and a ready-to-paste RADIANCE_LMHEAD_INT4_LEAN_CFG.
Decision: take a candidate if it is >= 5 % faster than the control and argmax-equal on every row.
"""
import sys
import time

import torch

from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as h

sys.path.insert(0, "/opt/vllm/lib/python3.12/site-packages")
N, K, G = 248320, 5120, 128
MS = [8, 16, 24, 32, 40, 48, 56, 64]
BUCKETS = (8, 16, 32, 40, 64)
dev = "cuda"
torch.manual_seed(0)

# synthetic head: gaussian rows, symmetric int4 g128 RTN (the kernel's numerics do not depend on the clip search)
w = torch.randn(N, K, dtype=torch.bfloat16, device=dev) * 0.02
rows = []
scs = []
for i in range(0, N, 8192):
    blk = w[i:i + 8192].float().view(-1, K // G, G)
    s = (blk.abs().amax(-1, keepdim=True) / 7).clamp_(min=1e-8).to(torch.bfloat16).float()
    q = torch.round(blk / s).clamp_(-8, 7)
    rows.append(h.pack_int4_exllama_shuffle((q.view(-1, K) + 8).to(torch.uint8)))
    scs.append(s.view(-1, K // G).to(torch.bfloat16))
del w
w32 = torch.cat(rows).view(torch.int8)
sc = torch.cat(scs).contiguous()
wt = h.radiance_w4a16_tile(w32).view(torch.int32)  # [N/16, K/128, 16, 16]
del rows, scs, w32
flush = torch.empty(256 << 20, dtype=torch.uint8, device=dev)

# fp32 reference rows (first 2048 rows, untiled dequant)
ref_w = h._radiance_w4_unpack_bf16(wt.view(torch.int8)[:2048 // 16], sc[:2048], G).float()


def bench(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


from cands import CANDS  # noqa: E402


def bucket_of(m):
    return next(b for b in BUCKETS if m <= b)


def ctrl(x):
    # production path; the split-K table must not hold an entry for the head here
    for b in BUCKETS:
        assert (G, K, N, b) not in h._gfx12x_splitk_table(), "split-K entry present: control polluted"
    return h.triton_w4a16_skinny_fmt_gemm(a=x, b_q=wt, scales=sc, group_size=G)


summary = {}
print(f"{'M':>3} {'control us':>10} {'GB/s':>5} | best candidate (us, vs control) | exact")
for m in MS:
    b = bucket_of(m)
    x = (torch.randn(m, K, dtype=torch.bfloat16, device=dev) * 0.5)
    ref = ctrl(x)
    ref2 = ctrl(x)
    assert torch.equal(ref, ref2), "control not deterministic"
    t_ctrl = bench(lambda: ctrl(x))
    r32 = x.float() @ ref_w.t()
    err_ref = (ref[:, :2048].float() - r32).abs().max().item() / r32.std().item()
    best = None
    for cfg in CANDS[b]:
        try:
            run = lambda: h.triton_w4a16_splitk_gemm(x, wt, sc, G, cfg)
            out = run()
            t = bench(run)
        except Exception as e:  # noqa: BLE001
            print(f"  M={m} cfg={cfg} FAILED {type(e).__name__}: {str(e)[:60]}")
            continue
        am = bool((out.argmax(-1) == ref.argmax(-1)).all())
        top_ok = bool((out.topk(20, -1).indices.sort(-1).values == ref.topk(20, -1).indices.sort(-1).values).all())
        rel = (out.float() - ref.float()).abs().max().item() / ref.float().std().item()
        ident = torch.equal(out, ref)
        print(f"  M={m:2d} {str(cfg):38s} {t:7.0f} us  argmax={am} top20={top_ok} maxdiff/std={rel:.4f} bit-equal={ident}")
        if am and top_ok and (best is None or t < best[0]):
            best = (t, cfg, ident, rel)
    gbs = (N * K // 2 + N * K // G * 2) / t_ctrl / 1e3
    if best:
        t, cfg, ident, rel = best
        print(f"{m:>3} {t_ctrl:10.0f} {gbs:5.0f} | {t:6.0f} us ({100*(t/t_ctrl-1):+.1f} %) {cfg} | "
              f"{'bit-equal' if ident else f'diff/std {rel:.4f}'}  (control vs fp32 ref: {err_ref:.4f} std)")
        if m == b or b not in summary:
            summary[b] = (t_ctrl, t, cfg)
    else:
        print(f"{m:>3} {t_ctrl:10.0f} {gbs:5.0f} | no exact candidate")

print("\n== per bucket (M = bucket size) ==")
spec = []
total_ctrl = total_best = 0
for b in BUCKETS:
    if b not in summary:
        continue
    tc, tb, cfg = summary[b]
    gain = 100 * (1 - tb / tc)
    print(f"bucket {b:2d}: control {tc:6.0f} us  best {tb:6.0f} us  ({gain:+.1f} %)  {cfg}")
    if gain >= 5 and tb < tc:
        spec.append(f"{b}=" + ",".join('none' if v is None else str(v) for v in cfg))
if spec:
    print("RECOMMENDED_LEAN_CFG=" + ";".join(spec))
else:
    print("RECOMMENDED_LEAN_CFG=   (no candidate >= 5 % faster: keep RADIANCE_LMHEAD_INT4_LEAN=0)")
