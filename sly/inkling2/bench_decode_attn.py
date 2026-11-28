"""Inkling 2 decode (verify) attention, global layer, 1 sequence x 8 query tokens, fp8 KV, on the R9700.

usage (inside the image, GPU free): python3 sly/inkling2/bench_decode_attn.py
Prints (a) max |diff| of the patched launch (bulk loop + BLOCK_N 64, many splits) against the stock launch
(masked loop, BLOCK_N 128, 4 splits) and, for ctx <= 8192, both against an fp32 torch reference;
(b) microseconds per call at 8k / 64k / 150k for the stock plan and the patched plan.
Measured 2026-11-27 (210 W): 150k stock 6690 us, patched 950 us per call, 8 calls per step.
"""
import json

import torch
import vllm.models.inkling2.nvidia.ops.attention as A
import vllm.models.inkling2.nvidia.ops.attention_sm90 as S0
from vllm.utils.platform_utils import num_compute_units

H, HKV, D, T, PAGE, REL = 24, 8, 128, 8, 128, 1024
dev = "cuda"
torch.manual_seed(0)
print("plan(1 seq, 8 tok) =", A.plan_decode_attention(False, num_reqs=1, max_query_len=T, num_heads=H, num_kv_heads=HKV, rel_extent=REL),
      "units", num_compute_units(), flush=True)


def inputs(ctx):
    nblk = (ctx + PAGE - 1) // PAGE
    kv = (torch.randn(nblk, HKV, PAGE, 2 * D, device=dev) * 2).to(torch.float8_e4m3fn)
    kc, vc = kv.transpose(1, 2).split(D, dim=-1)
    return dict(kv=kv, kc=kc, vc=vc, bt=torch.randperm(nblk, device=dev, dtype=torch.int32)[None],
                sl=torch.tensor([ctx], device=dev, dtype=torch.int32), cu=torch.tensor([0, T], device=dev, dtype=torch.int32),
                q=torch.randn(T, H, D, device=dev, dtype=torch.bfloat16),
                rel=torch.randn(T + 128, H, REL + 256, device=dev, dtype=torch.bfloat16) * 0.5,
                gate=torch.randn(T, H * D, device=dev, dtype=torch.bfloat16), sinks=torch.randn(T, H, device=dev, dtype=torch.bfloat16))


def call(x, s, out):
    A.rel_decode_attention(x["q"], x["kc"], x["vc"], x["rel"], x["gate"], x["sinks"], out, block_table=x["bt"], cache_seqlens=x["sl"],
                           cu_seqlens_q=x["cu"], max_query_len=T, num_actual_tokens=T, softmax_scale=D ** -0.5, is_local=False, num_splits=s)


def bench(fn, n=40):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1000


def torch_ref(x, ctx):
    """fp32 reference of one global decode call (all 8 tokens, causal, relative bias on the last REL keys)."""
    past = ctx - T
    nblk = x["bt"].shape[1]
    kk = x["kc"][x["bt"][0].long()]  # (nblk, PAGE, HKV, D) logical order
    vv = x["vc"][x["bt"][0].long()]
    K = kk.reshape(-1, HKV, D)[:ctx].float()
    V = vv.reshape(-1, HKV, D)[:ctx].float()
    out = torch.zeros(T, H, D, device=dev)
    lse = torch.zeros(T, H, device=dev)
    E_PAD = REL + 256
    for t in range(T):
        pos = past + t
        tile_end = min((t // 128 + 1) * 128, T)
        rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
        for h in range(H):
            kvh = h // (H // HKV)
            sc = (x["q"][t, h].float() @ K[: pos + 1, kvh].T) * D ** -0.5
            dist = pos - torch.arange(pos + 1, device=dev)
            m = dist < REL
            idx = torch.arange(pos + 1, device=dev) - rel_base
            b = torch.zeros(pos + 1, device=dev)
            b[m] = x["rel"][t, h, idx[m]].float()
            sc = sc + b
            p = torch.softmax(sc, 0)
            out[t, h] = p @ V[: pos + 1, kvh]
            lse[t, h] = torch.logsumexp(sc, 0)
    g = torch.sigmoid(x["gate"].float().view(T, H, D))
    sk = torch.sigmoid(lse - x["sinks"].float())
    return (out * g * sk[:, :, None]).reshape(T, H * D)



for ctx in (2000, 8192, 65536, 150000):
    x = inputs(ctx)
    outs = {}
    for bulk in (False, True):
        S0._BULK = bulk
        for s in (4, 64):
            o = torch.zeros(T, H * D, device=dev, dtype=torch.bfloat16)
            call(x, s, o)
            outs[(bulk, s)] = o.float().clone()
    base = outs[(False, 4)]
    r = {"ctx": ctx, "patched_vs_stock": round((outs[(True, 64)] - base).abs().max().item(), 5)}
    if ctx <= 8192:
        ref = torch_ref(x, ctx)
        r.update(stock_vs_fp32ref=round((base - ref).abs().max().item(), 5), patched_vs_fp32ref=round((outs[(True, 64)] - ref).abs().max().item(), 5))
    for name, bulk, s in (("stock_us", False, 4), ("patched_us", True, 64)):
        S0._BULK = bulk
        o = torch.zeros(T, H * D, device=dev, dtype=torch.bfloat16)
        r[name] = round(bench(lambda: call(x, s, o)), 1)
    print(json.dumps(r), flush=True)
