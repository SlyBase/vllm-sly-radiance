"""Inkling 2 decode (verify) attention, 1 sequence x 8 query tokens, fp8 KV, on the R9700: global and local layers.

usage (inside the image, GPU free): python3 sly/inkling2/bench_decode_attn.py [--sweep]
For global layers (full attention, 8k / 64k / 150k) and local sliding-window layers (window 1024, 8k / 64k) it prints
(a) max |diff| of the patched launch (RADIANCE_INKLING2_DECODE_SPLITS plan, tile, bulk loop and vector block load)
against the stock launch (DECODE_SPLITS=0 path) -- must stay <= 2e-3 -- and, for ctx <= 8192, both against an fp32
torch reference; (b) microseconds per call for the stock and the patched launch. --sweep adds a (BLOCK_N, warps)
sweep of the patched launch. Before the vector block load, 150k global: stock 6690 us, patched 950 us per call.
"""
import argparse
import json

import torch
import vllm.models.inkling2.nvidia.ops.attention as A
import vllm.models.inkling2.nvidia.ops.attention_sm90 as S0
from vllm.utils.platform_utils import num_compute_units

H, HKV, D, T, PAGE, REL = 24, 8, 128, 8, 128, 1024
MAX_ABS = 2e-3
dev = "cuda"
torch.manual_seed(0)


def decode_path(on):
    """DECODE_SPLITS=1 (patched plan, tile, bulk loop, vector load) or 0 (stock), as the env knob sets them at import."""
    A._RAD_DEC_SPLITS = S0._RAD_DEC_SPLITS = S0._BULK = S0._RAD_VEC_KV = on
    A._ROCM_CTAS_PER_SM = 16 if on else 0


def plan(is_local):
    return A.plan_decode_attention(is_local, num_reqs=1, max_query_len=T, num_heads=H, num_kv_heads=HKV, rel_extent=REL)


def inputs(ctx):
    nblk = (ctx + PAGE - 1) // PAGE
    kv = (torch.randn(nblk, HKV, PAGE, 2 * D, device=dev) * 2).to(torch.float8_e4m3fn)
    kc, vc = kv.transpose(1, 2).split(D, dim=-1)
    return dict(kv=kv, kc=kc, vc=vc, bt=torch.randperm(nblk, device=dev, dtype=torch.int32)[None],
                sl=torch.tensor([ctx], device=dev, dtype=torch.int32), cu=torch.tensor([0, T], device=dev, dtype=torch.int32),
                q=torch.randn(T, H, D, device=dev, dtype=torch.bfloat16),
                rel=torch.randn(T + 128, H, REL + 256, device=dev, dtype=torch.bfloat16) * 0.5,
                gate=torch.randn(T, H * D, device=dev, dtype=torch.bfloat16), sinks=torch.randn(T, H, device=dev, dtype=torch.bfloat16))


def call(x, s, out, is_local):
    A.rel_decode_attention(x["q"], x["kc"], x["vc"], x["rel"], x["gate"], x["sinks"], out, block_table=x["bt"], cache_seqlens=x["sl"],
                           cu_seqlens_q=x["cu"], max_query_len=T, num_actual_tokens=T, softmax_scale=D ** -0.5, is_local=is_local, num_splits=s)


def launch(x, on, is_local):
    """(output, splits) of one call on the stock (on=False) or patched (on=True) path; the plan picks the splits."""
    decode_path(on)
    s = plan(is_local)
    o = torch.zeros(T, H * D, device=dev, dtype=torch.bfloat16)
    call(x, s, o, is_local)
    return o.float().clone(), s


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


def torch_ref(x, ctx, is_local):
    """fp32 reference of one decode call (all 8 tokens, causal, relative bias on the last REL keys; window REL if local)."""
    past = ctx - T
    K = x["kc"][x["bt"][0].long()].reshape(-1, HKV, D)[:ctx].float()
    V = x["vc"][x["bt"][0].long()].reshape(-1, HKV, D)[:ctx].float()
    out = torch.zeros(T, H, D, device=dev)
    lse = torch.zeros(T, H, device=dev)
    E_PAD = REL + 256
    for t in range(T):
        pos = past + t
        lo = max(0, pos - REL + 1) if is_local else 0
        keys = torch.arange(lo, pos + 1, device=dev)
        tile_end = min((t // 128 + 1) * 128, T)
        rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
        near = (pos - keys) < REL
        for h in range(H):
            sc = (x["q"][t, h].float() @ K[keys, h // (H // HKV)].T) * D ** -0.5
            sc[near] += x["rel"][t, h, keys[near] - rel_base].float()
            out[t, h] = torch.softmax(sc, 0) @ V[keys, h // (H // HKV)]
            lse[t, h] = torch.logsumexp(sc, 0)
    g = torch.sigmoid(x["gate"].float().view(T, H, D))
    sk = torch.sigmoid(lse - x["sinks"].float())
    return (out * g * sk[:, :, None]).reshape(T, H * D)


def sweep(x, is_local):
    """us per call of the patched launch for each (BLOCK_N, warps); local layers re-plan their splits for the BLOCK_N."""
    saved = (S0._RAD_DEC_BN, S0._RAD_DEC_WARPS, S0._RAD_LOC_BN, S0._RAD_LOC_WARPS, A._RAD_LOC_BN)
    out = {}
    for bn in (32, 64, 128):
        for w in (2, 4, 8):
            if is_local:
                S0._RAD_LOC_BN, S0._RAD_LOC_WARPS, A._RAD_LOC_BN = bn, w, bn
            else:
                S0._RAD_DEC_BN, S0._RAD_DEC_WARPS = bn, w
            decode_path(True)
            o, s = torch.zeros(T, H * D, device=dev, dtype=torch.bfloat16), plan(is_local)
            out[f"bn{bn}_w{w}"] = round(bench(lambda: call(x, s, o, is_local)), 1)
    S0._RAD_DEC_BN, S0._RAD_DEC_WARPS, S0._RAD_LOC_BN, S0._RAD_LOC_WARPS, A._RAD_LOC_BN = saved
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="also sweep (BLOCK_N, warps) of the patched launch")
    args = ap.parse_args()
    print("units", num_compute_units(), flush=True)
    failed = False
    for is_local, ctxs in ((False, (2000, 8192, 65536, 150000)), (True, (2000, 8192, 65536))):
        for ctx in ctxs:
            x = inputs(ctx)
            base, s_stock = launch(x, False, is_local)
            new, s_new = launch(x, True, is_local)
            r = {"local": is_local, "ctx": ctx, "splits": [s_stock, s_new], "patched_vs_stock": round((new - base).abs().max().item(), 5)}
            if ctx <= 8192:
                ref = torch_ref(x, ctx, is_local)
                r.update(stock_vs_fp32ref=round((base - ref).abs().max().item(), 5), patched_vs_fp32ref=round((new - ref).abs().max().item(), 5))
            for name, on, s in (("stock_us", False, s_stock), ("patched_us", True, s_new)):
                decode_path(on)
                o = torch.zeros(T, H * D, device=dev, dtype=torch.bfloat16)
                r[name] = round(bench(lambda: call(x, s, o, is_local)), 1)
            r["ok"] = r["patched_vs_stock"] <= MAX_ABS
            failed |= not r["ok"]
            print(json.dumps(r), flush=True)
            if args.sweep and ctx >= 65536:
                print(json.dumps({"local": is_local, "ctx": ctx, "sweep_us": sweep(x, is_local)}), flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
