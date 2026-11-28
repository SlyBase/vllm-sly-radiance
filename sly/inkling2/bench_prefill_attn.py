"""Inkling 2 prefill attention, global layer, one chunk of the `_rel_attention_kernel` prefill launch (MODE_OUT_LSE), fp8 KV.

usage (inside the image, GPU free; apply sly/patch_inkling2_decode_attn.py + sly/patch_inkling2_prefill_attn.py first):
  python3 sly/inkling2/bench_prefill_attn.py            # CHUNK=2048, ctx 8k / 64k / 150k
  CTXS=8192,150000 CHUNK=2048 python3 sly/inkling2/bench_prefill_attn.py

Prints, per variant (masked loop = stock; unmasked bulk loop with tile BLOCK_M x BLOCK_N and num_warps, the knobs
RADIANCE_INKLING2_PREFILL_BLOCK_M / _BLOCK_N / _WARPS):
  (a) for a small shape, max |diff| of out and lse against an fp32 torch reference (stock arithmetic = the loop to beat),
  (b) per ctx: max |diff| against the stock launch and microseconds per call.
A variant that does not compile (LDS) prints its error and is skipped. Last line: JSON {"best": {...}} of the fastest
bulk variant at the largest ctx whose check against the stock launch passed and that is not slower than stock (+5 %)
at the smallest ctx: {"block_m": M, "block_n": N, "warps": W, "lazy": 0|1} (0 = stock value).
"""
import json
import os

import torch
import vllm.models.inkling2.nvidia.ops.attention_sm90 as S0

H, HKV, D, PAGE, REL = 24, 8, 128, 128, 1024
CHUNK = int(os.environ.get("CHUNK", "2048"))
CTXS = [int(c) for c in os.environ.get("CTXS", "8192,65536,150000").split(",")]
dev = "cuda"
torch.manual_seed(0)
G = H // HKV


def inputs(ctx, T):
    nblk = (ctx + PAGE - 1) // PAGE
    kv = (torch.randn(nblk, HKV, PAGE, 2 * D, device=dev) * 2).to(torch.float8_e4m3fn)
    kc, vc = kv.transpose(1, 2).split(D, dim=-1)
    return dict(
        kv=kv, kc=kc, vc=vc,
        bt=torch.randperm(nblk, device=dev, dtype=torch.int32)[None],
        sl=torch.tensor([ctx], device=dev, dtype=torch.int32),
        cu=torch.tensor([0, T], device=dev, dtype=torch.int32),
        q=torch.randn(T, H, D, device=dev, dtype=torch.bfloat16),
        rel=torch.randn(T + 128, H, REL + 256, device=dev, dtype=torch.bfloat16) * 0.5,
    )


def set_variant(bulk, block_m, block_n, warps, lazy=False):
    S0._PREFILL_LAZY = lazy
    S0._BULK = bulk
    S0._PREFILL_BULK = bulk
    S0._PREFILL_BLOCK_M = block_m
    S0._PREFILL_BLOCK_N = block_n
    S0._PREFILL_WARPS = warps


def call(x, T):
    out = torch.zeros(T, H, D, device=dev, dtype=torch.float32)
    lse = torch.zeros(H, T, device=dev, dtype=torch.float32)
    S0.rel_attention(x["q"], x["kc"], x["vc"], out, lse, x["rel"], block_table=x["bt"], cache_seqlens=x["sl"],
                     cu_seqlens_q=x["cu"], max_query_len=T, softmax_scale=D ** -0.5, window=0, num_splits=1,
                     mode=S0.MODE_OUT_LSE)
    return out, lse


def bench(fn, n):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n * 1000


def torch_ref(x, ctx, T, toks):
    """fp32 reference for the tokens `toks` of the chunk (all heads): causal, relative bias on the last REL keys."""
    past = ctx - T
    kk = x["kc"][x["bt"][0].long()].reshape(-1, HKV, D)[:ctx].float()
    vv = x["vc"][x["bt"][0].long()].reshape(-1, HKV, D)[:ctx].float()
    E_PAD = REL + 256
    out = torch.zeros(len(toks), H, D, device=dev)
    lse = torch.zeros(H, len(toks), device=dev)
    for i, t in enumerate(toks):
        pos = past + t
        tile_end = min((t // 128 + 1) * 128, T)
        rel_base = (past + tile_end + 127) // 128 * 128 - E_PAD
        keys = torch.arange(pos + 1, device=dev)
        m = (pos - keys) < REL
        for h in range(H):
            kvh = h // G
            sc = (x["q"][t, h].float() @ kk[: pos + 1, kvh].T) * D ** -0.5
            b = torch.zeros(pos + 1, device=dev)
            b[m] = x["rel"][t, h, (keys - rel_base)[m]].float()
            sc = sc + b
            out[i, h] = torch.softmax(sc, 0) @ vv[: pos + 1, kvh]
            lse[h, i] = torch.logsumexp(sc, 0)
    return out, lse


# (name, bulk, block_m (0 = 128), block_n, num_warps (0 = 8), lazy rescale); env VARIANTS = JSON list of the same
VARIANTS = [
    ("stock", False, 0, 32, 0),
    ("m128n32w8", True, 0, 32, 0), ("m128n16w8", True, 0, 16, 0), ("m128n32w4", True, 0, 32, 4),
    ("m128n32w16", True, 0, 32, 16), ("m128n64w16", True, 0, 64, 16),
    ("m64n32w4", True, 64, 32, 4), ("m64n32w8", True, 64, 32, 8), ("m64n64w4", True, 64, 64, 4),
    ("m64n64w8", True, 64, 64, 8), ("m64n16w4", True, 64, 16, 4),
    ("m32n64w4", True, 32, 64, 4), ("m32n128w4", True, 32, 128, 4),
]
if os.environ.get("VARIANTS"):
    VARIANTS = [tuple(v) for v in json.loads(os.environ["VARIANTS"])]
VARIANTS = [tuple(v) + (False,) * (6 - len(v)) for v in VARIANTS]

# (a) numerics against fp32 torch on a small shape that still has a bulk region (ctx 4096 > REL + chunk)
cx, ct = 4096, 256
toks = [0, 41, 42, 127, 128, 255]
x = inputs(cx, ct)
ref_o, ref_l = torch_ref(x, cx, ct, toks)
for name, bulk, bm, bn, w, lz in VARIANTS:
    set_variant(bulk, bm, bn, w, lz)
    try:
        o, l = call(x, ct)
    except Exception as e:  # noqa: BLE001 - report, keep going
        print(json.dumps({"check": name, "error": str(e)[:200]}), flush=True)
        continue
    print(json.dumps({"check": name, "out_vs_fp32": round((o[toks] - ref_o).abs().max().item(), 5),
                      "lse_vs_fp32": round((l[:, toks] - ref_l).abs().max().item(), 5)}), flush=True)

# (b) speed + agreement with the stock launch at the real shapes
times = {}
for ctx in CTXS:
    x = inputs(ctx, CHUNK)
    set_variant(False, 0, 32, 0)
    base_o, base_l = call(x, CHUNK)
    for name, bulk, bm, bn, w, lz in VARIANTS:
        set_variant(bulk, bm, bn, w, lz)
        try:
            o, l = call(x, CHUNK)
            us = bench(lambda: call(x, CHUNK), 3 if ctx > 100000 else 6)
        except Exception as e:  # noqa: BLE001
            print(json.dumps({"ctx": ctx, "variant": name, "error": str(e)[:200]}), flush=True)
            continue
        d = round((o - base_o).abs().max().item(), 5)
        dl = round((l - base_l).abs().max().item(), 5)
        print(json.dumps({"ctx": ctx, "chunk": CHUNK, "variant": name, "us": round(us), "out_vs_stock": d, "lse_vs_stock": dl}), flush=True)
        if d < 1e-2:
            times[(ctx, name)] = us
lo, hi = min(CTXS), max(CTXS)
ok = [v for v in VARIANTS[1:] if (hi, v[0]) in times and times.get((lo, v[0]), 1e18) <= 1.05 * times.get((lo, "stock"), 0)]
b = min(ok, key=lambda v: times[(hi, v[0])]) if ok else None
print(json.dumps({"best": {"block_m": b[2], "block_n": b[3], "warps": b[4], "lazy": int(b[5])} if b else {"block_m": 0, "block_n": 32, "warps": 0, "lazy": 0}}))
