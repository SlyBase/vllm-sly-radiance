#!/usr/bin/env python3
"""Offline numerics + timing check for the lm_head variants (bf16 reference, fp8, int4).

Runs in a throw-away container from the runtime image with the GPU exclusive and the HF cache
mounted; exercises the production classes RadianceLMHeadFp8 / RadianceLMHeadInt4 on the real
lm_head of amd/Qwen3.8-27B-Quark-AWQ-MXFP4 (bf16 [248320, 5120]) with two proxies for final
hidden states (random x (1+norm_w); RMS-normalised embedding rows x (1+norm_w)):

  * logit error vs the fp32 reference (max-abs, RMS relative to the logit RMS),
  * argmax flip rate and top-16 overlap vs the reference (what the target verify and the
    DFlash drafter bootstrap read from this layer), int4 also vs fp8 (today's production),
  * int4 with the MSE clip search vs plain RTN, group 128 vs 64,
  * per-call time at the DFlash M values (verify M = 8 x seqs, draft M = 7 x seqs).

  PYTHONPATH=/opt/patches/sly/mxfp4 RADIANCE_LMHEAD_FP8=1 RADIANCE_LMHEAD_INT4=1 \
      python3 check_lmhead_int4.py [--snapshot DIR] [--rows 2048]
"""
import argparse, glob, os, time

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", default=None)
ap.add_argument("--rows", type=int, default=2048, help="proxy hidden states per set")
ap.add_argument("--no-timing", action="store_true")
args = ap.parse_args()

import radiance_lmhead_fp8 as F8  # noqa: E402
import radiance_lmhead_int4 as I4  # noqa: E402

dev = torch.device("cuda")
snap = args.snapshot or sorted(glob.glob(
    "/root/.cache/huggingface/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/*"))[-1]
t0 = time.time()
with safe_open(f"{snap}/model.safetensors", framework="pt", device="cpu") as f:
    W = f.get_tensor("lm_head.weight").to(dev)            # bf16 [N, K]
    norm_w = f.get_tensor("model.language_model.norm.weight").to(dev).float()
    emb = f.get_slice("model.language_model.embed_tokens.weight")
    torch.manual_seed(0)
    ids = torch.randint(0, 150000, (args.rows,)).sort().values
    E = torch.stack([emb[i:i + 1][0] for i in ids.tolist()]).to(dev).float()
N, K = W.shape
print(f"lm_head {W.dtype} [{N}, {K}] loaded in {time.time() - t0:.1f}s; snapshot {snap}")

# --- proxy hidden states (bf16, like the model's final-norm output) ---
torch.manual_seed(1)
g = 1.0 + norm_w
H_rand = (torch.randn(args.rows, K, device=dev) * g).to(torch.bfloat16)
E = E * torch.rsqrt(E.square().mean(-1, keepdim=True) + 1e-6)
H_emb = (E * g).to(torch.bfloat16)
del E


def ref_logits(h):
    out = torch.empty((h.shape[0], N), dtype=torch.float32, device=dev)
    hf = h.float()
    for i in range(0, N, 32768):
        out[:, i:i + 32768] = hf @ W[i:i + 32768].float().t()
    return out


class Layer(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone(), requires_grad=False)


def build(method, **env):
    layer = Layer(W)
    for k, v in env.items():
        setattr(I4, k, v)
    t = time.time()
    method.process_weights_after_loading(layer)
    torch.cuda.synchronize()
    return layer, time.time() - t


variants = {}
layer_fp8, dt = build(F8.RadianceLMHeadFp8())
variants["fp8"] = (F8.RadianceLMHeadFp8(), layer_fp8)
print(f"fp8 quantised in {dt:.2f}s")
for name, gs, clip in (("int4_g128_mse", 128, "mse"), ("int4_g128_rtn", 128, "rtn"), ("int4_g64_mse", 64, "mse")):
    layer, dt = build(I4.RadianceLMHeadInt4(), GROUP_SIZE=gs, CLIP=clip)
    variants[name] = (I4.RadianceLMHeadInt4(), layer, gs)
    print(f"{name} quantised in {dt:.2f}s")
I4.GROUP_SIZE, I4.CLIP = 128, "mse"


def run(name, h):
    v = variants[name]
    if len(v) == 3:
        I4.GROUP_SIZE = v[2]
    out = v[0].apply(v[1], h)
    I4.GROUP_SIZE = 128
    return out


def stats(ref, out, ref_fp8=None):
    d = (out.float() - ref)
    rms = ref.square().mean().sqrt().item()
    am_r, am_o = ref.argmax(-1), out.argmax(-1)
    flip = (am_r != am_o).float().mean().item()
    k = 16
    tr, to = ref.topk(k, -1).indices, out.float().topk(k, -1).indices
    ov = (tr.unsqueeze(-1) == to.unsqueeze(-2)).any(-1).float().sum(-1).mean().item()
    # gap between top-1 and top-2 of the reference for the flipped rows (how close they were)
    top2 = ref.topk(2, -1).values
    gap = (top2[:, 0] - top2[:, 1])
    gap_flip = gap[am_r != am_o].mean().item() if flip > 0 else float("nan")
    s = (f"max|err| {d.abs().max().item():7.3f}  rms err {d.square().mean().sqrt().item():6.3f}"
         f" ({100 * d.square().mean().sqrt().item() / rms:4.2f} % of logit rms {rms:5.2f})"
         f"  argmax flips {100 * flip:5.2f} %  top16 overlap {ov:5.2f}/16  median top1-top2 gap {gap.median().item():.2f}"
         f" (flipped rows: {gap_flip:.2f})")
    if ref_fp8 is not None:
        s += f"  | vs fp8: argmax diff {100 * (out.argmax(-1) != ref_fp8.argmax(-1)).float().mean().item():5.2f} %"
    return s


for hname, H in (("rand", H_rand), ("emb", H_emb)):
    ref = ref_logits(H)
    print(f"\n=== proxy '{hname}' ({H.shape[0]} rows) ===")
    out8 = run("fp8", H)
    print(f"  {'fp8':14s} {stats(ref, out8)}")
    for name in ("int4_g128_mse", "int4_g128_rtn", "int4_g64_mse"):
        print(f"  {name:14s} {stats(ref, run(name, H), out8)}")
    del ref, out8

# --- consistency of the kernel paths (HIP skinny at M <= 5 vs Triton) ---
print("\n=== int4_g128_mse: M sweep vs its own M=64 result (path consistency) ===")
big = run("int4_g128_mse", H_rand[:64]).float()
for m in (1, 2, 3, 5, 6, 7, 8, 16, 63):
    o = run("int4_g128_mse", H_rand[:m]).float()
    print(f"  M={m:2d}: max|diff| vs M=64 slice {(o - big[:m]).abs().max().item():.4f}")

if args.no_timing:
    raise SystemExit(0)

print("\n=== per-call time (us, median of 30; weight 1.27 GB fp8 / 0.66 GB int4, DRAM-cold by size) ===")


def timeit(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) * 1000 for s, e in ev)
    return t[len(t) // 2]


print(f"  {'M':>3s} {'fp8':>9s} {'int4_g128':>10s} {'int4_g64':>9s}   (int4 = stock/table tiles of the installed image)")
for m in (1, 5, 7, 8, 14, 16, 28, 32, 40, 56, 64, 128):
    h = H_rand[:m].contiguous()
    t8 = timeit(lambda: run("fp8", h))
    t4 = timeit(lambda: run("int4_g128_mse", h))
    t64 = timeit(lambda: run("int4_g64_mse", h))
    print(f"  {m:3d} {t8:9.1f} {t4:10.1f} {t64:9.1f}   fp8 {1.271e9 / t8 / 1e3:4.0f} GB/s, int4 {0.656e9 / t4 / 1e3:4.0f} GB/s")
