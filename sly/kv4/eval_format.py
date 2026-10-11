#!/usr/bin/env python3
"""Which part of the KV4 format costs accuracy, on the model's own K/V.

Input: dumps written by radiance_kv4_store with RADIANCE_KV4_DUMP (kv_NNN.pt: bf16 K and V of one
store call, (tokens, kv_heads, 256), after rope). For every variant the script quantizes K and V,
then measures (a) the relative MSE of K and V themselves and (b) the relative error of a greedy
attention output over the dump's tokens (queries = the last 64 tokens' K rows reused as queries,
which is what a model with q/k norm roughly looks like), against the unquantized bf16 K/V.
fp8 with a static scale (today's path) is the yardstick.

  python3 sly/kv4/eval_format.py /path/to/dumps
"""
import glob
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import radiance_kv4_format as F  # noqa: E402


def fp8_static(x, amax):
    s = amax / 448.0
    return (x / s).to(F.FP8).to(torch.float32) * s


def quant(x, name, ngroup):
    return F.reconstruct(x, name, fp8_rounded=False, ngroup=ngroup)


def asym_group(x, g):
    return F.asym_reconstruct(x, g)


def asym_axis(x, g, axis_tokens, rotate=True):
    """Asymmetric int4 with the scale/zero per group of g along TOKENS (per channel, KIVI-style
    for K) instead of along dims. x (tokens, HEAD)."""
    xr = F.rotate(x) if rotate else x
    t = xr.shape[0]
    pad = (-t) % g
    xp = torch.cat([xr, xr[-1:].expand(pad, -1)], 0) if pad else xr
    xg = xp.reshape(-1, g, F.HEAD)                       # (blocks, g tokens, dims): scale per (block, dim)
    lo, hi = xg.amin(1, keepdim=True), xg.amax(1, keepdim=True)
    sc = ((hi - lo) / 15).clamp_min(1e-8)
    q = ((xg - lo) / sc).round().clamp(0, 15)
    y = (q * sc + lo).reshape(-1, F.HEAD)[:t]
    return F.unrotate(y) if rotate else y


def asym_dims(x, g, rotate=True):
    xr = F.rotate(x) if rotate else x
    xg = xr.reshape(xr.shape[0], -1, g)
    lo, hi = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
    sc = ((hi - lo) / 15).clamp_min(1e-8)
    q = ((xg - lo) / sc).round().clamp(0, 15)
    y = (q * sc + lo).reshape(xr.shape)
    return F.unrotate(y) if rotate else y


def smooth_asym(k, g):
    """Calibrated per-channel smoothing before the rotation: K / s_d, where s_d is the channel RMS
    (folded into q at run time for free), then asymmetric int4 per g dims."""
    s_d = k.pow(2).mean(0).sqrt().clamp_min(1e-6)
    return asym_dims(k / s_d, g) * s_d


def mixed(xq, x, keep_first, keep_last):
    """Rows kept at bf16 precision: the first keep_first (sink) and the last keep_last (residual)."""
    y = xq.clone()
    if keep_first:
        y[:keep_first] = x[:keep_first]
    if keep_last:
        y[-keep_last:] = x[-keep_last:]
    return y


def attn_out(q, k, v):
    s = (q @ k.T) / 16.0
    p = torch.softmax(s, dim=-1)
    return p @ v


def quant_query(q, qmax):
    """The backend's query quantization: rotate, amax / qmax per row, round, un-rotate (int8: 127, int4: 7)."""
    qr = F.rotate(q)
    sc = qr.abs().amax(-1, keepdim=True).clamp_min(1e-12) / qmax
    return F.unrotate(torch.round(qr / sc).clamp(-qmax, qmax) * sc)


def main(d):
    files = sorted(glob.glob(f"{d}/kv_*.pt"))
    assert files, d
    variants = {
        "fp8 static (today)": None,
        "K uniform/2 + V geo/1 (kernel)": (("uniform", 2), ("geo", 1)),
        "K uniform/2 + V uniform/1": (("uniform", 2), ("uniform", 1)),
        "K uniform/2 + V uniform/2": (("uniform", 2), ("uniform", 2)),
        "K uniform/8 + V uniform/8 (group 32)": (("uniform", 8), ("uniform", 8)),
        "K uniform/8 + V uniform/1 (kv4h: K group 32, V per token)": (("uniform", 8), ("uniform", 1)),
        "K uniform/4 + V uniform/1 (K group 64)": (("uniform", 4), ("uniform", 1)),
        "K uniform/2 + V uniform/1 (K group 128)": (("uniform", 2), ("uniform", 1)),
        "K uniform/1 + V uniform/1 (K per token)": (("uniform", 1), ("uniform", 1)),
        "kv4h + int8 query (production)": "q8",
        "kv4h + int4 query": "q4",
        "K per token + int4 query (fastest sweep format)": "q4k1",
        "K geo/2 + V geo/1": (("geo", 2), ("geo", 1)),
        "K lloyd/8 + V lloyd/8": (("lloyd", 8), ("lloyd", 8)),
        "asym int4 group 32 (K and V)": "asym32",
        "asym int4 group 16 (K and V)": "asym16",
        "KIVI: K per-channel/32 tok, V per-token/32 dims (rotated)": "kivi32",
        "KIVI: K per-channel/32 tok, V per-token/32 dims (no rotation)": "kivi32_norot",
        "K per-channel/64 tok + V/32 (rotated)": "kivi64",
        "smoothed K asym32 + V asym32": "smooth32",
        "asym32 without rotation": "asym32_norot",
        "K asym16 + V fp8 (which side hurts?)": "k4_v8",
        "K fp8 + V asym16": "k8_v4",
        "K asym16 + V asym int8/16": "k4_v8i",
    }
    keep = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    if keep:
        variants = {n: v for n, v in variants.items() if any(k in n for k in keep)}
    acc = {n: [0.0, 0.0, 0.0, 0] for n in variants}
    for f in files:
        dd = torch.load(f)
        k_all, v_all = dd["k"].float(), dd["v"].float()
        for h in range(k_all.shape[1]):
            k, v = k_all[:, h], v_all[:, h]
            q = k[-64:]
            ref = attn_out(q, k, v)
            kmax, vmax = k.abs().max(), v.abs().max()
            for name, spec in variants.items():
                qq = q
                if spec in ("q8", "q4", "q4k1"):
                    kq = quant(k, "uniform", 1 if spec == "q4k1" else 2)
                    vq = quant(v, "uniform", 1)
                    qq = quant_query(q, 127.0 if spec == "q8" else 7.0)
                elif spec is None:
                    kq, vq = fp8_static(k, kmax), fp8_static(v, vmax)
                elif spec == "asym32":
                    kq, vq = asym_group(k, 32), asym_group(v, 32)
                elif spec == "asym16":
                    kq, vq = asym_group(k, 16), asym_group(v, 16)
                elif spec == "kivi32":
                    kq, vq = asym_axis(k, 32, True), asym_dims(v, 32)
                elif spec == "kivi32_norot":
                    kq, vq = asym_axis(k, 32, True, rotate=False), asym_dims(v, 32, rotate=False)
                elif spec == "kivi64":
                    kq, vq = asym_axis(k, 64, True), asym_dims(v, 32)
                elif spec == "smooth32":
                    kq, vq = smooth_asym(k, 32), asym_dims(v, 32)
                elif spec == "asym32_norot":
                    kq, vq = asym_dims(k, 32, rotate=False), asym_dims(v, 32, rotate=False)
                elif spec == "k4_v8":
                    kq, vq = asym_group(k, 16), fp8_static(v, vmax)
                elif spec == "k8_v4":
                    kq, vq = fp8_static(k, kmax), asym_group(v, 16)
                elif spec == "k4_v8i":
                    xr = F.rotate(v)
                    xg = xr.reshape(xr.shape[0], -1, 16)
                    lo, hi = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
                    sc = ((hi - lo) / 255).clamp_min(1e-8)
                    vq = F.unrotate((((xg - lo) / sc).round().clamp(0, 255) * sc + lo).reshape(xr.shape))
                    kq = asym_group(k, 16)
                elif spec in ("sink", "resid", "both", "lloyd_both"):
                    if spec == "lloyd_both":
                        kq, vq = quant(k, "lloyd", 8), quant(v, "lloyd", 8)
                    else:
                        kq, vq = quant(k, "uniform", 2), quant(v, "geo", 1)
                    kf, kl = (4, 0) if spec == "sink" else (0, 128) if spec == "resid" else (4, 128)
                    kq, vq = mixed(kq, k, kf, kl), mixed(vq, v, kf, kl)
                else:
                    (ck, gk), (cv, gv) = spec
                    kq, vq = quant(k, ck, gk), quant(v, cv, gv)
                a = acc[name]
                a[0] += float(((kq - k) ** 2).mean() / (k ** 2).mean())
                a[1] += float(((vq - v) ** 2).mean() / (v ** 2).mean())
                out = attn_out(qq, kq, vq)
                a[2] += float((out - ref).norm() / ref.norm())
                a[3] += 1
    print(f"{len(files)} dumps; relative MSE of K, of V, and relL2 of the attention output vs bf16:")
    for name, (mk, mv, me, n) in acc.items():
        print(f"  {name:40s} K {mk / n:.2e}   V {mv / n:.2e}   attn {me / n:.4f}")


if __name__ == "__main__":
    main(sys.argv[1])
