"""Offline accuracy hunt for the int4 libr4d route (R4D_KV4I): one dumped launch (R4D_KV4I_DUMP) against
the fp32 truth built from the same int4 bytes, plus a torch emulation of the kernel with each rounding
switched on alone. CPU only.

    python kv4i_offline.py <dump.pt> [--rows 256]

Truth: rotate(q) . decode(K) * scale / HEAD, fp32 softmax, P . decode(V), unrotate -- the int4 cache is
the input both kernels share, so this isolates what each kernel adds on top of it.
Kernel emulation (r4d_attn_prefill O_QK8|O_PV8|O_FOLDQ|O_QF8|O_KV4I): q_in = bf16(q8 * q_scale),
q_fold = e4m3(q_in * scale/HEAD * log2e), K8 = e4m3(code * s), p = 2^(s - rowmax), P8 = e4m3(p),
V8 = e4m3(code * s), out = P8 . V8 / sum(p). ponytail: the kernel's lazy max (p up to 2^8) is
replaced by the exact row max, which is the worst case for P underflow.
"""
import argparse
import math

import torch

import radiance_kv4_format as F
import radiance_kv4h_format as H

LOG2E = 1.0 / math.log(2.0)
BS = 1664


def e4m3(x):
    """RNE to e4m3 with saturation at 448 (v_cvt_pk_fp8_f32; torch alone would make NaN)."""
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).to(torch.float32)


def decode(kv, ctx):
    """kv uint8 (nblk, kv_heads, BS, 272) -> codes*scales k_rot, v_rot (kv_heads, ctx, 256) fp32."""
    ks, vs = [], []
    for h in range(kv.shape[1]):
        kh, vh = zip(*(H.decode_block(kv[b, h].reshape(-1), BS, min(BS, ctx - b * BS))
                       for b in range(kv.shape[0])))
        ks.append(torch.cat(kh))
        vs.append(torch.cat(vh))
    return torch.stack(ks), torch.stack(vs)


def attend(qf, k, v, qpos, kround=False, pround=False, vround=False, kgain=1.0):
    """qf (R, 256) already folded (log2 domain), k/v (ctx, 256) -> rotated output (R, 256)."""
    k = e4m3(k / kgain) if kround else k / kgain
    v = e4m3(v) if vround else v
    s = qf @ k.T
    s = s.masked_fill(torch.arange(k.shape[0])[None, :] > qpos[:, None], -math.inf)
    p = torch.exp2(s - s.amax(-1, keepdim=True))
    return ((e4m3(p) if pround else p) @ v) / p.sum(-1, keepdim=True)


def attend_lazy(qf, k, v, qpos, pshift=0.0, pgrow=8.0, lq=False, vround=True, kround=True, pround=True):
    """The kernel's softmax: 16-key steps, reference max m_ref moved only when a row exceeds it by more
    than pgrow octaves, p = 2^(s - m_ref + pshift) packed e4m3, l sums p (lq: the packed p) in f32.
    ponytail: the rescale decision is per row, the kernel's is per wave (wave_any) -- same m_ref values
    for a row that triggers, a row that does not may keep an older (smaller) m_ref in the kernel."""
    k = e4m3(k) if kround else k
    v = e4m3(v) if vround else v
    R = qf.shape[0]
    m = torch.full((R, 1), -3.0e38)
    acc, l = torch.zeros(R, v.shape[1]), torch.zeros(R, 1)
    keys = torch.arange(k.shape[0])
    for k0 in range(0, k.shape[0], 16):
        s = (qf @ k[k0:k0 + 16].T).masked_fill(keys[None, k0:k0 + 16] > qpos[:, None], -math.inf)
        rmax = s.amax(-1, keepdim=True)
        up = rmax > m + pgrow
        mnew = torch.where(up, torch.maximum(m, rmax).clamp_min(-3.0e38), m)
        a = torch.exp2(m - mnew)
        acc, l, m = acc * a, l * a, mnew
        p = torch.exp2(s - m + pshift)
        p8 = e4m3(p) if pround else p
        l = l + (p8 if lq else p).sum(-1, keepdim=True)
        acc = acc + p8 @ v[k0:k0 + 16]
    return acc / l


def _lift_pow2(x, amax_target=448.0):
    """per-row power of two L with amax(x*L) in (target/2, target]; L=1 for all-zero rows."""
    a = x.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(amax_target / a.clamp_min(1e-30)))
    return torch.where(a > 0, torch.exp2(e.clamp(-120, 120)), torch.ones_like(a))


RLIFT = 8.0   # 1/sV is staged as two e4m3 terms of RLIFT/sV (valid for 0.018 <= sV; the real caches have 3..16)


def attend_b(q, codeK, sK, codeV, sV, qpos, sc, rnd=True, qrnd=True, pgrow=8.0, tile=16, probs=False, lazy=False,
             split=True):
    """Variant B (docs/R4D-INT4-PREFILL.md): bare int4 codes, K token scale on the fp32 score, V token scale on P
    before the e4m3 pack. q (R,256) un-folded rotated query, codes (ctx,256), scales (ctx,1), sc = scale/HEAD*log2e.
    lazy=False: per-row per-tile power-of-two lift of P' into [224,448] (the spec; the accumulator would have to be
    un-lifted every tile, so no kernel does it). lazy=True: what libr4d does -- the reference max lives in the log
    domain x = s + log2 sV, P' = 2^(x - m_ref) packed without a lift, m_ref moves (acc rescales) only when a tile
    exceeds it by pgrow octaves. lazy also weights the packed denominator with the two-term e4m3 split of 1/sV (split=False: exact 1/sV).
    Returns (out/l unquantized, out/lq packed denominator[, P (R,ctx) effective probs]).
    """
    R, ctx = q.shape[0], codeK.shape[0]
    if rnd and qrnd:
        Lq = _lift_pow2(q)
        q = e4m3(q * Lq) / Lq
    m = torch.full((R, 1), -3.0e38)
    acc = torch.zeros(R, codeV.shape[1])
    l, lq = torch.zeros(R, 1), torch.zeros(R, 1)
    keys, tiles = torch.arange(ctx), []
    for k0 in range(0, ctx, tile):
        sl = slice(k0, k0 + tile)
        s = ((q @ codeK[sl].T) * sK[sl, 0][None, :] * sc).masked_fill(keys[None, sl] > qpos[:, None], -math.inf)
        sv = sV[sl, 0][None, :]
        x = s + torch.log2(sv) if lazy else s
        rmax = x.amax(-1, keepdim=True)
        mnew = torch.where(rmax > m + pgrow, torch.maximum(m, rmax).clamp_min(-3.0e38), m)
        a = torch.exp2(m - mnew)
        acc, l, lq, m = acc * a, l * a, lq * a, mnew
        p = torch.exp2(x - m)                       # lazy: already times sV
        pv = p if lazy else p * sv
        rsv = (1.0 / sv.clamp_min(1e-30)) * (sv > 0)
        wsv = rsv                                   # weights of the packed denominator
        if lazy and split:                          # kernel: v_dot4_f32_fp8_fp8 with hi + lo, both e4m3
            hi = e4m3(rsv * RLIFT)
            wsv = (hi + e4m3(rsv * RLIFT - hi)) / RLIFT
        if rnd:
            Lt = torch.ones_like(m) if lazy else _lift_pow2(pv)
            p8 = e4m3(pv * Lt)
            acc = acc + (p8 @ codeV[sl]) / Lt
            w = p8 / Lt * wsv
        else:
            acc, w = acc + pv @ codeV[sl], pv * wsv
        lq = lq + w.sum(-1, keepdim=True)
        l = l + (p * rsv if lazy else p).sum(-1, keepdim=True)
        tiles.append((w, m))
    if probs:
        return acc / l, acc / lq, torch.cat([w * torch.exp2(mt - m) for w, mt in tiles], -1) / lq
    return acc / l, acc / lq


ARMS = {   # name: (q fold e4m3, K e4m3, P e4m3, V e4m3, K gain)
    "Q only": (1, 0, 0, 0, 1.0), "K only": (0, 1, 0, 0, 1.0), "P only": (0, 0, 1, 0, 1.0),
    "V only": (0, 0, 0, 1, 1.0), "kernel (all)": (1, 1, 1, 1, 1.0),
    "fix K/16": (1, 1, 1, 1, 16.0),    # stage K as code*s/16 and fold 16x into the query
}


def analyse(d, rows=256):
    n, nh = d["query"].shape[:2]
    ctx, qstart = int(d["seq_lens"][0]), d["query_start_loc"]
    qlen = int(qstart[1] - qstart[0])
    gqa = nh // d["num_kv_heads"]
    r = torch.linspace(0, qlen - 1, min(rows, qlen)).round().long().unique()
    qpos = ctx - qlen + r
    scale = d["scale"] / H.HEAD
    k, v = decode(d["kv"], ctx)
    q_true = F.rotate(d["query"][r].float())                                      # (R, nh, 256)
    q_in = (d["q8"].float()[r] * d["q_scale"]).to(torch.bfloat16).float()         # what the kernel loads
    unrot = lambda o: F.unrotate(o)
    res = {name: torch.zeros(nh) for name in ("route (dump)", "tetra (dump)", *ARMS, "emul vs dump")}
    den = torch.zeros(nh)
    fold = torch.zeros(nh)
    for hq in range(nh):
        kh, vh = k[hq // gqa], v[hq // gqa]
        truth = unrot(attend(q_true[:, hq] * scale * LOG2E, kh, vh, qpos))
        den[hq] = truth.norm()
        for name, (qr, kr, pr, vr, g) in ARMS.items():
            qf = (q_in if qr else q_true)[:, hq] * scale * LOG2E * g
            out = unrot(attend(e4m3(qf) if qr else qf, kh, vh, qpos, kr, pr, vr, g))
            res[name][hq] = (out - truth).norm()
            if name == "kernel (all)":
                res["emul vs dump"][hq] = (out - d["out"][r, hq].float()).norm()
                fold[hq] = ((qf.abs() < 2.0 ** -6) & (qf != 0)).float().mean()
        res["route (dump)"][hq] = (d["out"][r, hq].float() - truth).norm()
        res["tetra (dump)"][hq] = (d["ref"][r, hq].float() - truth).norm()
    return {name: e / den for name, e in res.items()}, fold, dict(ctx=ctx, qlen=qlen, rows=len(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--rows", type=int, default=256)
    a = ap.parse_args()
    rel, fold, info = analyse(torch.load(a.dump), a.rows)
    print(f"{a.dump}: ctx {info['ctx']}, q_len {info['qlen']}, {info['rows']} rows; rel-L2 vs fp32 truth")
    print(f"{'arm':14s} {'mean':>7s} {'max':>7s} {'worst heads (rel-L2)'}")
    for name, e in rel.items():
        worst = " ".join(f"{int(i)}:{e[i]:.3f}" for i in e.argsort(descending=True)[:4])
        print(f"{name:14s} {e.mean():7.4f} {e.max():7.4f}  {worst}")
    print("q fold subnormal share per head: " + " ".join(f"{x:.2f}" for x in fold.tolist()))


if __name__ == "__main__":
    main()
