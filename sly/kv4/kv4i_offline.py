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
