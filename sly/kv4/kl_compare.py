"""Compare arms against the bf16 reference: KL(ref||arm) on the top-K support (+ tail bucket),
top-1 agreement, mean NLL difference of the true token; per context-position bin and overall."""
import os, sys
import numpy as np

D = os.path.abspath(os.environ.get("KVACC_DIR", "kvacc"))   # raw/ texts in, ids/ + out/<arm>/ written
REF = "bf16"
ARMS = sys.argv[1:] or ["bf16r", "fp8", "kv4"]
BINS = [(1, 4096), (4096, 16384), (16384, 32768), (32768, 65536), (65536, 131072)]


def load(arm, doc):
    z = np.load(f"{D}/out/{arm}/{doc}.npz")
    return z["lp_true"], z["top_ids"], z["top_lp"]


def per_pos(ref, arm):
    """per position: KL estimate, top1 match, dNLL (arm - ref) of the true token"""
    r_true, r_ids, r_lp = ref
    a_true, a_ids, a_lp = arm
    n = len(r_true)
    kl = np.zeros(n); top1 = np.zeros(n, bool)
    for i in range(1, n):
        amap = dict(zip(a_ids[i].tolist(), a_lp[i].tolist()))
        afloor = a_lp[i][-1]  # tokens outside the arm's top-K: at most this logprob
        pr = np.exp(r_lp[i])
        la = np.array([amap.get(t, afloor) for t in r_ids[i].tolist()])
        k = float(np.sum(pr * (r_lp[i] - la)))
        rt = max(1e-9, 1 - pr.sum()); at = max(1e-9, 1 - np.exp(la).sum())
        kl[i] = max(0.0, k + rt * np.log(rt / at))
        top1[i] = r_ids[i][0] == a_ids[i][0]
    dnll = -(a_true - r_true)
    return kl, top1, dnll


def main():
    docs = sorted(f[:-4] for f in os.listdir(f"{D}/out/{REF}") if f.endswith(".npz"))
    for arm in ARMS:
        if not os.path.isdir(f"{D}/out/{arm}"):
            continue
        allk, allt, alld, pos = [], [], [], []
        print(f"\n=== {arm} vs {REF}")
        print(f"{'doc':16s} {'KL mean':>10s} {'KL p99':>9s} {'top1 %':>8s} {'dNLL':>9s} {'ppl ref':>8s} {'ppl arm':>8s}")
        for doc in docs:
            if not os.path.exists(f"{D}/out/{arm}/{doc}.npz"):
                continue
            ref, a = load(REF, doc), load(arm, doc)
            kl, t1, dn = per_pos(ref, a)
            s = slice(1, None)
            print(f"{doc:16s} {kl[s].mean():10.5f} {np.percentile(kl[s], 99):9.4f} {100 * t1[s].mean():8.2f} "
                  f"{np.nanmean(dn[s]):+9.5f} {np.exp(-np.nanmean(ref[0][s])):8.3f} {np.exp(-np.nanmean(a[0][s])):8.3f}")
            allk.append(kl[s]); allt.append(t1[s]); alld.append(dn[s]); pos.append(np.arange(1, len(kl)))
        k, t, d, p = map(np.concatenate, (allk, allt, alld, pos))
        print(f"{'ALL':16s} {k.mean():10.5f} {np.percentile(k, 99):9.4f} {100 * t.mean():8.2f} {np.nanmean(d):+9.5f}")
        print("  by context position:   " + "  ".join(f"{a // 1024}-{b // 1024}K" .rjust(16) for a, b in BINS))
        row = lambda f: "  ".join(f(m).rjust(16) for m in [(p >= a) & (p < b) for a, b in BINS])
        print("  KL mean               " + row(lambda m: f"{k[m].mean():.5f}"))
        print("  top-1 agree %         " + row(lambda m: f"{100 * t[m].mean():.2f}"))
        print("  dNLL (nats/token)     " + row(lambda m: f"{np.nanmean(d[m]):+.5f}"))


main()
