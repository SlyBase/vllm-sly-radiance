#!/usr/bin/env python3
"""Compare sly/bench_fidelity.py arms against a reference arm on the same tokens.

Per arm: mean NLL, paired dNLL vs the reference with a 95 % interval (tokens as the unit), KL(ref||arm)
over the reference's top-K (a token missing from the arm's top-K gets the arm's smallest top-K logprob,
so KL is a lower bound), and top-1 agreement. Exit 1 with --max-dnll if any arm exceeds the bound.

usage: check_fidelity.py REF.json ARM.json [ARM.json ...] [--max-dnll 0.02]
"""
import argparse
import json
import math
import sys


def compare(ref, x):
    d, kl, agree, n = [], 0.0, 0, 0
    for name, rr in ref["texts"].items():
        xr = x["texts"].get(name)
        if not xr or len(xr) != len(rr):
            continue
        for (tid, lpr, topr), (tid2, lpx, topx) in zip(rr, xr):
            assert tid == tid2, f"{name}: token mismatch (different tokenizer or text?)"
            d.append(lpr - lpx)
            floor = min(topx.values())
            kl += sum(math.exp(lp) * (lp - topx.get(t, floor)) for t, lp in topr.items())
            agree += max(topr, key=topr.get) == max(topx, key=topx.get)
            n += 1
    m = sum(d) / n
    se = (sum((v - m) ** 2 for v in d) / (n - 1)) ** 0.5 / n ** 0.5
    nll = -sum(r[1] for t in x["texts"].values() for r in t) / sum(len(t) for t in x["texts"].values())
    return {"n": n, "nll": nll, "dnll": m, "ci95": 1.96 * se, "kl": kl / n, "top1": agree / n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--max-dnll", type=float)
    a = ap.parse_args()
    ref = json.load(open(a.ref))
    print(f"reference: {a.ref}")
    bad = False
    for path in a.arms:
        r = compare(ref, json.load(open(path)))
        print(f"{path:34} n={r['n']} NLL={r['nll']:.4f}  dNLL={r['dnll']:+.4f} (+-{r['ci95']:.4f})  "
              f"KL={r['kl']:.4f}  top1={r['top1']:.4f}")
        if a.max_dnll is not None and r["dnll"] > a.max_dnll:
            bad = True
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
