"""Micro-benchmark for the GPU window: DSpark Markov fold + softcap, bf16 fold weight vs the int8 copy
(RADIANCE_INKLING2_FOLD_INT8=1).

Real shape: V = 201024, r = 256, rows 1..8 (one draft row per request; 7 Markov steps per verify step). The fold
weight is 103 MB (bf16) / 51 MB (int8) -- both above the 64 MB Infinity Cache, so repeated calls on one weight
stream cold bytes. Replay of one CUDA graph, events. Error vs the bf16 fold: max |d logit|, share of rows with
the same argmax, top-20 overlap (what the sampler's top_k = 20 keeps).

  python3 bench_fold.py [--rows 1,2,4,8] [--json out.json]
Output per row count: `rows arm us GBps maxdiff argmax_same top20_overlap`; last line: saving per verify step (7 steps).
"""
import argparse
import json

import torch

import radiance_inkling2_draft as d
from bench_gemv import graph_us

V, R = 201024, 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="1,2,4,8")
    ap.add_argument("--json")
    a = ap.parse_args()
    g = torch.Generator(device="cuda").manual_seed(0)
    fw = (torch.randn(V, R, device="cuda", generator=g) * 0.1).bfloat16()
    fb = torch.randn(V, device="cuda", generator=g).bfloat16()
    res = []
    for rows in (int(v) for v in a.rows.split(",")):
        base = (torch.randn(rows, V, device="cuda", generator=g) * 6).bfloat16()
        lat = torch.tanh(torch.randn(rows, R, device="cuda", generator=g)).bfloat16()
        outs, us = {}, {}
        for name, flag in (("bf16", False), ("int8", True)):
            d.FOLD_INT8 = flag
            outs[name] = d.fold_softcap(base, fw, fb, lat, 30.0).float()  # eager first: makes the int8 copy
            torch.cuda.synchronize()
            us[name] = graph_us([lambda: d.fold_softcap(base, fw, fb, lat, 30.0)] * 4)
        d.FOLD_INT8 = False
        diff = float((outs["int8"] - outs["bf16"]).abs().max())
        same = float((outs["int8"].argmax(1) == outs["bf16"].argmax(1)).float().mean())
        a20, b20 = outs["int8"].topk(20, 1).indices.tolist(), outs["bf16"].topk(20, 1).indices.tolist()
        t20 = sum(len(set(i) & set(j)) / 20 for i, j in zip(a20, b20)) / len(a20)
        for name, nbytes in (("bf16", V * R * 2), ("int8", V * R + V * 4)):
            print(f"{rows} {name} {us[name]:.1f} {nbytes / us[name] / 1e3:.0f} {diff:.3g} {same:.3f} {t20:.3f}", flush=True)
        res.append(dict(rows=rows, us=us, maxdiff=diff, argmax_same=same, top20=t20))
    r7 = next((r for r in res if r["rows"] == 8), res[-1])
    print(f"saving per verify step (7 Markov steps): {7 * (r7['us']['bf16'] - r7['us']['int8']) / 1e3:.3f} ms", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
