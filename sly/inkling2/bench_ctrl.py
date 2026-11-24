"""Micro-benchmark for the GPU window: Inkling-2-Mini loop message controllers, torch.bmm vs the batched HIP GEMV.

  bmm     the two torch.bmm of Inkling2LoopDecoderLayer.blend (hipBLASLt): the model's own path
  hip     radiance_inkling2_gemv_hip.controller_gemms (RADIANCE_INKLING2_HIP_CTRL=1): two gemv_bf16_batched launches

Real shapes: S = 3 streams (attn, k, v; local layers) or 1 (mlp), T = 1..8 verify rows, 2D = 4608, r = 64.
Weights ROTATE through a pool of >= 280 MB (4x the Infinity Cache; 12 layers are only 57 MB), calls are captured
into ONE CUDA graph and the replay is timed with events (no host launch cost, like the decode graph). Every cell
is checked against an fp64 reference first (hidden rounded to bf16 as in the model).
`step_ms` = us per call x (24 S=3 + 24 S=1 visits per verify step) = what the controllers cost per step.

  python3 bench_ctrl.py [--ts 1,2,4,8] [--json out.json]
Output: one flushed line per cell: `S T arm us ok maxabs`; last line: projected step_ms per arm at T = 8.
"""
import argparse
import json
import sys

import torch

import radiance_inkling2_gemv_hip as disp
from bench_gemv import graph_us

D2, R = 4608, 64
POOL_BYTES = 280 * 1024 * 1024
VISITS = {3: 24, 1: 24}  # per verify step: attn blend (S=3) and mlp blend (S=1) in 2 passes x 12 layers


def bmm_path(pair, down, gf):
    hidden = torch.bmm(pair, down.transpose(1, 2))
    return torch.bmm(hidden, gf.transpose(1, 2), out_dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ts", default="1,2,4,8")
    ap.add_argument("--json")
    a = ap.parse_args()
    if not (disp._ext is not None and hasattr(disp._ext, "gemv_bf16_batched")):
        sys.exit("no radiance_inkling2_gemv build with gemv_bf16_batched")
    disp.HIP_CTRL = True
    g = torch.Generator(device="cuda").manual_seed(0)
    per = (R * D2 * 2) * 2  # bytes of one stream's down + gain/forget
    res, step = [], {"bmm": 0.0, "hip": 0.0}
    for s in (3, 1):
        npool = max(2, POOL_BYTES // (per * s) + 1)
        down = [(torch.randn(s, R, D2, device="cuda", generator=g) * 0.05).bfloat16() for _ in range(npool)]
        gf = [(torch.randn(s, D2, R, device="cuda", generator=g) * 0.2).bfloat16() for _ in range(npool)]
        for t in (int(v) for v in a.ts.split(",")):
            pair = torch.randn(s, t, D2, device="cuda", generator=g).bfloat16()
            arms = {"bmm": bmm_path, "hip": disp.controller_gemms}
            ref = (torch.bmm(torch.bmm(pair.double(), down[0].double().transpose(1, 2)).bfloat16().double(),
                             gf[0].double().transpose(1, 2))).float()
            for name, fn in arms.items():
                out = fn(pair, down[0], gf[0])
                torch.cuda.synchronize()
                err = float((out - ref).abs().max())
                ok = bool(torch.allclose(out, ref, rtol=1e-2, atol=1e-2))  # 1 bf16 ulp of hidden at most
                us = graph_us([(lambda i=i: fn(pair, down[i], gf[i])) for i in range(npool)])
                print(f"{s} {t} {name} {us:.2f} {ok} {err:.3g}", flush=True)
                res.append(dict(S=s, T=t, arm=name, us=us, ok=ok, maxabs=err))
                if t == 8:
                    step[name] += us * VISITS[s] / 1e3
    # one pass of the whole blend = 2 bmm + the surrounding norm kernels (not timed here): the cells above
    # are the two products only
    print(f"projected controller GEMM cost per verify step at T=8 (ms): bmm {step['bmm']:.3f}  hip {step['hip']:.3f}"
          f"  saving {step['bmm'] - step['hip']:.3f}", flush=True)
    if a.json:
        json.dump(dict(cells=res, step_ms=step), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
