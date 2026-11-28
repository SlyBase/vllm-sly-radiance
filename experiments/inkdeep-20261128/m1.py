"""M1: global decode tile micro-sweep at short AND deep ctx, warps x CTAS_PER_SM.
Question: can warps=2 keep its deep-ctx win and recover the short-ctx (MIA) loss
via more splits (higher CTAS_PER_SM)? Runs inside the 1.15.1 image, GPU free.
"""
import importlib.util
import json

import torch

spec = importlib.util.spec_from_file_location("bd", "/work/bench_decode_attn.py")
bd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bd)

for ctx in (2000, 8192, 65536, 150000):
    x = bd.inputs(ctx)
    base, _ = bd.launch(x, False, False)
    for w in (2, 4, 8):
        for c in (16, 32, 64):
            bd.S0._RAD_DEC_BN, bd.S0._RAD_DEC_WARPS = 32, w
            bd.decode_path(True)
            bd.A._ROCM_CTAS_PER_SM = c
            s = bd.plan(False)
            o = torch.zeros(bd.T, bd.H * bd.D, device="cuda", dtype=torch.bfloat16)
            bd.call(x, s, o, False)
            torch.cuda.synchronize()
            err = (o.float() - base).abs().max().item()
            us = bd.bench(lambda: bd.call(x, s, o, False))
            print(json.dumps(dict(ctx=ctx, warps=w, ctas=c, splits=s,
                                  us=round(us, 1), err=round(err, 5))), flush=True)
print("M1 DONE", flush=True)
