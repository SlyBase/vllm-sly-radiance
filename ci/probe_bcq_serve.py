"""In-process BCQ serve probe: load the BCQ checkpoint with vLLM (eager, no multiproc), find the first module whose
output goes non-finite on a short prompt, and compare a few served linears against the checkpoint dequant.
  VLLM_ENABLE_V1_MULTIPROCESSING=0 RADIANCE_MXFP4_BCQ=/m/bcq_tables.pt python3 probe_bcq_serve.py /m
"""
import json, os, sys
import torch
from safetensors import safe_open
from vllm import LLM, SamplingParams

ck = sys.argv[1]
llm = LLM(model=ck, enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.85, max_num_seqs=4,
          limit_mm_per_prompt={"image": 0, "video": 0})
core = llm.llm_engine.engine_core.engine_core
model = core.model_executor.driver_worker.worker.model_runner.model
bad = []
import radiance_mxfp4 as _RM0
print("GET_HAD after load:", _RM0._ext.get_had() if hasattr(_RM0._ext, "get_had") else "n/a",
      "ext file:", getattr(_RM0._ext, "__file__", "?"), flush=True)
import sys as _s
print("loaded fp8 modules:", [m for m in _s.modules if "mxfp4_fp8" in m], flush=True)
if _RM0.HAD:
    _hs = _RM0._ext.get_had_signs()
    _want = _RM0.had_signs(128).tolist()
    print("HAD SIGNS on device match:", max(abs(a - b) for a, b in zip(_hs, _want)) if _hs else "read failed", _hs[:4], flush=True)
    _gu = torch.randn(4, 2 * 17408, device="cuda", dtype=torch.bfloat16)
    _q, _sc = torch.ops.radiance.silu_mul_quant(_gu)
    _v = (torch.nn.functional.silu(_gu[:, :17408].float()).to(torch.bfloat16).float() * _gu[:, 17408:].float()).to(torch.bfloat16)
    _ref = (_v.float().view(4, -1, 128) @ _RM0.had_matrix(128, "cuda")).view(4, -1)
    _got = _q.float() * _sc.view(-1, 1)
    print("SILU OP rel vs rotated ref:", float((_got - _ref).norm() / _ref.norm()),
          "vs unrotated:", float((_got - _v.float()).norm() / _v.float().norm()), flush=True)


def hook(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        if torch.is_tensor(o) and o.is_floating_point() and not bool(torch.isfinite(o).all()) and len(bad) < 8:
            i = inp[0] if inp and torch.is_tensor(inp[0]) else None
            bad.append(f"{name} out-nonfinite in-finite={None if i is None else bool(torch.isfinite(i).all())}")
    return f


for n, m in model.named_modules():
    if n.count(".") <= 7:
        m.register_forward_hook(hook(n))
out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
print("GEN", repr(out[0].outputs[0].text))
print("FIRST NONFINITE:", *bad, sep="\n  ")
if os.environ.get("PROBE_GEN_ONLY") == "1":
    sys.exit(0)

wm = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]
import os
import radiance_mxfp4 as RM
HAD = RM.HAD
tf = f"{ck}/bcq_tables.pt" if os.path.exists(f"{ck}/bcq_tables.pt") else f"{ck}/mcb_tables.pt"
T = torch.load(tf).float().cuda()


def g(k):
    with safe_open(f"{ck}/{wm[k]}", "pt") as h:
        return h.get_tensor(k).cuda()


def deq(b):
    c, s, r = g(b + ".weight_packed"), g(b + ".weight_scale"), g(b + ".weight_row_exp")
    nib = torch.stack([c & 15, c >> 4], -1).reshape(c.shape[0], -1).long()
    m = T[s.long().repeat_interleave(32, 1), nib & 7]
    W = torch.where(nib >= 8, -m, m) * torch.exp2(r.float() - 127)[:, None]
    if HAD:  # stored W R: un-rotate to the logical weight
        n, k = W.shape
        W = (W.view(n, k // HAD, HAD) @ RM.had_matrix(HAD, W.device).T).view(n, k)
    return W


P = "model.language_model.layers."
mods = dict(model.named_modules())
for vname, parts in [("0.mlp.gate_up_proj", ["0.mlp.gate_proj", "0.mlp.up_proj"]), ("0.mlp.down_proj", ["0.mlp.down_proj"]),
                     ("0.linear_attn.out_proj", ["0.linear_attn.out_proj"]),
                     ("3.self_attn.qkv_proj", ["3.self_attn.q_proj", "3.self_attn.k_proj", "3.self_attn.v_proj"]),
                     ("3.self_attn.o_proj", ["3.self_attn.o_proj"])]:
    key = next((k for k in mods if k.endswith("layers." + vname)), None)
    if key is None:
        print("no module", vname); continue
    lin = mods[key]
    W = torch.cat([deq(P + p) for p in parts])
    for M in (1, 8, 64, 600):
        x = torch.randn(M, W.shape[1], device="cuda", dtype=torch.bfloat16)
        y = lin.quant_method.apply(lin, x)
        y = y[0] if isinstance(y, tuple) else y
        ref = x.float() @ W.T
        print(f"{vname} M={M} rel={float((y.float() - ref).norm() / ref.norm()):.4f} finite={bool(torch.isfinite(y).all())} "
              f"wref[{int(lin.radiance_wref.min())},{int(lin.radiance_wref.max())}]")
