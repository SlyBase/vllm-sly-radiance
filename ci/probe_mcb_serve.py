"""In-process MCB serve probe: load the MCB checkpoint with vLLM (eager, no multiproc), find the first module whose
output goes non-finite on a short prompt, and compare a few served linears against the checkpoint dequant.
  VLLM_ENABLE_V1_MULTIPROCESSING=0 RADIANCE_MXFP4_MCB=/m/mcb_tables.pt python3 probe_mcb_serve.py /m
"""
import json, sys
import torch
from safetensors import safe_open
from vllm import LLM, SamplingParams

ck = sys.argv[1]
llm = LLM(model=ck, enforce_eager=True, max_model_len=2048, gpu_memory_utilization=0.85, max_num_seqs=4,
          limit_mm_per_prompt={"image": 0, "video": 0})
core = llm.llm_engine.engine_core.engine_core
model = core.model_executor.driver_worker.worker.model_runner.model
bad = []


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

wm = json.load(open(f"{ck}/model.safetensors.index.json"))["weight_map"]
T = torch.load(f"{ck}/mcb_tables.pt").float().cuda()


def g(k):
    with safe_open(f"{ck}/{wm[k]}", "pt") as h:
        return h.get_tensor(k).cuda()


def deq(b):
    c, s, r = g(b + ".weight_packed"), g(b + ".weight_scale"), g(b + ".weight_row_exp")
    nib = torch.stack([c & 15, c >> 4], -1).reshape(c.shape[0], -1).long()
    m = T[s.long().repeat_interleave(32, 1), nib & 7]
    return torch.where(nib >= 8, -m, m) * torch.exp2(r.float() - 127)[:, None]


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
