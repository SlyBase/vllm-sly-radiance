"""M2: ISA spill probe on the IMAGE's triton (3.8), patched kernel as shipped.
warps 2/4/8 x window 0/1024, BLOCK_N=32, VEC_KV=True, num_stages=1.
"""
import importlib
import json
import re

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

m = importlib.import_module("vllm.models.inkling2.nvidia.ops.attention_sm90")
kern = m._rel_attention_kernel
POINTERS = {"Q": "*bf16", "K": "*fp8e4nv", "V": "*fp8e4nv", "Rel": "*bf16",
            "Out": "*fp32", "Lse": "*fp32", "Gate": "*bf16", "Sink": "*bf16",
            "BlockTable": "*i32", "SeqLens": "*i32", "CuSeqlensQ": "*i32",
            "softmax_scale": "fp32"}
for window in (0, 1024):
    for w in (2, 4, 8):
        consts = dict(H=24, G=3, D=128, E=1024, PAGE=128, WINDOW=window,
                      DRAFT=False, MODE=2, BLOCK_M=32, BLOCK_N=32,
                      BULK=True, VEC_KV=True)
        if "LAZY" in kern.arg_names:
            consts["LAZY"] = False
        sig = {n: POINTERS.get(n, "i32") for n in kern.arg_names if n not in consts}
        code = triton.compile(ASTSource(kern, sig, consts),
                              target=GPUTarget("hip", "gfx1201", 32),
                              options={"num_warps": w, "num_stages": 1})
        asm = code.asm["amdgcn"]
        spills = int(re.search(r"\.vgpr_spill_count:\s*(\d+)", asm).group(1))
        print(json.dumps(dict(window=window, warps=w, spills=spills,
                              b128=len(re.findall(r"global_load_b128", asm)),
                              u8="global_load_u8" in asm,
                              triton=triton.__version__)), flush=True)
print("M2 DONE", flush=True)
