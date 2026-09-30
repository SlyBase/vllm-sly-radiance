#!/usr/bin/env python3
"""CPU check of radiance_nvfp4_diag's schemes on real NVFP4 layers of a compressed-tensors checkpoint.

Builds each diagnostic scheme on a layer, runs apply_weights on a random activation and reports the
output error against an fp32 reference dequant of the checkpoint (radiance_nvfp4.dequant_nvfp4).
Expected (ThinkingCap-Qwen3.8-27B-NVFP4, 2026-09-29): native ~0.0025 (bf16 rounding only), requant ~0.11,
fold ~0.021, fold + A8 ~0.034. Runs inside the image without a GPU:

  docker run --rm --entrypoint python3 -e HIP_VISIBLE_DEVICES= -v <ckpt>:/ckpt:ro -v $PWD:/w:ro \
      vllm-sly-radiance:<ver>-rocm10.0 -P /w/sly/check_nvfp4_diag.py /ckpt

-P keeps the script directory off sys.path; the repo root is APPENDED so an image that predates
radiance_nvfp4_diag.py still finds the checkout's copy, while every other module comes from the image.
"""
import importlib
import json
import os
import sys

import torch
from safetensors import safe_open

import vllm.model_executor.parameter as _P

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# the parameter classes ask for the TP rank at construction; there is no process group on CPU
_P.get_tensor_model_parallel_rank = lambda: 0
_P.get_tensor_model_parallel_world_size = lambda: 1

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/ckpt"
LAYERS = ("model.language_model.layers.3.mlp.down_proj", "model.language_model.layers.3.self_attn.o_proj")
MODES = ("native", "requant", "fold", "fold+a8")
BOUND = {"native": 0.01, "requant": 0.13, "fold": 0.03, "fold+a8": 0.05}


def tensors(idx, name):
    t = {}
    for suf in ("weight_packed", "weight_scale", "weight_global_scale"):
        k = f"{name}.{suf}"
        with safe_open(f"{CKPT}/{idx[k]}", "pt", device="cpu") as f:
            t[suf] = f.get_tensor(k)
    return t


def main():
    import radiance_nvfp4 as R
    idx = json.load(open(f"{CKPT}/model.safetensors.index.json"))["weight_map"]
    fails = 0
    for name in LAYERS:
        t = tensors(idx, name)
        n, k2 = t["weight_packed"].shape
        torch.manual_seed(0)
        x = torch.randn(16, k2 * 2, dtype=torch.bfloat16)
        g = float(t["weight_global_scale"].reshape(-1)[0])
        yref = x.float() @ R.dequant_nvfp4(t["weight_packed"], t["weight_scale"], g).T
        for mode in MODES:
            os.environ["RADIANCE_NVFP4_DIAG"] = mode.split("+")[0]
            os.environ["RADIANCE_NVFP4_DIAG_A8"] = "1" if mode.endswith("a8") else "0"
            import radiance_nvfp4_diag
            D = importlib.reload(radiance_nvfp4_diag)
            cls = D.scheme_class()
            layer = torch.nn.Module()
            cls().create_weights(layer, [n], k2 * 2, torch.bfloat16, lambda p, w: p.data.copy_(w))
            layer.weight_packed.data.copy_(t["weight_packed"])
            layer.weight_scale.data.copy_(t["weight_scale"])
            layer.weight_global_scale.data.fill_(g)
            cls().process_weights_after_loading(layer)
            y = cls().apply_weights(layer, x).float()
            err = float((y - yref).norm() / yref.norm())
            ok = err <= BOUND[mode]
            fails += not ok
            print(f"{'OK ' if ok else 'BAD'} {name.split('layers.')[1]:22} {mode:8} out relErr {err:.4f} (<= {BOUND[mode]})")
    print("PASS" if not fails else f"FAIL: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
