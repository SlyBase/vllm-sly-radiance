"""CPU-only (no GPU): MXFP4 quantization error on the REAL Inkling-2-Mini dense weights.
Validates the load-time quantizer against the format-spec dequant and reports relRMS per tensor
(the weight-quantization quality proxy; MXFP4 expected ~0.11-0.13). Runs in the image with the GPU
hidden, prod keeps the card. Also reports the per-step dense byte budget (bf16 vs packed, x visits).
"""
import struct
import json
import torch
from safetensors import safe_open

import radiance_inkling2_mxfp4dense as MD

P = "/model/model-00001-of-00001.safetensors"
E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])


def ref_dequant(packed, scale_kn):
    n, kh = packed.shape
    k = kh * 2
    c = torch.stack([packed & 0x0F, (packed >> 4) & 0x0F], -1).reshape(n, k).long()
    sc = torch.exp2(scale_kn.float() - 127.0).T.repeat_interleave(32, 1)
    return E2M1[c] * sc


# representative dense tensors (one per class) + the head
SAMPLES = [
    "model.llm.layers.0.attn.wq_du.weight",
    "model.llm.layers.0.attn.gate_attn.weight",
    "model.llm.layers.0.attn.wk_dv.weight",
    "model.llm.layers.0.attn.wo_ud.weight",
    "model.llm.layers.11.controllers.attn.down_weight",
    "model.llm.unembed.weight",
]
f = safe_open(P, framework="pt")
print(f"{'tensor':55s} {'shape':>16} {'rtn_relRMS':>10} {'mse_relRMS':>10} {'dequant_ok':>10}")
for name in SAMPLES:
    if name not in f.keys():
        print(name, "MISSING")
        continue
    w = f.get_tensor(name).to(torch.bfloat16)
    if w.ndim != 2 or w.shape[1] % 32:
        print(name, "skip (not 2D / K%32)")
        continue
    p, s, r = MD.quantize_mxfp4(w, clip="rtn")
    pm, sm, _ = MD.quantize_mxfp4(w, clip="mse")
    dq = MD.dequant_mxfp4(p, s).float()
    ok = torch.equal(dq, ref_dequant(p, s))
    wf = w.float()
    rel_rtn = (dq - wf).norm() / wf.norm()
    rel_mse = (MD.dequant_mxfp4(pm, sm).float() - wf).norm() / wf.norm()
    print(f"{name:55s} {str(tuple(w.shape)):>16} {rel_rtn:10.4f} {rel_mse:10.4f} {str(ok):>10}")

# per-step dense byte budget: bf16 vs MXFP4 packed, weighted by layer visits
cfg = json.load(open("/model/config.json"))
tc = cfg.get("text_config", cfg)
lc = tc.get("loop_config", {})
core = set(range(lc.get("core_start", 6), lc.get("core_end", 18)))
npass = lc.get("num_passes", 3)
nl = tc.get("num_hidden_layers", 24)


def visits(i):
    return npass if i in core else 1


hdr = {}
with open(P, "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n))
bf16_bytes = packed_bytes = 0
for k, v in hdr.items():
    if k == "__metadata__" or v.get("dtype") != "BF16":
        continue
    sh = v.get("shape", [])
    if len(sh) != 2 or sh[0] < 256 or sh[1] % 64:
        continue  # only quantizable dense (N>=256, K%64)
    if ".layers." in k:
        i = int(k.split(".layers.")[1].split(".")[0])
        vis = visits(i)
    elif "unembed" in k:
        vis = 1
    else:
        continue
    nb = sh[0] * sh[1]
    bf16_bytes += nb * 2 * vis
    packed_bytes += (nb // 2 + nb // 32) * vis  # 4-bit codes + 1 E8M0 byte / 32
print(f"\nPer-step quantizable dense reads: bf16 {bf16_bytes/1e9:.3f} GB -> MXFP4 {packed_bytes/1e9:.3f} GB "
      f"({bf16_bytes/packed_bytes:.2f}x fewer bytes)")
print("CPU_QUANT_DONE")