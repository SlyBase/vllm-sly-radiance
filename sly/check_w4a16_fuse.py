#!/usr/bin/env python3
"""End-to-end check of the W4A16 post-load transforms of sly/patch_w4a16_tiles.py (sly/patch_w4a16_fuse.py
call sites) on stand-in layers with the RDNAHybridW4A16LinearKernel interface (scheme.kernel,
weight_packed / weight_scale, _get_weight_params / _transform_param), against the unfused computation:

  * MLP gate_up + silu: rows interleaved by the transform, the fused op body (_radiance_w4a16_silu_impl)
    equals silu(gate) * up of the untransformed layer -- on the torch fallback and on the EPI 2 kernel
    (default decode config); a plain call de-interleaved as apply_weights does is bit-identical to the
    original layer's output,
  * GDN qkvz + ba merge (RADIANCE_GDN_BA_W4=1): qkvz columns match, ba within the int4 re-quantization
    error, the bf16 ba weight is released, the merge is off by default,
  * DFlash context-KV on the drafter's own int4 K/V rows: equal to the dequantized bf16 fused weight the
    stock path would build (same codes, same scales),
  * DFlash2 grouped-conv kernel_projection -> RadianceW4Linear within the int4 error,
  * the model-source markers: no transform where the call site is missing.

CPU (Triton interpreter), in a plain `docker run` without device flags, after applying the patch:
  BENCH_DEVICE=cpu TRITON_INTERPRET=1 python3 check_w4a16_fuse.py
"""
import os
import sys
import types

import torch
from torch import nn

from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as H

_RADIANCE_W4_SILU = True      # this module plays the patched model source (sly/patch_w4a16_fuse.py markers)
_RADIANCE_W4_QKVZ_BA = True

dev = torch.device(os.environ.get("BENCH_DEVICE", "cuda"))
DT = torch.float16 if dev.type == "cpu" else torch.bfloat16  # the interpreter mis-casts int -> bf16
GS = 128
K = 1024
torch.manual_seed(0)
SHIFTS = torch.tensor([(j // 2) * 4 + (j % 2) * 16 for j in range(8)], dtype=torch.int32)
fails = 0


def report(ok, what, detail=""):
    global fails
    print(f"{'ok  ' if ok else 'FAIL'} {what} {detail}", flush=True)
    fails += not ok


def dequant(rows_i32, scales):
    q = ((rows_i32.cpu()[:, :, None] >> SHIFTS[None, None, :]) & 0xF).reshape(rows_i32.shape[0], -1).float()
    return (q - 8) * scales.cpu().float().repeat_interleave(GS, dim=1)


class RDNAHybridW4A16LinearKernel:  # the name is what _radiance_w4_parts checks
    w_q_name, w_s_name = "weight_packed", "weight_scale"

    def __init__(self):
        self.config = types.SimpleNamespace(group_size=GS)

    def _get_weight_params(self, layer):
        return getattr(layer, self.w_q_name), getattr(layer, self.w_s_name), None

    def _transform_param(self, layer, name, fn):
        setattr(layer, name, nn.Parameter(fn(getattr(layer, name)), requires_grad=False))


class W4Linear(nn.Module):
    def __init__(self, n, k=K):
        super().__init__()
        self.rows = torch.randint(-2**31, 2**31 - 1, (n, k // 8), dtype=torch.int32, device=dev)
        self.weight_packed = nn.Parameter(H.radiance_w4a16_tile(self.rows), requires_grad=False)
        self.weight_scale = nn.Parameter((torch.rand(n, k // GS) * 0.02 + 0.001).to(DT).to(dev), requires_grad=False)
        self.scheme = types.SimpleNamespace(kernel=RDNAHybridW4A16LinearKernel())
        self.ref_w = dequant(self.rows, self.weight_scale)  # fp32 [n, k]


class MLP(nn.Module):
    def __init__(self, inter):
        super().__init__()
        self.gate_up_proj = W4Linear(2 * inter)
        self.down_proj = nn.Identity()


class GDN(nn.Module):
    def __init__(self, n1):
        super().__init__()
        self.in_proj_qkvz = W4Linear(n1)
        self.in_proj_ba = nn.Linear(K, 96, bias=False).to(DT).to(dev)


class DFlashGroupedConv(nn.Module):  # the name is what radiance_w4a16_postload checks
    def __init__(self):
        super().__init__()
        self.kernel_projection = nn.Linear(K, 64, bias=False).to(DT).to(dev)


def call(op, impl, *args):
    """a registered op when the CPU dispatch has it, else the op's body (same code path below the op)"""
    try:
        return op(*args)
    except Exception:  # noqa: BLE001 -- no CPU kernel registered for the vllm op namespace
        return impl(*args)


x = torch.randn(8, K).to(DT).to(dev)
fl = lambda ref: ref.abs().max().item() * 2 ** -6  # noqa: E731
on_gfx12x = H._on_gfx12x

# --- MLP gate_up + silu ---
mlp = MLP(64)
lin = mlp.gate_up_proj
ref = x.cpu().float() @ lin.ref_w.t()
ref_silu = torch.nn.functional.silu(ref[:, :64]) * ref[:, 64:]
plain_before = H.triton_w4a16_skinny_fmt_gemm(x, lin.weight_packed.view(torch.int32), lin.weight_scale, GS).cpu()
report(H._radiance_postload_mlp(mlp) and lin._radiance_silu, "MLP transform applied")
for gfx in (False, True):
    H._on_gfx12x = (lambda v: (lambda: v))(gfx)
    out = H._radiance_w4a16_silu_impl(x, lin.weight_packed, lin.weight_scale, GS).cpu()
    err = (out.float() - ref_silu).abs().max().item()
    report(out.shape == (8, 64) and err <= fl(ref_silu), f"MLP fused silu ({'EPI 2 kernel' if gfx else 'torch fallback'})",
           f"err {err:.3e}")
H._on_gfx12x = on_gfx12x
inter = H.triton_w4a16_skinny_fmt_gemm(x, lin.weight_packed.view(torch.int32), lin.weight_scale, GS).cpu()
restored = inter.view(-1, 64, 2).transpose(1, 2).reshape(-1, 128)  # what apply_weights does for a plain call
report(torch.equal(restored, plain_before), "MLP plain call de-interleaved == original layer (bit-identical)")

# --- GDN qkvz + ba ---
g = GDN(256)
report(H.radiance_w4a16_postload(g)["gdn_ba"] == 0 and not getattr(g.in_proj_qkvz, "_radiance_ba", 0),
       "GDN merge off by default")
ref_q = x.cpu().float() @ g.in_proj_qkvz.ref_w.t()
ref_ba = x.cpu().float() @ g.in_proj_ba.weight.cpu().float().t()
os.environ["RADIANCE_GDN_BA_W4"] = "1"
report(H.radiance_w4a16_postload(g)["gdn_ba"] == 1 and g.in_proj_qkvz._radiance_ba == 256
       and g.in_proj_ba.weight.numel() == 0, "GDN merge applied, bf16 ba released")
del os.environ["RADIANCE_GDN_BA_W4"]
for gfx in (False, True):
    H._on_gfx12x = (lambda v: (lambda: v))(gfx)
    c1, c2 = H._radiance_w4a16_split_impl(x, g.in_proj_qkvz.weight_packed, g.in_proj_qkvz.weight_scale, 256, GS)
    e1 = (c1.cpu().float() - ref_q).abs().max().item()
    rel = ((c2.cpu().float() - ref_ba).square().mean() / ref_ba.square().mean()).sqrt().item()
    report(c1.shape == (8, 256) and c2.shape == (8, 96) and c1.is_contiguous() and c2.is_contiguous()
           and e1 <= fl(ref_q) and rel < 0.15, f"GDN merged qkvz + ba ({'EPI 1 kernel' if gfx else 'torch fallback'})",
           f"qkvz err {e1:.3e}, ba rel rms {rel:.3f}")
H._on_gfx12x = on_gfx12x

# --- DFlash context-KV on the int4 rows ---
layers = [types.SimpleNamespace(qkv_proj=W4Linear(128 + 64), q_size=128) for _ in range(5)]
model = types.SimpleNamespace(_kv_source_attn=layers, _fused_kv_bias=None)
ref_kv = x.cpu().float() @ torch.cat([a.qkv_proj.ref_w[a.q_size:] for a in layers]).t()
try:
    kv = H.radiance_dflash_kv_project(model, x)
except Exception:  # noqa: BLE001 -- op not callable on the CPU: its body on the state the call just built
    q, s, gs = model._radiance_kv_w4
    kv = H._radiance_w4a16_gemm_impl(x, q, s, gs)
err = (kv.cpu().float() - ref_kv).abs().max().item()
report(kv.shape == (8, 5 * 64) and err <= fl(ref_kv) and model._radiance_kv_w4[0].dim() == 4,
       "DFlash context-KV on int4 rows (tiled)", f"err {err:.3e}")

# --- DFlash2 conv kernel_projection -> int4 ---
conv = DFlashGroupedConv()
ref_c = x.cpu().float() @ conv.kernel_projection.weight.cpu().float().t()
cnt = H.radiance_w4a16_postload(conv)
report(cnt["conv_w4"] == 1 and isinstance(conv.kernel_projection, H.RadianceW4Linear), "conv projection replaced")
kp = conv.kernel_projection
out = call(lambda *a: kp(x), lambda *a: H._radiance_w4a16_gemm_impl(x, kp.w_q, kp.w_s, kp.group_size))
rel = ((out.cpu().float() - ref_c).square().mean() / ref_c.square().mean()).sqrt().item()
report(out.shape == (8, 64) and rel < 0.15, "conv projection int4", f"rel rms {rel:.3f}")

# --- markers: a model source without the call site is left alone ---
NoSite = type("NoSite", (MLP,), {"__module__": "radiance_no_call_site"})
ns = NoSite(64)
report(not H._radiance_postload_mlp(ns) and not getattr(ns.gate_up_proj, "_radiance_silu", False),
       "no transform without the model-source marker")

print(f"{fails} failures")
sys.exit(1 if fails else 0)
