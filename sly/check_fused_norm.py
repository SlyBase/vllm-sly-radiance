#!/usr/bin/env python3
"""Numerics + smoke test of the RADIANCE_FUSED_NORM_QUANT fusions against the unfused chain.

Runs in a throw-away container from the 0.1.6 image with the GPU exclusive and the HF cache mounted,
on the real weights of amd/Qwen3.8-27B-Quark-AWQ-MXFP4 layer 4 (GDN) and layer 3 (attention):

  residual/y -> input_layernorm(4) --(q,s)--> in_proj_qkvz, in_proj_ba
  core_attn_out, z(qkvz split view) -> linear_attn.norm --(q,s)--> out_proj
  out_proj, residual -> post_attention_layernorm(4) --(q,s)--> gate_up_proj
  gate_up -> silu*up --(q,s)--> down_proj
  down, residual -> input_layernorm(3) --(q,s)--> qkv_proj

Reference per site: vLLM's own module forward_native (eager) and a torch.compile'd native chain
(what Inductor generates in production, custom_ops []), each followed by
_custom_ops.scaled_fp8_quant(per-token); the consumer runs on the SAME W4A8 GEMM
(radiance::mxfp4_linear_pq) so every output difference is the fusion's. Also the unfused production
op radiance::mxfp4_linear on the reference activation, _exact_ref (fp32 ground truth, M <= 128),
torch.compile(fullgraph) of the fused chain, a CUDA-graph capture/replay, and per-call timing.

  RADIANCE_MXFP4=1 RADIANCE_MXFP4_W4A8=1 RADIANCE_MXFP4_W4A8_MIN_M=0 RADIANCE_MXFP4_DECODE_MAX_M=128 \
  RADIANCE_FUSED_NORM_QUANT=1 python3 check_fused_norm.py [--ms 1,8,16,64,128,512,2048]
"""
import argparse, glob, inspect, os, sys, time

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--snapshot", default=None)
ap.add_argument("--ms", default="1,8,16,64,128,512,2048")
ap.add_argument("--no-timing", action="store_true")
args = ap.parse_args()
MS = [int(m) for m in args.ms.split(",")]

import radiance_mxfp4 as R          # noqa: E402  (loads the HIP extension, registers the ops)
import radiance_fused_norm as FN    # noqa: E402
from vllm import _custom_ops as ops  # noqa: E402

assert FN.ENABLED and FN._ext is not None, "set RADIANCE_FUSED_NORM_QUANT=1 and the W4A8 envs"
dev = torch.device("cuda")
BF = torch.bfloat16
EPS = 1e-6
torch.manual_seed(0)

# ---------------------------------------------------------------- weights
snap = args.snapshot or sorted(glob.glob(
    "/root/.cache/huggingface/hub/models--amd--Qwen3.8-27B-Quark-AWQ-MXFP4/snapshots/*"))[-1]
P = "model.language_model.layers."
f = safe_open(f"{snap}/model.safetensors", framework="pt", device="cpu")


def T(k):
    return f.get_tensor(P + k).to(dev)


class Lin:
    def __init__(self, *names):
        w = torch.cat([T(n + ".weight") for n in names], 0).contiguous()
        ws = torch.cat([T(n + ".weight_scale") for n in names], 0)
        self.weight = w
        self.weight_scale = ws.T.contiguous()          # [K/32, N], as process_weights leaves it
        self.wref = R.make_row_ref(self.weight_scale)  # folded
        self.N, self.K = w.shape[0], w.shape[1] * 2
        self.names = names

    def pq(self, q, s):
        return torch.ops.radiance.mxfp4_linear_pq(q, s, self.weight, self.weight_scale, self.wref)

    def unfused(self, x):
        return torch.ops.radiance.mxfp4_linear(x, self.weight, self.weight_scale, self.wref)

    def exact(self, q, s):
        return R._exact_ref(q, s.view(-1), self.weight, self.weight_scale, self.N, self.K)


L4 = {
    "in_w": T("4.input_layernorm.weight"), "post_w": T("4.post_attention_layernorm.weight"),
    "norm_w": T("4.linear_attn.norm.weight"),
    "qkvz": Lin("4.linear_attn.in_proj_qkv", "4.linear_attn.in_proj_z"),
    "ba": Lin("4.linear_attn.in_proj_b", "4.linear_attn.in_proj_a"),
    "out": Lin("4.linear_attn.out_proj"),
    "gate_up": Lin("4.mlp.gate_proj", "4.mlp.up_proj"), "down": Lin("4.mlp.down_proj"),
}
L3 = {"in_w": T("3.input_layernorm.weight"),
      "qkv": Lin("3.self_attn.q_proj", "3.self_attn.k_proj", "3.self_attn.v_proj")}
for k, v in list(L4.items()) + list(L3.items()):
    if isinstance(v, Lin):
        print(f"  {k:8s} N={v.N:6d} K={v.K:5d} ws={tuple(v.weight_scale.shape)}")

# decode-kernel scratch, exactly as process_weights_after_loading allocates it
if R.DECODE_MAX_M > 0 and not R._decode_scratch_ready[0]:
    R._decode_scratch_ready[0] = True
    R._decode_scratch[0] = torch.empty(4 * max(64, R.DECODE_MAX_M) * 36864, dtype=torch.float32, device=dev)
    R._decode_scratch[1] = torch.zeros(36864 // 128 + 8, dtype=torch.int32, device=dev)
    R._ext.set_decode_scratch(R._decode_scratch[0].data_ptr(), R._decode_scratch[0].numel() * 4,
                              R._decode_scratch[1].data_ptr())

# ---------------------------------------------------------------- references
from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.activation import SiluAndMul  # noqa: E402
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNormGated  # noqa: E402

with set_current_vllm_config(VllmConfig()):
    g_in4, g_post4, g_in3 = GemmaRMSNorm(5120, EPS), GemmaRMSNorm(5120, EPS), GemmaRMSNorm(5120, EPS)
    gnorm = RMSNormGated(128, eps=EPS, group_size=None, norm_before_gate=True, activation="silu")
    silu_mul = SiluAndMul()
for mod, w in ((g_in4, L4["in_w"]), (g_post4, L4["post_w"]), (g_in3, L3["in_w"]), (gnorm, L4["norm_w"])):
    mod.weight.data = w.clone()
    mod.to(dev)
print("GemmaRMSNorm.forward_native:", inspect.getsource(GemmaRMSNorm.forward_native).strip().splitlines()[-1])


def quant(x):
    q, s = ops.scaled_fp8_quant(x, None, use_per_token_if_dynamic=True)
    return q, s.view(-1).float()


# manual native chains (the arithmetic Inductor lowers in production), compiled once, dynamic M
def m_add_rms(y, res, w):
    x = y.float() + res.float()
    r = x.to(BF)
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * (w.float() + 1.0)
    return out.to(BF), r


def m_silu(gu):
    d = gu.shape[-1] // 2
    return torch.nn.functional.silu(gu[..., :d]) * gu[..., d:]


def m_gdn(x, z, w):
    M = x.shape[0]
    x = x.reshape(-1, 128).float()
    z = z.reshape(-1, 128).float()
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w.float()
    return (out * torch.nn.functional.silu(z)).to(BF).reshape(M, -1)


c_add_rms = torch.compile(m_add_rms, dynamic=True)
c_silu = torch.compile(m_silu, dynamic=True)
c_gdn = torch.compile(m_gdn, dynamic=True)


def e_add_rms(mod, y, res):
    h, r = mod.forward_native(y.clone(), res.clone())
    return h, r


def e_gdn(x, z):
    M = x.shape[0]
    return gnorm.forward_native(x.reshape(-1, 128), z.reshape(-1, 128)).reshape(M, -1)


# ---------------------------------------------------------------- metrics
worst = {}


def track(key, val):
    worst[key] = max(worst.get(key, 0.0), float(val))


def cmp_q(tag, M, q, s, qr, sr):
    dq = (q.float() - qr.float()).abs()
    ds = (s - sr).abs()
    mism = (q.view(torch.uint8) != qr.view(torch.uint8)).float().mean().item()
    srel = (ds / sr.abs().clamp_min(1e-30)).max().item()
    track(f"{tag} code_maxdiff", dq.max().item()); track(f"{tag} scale_rel", srel)
    return f"code mism {mism:.2e} max|d| {dq.max().item():.3g}  scale max|d| {ds.max().item():.3g} rel {srel:.2e}"


def cmp_o(tag, M, o, oref, lin=None, q=None, s=None):
    d = (o.float() - oref.float())
    rel = (d.norm() / oref.float().norm().clamp_min(1e-12)).item()
    track(f"{tag} out_maxdiff", d.abs().max().item()); track(f"{tag} out_rel", rel)
    msg = f"out max|d| {d.abs().max().item():.3g} rel {rel:.2e}"
    if lin is not None and M <= 128:
        ex = lin.exact(q, s)
        erel = ((o.float() - ex).norm() / ex.norm().clamp_min(1e-12)).item()
        track(f"{tag} exact_rel", erel)
        msg += f"  exact_rel {erel:.4f} {'**WRONG**' if erel > 0.02 else 'ok'}"
    return msg


def site(tag, M, fused_q, fused_s, h_eager, h_comp, lin):
    """fused (q,s) vs eager/compiled reference activation, then the consumer GEMM."""
    qe, se = quant(h_eager)
    qc, sc = quant(h_comp)
    o_f = lin.pq(fused_q, fused_s)
    o_e = lin.pq(qe, se)
    o_c = lin.pq(qc, sc)
    o_u = lin.unfused(h_comp)
    print(f"  {tag:14s} M={M:4d} vs eager:    {cmp_q(tag + '/eager', M, fused_q, fused_s, qe, se)}")
    print(f"  {tag:14s} M={M:4d} vs compiled: {cmp_q(tag + '/comp', M, fused_q, fused_s, qc, sc)}")
    print(f"  {'':14s}        {lin.names[0].split('.')[-1]:>12s} fused vs compiled: "
          f"{cmp_o(tag + '/comp', M, o_f, o_c, lin, fused_q, fused_s)}")
    print(f"  {'':14s}        {'':>12s} fused vs eager: {cmp_o(tag + '/eager', M, o_f, o_e)}"
          f"  | unfused prod op vs pq(compiled): {cmp_o(tag + '/unfused', M, o_u, o_c)}")
    act_d = (h_eager.float() - h_comp.float()).abs().max().item()
    track("eager_vs_compiled_act", act_d)
    return o_c


with torch.no_grad():
    for M in MS:
        print(f"--- M={M}")
        g = torch.Generator(device=dev).manual_seed(M)
        out_ch = torch.randint(0, 5120, (8,), generator=g, device=dev)
        res = torch.randn(M, 5120, device=dev, generator=g) * 0.4
        res[:, out_ch] *= 60.0                                   # residual outlier channels
        res = res.to(BF)
        y = (torch.randn(M, 5120, device=dev, generator=g) * 0.3).to(BF)

        # 1) input_layernorm(4) -> in_proj_qkvz + in_proj_ba
        q, s, r = torch.ops.radiance.add_rms_quant(y, res, L4["in_w"], EPS)
        he, re_ = e_add_rms(g_in4, y, res)
        hc, rc = c_add_rms(y, res, L4["in_w"])
        rd = max((r.float() - re_.float()).abs().max().item(), (r.float() - rc.float()).abs().max().item())
        track("add_rms res_out_maxdiff", rd)
        print(f"  add_rms(in4)   M={M:4d} res_out max|d| {rd:.3g}")
        qkvz = site("add_rms(in4)", M, q, s, he, hc, L4["qkvz"])
        site("add_rms(in4)", M, q, s, he, hc, L4["ba"])
        res = rc

        # 2) GDN gated norm -> out_proj; z is the split VIEW of the projection (row stride 16384)
        z = qkvz.split([10240, 6144], dim=-1)[1].reshape(M, 48, 128)
        core = (torch.randn(M, 48, 128, device=dev, generator=g) * 0.05).to(BF)
        q, s = torch.ops.radiance.gdn_norm_quant(core.reshape(M, -1), z.reshape(M, -1), L4["norm_w"], EPS)
        q2, s2 = torch.ops.radiance.gdn_norm_quant(core.reshape(M, -1), z.reshape(M, -1).contiguous(),
                                                   L4["norm_w"], EPS)
        same = torch.equal(q.view(torch.uint8), q2.view(torch.uint8)) and torch.equal(s, s2)
        track("gdn view_vs_contig_mismatch", 0.0 if same else 1.0)
        print(f"  gdn_norm       M={M:4d} z stride {tuple(z.reshape(M, -1).stride())} view==contiguous: {same}")
        he = e_gdn(core, z)
        hc = c_gdn(core, z, L4["norm_w"])
        y = site("gdn_norm", M, q, s, he, hc, L4["out"])

        # 3) post_attention_layernorm(4) -> gate_up
        q, s, r = torch.ops.radiance.add_rms_quant(y, res, L4["post_w"], EPS)
        he, re_ = e_add_rms(g_post4, y, res)
        hc, rc = c_add_rms(y, res, L4["post_w"])
        rd = max((r.float() - re_.float()).abs().max().item(), (r.float() - rc.float()).abs().max().item())
        track("add_rms res_out_maxdiff", rd)
        gu = site("add_rms(post4)", M, q, s, he, hc, L4["gate_up"])
        res = rc

        # 4) silu * up -> down
        q, s = torch.ops.radiance.silu_mul_quant(gu)
        he = silu_mul.forward_native(gu)
        hc = c_silu(gu)
        y = site("silu_mul", M, q, s, he, hc, L4["down"])

        # 5) input_layernorm(3) -> qkv_proj (attention consumer)
        q, s, r = torch.ops.radiance.add_rms_quant(y, res, L3["in_w"], EPS)
        he, re_ = e_add_rms(g_in3, y, res)
        hc, rc = c_add_rms(y, res, L3["in_w"])
        site("add_rms(in3)", M, q, s, he, hc, L3["qkv"])
        torch.cuda.synchronize()

    # ------------------------------------------------------------ compile + CUDA graph
    def fused_chain(y, res, core, qkvz_in):
        q, s, r = torch.ops.radiance.add_rms_quant(y, res, L4["post_w"], EPS)
        gu = torch.ops.radiance.mxfp4_linear_pq(q, s, L4["gate_up"].weight, L4["gate_up"].weight_scale,
                                                L4["gate_up"].wref)
        q2, s2 = torch.ops.radiance.silu_mul_quant(gu)
        d = torch.ops.radiance.mxfp4_linear_pq(q2, s2, L4["down"].weight, L4["down"].weight_scale,
                                               L4["down"].wref)
        M = y.shape[0]
        z = qkvz_in.split([10240, 6144], dim=-1)[1].reshape(M, 48, 128)
        q3, s3 = torch.ops.radiance.gdn_norm_quant(core.reshape(M, -1), z.reshape(M, -1), L4["norm_w"], EPS)
        o = torch.ops.radiance.mxfp4_linear_pq(q3, s3, L4["out"].weight, L4["out"].weight_scale, L4["out"].wref)
        return d, o, r

    cf = torch.compile(fused_chain, fullgraph=True, dynamic=True)
    for M in (1, 16, 64):
        ins = [(torch.randn(M, 5120, device=dev) * 0.3).to(BF), (torch.randn(M, 5120, device=dev) * 5).to(BF),
               (torch.randn(M, 48, 128, device=dev) * 0.05).to(BF), (torch.randn(M, 16384, device=dev)).to(BF)]
        e = fused_chain(*ins)
        c = cf(*ins)
        dd = max((a.float() - b.float()).abs().max().item() for a, b in zip(e, c))
        track("compile_vs_eager", dd)
        print(f"  torch.compile(fullgraph) fused chain M={M}: max|d| vs eager {dd:.3g}")

    M = 16
    st = [(torch.randn(M, 5120, device=dev) * 0.3).to(BF), (torch.randn(M, 5120, device=dev) * 5).to(BF),
          (torch.randn(M, 48, 128, device=dev) * 0.05).to(BF), (torch.randn(M, 16384, device=dev)).to(BF)]
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(3):
            cf(*st)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gout = cf(*st)
    for it in range(3):
        for t, sc in zip(st, (0.3, 5.0, 0.05, 1.0)):
            t.copy_((torch.randn(t.shape, device=dev) * sc).to(BF))
        graph.replay()
        e = fused_chain(*st)
        dd = max((a.float() - b.float()).abs().max().item() for a, b in zip(gout, e))
        track("cudagraph_vs_eager", dd)
        print(f"  CUDA graph replay {it}: max|d| vs eager {dd:.3g}")

    # ------------------------------------------------------------ timing
    if not args.no_timing:
        def bench(fn, n=300):
            for _ in range(20):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / n * 1e6

        for M in (1, 8, 16, 64):
            y = (torch.randn(M, 5120, device=dev) * 0.3).to(BF)
            res = (torch.randn(M, 5120, device=dev) * 5).to(BF)
            gu = torch.randn(M, 34816, device=dev).to(BF)
            core = (torch.randn(M, 6144, device=dev) * 0.05).to(BF)
            z = torch.randn(M, 16384, device=dev).to(BF)[:, 10240:]
            t = [bench(lambda: torch.ops.radiance.add_rms_quant(y, res, L4["in_w"], EPS)),
                 bench(lambda: quant(c_add_rms(y, res, L4["in_w"])[0])),
                 bench(lambda: torch.ops.radiance.silu_mul_quant(gu)),
                 bench(lambda: quant(c_silu(gu))),
                 bench(lambda: torch.ops.radiance.gdn_norm_quant(core, z, L4["norm_w"], EPS)),
                 bench(lambda: quant(c_gdn(core, z, L4["norm_w"])))]
            print(f"  timing M={M:3d} (us/call, host+device, compiled ref not cudagraphed): "
                  f"add_rms {t[0]:.0f} vs {t[1]:.0f} | silu_mul {t[2]:.0f} vs {t[3]:.0f} | gdn {t[4]:.0f} vs {t[5]:.0f}")

print("--- worst over all M")
for k in sorted(worst):
    print(f"  {k:36s} {worst[k]:.4g}")
bad = [k for k in worst if k.endswith("exact_rel") and worst[k] > 0.02] + \
      [k for k in ("compile_vs_eager", "cudagraph_vs_eager", "gdn view_vs_contig_mismatch") if worst.get(k, 0) > 0]
print("CHECK_FUSED_NORM", "FAIL " + ",".join(bad) if bad else "PASS")
