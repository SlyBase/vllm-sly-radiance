"""Online Hadamard-128 in the fp8 producers (set_had) vs a torch reference, every launcher variant.
For add_rms_quant / silu_mul_quant / gdn_norm_quant (non-tiled, REG as in prod) at M in MS:
  ref = per-token e4m3 quant of (bf16-rounded producer output) @ R, R = blockdiag(H128 diag(s))/sqrt(128)
Checks: dequantized q * scale matches the reference within e4m3 rounding (rel <= 0.03, same as the
plain producer vs its own reference), set_had(.., 0) restores the unrotated digest bit-exactly, and
q @ R^T recovers the unrotated activation (rel <= 0.03). Prints TOTAL OK/BAD.
"""
import hashlib, os, sys
os.environ.setdefault("RADIANCE_FUSED_NQ_REG", "1")
import torch
sys.path.insert(0, "/w")
import radiance_mxfp4 as R

E = R._ext
dev = torch.device("cuda"); st = torch.cuda.current_stream().cuda_stream
MS = [int(x) for x in os.environ.get("MS", "1,8,64,600,2048").split(",")]
Hm = R.had_matrix(128, dev)
sg = R.had_signs(128)
g = torch.Generator(device="cpu").manual_seed(0)


def h(t):
    return hashlib.sha256(t.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def rot(x):
    M, K = x.shape
    return (x.float().view(M, K // 128, 128) @ Hm).view(M, K)


def deq(q, s):
    return q.view(torch.float8_e4m3fn).float() * s[:, None]


def run(kind, M):
    if kind == "add_rms":
        K = 5120
        y = torch.randn(M, K, generator=g).to(dev, torch.bfloat16); r = torch.randn(M, K, generator=g).to(dev, torch.bfloat16)
        w = (0.1 * torch.randn(K, generator=g)).to(dev, torch.bfloat16)
        q = torch.empty(M, K, dtype=torch.uint8, device=dev); s = torch.empty(M, device=dev); ro = torch.empty_like(y)
        go = lambda: E.launch_add_rms_quant(y.data_ptr(), r.data_ptr(), w.data_ptr(), q.data_ptr(), s.data_ptr(),
                                            ro.data_ptr(), M, K, 1e-6, st, 0)
        v = (y.float() + r.float()); v = (v * torch.rsqrt(v.pow(2).mean(1, keepdim=True) + 1e-6) * (w.float() + 1)).to(torch.bfloat16)
    elif kind == "silu":
        K = 17408
        gu = torch.randn(M, 2 * K, generator=g).to(dev, torch.bfloat16)
        q = torch.empty(M, K, dtype=torch.uint8, device=dev); s = torch.empty(M, device=dev)
        go = lambda: E.launch_silu_mul_quant(gu.data_ptr(), q.data_ptr(), s.data_ptr(), M, K, st, 0)
        a = gu[:, :K].float()
        v = (torch.nn.functional.silu(a).to(torch.bfloat16).float() * gu[:, K:].float()).to(torch.bfloat16)
    else:
        K = 6144
        x = torch.randn(M, K, generator=g).to(dev, torch.bfloat16); z = torch.randn(M, K, generator=g).to(dev, torch.bfloat16)
        w = torch.randn(128, generator=g).to(dev, torch.bfloat16)
        q = torch.empty(M, K, dtype=torch.uint8, device=dev); s = torch.empty(M, device=dev)
        go = lambda: E.launch_gdn_norm_quant(x.data_ptr(), z.data_ptr(), K, w.data_ptr(), q.data_ptr(), s.data_ptr(),
                                             M, K, 1e-6, st, 0)
        xh = x.float().view(M, -1, 128)
        v = ((xh * torch.rsqrt(xh.pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()).view(M, K)
        v = (v * torch.nn.functional.silu(z.float())).to(torch.bfloat16)
    E.set_had(sg.data_ptr(), 0); go(); torch.cuda.synchronize(); d0 = h(q)
    E.set_had(sg.data_ptr(), 1); go(); torch.cuda.synchronize()
    got = deq(q, s); ref = rot(v)
    rel = float((got - ref).norm() / ref.norm())
    back = (got.view(M, -1, 128) @ Hm.T).view(M, -1)
    relb = float((back - v.float()).norm() / v.float().norm())
    E.set_had(sg.data_ptr(), 0); go(); torch.cuda.synchronize(); d1 = h(q)
    ok = rel <= 0.03 and relb <= 0.03 and d0 == d1
    print(f"HAD {kind} M={M} rel={rel:.4f} back={relb:.4f} off_digest {'SAME' if d0 == d1 else 'CHANGED'} {'OK' if ok else 'WRONG'}", flush=True)
    return ok


ok = bad = 0
for kind in ("add_rms", "silu", "gdn"):
    for M in MS:
        r_ = run(kind, M); ok += r_; bad += not r_
print(f"TOTAL OK={ok} BAD={bad}")
