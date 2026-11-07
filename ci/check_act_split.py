"""fp8 hi/lo split in the producers (set_act_split) vs the exact bf16 producer output.
For add_rms_quant / silu_mul_quant / gdn_norm_quant (row-major and tiled where the launcher has it),
with set_act_split(64): M <= 64 must write residual rows M..2M-1 so that deq(hi) + deq(lo) matches
the bf16 values far better than deq(hi) alone (rel_split <= rel_hi / 6), and the hi rows must be
bit-identical to set_act_split(0); M = 65 must leave rows M.. untouched. With HAD=1 the same checks
run on top of the online rotation. Prints TOTAL OK/BAD.
"""
import os, sys
os.environ.setdefault("RADIANCE_FUSED_NQ_REG", "1")
import torch
sys.path.insert(0, "/w")
import radiance_mxfp4 as R
from tile_layout import tile

E = R._ext
dev = torch.device("cuda"); st = torch.cuda.current_stream().cuda_stream
HAD = os.environ.get("HAD", "0") == "1"
Hm = R.had_matrix(128, dev); sg = R.had_signs(128)
E.set_had(sg.data_ptr(), 1 if HAD else 0)
g = torch.Generator().manual_seed(0)


def deq(q, s):
    return q.view(torch.float8_e4m3fn).float() * s[:, None]


def rot(v):
    M, K = v.shape
    return (v.float().view(M, K // 128, 128) @ Hm).view(M, K) if HAD else v.float()


def case(kind, M, tiled):
    if kind == "add_rms":
        K = 5120
        y = torch.randn(M, K, generator=g).to(dev, torch.bfloat16); r = torch.randn(M, K, generator=g).to(dev, torch.bfloat16)
        w = (0.1 * torch.randn(K, generator=g)).to(dev, torch.bfloat16); ro = torch.empty_like(y)
        v = y.float() + r.float(); v = (v * torch.rsqrt(v.pow(2).mean(1, keepdim=True) + 1e-6) * (w.float() + 1)).to(torch.bfloat16)
        run = lambda q, s: E.launch_add_rms_quant(y.data_ptr(), r.data_ptr(), w.data_ptr(), q.data_ptr(), s.data_ptr(),
                                                  ro.data_ptr(), M, K, 1e-6, st, tiled)
    elif kind == "silu":
        K = 17408
        gu = torch.randn(M, 2 * K, generator=g).to(dev, torch.bfloat16)
        v = (torch.nn.functional.silu(gu[:, :K].float()).to(torch.bfloat16).float() * gu[:, K:].float()).to(torch.bfloat16)
        run = lambda q, s: E.launch_silu_mul_quant(gu.data_ptr(), q.data_ptr(), s.data_ptr(), M, K, st, tiled)
    else:
        K = 6144
        x = torch.randn(M, K, generator=g).to(dev, torch.bfloat16); z = torch.randn(M, K, generator=g).to(dev, torch.bfloat16)
        w = torch.randn(128, generator=g).to(dev, torch.bfloat16)
        xh = x.float().view(M, -1, 128)
        v = (((xh * torch.rsqrt(xh.pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()).view(M, K) * torch.nn.functional.silu(z.float())).to(torch.bfloat16)
        run = lambda q, s: E.launch_gdn_norm_quant(x.data_ptr(), z.data_ptr(), K, w.data_ptr(), q.data_ptr(), s.data_ptr(),
                                                   M, K, 1e-6, st, tiled)
    rows = (2 * M + 15) // 16 * 16
    out = {}
    for lim in (0, 64):
        E.set_act_split(lim)
        q = torch.full((rows, K), 0x7F, dtype=torch.uint8, device=dev); s = torch.full((rows,), -1.0, device=dev)
        run(q, s); torch.cuda.synchronize()
        if tiled:   # back to row-major for the comparison
            q = untile(q, rows, K)
        out[lim] = (q, s)
    E.set_act_split(0)
    ref = rot(v)
    qh, sh = out[64]
    hi = deq(qh[:M], sh[:M])
    same_hi = torch.equal(out[0][0][:M], qh[:M])
    rel_hi = float((hi - ref).norm() / ref.norm())
    if M <= 64:
        both = hi + deq(qh[M:2 * M], sh[M:2 * M])
        rel = float((both - ref).norm() / ref.norm())
        ok = same_hi and rel <= rel_hi / 6
    else:
        rel = float("nan")
        ok = same_hi and bool((sh[M:] == -1.0).all())
    print(f"SPLIT {kind} tiled={tiled} M={M} rel_hi={rel_hi:.4f} rel_split={rel:.5f} hi_same={same_hi} {'OK' if ok else 'WRONG'}", flush=True)
    return ok


def untile(q, rows, K):
    """inverse of the producers' fragment-tiled store: uint2 at ((r>>4)*kst + g>>1)*32 + (r&15) + 16*(g&1)."""
    kst = K >> 4
    flat = q.reshape(-1).view(torch.int64)            # uint2 = 8 bytes
    r = torch.arange(rows, device=q.device)[:, None]; gg = torch.arange(K // 8, device=q.device)[None, :]
    idx = ((r >> 4) * kst + (gg >> 1)) * 32 + (r & 15) + 16 * (gg & 1)
    return flat[idx.reshape(-1)].view(torch.uint8).reshape(rows, K)


ok = bad = 0
for kind in ("add_rms", "silu", "gdn"):
    for tiled in (0, 1):
        for M in (1, 8, 64, 65):
            r_ = case(kind, M, tiled); ok += r_; bad += not r_
print(f"TOTAL OK={ok} BAD={bad}")
