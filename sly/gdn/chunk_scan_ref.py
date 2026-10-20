"""fp64 CPU reference for r4d_gdn_chunk_scan (formulas of the kernel header, scan:18-23).

Layouts follow the kernel: q,k [T,Hg,K]  v [T,H,V]  A [T,H,64]  g (chunk-local cumsum),beta [T,H]
h0,ht [N,H,V,K]  cu [N+1].  Pure torch, no GPU.  Used by the CPU tests and by the GPU window test
(sly/gdn/tests/gpu_chunk_scan_v2.py) as the oracle.
"""
import torch

BT = 64


def make_inputs(T_list, H=2, Hg=1, K=16, V=8, span=3.0, seed=0, h0_scale=0.5):
    """Random fp64 inputs for len(T_list) sequences. `span` = total gate decay per 64-chunk."""
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen, dtype=torch.float64)
    cu = [0]
    for t in T_list: cu.append(cu[-1] + t)
    T, N = cu[-1], len(T_list)
    k = r(T, Hg, K); k = k / k.norm(dim=-1, keepdim=True)
    q = r(T, Hg, K); q = q / q.norm(dim=-1, keepdim=True)
    v = r(T, H, V)
    beta = torch.rand(T, H, generator=gen, dtype=torch.float64)
    glog = -torch.rand(T, H, generator=gen, dtype=torch.float64) * (2.0 * span / BT)  # per-token log decay
    return dict(q=q, k=k, v=v, beta=beta, glog=glog, h0=r(N, H, V, K) * h0_scale,
                cu=cu, scale=K ** -0.5)


def chunk_cumsum(glog, cu):
    """g as the kernel receives it: cumsum restarted at every 64-token chunk of each sequence."""
    g = torch.empty_like(glog)
    for n in range(len(cu) - 1):
        for c0 in range(cu[n], cu[n + 1], BT):
            c1 = min(c0 + BT, cu[n + 1])
            g[c0:c1] = glog[c0:c1].cumsum(0)
    return g


def naive(inp):
    """Token-by-token gated delta rule: S <- a S; d = beta (v - S k); S <- S + d k^T; o = scale S q."""
    q, k, v, beta, glog, cu = (inp[x] for x in ("q", "k", "v", "beta", "glog", "cu"))
    H, Hg = v.shape[1], q.shape[1]
    o = torch.zeros_like(v); ht = torch.zeros_like(inp["h0"])
    for n in range(len(cu) - 1):
        for h in range(H):
            S = inp["h0"][n, h].clone()                       # [V,K]
            for t in range(cu[n], cu[n + 1]):
                S = S * torch.exp(glog[t, h])
                kt, qt = k[t, h // (H // Hg)], q[t, h // (H // Hg)]
                S = S + beta[t, h] * torch.outer(v[t, h] - S @ kt, kt)
                o[t, h] = inp["scale"] * (S @ qt)
            ht[n, h] = S
    return o, ht


def solve_A(inp, g):
    """A = (I + strict_lower(diag(beta) K K^T e^{g_i-g_j}))^-1 per chunk and head ([T,H,64])."""
    k, beta, cu = inp["k"], inp["beta"], inp["cu"]
    H, Hg = beta.shape[1], k.shape[1]
    A = torch.zeros(beta.shape[0], H, BT, dtype=torch.float64)
    for n in range(len(cu) - 1):
        for c0 in range(cu[n], cu[n + 1], BT):
            c1 = min(c0 + BT, cu[n + 1]); m = c1 - c0
            for h in range(H):
                kk = k[c0:c1, h // (H // Hg)]
                L = torch.tril(beta[c0:c1, h, None] * (kk @ kk.T) *
                               torch.exp(g[c0:c1, h, None] - g[c0:c1, h][None]), -1)
                A[c0:c1, h, :m] = torch.linalg.inv(torch.eye(m, dtype=torch.float64) + L)
    return A


def chunked(inp, split=False, clamp=80.0):
    """The kernel's math. split=False: exact e^{g_i-g_j}. split=True: the kernel's factorisation
    e^{g_i-c} * e^{c-g_j} about the chunk midpoint c with the growing half clamped at `clamp`."""
    q, k, v, beta, cu = (inp[x] for x in ("q", "k", "v", "beta", "cu"))
    g = chunk_cumsum(inp["glog"], cu)
    A = solve_A(inp, g)
    H, Hg = v.shape[1], q.shape[1]
    o = torch.zeros_like(v); ht = torch.zeros_like(inp["h0"])
    for n in range(len(cu) - 1):
        for h in range(H):
            S = inp["h0"][n, h].clone()
            hk = h // (H // Hg)
            for c0 in range(cu[n], cu[n + 1], BT):
                c1 = min(c0 + BT, cu[n + 1]); m = c1 - c0
                gg = g[c0:c1, h]; kk, qq, vv = k[c0:c1, hk], q[c0:c1, hk], v[c0:c1, h]
                c = 0.5 * (gg[0] + gg[-1])
                ge = torch.exp(gg)
                W = beta[c0:c1, h, None] * (vv - ge[:, None] * (kk @ S.T))
                if split:
                    gv = torch.exp(torch.clamp(c - gg, max=clamp)); rs = inp["scale"] * torch.exp(torch.clamp(gg - c, max=clamp))
                    Vp = gv[:, None] * (A[c0:c1, h, :m] @ W)
                    sA = rs[:, None] * torch.tril(qq @ kk.T) * 1.0
                    # sA_ij = rs_i * P_ij, applied to V' = gv_j * (AW)_j  (so e^{g_i-g_j} = rs_i gv_j)
                    o[c0:c1, h] = inp["scale"] * ge[:, None] * (qq @ S.T) + sA @ Vp
                    lam = torch.exp(gg[-1] - c)
                    S = lam * (torch.exp(c) * S + Vp.T @ kk)
                else:
                    u = A[c0:c1, h, :m] @ W
                    D = torch.exp(gg[:, None] - gg[None])
                    o[c0:c1, h] = inp["scale"] * (ge[:, None] * (qq @ S.T)) + inp["scale"] * (torch.tril(qq @ kk.T) * D) @ u
                    S = torch.exp(gg[-1]) * S + (torch.exp(gg[-1] - gg)[:, None] * u).T @ kk
            ht[n, h] = S
    return o, ht


def rel_rmse(a, b):
    return float(((a - b) ** 2).sum().sqrt() / (b ** 2).sum().sqrt())
