"""Kernel mutants for the GPU-window mutation check of radiance_inkling2_mx4.hip.

Usage: python3 mutate_mx4_hip.py <radiance_inkling2_mx4.hip> <outdir>   -> outdir/m_<name>.hip
Each mutation is an exact str.replace with a unique anchor (asserted); the mutants are compiled like the real
kernel (own dir per build, see test_gemv_mx4_hip.py) and run through tests/test_gemv_mx4_hip.py with
RADIANCE_MX4_BUILD=<dir>: every one must turn a test red. Only mutations that stay inside the allocated
buffers (no OOB access on the shared GPU) are listed: the row / K guards are never mutated.
"""
import sys
from pathlib import Path

MUTATIONS = {
    # decode chain (shared by the GEMV and the dequant kernel)
    "lut_h_even": ("__builtin_amdgcn_perm(0x40404040u, 0x3F3F3F00u, se)", "__builtin_amdgcn_perm(0x40404040u, 0x3F3F4000u, se)"),
    "lut_h_odd": ("__builtin_amdgcn_perm(0x40404040u, 0x3F3F3F00u, so)", "__builtin_amdgcn_perm(0x40404040u, 0x3F403F00u, so)"),
    "lut_h_hi": ("__builtin_amdgcn_perm(0x40404040u, 0x3F3F3F00u, se)", "__builtin_amdgcn_perm(0x40404000u, 0x3F3F3F00u, se)"),
    "lut_l_even": ("__builtin_amdgcn_perm(0xC0804000u, 0xC0800000u, se)", "__builtin_amdgcn_perm(0xC0804000u, 0xC0400000u, se)"),
    "lut_l_odd": ("__builtin_amdgcn_perm(0xC0804000u, 0xC0800000u, so)", "__builtin_amdgcn_perm(0xC0004000u, 0xC0800000u, so)"),
    "sign_even": ("((w << 4) & 0x80808080u)", "((w << 3) & 0x80808080u)"),
    "sign_odd": ("| (w & 0x80808080u);", "| ((w << 1) & 0x80808080u);"),
    "mask_even": ("const uint32_t se = w & 0x07070707u;", "const uint32_t se = w & 0x03030303u;"),
    "mask_odd": ("const uint32_t so = (w >> 4) & 0x07070707u;", "const uint32_t so = (w >> 4) & 0x03030303u;"),
    "il_even_swap": ("__builtin_amdgcn_perm(he, le, 0x05010400u), __builtin_amdgcn_perm(he, le, 0x07030602u)",
                     "__builtin_amdgcn_perm(he, le, 0x07030602u), __builtin_amdgcn_perm(he, le, 0x05010400u)"),
    "il_odd_swap": ("__builtin_amdgcn_perm(ho, lo, 0x05010400u), __builtin_amdgcn_perm(ho, lo, 0x07030602u)",
                    "__builtin_amdgcn_perm(ho, lo, 0x07030602u), __builtin_amdgcn_perm(ho, lo, 0x05010400u)"),
    "il_lohi": ("__builtin_amdgcn_perm(he, le, 0x05010400u),", "__builtin_amdgcn_perm(le, he, 0x05010400u),"),
    "scale_plus1": ("__uint_as_float((e & 0xFFu) << 23)", "__uint_as_float(((e & 0xFFu) + 1) << 23)"),
    # gemv_mx4
    "noscale": ("acc[t][e] = __builtin_fmaf(tmp[e], sc, acc[t][e]);", "acc[t][e] = __builtin_fmaf(tmp[e], 1.f, acc[t][e]);"),
    "scaleblock": ("e8m0_factor(b.sc[t] >> (8 * k))", "e8m0_factor(b.sc[t])"),
    "scaleswap": ("e8m0_factor(b.sc[t] >> (8 * k))", "e8m0_factor(b.sc[t] >> (8 * (1 - k)))"),
    "stepswap": ("__builtin_bit_cast(bf16x8, xb.x[k][0]),", "__builtin_bit_cast(bf16x8, xb.x[k][1]),"),
    "opsame": ("__builtin_bit_cast(bf16x8, w0), tmp);", "__builtin_bit_cast(bf16x8, xb.x[k][0]), tmp);"),
    "blockswap": ("mx4_decode(k ? b.w[t].z : b.w[t].x)", "mx4_decode(k ? b.w[t].x : b.w[t].z)"),
    "dwordswap": ("mx4_decode(k ? b.w[t].w : b.w[t].y)", "mx4_decode(k ? b.w[t].z : b.w[t].x)"),
    "xhalf": ("xr + g * MX_GROUP + k * 32 + j * 8)", "xr + g * MX_GROUP + k * 32 + (1 - j) * 8)"),
    "xrow": ("(int64_t)(xlive ? c : 0) * SX + h * 16;", "(int64_t)(xlive ? 0 : 0) * SX + h * 16;"),
    "xk": ("(int64_t)(xlive ? c : 0) * SX + h * 16;", "(int64_t)(xlive ? c : 0) * SX + 0;"),
    "xstride": ("(int64_t)(xlive ? c : 0) * SX + h * 16;", "(int64_t)(xlive ? c : 0) * ngroups * MX_GROUP + h * 16;"),
    "xlive_off1": ("const bool xlive = c < M;", "const bool xlive = c < M - (M > 1);"),
    "wgroup": ("(qp[t] + (int64_t)g * MX_GROUP_BYTES)", "(qp[t] + (int64_t)g * 256)"),
    "wlane": ("qp[t] = q + tile * ngroups * MX_GROUP_BYTES + lane * 16;", "qp[t] = q + tile * ngroups * MX_GROUP_BYTES + c * 16;"),
    "wtile": ("qp[t] = q + tile * ngroups * MX_GROUP_BYTES", "qp[t] = q + (int64_t)tile0 * ngroups * MX_GROUP_BYTES"),
    "sgroup": ("(sp[t] + (int64_t)g * MX_GROUP_SCALES)", "(sp[t] + (int64_t)g * 16)"),
    "srow": ("sp[t] = s + tile * ngroups * MX_GROUP_SCALES + c * 2;", "sp[t] = s + tile * ngroups * MX_GROUP_SCALES + 0;"),
    "split": ("  const int g1 = (int)((int64_t)ngroups * (wv + 1) / W);", "  const int g1 = (int)((int64_t)ngroups / W);"),
    "pipe_w": ("loadw(wb[(d + D - 1) % D], g + d + D - 1);", "loadw(wb[(d + D - 1) % D], g + d + D - 2);"),
    "pipe_x": ("loadx(xb[(d + 1) & 1], g + d + 1);", "loadx(xb[(d + 1) & 1], g + d);"),
    "prologue_w": ("if (g0 + d < g1) loadw(wb[d], g0 + d);", "if (g0 + d < g1) loadw(wb[d], g0);"),
    "prologue_x": ("if (g0 < g1) loadx(xb[0], g0);", "if (g0 < g1) loadx(xb[0], g0 + 1 < g1 ? g0 + 1 : g0);"),
    "nosum": ("for (int g = 0; g < W; g++) v += redq[(g * T + t) * 32 + l][e];", "for (int g = 0; g < 1; g++) v += redq[(g * T + t) * 32 + l][e];"),
    "rowhalf": ("const int m = 8 * (l >> 4) + e,", "const int m = 4 * (l >> 4) + e,"),
    "ncol": ("(tile0 + t) * 16 + (l & 15);", "(tile0 + t) * 16 + (l & 7);"),
    "ystride": ("else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(v);", "else ((uint16_t*)y)[(int64_t)m * N + n] = f2bf(v);"),
    "y32stride": ("if (out32) ((float*)y)[(int64_t)m * SY + n] = v;", "if (out32) ((float*)y)[(int64_t)m * N + n] = v;"),
    "blocks": ("dim3((ntiles + T - 1) / T)", "dim3((ntiles + T - 1) / T - (ntiles > T))"),
    "rne": ("(u + 0x7FFFu + ((u >> 16) & 1u)) >> 16", "(u + 0x7FFFu) >> 16"),
    "cfgpick_head": ("if (N >= (1 << 17)) return 11;", "if (N >= (1 << 17)) return 1;"),
    "cfgpick_small": ("if (N < 2048) return 2;", "if (N < 2048) return 1;"),
    "cfgpick_q": ("if (N >= 6000 && N < 7000) return 7;", "if (N >= 6000 && N < 7000) return 1;"),
    # dequant_mx4
    "dq_scale": ("const float sc = e8m0_factor(sc2 >> (8 * b));", "const float sc = e8m0_factor(sc2);"),
    "dq_halves": ("o[i] = (uint32_t)f2bf(lo) | ((uint32_t)f2bf(hi) << 16);", "o[i] = (uint32_t)f2bf(hi) | ((uint32_t)f2bf(lo) << 16);"),
    "dq_half": ("g * MX_GROUP + b * 32 + h * 16 + j * 8)", "g * MX_GROUP + b * 32 + (1 - h) * 16 + j * 8)"),
    "dq_dword": ("mx4_decode(b ? (j ? v.w : v.z) : (j ? v.y : v.x))", "mx4_decode(b ? (j ? v.z : v.w) : (j ? v.y : v.x))"),
    "dq_row": ("const int row = tile * MX_TILE + c;", "const int row = tile * MX_TILE + (c ^ 1 < 16 ? c ^ 1 : c);"),
}


def write_mutants(src: Path, outdir: Path):
    text = Path(src).read_text()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for name, (old, new) in MUTATIONS.items():
        assert text.count(old) == 1, f"mutation anchor not unique: {name} ({text.count(old)})"
        (Path(outdir) / f"m_{name}.hip").write_text(text.replace(old, new, 1))
    return list(MUTATIONS)


if __name__ == "__main__":
    print(" ".join(write_mutants(Path(sys.argv[1]), Path(sys.argv[2]))))
