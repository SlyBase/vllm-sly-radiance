"""Kernel mutants for the GPU-window mutation check of radiance_inkling2_gemv.hip.

Usage: python3 mutate_gemv_hip.py <radiance_inkling2_gemv.hip> <outdir>   -> outdir/m_<name>.hip
Each mutation is an exact str.replace with a unique anchor (asserted); the mutants are compiled
like the real kernel and run through tests/test_gemv_hip.py with RADIANCE_GEMV_SO=<mutant.so>:
every one must turn a test red. Only mutations that stay inside the allocated buffers (no OOB
access on the shared GPU) are listed.
"""
import sys
from pathlib import Path

MUTATIONS = {
    # gemv_q8 (tests/test_gemv_hip.py). Same rule as below: only mutations that stay inside the buffers.
    "q_noscale": ("acc[t][e] = __builtin_fmaf(tmp[e], sc, acc[t][e]);", "acc[t][e] = __builtin_fmaf(tmp[e], 1.f, acc[t][e]);"),
    "q_scaleblock": ("const float sc = k ? b.sc[t].y : b.sc[t].x;", "const float sc = b.sc[t].x;"),
    "q_scaleswap": ("const float sc = k ? b.sc[t].y : b.sc[t].x;", "const float sc = k ? b.sc[t].x : b.sc[t].y;"),
    "q_negbias": ("    if (bias) v += __uint_as_float((uint32_t)bias[n] << 16);", "    if (bias) v -= __uint_as_float((uint32_t)bias[n] << 16);"),
    "q_unsigned": ("const uint32_t u = w ^ 0x80808080u;", "const uint32_t u = w;"),
    "q_magic": ("(u & 0xFFu)) - 8388736.f;", "(u & 0xFFu)) - 8388608.f;"),
    "q_byte1": ("((u >> 8) & 0xFFu)", "((u >> 16) & 0xFFu)"),
    "q_packswap": ("lo = (__float_as_uint(f0) >> 16) | (__float_as_uint(f1) & 0xFFFF0000u);", "lo = (__float_as_uint(f1) >> 16) | (__float_as_uint(f0) & 0xFFFF0000u);"),
    "q_stepswap": ("__builtin_bit_cast(bf16x8, b.x[k][0]),", "__builtin_bit_cast(bf16x8, b.x[k][1]),"),
    "q_opsame": ("__builtin_bit_cast(bf16x8, w0), tmp);", "__builtin_bit_cast(bf16x8, b.x[k][0]), tmp);"),
    "q_xhalf": ("xr + g * Q_GROUP + k * 32 + j * 8)", "xr + g * Q_GROUP + k * 32 + (1 - j) * 8)"),
    "q_xrow": ("(int64_t)(xlive ? c : 0) * SX + h * 16;", "(int64_t)(xlive ? 0 : 0) * SX + h * 16;"),
    "q_xk": ("(int64_t)(xlive ? c : 0) * SX + h * 16;", "(int64_t)(xlive ? c : 0) * SX + 0;"),
    "q_xstride": ("(int64_t)(xlive ? c : 0) * SX + h * 16;", "(int64_t)(xlive ? c : 0) * ngroups * Q_GROUP + h * 16;"),
    "q_xlive_off1": ("const bool xlive = c < M;", "const bool xlive = c < M - (M > 1);"),
    "q_wblock": ("(int64_t)g * Q_GROUP_BYTES + k * 512));", "(int64_t)g * Q_GROUP_BYTES + 0));"),
    "q_wgroup": ("(int64_t)g * Q_GROUP_BYTES + k * 512));", "(int64_t)g * 512 + k * 512));"),
    "q_wlane": ("qp[t] = q + tile * ngroups * Q_GROUP_BYTES + lane * 16;", "qp[t] = q + tile * ngroups * Q_GROUP_BYTES + c * 16;"),
    "q_wtile": ("qp[t] = q + tile * ngroups * Q_GROUP_BYTES + lane * 16;", "qp[t] = q + (int64_t)(tile0 + 0 * t) * ngroups * Q_GROUP_BYTES + lane * 16;"),
    "q_sgroup": ("(int64_t)g * Q_GROUP_SCALES));", "(int64_t)g * 16));"),
    "q_srow": ("sp[t] = s + tile * ngroups * Q_GROUP_SCALES + c * 2;", "sp[t] = s + tile * ngroups * Q_GROUP_SCALES + 0;"),
    "q_split": ("  const int g1 = (int)((int64_t)ngroups * (wv + 1) / W);", "  const int g1 = (int)((int64_t)ngroups / W);"),
    "q_pipe": ("load(buf[(d + D - 1) % D], g + d + D - 1);", "load(buf[(d + D - 1) % D], g + d + D - 2);"),
    "q_prologue": ("load(buf[d], g0 + d);", "load(buf[d], g0);"),
    "q_nosum": ("for (int g = 0; g < W; g++) v += redq[(g * T + t) * 32 + l][e];", "for (int g = 0; g < 1; g++) v += redq[(g * T + t) * 32 + l][e];"),
    "q_rowhalf": ("const int m = 8 * (l >> 4) + e,", "const int m = 4 * (l >> 4) + e,"),
    "q_ncol": ("(tile0 + t) * 16 + (l & 15);", "(tile0 + t) * 16 + (l & 7);"),
    "q_ystride": ("else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(v);\n  }\n}\n\nstruct ArgsQ", "else ((uint16_t*)y)[(int64_t)m * N + n] = f2bf(v);\n  }\n}\n\nstruct ArgsQ"),
    "q_y32stride": ("if (out32) ((float*)y)[(int64_t)m * SY + n] = v;\n    else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(v);\n  }\n}\n\nstruct ArgsQ", "if (out32) ((float*)y)[(int64_t)m * N + n] = v;\n    else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(v);\n  }\n}\n\nstruct ArgsQ"),
    "q_blocks": ("dim3((ntiles + T - 1) / T)", "dim3((ntiles + T - 1) / T - (ntiles > T))"),
    "q_cfgpick": ("if (N >= (1 << 17)) return 6;", "if (N >= (1 << 17)) return 1;"),
    # gemv_bf16 (tests/test_gemv_bf16_hip.py). Only mutations that stay inside the allocated buffers:
    # the OOB guards (row clamp, K clamps and masks, columns >= M, stores past N) are never mutated on the
    # shared GPU.
    "bf_opsame": ("__builtin_bit_cast(bf16x8, a),", "__builtin_bit_cast(bf16x8, xv),"),
    "bf_swz_write": ("tile[(t * 16 + row) * 8 + (lc ^ (row & 7))] =", "tile[(t * 16 + row) * 8 + lc] ="),
    "bf_swz_read": ("tile[(t * 16 + r) * 8 + ((2 * j + h) ^ (r & 7))]", "tile[(t * 16 + r) * 8 + (2 * j + h)]"),
    "bf_chunk_h": ("((2 * j + h) ^ (r & 7))", "((2 * j) ^ (r & 7))"),
    "bf_step_j": ("((2 * j + h) ^ (r & 7))", "((2 * (j ^ 1) + h) ^ (r & 7))"),
    "bf_xrow": ("const uint16_t* xr0 = x + (int64_t)(xlive ? r : 0) * SX;", "const uint16_t* xr0 = x + (int64_t)(xlive ? 0 : 0) * SX;"),
    "bf_xlive_off1": ("const bool xlive = r < M;", "const bool xlive = r < M - (M > 1);"),
    "bf_wstride": ("N - 1) * SW;", "N - 1) * K;"),
    "bf_wrow": ("min(nbase + 16 * t + 4 * q + lr, N - 1)", "min(nbase + 16 * t + lr, N - 1)"),
    "bf_wtile": ("min(nbase + 16 * t + 4 * q + lr, N - 1)", "min(nbase + 4 * q + lr, N - 1)"),
    "bf_split": ("const int g0 = (int)((int64_t)ngroups * wv / W), g1 = (int)((int64_t)ngroups * (wv + 1) / W);", "const int g0 = 0, g1 = (int)((int64_t)ngroups / W);"),
    "bf_gstep": ("const uint16_t* p = wl[t][q] + g * BF_GROUP + lc * 8;", "const uint16_t* p = wl[t][q] + g * 32 + lc * 8;"),
    "bf_pipe": ("      if (g + 2 < g1) load(wa, xa, g + 2);\n", "      if (g + 2 < g1) load(wa, xa, g + 1);\n"),
    "bf_nosum": ("for (int g = 0; g < W; g++) v += redb[g * T * 256 + id];", "for (int g = 0; g < 1; g++) v += redb[g * T * 256 + id];"),
    "bf_rowhalf": ("n = nbase + 16 * t + (l >> 4) * 8 + e;", "n = nbase + 16 * t + (l >> 4) * 4 + e;"),
    "bf_negbias": ("v += bias ? __uint_as_float", "v -= bias ? __uint_as_float"),
    "bf_bias0": ("v += bias ? __uint_as_float((uint32_t)bias[n] << 16)", "v += bias ? __uint_as_float((uint32_t)bias[0] << 16)"),
    "bf_ystride": ("else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(v);\n  }\n}\n\nstruct ArgsB", "else ((uint16_t*)y)[(int64_t)m * N + n] = f2bf(v);\n  }\n}\n\nstruct ArgsB"),
    "bf_y32stride": ("if (out32) ((float*)y)[(int64_t)m * SY + n] = v;\n    else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(v);\n  }\n}\n\nstruct ArgsB", "if (out32) ((float*)y)[(int64_t)m * N + n] = v;\n    else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(v);\n  }\n}\n\nstruct ArgsB"),
    "bf_rne": ("return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);", "return (uint16_t)((u + 0x7FFFu) >> 16);"),
    "bf_blocks": ("const int blocks = (a.N + 16 * T - 1) / (16 * T);", "const int blocks = (a.N + 16 * T - 1) / (16 * T) - (a.N > 16 * T);"),
}


def write_mutants(src: Path, outdir: Path):
    text = Path(src).read_text()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for name, (old, new) in MUTATIONS.items():
        assert text.count(old) == 1, f"mutation anchor not unique: {name}"
        (Path(outdir) / f"m_{name}.hip").write_text(text.replace(old, new, 1))
    return list(MUTATIONS)


if __name__ == "__main__":
    print(" ".join(write_mutants(Path(sys.argv[1]), Path(sys.argv[2]))))
