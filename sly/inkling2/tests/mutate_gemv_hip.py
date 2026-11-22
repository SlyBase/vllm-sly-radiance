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
    "noscale": ("acc[j][m] += s[j][m] * sc[j][u];", "acc[j][m] += s[j][m];"),
    "scaleblock": ("sc[j][u] = ok ? sr[j][itu * 8] : 0.f;", "sc[j][u] = ok ? sr[j][0] : 0.f;"),
    "negbias": ("const float v = tot[j][m] + (bias", "const float v = tot[j][m] - (bias"),
    "negbias_reduce": ("s += bias ? __uint_as_float((uint32_t)bias[n] << 16) : 0.f;",
                       "s -= bias ? __uint_as_float((uint32_t)bias[n] << 16) : 0.f;"),
    "noshuffle": ("  v += dpp<0x164>(v);  // row_xmask:4\n", ""),
    "dupshuffle": ("v += dpp<0xB1>(v);   // quad_perm [1,0,3,2]", "v += dpp<0x4E>(v);"),
    "unsigned": ("(float)(int)(int8_t)(w >> (8 * byte))", "(float)(int)(uint8_t)(w >> (8 * byte))"),
    "halfswap": ("const uint4 v = h ? wb[j][u] : wa[j][u];", "const uint4 v = h ? wa[j][u] : wb[j][u];"),
    "xhalf": ("xc = xs[m * kc8 + xb + (2 * h + 1) * 8];", "xc = xs[m * kc8 + xb + (2 * h) * 8];"),
    "noswizzle": ("((cc & 3) << 3) | (cc >> 2)", "((cc >> 2) << 2) | (cc & 3)"),
    "xstride": ("v = *(const uint4*)(x + (int64_t)m * SX + k);", "v = *(const uint4*)(x + (int64_t)m * K + k);"),
    "splitbase": ("partials[((int64_t)split * M + m) * N + n] = tot[j][m];",
                  "partials[((int64_t)0 * M + m) * N + n] = tot[j][m];"),
    "splitstart": ("const int i0 = (int)(((int64_t)nit * split) / nsplit);", "const int i0 = 0;"),
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
