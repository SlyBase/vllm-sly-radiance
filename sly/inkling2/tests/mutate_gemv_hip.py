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
