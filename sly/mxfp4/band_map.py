#!/usr/bin/env python3
"""Which kernel/config serves every M in 1..2048 for each production GEMM shape (no GPU needed).

Pure-Python replica of launch_impl()/launch_at_impl() in radiance_mxfp4_fp8.hip plus the Python
side gates in radiance_mxfp4.py (A_TILED_MIN_M). Keep it in step with the .hip: it is a map, not
the source of truth, and the GPU bench (tests-lessons/A/bench_wide_cells.py) is what measures.

  python3 band_map.py                         # production env (DECODE_MAX_M=128, WPERM=1, A_TILED 513)
  python3 band_map.py --wide 256              # with RADIANCE_MXFP4_WIDE_MAX_M=256
  python3 band_map.py --wide 256 --tn4 1024   # also RADIANCE_MXFP4_TN4_MIN_M=1024

Output per shape: contiguous M ranges with the same kernel label, and for the folded/atiled rows
the 256-row-tile padding (rows computed but not asked for) at the worst M of the range.
"""
import argparse

# (name, N, K, tiled producer in production: add_rms_quant / silu feed the atiled kernel from M>=513)
SHAPES = [
    ("gate_up", 34816, 5120, True),
    ("qkvz",    16384, 5120, True),
    ("qkv",     14336, 5120, True),
    ("o_proj",   5120, 6144, False),   # gdn_norm_quant / attn out: row-major in production
    ("out_proj", 5120, 6144, False),
    ("down",     5120, 17408, False),  # no tiled producer (K > add_rms limit)
    ("in_proj_ba", 96, 5120, True),
]
DEC_KS, DEC_MAX_N, DEC_MTILE, DEC_MAX_TM, BMF = 4, 36864, 16, 8, 256


def split_k_for(nblk, M):
    fill = 110 if M <= 24 else 78
    ks = 1
    while ks < DEC_KS:
        if nblk * ks >= fill:
            return ks
        ks <<= 1
    return DEC_KS


def decode_cfg(M, N, K, tune16=True):
    nblk = (N + 127) // 128
    tm = (M + 15) // 16
    if M > 64:
        if nblk >= 110:   dks, bk = 1, 64
        elif nblk >= 48:  dks, bk = 1, 128
        elif nblk <= 8:   dks, bk = DEC_KS, 128
        elif K >= 6144:   dks, bk = (2, 128) if M <= 80 else (DEC_KS, 64)
        else:             dks, bk = 1, 128
    elif tune16 and M > 8:
        dks = split_k_for(nblk, M)
        bk = 128
        if dks == DEC_KS and nblk > 8 and K < 8192 and 16 <= M <= 24:
            dks = 2
    else:
        dks = split_k_for(nblk, M)
        bk = 64 if (tm == 4 and dks == 1) else 128
    if K % 128:
        bk = 64
        if dks == 2:
            dks = DEC_KS
    return dks, bk, tm


def wide_cfg(M, N, K):
    nblk = (N + 127) // 128
    if nblk >= 48:
        return 1
    if nblk <= 8:
        return DEC_KS
    if K >= 6144:
        return DEC_KS if M <= 192 else 2
    return 1


def label(M, N, K, tiled_prod, decode_max, wide_max, tn4_min, a_tiled_min, scratch_bytes):
    if 0 < M <= decode_max and M <= DEC_MTILE * DEC_MAX_TM and N <= DEC_MAX_N:
        dks, bk, tm = decode_cfg(M, N, K)
        if dks == 1 or dks * M * N * 4 <= scratch_bytes:
            return f"decode d{dks}/b{bk}/tm{tm}", 0
    if wide_max > 128 and 128 < M <= wide_max and N <= DEC_MAX_N:
        dks = wide_cfg(M, N, K)
        if dks == 1 or dks * M * N * 4 <= scratch_bytes:
            tm = (M + 15) // 16
            return f"WIDE d{dks}/b64/tm{tm}", 0
    tn = 4 if (M >= tn4_min and N >= 128) else 2
    kind = "atiled" if (a_tiled_min and M >= a_tiled_min and tiled_prod) else "folded"
    pad = -(-M // BMF) * BMF - M
    return f"{kind} TN{tn} (BMF256)", pad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decode-max", type=int, default=128)
    ap.add_argument("--wide", type=int, default=0)
    ap.add_argument("--tn4", type=int, default=2048)
    ap.add_argument("--a-tiled", type=int, default=513)
    ap.add_argument("--maxm", type=int, default=2048)
    a = ap.parse_args()
    scratch = 4 * max(64, a.decode_max) * 36864 * 4
    print(f"env: DECODE_MAX_M={a.decode_max} WIDE_MAX_M={a.wide} TN4_MIN_M={a.tn4} "
          f"A_TILED_MIN_M={a.a_tiled} WPERM=1 TUNE16=1; scratch {scratch >> 20} MiB")
    for name, N, K, tp in SHAPES:
        print(f"\n{name} N={N} K={K} (nblk {(N + 127) // 128}, tiled producer: {'yes' if tp else 'no'})")
        runs = []
        for M in range(1, a.maxm + 1):
            lab, pad = label(M, N, K, tp, a.decode_max, a.wide, a.tn4, a.a_tiled, scratch)
            if runs and runs[-1][2] == lab:
                runs[-1][1] = M
                runs[-1][3] = max(runs[-1][3], pad)
            else:
                runs.append([M, M, lab, pad])
        # fold the many 1-wide decode runs: merge neighbours that differ only in tm
        for lo, hi, lab, pad in runs:
            extra = f"   worst padding {pad} rows" if pad and hi > lo else (f"   padding {pad} rows" if pad else "")
            print(f"  M {lo:>4}..{hi:<4} {lab}{extra}")


if __name__ == "__main__":
    main()
