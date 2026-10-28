"""[GPU WINDOW ONLY] chunk_scan with r4d_gdn_exact_decay.patch: every unit against the fp64 recurrence on
chunks whose gate span g_0 - g_last exceeds 160 (libr4d issue #4), where the unpatched split form's e^80
clamp attenuates the chunk.

Same requirements as gpu_chunk_scan_variants.py (GPU, an r4d.so with the s1/s2/s3 variants, torch+ROCm, an
acquired gpu-window; the script does NOT acquire it).  Run:
    python3 -u sly/gdn/tests/gpu_chunk_scan_exact_decay.py      (exit 0 = no WRONG; ~1 min)
Exit codes: 0 = every unit within 1 % on every case, 1 = WRONG, 2 = SKIP (nothing ran).

Oracle: `naive` from sly/gdn/chunk_scan_ref.py (token-by-token, fp64), fed the same bf16-rounded q/k/v the
kernel reads.  The kernel also reads A = solve_A in bf16, so the bar is a tolerance (1 % relRMSE on `o` and
`ht`), not a digest.  Each case also prints what the unpatched split form gives on the same inputs
(`chunked(split=True)`, fp64), so a passing run shows the patch is what makes the difference.
Bit-identity below the threshold is gpu_chunk_scan_variants.py's DIGEST gate, which is unchanged.
"""
import pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    from gpu_chunk_scan_variants import FNS, lens, run, to_dev, torch
except BaseException:  # its module-level pytest.skip (no GPU / no variants) is a BaseException
    if __name__ == "__main__":
        print("SKIP: runs only in the GPU window (needs GPU + r4d with the s1/s2/s3 variants)")
        sys.exit(2)                                  # not a pass: nothing was measured
    raise

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from chunk_scan_ref import make_inputs, naive, chunked, rel_rmse

TOL = 0.01
# (T, N, span): span 3 is an ordinary chunk (split path), span 160 mixes split and exact chunks in one
# sequence, 200-1000 are exact-path chunks; N = 2 is a varlen split with unequal lengths.
CASES = [(200, 1, 3.0), (200, 1, 160.0), (200, 1, 200.0), (200, 1, 400.0), (195, 2, 400.0), (200, 1, 1000.0)]


def bf16_rounded(inp):
    out = dict(inp)
    for x in ("q", "k", "v"):
        out[x] = inp[x].to(torch.bfloat16).double()
    out["h0"] = inp["h0"].to(torch.float32).double()
    return out


def exact_decay_gate():
    wrong = 0
    for T, n, span in CASES:
        inp = make_inputs(lens(T, n), H=48, Hg=16, K=128, V=128, span=span, seed=T)
        ref = bf16_rounded(inp)
        o0, s0 = naive(ref)
        so, ss = chunked(ref, split=True)
        t = to_dev(inp)
        for name, fn in FNS.items():
            o, ht = run(fn, t, inp["scale"])
            eo, es = rel_rmse(o.double().cpu(), o0), rel_rmse(ht.double().cpu(), s0)
            ok = eo <= TOL and es <= TOL          # NaN compares False: a non-finite output is WRONG
            wrong += not ok
            print(f"EXACT T={T} N={n} span={span:g} {name} o={eo:.2e} ht={es:.2e} "
                  f"(split form {rel_rmse(so, o0):.2e}) {'ok' if ok else 'WRONG'}")
    return wrong


def test_gpu_chunk_scan_exact_decay():
    assert exact_decay_gate() == 0


if __name__ == "__main__":
    bad = exact_decay_gate()
    print("WRONG" if bad else "ALL OK")
    sys.exit(1 if bad else 0)
