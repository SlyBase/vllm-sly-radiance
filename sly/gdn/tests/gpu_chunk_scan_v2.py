"""[GPU WINDOW ONLY] chunk_scan v2 vs stock: DIGEST, accuracy and timing.

Needs a GPU, the r4d build that carries the v2 kernel (bind-mounted r4d.so) and torch+ROCm, i.e. the
release image inside an acquired gpu-window.  Without them the module skips ("runs only in the GPU
window").  Run:   python3 sly/gdn/tests/gpu_chunk_scan_v2.py   (exit 0 = no WRONG)

Checks
  1. DIGEST: sha256 over the bytes of `o` and of `ht`, v2 == stock, for T in SHAPES, N in {1, 2 varlen},
     H=48/Hg=16.  Reported separately for o and ht.  Any mismatch prints WRONG and fails.  v2 moves one
     prefetch issue point and not one arithmetic instruction, so this is a construction check, not a
     tolerance: 28 of 28 must be identical.
  2. Accuracy: relRMSE of v2 against the fp64 CPU reference (sly/gdn/chunk_scan_ref.py), H=6/Hg=2:
     o <= 5e-3, state <= 4e-3 (unchanged from stock; kept as the guard against a build accident).
  3. Timing (T=4089, H=48): ABBA stock/v2/v2/stock, 5 warmup + 20 timed, median; ratio v2/stock.
     GO if <= 0.92, ABORT if > 0.95 (docs/GDN-CHUNK-SCAN-V2.md).

The kernel source carries the issue point as PLATE (0 = stock position, 1 = after the main loop,
2 = before the state update).  They are three separate builds of the same unit; run this script once
per build, the ratio printed is the one for the PLATE that build was compiled with.
"""
import hashlib, statistics, sys

try:
    import pytest
except ImportError:
    pytest = None
try:
    import pathlib
    import torch
    torch.set_num_threads(1)  # 8 threads made the fp64 CPU oracle ~240x slower in the container (spin-wait)
    import r4d
    _ok = torch.cuda.is_available() and hasattr(r4d, "gdn_chunk_scan_k128_v128_c64_bf16_v2")
except Exception:  # no torch / no r4d / no GPU
    _ok = False
if not _ok:
    if __name__ == "__main__":
        print("SKIP: runs only in the GPU window (needs GPU + r4d with the v2 kernel)")
        sys.exit(0)
    if pytest is not None:
        pytest.skip("runs only in the GPU window (needs GPU + r4d with the v2 kernel)", allow_module_level=True)
    raise ImportError("GPU window only")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from chunk_scan_ref import make_inputs, chunk_cumsum, solve_A, chunked, rel_rmse, BT

STOCK = getattr(r4d, "gdn_chunk_scan_k128_v128_c64_bf16")
V2 = getattr(r4d, "gdn_chunk_scan_k128_v128_c64_bf16_v2")
SHAPES = [64, 65, 513, 880, 1514, 1760, 4089]
LAYOUTS = [[1], [2]]          # N=1: one sequence of T; N=2: two sequences summing to T (varlen)


def lens(T, n):
    return [T] if n == 1 else [T // 3 + 1, T - (T // 3 + 1)] if T > 3 else [T]


def to_dev(inp, dtype_gate=torch.float32):
    cu = inp["cu"]
    g = chunk_cumsum(inp["glog"], cu)
    A = solve_A(inp, g)
    d = "cuda"
    return dict(
        q=inp["q"].to(torch.bfloat16).to(d), k=inp["k"].to(torch.bfloat16).to(d),
        v=inp["v"].to(torch.bfloat16).to(d), A=A.to(torch.bfloat16).to(d),
        g=g.to(dtype_gate).to(d), beta=inp["beta"].to(dtype_gate).to(d),
        h0=inp["h0"].to(torch.float32).to(d), cu=torch.tensor(cu, dtype=torch.int32, device=d)), g


def run(fn, t, scale, H, Hg):
    o = torch.empty_like(t["v"]); ht = torch.empty_like(t["h0"])
    N = t["cu"].numel() - 1
    fn(t["q"].data_ptr(), t["k"].data_ptr(), t["v"].data_ptr(), t["A"].data_ptr(), t["g"].data_ptr(),
       t["beta"].data_ptr(), t["h0"].data_ptr(), o.data_ptr(), ht.data_ptr(), t["cu"].data_ptr(),
       N, H, Hg, 128, 128, BT, float(scale), torch.cuda.current_stream().cuda_stream)
    return o, ht


def digest(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def digest_gate():
    wrong = 0
    for T in SHAPES:
        for n in (1, 2):
            inp = make_inputs(lens(T, n), H=48, Hg=16, K=128, V=128, seed=T)
            t, _ = to_dev(inp)
            (o0, s0), (o1, s1) = run(STOCK, t, inp["scale"], 48, 16), run(V2, t, inp["scale"], 48, 16)
            torch.cuda.synchronize()
            for name, a, b in (("o", o0, o1), ("ht", s0, s1)):
                same = digest(a) == digest(b)
                wrong += (not same)
                print(f"DIGEST T={T} N={n} {name} stock={digest(a)} v2={digest(b)} {'ok' if same else 'WRONG'}")
    return wrong


def accuracy_gate():
    wrong = 0
    for T in (65, 513, 1514):
        inp = make_inputs([T], H=6, Hg=2, K=128, V=128, seed=T)
        t, _ = to_dev(inp)
        ro, rs = chunked(inp)
        o, s = run(V2, t, inp["scale"], 6, 2)
        eo, es = rel_rmse(o.double().cpu(), ro), rel_rmse(s.double().cpu(), rs)
        bad = eo > 5e-3 or es > 4e-3
        wrong += bad
        print(f"ACC T={T} o={eo:.2e} state={es:.2e} {'WRONG' if bad else 'ok'}")
    return wrong


def timing_gate():
    inp = make_inputs([4089], H=48, Hg=16, K=128, V=128, seed=1)
    t, _ = to_dev(inp)
    def one(fn):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); run(fn, t, inp["scale"], 48, 16); b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b)
    res = {"stock": [], "v2": []}
    for fn in (STOCK, V2): [one(fn) for _ in range(5)]
    for name, fn in (("stock", STOCK), ("v2", V2), ("v2", V2), ("stock", STOCK)):   # ABBA
        res[name] += [one(fn) for _ in range(20)]
    ms = {k: statistics.median(v) for k, v in res.items()}
    r = ms["v2"] / ms["stock"]
    verdict = "GO" if r <= 0.92 else "ABORT" if r > 0.95 else "GREY"
    print(f"TIME T=4089 H=48 stock={ms['stock']:.3f} ms v2={ms['v2']:.3f} ms ratio={r:.3f} {verdict}")
    return r > 0.95


def test_gpu_chunk_scan_v2():
    assert digest_gate() == 0 and accuracy_gate() == 0 and not timing_gate()


if __name__ == "__main__":
    bad = digest_gate() + accuracy_gate() + int(timing_gate())
    print("WRONG" if bad else "ALL OK")
    sys.exit(1 if bad else 0)
