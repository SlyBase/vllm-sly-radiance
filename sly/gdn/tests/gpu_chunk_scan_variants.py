"""[GPU WINDOW ONLY] chunk_scan build variants s1 / s2 / s3 vs stock: DIGEST bit-identity and ABBA timing.

Needs a GPU, an r4d.so that carries the variants (all of them live in ONE .so, next to the stock entry
point, so the timing runs round-robin in one process) and torch+ROCm, i.e. the release image inside an
acquired gpu-window.  The script does NOT acquire the window.  Without a GPU or the variants the module
skips.  Run:   python3 -u sly/gdn/tests/gpu_chunk_scan_variants.py     (exit 0 = no WRONG; ~3-4 min)
Exit codes: 0 = every DIGEST/STABLE check passed, 1 = WRONG, 2 = SKIP (nothing ran -- no GPU or an r4d.so
without the variants). A skipped gate is not a passed gate.

  s1  workgroup-processor mode (no -mcumode)
  s2  s1 + LDS 60,928 -> 43,008 B, so three workgroups fit one WGP
  s3  s2 + the next chunk's staging loads issued after barrier (3)

What the three slices compile to on gfx1201 (hipcc -Rpass-analysis=kernel-resource-usage):

  variant  VGPR  spill  LDS B   waves/SIMD
  stock     256      4   60928  4
  s1        256      4   60928  4
  s2        256      0   43008  5
  s3        189      0   43008  6

Checks
  1. DIGEST: sha256 over the bytes of `o` and of `ht`, variant == stock, on the shapes below, N in {1, 2
     varlen}, H=48/Hg=16.  Any mismatch prints WRONG and fails.  Arithmetic order is unchanged by
     construction, so this is bit-identity, not a tolerance.
  2. STABLE: every variant five times on T=4089; a missing barrier shows up as run-to-run drift.
  3. Timing (1 layer = one launch, H=48): ABBA-palindromic rounds stock,s1,s2,s3,s3,s2,s1,stock, each slot
     a batch of 20 back-to-back launches; per-launch median per variant, ratio to stock, T=4089 (the gate
     number) and T=1760 / 880 (the ladder's other rows).  Timing never fails the run: every slice is a
     measurement point, the verdict is the ratio column.
"""
import hashlib, pathlib, statistics, sys

try:
    import torch
    torch.set_num_threads(1)  # 8 threads made the fp64 CPU oracle ~240x slower in the container (spin-wait)
    import r4d
    NAMES = {"stock": "gdn_chunk_scan_k128_v128_c64_bf16"}
    NAMES.update({v: f"gdn_chunk_scan_k128_v128_c64_bf16_{v}" for v in ("s1", "s2", "s3")})
    _ok = torch.cuda.is_available() and all(hasattr(r4d, n) for n in NAMES.values())
except Exception:  # no torch / no r4d / no GPU
    _ok = False
if not _ok:
    if __name__ == "__main__":
        print("SKIP: runs only in the GPU window (needs GPU + r4d with the s1/s2/s3 variants)")
        sys.exit(2)                                  # not a pass: nothing was measured
    import pytest
    pytest.skip("runs only in the GPU window (needs GPU + r4d with the variants)", allow_module_level=True)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from chunk_scan_ref import make_inputs, chunk_cumsum, solve_A, BT

FNS = {k: getattr(r4d, n) for k, n in NAMES.items()}
VARIANTS = ("s1", "s2", "s3")
# (T, N): one exact chunk, one chunk plus a single token, two full ladder rows, and the T=4089 gate shape
# with a varlen split.  Each (T, N) costs one fp64 solve_A on the CPU, so the list is kept short on purpose.
CASES = [(64, 1), (65, 1), (880, 1), (4089, 1), (4089, 2)]
ROUND = ("stock", "s1", "s2", "s3", "s3", "s2", "s1", "stock")   # palindromic: every position bias cancels
BATCH, ROUNDS, WARM = 20, 5, 3


def lens(T, n):
    return [T] if n == 1 else [T // 3 + 1, T - (T // 3 + 1)] if T > 3 else [T]


def to_dev(inp):
    g = chunk_cumsum(inp["glog"], inp["cu"])
    A = solve_A(inp, g)
    d = "cuda"
    return dict(
        q=inp["q"].to(torch.bfloat16).to(d), k=inp["k"].to(torch.bfloat16).to(d),
        v=inp["v"].to(torch.bfloat16).to(d), A=A.to(torch.bfloat16).to(d),
        g=g.to(torch.float32).to(d), beta=inp["beta"].to(torch.float32).to(d),
        h0=inp["h0"].to(torch.float32).to(d), cu=torch.tensor(inp["cu"], dtype=torch.int32, device=d))


def launch(fn, t, o, ht, scale, H=48, Hg=16):
    fn(t["q"].data_ptr(), t["k"].data_ptr(), t["v"].data_ptr(), t["A"].data_ptr(), t["g"].data_ptr(),
       t["beta"].data_ptr(), t["h0"].data_ptr(), o.data_ptr(), ht.data_ptr(), t["cu"].data_ptr(),
       t["cu"].numel() - 1, H, Hg, 128, 128, BT, float(scale), torch.cuda.current_stream().cuda_stream)


def run(fn, t, scale):
    o = torch.full_like(t["v"], float("nan"))     # poison: a skipped store cannot hide behind stale zeros
    ht = torch.full_like(t["h0"], float("nan"))
    launch(fn, t, o, ht, scale)
    torch.cuda.synchronize()
    return o, ht


def digest(x):
    return hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()[:16]


def digest_gate():
    wrong = 0
    for T, n in CASES:
        inp = make_inputs(lens(T, n), H=48, Hg=16, K=128, V=128, seed=T)
        t = to_dev(inp)
        ref = run(FNS["stock"], t, inp["scale"])
        rh = (digest(ref[0]), digest(ref[1]))
        for v in VARIANTS:
            got = run(FNS[v], t, inp["scale"])
            for name, a, b in (("o", rh[0], digest(got[0])), ("ht", rh[1], digest(got[1]))):
                same = a == b
                wrong += not same
                print(f"DIGEST T={T} N={n} {name} {v} stock={a} {v}={b} {'ok' if same else 'WRONG'}")
    return wrong


def stable_gate():
    inp = make_inputs([4089], H=48, Hg=16, K=128, V=128, seed=4089)
    t = to_dev(inp)
    wrong = 0
    for v in VARIANTS:
        ds = {digest(torch.cat([x.flatten().view(torch.uint8) for x in run(FNS[v], t, inp["scale"])]))
              for _ in range(5)}
        wrong += len(ds) != 1
        print(f"STABLE T=4089 {v} distinct_digests={len(ds)} {'ok' if len(ds) == 1 else 'WRONG'}")
    return wrong


def timing(T):
    inp = make_inputs([T], H=48, Hg=16, K=128, V=128, seed=1)
    t = to_dev(inp)
    o = torch.empty_like(t["v"]); ht = torch.empty_like(t["h0"])

    def batch(name):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(BATCH):
            launch(FNS[name], t, o, ht, inp["scale"])
        b.record(); torch.cuda.synchronize()
        return a.elapsed_time(b) / BATCH
    for name in FNS:
        for _ in range(WARM):
            batch(name)
    res = {k: [] for k in FNS}
    for _ in range(ROUNDS):
        for name in ROUND:
            res[name].append(batch(name))
    ms = {k: statistics.median(v) for k, v in res.items()}
    print(f"TIME T={T} H=48 stock={ms['stock']:.4f} ms " +
          " ".join(f"{v}={ms[v]:.4f} ms ratio={ms[v] / ms['stock']:.3f}" for v in VARIANTS))
    # spread of the stock arm against itself (first vs last slot of each round): the noise floor
    first, last = statistics.median(res["stock"][0::2]), statistics.median(res["stock"][1::2])
    print(f"NOISE T={T} stock first-slot/last-slot = {first / last:.3f}")


def test_gpu_chunk_scan_variants():
    assert digest_gate() == 0 and stable_gate() == 0
    for T in (4089, 1760, 880):
        timing(T)


if __name__ == "__main__":
    bad = digest_gate() + stable_gate()
    for T in (4089, 1760, 880):
        timing(T)
    print("WRONG" if bad else "ALL OK")
    sys.exit(1 if bad else 0)
