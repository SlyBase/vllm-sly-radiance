#!/usr/bin/env python3
"""Check for the large-M tile path of the DFlash2 drafter's fc (sly/patch_w4a16_tiles.py).

`fc` (combine_hidden_states, ReplicatedLinear N=5120 K=25600, gs=128, bf16) is the one drafter
linear that sees the TARGET's token count: the 47k prefill calls it 12x with M = 4089, 24.6 ms
each, on _triton_w4a16_skinny_fmt_kernel with the gfx12x heuristic's 128x64x64 tile (measured on
decode batches). Its cost is the dequantized weight volume per call, (M / BLOCK_M) x the whole
65 MiB weight, so a wider BLOCK_M halves it. RADIANCE_DFLASH_FC_TILED_MIN_M routes M above the tile
table's horizon (M > 64) to that wider tile; default 0 = off, the stock heuristic unchanged.

Checked here:
  * the dispatch (_radiance_w4a16_big_m_cfg): off by default, the threshold, M > 64 only, the
    config override, a garbage env value, and that the tile really is wider than the stock one,
  * through the public path (triton_w4a16_skinny_fmt_gemm): the knob changes the launched BLOCK_M
    and nothing else -- same bytes out (DIGEST) with it off and on,
  * the kernel at both configs against the stock config (bit-identical) and against an fp32
    reference (unpacked ExLlama nibbles, (q - 8) * scale, fp32 matmul).

CPU (Triton interpreter, reduced N/K, real group size), in a plain container after the patch:
  BENCH_DEVICE=cpu TRITON_INTERPRET=1 python3 check_w4a16_fc_big_m.py
GPU (production shapes, DIGEST over M = 513, 880, 1514, 1760, 4096; timings with --bench):
  python3 check_w4a16_fc_big_m.py --gpu --bench
The dispatch part needs no torch and also runs on a machine without vLLM (it reads the patched
source): without torch/vLLM, or without a GPU for --gpu, the rest is skipped, not failed.
"""
import argparse
import ast
import contextlib
import hashlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
GS = 128
fails = 0
checked = 0

ap = argparse.ArgumentParser()
ap.add_argument("--gpu", action="store_true", help="production shapes N=5120 K=25600, M=513..4096")
ap.add_argument("--bench", action="store_true", help="also time both configs (GPU only)")
ap.add_argument("--src", default="", help="pristine rdna_hybrid_w4a16.py to patch (else searched)")
args = ap.parse_args()


def report(ok, tag, detail=""):
    global fails, checked
    checked += 1
    if not ok:
        fails += 1
    print(f"{'ok  ' if ok else 'FAIL'} {tag}" + (f"  {detail}" if detail and not ok else ""), flush=True)


def digest(t):
    return hashlib.sha256(t.contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()[:16]


@contextlib.contextmanager
def knobs(**kv):
    old = {k: os.environ.get(k) for k in kv}
    for k, v in kv.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---- the dispatch, read out of the patched source (no torch, no vLLM needed) --------------------

PRISTINE = [
    "/opt/data/vllm-sly-radiance/.dryrun/src/vllm/vllm/model_executor/kernels/linear/"
    "mixed_precision/rdna_hybrid_w4a16.py",
    "/opt/data/repos/vllm-sly-radiance/.dryrun/src/vllm/vllm/model_executor/kernels/linear/"
    "mixed_precision/rdna_hybrid_w4a16.py",
]


def patched_copy(tmp):
    """Apply the patch to a copy of the pristine vLLM module; return the copy's path."""
    src = args.src or next((p for p in PRISTINE if Path(p).is_file()
                            and "_radiance" not in Path(p).read_text()), "")
    if not src:
        raise SystemExit("FAIL no pristine rdna_hybrid_w4a16.py found (pass --src)")
    dst = Path(tmp) / "vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    import sysconfig
    real = sysconfig.get_paths
    sysconfig.get_paths = lambda *a, **k: {"purelib": tmp}  # the patch resolves its target from this
    sys.path.insert(0, str(HERE.parent))
    try:
        import patch_w4a16_tiles  # noqa: F401  -- importing applies every hunk
    finally:
        sysconfig.get_paths = real
    return dst


def extract(path, name):
    """The named top-level function of the patched file, compiled standalone."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            ns = {"os": os}
            exec(compile(ast.Module([node], []), str(path), "exec"), ns)
            return ns[name]
    raise SystemExit(f"FAIL {name} missing from the patched module")


with tempfile.TemporaryDirectory() as tmp:
    SRC = patched_copy(tmp)
    big_m_cfg = extract(SRC, "_radiance_w4a16_big_m_cfg")

    FC = (GS, 25600, 5120)  # group_size, K, N of combine_hidden_states
    STOCK_BLOCK_M = 128     # the gfx12x heuristic's tile for this shape at M > 512 (K >= 2 * N)
    with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M=None, RADIANCE_DFLASH_FC_TILED_CFG=None):
        report(big_m_cfg(*FC, 4089) is None and big_m_cfg(*FC, 513) is None, "default off")
    with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M=0):
        report(big_m_cfg(*FC, 4089) is None, "MIN_M=0 off")
    with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M=513):
        report(big_m_cfg(*FC, 512) is None, "MIN_M=513 leaves M=512 alone")
        report(big_m_cfg(*FC, 513) == (256, 128, 64, 8, 1), "MIN_M=513 takes M=513",
               f"got {big_m_cfg(*FC, 513)}")
        report(all(big_m_cfg(*FC, m) == (256, 128, 64, 8, 1) for m in (880, 1514, 1760, 4089)),
               "MIN_M=513 takes the prefill M values")
        cfg = big_m_cfg(*FC, 4089)
        report(cfg[0] // STOCK_BLOCK_M >= 2, "BLOCK_M at least 2x the stock tile",
               f"BM {cfg[0]} vs stock {STOCK_BLOCK_M}")
    with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M=1):
        report(big_m_cfg(*FC, 65) is not None and big_m_cfg(*FC, 64) is None,
               "MIN_M=1 still starts above 64")
    with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M=65):
        report(big_m_cfg(*FC, 65) == (256, 128, 64, 8, 1) and big_m_cfg(*FC, 64) is None,
               "MIN_M=65: M=65 taken, M=64 (tile-table horizon) untouched")
    with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M=513, RADIANCE_DFLASH_FC_TILED_CFG="128,128,128,4"):
        report(big_m_cfg(*FC, 4089) == (128, 128, 128, 4, None), "CFG override, 4 fields",
               f"got {big_m_cfg(*FC, 4089)}")
    with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M=513, RADIANCE_DFLASH_FC_TILED_CFG="192, 64, 128, 8, 2"):
        report(big_m_cfg(*FC, 4089) == (192, 64, 128, 8, 2), "CFG override, 5 fields with spaces",
               f"got {big_m_cfg(*FC, 4089)}")
    with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M=513, RADIANCE_DFLASH_FC_TILED_CFG="nonsense"):
        report(big_m_cfg(*FC, 4089) == (256, 128, 64, 8, 1), "garbage CFG keeps the default tile")
    with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M="abc"):
        report(big_m_cfg(*FC, 4089) is None, "garbage MIN_M is off, not a crash")
    with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M=513):
        report(big_m_cfg(64, 25600, 5120, 4089) == (256, 128, 64, 8, 1), "other group sizes pass through")

# ---- numerics -----------------------------------------------------------------------------------
try:
    import torch
    import triton  # noqa: F401
    from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as H
except Exception as e:  # no torch / no vLLM on this machine
    print(f"SKIP numerics: {e!r}")
else:
    if args.gpu and not torch.cuda.is_available():
        print("SKIP numerics: --gpu without a GPU")
    else:
        gpu = args.gpu
        dev = torch.device("cuda" if gpu else "cpu")
        # the Triton interpreter mis-casts int -> bf16; the kernel body is dtype-generic, so the CPU
        # run checks fp16 and the GPU run bf16 (production)
        DT = torch.bfloat16 if gpu else torch.float16
        N, K = (5120, 25600) if gpu else (96, 2048)
        MS = [513, 880, 1514, 1760, 4096] if gpu else [133]
        BIG = (256, 128, 64, 8, 1)
        STOCK = (STOCK_BLOCK_M, 64, 64, 8, None)
        torch.manual_seed(0)
        b_q = torch.randint(-2 ** 31, 2 ** 31 - 1, (N, K // 8), dtype=torch.int32, device=dev)
        scales = (torch.rand(N, K // GS, dtype=torch.float32) * 0.02 + 0.001).to(DT).to(dev)
        SHIFTS = torch.tensor([(j // 2) * 4 + (j % 2) * 16 for j in range(8)], dtype=torch.int32)

        def reference(a):
            q = ((b_q.cpu()[:, :, None] >> SHIFTS[None, None, :]) & 0xF)
            q = q.reshape(N, -1).float()
            s = scales.cpu().float().repeat_interleave(GS, dim=1)
            return a.cpu().float() @ ((q - 8) * s).t()

        class _Rec:  # records the BLOCK sizes the public path launches with
            def __init__(self, kern):
                self.kern, self.calls = kern, []

            def __getitem__(self, grid):
                def launch(*a, **kw):
                    self.calls.append(kw)
                    return self.kern[grid](*a, **kw)
                return launch

        for M in MS:
            a = torch.randn(M, K, dtype=torch.float32).to(DT).to(dev)
            ref = reference(a)
            floor = ref.abs().max().item() * 2 ** -7  # output rounding + fp32 reassociation
            outs = {}
            for tag, cfg in (("stock", STOCK), ("big", BIG)):
                outs[tag] = H.triton_w4a16_splitk_gemm(a, b_q, scales, GS, tuple(cfg) + (1, 0)).cpu()
                err = (outs[tag].float() - ref).abs().max().item()
                report(outs[tag].shape == (M, N) and err <= floor, f"{'bf16' if gpu else 'fp16'} M={M} {tag}",
                       f"max err {err:.3e} > floor {floor:.3e}")
            same = torch.equal(outs["stock"], outs["big"])
            report(same, f"M={M} big tile bit-identical to stock",
                   f"DIGEST stock {digest(outs['stock'])} big {digest(outs['big'])}")
            print(f"     M={M:4d} DIGEST stock {digest(outs['stock'])} big {digest(outs['big'])} "
                  f"{'SAME' if same else 'WRONG'}")

            # public path: the gfx12x branch is pure Python, so the dispatch is exercised on CPU too
            on_gfx12x, rec = H._on_gfx12x, _Rec(H._triton_w4a16_skinny_fmt_kernel)
            H._on_gfx12x = lambda: True
            H._triton_w4a16_skinny_fmt_kernel = rec
            try:
                with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M=None):
                    off = H.triton_w4a16_skinny_fmt_gemm(a, b_q, scales, GS)
                off_bm = rec.calls[-1]["BLOCK_M"]
                with knobs(RADIANCE_DFLASH_FC_TILED_MIN_M=min(513, M)):  # M=133 (CPU) is below 513
                    on = H.triton_w4a16_skinny_fmt_gemm(a, b_q, scales, GS)
                on_bm = rec.calls[-1]["BLOCK_M"]
            finally:
                H._on_gfx12x, H._triton_w4a16_skinny_fmt_kernel = on_gfx12x, rec.kern
            report(off_bm == STOCK_BLOCK_M, f"M={M} knob off keeps BLOCK_M {STOCK_BLOCK_M}",
                   f"got {off_bm}")
            report(on_bm == BIG[0], f"M={M} knob on raises BLOCK_M to {BIG[0]}", f"got {on_bm}")
            report(digest(off) == digest(on), f"M={M} public path: same bytes with the knob on",
                   f"DIGEST off {digest(off)} on {digest(on)}")

            if gpu and args.bench:
                for tag, cfg in (("stock", STOCK), ("big", BIG)):
                    call = (lambda c: (lambda: H.triton_w4a16_splitk_gemm(
                        a, b_q, scales, GS, tuple(c) + (1, 0))))(cfg)
                    for _ in range(5):
                        call()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(20):
                        call()
                    torch.cuda.synchronize()
                    print(f"     M={M:4d} {tag:5s} {(time.perf_counter() - t0) / 20e-3:8.3f} ms",
                          flush=True)

print(f"\n{checked} checks, {fails} failures")
sys.exit(1 if fails else 0)
