#!/usr/bin/env python3
"""CPU tests for RADIANCE_ATTN_Q_FP8: the prefill kernel widens the e4m3 query itself.

With `--kv-cache-dtype fp8` the attention layer hands the R4D impl an e4m3 query, and the glue used to
run `q.to(bfloat16)` (and `q * q_scale` when the static scale is not 1) before every prefill launch:
a 25 MB read + 50 MB write elementwise copy per (layer, chunk), 176 launches / ~57 ms over a 47k-token
prefill. O_QF8 (bit 24) reads the e4m3 bytes in the kernel prologue instead and rebuilds the very
dwords the bf16 load would have produced. The claim rests on one formula,

    bf16_rne( float(e4m3 byte) * float(q_scale) )   ==   torch: q8.to(bfloat16) * q_scale

(torch computes a bf16 tensor times a python scalar in fp32 with the scalar rounded to fp32 and rounds
the product to bf16 RNE), checked here on the CPU over all 256 codes x a set of scales. The e4m3
widening itself is the hardware `v_cvt_pk_f32_fp8` on the device; the GPU gate (gate_v4.sh) re-checks
the whole path, hashing kernel output of the fp8 arm against the cast arm on the device.

Plus the launcher toggle logic and the patch-text invariants. Plain torch + numpy, no vllm, no GPU.

Run:  python3 sly/tests/test_attn_q_fp8_cpu.py     (exit 0 = pass, also collectable by pytest)
"""
import ast
import os
import re
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "radiance_r4d_hybrid_attn.py"
PATCH = (ROOT / "sly" / "r4d" / "r4d_extras_rx10.patch").read_text()
ADDED = "\n".join(line[1:] for line in PATCH.splitlines() if line.startswith("+"))

O_QF8 = 16777216          # bit 24

SCALES = [1.0, 0.0625, 0.5, 2.0, 3.0, 0.1, 1.0 / 3.0, 448.0 / 7.0, 1e-3, 123.456, 1e-30, 1e30, -0.75]


def kernel_formula():
    """The text of r4d_bf16_rne and of the QF8 prologue, as written in the patch."""
    m = re.search(r"uint32_t r4d_bf16_rne\(float f\) \{(.*?)\n\}", ADDED, re.S)
    assert m, "the r4d_bf16_rne helper disappeared from the patch"
    body = m.group(1)
    assert re.search(r"\(u \+ 0x7fffu \+ \(\(u >> 16\) & 1u\)\) >> 16", body), \
        "the rounding is no longer round-to-nearest-even on the low 16 bits"
    assert re.search(r"\(u >> 16\) \| 0x0040u", body), "NaN is no longer kept as a quiet bf16 NaN"
    return body


def e4m3_to_float(codes):
    """OCP e4m3fn (torch.float8_e4m3fn, what the fp8 query is): independent of torch's decoder."""
    c = np.asarray(codes, dtype=np.uint32)
    sign = np.where(c & 0x80, -1.0, 1.0).astype(np.float32)
    e, m = (c >> 3) & 0xF, c & 0x7
    val = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1.0 + m / 8.0) * 2.0 ** (e.astype(np.float64) - 7))
    val = np.where((e == 0xF) & (m == 0x7), np.nan, val)
    return (sign * val).astype(np.float32)


def rne(x):
    """r4d_bf16_rne, transcribed: fp32 in, bf16 bits (uint16) out."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF) >> 16
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    return np.where(nan, ((u >> 16) | 0x0040) & 0xFFFF, r).astype(np.uint16)


def kernel_q(codes, q_scale):
    """The prologue: widen exactly, multiply in fp32 by the fp32 scale, round once."""
    return rne(e4m3_to_float(codes) * np.float32(q_scale))


def glue_q(codes, q_scale):
    """Today's glue, on the CPU: q.to(bfloat16), then * q_scale when it is not 1."""
    q = torch.as_tensor(np.asarray(codes, dtype=np.uint8)).view(torch.float8_e4m3fn).to(torch.bfloat16)
    if q_scale != 1.0:
        q = q * q_scale
    return q.view(torch.uint16).numpy()


def test_formula_matches_the_glue_over_all_codes_and_scales():
    kernel_formula()
    codes = np.arange(256, dtype=np.uint8)
    ref = torch.as_tensor(codes).view(torch.float8_e4m3fn).float().numpy()
    ok = ~np.isnan(ref)
    assert np.array_equal(np.isnan(e4m3_to_float(codes)), ~ok)
    assert np.array_equal(e4m3_to_float(codes)[ok], ref[ok]), "the e4m3 decoder is wrong"
    g = np.random.default_rng(20261021)
    scales = SCALES + [float(s) for s in 10.0 ** g.uniform(-4, 3, 200)] + \
        [float(np.float32(s)) for s in g.uniform(0.001, 8, 100)]
    for s in scales:
        want, got = glue_q(codes, s), kernel_q(codes, s)
        nan = np.isnan(e4m3_to_float(codes))
        # torch's CPU fp32 -> bf16 collapses every NaN to one pattern; the payload is the GPU gate's.
        assert np.array_equal(want[~nan], got[~nan]), f"q_scale {s}: kernel formula != glue"
        assert np.all((got[nan] & 0x7F80) == 0x7F80) and np.all(got[nan] & 0x7F), "NaN must stay NaN"


def test_scale_one_is_a_plain_widen():
    codes = np.arange(256, dtype=np.uint8)
    nan = np.isnan(e4m3_to_float(codes))
    assert np.array_equal(kernel_q(codes, 1.0)[~nan], glue_q(codes, 1.0)[~nan])
    assert np.array_equal(kernel_q(codes, 1.0)[~nan], (e4m3_to_float(codes).view(np.uint32) >> 16)[~nan]
                          .astype(np.uint16)), "e4m3 -> bf16 must be exact (the high half of the fp32)"


def _launcher_ns(push, said):
    """`_q_fp8_ok` and `_q_kernel` with the module names they close over, no vllm import."""
    src = LAUNCHER.read_text()
    tree = ast.parse(src)
    ns = {"os": os, "torch": torch, "math": __import__("math"), "_set_q_fp8": push, "_logged": set(),
          "_say_once": lambda key, msg: said.append(msg)}
    for name in ("_q_fp8_ok", "_q_kernel"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.get_source_segment(src, fn), str(LAUNCHER), "exec"), ns)
    return ns


def _with_knob(value, fn):
    env = os.environ.pop("RADIANCE_ATTN_Q_FP8", None)
    if value is not None:
        os.environ["RADIANCE_ATTN_Q_FP8"] = value
    try:
        return fn()
    finally:
        os.environ.pop("RADIANCE_ATTN_Q_FP8", None)
        if env is not None:
            os.environ["RADIANCE_ATTN_Q_FP8"] = env


def test_toggle_follows_the_pointer():
    """The kernel is told, per launch, exactly what the pointer it gets is."""
    pushed, said = [], []
    kern = _launcher_ns(pushed.append, said)["_q_kernel"]
    q8 = torch.randn(8, 2, 256).to(torch.float8_e4m3fn)

    def run():
        out = kern(q8, 1.0)
        assert pushed == [0.0] and out.dtype == torch.bfloat16, "the knob must default off"
        assert torch.equal(out, q8.to(torch.bfloat16))

    _with_knob(None, run)

    def run_on():
        pushed.clear()
        out = kern(q8, 0.0625)
        assert pushed == [0.0625] and out.dtype == torch.float8_e4m3fn, \
            "contiguous e4m3 q with the knob on goes straight to the kernel, with its scale"
        assert out.data_ptr() == q8.data_ptr(), "the qualifying query must not be copied"
        pushed.clear()
        assert kern(q8, 1.0).data_ptr() == q8.data_ptr() and pushed == [1.0]

        pushed.clear()
        q16 = q8.to(torch.bfloat16)
        out = kern(q16, 0.5)
        assert pushed == [0.0] and out.data_ptr() == q16.data_ptr(), "bf16 q is passed as is, declared bf16"

        pushed.clear()
        out = kern(torch.randn(4, 2, 256), 0.5)
        assert pushed == [0.0] and out.dtype == torch.bfloat16, "fp32 q is cast, never read as e4m3"

        pushed.clear()
        nc = q8.transpose(0, 1).contiguous().transpose(0, 1)
        out = kern(nc, 0.5)
        assert pushed == [0.0] and out.dtype == torch.bfloat16
        assert torch.equal(out, nc.to(torch.bfloat16) * 0.5), "the cast path is exactly today's"

        for bad in (0.0, float("nan"), float("inf")):
            pushed.clear()
            assert kern(q8, bad).dtype == torch.bfloat16 and pushed == [0.0], f"q_scale {bad} is declined"

    _with_knob("1", run_on)
    pushed.clear()
    assert _with_knob("0", lambda: kern(q8, 1.0)).dtype == torch.bfloat16 and pushed == [0.0]
    assert any("RADIANCE_ATTN_Q_FP8" in m for m in said), "the fp8 path must say once that it took effect"
    assert any("not contiguous" in m for m in said) and any("not e4m3" in m for m in said), \
        "every decline reason must be logged"


def test_older_lib_never_gets_fp8():
    pushed, said = [], []
    kern = _launcher_ns(None, said)["_q_kernel"]
    q8 = torch.randn(4, 2, 256).to(torch.float8_e4m3fn)
    out = _with_knob("1", lambda: kern(q8, 1.0))
    assert out.dtype == torch.bfloat16 and pushed == [], "an old libr4d must never be handed e4m3 q"
    assert any("no set_q_fp8" in m for m in said), "the decline must be logged once, with the reason"
    assert '_set_q_fp8 = getattr(r4d, "set_q_fp8", None)' in LAUNCHER.read_text()


def test_dead_fp32_path_is_gone():
    for what in ("O_QF32", "set_q_fp32", "r4d_q_fp32", "RADIANCE_ATTN_Q_FP32", "g_q_fp32"):
        assert what not in PATCH, f"{what} must be removed from the patch"
        assert what not in LAUNCHER.read_text(), f"{what} must be removed from the glue"


def test_patch_wiring():
    bits = [int(v) for v in re.findall(r"#define O_[A-Z0-9]+\s+(\d+)", ADDED)]
    assert bits.count(O_QF8) == 1 and "#define O_QF8  16777216" in ADDED, "O_QF8 is bit 24, defined once"
    assert 8388608 not in bits, "bit 23 stays unused"
    assert not re.search(r"^[+-]\s*#define P_OPT", PATCH, re.M), "the shipped prefill mask must not move"
    assert "OPTX | O_QF8" in ADDED, "the launcher must add O_QF8 to the instance it dispatches"
    assert "getenv(\"RADIANCE_ATTN_Q_FP8\")" not in ADDED and "RADIANCE_ATTN_Q_FP8" not in ADDED, \
        "libr4d must not read the format knob from the environment"
    assert "static float g_q_fp8 = 0.0f;" in ADDED, "the toggle must default off"
    assert "float r4d_q_fp8(void);" in ADDED, "the launch side needs the accessor"
    assert "const float qs = r4d_q_fp8();" in ADDED, "prefill_launch must read the toggle"
    assert 'm.def("set_q_fp8", &set_q_fp8' in ADDED, "the Python binding must exist"
    assert "float q_scale;" in ADDED, "the scale travels in R4DArgs"
    assert "__builtin_amdgcn_cvt_pk_f32_fp8" in ADDED and "r4d_bf16x2(" in ADDED
    assert "a.q_scale" in ADDED, "the prologue multiplies by the launch's q_scale"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all q-fp8 checks passed")
