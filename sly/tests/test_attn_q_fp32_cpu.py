#!/usr/bin/env python3
"""CPU tests for RADIANCE_ATTN_Q_FP32: the prefill kernel rounds the fp32 query itself.

Gemma's rms_norm leaves q fp32, and the glue used to cast 12.6 M elements to bf16 per (layer,
chunk) -- 50 MB read + 25 MB write, 54 ms over a 47k-token prefill -- only for libr4d to read bf16
again. O_QF32 (bit 24, the fp32-query instance) rounds it inside the kernel prologue instead, and
the whole claim rests on that rounding producing exactly the bits the cast would have produced:

  * the formula the kernel uses (r4d_bf16_rne, whose text is checked against the patch) against
    torch's own .to(torch.bfloat16), bit for bit, on ties, subnormals, infinities, NaNs and a large
    sample of the fp32 bit space;
  * the launcher helper that decides whether a query can go unconverted (fp32, q_scale 1, knob on)
    and the patch-text invariants that keep the bit and the knob wired to the shipped instances.

Plain torch + numpy, no vllm, no GPU, no compiler.

Run:  python3 sly/tests/test_attn_q_fp32_cpu.py     (exit 0 = pass, also collectable by pytest)
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

O_QF32 = 16777216          # bit 24


def kernel_formula():
    """The body of r4d_bf16_rne as it is written in the patch (the source of truth)."""
    m = re.search(r"uint32_t r4d_bf16_rne\(float f\) \{(.*?)\n\}", ADDED, re.S)
    assert m, "the r4d_bf16_rne helper disappeared from the patch"
    body = m.group(1)
    for token in ("0x7f800000u", "0x007fffffu", "0x7fffu", "0x0040u"):
        assert token in body, f"the rounding formula lost {token}: {body}"
    assert re.search(r"\(u \+ 0x7fffu \+ \(\(u >> 16\) & 1u\)\) >> 16", body), \
        "the rounding is no longer round-to-nearest-even on the low 16 bits"
    assert re.search(r"\(u >> 16\) \| 0x0040u", body), "NaN is no longer kept as a quiet bf16 NaN"
    return body


def rne(x):
    """The kernel's formula, transcribed: fp32 bits in, bf16 bits out (uint16, low half)."""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rne_bits = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFFFFFF) >> 16
    nan = ((u & 0x7F800000) == 0x7F800000) & ((u & 0x007FFFFF) != 0)
    return np.where(nan, ((u >> 16) | 0x0040) & 0xFFFF, rne_bits).astype(np.uint16)


def torch_bits(x):
    """torch's own fp32 -> bf16, as the bf16 bit patterns it produced."""
    return torch.as_tensor(np.asarray(x, dtype=np.float32)).to(torch.bfloat16).view(torch.uint16).numpy()


def compare(x, what):
    x = np.asarray(x, dtype=np.float32)
    want, got = torch_bits(x), rne(x)
    nan = np.isnan(x)
    if not np.array_equal(want[~nan], got[~nan]):
        bad = np.nonzero(want != got)[0][:8]
        detail = ", ".join(f"{x[i]!r}: torch {want[i]:#06x} kernel {got[i]:#06x}" for i in bad)
        raise AssertionError(f"{what}: {len(bad)} mismatches, first ones {detail}")
    if nan.any():
        # torch's CPU cast collapses EVERY NaN to 0xffff (sign and payload both dropped), while the
        # HIP cast this kernel replaces quiets the payload and keeps the sign -- so the payload is
        # not comparable here at all. What is comparable is that a NaN stays a NaN and keeps its
        # sign; the gate compares full outputs, and q is never NaN in a real step.
        assert np.all((got[nan] & 0x7F80) == 0x7F80), "a NaN input must not produce a finite bf16"
        assert np.all((got[nan] & 0x007F) != 0), "a NaN input must stay NaN, not become infinity"
        assert np.array_equal(got[nan] & 0x8000, (x[nan].view(np.uint32) >> 16) & 0x8000), \
            "a NaN must keep its sign"


def test_rounding_matches_torch_on_the_hard_cases():
    kernel_formula()
    # Every bf16-representable boundary and the exact halfway point above and below it: the tie
    # cases are what separate RNE from truncation. Built from raw bit patterns.
    mid = np.arange(0x0000, 0x7F81, dtype=np.uint32) << 16          # +0 .. +inf, bf16 grid
    ties_up = mid | 0x8000                                          # exactly halfway: rounds to even
    ties_up_odd = ties_up | 0x0001                                  # halfway plus one ULP: rounds up
    with np.errstate(invalid="ignore"):
        cases = np.concatenate([
            mid, ties_up, ties_up_odd,
            np.array([0x00000000, 0x80000000,                            # +/- zero
                      0x00000001, 0x0000FFFF, 0x00400000,                # fp32 subnormals
                      0x007FFFFF, 0x00800000, 0x33000000, 0x33000001,    # smallest normals / bf16 subnormals
                      0x7F7FFFFF, 0x7F800000, 0xFF800000,                # max finite, +/- inf
                      0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0xFFC00001,    # signalling/quiet NaNs
                      0x40490FDB, 0x402DF854, 0x3FC00000, 0xBF800001],   # pi, e, 1.5, -1+ulp
                      dtype=np.uint32),
        ]).view(np.float32)
    compare(cases, "bf16 grid, ties, subnormals, infinities and NaNs")


def test_rounding_matches_torch_on_random_values():
    g = np.random.default_rng(20261021)
    # 1e6 values the way q actually arrives: normal-ish fp32 with a wide dynamic range ...
    x = (g.standard_normal(1_000_000) * np.float32(10.0 ** g.uniform(-18, 18, 1_000_000))).astype(np.float32)
    compare(x, "1e6 scaled normals")
    # ... and 1e6 random BIT PATTERNS, which covers the subnormal and NaN tails densely.
    compare(g.integers(0, 1 << 32, 1_000_000, dtype=np.uint64).astype(np.uint32).view(np.float32),
            "1e6 random fp32 bit patterns")


def _launcher_ns(push, said):
    """`_q_fp32_ok` and `_q_kernel` with the module names they close over, no vllm import."""
    src = LAUNCHER.read_text()
    tree = ast.parse(src)
    ns = {
        "os": os,
        "torch": torch,
        "_set_q_fp32": push,
        "_logged": set(),
        "_say_once": lambda key, msg: said.append(msg),
    }
    for name in ("_q_fp32_ok", "_q_kernel"):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.get_source_segment(src, fn), str(LAUNCHER), "exec"), ns)
    return ns


def test_toggle_follows_the_pointer():
    """The kernel is told, per launch, exactly whether the pointer it gets is fp32.

    This is the whole fix: the answer used to be an environment read on both sides, and the two
    disagree on every query the glue declines -- where the kernel then reads 4 bytes per element
    out of a 2-byte buffer (host reset on a 1515-token prefill).
    """
    pushed, said = [], []
    kern = _launcher_ns(pushed.append, said)["_q_kernel"]
    q32 = torch.randn(8, 2, 256)
    env = os.environ.pop("RADIANCE_ATTN_Q_FP32", None)
    try:
        out = kern(q32, 1.0)
        assert pushed == [0] and out.dtype == torch.bfloat16, "the knob must default off"

        os.environ["RADIANCE_ATTN_Q_FP32"] = "1"
        pushed.clear()
        out = kern(q32, 1.0)
        assert pushed == [1], "contiguous fp32 q with q_scale 1 must tell the kernel fp32"
        assert out.data_ptr() == q32.data_ptr(), "the qualifying query must not be copied"

        pushed.clear()
        q16 = q32.to(torch.bfloat16)
        out = kern(q16, 1.0)
        assert pushed == [0] and out.data_ptr() == q16.data_ptr(), \
            "a bf16 q with the knob on must still be declared bf16 (the fault case)"

        pushed.clear()
        out = kern(q32.transpose(0, 1).contiguous().transpose(0, 1), 1.0)
        assert pushed == [0] and out.dtype == torch.bfloat16, "a non-contiguous q must be copied"

        pushed.clear()
        out = kern(q32, 0.5)
        assert pushed == [0] and out.dtype == torch.bfloat16, \
            "q_scale must be folded by the cast path, never by the kernel"
        assert torch.equal(out.float(), q16.float() * 0.5), "the fold must stay on the cast path"

        pushed.clear()
        out = kern(torch.zeros(4, 2, 256, dtype=torch.float8_e4m3fn), 1.0)
        assert pushed == [0] and out.dtype == torch.bfloat16, "the fp8 query path has no fp32 form"

        os.environ["RADIANCE_ATTN_Q_FP32"] = "0"
        pushed.clear()
        assert kern(q32, 1.0).dtype == torch.bfloat16 and pushed == [0]
    finally:
        os.environ.pop("RADIANCE_ATTN_Q_FP32", None)
        if env is not None:
            os.environ["RADIANCE_ATTN_Q_FP32"] = env
    assert any("RADIANCE_ATTN_Q_FP32" in m for m in said), \
        "the fp32 path must say once that it took effect"


def test_older_lib_never_gets_fp32():
    """A libr4d without set_q_fp32 cannot be told, so it is handed bf16 and told why."""
    pushed, said = [], []
    kern = _launcher_ns(None, said)["_q_kernel"]
    env = os.environ.pop("RADIANCE_ATTN_Q_FP32", None)
    os.environ["RADIANCE_ATTN_Q_FP32"] = "1"
    try:
        out = kern(torch.randn(4, 2, 256), 1.0)
    finally:
        os.environ.pop("RADIANCE_ATTN_Q_FP32", None)
        if env is not None:
            os.environ["RADIANCE_ATTN_Q_FP32"] = env
    assert out.dtype == torch.bfloat16, "an old libr4d must never be handed fp32 q"
    assert pushed == [], "there is no toggle to push"
    assert any("no set_q_fp32" in m for m in said), "the decline must be logged once, with the reason"
    src = LAUNCHER.read_text()
    assert '_set_q_fp32 = getattr(r4d, "set_q_fp32", None)' in src, \
        "the toggle must be looked up with getattr so an old libr4d loads"


def test_patch_wiring():
    bits = [int(v) for v in re.findall(r"#define O_[A-Z0-9]+\s+(\d+)", ADDED)]
    assert bits.count(O_QF32) == 1, "O_QF32 must be defined exactly once, as bit 24"
    assert 8388608 not in bits, "bit 23 stays unused (the DB bit's neighbour)"
    assert all(b & (b - 1) == 0 for b in bits if b == O_QF32)
    # P_OPT lives outside the touched hunks and must stay untouched: the shipped instance is
    # dispatched from that mask, and an unchanged mask is what keeps its code object identical.
    assert not re.search(r"^[+-]\s*#define P_OPT", PATCH, re.M), (
        "the patch must not move the shipped prefill mask")
    assert "OPTX | O_QF32" in ADDED, "the launcher must add O_QF32 to the instance it dispatches"
    # The kernel must NOT decide from the environment: the env is process-wide, the answer is
    # per query. It reads the toggle the caller pushed instead.
    assert 'getenv("RADIANCE_ATTN_Q_FP32")' not in ADDED, (
        "libr4d must not read RADIANCE_ATTN_Q_FP32 from the environment")
    assert "static int g_q_fp32 = 0;" in ADDED, "the toggle must default off"
    assert "int  r4d_q_fp32(void);" in ADDED, "the launch side needs the accessor"
    assert "const bool qf = r4d_q_fp32() != 0;" in ADDED, "prefill_launch must read the toggle"
    assert 'm.def("set_q_fp32", &set_q_fp32' in ADDED, "the Python binding must exist"
    assert "g_q_fp32 = on ? 1 : 0;" in ADDED, "set_q_fp32 must store a normalised 0/1"
    # The rounding must be applied before the folded query scale, i.e. the fp32 path must feed the
    # same bf16 dwords the bf16 path loads, not a re-derived scale.
    assert "r4d_bf16x2(f0.x, f0.y)" in ADDED and "r4d_qcvt<D, FOLDQ, F16>(l0.x, mul)" in ADDED


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all q-fp32 checks passed")
