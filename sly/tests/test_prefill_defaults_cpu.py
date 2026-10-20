#!/usr/bin/env python3
"""Wiring test: the measured prefill stack is what `serve-mxfp4.sh` ships, every knob of it is
still overridable, and the kernel-side defaults stay off. No torch, no triton, no GPU.

The measurement (210 W, R9700, BetterBench prefill, median of 8, APC off) compared the previous
launcher defaults against this combination, on a tree that carries all of its kernels:

    CHUNK=4096  RADIANCE_MXFP4_TN4_MIN_M=513  R4D_ATTN_FP8=3
    RADIANCE_FUSED_NORM_QUANT=1  RADIANCE_FUSED_NQ_REG=1
    RADIANCE_MXFP4_AT_SKIP_PAD=1  RADIANCE_MXFP4_TILED_OPROJ=1

Two rules follow from how the kernels read their knobs, and both are pinned below:

* the default lives in the LAUNCHER (`${VAR:-value}`), not in the kernel. The kernels keep their
  built-in/env fallback at 0, so a build of this tree with no explicit env is the stock path and
  `RADIANCE_...=0 ./serve-mxfp4.sh` reproduces it. A kernel-side default would mean a rebuild to
  turn a measured lever back off, and `--offload-device-only` assembly would stop matching.
* `RADIANCE_FUSED_NQ_REG` and `RADIANCE_MXFP4_TILED_OPROJ` are read inside the fused producers, so
  they only act when the master `RADIANCE_FUSED_NORM_QUANT` is on. Shipping them without the master
  would be a default that silently does nothing.

Run:  python3 sly/tests/test_prefill_defaults_cpu.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MXFP4_SERVE = ROOT / "serve-mxfp4.sh"
README = ROOT / "README.md"
DOCKERHUB = ROOT / "DOCKERHUB.md"
HIP = ROOT / "sly/mxfp4/radiance_mxfp4_fp8.hip"
FUSED_NORM = ROOT / "sly/mxfp4/radiance_fused_norm.py"

# The measured combination: knob -> value the launcher ships.
STACK = {
    "CHUNK": "4096",
    "RADIANCE_MXFP4_TN4_MIN_M": "513",
    "R4D_ATTN_FP8": "3",
    "RADIANCE_FUSED_NORM_QUANT": "1",
    "RADIANCE_FUSED_NQ_REG": "1",
    "RADIANCE_MXFP4_AT_SKIP_PAD": "1",
    "RADIANCE_MXFP4_TILED_OPROJ": "1",
}


def _launcher_value(text, name):
    """`-e NAME="${NAME:-value}"` (docker argument) or `NAME=${NAME:-value}` (shell default) -> the
    value, or None when the launcher does not carry the knob in the overridable form."""
    m = re.search(rf'-e {name}="\$\{{{name}:-([^"}}]*)\}}"', text)
    if m:
        return m.group(1)
    m = re.search(rf"^{name}=\$\{{{name}:-([^}}]*)\}}", text, re.M)
    return m.group(1) if m else None


def test_launcher_ships_the_measured_stack():
    txt = MXFP4_SERVE.read_text()
    for name, want in STACK.items():
        got = _launcher_value(txt, name)
        assert got == want, (
            f"serve-mxfp4.sh ships {name}={got!r}, the measured prefill stack needs {want!r} "
            '(as `-e NAME="${NAME:-value}"`, so the environment still wins)')


def test_launcher_defaults_are_overridable():
    """Every knob must be a `${VAR:-value}` default, never a literal: turning the lever off has to
    stay an env var, not a source edit."""
    txt = MXFP4_SERVE.read_text()
    for name in STACK:
        assert f"${{{name}:-" in txt, (
            f"serve-mxfp4.sh sets {name} without the ${{{name}:-...}} override form")
        assert not re.search(rf'-e {name}="?[0-9]', txt), (
            f"serve-mxfp4.sh passes {name} as a literal -- the environment could not override it")


def test_chunk_default_is_the_measured_4096():
    txt = MXFP4_SERVE.read_text()
    chunk = _launcher_value(txt, "CHUNK")
    assert chunk == "4096", (
        f"serve-mxfp4.sh ships CHUNK={chunk!r}; the measured prefill stack is 4096 "
        "(~3.5 % of the KV pool against 8192, prefill +3.5..4.8 % over 2048 at 8k-64k prompts)")
    usage = re.search(r"CHUNK=(\d+)\s+prefill chunk", txt)
    assert usage, "the usage text no longer documents CHUNK"
    assert usage.group(1) == chunk, f"usage says CHUNK={usage.group(1)}, the launcher ships {chunk}"


def test_mxfp4_tn4_threshold_ships_the_measured_513():
    val = _launcher_value(MXFP4_SERVE.read_text(), "RADIANCE_MXFP4_TN4_MIN_M")
    assert val == "513", (
        f"serve-mxfp4.sh ships RADIANCE_MXFP4_TN4_MIN_M={val!r}; the measured prefill stack "
        "switches to the TN=4 tile from M = 513")


def test_attention_fp8_legs_ship_on():
    """R4D_ATTN_FP8=3 (both 8-bit legs) is part of the measured stack. The libr4d fallback stays
    0 on purpose -- that is a compiled-in default and would need a rebuild to change; the launcher
    exports the value, which is the single place it is set."""
    val = _launcher_value(MXFP4_SERVE.read_text(), "R4D_ATTN_FP8")
    assert val == "3", f"serve-mxfp4.sh ships R4D_ATTN_FP8={val!r}, the measured stack needs 3"


def test_producer_knobs_ship_with_their_master():
    """`RADIANCE_FUSED_NQ_REG` and `RADIANCE_MXFP4_TILED_OPROJ` are read by the fused producers,
    which only run while `RADIANCE_FUSED_NORM_QUANT` is on -- defaulting them without the master
    would be a no-op."""
    txt = MXFP4_SERVE.read_text()
    assert _launcher_value(txt, "RADIANCE_FUSED_NORM_QUANT") == "1", (
        "serve-mxfp4.sh does not enable RADIANCE_FUSED_NORM_QUANT: the producer knobs of the "
        "measured stack (NQ_REG, TILED_OPROJ) would be inert in the shipped launcher")
    fused = FUSED_NORM.read_text()
    assert re.search(r'^ENABLED = _flag\("RADIANCE_FUSED_NORM_QUANT", "0"\)', fused, re.M), (
        "radiance_fused_norm.py no longer gates on the master knob -- revisit this test")
    # The Python side gates TILED_OPROJ on the master explicitly; NQ_REG is read in the .hip and is
    # master-gated by its call sites (the radiance::*_quant ops, which only the fused producers use).
    assert re.search(r'= ENABLED and _flag\("RADIANCE_MXFP4_TILED_OPROJ"', fused), (
        "RADIANCE_MXFP4_TILED_OPROJ is no longer master-gated in radiance_fused_norm.py")
    assert re.search(r'^\s*if ENABLED:$', fused, re.M), (
        "radiance_fused_norm.py no longer has the ENABLED block that imports the W4A8 kernel")



def test_kernel_defaults_stay_off():
    """The kernels read the env but fall back to 0: the serving default is a launcher decision, the
    kernel default is the stock path. Unsetting the knob must reproduce the old numerics without a
    rebuild."""
    hip = HIP.read_text()
    for name, fallback in (("RADIANCE_FUSED_NQ_REG", r"return e && atoi\(e\) != 0;"),
                           ("RADIANCE_MXFP4_AT_SKIP_PAD", r"return e \? atoi\(e\) : 0;")):
        line = re.search(rf'^static int \w+ = \[\] \{{.*getenv\("{name}"\).*$', hip, re.M)
        assert line, f"no env read for {name} in radiance_mxfp4_fp8.hip"
        assert re.search(fallback, line.group(0)), (
            f"{name} no longer falls back to 0 in radiance_mxfp4_fp8.hip: {line.group(0)!r}")
    assert re.search(r'_flag\("RADIANCE_MXFP4_TILED_OPROJ", "0"\)', FUSED_NORM.read_text()), (
        "RADIANCE_MXFP4_TILED_OPROJ no longer defaults to 0 in radiance_fused_norm.py")


def test_knob_sources_are_in_this_tree():
    """The other half of the guard that used to forbid these defaults: a launcher default is only
    allowed while the code that reads the knob is in this tree."""
    hip = HIP.read_text()
    for name in ("RADIANCE_FUSED_NQ_REG", "RADIANCE_MXFP4_AT_SKIP_PAD",
                 "RADIANCE_MXFP4_TN4_MIN_M"):
        assert f'getenv("{name}")' in hip, f"{name} has no env read in radiance_mxfp4_fp8.hip"
    assert '_flag("RADIANCE_MXFP4_TILED_OPROJ"' in FUSED_NORM.read_text(), (
        "RADIANCE_MXFP4_TILED_OPROJ has no reader in radiance_fused_norm.py")


def test_docs_ship_the_same_defaults():
    row = re.search(r"\|\s*`--max-num-batched-tokens`\s*\|\s*`(\d+)`\s*\|", README.read_text())
    assert row, "README has no --max-num-batched-tokens row in the vLLM arguments table"
    assert row.group(1) == _launcher_value(MXFP4_SERVE.read_text(), "CHUNK"), (
        f"README documents {row.group(1)} as the chunk default, the launcher ships another value")
    dh = re.search(r"\|\s*`RADIANCE_MXFP4_TN4_MIN_M`\s*\|\s*`(\d+)`\s*\|", DOCKERHUB.read_text())
    assert dh, "DOCKERHUB.md has no RADIANCE_MXFP4_TN4_MIN_M row"
    assert dh.group(1) == _launcher_value(MXFP4_SERVE.read_text(), "RADIANCE_MXFP4_TN4_MIN_M"), (
        f"DOCKERHUB.md documents TN4_MIN_M={dh.group(1)}, the launcher ships another value")


TESTS = (test_launcher_ships_the_measured_stack,
         test_launcher_defaults_are_overridable,
         test_chunk_default_is_the_measured_4096,
         test_mxfp4_tn4_threshold_ships_the_measured_513,
         test_attention_fp8_legs_ship_on,
         test_producer_knobs_ship_with_their_master,
         test_kernel_defaults_stay_off,
         test_knob_sources_are_in_this_tree,
         test_docs_ship_the_same_defaults)


if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    if failed:
        sys.exit(1)
    print("prefill defaults wiring tests OK")
