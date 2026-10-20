#!/usr/bin/env python3
"""Wiring test: the measured prefill stack is what the launcher ships, and a knob is only ever
defaulted on when its implementation is in this tree. No torch, no triton, no GPU.

The measurement (210 W, R9700, BetterBench prefill, median of 8, APC off) compared the stock
launcher against a combination of five env knobs plus a 4096 prefill chunk. Three of those five
knobs (register-resident fused-norm producers, SKIPPAD in the A-tiled GEMM, tiled out_proj) live
only on the experiment branches and are NOT in this tree -- an `-e NAME=1` default for them would
be inert, i.e. a shipped default that silently does nothing until an image is built from one of
those branches. The checks below pin down both halves of that: the two knobs that do exist ship
the measured value, and the three that do not are never defaulted here.

Run:  python3 sly/tests/test_prefill_defaults_cpu.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MXFP4_SERVE = ROOT / "serve-mxfp4.sh"
SERVES = tuple(sorted(ROOT.glob("serve*.sh"))) + tuple(sorted(ROOT.glob("run_*.sh")))
README = ROOT / "README.md"
DOCKERHUB = ROOT / "DOCKERHUB.md"

# Knobs of the measured combination whose kernels are on the experiment branches only. They must
# not appear in a launcher until those kernels are merged, or the default is a no-op.
NIGHT_ONLY = ("RADIANCE_FUSED_NQ_REG", "RADIANCE_MXFP4_AT_SKIP_PAD", "RADIANCE_MXFP4_TILED_OPROJ")


def _launcher_default(text, name):
    """`-e NAME="${NAME:-value}"` -> value, or `NAME=${NAME:-value}` -> value."""
    m = re.search(rf'{name}="?\$\{{{name}:-([^"}}]*)\}}', text)
    if m:
        return m.group(1)
    m = re.search(rf"^{name}=\$\{{{name}:-([^}}]*)\}}", text, re.M)
    return m.group(1) if m else None


def test_chunk_default_is_the_measured_4096():
    txt = MXFP4_SERVE.read_text()
    chunk = _launcher_default(txt, "CHUNK")
    assert chunk == "4096", (
        f"serve-mxfp4.sh ships CHUNK={chunk!r}; the measured prefill stack is 4096 "
        "(~3.5 % of the KV pool against 8192, prefill +3.5..4.8 % over 2048 at 8k-64k prompts)")
    usage = re.search(r"CHUNK=(\d+)\s+prefill chunk", txt)
    assert usage, "the usage text no longer documents CHUNK"
    assert usage.group(1) == chunk, f"usage says CHUNK={usage.group(1)}, the launcher ships {chunk}"


def test_mxfp4_tn4_threshold_ships_the_measured_513():
    txt = MXFP4_SERVE.read_text()
    val = _launcher_default(txt, "RADIANCE_MXFP4_TN4_MIN_M")
    assert val == "513", (
        f"serve-mxfp4.sh ships RADIANCE_MXFP4_TN4_MIN_M={val!r}; the measured prefill stack "
        "switches to the TN=4 tile from M = 513")


def test_attention_fp8_legs_ship_on():
    """R4D_ATTN_FP8=3 (both 8-bit legs) is part of the measured stack. The libr4d fallback stays
    0 on purpose -- that is a compiled-in default and would need a rebuild to change; the launcher
    exports the value, which is the single place it is set."""
    val = _launcher_default(MXFP4_SERVE.read_text(), "R4D_ATTN_FP8")
    assert val == "3", f"serve-mxfp4.sh ships R4D_ATTN_FP8={val!r}, the measured stack needs 3"


def test_no_default_for_a_knob_without_kernel_support():
    """A default for a knob whose kernel is not in this tree is a lie the log will not show."""
    for serve in SERVES:
        txt = serve.read_text()
        for name in NIGHT_ONLY:
            assert name not in txt, (
                f"{serve.name} passes {name}, but its kernel is not in this tree (it lives on the "
                "experiment branches): the env knob would be inert in an image built here")
    for name in NIGHT_ONLY:
        hits = [p for p in ROOT.rglob("*.hip") if name in p.read_text(errors="ignore")]
        assert not hits, f"{name} is referenced by {hits} -- update this test when the kernel lands"


def test_docs_ship_the_same_defaults():
    row = re.search(r"\|\s*`--max-num-batched-tokens`\s*\|\s*`(\d+)`\s*\|", README.read_text())
    assert row, "README has no --max-num-batched-tokens row in the vLLM arguments table"
    assert row.group(1) == _launcher_default(MXFP4_SERVE.read_text(), "CHUNK"), (
        f"README documents {row.group(1)} as the chunk default, the launcher ships another value")
    dh = re.search(r"\|\s*`RADIANCE_MXFP4_TN4_MIN_M`\s*\|\s*`(\d+)`\s*\|", DOCKERHUB.read_text())
    assert dh, "DOCKERHUB.md has no RADIANCE_MXFP4_TN4_MIN_M row"
    assert dh.group(1) == _launcher_default(MXFP4_SERVE.read_text(), "RADIANCE_MXFP4_TN4_MIN_M"), (
        f"DOCKERHUB.md documents TN4_MIN_M={dh.group(1)}, the launcher ships another value")


if __name__ == "__main__":
    failed = 0
    for fn in (test_chunk_default_is_the_measured_4096,
               test_mxfp4_tn4_threshold_ships_the_measured_513,
               test_attention_fp8_legs_ship_on,
               test_no_default_for_a_knob_without_kernel_support,
               test_docs_ship_the_same_defaults):
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    if failed:
        sys.exit(1)
    print("prefill defaults wiring tests OK")
