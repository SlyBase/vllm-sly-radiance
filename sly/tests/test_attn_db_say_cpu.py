#!/usr/bin/env python3
"""Pins the one-shot stderr line that proves R4D_ATTN_DB / R4D_ATTN_TILE reached the prefill kernel.

The knobs are read in libr4d's prefill launcher, so a served run could not tell from its own logs
whether a measurement arm actually ran the O_DB instance or the shipped tile-48 one. The Python
launcher writes one line per process when -- and only when -- a non-default pair is in force, and
that line has to name the instance and the fp8 mode. Silent on the default path, exactly as 1.6.0.

The launcher imports torch and vLLM at module level, so the helpers are lifted out of its AST and
run against a scratch namespace: no torch, no GPU, no image.

Run:  python3 sly/tests/test_attn_db_say_cpu.py
"""
import ast
import contextlib
import io
import os
import sys
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[2] / "radiance_r4d_hybrid_attn.py"
HELPERS = ("_say", "_say_once", "_o_db_note", "_o_db_say")


def _namespace():
    """The stderr helpers of the launcher, exec'd without importing it."""
    src = LAUNCHER.read_text()
    tree = ast.parse(src)
    chunks = [seg for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in HELPERS
              for seg in (ast.get_source_segment(src, n),) if seg]
    assert len(chunks) == len(HELPERS), f"launcher lost one of {HELPERS}"
    ns = {"os": os, "sys": sys, "_logged": set()}
    exec(compile("\n\n".join(chunks), str(LAUNCHER), "exec"), ns)
    return ns


def _run(ns, env):
    """(line count, text) of one _o_db_say call with the given attention env."""
    keys = ("R4D_ATTN_TILE", "R4D_ATTN_DB", "R4D_ATTN_FP8")
    saved = {k: os.environ.pop(k, None) for k in keys}
    os.environ.update({k: v for k, v in env.items() if isinstance(v, str)})
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            ns["_o_db_say"]("r4d_attn_prefill_h256_gqa6", os.environ.get("R4D_ATTN_FP8", "0"))
    finally:
        for k in keys:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]
    lines = err.getvalue().splitlines()
    return len(lines), (lines[0] if lines else "")


def test_unset_is_silent():
    """Default-off silence: an unset pair must not add a byte to the log."""
    assert _run(_namespace(), {}) == (0, "")


def test_explicit_defaults_are_silent():
    assert _run(_namespace(), {"R4D_ATTN_TILE": "48", "R4D_ATTN_DB": "0"}) == (0, "")


def test_db_prints_one_line_naming_instance_and_fp8():
    n, line = _run(_namespace(), {"R4D_ATTN_DB": "1", "R4D_ATTN_FP8": "3"})
    assert n == 1, n
    for want in ("r4d_attn_prefill_h256_gqa6", "tile 48", "O_DB on", "fp8 legs 3"):
        assert want in line, (want, line)
    assert "inert" not in line, line


def test_tile_alone_prints():
    n, line = _run(_namespace(), {"R4D_ATTN_TILE": "16", "R4D_ATTN_FP8": "3"})
    assert n == 1 and "tile 16" in line and "O_DB off" in line, line


def test_non_fp8_legs_says_inert():
    """Only fp8mode 3 acts on the pair; claiming otherwise would be a false log line."""
    n, line = _run(_namespace(), {"R4D_ATTN_DB": "1", "R4D_ATTN_FP8": "0"})
    assert n == 1 and "inert" in line and "fp8 legs 0" in line, line


def test_called_per_layer_still_one_line():
    """The impl's forward runs once per layer: three calls, one line."""
    ns = _namespace()
    assert _run(ns, {"R4D_ATTN_DB": "1", "R4D_ATTN_FP8": "3"})[0] == 1
    assert _run(ns, {"R4D_ATTN_DB": "1", "R4D_ATTN_FP8": "3"})[0] == 0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("ok")
