#!/usr/bin/env python3
"""CPU test: `DET_COMPILE=1` pins Inductor to deterministic mode in `serve-mxfp4.sh`, default off.

Why (issue #189): every fresh compile re-times its Triton reductions and keeps the fastest config
(vllm#58899, open). Different configs sum in a different fp32 order, so two fresh-compile starts can
serve slightly different numerics -- noise in A/B arms and in disk-tier KV rows reused across
restarts. `"deterministic": true` in `inductor_compile_config` skips the on-device benchmarking
(torch 2.13 `torch/_inductor/config.py`), so every compile picks the same config.

Opt-in until a GPU A/B also clears decode ms/step and drafter acceptance (only prefill is measured).
The compilation-config block is executed here in bash, so the test checks the emitted flag.
No torch, no GPU.

Run:  python3 sly/tests/test_launcher_deterministic_compile_cpu.py
"""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MXFP4_SERVE = ROOT / "serve-mxfp4.sh"


def _cc_block():
    """From `CC_ITEMS=""` through the `fi` that appends --compilation-config to EXTRA."""
    lines = MXFP4_SERVE.read_text().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines) if l.strip() == 'CC_ITEMS=""')
    add = next(i for i in range(start, len(lines)) if "--compilation-config {$CC_ITEMS}" in lines[i])
    return "".join(lines[start:add + 2])


def _compilation_config(**env):
    """Run the block; return the parsed --compilation-config JSON (None if the flag is absent)."""
    script = ("set -euo pipefail\nEXTRA=''\nCAPTURE_SIZES=none\nNQF=0\n" + _cc_block()
              + 'printf "%s" "$EXTRA"\n')
    clean = {k: v for k, v in os.environ.items() if k not in ("DET_COMPILE", "COOP_RED", "COMPILE_SIZES")}
    clean.update(env)
    r = subprocess.run(["bash", "-c", script], env=clean, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    extra = r.stdout.strip()
    if not extra:
        return None
    flag, _, payload = extra.partition(" ")
    assert flag == "--compilation-config", extra

    def no_dupes(pairs):
        keys = [k for k, _ in pairs]
        assert len(keys) == len(set(keys)), f"duplicate keys (the last silently wins): {keys}"
        return dict(pairs)

    return json.loads(payload, object_pairs_hook=no_dupes)


def test_default_leaves_the_command_unchanged():
    assert _compilation_config() is None


def test_det_compile_pins_inductor_deterministic():
    cc = _compilation_config(DET_COMPILE="1")
    assert cc == {"inductor_compile_config": {"deterministic": True}}, cc


def test_det_compile_merges_with_cooperative_reductions():
    cc = _compilation_config(DET_COMPILE="1", COOP_RED="1")
    assert cc == {"inductor_compile_config": {"triton.cooperative_reductions": True,
                                              "deterministic": True}}, cc


def test_cooperative_reductions_alone_is_unchanged():
    cc = _compilation_config(COOP_RED="1")
    assert cc == {"inductor_compile_config": {"triton.cooperative_reductions": True}}, cc


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("launcher deterministic compile tests OK")
