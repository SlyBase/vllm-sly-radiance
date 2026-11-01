#!/usr/bin/env python3
"""CPU test: `serve-mxfp4.sh` ships async scheduling OFF by default, and every state is reachable.

Measured at 300 W on one R9700, same image, prefix cache off, chunk 4096, `--max-model-len 65536`,
8 timed runs per prefill depth, 32 requests per decode level, two arms whose only difference is
the flag (N1 = no flag, so the dflash drafter auto-enables async scheduling; N2 = the launcher
default):

    prefill 2K/8K/16K/32K/64K tok/s   N1 3544.1/3759.4/3550.4/3449.8/3199.2
                                      N2 3789.5/3803.3/3704.0/3487.3/3170.6
    decode weighted / c1 / c2 / c4    N1 160.0/127.6/239.0/330.5
                                      N2 159.8/129.0/210.9/340.3

So the default is `--no-async-scheduling` (`RADIANCE_NO_ASYNC_SCHED=1`, appended as a flag, not an
env var passed into the container) and it stays overridable in both directions: `=0` drops the flag
entirely (dflash then auto-enables async scheduling -- the N1 arm), `ASYNC=1` forces
`--async-scheduling`. The decision block is executed here in bash so the test checks behaviour, not
just text. No torch, no triton, no GPU.

Run:  python3 sly/tests/test_launcher_async_sched_cpu.py
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MXFP4_SERVE = ROOT / "serve-mxfp4.sh"
SERVE = ROOT / "serve.sh"


def _async_block():
    """The defaults as written in the launcher, from `ASYNC=${ASYNC:-0}` through its closing `fi`."""
    txt = MXFP4_SERVE.read_text()
    lines = txt.splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if l.startswith("ASYNC=${ASYNC:-0}")), None)
    assert start is not None, "serve-mxfp4.sh lost its ASYNC default"
    for end in range(start + 1, len(lines)):
        stripped = lines[end].rstrip("\n").strip()
        if stripped == "fi" or stripped.endswith("fi"):
            return txt, "".join(lines[start:end + 1])
    raise AssertionError("serve-mxfp4.sh lost the ASYNC_FLAG decision block (`fi`)")


def _decide(**env):
    """Run the launcher's decision block in bash with a controlled environment."""
    _, block = _async_block()
    script = "set -euo pipefail\n" + block + 'printf "%s|%s\\n" "${ASYNC_FLAG}" "${UNPAD}"\n'
    clean = {k: v for k, v in os.environ.items()
             if k not in ("ASYNC", "RADIANCE_NO_ASYNC_SCHED")}
    clean.update(env)
    r = subprocess.run(["bash", "-c", script], env=clean, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    flag, unpad = r.stdout.strip().split("|")
    return flag, unpad == "true"


def test_default_is_no_async_scheduling():
    """No environment at all: the shipping default appends --no-async-scheduling."""
    flag, unpad = _decide()
    assert flag == "--no-async-scheduling", flag
    # vLLM refuses async scheduling together with disable_padded_drafter_batch, so sync ships unpadded
    assert unpad is True


def test_knob_is_explicit_and_defaults_on():
    txt, _ = _async_block()
    assert "RADIANCE_NO_ASYNC_SCHED:-1" in txt, \
        "serve-mxfp4.sh must carry RADIANCE_NO_ASYNC_SCHED with a default of 1"
    assert _decide(RADIANCE_NO_ASYNC_SCHED="1") == ("--no-async-scheduling", True)


def test_knob_off_drops_the_flag_for_the_dflash_auto_arm():
    """=0 leaves the flag off, which is how the async (N1) arm is reproduced without editing."""
    flag, unpad = _decide(RADIANCE_NO_ASYNC_SCHED="0")
    assert flag == "", f"RADIANCE_NO_ASYNC_SCHED=0 must not pass any scheduling flag, got {flag!r}"
    assert unpad is False, "async scheduling cannot ship with padded drafter batches disabled"


def test_async_override_wins():
    flag, unpad = _decide(ASYNC="1")
    assert flag == "--async-scheduling", flag
    assert unpad is False
    # and it wins over the default knob, not the other way round
    assert _decide(ASYNC="1", RADIANCE_NO_ASYNC_SCHED="1") == ("--async-scheduling", False)


def test_decision_reaches_the_serve_command():
    txt = MXFP4_SERVE.read_text()
    assert re.search(r"^\s*\$ASYNC_FLAG \$EXTRA \\$", txt, re.M), \
        "the ASYNC_FLAG decision must be interpolated into the vllm serve command"


def test_serve_sh_stays_consistent():
    """serve.sh is the other production launcher and already defaults ASYNC=0; keep it that way."""
    txt = SERVE.read_text()
    assert "ASYNC=${ASYNC:-0}" in txt
    assert "--no-async-scheduling" in txt


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("launcher async scheduling tests OK")
