#!/usr/bin/env python3
"""The measured prefill stack is what the IMAGE ships (Dockerfile ENV), and PSW is "auto".

`serve-mxfp4.sh` is a launcher; the production unit runs the image entrypoint directly, so the
defaults that matter live in the Dockerfile ENV (an explicit `-e` still wins) or in the module that
reads the knob. PSW (`RADIANCE_MXFP4_AT_PSW`) turns itself on unless the launch conflicts with it
(then it stays off and logs why); an explicit `=2` keeps refusing. The PSW decision block of
`sly/mxfp4/radiance_mxfp4.py` is executed here, so the test checks behaviour. No torch, no GPU.

Run:  python3 sly/tests/test_v6p_defaults_cpu.py
"""
import io
import os
import re
import sys
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MXFP4 = ROOT / "sly/mxfp4/radiance_mxfp4.py"

STACK = {
    "RADIANCE_MXFP4_TN4_MIN_M": "513", "RADIANCE_FUSED_NQ_REG": "1",
    "RADIANCE_MXFP4_AT_SKIP_PAD": "1", "RADIANCE_MXFP4_TILED_OPROJ": "1",
    "RADIANCE_MXFP4_TALL": "1", "RADIANCE_ATTN_Q_FP8": "1", "R4D_ATTN_FP8": "3",
    "R4D_ATTN_DB": "1", "R4D_ATTN_TILE": "16", "RADIANCE_GDN_SCAN_VARIANT": "s3",
    "RADIANCE_DFLASH_PREFILL_WINDOW_SKIP": "1", "RADIANCE_DFLASH_FC_TILED_MIN_M": "513",
}


def _image_env():
    txt = (ROOT / "Dockerfile").read_text()
    m = re.search(r"^ENV (RADIANCE_MXFP4_TN4_MIN_M=.*?)\n\n", txt, re.M | re.S)
    assert m, "the prefill-stack ENV block is gone from the Dockerfile"
    return dict(re.findall(r"(\w+)=(\S+)", m.group(1).replace("\\\n", " ")))


def test_image_env_ships_the_stack():
    assert _image_env() == STACK


def test_tall_for_in_proj_stays_off():
    # measured slower on the merged GDN in_proj: the knob is gate_up only in the kernel source
    hip = (ROOT / "sly/mxfp4/radiance_mxfp4_fp8.hip").read_text()
    assert "N == 34816" in hip


def _psw(env, wperm=True, min_m=0):
    """Run the PSW decision block of radiance_mxfp4.py; returns (AT_PSW, stderr) or the refusal."""
    src = MXFP4.read_text()
    block = src[src.index("_psw_auto = "):src.index("_psw_on = [False]")]
    ns = {"os": os, "sys": sys, "WPERM": wperm, "MIN_M": min_m}
    old = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            exec(block, ns)
    finally:
        os.environ.clear()
        os.environ.update(old)
    return ns["AT_PSW"], err.getvalue()


PROD = {"RADIANCE_MXFP4_WPERM": "1", "RADIANCE_MXFP4_DECODE_MAX_M": "128"}


def test_default_is_auto_and_on_for_the_production_launch():
    psw, log = _psw(PROD)
    assert psw == 2 and "PSW mode 2" in log
    assert re.search(r'environ\.get\("RADIANCE_MXFP4_AT_PSW", "auto"\)', MXFP4.read_text())


def test_auto_declines_on_every_refusal_trigger():
    for extra, wperm, min_m in (({"RADIANCE_MXFP4_DECODE_NT": "1"}, True, 0),
                                ({}, False, 0), ({}, True, 256),
                                ({"RADIANCE_MXFP4_R4D_DECODE_MAX_M": "8"}, True, 0),
                                ({"RADIANCE_MXFP4_DEBUG": "1"}, True, 0),
                                ({"RADIANCE_PAROQUANT": "1"}, True, 0)):
        psw, log = _psw({**PROD, **extra}, wperm, min_m)
        assert psw == 0 and "off (auto)" in log, (extra, wperm, min_m, log)


def test_explicit_values_keep_their_meaning():
    assert _psw({**PROD, "RADIANCE_MXFP4_AT_PSW": "0"})[0] == 0
    assert _psw({**PROD, "RADIANCE_MXFP4_AT_PSW": "2"})[0] == 2
    try:
        _psw({**PROD, "RADIANCE_MXFP4_AT_PSW": "2", "RADIANCE_MXFP4_DECODE_NT": "1"})
    except RuntimeError as e:
        assert "refused" in str(e)
    else:
        raise AssertionError("explicit =2 with a conflicting knob must still refuse")


TESTS = (test_image_env_ships_the_stack, test_tall_for_in_proj_stays_off,
         test_default_is_auto_and_on_for_the_production_launch,
         test_auto_declines_on_every_refusal_trigger, test_explicit_values_keep_their_meaning)

if __name__ == "__main__":
    failed = 0
    for fn in TESTS:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    sys.exit(1 if failed else print("v6p defaults tests OK"))
