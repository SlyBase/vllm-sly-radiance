#!/usr/bin/env python3
"""Aiter contract gate: sly/radiance_attn_decode.py must survive the pinned aiter release.

aiter (AMD's inference toolkit) is built from source and declares its compatibility nowhere
machine-readable -- there is no PyPI metadata, no contract file. Its sanctioned stand for this
image is what vLLM itself pins per release (docker/Dockerfile.rocm_base: AITER_BRANCH); this
check therefore has two parts:

  1. CONTRACT (fail): the aiter internals radiance_attn_decode wraps must still exist with a
     compatible surface, verified by AST against the aiter source tree at the pinned tag
     (the one ci/patch_dryrun.sh already checked out -- no GPU, no import, no wheel):
       * _UAParams must carry every field the wrapper reads (surface self-maintaining: it is
         re-derived from radiance_attn_decode.py itself on every run)
       * use_2d_kernel(params, ...) -- first arg positional, extras default (the wrapper calls
         orig_2d(params))
       * get_unified_attention_config(op, params, ...) -- op+params positional, extras default
     A renamed/moved field or signature is exactly what a bi-weekly aiter release can introduce
     between two Renovate bumps -- invisible to the patch anchors, caught here before the merge.
  2. DRIFT (warn): the pinned AITER_VERSION vs vLLM v<VLLM_VERSION>'s sanctioned AITER_BRANCH.
     We deliberately run above the sanctioned stand (gfx1201 from-source, see Dockerfile head
     comment), so this is a visibility line, not a gate.

Usage: python3 ci/check_aiter_contract.py --aiter-src DIR [--repo ROOT]
       --aiter-src = a dir whose aiter/ subdirectory is the package (the dry-run's sparse tree)

Exit 1 with FAIL lines on a contract break; drift and degraded lookups only ever warn.
"""
import argparse
import ast
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_pins import read_pins  # noqa: E402

# attribute reads that are dict methods, not _UAParams fields (the wrapper keeps small dicts
# under the same variable names)
_DICT_METHODS = {"setdefault", "get", "keys", "values", "items", "copy", "update", "pop"}

failures = []


def fail(msg):
    failures.append(msg)
    print(f"::error::{msg}")


def warn(msg):
    print(f"::warning::{msg}")


def find_defining_file(aiter_pkg, marker):
    """Locate the module defining `marker` (marker = source snippet, e.g. 'class _UAParams').

    Known paths first; a bounded scan of the triton ops tree as fallback so a module MOVE is
    tolerated but a disappearance is not."""
    for cand in (
        aiter_pkg / "ops/triton/attention/unified_attention.py",
        aiter_pkg / "ops/triton/utils/unified_attention_utils.py",
    ):
        if cand.is_file() and marker in cand.read_text():
            return cand
    for py in sorted((aiter_pkg / "ops/triton").rglob("*.py"))[:400]:
        try:
            if marker in py.read_text():
                return py
        except OSError:
            continue
    return None


def named_tuple_fields(src):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        fail(f"aiter unified_attention.py no longer parses ({e.msg} line {e.lineno}) -- cannot verify _UAParams")
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "_UAParams":
            fields = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.append(stmt.target.id)
            return fields
    return None


def func_args(src, name):
    """(positional names, number of params with a default) for def <name>."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        fail(f"aiter module for {name}() no longer parses ({e.msg} line {e.lineno}) -- cannot verify signature")
        return None, 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            args = [a.arg for a in node.args.args]
            n_defaults = len(node.args.defaults)
            return args, n_defaults
    return None, 0


def wrapper_read_surface(wrapper_py):
    """Attributes the wrapper reads off its _UAParams instance (self-maintaining contract)."""
    tree = ast.parse(Path(wrapper_py).read_text())
    reads = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id in ("params", "p", "pp"):
                reads.add(node.attr)
    return reads


def check_contract(aiter_src, repo_root):
    aiter_pkg = aiter_src / "aiter"
    if not aiter_pkg.is_dir():
        aiter_pkg = aiter_src  # also accept a dir that IS the package
    if not (aiter_pkg / "ops/triton").is_dir():
        fail(f"aiter package layout not found under {aiter_src} (ops/triton missing)")
        return

    ua_file = find_defining_file(aiter_pkg, "class _UAParams")
    cfg_file = find_defining_file(aiter_pkg, "def get_unified_attention_config")
    if ua_file is None:
        fail("aiter: module defining _UAParams not found (moved or removed?)")
    if cfg_file is None:
        fail("aiter: module defining get_unified_attention_config not found (moved or removed?)")

    ua_src = ua_file.read_text() if ua_file else ""
    cfg_src = cfg_file.read_text() if cfg_file else ""

    fields = named_tuple_fields(ua_src)
    if fields is None:
        fail(f"aiter {ua_file if ua_file else '?'}: class _UAParams not parseable")
    else:
        surface = wrapper_read_surface(repo_root / "sly/radiance_attn_decode.py")
        missing = sorted(f for f in surface - _DICT_METHODS if f not in fields)
        for f in missing:
            fail(f"_UAParams no longer has '{f}' (read by radiance_attn_decode) -- aiter renamed or dropped the field")

    for fname, src, first, extras_must_default in (
        ("use_2d_kernel", ua_src, "params", True),
        ("get_unified_attention_config", cfg_src, None, True),
    ):
        args, n_defaults = func_args(src, fname)
        if args is None:
            fail(f"aiter: {fname}() not found (removed or renamed?)")
            continue
        if first and args[0] != first:
            fail(f"aiter {fname}(): first positional is '{args[0]}', the wrapper calls it with a _UAParams instance")
        if extras_must_default and n_defaults < max(0, len(args) - (1 if first else 2)):
            fail(f"aiter {fname}({', '.join(args)}): a new required argument -- the wrapper's positional call breaks")

    if not failures:
        print(f"aiter contract OK: _UAParams {len(fields) if fields else 0} fields, wrapper surface covered; use_2d_kernel/get_unified_attention_config signatures compatible")


def check_drift(repo_root):
    pins = read_pins(repo_root / "Dockerfile")
    aiter_pin, vllm_pin = pins.get("AITER_VERSION"), pins.get("VLLM_VERSION")
    if not (aiter_pin and vllm_pin):
        warn("drift check skipped: AITER_VERSION/VLLM_VERSION pins missing")
        return
    try:
        req = urllib.request.Request(
            f"https://raw.githubusercontent.com/vllm-project/vllm/v{vllm_pin}/docker/Dockerfile.rocm_base",
            headers={"User-Agent": "hermes-ci"})
        base = urllib.request.urlopen(req, timeout=30).read().decode()
    except Exception as e:
        warn(f"drift check degraded (could not fetch vLLM v{vllm_pin} rocm_base: {type(e).__name__})")
        return
    m = re.search(r'ARG\s+AITER_BRANCH="([^"]+)"', base)
    if not m:
        warn(f"drift check degraded: no AITER_BRANCH in vLLM v{vllm_pin} rocm_base")
        return
    sanctioned = m.group(1).lstrip("v")
    if aiter_pin != sanctioned:
        warn(f"aiter {aiter_pin} is beyond vLLM v{vllm_pin}'s sanctioned AITER_BRANCH v{sanctioned} "
             f"(from-source on gfx1201 is the documented reason; the GPU accept gate is the proof)")
    else:
        print(f"aiter drift OK: {aiter_pin} == vLLM v{vllm_pin}'s sanctioned AITER_BRANCH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aiter-src", required=True, help="dir holding the aiter source tree at the pinned tag")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent), help="repo root (for pins + wrapper)")
    a = ap.parse_args()
    t0 = time.time()
    check_contract(Path(a.aiter_src), Path(a.repo))
    check_drift(Path(a.repo))
    if failures:
        print(f"::error::aiter contract: {len(failures)} finding(s)")
        sys.exit(1)
    print(f"aiter contract gate OK ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
