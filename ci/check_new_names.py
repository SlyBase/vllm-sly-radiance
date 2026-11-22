#!/usr/bin/env python3
"""Fail when a patch introduces a name its function cannot resolve (latent NameError).

An anchor can survive an upstream refactor while a local the hunk reads is gone: the dry run applies
it cleanly and the image dies only when the code runs. For every .py file that differs between the
pristine tree and the patched tree, list each function's implicit globals that are neither module
names nor builtins (symtable, stdlib only), and report the ones the pristine file did not have.
Files with a star import are skipped (their module names are unknowable statically).

Usage: ci/check_new_names.py PRISTINE_DIR PATCHED_DIR
"""
import builtins
import filecmp
import symtable
import sys
from pathlib import Path


def _undefined(src):
    top = symtable.symtable(src, "<patched>", "exec")
    if "*" in src and "import *" in src:
        return set()
    module = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported()}
    known = module | set(dir(builtins)) | {"__file__", "__name__", "__spec__", "__builtins__"}
    out = set()

    def walk(t):
        for s in t.get_symbols():
            if t.get_type() == "function" and s.is_referenced() and s.is_global() \
                    and s.get_name() not in known:
                out.add((t.get_name(), s.get_name()))
        for c in t.get_children():
            walk(c)

    walk(top)
    return out


def new_undefined(pristine, patched):
    return _undefined(patched) - _undefined(pristine)


def check_trees(src, dst):
    found = []
    for p in sorted(Path(dst).rglob("*.py")):
        rel = p.relative_to(dst)
        q = Path(src) / rel
        if not q.is_file() or filecmp.cmp(p, q, shallow=False):
            continue
        for fn, name in sorted(new_undefined(q.read_text(), p.read_text())):
            found.append((str(rel), fn, name))
    return found


if __name__ == "__main__":
    bad = check_trees(sys.argv[1], sys.argv[2])
    for rel, fn, name in bad:
        print(f"FAIL: {rel}: {fn}() reads undefined name '{name}' (introduced by a patch)")
    print(f"new-names check: {len(bad)} finding(s)")
    sys.exit(1 if bad else 0)
