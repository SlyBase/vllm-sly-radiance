#!/usr/bin/env python3
"""Print the patch names from the Dockerfile's `for p in ... ; do` apply loop, in order.

The loop in the assemble stage is the one place that decides which patches ship and in what
order (anchor dependencies). CI extracts it from there instead of keeping a copy.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOOP_RE = re.compile(r"for p in (.*?);\s*do", re.S)


def patch_names(dockerfile=ROOT / "Dockerfile"):
    m = LOOP_RE.search(dockerfile.read_text())
    if not m:
        raise SystemExit("FAIL: patch loop `for p in ...; do` not found in Dockerfile")
    return m.group(1).replace("\\\n", " ").split()


if __name__ == "__main__":
    sep = "\n" if "--lines" in sys.argv else " "
    print(sep.join(patch_names()))
