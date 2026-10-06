#!/usr/bin/env python3
"""cmp_greedy.py A.json B.json -- first differing character per prompt. Exit 1 if any SINGLE-stream
completion differs (the hard criterion); conc8 diffs are reported but only warn (batch composition
and arrival timing can legitimately move a greedy trajectory through verify-shape-dependent
numerics even when two arms are identical)."""
import json
import sys

a, b = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
bad = {"single": 0, "conc8": 0}
for k in ("single", "conc8"):
    for i, (x, y) in enumerate(zip(a[k], b[k])):
        if x != y:
            bad[k] += 1
            j = next((n for n, (c, d) in enumerate(zip(x, y)) if c != d), min(len(x), len(y)))
            print(f"DIFF {k}[{i}] at char {j} of {len(x)}/{len(y)}")
print(f"GREEDY single: {'IDENTICAL' if not bad['single'] else str(bad['single']) + '/8 DIFFER'}; "
      f"conc8: {'IDENTICAL' if not bad['conc8'] else str(bad['conc8']) + '/8 differ (warn only)'}")
sys.exit(1 if bad["single"] else 0)
