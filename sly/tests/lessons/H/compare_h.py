#!/usr/bin/env python3
"""compare_h.py A.json B.json -- probe_h.py results of two arms: exact equality of the short texts, first
divergence of the long texts, and the prompt-NLL difference."""
import json
import sys

a, b = (json.load(open(p)) for p in sys.argv[1:3])
bad = 0
for key, must_equal in (("short", True), ("long", False)):
    for x, y in zip(a[key], b[key]):
        eq = x["text"] == y["text"]
        if eq:
            print(f"  {key:5s} words={x['words']:4d} tokens={x['prompt_tokens']:5d}: identical")
            continue
        n = next((i for i, (p, q) in enumerate(zip(x["text"], y["text"])) if p != q), min(len(x["text"]), len(y["text"])))
        print(f"  {key:5s} words={x['words']:4d} tokens={x['prompt_tokens']:5d}: DIFFERS at char {n}"
              + ("   <-- must be identical" if must_equal else "   (expected possible: other prefill kernel)"))
        bad += 1 if must_equal else 0
if a["lp"] and b["lp"]:
    d = [(p, q) for p, q in zip(a["lp"], b["lp"]) if p is not None and q is not None]
    ma, mb = (sum(p for p, _ in d) / len(d), sum(q for _, q in d) / len(d))
    dmax = max(abs(p - q) for p, q in d)
    dmean = sum(abs(p - q) for p, q in d) / len(d)
    print(f"  prompt NLL over {len(d)} tokens: A {ma:.5f}  B {mb:.5f}  delta {mb - ma:+.5f}   |diff| mean {dmean:.5f} max {dmax:.4f}")
print("SHORT-TEXT EQUALITY: " + ("PASS" if bad == 0 else f"FAIL ({bad})"))
sys.exit(1 if bad else 0)
