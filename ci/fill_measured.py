#!/usr/bin/env python3
"""Fill the CHANGELOG `### Measured` skeleton of a stack release from the night gate's report.json.

ci/resolve_stack.py --apply writes `- Night gate pending.` under `### Measured`; after a green gate
accept-night replaces that bullet with the numbers the gate measured (they become the GitHub release body).

    python3 ci/fill_measured.py --version 1.1.0 --report out/accept-.../report.json --run-url <url>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ci"))
from resolve_stack import PENDING  # noqa: E402


def bullets(report: dict, run_url: str) -> list[str]:
    meta, res = report["meta"], report["results"]
    start, spec, bb, gsm = res.get("start", {}), res.get("spec", {}), res.get("betterbench", {}), res.get("gsm8k")
    day = meta["started"].split(" ")[0]
    head = (f"Night gate green ({day}, `{meta['mode']}` mode, production launch, Swift-1.5 checkpoint, 210 W, "
            f"second start, against the baseline of the previous release): ")
    parts = []
    if start.get("kv_tokens"):
        parts.append(f"KV pool {start['kv_tokens']:,} tokens")
    if start.get("startup_s"):
        parts.append(f"startup {start['startup_s']:.0f} s")
    if spec.get("mean_tokens_per_step"):
        parts.append(f"{spec['mean_tokens_per_step']} tokens/step, acceptance {spec.get('acceptance_rate')}")
    out = [head + ", ".join(parts) + "."]
    if bb.get("aggregate_tps"):
        out.append("BetterBench aggregate t/s: " + ", ".join(f"conc {k} = {v}" for k, v in bb["aggregate_tps"].items()) + ".")
    if gsm:
        out.append(f"GSM8K 200 (cot zero-shot, greedy): {gsm['exact_match']}.")
    out.append(f"All gate checks passed ({len(report.get('checks', []))} checks); report: {run_url}")
    return [f"- {b}" for b in out]


def fill(changelog: Path, version: str, report: dict, run_url: str) -> bool:
    text = changelog.read_text()
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.S | re.M)
    if not m or f"- {PENDING}" not in m.group(0):
        return False
    section = m.group(0).replace(f"- {PENDING}", "\n".join(bullets(report, run_url)))
    changelog.write_text(text[: m.start()] + section + text[m.end():])
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--run-url", required=True)
    ap.add_argument("--changelog", default=str(ROOT / "CHANGELOG.md"))
    args = ap.parse_args()
    report = json.loads(Path(args.report).read_text())
    if report.get("verdict") != "PASS":
        print("::error::refusing to write measurements of a gate that did not pass")
        return 1
    if not fill(Path(args.changelog), args.version, report, args.run_url):
        print(f"::warning::CHANGELOG [{args.version}] has no '{PENDING}' bullet: left as it is")
    return 0


if __name__ == "__main__":
    sys.exit(main())
