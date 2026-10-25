#!/usr/bin/env python3
"""Diff a fresh research-corpus checkout against the stored read ledger.

Reports, per swept subcategory, the papers in the current top-10 that are
   NEW      - never judged (id not in the ledger)
   CHANGED  - judged before, but the digest file was rewritten (corpus rebuild)
Papers that are judged and unchanged are silent, so a re-run after a corpus
update only shows the work that is actually left.

    check_corpus.py <corpus-dir> [--update] [--top N]

--update appends the NEW ids as {depth: ranking-only, verdict: unjudged}.
"""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

STATE = Path(__file__).resolve().parent / "corpus-state.json"


def digest_hash(p):
    return hashlib.sha1(p.read_bytes()).hexdigest()[:12]


def top_ids(subdir: Path, n: int):
    """First n entries of the 'Best of the best' table: [(arxiv_id, path)]."""
    txt = (subdir / "README.md").read_text()
    i = txt.find("Best of the best")
    if i < 0:
        return []
    out = []
    for _, fn in re.findall(r"^\d+\. \*\*\[([^\]]+)\]\(([^)]+\.md)\)\*\*", txt[i:], re.M)[:n]:
        p = subdir / fn
        out.append((Path(fn).name[:10], p))
    return out


def main(argv):
    corpus = Path(argv[1])
    update = "--update" in argv
    top = 10
    if "--top" in argv:
        top = int(argv[argv.index("--top") + 1])

    state = json.loads(STATE.read_text())
    entries = state["entries"]

    # --mark AID=verdict:depth   (after reading a pending paper)
    for a in argv:
        if a.startswith("--mark"):
            continue
        if "=" in a and a.split("=")[0] in entries:
            aid, val = a.split("=")
            verdict, _, depth = val.partition(":")
            entries[aid]["verdict"] = verdict
            entries[aid]["depth"] = depth or "tldr"
            entries[aid]["checked"] = "2026-10-25"
            STATE.write_text(json.dumps(state, indent=1) + "\n")
            print(f"marked {aid} = {verdict}/{entries[aid]['depth']}")
            return
    new, changed = [], []

    for sub in state["subcategories"]:
        for aid, p in top_ids(corpus / sub, top):
            if not p.exists():
                continue
            if aid not in entries:
                new.append((sub, aid, p.name))
            elif entries[aid]["digest"] != digest_hash(p):
                changed.append((sub, aid, entries[aid]["digest"], digest_hash(p)))

    commit = subprocess.run(["git", "-C", str(corpus), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"corpus {commit}  ledger {state['corpus_commit'][:8]}  "
          f"{len(entries)} registered, {sum(e['verdict'] != 'unjudged' for e in entries.values())} judged, "
          f"{len(state['subcategories'])} subcategories, top {top}")
    print(f"\nNEW {len(new)}")
    for sub, aid, fn in new:
        print(f"  {sub:45} {aid}  {fn[11:75]}")
    print(f"\nCHANGED {len(changed)}  (digest rewritten since the sweep)")
    for sub, aid, old, now in changed:
        print(f"  {sub:45} {aid}  {old} -> {now}")

    if update and (new or changed):
        for sub, aid, fn in new:
            entries[aid] = {"sub": sub, "depth": "ranking-only", "verdict": "unjudged",
                            "digest": digest_hash(corpus / sub / fn), "checked": "PENDING"}
        for sub, aid, old, now in changed:
            entries[aid]["digest"] = now
            entries[aid]["checked"] = "PENDING (digest changed)"
        state["corpus_commit"] = subprocess.run(
            ["git", "-C", str(corpus), "rev-parse", "HEAD"],
            capture_output=True, text=True).stdout.strip()
        STATE.write_text(json.dumps(state, indent=1) + "\n")
        print(f"\nledger updated: {len(entries)} entries")


if __name__ == "__main__":
    main(sys.argv)
