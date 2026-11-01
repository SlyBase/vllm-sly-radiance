#!/usr/bin/env python3
"""Cut a release: turn the CHANGELOG `## [Unreleased]` section into `## [<version>] - <date>` and set VERSION.

    ci/cut_release.py 1.10.0      -> edits CHANGELOG.md and VERSION; commit them as the release PR

Feature PRs collect their notes under [Unreleased] and leave VERSION alone; the PR made from this
script's result is the only one that moves VERSION, so it is the one that builds, gates and tags.
"""
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def cut(text: str, version: str, date: str) -> str:
    m = re.search(r"^## \[Unreleased\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.S | re.M)
    if not m or not m.group(1).strip():
        sys.exit("CHANGELOG.md has no non-empty '## [Unreleased]' section: nothing to release")
    return text[:m.start()] + f"## [{version}] - {date}\n" + text[m.start() + m.group(0).index("\n") + 1:]


def main() -> None:
    if len(sys.argv) != 2 or not re.fullmatch(r"\d+\.\d+\.\d+", sys.argv[1]):
        sys.exit(__doc__)
    ver = sys.argv[1]
    log = ROOT / "CHANGELOG.md"
    log.write_text(cut(log.read_text(), ver, datetime.date.today().isoformat()))
    (ROOT / "VERSION").write_text(ver + "\n")
    print(f"released {ver}: CHANGELOG [Unreleased] -> [{ver}], VERSION bumped")


if __name__ == "__main__":
    main()
