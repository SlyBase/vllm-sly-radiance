#!/usr/bin/env python3
"""Print the CHANGELOG.md section of one version (the release notes of v<version>).

    ci/changelog_section.py 0.4.0        -> body of "## [0.4.0] - <date>", without the heading
    ci/changelog_section.py --check X    -> exit 1 (with a message) if the section is missing or empty
    ci/changelog_section.py --ref v0.4.0 X -> relative links made absolute (blob/<ref>/...), for the
                                            GitHub release page, where repo-relative links do not resolve

Used by ci/release_tag.sh (GitHub release body) and ci/check_consistency.py (a VERSION bump needs one).
"""
import re
import sys
from pathlib import Path

CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def section(version: str) -> str | None:
    text = CHANGELOG.read_text() if CHANGELOG.exists() else ""
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.S | re.M)
    if not m:
        return None
    body = m.group(1).strip()
    return body or None


REPO_URL = "https://github.com/SlyBase/vllm-sly-radiance"


def absolute_links(body: str, ref: str) -> str:
    """](docs/X.md#a) -> ](https://github.com/.../blob/<ref>/docs/X.md#a); URLs and anchors stay."""
    return re.sub(r"\]\((?!https?://|#|mailto:)([^)]+)\)", lambda m: f"]({REPO_URL}/blob/{ref}/{m.group(1)})", body)


def main() -> int:
    args = sys.argv[1:]
    check = "--check" in args
    args = [a for a in args if a != "--check"]
    ref = None
    if "--ref" in args:
        i = args.index("--ref")
        ref = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    body = section(args[0])
    if body is None:
        print(f"CHANGELOG.md has no (non-empty) section '## [{args[0]}]'", file=sys.stderr)
        return 1
    if not check:
        print(absolute_links(body, ref) if ref else body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
