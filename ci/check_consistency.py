#!/usr/bin/env python3
"""Consistency checks between the Dockerfile's patch loop, the patch files and the docs.

  1. every name in the Dockerfile loop is an existing `<name>.py`
  2. every `patch_*.py` / `sly/patch_*.py` in the tree is in the loop or listed (with a reason)
     in ci/unused_patches.txt -- no patch may silently fall out of the image
  3. every `sly/patch_*.py` is documented in sly/README.md
  4. ci/unused_patches.txt and ci/patch_dryrun_skip.txt only name files / loop entries that exist
  5. (--base <ref>) a change to anything that ends up in the image bumps VERSION

Exit 1 with a `FAIL` line per finding; `::error::` annotations for GitHub Actions.
"""
import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ci"))
from patch_list import patch_names  # noqa: E402

# paths whose change does not alter the image and therefore needs no VERSION bump
NO_BUMP_PREFIXES = (".github/", "ci/", "docs/")
NO_BUMP_FILES = {".gitignore", ".dockerignore", ".hadolint.yaml", "renovate.json", "Makefile",
                 "docker-compose.yml", "DOCKERHUB.md", "LICENSE"}
NO_BUMP_SUFFIXES = (".md",)
# Offline measurement / check tools under sly/: COPYed into the assemble stage with the rest of
# sly/, but never executed by a build step and not part of the final image (only /opt/vllm is).
NO_BUMP_GLOBS = ("sly/bench_*.py", "sly/check_*.py", "sly/mxfp4/bench_*.py", "sly/mxfp4/check_*.py")

failures = []


def fail(msg):
    failures.append(msg)
    print(f"::error::{msg}")


def read_list(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, reason = line.partition(" ")
        out[name] = reason.strip()
    return out


def check_patches():
    loop = patch_names()
    for name in loop:
        if not (ROOT / f"{name}.py").is_file():
            fail(f"Dockerfile loop names {name} but {name}.py does not exist")
    unused = read_list(ROOT / "ci" / "unused_patches.txt")
    for path, reason in unused.items():
        if not (ROOT / path).is_file():
            fail(f"ci/unused_patches.txt lists {path} which does not exist")
        if not reason:
            fail(f"ci/unused_patches.txt: {path} has no reason")
        if path.removesuffix(".py") in loop:
            fail(f"ci/unused_patches.txt lists {path} but it IS in the Dockerfile loop")
    tree = sorted(p.relative_to(ROOT).as_posix()
                  for p in list(ROOT.glob("patch_*.py")) + list(ROOT.glob("sly/patch_*.py")))
    for path in tree:
        if path.removesuffix(".py") not in loop and path not in unused:
            fail(f"{path} is neither in the Dockerfile loop nor listed in ci/unused_patches.txt")
    skip = read_list(ROOT / "ci" / "patch_dryrun_skip.txt")
    for name, reason in skip.items():
        if name not in loop:
            fail(f"ci/patch_dryrun_skip.txt: {name} is not in the Dockerfile loop")
        if not reason:
            fail(f"ci/patch_dryrun_skip.txt: {name} has no reason")
    readme = (ROOT / "sly" / "README.md").read_text()
    for path in tree:
        if path.startswith("sly/") and Path(path).name not in readme:
            fail(f"{path} is not mentioned in sly/README.md")
    print(f"patch loop: {len(loop)} entries, {len(tree)} patch files, {len(unused)} unused, {len(skip)} dry-run skips")


def check_version_bump(base):
    files = subprocess.run(["git", "diff", "--name-only", f"{base}...HEAD"], cwd=ROOT,
                           check=True, capture_output=True, text=True).stdout.split()
    image_files = [f for f in files
                   if not f.startswith(NO_BUMP_PREFIXES) and f not in NO_BUMP_FILES
                   and not f.endswith(NO_BUMP_SUFFIXES)
                   and not any(fnmatch.fnmatch(f, g) for g in NO_BUMP_GLOBS)]
    if not image_files:
        print("version bump: no image-relevant change")
        return
    old = subprocess.run(["git", "show", f"{base}:VERSION"], cwd=ROOT, capture_output=True,
                         text=True).stdout.strip()
    new = (ROOT / "VERSION").read_text().strip()
    if old == new:
        fail(f"VERSION is still {new} although image-relevant files changed: "
             + ", ".join(image_files[:8]) + (" ..." if len(image_files) > 8 else ""))
    else:
        print(f"version bump: {old} -> {new} ({len(image_files)} image-relevant files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", help="git ref of the PR base; enables the VERSION-bump check")
    args = ap.parse_args()
    check_patches()
    if args.base:
        check_version_bump(args.base)
    if failures:
        print(f"FAIL: {len(failures)} finding(s)")
        sys.exit(1)
    print("consistency OK")


if __name__ == "__main__":
    main()
