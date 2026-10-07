#!/usr/bin/env python3
"""accept-night: pick the stack PR the night gate should test.

Candidates are open PRs from this repository's own `renovate/stack-*` branches with the label `stack`.
A PR is picked only when
  * it does not conflict with main,
  * its CI is fully green for the CURRENT head sha: the aggregate check `ci` succeeded and the image
    build (build.yml job `image`, which leaves vllm-sly-radiance:<VERSION>-rocm<mm> in the local docker
    of LXC 2408 -- the image the gate then runs) succeeded for that sha, and no commit status is failing,
  * the gate has not already failed for exactly this head sha (comment marker FAILED_MARKER): a new
    sha (Renovate rebase, a Hermes push) is a new candidate, the same sha is not retried automatically.
The lowest PR number wins; one PR per night (there is one GPU).

Without an eligible stack PR, the second candidate is an UNRELEASED main: main's VERSION has no tag
v<VERSION> yet (a version merged by hand, e.g. a stack PR merged before the night gate got to it). It is
picked when `ci` is green on main's head, the newest main commit that ran the image build (build.yml runs
on main only when VERSION changes) built it successfully with the same VERSION, and the gate has not
already failed for main's head (marker in a commit comment). The output then has "kind": "main" and no
PR number; the night gate releases main's head on green instead of merging a PR.

    python3 ci/night_pick.py --repo SlyBase/vllm-sly-radiance            # prints JSON, {} when nothing
    python3 ci/night_pick.py --repo ... --pr 107                         # only that PR (workflow_dispatch)
    python3 ci/night_pick.py --repo ... --pr main                        # only the unreleased main

Needs the `gh` CLI with a token (GH_TOKEN).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys

FAILED_MARKER = "<!-- accept-night:failed sha={sha} -->"
OK_CONCLUSIONS = {"success", "skipped", "neutral"}


def gh_json(*args: str):
    out = subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout
    return json.loads(out) if out.strip() else None


def judge(pr: dict, check_runs: list[dict], statuses: list[dict], comments: list[str]) -> str | None:
    """None = eligible, else the reason it is not. Pure function (unit-tested)."""
    if pr.get("isCrossRepository"):
        return "from a fork"
    ref = pr.get("headRefName", "")
    if not ref.startswith("renovate/stack-"):
        return f"branch {ref} is not renovate/stack-*"
    if "stack" not in {lab["name"] for lab in pr.get("labels", [])}:
        return "no `stack` label"
    if pr.get("mergeable") == "CONFLICTING":
        return "conflicts with the base branch"
    if FAILED_MARKER.format(sha=pr["headRefOid"]) in "\n".join(comments):
        return "the gate already failed for this head sha"
    by_name = {c["name"]: c for c in check_runs}
    for need in ("ci", "image"):
        c = by_name.get(need)
        if not c:
            return f"check `{need}` has not run for this sha"
        if c.get("status") != "completed" or c.get("conclusion") != "success":
            return f"check `{need}` is {c.get('conclusion') or c.get('status')}"
    bad = [c["name"] for c in check_runs if c.get("status") == "completed" and c.get("conclusion") not in OK_CONCLUSIONS]
    if bad:
        return "failing checks: " + ", ".join(sorted(set(bad)))
    pending = [c["name"] for c in check_runs if c.get("status") != "completed"]
    if pending:
        return "checks still running: " + ", ".join(sorted(set(pending)))
    badst = [s["context"] for s in statuses if s.get("state") in ("failure", "error", "pending")]
    if badst:
        return "commit statuses not green: " + ", ".join(badst)
    return None


def judge_main(version: str, tagged: bool, head_checks: list[dict], image_checks: list[dict] | None,
               image_version: str | None, comments: list[str], head_sha: str) -> str | None:
    """None = main's head is an eligible release candidate, else the reason. Pure function (unit-tested).

    head_checks: check runs of main's head; image_checks: check runs of the newest main commit that ran
    `image` (None when none of the recent commits did); image_version: VERSION at that commit."""
    if not version:
        return "main has no VERSION"
    if tagged:
        return f"v{version} is already released"
    if FAILED_MARKER.format(sha=head_sha) in "\n".join(comments):
        return "the gate already failed for main's head"
    ci = [c for c in head_checks if c["name"] == "ci"]
    if not ci or ci[-1].get("status") != "completed" or ci[-1].get("conclusion") != "success":
        return "check `ci` on main's head is " + ((ci[-1].get("conclusion") or ci[-1].get("status")) if ci else "missing")
    if image_checks is None:
        return "no recent main commit ran the image build"
    img = [c for c in image_checks if c["name"] == "image"]
    if not img or img[-1].get("conclusion") != "success":
        return "the last main image build is " + ((img[-1].get("conclusion") or img[-1].get("status")) if img else "missing")
    if image_version != version:
        return f"the last main image build is v{image_version}, main is v{version}"
    return None


def pick_main(repo: str) -> dict | None:
    import base64
    commits = gh_json("api", f"repos/{repo}/commits?sha=main&per_page=30") or []
    if not commits:
        return None
    head = commits[0]["sha"]

    def version_at(sha: str) -> str:
        c = gh_json("api", f"repos/{repo}/contents/VERSION?ref={sha}")
        return base64.b64decode(c["content"]).decode().strip() if c else ""

    def checks(sha: str) -> list[dict]:
        runs = gh_json("api", "--paginate", "--slurp", f"repos/{repo}/commits/{sha}/check-runs?per_page=100")
        return [c for page in runs or [] for c in page.get("check_runs", [])]

    ver = version_at(head)
    tagged = subprocess.run(["gh", "api", f"repos/{repo}/git/ref/tags/v{ver}"], capture_output=True).returncode == 0
    head_checks = checks(head)
    image_checks = image_version = None
    for c in commits:
        cr = head_checks if c["sha"] == head else checks(c["sha"])
        if any(x["name"] == "image" for x in cr):
            image_checks, image_version = cr, version_at(c["sha"])
            break
    comments = gh_json("api", "--paginate", "--slurp", f"repos/{repo}/commits/{head}/comments?per_page=100")
    bodies = [c.get("body") or "" for page in comments or [] for c in page]
    why = judge_main(ver, tagged, head_checks, image_checks, image_version, bodies, head)
    print(f"main @ {head[:12]} (v{ver}): " + ("ELIGIBLE" if why is None else f"skipped, {why}"), file=sys.stderr)
    if why is not None:
        return None
    return {"kind": "main", "number": "", "sha": head, "ref": "main",
            "url": f"https://github.com/{repo}/commit/{head}", "title": f"unreleased main v{ver}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", help="a PR number, or `main` for only the unreleased-main candidate")
    args = ap.parse_args()
    if args.pr == "main":
        m = pick_main(args.repo)
        print(json.dumps(m) if m else "{}")
        return 0
    args.pr = int(args.pr) if args.pr else None

    prs = gh_json("pr", "list", "--repo", args.repo, "--state", "open", "--label", "stack", "--limit", "50",
                  "--json", "number,headRefName,headRefOid,labels,isCrossRepository,mergeable,url,title")
    prs = sorted(prs or [], key=lambda p: p["number"])
    for pr in prs:
        if args.pr and pr["number"] != args.pr:
            continue
        sha = pr["headRefOid"]
        runs = gh_json("api", "--paginate", "--slurp", f"repos/{args.repo}/commits/{sha}/check-runs?per_page=100")
        check_runs = [c for page in runs or [] for c in page.get("check_runs", [])]
        st = gh_json("api", f"repos/{args.repo}/commits/{sha}/status")
        comments = gh_json("api", "--paginate", "--slurp", f"repos/{args.repo}/issues/{pr['number']}/comments?per_page=100")
        bodies = [c.get("body") or "" for page in comments or [] for c in page]
        why = judge(pr, check_runs, (st or {}).get("statuses", []), bodies)
        print(f"PR #{pr['number']} ({pr['headRefName']} @ {sha[:12]}): " + ("ELIGIBLE" if why is None else f"skipped, {why}"),
              file=sys.stderr)
        if why is None:
            print(json.dumps({"kind": "pr", "number": pr["number"], "sha": sha, "ref": pr["headRefName"],
                              "url": pr["url"], "title": pr["title"]}))
            return 0
    m = None if args.pr else pick_main(args.repo)
    print(json.dumps(m) if m else "{}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
