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

    python3 ci/night_pick.py --repo SlyBase/vllm-sly-radiance            # prints JSON, {} when nothing
    python3 ci/night_pick.py --repo ... --pr 107                         # only that PR (workflow_dispatch)

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--pr", type=int)
    args = ap.parse_args()

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
            print(json.dumps({"number": pr["number"], "sha": sha, "ref": pr["headRefName"], "url": pr["url"],
                              "title": pr["title"]}))
            return 0
    print("{}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
