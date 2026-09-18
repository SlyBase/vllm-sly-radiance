#!/usr/bin/env bash
# Mirror one upstream's main into origin/upstream/<name> (fast-forward only) and open or refresh
# the PR upstream/<name> -> main. Never merges, never force-pushes. Used by
# .github/workflows/upstream-sync.yml; runnable locally with a checkout that has `origin`.
#
#   ci/upstream_sync.sh <name> <git url>          e.g. ci/upstream_sync.sh ggz14 https://codeberg.org/ggz14/radiance-vllm-mxfp4.git
#
# Needs: git identity (for the test merge), gh authenticated with contents + pull-requests write.
# DRY_RUN=1 skips the mirror push and the PR and prints the PR body instead.
set -euo pipefail
DRY_RUN=${DRY_RUN:-0}
NAME=$1 URL=$2
MIRROR="upstream/$NAME"
REPO=${GH_REPO:-SlyBase/vllm-sly-radiance}
MAX_COMMITS=${MAX_COMMITS:-60}

git remote remove up 2>/dev/null || true
git remote add up "$URL"
git fetch --quiet up main
git fetch --quiet origin main "+refs/heads/$MIRROR:refs/remotes/origin/$MIRROR" 2>/dev/null || true
UP=$(git rev-parse up/main)

# --- 1. mirror branch, fast-forward only ---
if git rev-parse --verify --quiet "origin/$MIRROR" >/dev/null; then
  CUR=$(git rev-parse "origin/$MIRROR")
  if [ "$CUR" = "$UP" ]; then
    echo "mirror $MIRROR already at ${UP:0:7}"
  elif git merge-base --is-ancestor "$CUR" "$UP"; then
    [ "$DRY_RUN" = 1 ] || git push --quiet origin "$UP:refs/heads/$MIRROR"
    echo "mirror $MIRROR: ${CUR:0:7} -> ${UP:0:7} (fast-forward)"
  else
    echo "::error::$MIRROR is not an ancestor of $URL main (upstream rewrote history: ${CUR:0:7} vs ${UP:0:7}); refusing to force-push -- resolve by hand"
    exit 1
  fi
else
  [ "$DRY_RUN" = 1 ] || git push --quiet origin "$UP:refs/heads/$MIRROR"
  echo "mirror $MIRROR created at ${UP:0:7}"
fi

# --- 2. anything new for main? ---
N=$(git rev-list --count "origin/main..$UP")
if [ "$N" -eq 0 ]; then
  echo "$MIRROR: nothing that is not already in main"; exit 0
fi
BASE=$(git merge-base origin/main "$UP")
echo "$MIRROR: $N commit(s) not in main (merge-base ${BASE:0:7})"

# --- 3. test merge for the conflict list + consistency check (throw-away worktree) ---
WT=$(mktemp -d)
git worktree add --quiet --detach "$WT" origin/main
CONFLICTS=""
MERGED=1
if ! git -C "$WT" merge --no-commit --no-ff --quiet "$UP" >/dev/null 2>&1; then
  CONFLICTS=$(git -C "$WT" diff --name-only --diff-filter=U || true)
  # README is never taken from upstream -- the fork keeps its own. Resolve it here (only in
  # this throwaway tree) so a lone README conflict doesn't block the consistency check below;
  # the CONFLICTS list above still reports it to the human untouched.
  if [ "$CONFLICTS" = "README.md" ] && git -C "$WT" checkout --ours -- README.md 2>/dev/null; then
    git -C "$WT" add README.md
  else
    MERGED=0
  fi
fi
CONSISTENCY=""
if [ "$MERGED" = 1 ]; then
  git -C "$WT" commit --quiet --no-verify -m "throwaway test merge" || MERGED=0
fi
if [ "$MERGED" = 1 ]; then
  CONSISTENCY=$(cd "$WT" && python3 ci/check_consistency.py --base origin/main 2>&1 || true)
fi
git -C "$WT" merge --abort >/dev/null 2>&1 || true
git worktree remove --force "$WT"

# --- 4. files that matter for the image (Dockerfile loop, COPY'd modules, sly/ layer) ---
CHANGED=$(git diff --name-only "$BASE" "$UP")
IMAGE_RE='^(Dockerfile|VERSION|_patchlib\.py|install_radiance_hooks\.py|patch_[^/]*\.py|radiance_[^/]*\.(py|pth|sh)|prune_rocm\.sh|sly/.*|fp8-configs/.*|moe-configs/.*|dflash2/.*|[^/]*\.jinja)$'
IMAGE_FILES=$(echo "$CHANGED" | grep -E "$IMAGE_RE" || true)

# --- 5. PR body ---
BODY=$(mktemp)
{
  echo "Automated mirror of \`$URL\` (branch \`main\`) into \`$MIRROR\`, now at \`${UP:0:7}\`."
  echo
  echo "**$N commit(s)** not in \`main\` (merge-base \`${BASE:0:7}\`, $(echo "$CHANGED" | grep -c . ) files changed). Nothing is merged automatically: review, resolve, bump \`VERSION\`, build, A/B."
  echo
  if [ -n "$CONFLICTS" ]; then
    echo "### Test merge into main: CONFLICTS"; echo '```'; echo "$CONFLICTS"; echo '```'
  else
    echo "### Test merge into main: clean (no textual conflicts -- still review the overlaps below)"
  fi
  echo
  if [ "$MERGED" = 1 ]; then
    if echo "$CONSISTENCY" | grep -q "^FAIL:"; then
      echo "### ci/check_consistency.py against the test merge: FAIL"
    else
      echo "### ci/check_consistency.py against the test merge: OK"
    fi
    echo '```'; echo "$CONSISTENCY"; echo '```'
  else
    echo "### ci/check_consistency.py: skipped (unresolved conflicts beyond README.md)"
  fi
  echo
  if [ -n "$IMAGE_FILES" ]; then
    echo "### Changed files that are part of the image (Dockerfile loop / COPY / sly)"; echo '```'; echo "$IMAGE_FILES"; echo '```'
  else
    echo "### No changed file is part of the image build"
  fi
  echo
  echo "### Commits (newest first, max $MAX_COMMITS)"
  git log --format='- `%h` %s (%ad)' --date=short -n "$MAX_COMMITS" "origin/main..$UP"
  [ "$N" -gt "$MAX_COMMITS" ] && echo "- … $((N - MAX_COMMITS)) more"
  echo
  echo "<sub>upstream-sync.yml, $(date -u +%Y-%m-%dT%H:%MZ)</sub>"
} > "$BODY"

# --- 6. open or refresh the PR ---
if [ "$DRY_RUN" = 1 ]; then
  echo "--- DRY_RUN: PR body ---"; cat "$BODY"; rm -f "$BODY"; exit 0
fi
gh label create upstream-sync --repo "$REPO" --color 0E8A16 --description "automated upstream mirror PR" --force >/dev/null 2>&1 || true
TITLE="upstream sync: $NAME ($N new commits, ${UP:0:7})"
EXISTING=$(gh pr list --repo "$REPO" --head "$MIRROR" --base main --state open --json number --jq '.[0].number // empty')
if [ -n "$EXISTING" ]; then
  gh pr edit "$EXISTING" --repo "$REPO" --title "$TITLE" --body-file "$BODY" >/dev/null
  echo "PR #$EXISTING refreshed: $(gh pr view "$EXISTING" --repo "$REPO" --json url --jq .url)"
  PR_NUM=$EXISTING PR_ACTION=edited
else
  if ! URLOUT=$(gh pr create --repo "$REPO" --head "$MIRROR" --base main --label upstream-sync --title "$TITLE" --body-file "$BODY" 2>&1); then
    echo "$URLOUT"
    echo "::error::could not create the PR $MIRROR -> main. With the default GITHUB_TOKEN this needs the org/repo setting 'Allow GitHub Actions to create and approve pull requests'; otherwise set up a GitHub App (Contents + Pull requests R/W) and add its ID/private key as the SYNC_APP_ID/SYNC_APP_PRIVATE_KEY secrets (see upstream-sync.yml's header)."
    exit 1
  fi
  echo "PR created: $URLOUT"
  PR_NUM=${URLOUT##*/} PR_ACTION=opened
fi
rm -f "$BODY"

# GitHub does not deliver repository webhooks for pull_request edits/creates made with a
# GitHub App installation token (verified empirically 2026-09-18: the same gh pr edit fires a
# webhook delivery under a user token but not under the App token this script normally runs
# with) -- so upstream-sync.yml pings the Hermes triage webhook itself right after this script,
# using PR_NUM/PR_ACTION below (unset/empty when step 2 exited early with nothing to sync).
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "pr_number=$PR_NUM" >> "$GITHUB_OUTPUT"
  echo "pr_action=$PR_ACTION" >> "$GITHUB_OUTPUT"
fi
