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

# --- 3. test merge for the conflict list (throw-away worktree) ---
WT=$(mktemp -d)
git worktree add --quiet --detach "$WT" origin/main
CONFLICTS=""
if ! git -C "$WT" merge --no-commit --no-ff --quiet "$UP" >/dev/null 2>&1; then
  CONFLICTS=$(git -C "$WT" diff --name-only --diff-filter=U || true)
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
else
  if ! URLOUT=$(gh pr create --repo "$REPO" --head "$MIRROR" --base main --label upstream-sync --title "$TITLE" --body-file "$BODY" 2>&1); then
    echo "$URLOUT"
    echo "::error::could not create the PR $MIRROR -> main. With the default GITHUB_TOKEN this needs the org/repo setting 'Allow GitHub Actions to create and approve pull requests'; otherwise set the SYNC_TOKEN secret (fine-grained PAT, contents + pull-requests write)."
    exit 1
  fi
  echo "PR created: $URLOUT"
fi
rm -f "$BODY"
