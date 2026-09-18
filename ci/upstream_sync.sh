#!/usr/bin/env bash
# Mirror one upstream's main into origin/upstream/<name> (fast-forward only) and open or refresh
# the PR upstream/<name> -> main. Never merges, never force-pushes. Used by
# .github/workflows/upstream-sync.yml; runnable locally with a checkout that has `origin`.
#
#   ci/upstream_sync.sh <name> <git url>          e.g. ci/upstream_sync.sh ggz14 https://codeberg.org/ggz14/radiance-vllm-mxfp4.git
#
# Needs: git identity (for the test merge), gh authenticated with contents + pull-requests write.
# DRY_RUN=1 skips the mirror push and the PR and prints the PR body and the triage brief instead.
set -euo pipefail
DRY_RUN=${DRY_RUN:-0}
NAME=$1 URL=$2
MIRROR="upstream/$NAME"
REPO=${GH_REPO:-SlyBase/vllm-sly-radiance}
MAX_COMMITS=${MAX_COMMITS:-60}
# The Hermes triage route (homelab repo, hermes-config.yaml.j2) has no terminal and reads GitHub
# only through web_extract, which cuts every response at 50k chars and can serve a stale copy
# (observed 2026-09-18, PR #12: `pulls/12/files` was cut inside its first entry, so no image-file
# diff ever reached the model). So this script also writes a "triage brief" -- the PR body plus
# an overlap table and per-file diffs -- that upstream-sync.yml sends to the route inside the
# webhook payload. The PR body itself stays the short one for humans.
BRIEF_BUDGET=${BRIEF_BUDGET:-56000}   # bytes, the whole brief (~17k tokens in the first prompt)
PER_FILE_CAP=${PER_FILE_CAP:-12000}   # bytes, one file's diff inside the brief

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

# This fork's own current image-layer files (same IMAGE_RE, against origin/main instead of
# the diff), listed once here so the Hermes triage route (see upstream-sync.yml) can match a
# changed upstream file against a differently-named fork equivalent (e.g. a sly/ module or a
# patch_*.py) by name, instead of having to improvise its own directory-listing tool calls --
# that improvisation is what made it read whole directory trees' worth of file contents and
# never finish (observed 2026-09-18, PR #12: 3 context compactions, no comment after 57min).
FORK_IMAGE_FILES=$(git ls-tree -r --name-only origin/main | grep -E "$IMAGE_RE" || true)

# Other paths in the fork's image layer with the same file name as $1 (e.g. the sly/ copy of a
# root-level radiance_*.py), one per line.
counterparts() {
  echo "$FORK_IMAGE_FILES" | awk -v b="${1##*/}" -v f="$1" -F/ '$NF == b && $0 != f'
}

# Markdown for the diffs of $4 between refs $2 and $3. $4 has one entry per line: "path" (diff of
# that path from $2 to $3) or "path<TAB>other" (blob diff of $2:path against $3:other, for a copy of
# the file living at another path). Shares the byte budget $1 out smallest-first, so small diffs
# stay whole; a diff over its share (or over PER_FILE_CAP) is cut at a line boundary and says how
# much was dropped. Empty diffs are skipped.
emit_diffs() {
  local budget=$1 from=$2 to=$3 files=$4 d n=0 left i j f g size share cap kept used
  d=$(mktemp -d)
  while IFS=$'\t' read -r f g; do
    [ -n "$f" ] || continue
    if [ -n "$g" ]; then
      git diff --no-color --no-ext-diff "$from:$f" "$to:$g" > "$d/next"
    else
      git diff --no-color --no-ext-diff "$from" "$to" -- "$f" > "$d/next"
    fi
    [ -s "$d/next" ] || continue
    n=$((n + 1))
    mv "$d/next" "$d/$n.diff"
    if [ -n "$g" ]; then
      printf '#### `%s` (the fork'\''s copy at another path) vs the merge-base `%s`' "$g" "$f" > "$d/$n.head"
    else
      printf '#### `%s`' "$f" > "$d/$n.head"
    fi
    wc -c < "$d/$n.diff" | tr -d ' ' > "$d/$n.size"
  done <<< "$files"
  if [ "$n" -eq 0 ]; then rm -rf "$d"; return 0; fi
  left=$n
  for i in $(for j in $(seq 1 "$n"); do echo "$(cat "$d/$j.size") $j"; done | sort -n | cut -d' ' -f2); do
    size=$(cat "$d/$i.size")
    share=$((budget / left))
    cap=$((share < PER_FILE_CAP ? share : PER_FILE_CAP))
    used=$((size < cap ? size : cap))
    echo "$cap" > "$d/$i.cap"
    budget=$((budget - used))
    left=$((left - 1))
  done
  for i in $(seq 1 "$n"); do
    size=$(cat "$d/$i.size")
    cap=$(cat "$d/$i.cap")
    cat "$d/$i.head"; echo
    echo '````diff'
    if [ "$size" -gt "$cap" ]; then
      head -c "$cap" "$d/$i.diff" | sed '$d' > "$d/$i.cut"
      kept=$(wc -c < "$d/$i.cut" | tr -d ' ')
      cat "$d/$i.cut"
      echo "... [truncated $((size - kept)) more bytes of this file's diff]"
    else
      cat "$d/$i.diff"
    fi
    echo '````'
  done
  rm -rf "$d"
}

# One row per changed image file: what the fork has at that path and whether it touched it since
# the merge-base. The deterministic half of the overlap check, so the triage doesn't derive it.
emit_overlap_table() {
  local f up_stat fork_here fork_stat others
  echo "| upstream changed file | upstream diff | same path in fork main | fork's own change since merge-base | same file name elsewhere in fork main |"
  echo "|---|---|---|---|---|"
  while IFS= read -r f; do
    [ -n "$f" ] || continue
    up_stat=$(git diff --numstat "$BASE" "$UP" -- "$f" | awk '{print "+"$1"/-"$2}')
    fork_stat=-
    if git cat-file -e "origin/main:$f" 2>/dev/null; then
      fork_here=yes
      if git cat-file -e "$BASE:$f" 2>/dev/null; then
        fork_stat=$(git diff --numstat "$BASE" origin/main -- "$f" | awk '{print "+"$1"/-"$2}')
        fork_stat=${fork_stat:-unchanged}
      else
        fork_stat="n/a (path is not in the merge-base)"
      fi
    else
      fork_here=no
    fi
    others=$(counterparts "$f" | paste -sd, -)
    echo "| \`$f\` | ${up_stat:-?} | $fork_here | $fork_stat | ${others:-none} |"
  done <<< "$IMAGE_FILES"
}

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
    echo
    echo "### This fork's own current image-layer files (sly/, patch_*.py, radiance_*.*, ...) -- for the semantic-overlap check above"
    echo '```'; echo "$FORK_IMAGE_FILES"; echo '```'
  else
    echo "### No changed file is part of the image build"
  fi
  echo
  echo "### Commits (newest first, max $MAX_COMMITS)"
  git log --format='- `%h` %s (%ad)' --date=short -n "$MAX_COMMITS" "origin/main..$UP"
  [ "$N" -gt "$MAX_COMMITS" ] && echo "- … $((N - MAX_COMMITS)) more"
} > "$BODY"

# --- 5b. triage brief: the body sections above, then the overlap table and the diffs ---
BRIEF=$(mktemp "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/triage-brief.XXXXXX")
cp "$BODY" "$BRIEF"
printf '\n<sub>upstream-sync.yml, %s</sub>\n' "$(date -u +%Y-%m-%dT%H:%MZ)" >> "$BODY"
if [ -n "$IMAGE_FILES" ]; then
  # What the fork made of each file upstream changed, to hold against the upstream diff: its own
  # diff of the same path, and the diff of a same-named copy at another path (typically the
  # fork's sly/ version of a root-level radiance_*.py) against the merge-base version.
  FORK_SIDE=""
  while IFS= read -r f; do
    git cat-file -e "$BASE:$f" 2>/dev/null || continue   # new upstream file: nothing to diff from
    if git cat-file -e "origin/main:$f" 2>/dev/null; then
      FORK_SIDE+="$f"$'\n'
    fi
    while IFS= read -r g; do
      if [ -n "$g" ]; then FORK_SIDE+="$f"$'\t'"$g"$'\n'; fi
    done < <(counterparts "$f")
  done <<< "$IMAGE_FILES"
  {
    echo
    echo "### Overlap table (computed by the workflow, not by a model; +added/-removed lines)"
    emit_overlap_table
  } >> "$BRIEF"
  REST=$((BRIEF_BUDGET - $(wc -c < "$BRIEF" | tr -d ' ') - 1500))   # 1500: headings/notes below
  [ "$REST" -ge 3000 ] || REST=3000
  UPDIFF=$(mktemp) FORKDIFF=$(mktemp)
  # fork side first (usually small), the upstream diffs get whatever it leaves over
  emit_diffs $((REST / 3)) "$BASE" origin/main "$FORK_SIDE" > "$FORKDIFF"
  emit_diffs $((REST - $(wc -c < "$FORKDIFF" | tr -d ' '))) "$BASE" "$UP" "$IMAGE_FILES" > "$UPDIFF"
  {
    echo
    echo "### Upstream diffs of the changed image files (merge-base..upstream head)"
    echo "\`-\` = merge-base, \`+\` = upstream head. A \`... [truncated N more bytes ...]\` line marks a diff cut to fit the brief."
    cat "$UPDIFF"
    echo
    echo "### What the fork made of those files (fork main vs the merge-base version)"
    if [ -s "$FORKDIFF" ]; then
      echo "\`-\` = merge-base (what upstream had), \`+\` = the fork. Only the fork's own changes appear here (same path, and same-named copies at other paths); files the fork left alone show as unchanged in the table above."
      cat "$FORKDIFF"
    else
      echo "(none: the fork changed none of them, and has no same-named copy at another path)"
    fi
  } >> "$BRIEF"
  rm -f "$UPDIFF" "$FORKDIFF"
fi

# --- 6. open or refresh the PR ---
if [ "$DRY_RUN" = 1 ]; then
  echo "--- DRY_RUN: PR body ($(wc -c < "$BODY" | tr -d ' ') bytes) ---"; cat "$BODY"
  echo "--- DRY_RUN: triage brief ($(wc -c < "$BRIEF" | tr -d ' ') bytes) ---"; cat "$BRIEF"
  rm -f "$BODY" "$BRIEF"; exit 0
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
  echo "brief_file=$BRIEF" >> "$GITHUB_OUTPUT"
fi
