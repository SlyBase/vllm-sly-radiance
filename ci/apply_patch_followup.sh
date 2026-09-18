#!/usr/bin/env bash
# After an upstream-sync PR merges, check its Hermes triage comment for a "### PATCH-FOLLOWUP"
# block (path this fork's own patch that upstream now supersedes) and, if present and the named
# patch is actually active, open a small follow-up PR that drops it from the Dockerfile's patch
# loop and deletes the file. Never merges; a human still reviews and approves.
#
#   ci/apply_patch_followup.sh <merged-pr-number>
#
# Needs: gh authenticated with contents + pull-requests write, on a checkout of the now-merged main.
set -euo pipefail
PR=$1
REPO=${GH_REPO:-SlyBase/vllm-sly-radiance}

COMMENT=$(gh pr view "$PR" --repo "$REPO" --json comments \
  --jq '[.comments[] | select(.body | contains("### PATCH-FOLLOWUP"))] | last | .body // empty')
if [ -z "$COMMENT" ]; then
  echo "no PATCH-FOLLOWUP block on PR #$PR"; exit 0
fi

PATCH_PATH=$(echo "$COMMENT" | sed -n 's/^remove: *//p' | head -1 | tr -d '\r')
REASON=$(echo "$COMMENT" | sed -n 's/^reason: *//p' | head -1 | tr -d '\r')

if [ -z "$PATCH_PATH" ] || [ -z "$REASON" ]; then
  echo "::warning::PATCH-FOLLOWUP block found but missing remove:/reason: fields, skipping"; exit 0
fi
case "$PATCH_PATH" in
  patch_*.py|sly/patch_*.py) ;;
  *) echo "::warning::PATCH-FOLLOWUP names '$PATCH_PATH', outside the allowed patch_*.py / sly/patch_*.py scope -- refusing"; exit 0 ;;
esac
if [ ! -f "$PATCH_PATH" ]; then
  echo "$PATCH_PATH already gone, nothing to do"; exit 0
fi

NAME=$(basename "$PATCH_PATH" .py)
if ! grep -qE "(^| )$NAME( |\\\\|\$)" Dockerfile; then
  echo "$NAME is not in the Dockerfile patch loop (already inactive), just untracked -- skipping automated PR (needs a human look)"; exit 0
fi

BRANCH="patch-followup/pr$PR-$NAME"
git checkout -b "$BRANCH"

python3 - "$NAME" <<'PYEOF'
import re, sys
from pathlib import Path
name = sys.argv[1]
p = Path("Dockerfile")
text = p.read_text()
m = re.search(r"for p in (.*?);\s*do", text, re.S)
if not m:
    sys.exit("FAIL: patch loop not found in Dockerfile")
names = m.group(1).replace("\\\n", " ").split()
if name not in names:
    sys.exit(f"FAIL: {name} not found in the Dockerfile loop")
names.remove(name)
new_loop = "for p in " + " ".join(names) + "; do"
p.write_text(text[:m.start()] + new_loop + text[m.end():])
PYEOF

git rm --quiet "$PATCH_PATH"

OLD_VERSION=$(cat VERSION)
IFS=. read -r MA MI PA <<< "$OLD_VERSION"
NEW_VERSION="$MA.$MI.$((PA + 1))"
echo "$NEW_VERSION" > VERSION

git add Dockerfile VERSION
git commit --quiet -m "drop $PATCH_PATH: superseded by the upstream change merged in #$PR

$REASON

VERSION: $OLD_VERSION -> $NEW_VERSION"
git push --quiet -u origin "$BRANCH"

BODY=$(mktemp)
{
  echo "Follow-up to #$PR: drops \`$PATCH_PATH\` from the Dockerfile's patch loop."
  echo
  echo "$REASON"
  echo
  echo "Nothing merged automatically -- review and approve like any other PR."
  echo
  echo "<sub>Hermes triage comment on #$PR, applied by upstream-sync-followup.yml, $(date -u +%Y-%m-%dT%H:%MZ)</sub>"
} > "$BODY"
gh label create patch-followup --repo "$REPO" --color 5319E7 --description "drops a fork patch superseded by upstream" --force >/dev/null 2>&1 || true
gh pr create --repo "$REPO" --base main --head "$BRANCH" --label patch-followup \
  --title "drop $PATCH_PATH (superseded by #$PR)" --body-file "$BODY"
rm -f "$BODY"
