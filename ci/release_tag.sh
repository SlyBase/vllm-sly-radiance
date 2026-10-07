#!/usr/bin/env bash
# Cut the release tag v<VERSION> on a commit and push it.
#
# A release in this repo is an annotated v* tag on a commit main already carries (the VERSION history,
# README), and build.yml publishes every v* tag to ghcr.io. Two callers share this script:
#   accept.yml   the acceptance gate was green      -> tags the gated commit, no human involved
#   release.yml  someone releases without a green gate -> the reason ends up in the tag message
#
#   ci/release_tag.sh <version> <sha> <reason> [<run-url>]
#
# The checkout must have full history and a token whose push starts workflows (an App token: a push made
# with the default GITHUB_TOKEN does not trigger build.yml, so the tag would exist and nothing would be
# published). Idempotent: the tag already on that commit is a no-op; on another commit it is an error,
# a tag is never moved.
#
# RELEASE_NOTIFY (env, optional): GitHub handles mentioned in the release body, e.g. "@slydlake". A
# mention is what makes the GitHub app push a notification for the new release (a release alone only
# reaches people watching the repo with "Releases"). The callers set it from the repository variable of
# the same name, default @slydlake; an empty variable value cannot be set, so "-" turns it off.
set -euo pipefail

VER=${1:?usage: release_tag.sh <version> <sha> <reason> [<run-url>]}
SHA=${2:?sha}
WHY=${3:?reason}
RUN=${4:-}
TAG="v${VER}"

fail() { echo "::error::$*"; exit 1; }

# The GitHub release of the tag: its body is the version's CHANGELOG.md section (ci/changelog_section.py),
# so what a tag contains is readable on the releases page. Needs GH_TOKEN (the App token the caller
# already uses for the push). Idempotent: an existing release is left alone.
publish_release() {
  if [ -z "${GH_TOKEN:-}" ] || ! command -v gh >/dev/null; then
    echo "::warning::no GH_TOKEN/gh: tag pushed, GitHub release for ${TAG} not created"; return 0
  fi
  if gh release view "$TAG" >/dev/null 2>&1; then
    echo "GitHub release ${TAG} exists"; return 0
  fi
  local notes
  notes=$(mktemp)
  git show "${COMMIT}:ci/changelog_section.py" >/dev/null 2>&1 \
    && python3 ci/changelog_section.py --ref "$TAG" "$VER" > "$notes" \
    || echo "No CHANGELOG.md section for ${VER} at this commit." > "$notes"
  # Append the commit list since the previous release tag (auto-generated; no manual curation).
  # `git describe` finds the most recent tag before $SHA; if none exists, skip silently.
  local prev_tag
  prev_tag=$(git describe --tags --abbrev=0 "${SHA}^" 2>/dev/null || true)
  if [ -n "$prev_tag" ]; then
    {
      echo ""
      echo "### Commits since ${prev_tag}"
      echo ""
      git log --oneline --no-merges "${prev_tag}..${SHA}" | sed 's/^\([0-9a-f]*\) /-\`\1\` /'
    } >> "$notes"
  fi
  # the tag suffix is the ROCm major.minor of ROCM_BASE at that commit (same rule as build.yml)
  local rocm_mm
  rocm_mm=$(git show "${COMMIT}:Dockerfile" | sed -nE 's/^ARG ROCM_BASE=[^:]+:([0-9]+\.[0-9]+)\..*/\1/p' | head -1)
  { echo; echo "---"; echo "Image: \`ghcr.io/slybase/vllm-sly-radiance:${VER}-rocm${rocm_mm:-10.1}\` (published by the build workflow for this tag)."; echo "${WHY}"; } >> "$notes"
  if [ -n "${RELEASE_NOTIFY:-}" ] && [ "${RELEASE_NOTIFY}" != "-" ]; then
    { echo; echo "cc ${RELEASE_NOTIFY}"; } >> "$notes"
  fi
  gh release create "$TAG" --verify-tag --title "vllm-sly-radiance ${VER}" --notes-file "$notes"
  rm -f "$notes"
}

[[ "$VER" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || fail "'$VER' is not a MAJOR.MINOR.PATCH version"
git fetch --quiet origin main --tags
COMMIT=$(git rev-parse --verify --quiet "${SHA}^{commit}") || fail "$SHA is not a commit of this repository"

# A release is something main carries, never a side branch.
git merge-base --is-ancestor "$COMMIT" origin/main || fail "$COMMIT is not on main"

# The tag has to say what the commit says: the image tag comes from VERSION at that commit.
AT=$(git show "${COMMIT}:VERSION" 2>/dev/null | tr -d '[:space:]' || true)
[ "$AT" = "$VER" ] || fail "VERSION at ${COMMIT:0:12} is '${AT}', not '${VER}'"

if HAVE=$(git rev-parse --verify --quiet "refs/tags/${TAG}^{commit}"); then
  if [ "$HAVE" = "$COMMIT" ]; then
    echo "${TAG} is already on ${COMMIT:0:12}: nothing to tag"
    publish_release
    exit 0
  fi
  fail "${TAG} already exists on ${HAVE:0:12}, refusing to move it to ${COMMIT:0:12}"
fi

if [ -n "$RUN" ]; then
  git tag -a "$TAG" "$COMMIT" -m "vllm-sly-radiance ${VER}" -m "$WHY" -m "$RUN"
else
  git tag -a "$TAG" "$COMMIT" -m "vllm-sly-radiance ${VER}" -m "$WHY"
fi
git push origin "refs/tags/${TAG}"
echo "released ${TAG} at ${COMMIT}"
publish_release
