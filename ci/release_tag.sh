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
set -euo pipefail

VER=${1:?usage: release_tag.sh <version> <sha> <reason> [<run-url>]}
SHA=${2:?sha}
WHY=${3:?reason}
RUN=${4:-}
TAG="v${VER}"

fail() { echo "::error::$*"; exit 1; }

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
    echo "${TAG} is already on ${COMMIT:0:12}: nothing to do"
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
