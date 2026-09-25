# Working on vllm-sly-radiance (humans and coding agents)

Read this before changing anything. It keeps the image, the docs and the release notes in step.

## Every change that ends up in the image

1. Work on a branch (or a `git worktree` if someone else uses the checkout) — never on `main`.
2. Bump `VERSION` (MAJOR.MINOR.PATCH; minor for a dependency/stack bump or a new feature, patch for a
   fix). `ci/check_consistency.py` fails a PR that changes an image file without a bump.
3. Add a `## [<version>] - <date>` section at the top of `CHANGELOG.md` (Added / Changed / Fixed /
   Removed, plus a *Measured* line with the numbers that justify the change). The consistency check
   fails without it, and `ci/release_tag.sh` publishes exactly that section as the GitHub release body
   when the `v<version>` tag is cut. Write it for users: what changes for them, and by how much.
4. New patch: `sly/patch_*.py`, anchored with `_patchlib.apply` / `apply_any`, idempotent, listed in the
   Dockerfile loop and documented in `sly/README.md`. A patch that is kept out of the loop goes into
   `ci/unused_patches.txt` with the reason.
5. New environment knob: add it to the options table in `README.md` with default, recommended value
   and why.
6. Run `python3 ci/check_consistency.py --base origin/main` and `ci/patch_dryrun.sh` before pushing.

## Measuring (anything that claims to be faster)

- One R9700, production arguments, **300 W with the firmware fan curve** for A/Bs; the production
  setting is 210 W / fan capped at 2,800 rpm. Restore it afterwards.
- A/B in one GPU window, control arm first and repeated last; only compare arms of the same window
  (prefill differs by ~±3 % between sessions, the same image repeats within 0.3 % inside one).
- The first start after an image or flag change compiles fresh and shows a smaller KV pool; measure
  the second start. The KV pool must stay at 384,316 tokens (262k context) for the production setup.
- Decode: rank by the step gap (deterministic) or by ≥ 128 BetterBench runs per arm — a single 64-run
  BetterBench sample carries ±5 % on the weighted decode (every run is one sampling trajectory).
- Quality: GSM8K 200 (cot zero-shot, greedy) against the baseline in `ci/accept/baselines/`.

## Docs layout

| File | Contents | Update when |
|---|---|---|
| `README.md` | stack, current reference numbers + conditions, quickstart, what changed and why (short), options with recommendations | a release changes a number, a knob or the recommended launch |
| `CHANGELOG.md` | one section per version (= release notes) | every VERSION bump |
| `docs/TECHNICAL.md` | deep dives per change | a change needs more than two lines of explanation |
| `docs/BENCHMARKS.md` | methodology, full tables, history, checkpoint comparison | a new reference run; move the old README numbers here |
| `docs/NOT-ADOPTED.md` | what other stacks do and why this one does not | an A/B rejects something |
| `docs/MULTI-GPU.md`, `docs/DEVELOPMENT.md` | TP>1 notes; build, CI, Renovate, releases | the respective area changes |
| `sly/README.md` | per-patch reference (German) | a `sly/` file is added or changed |

## Releases

A green acceptance gate (`accept.yml`, GPU window approved by a maintainer) tags `v<VERSION>` on the
gated commit; the tag starts `build.yml`, which publishes the image to ghcr.io, and `release_tag.sh`
creates the GitHub release from the CHANGELOG section. A release without a green gate goes through
Actions → release (reason required).

## Upstreams

`upstream/stilldeadcode` and `upstream/ggz14` are read-only mirrors (daily sync PRs). Credit the authors
in `README.md` → Credits when you take something over, and say what was taken.
