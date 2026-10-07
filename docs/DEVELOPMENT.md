# Development: build, CI, releases

How the image is built, tested and released. Contributor and agent rules are in [AGENTS.md](../AGENTS.md).

## Build

Everything the build needs is in this directory (flat Docker context). Multi-stage: **buildbase**
→ **torch** (AMD's prebuilt PyTorch wheel for ROCm 10.1 by default, or PyTorch from source, see
below) → **builder** (Triton as the hash-pinned PyPI wheel; torchvision, AITER, vLLM from source)
→ **rocmprune** → **assemble**
(wheels, upstream RDNA4 patches, then the `sly/` patches, libr4d, the HIP kernel) → **venvsplit**
→ **final** (clean `ubuntu:24.04` + pruned ROCm + venv + entrypoint).

The final image has three large layers, from stable to volatile: the pruned ROCm tree
([`prune_rocm.sh`](prune_rocm.sh): gfx1201 only, no static archives, no Flang/MLIR), the **cold**
venv (the installed stack: torch, triton, aiter's metadata, the Python dependencies) and the **hot**
venv (the patched vLLM and aiter trees, the few files patched elsewhere, the radiance modules and
kernels). [`split_venv.py`](split_venv.py) makes the split and resets every mtime, so the cold layer
is byte-identical from release to release as long as the stack is: a `docker pull` of the next
release downloads the hot layer only, until the next ROCm or stack bump.

`docker inspect --format '{{json .Config.Labels}}' <image>` identifies an image: the
`org.opencontainers.image.*` labels (version, git revision, build date, source) and the
`io.slybase.radiance.*` component pins (vLLM, torch, triton, aiter, transformers, libr4d, ROCm base).

```bash
git clone https://github.com/SlyBase/vllm-sly-radiance.git
cd vllm-sly-radiance

docker build -t vllm-sly-radiance:$(cat VERSION)-rocm10.1 .
```

A cold build takes about 30 minutes on the runner (`MAX_JOBS=4` by default): PyTorch is AMD's
wheel from `stable.repo.amd.com/rocm/whl-next` (`torch==2.12.0+rocm10.1.0` plus
`amd-torch-device-gfx1201`), installed against the image's own `/opt/rocm`. How that fits
(metadata rewrite, `rocm_sdk/` stand-in, no strip of `torch/lib`) is in the comment on the
`torch-amd` stage. With the builder stage cached, a change to the `sly/` layer rebuilds in ~10 minutes.

For a ROCm release AMD has no wheel for yet (a 10.1 beta, say), `--build-arg TORCH_FROM=torch-wheel`
compiles PyTorch from source instead (~2 h, torch at `MAX_JOBS=3`). That compile can also come from an
image: `--build-arg TORCH_FROM=<image>` takes the `torch-wheel` target from there. CI keeps one per torch definition at
`ghcr.io/slybase/vllm-sly-radiance-torch:<tag>`, the tag from [`ci/torch_key.py`](../ci/torch_key.py)
(torch version, ROCm major.minor, GPU arch, hash of the torch stages and the base image). A local
build can use it too when logged in to ghcr.io:

```bash
docker build --build-arg TORCH_FROM=ghcr.io/slybase/vllm-sly-radiance-torch:$(python3 ci/torch_key.py) \
  -t vllm-sly-radiance:$(cat VERSION)-rocm10.1 .
```
The build needs no GPU, so it can run next to a serving container.

Smoke test:

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri -e HIP_VISIBLE_DEVICES=0 \
  --entrypoint python3 vllm-sly-radiance:$(cat VERSION)-rocm10.1 \
  -c "from vllm.model_executor.layers.quantization.quark import QuarkConfig; print('OK')"
```

## Repository layout

```
Dockerfile                    build pipeline (upstream + sly/ patch loop + HIP kernel compile)
VERSION                       image version
_patchlib.py                  anchor-based, idempotent patch helper (upstream)
patch_*.py, radiance_*.py     upstream vllm-radiance RDNA4 patches and runtime modules
sly/
  README.md                   per-patch reference (German)
  patch_quark_mxfp4.py        Quark/MXFP4 loader gates for vLLM 0.29.0 + kernel plugin registration
  patch_short_prefill.py      GDN 1-token-prefill fix
  patch_dflash_w4_packed.py   W4A16 (compressed-tensors) DFlash drafter
  patch_gdn_nonspec_mask.py   non_spec_sequence_masks_cpu on the numpy path
  patch_lmhead_fp8.py         hook radiance_lmhead_fp8 into QuarkConfig
  patch_w4a16_tiles.py        gfx1201 tile table, split-K, tiled layout and fused epilogues for the W4A16 GEMMs
  bench_w4a16_tiles.py        tile / split-K sweep that produced the tables
  check_w4a16_splitk.py       W4A16 kernel numerics vs fp32 reference + bit-identity of new paths (CPU or GPU)
  check_w4a16_fuse.py         post-load transforms end to end on stand-in layers (CPU interpreter)
  patch_w4a16_fuse.py         call sites of the W4A16 fusions (post-load hook, MLP silu, GDN qkvz+ba, DFlash context-KV)
  mxfp4/radiance_mxfp4.py     RadianceMxfp4W4A8LinearKernel plugin (dispatch, scratch, knobs)
  mxfp4/radiance_mxfp4_fp8.hip  fp8-WMMA W4A8 GEMM: folded prefill + split-K decode kernels
  mxfp4/radiance_lmhead_fp8.py  fp8 lm_head
  mxfp4-configs/              AITER gemm_afp4wfp4 config for gfx1201
```

## Branches, CI and upstream sync

```
main                    integration branch = upstream vllm-radiance + sly/ (protected: PR + green `ci`)
upstream/stilldeadcode  read-only mirror of StillDeadcode/vllm-radiance `main` (Codeberg)
upstream/ggz14          read-only mirror of ggz14/radiance-vllm-mxfp4 `main` (Codeberg)
archive/*               tags freezing the pre-2026-09 layout (sly/main, sly/b-*, sly/e2-*) — kept forever
v0.1.1 … v0.1.6         annotated tags = the VERSION history (v0.1.0 has no unambiguous commit)
```

Remotes on a dev machine: `origin` (GitHub) plus the two Codeberg upstreams:

```bash
git remote add stilldeadcode https://codeberg.org/StillDeadcode/vllm-radiance.git
git remote add ggz14 https://codeberg.org/ggz14/radiance-vllm-mxfp4.git
```

Workflows (`.github/workflows/`):

| Workflow | Trigger | What it does |
|---|---|---|
| `ci` | PR, push to `main`, manual | `lint` (ruff E9/F63/F7/F82, shellcheck, hadolint, actionlint, `docker buildx build --check`), `patch-dryrun` (`ci/patch_dryrun.sh`: the pinned upstream sources from the Dockerfile ARGs in a venv, then the Dockerfile patch loop twice — pass 1 must apply every hunk, pass 2 must be all NOOP; `ci/patch_dryrun_skip.txt` is the documented skip allowlist), `constraints` (`ci/check_constraints.py`, see below), `consistency` (`ci/check_consistency.py`: every patch file is in the loop or in `ci/unused_patches.txt`, every `sly/patch_*.py` is documented in `sly/README.md`, image changes bump `VERSION`). The aggregate status **`ci`** is the required check on `main`. |
| `build` | push to `main` touching `VERSION`, tags `v*`, manual | Self-hosted runner (`rocm-build`, LXC 2408, CPU only — no `--device`, no GPU test, no deploy): `docker build` with AMD's PyTorch wheel (input `torch_from_source`: PyTorch from `ghcr.io/slybase/vllm-sly-radiance-torch:<ci/torch_key.py>`, missing → `--target torch-wheel` once, ~2 h, and pushed) → import smoke test → push to `ghcr.io/slybase/vllm-sly-radiance:<VERSION>-rocm10.1` only for `v*` tags or the `push_ghcr` input. Build log is an artifact. |
| `upstream-sync` | daily 04:00 UTC, manual | Fast-forwards `upstream/*` from Codeberg (never force) and opens/updates a PR `upstream/<name>` → `main` (label `upstream-sync`) listing the new commits, the test-merge conflict status and the image-relevant files. Never merges. Needs the `SYNC_TOKEN` secret (fine-grained PAT, contents + pull-requests write) to create PRs. |
| `renovate` | every 6 h, push to `main` touching the config, manual | Self-hosted Renovate (same setup as SlyBase/helm-charts) with the upstream-sync App token (`SYNC_APP_*`). Optional secret `RENOVATE_GITHUB_COM_TOKEN` (read-only PAT) lifts the rate limit for lookups in other repositories. |

Renovate (`renovate.json`) tracks every Dockerfile ARG pin via the `# renovate:` markers above the
ARGs (ROCm base image tag + digest, ubuntu digest, torch/triton/torchvision as one group, vLLM,
AITER, transformers, rocm_bandwidth_test, libr4d commit), the CI tool versions, the actions and
`constraints.txt`. No automerge — every bump goes through `ci`, and vLLM/AITER/ROCm bumps
additionally need a build and an A/B run. `renovate/*` PRs are exempt from the VERSION-bump check;
they collect on `main` and ship with the next VERSION bump.

**Python dependencies.** Everything the image installs from PyPI is pinned in
[`constraints.txt`](constraints.txt) (`pip install -c` in the assemble stage). Without it every
rebuild that misses the build cache resolved vLLM's open ranges to whatever was newest that day.
Renovate opens one weekly PR (`renovate/python-deps`) for the minor/patch updates and one for the
majors (`renovate/major-python-deps`). The `constraints` job in `ci` replays the image's pip
resolution for the pinned vLLM ([`ci/check_constraints.py`](ci/check_constraints.py), ~30 s, no
build) and fails on a missing, stale or out-of-range pin. Two things keep those PRs green:

- **Generated limits.** Renovate looks each package up on its own and cannot know that numba pins
  llvmlite `<0.48` or that the OpenTelemetry packages pin each other exactly. The script derives
  these from the resolution and keeps them as `GENERATED` rules in `renovate.json` (`enabled: false`
  for exact pins, `allowedVersions` for upper bounds). CI fails when they are out of date.
- **Repair.** Tightly coupled pairs can still arrive half-updated (httpx2 without the httpcore2 it
  pins exactly, pydantic-core ahead of pydantic). On a `renovate/*` branch a failing check runs
  `--repair`: the changed pins become ranges between `main`'s version and the proposal, pip picks the
  newest consistent set (never newer than proposed, never older than `main`), and the job pushes it
  to the branch with the App token. Renovate leaves the branch alone after that.

After a vLLM bump (or when a pin stops resolving) re-resolve the whole file:

```bash
python3 ci/check_constraints.py --update   # needs python 3.12 on linux x86_64, like the runner
```

Changing code:

```bash
git switch -c feat/my-change main
# edit; bump VERSION for anything that changes the image (consistency enforces it on PRs)
python3 ci/check_consistency.py --base origin/main && ci/patch_dryrun.sh
git push -u origin feat/my-change && gh pr create
# merge when `ci` is green — the VERSION bump on main then triggers `build`
```

Merging upstream: take the `upstream-sync` PR (or `git merge origin/upstream/<name>` on a branch),
resolve the usual conflicts (Dockerfile pins + patch loop, `README.md`, `VERSION`) and re-verify
every `sly/` anchor — `ci/patch_dryrun.sh` fails hard when an anchor is gone.

## Releases and the changelog

- Every `VERSION` bump carries a `## [<version>] - <date>` section in `CHANGELOG.md`
  (`ci/check_consistency.py` fails without it; `ci/changelog_section.py <version>` prints it).
- A green acceptance gate (`accept.yml`) runs `ci/release_tag.sh`: annotated tag `v<VERSION>` on the
  gated commit, then a GitHub release whose body is that CHANGELOG section plus the image reference.
  `release.yml` does the same by hand (reason required). The tag starts `build.yml`, which pushes the
  image to `ghcr.io/slybase/vllm-sly-radiance:<version>-rocm10.1`.
- After a release that changes the numbers, run the reference benchmark (see `AGENTS.md`), put the
  new values into the README's *Performance* table and move the previous table to
  `docs/BENCHMARKS.md`.

## Renovate

Renovate runs without an hourly or concurrent PR cap (`prHourlyLimit` / `prConcurrentLimit` 0): with
the default 2/h and a cap of 5, updates sat rate-limited on the dashboard for days. Grouping keeps the
PR count small (torch stack, Python deps, CI tools, actions); a dependency bump with image impact is
integrated like any other change (VERSION, CHANGELOG, build, GPU gate).

Image-relevant bumps are not made by hand any more: only a vLLM release or a ROCm base tag opens a `stack`
PR, and `ci/resolve_stack.py` derives the rest. Pipeline, night gate, Hermes rules and the pause switch:
[MAINTENANCE.md](MAINTENANCE.md).
