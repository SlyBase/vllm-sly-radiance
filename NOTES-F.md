# NOTES-F: ROCm 10.0 -> 10.1 image bump

## What exists (checked 2026-10-07 against the public indexes)

- Base image: `rocm/dev-ubuntu-24.04:10.1.0-full`, pushed 2026-10-05, digest
  `sha256:5ed1362ea542a928651e4c710b44024ebe98870f74159cb70f0265aec8ef0abe` (10.0.0-full was 2026-08-26). Same
  TheRock layout as 10.0 (`/opt/rocm` -> `core-10.1`, `core/.info/version` = 10.1.0, no `/opt/rocm/.info/version`,
  python 3.12.3). HIP 7.16.26385 (10.0: 7.15.26333), clang 24 (10.0: 23), libhsa-runtime64 1.21.0 (same soname),
  4 more libs than 10.0, tree 9.9 GB vs 9.2 GB.
- AMD torch wheels on `stable.repo.amd.com/rocm/whl-next` (redirect target of `/rocm/whl-next`, no auth), cp312 linux:
  torch **2.12.0 / 2.13.0 / 2.14.0** `+rocm10.1.0`; **there is no 2.11 wheel for rocm10.1** (2.11.0 exists only for
  rocm10.0.0). `amd-torch-device-gfx1201` has the same three versions (`torch/.kpack/torch_gfx1201.kpack`).
  torchvision `+rocm10.1.0`: 0.27.0 / 0.28.0 / 0.29.0a0; AMD triton only `3.8.0+git669b31ac.rocm10.1.0`.
- The 10.1 torch wheel has the same metadata shape as the 10.0 one (`rocm[libraries]~=10.1.0`,
  `triton==3.8.0+git669b31ac.rocm10.1.0`, `rocm-bootstrap`; the device wheel `rocm-sdk-device-gfx1201`).
  `fix_amd_torch_metadata.py` and the `rocm_sdk/` stand-in therefore work unchanged.
- PyPI triton: 3.6.0, 3.7.0, 3.7.1, 3.8.0 (3.7.0 cp312 x86_64 sha256 `8f111161...efcf`, 3.7.1 `7e408699...a728`).
- ROCm/ROCm#6406 (the reason Renovate held torch on 2.11, "torch >= 2.12 spins 100 % CPU after the first GPU op")
  is closed: fixed by rocm-systems PR 7898 (backoff in the `AsyncEventsLoop` of libhsa-runtime64), in TheRock
  nightlies since 2026-08-22, so in 10.1.0. That library comes from the image's `/opt/rocm` (the stand-in loads
  torch's libs from there). The test script checks the idle CPU of the container.

## Decision: torch 2.12.0, triton stays 3.6.0, torchvision 0.27.0, aiter unchanged

- torch 2.12.0 is the lowest AMD ships for 10.1 and the only step that does not move further than necessary
  (2.13 / 3.7.1 / torchvision 0.28 is the combination that hung the GPU under load in 0.5.0-0.5.4).
- torchvision follows torch (0.27.0 for 2.12, compiled from source against it as before).
- triton: PyPI 3.6.0 kept (one moving part less in the A/B; its LLVM is bundled, it only dlopens libamdhip64).
  Upstream pairs torch 2.12 with triton 3.7.x; the next step if inductor complains is `TRITON_VERSION=3.7.0` +
  `TRITON_SHA256=8f111161d49bf903c0eaedde3962353a3d841c08a836839b7cc1025b8426efcf`.
- aiter stays 0.1.22.post1 (held, unrelated to ROCm; AITER JIT-compiles with the image's hipcc 10.1 on first use).
- libr4d and the radiance HIP extensions are compiled in the `kernels` stage with the base image's hipcc and
  `--offload-arch=gfx1201`; no flag change needed, but the compiler is a new major (clang 23 -> 24): the build log
  tells if a kernel fails; spill/scratch numbers and speed of the .so files are only known after the GPU test.
- Renovate: `ROCM_BASE` is pinned by tag+digest (`rocm/dev-ubuntu-24.04`, own rule "major/minor is a toolchain change").
  The torch cap `allowedVersions` moved from `2.11.x` to `2.12.x` and its text no longer says "until fixed".
  `TORCH_AMD_ROCM` (10.1.0) is not Renovate-managed; it must be changed together with `ROCM_BASE`.

## Build result (2408, CPU only, nice 10)

First attempt failed in the `stack` stage: ROCm 10.1 ships amdsmi's Python package only as a bare relocatable tree
(`/opt/rocm/share/amd_smi/amdsmi`, no setup.py / pyproject), so `pip install /opt/rocm/share/amd_smi` died. Fix: when
`setup.py` is absent a `amdsmi_rocm.pth` puts `/opt/rocm/share/amd_smi` on sys.path (the wrapper then finds
`<root>/lib/libamd_smi.so.27` itself; the .pth sorts before `radiance_amdsmi.pth`), plus a build-time assertion that the
library actually loaded (the wrapper otherwise degrades silently to a `_MissingLibrary`). Second build: **OK**, ~8 min
with the cached builder stages, image `vllm-sly-radiance:0.8.0-rc-rocm10.1`, 5.62 GB (0.7.0: 5.28 GB; the 10.1 ROCm tree
is 0.7 GB larger before pruning). All sly patches applied (anchors held on torch 2.12). Release-stage check:
`stack OK | vllm 0.30.0 | torch 2.12.0+rocm10.1.0 | aiter 0.1.22.post1 | torchvision 0.27.0 | triton 3.6.0 | transformers 5.18.0 | r4d 0.5.0`.
`import torch` with `HIP_VISIBLE_DEVICES=-1` reports HIP 7.16.26385. Not run: any GPU code.

## Changes in this branch

- `Dockerfile`: amdsmi install (see build result), `ROCM_BASE` 10.1.0-full@sha256, `TORCH_VERSION=2.12.0`, `TORCHVISION_VERSION=0.27.0`,
  `TORCH_AMD_ROCM=10.1.0`, comments (stack line, why 2.12, how to go back to 10.0).
- `renovate.json` (torch cap 2.12.x), `ci/release_tag.sh` (release body named `-rocm10.0` hard-coded; now derived from
  `ARG ROCM_BASE` at the tagged commit like `build.yml` does), `.github/workflows/build.yml` (comment),
  `docs/DEVELOPMENT.md`, `docs/TECHNICAL.md`, `DOCKERHUB.md`, `sly/README.md`: ROCm version / tag suffix `-rocm10.1`.
- NOT touched (main session): VERSION, CHANGELOG, README (its `ROCm | 10.0` table row and `0.4.6-rocm10.0` pull
  examples), issue-template placeholders, historic baselines in `ci/accept/baselines/`.
- `tests-lessons/F/test.sh` + `greedy.py` (copy in `/root/lessons/F/`).
- Image tag for the CI: `build.yml`/`accept.yml` derive `-rocm10.1` from `ROCM_BASE`, nothing else to change. Note that
  `ci/torch_key.py` changes the key, so the `ghcr.io/.../vllm-sly-radiance-torch` source-build wheel is not used by
  default (the AMD wheel is); `ci/check_consistency.py` needs the VERSION bump.

## Proposed CHANGELOG text

```
## [0.8.0] - <date>

### Changed
- **ROCm 10.1.** Base image `rocm/dev-ubuntu-24.04:10.1.0-full` (HIP 7.16, clang 24) instead of 10.0.0; the image tag is
  `<version>-rocm10.1`. PyTorch is AMD's `torch==2.12.0+rocm10.1.0` wheel (`amd-torch-device-gfx1201`) -- AMD ships no
  2.11 build for 10.1, so torch moves 2.11 -> 2.12 and torchvision 0.24.1 -> 0.27.0 (compiled against it). triton stays
  3.6.0 (PyPI), aiter 0.1.22.post1, vLLM 0.30.0. The torch 2.12 CPU-spin bug (ROCm/ROCm#6406) that held torch on 2.11 is
  fixed in ROCm 10.1; Renovate's torch cap is now 2.12.x.
- To stay on ROCm 10.0 build with `--build-arg ROCM_BASE=rocm/dev-ubuntu-24.04:10.0.0-full TORCH_AMD_ROCM=10.0.0
  TORCH_VERSION=2.11.0 TORCHVISION_VERSION=0.24.1`.

### Fixed
- `ci/release_tag.sh` named the image `-rocm10.0` in every release body regardless of the base.

### Measured
- <fill in from tests-lessons/F/test.sh: KV pool, second-start time, step ms, tok/upd, TTFT, prefill 2k/8k, idle CPU>
```

## Risks

- torch 2.12 + triton 3.6.0 is not an upstream-tested pair; `patch_dynamo_metrics` (torch/_dynamo/utils.py) and
  `patch_gfx1201` (triton driver.py) anchors may drift; the Docker build runs every patch and fails loudly.
- New compiler (clang 24): radiance HIP kernels / aiter JIT may change register use and speed; compare ms/step.
- First start on 10.1 recompiles everything (own cache dirs in the test; production cache dirs hold 10.0 artefacts, mount
  fresh ones or accept the double start on rollout).
- The #6406 fix is inferred from the issue and TheRock's nightly timeline (10.1.0a20260822 nightly had it); the
  test measures it, it has not been run.
- torch 2.13/2.14 exist for 10.1; moving to them is a separate A/B.
