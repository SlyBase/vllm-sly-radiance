# vllm-radiance: vLLM/torch/triton/aiter stack for RDNA4 (gfx1201 / R9700), plus the radiance
# patches and kernels. Single multistage build on the official AMD ROCm image, in five stages:
#   1. builder    compile torch/triton/torchvision/aiter/vLLM from source into /wheels
#   2. rocmprune  cut the 19 GB ROCm tree down to this one GPU architecture
#   3. assemble   install the wheels, apply the patches, build the R4D kernel library
#   3b. venvsplit split the venv into a cold (stack) and a hot (radiance) layer
#   4. final      the release image: a clean Ubuntu with only the pruned ROCm and the venv
# No prebuilt component wheels and no checked-in binaries. The release image carries neither the
# build toolchain nor the wheels, which is most of the reason it is far smaller than the base.
#
# stack: torch 2.14.0, triton 3.8.0, torchvision 0.24.1, aiter v0.1.21.post2, vLLM v0.29.0,
# all compiled for PYTORCH_ROCM_ARCH=gfx1201 against the base image's ROCm 10.0 (the default
# ROCM_BASE below is what the homelab's production image is built from; 7.14 needs --build-arg).
# renovate: datasource=docker depName=rocm/dev-ubuntu-24.04 versioning=regex:^(?<major>\d+)\.(?<minor>\d+)\.(?<patch>\d+)-full$
ARG ROCM_BASE=rocm/dev-ubuntu-24.04:10.0.0-full@sha256:a90cf047f615abe70fbef83c64def0a2d549ef37a39c8ea545430aba4981b374
ARG GFX_ARCH=gfx1201
# The release stage starts from a clean distro image rather than the ROCm base, and COPYs in only
# the pruned ROCm tree plus the venv. Same Ubuntu release as the ROCm base (24.04), so the venv's
# interpreter (python 3.12.3) matches.
# renovate: datasource=docker depName=ubuntu versioning=ubuntu
ARG RELEASE_BASE=ubuntu:24.04@sha256:008173c23f95b170204355c12626cb5a965d779a7e1283b09e9cffbb1bf33ca3

# Component pins, in one place. Each is both the git tag that gets compiled and the version the
# resulting wheel reports, so `pip show`, `importlib.metadata`, and the startup banner all agree
# with what was actually built.
# torch/triton/torchvision are NOT free choices, and the number to read is not the one in
# pyproject.toml. At 0.27.1 upstream's own ROCm image (docker/Dockerfile.rocm_base) built
# PYTORCH_BRANCH=release/2.11 with torchvision v0.24.1 and triton 3.6.0 -- the sanctioned trio this
# fork used to build against, and requirements/rocm.txt itself pins no torch at all.
# RE-VERIFIED AT THIS BUMP (2026-09, fetched docker/Dockerfile.rocm_base from the v0.29.0 tag
# directly): upstream's OWN rocm_base Dockerfile has since moved to PYTORCH_BRANCH="6bbd260"
# (release/2.12 as of 08/02), TRITON_BRANCH="f0b55c0" (release/internal/3.7.x as of 08/18) and
# AITER_BRANCH="v0.1.19" -- none of which is torch 2.14.0 / triton 3.8.0 / aiter 0.1.21.post2.
# This bump is therefore, again, a combination upstream does not itself test on ROCm -- the exact
# risk category that made 0.5.0-0.5.4 hang the GPU under load (torch 2.13 / triton 3.7.1 /
# torchvision 0.28, reachable only because `use_existing_torch.py` strips the pin). Carried forward
# anyway because it is what this fork's target stack (vLLM 0.29.0 + newer aiter for gfx1201 MXFP4)
# needs; flagged here rather than silently assumed safe -- watch for the same symptom (fluent
# startup, hang under load) and be ready to fall back to upstream's own pinned trio if it appears.
# renovate: datasource=github-releases depName=pytorch/pytorch extractVersion=^v(?<version>\d+\.\d+\.\d+)$
ARG TORCH_VERSION=2.14.0
# renovate: datasource=github-releases depName=triton-lang/triton extractVersion=^v(?<version>\d+\.\d+\.\d+)$
ARG TRITON_VERSION=3.8.0
# renovate: datasource=github-releases depName=pytorch/vision extractVersion=^v(?<version>\d+\.\d+\.\d+)$
ARG TORCHVISION_VERSION=0.29.0
# renovate: datasource=github-tags depName=ROCm/aiter versioning=pep440 extractVersion=^v(?<version>.+)$
ARG AITER_VERSION=0.1.23
# renovate: datasource=github-releases depName=vllm-project/vllm extractVersion=^v(?<version>\d+\.\d+\.\d+)$
ARG VLLM_VERSION=0.30.0
# transformers is pinned here because vLLM does not pin it: requirements/common.txt asks only for
# `transformers >= 5.5.3`, so an unpinned rebuild silently picks up whatever is newest and the
# stack changes underneath the build. 5.15.0 made Gemma-4's head_dim a per-layer attribute and
# turned the global read into AmbiguousGlobalPerLayerAttributeError, which no released vLLM config
# convertor handled at the time -- a Gemma-4 checkpoint then failed during argument parsing, before
# a model or an attention backend exists. 5.14.1 was the last release before that change.
# BUMPED to 5.17.0 for the vLLM 0.29.0 stack. NOT independently re-verified against a live Gemma-4
# checkpoint load (no GPU in this environment) -- static check only: the exact
# `AmbiguousGlobalPerLayerAttributeError` class name is absent from configuration_utils.py and
# configuration_gemma3.py at the v5.17.0 tag, and configuration_utils.py's per-layer-config handling
# (`per_layer_config` / HeterogeneousConfigMixin) reads as a broader rewrite of that mechanism, not
# merely a patch on top of 5.15.0's -- consistent with the bug having been designed away rather than
# left in place, but this is inference from source shape, not a passing load test. Re-check this the
# first time a Gemma-4 checkpoint is actually served on this image.
# renovate: datasource=pypi depName=transformers versioning=pep440
ARG TRANSFORMERS_VERSION=5.17.0
# rocm-bandwidth-test for the startup topology/bandwidth sweep. Pinned to the NEWEST tag that still
# has a plain CMakeLists: the rocm-7.x tags moved to a cmake framework that demands clang>=19 on PATH
# plus vendored boost/fmt/curl submodules, none of which this tool needs.
# renovate: datasource=github-tags depName=ROCm/rocm_bandwidth_test versioning=regex:^rocm-(?<major>\d+)\.(?<minor>\d+)\.(?<patch>\d+)$
ARG RBT_VERSION=rocm-6.4.4
# R4D: the HIP kernel library for this GPU -- attention, gated delta net, all-reduce, a skinny bf16
# GEMM and (since this pin) the OCP-MXFP4 x fp8 skinny GEMM gemm_mxfp4a8_nt_m64. It is a library of
# gfx1201 kernels rather than a part of this image, so it lives in its own repository and is pinned
# here like any other component; R4D_REPO exists so a fork or a local mirror can be substituted
# without editing the build.
# R4D_VERSION is a raw commit SHA, not a tag -- see the full rationale and the matching clone/
# assertion change at the point of use in the assemble stage (grep this file for "libr4d"). Short
# version: no tag has gemm_mxfp4a8_nt_m64 yet (`git ls-remote --tags` still shows only v0.4.0 and
# v0.5.0), so this pins to a fixed SHA on main rather than to the moving branch name -- a moving
# branch as a pin has already run a production service into a hanging download once in this
# homelab, which is the entire reason this is a SHA and not just `main`.
ARG R4D_REPO=https://codeberg.org/StillDeadcode/libr4d.git
# renovate: datasource=git-refs depName=https://codeberg.org/StillDeadcode/libr4d.git branch=main (digest pin; keep R4D_REPO above in sync)
ARG R4D_VERSION=5dc6302b87d598d1d3bf2ad3b50aab365461a63c

# =====================================================================================
# STAGE 1 builder: compile the stack from source into /wheels
# =====================================================================================
FROM ${ROCM_BASE} AS builder
ARG GFX_ARCH
ARG TORCH_VERSION
ARG TRITON_VERSION
ARG TORCHVISION_VERSION
ARG AITER_VERSION
# Build parallelism, as an ARG so a build can be told to leave the box some headroom:
#   docker build --build-arg MAX_JOBS=16 .
# Lowered from 32 (effectively 14, one per LXC core): this host has only 29 GB physical RAM shared
# with a 6 GB VM and another LXC, on top of this LXC's own 24 GB memory cgroup -- already
# oversubscribed on paper before any build runs. A wide ninja job here does not just risk a
# cgroup-local OOM-kill, it can push the WHOLE HOST into a global OOM (kernel picks victims
# system-wide, "global_oom" in dmesg, observed killing unrelated processes in other cgroups) and
# has crashed pve3 outright more than once. 4 keeps every stage's worst-case peak well under budget.
ARG MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive \
    PYTORCH_ROCM_ARCH=${GFX_ARCH} \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
    USE_ROCM=1 USE_CUDA=0 MAX_JOBS=${MAX_JOBS} CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS}

# Build tooling the base dev image lacks (git/venv/pkg-config + the -dev packages torch's cmake
# probes: libdrm for rocm_smi, libnuma, libelf).
RUN apt-get update && apt-get install -y --no-install-recommends \
      git python3.12-venv build-essential ccache pkg-config \
      libdrm-dev libnuma-dev libelf-dev \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/py
ENV PATH=/opt/py/bin:$PATH
# setuptools-scm is a build requirement of aiter and vLLM (both take their version from the git
# tag). --no-build-isolation means it is NOT auto-installed: without it here setuptools silently
# ignores `use_scm_version` and the wheel is stamped 0.0.0.
RUN pip install -U pip wheel setuptools "setuptools-scm>=8.0" "cmake<4" ninja pybind11 numpy \
      pyyaml typing_extensions cffi requests build
RUN mkdir -p /wheels

# --- torch (AOTriton off: its gfx1201 source-configure fails and vLLM never uses torch
#     SDPA-flash; USE_MAGMA=0: base has no magma) ---
# Cloning is its own layer, split from the ~90min compile below: --recurse-submodules pulls
# dozens of third-party repos one after another, and any single flaky GitHub fetch aborts the
# whole clone (observed in practice -- transient, not a real outage, but git only retries a
# given submodule once before giving up). A shallow submodule clone can't be resumed in place,
# so each retry starts clean; `test -d .../.git` makes the RUN fail loudly if all attempts do.
RUN for i in 1 2 3 4 5; do \
        rm -rf /src/pytorch; \
        git clone --depth 1 -b v${TORCH_VERSION} --recurse-submodules --shallow-submodules \
            https://github.com/pytorch/pytorch.git /src/pytorch && break; \
        echo "torch clone attempt $i failed, retrying in 10s..." >&2; sleep 10; \
    done; test -d /src/pytorch/.git
# MAX_JOBS caps ninja's parallelism (default: one job per core, 14 here). Some of ATen's
# generated Register*.cpp translation units are heavy enough under GCC (3-4GB+ RSS each) that
# 14-wide hit the LXC's 24GB cgroup limit and the kernel OOM-killer took cc1plus out mid-build
# (confirmed via dmesg on the pve host) -- silent from ninja's side, just "subcommand failed"
# with no compiler error above it.
# 6 was NOT enough of a cut: it still OOM-killed on the same file (RegisterCompositeExplicitAutograd)
# in two separate attempts, and raising the LXC's own swap ceiling to compensate was the wrong lever
# -- it let this one cgroup's virtual footprint (mem + swap) exceed the ENTIRE host's physical
# budget once the co-resident VM and other containers are counted, so instead of a contained,
# cgroup-local OOM-kill, the kernel's OOM killer went "global_oom" and started taking down
# unrelated processes / the host itself (repeated pve3 crashes, confirmed via dmesg across boots).
# 3 is the actual fix: keep peak concurrent heavy-file RSS low enough that this LXC never gets
# close to its own 24 GB ceiling in the first place, so any OOM (if it still happens) stays local.
RUN cd /src/pytorch \
    && pip install -r requirements.txt \
    && python tools/amd_build/build_amd.py \
    && USE_MAGMA=0 USE_MKLDNN=1 BUILD_TEST=0 USE_NCCL=1 USE_RCCL=1 \
       USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_AOTRITON=0 \
       PYTORCH_BUILD_VERSION=${TORCH_VERSION}+rocm7.14 PYTORCH_BUILD_NUMBER=1 \
       MAX_JOBS=3 \
       python -m build --wheel --no-isolation --outdir /wheels . \
    && pip install /wheels/torch-*.whl && rm -rf /src/pytorch

# --- triton ---
RUN git clone --depth 1 -b v${TRITON_VERSION} https://github.com/triton-lang/triton.git /src/triton \
    && cd /src/triton && pip wheel --no-build-isolation --no-deps . -w /wheels \
    && pip install /wheels/triton-*.whl && rm -rf /src/triton

# --- torchvision ---
# FORCE_CUDA=1 is REQUIRED: torchvision's BUILD_CUDA_SOURCES gates on torch.cuda.is_available(),
# which is false in `docker build` (no GPU) -> it picks CppExtension, where torch's build-time hipify
# double-compiles vision.cpp + vision_hip.cpp -> "multiple definition of vision::cuda_version()".
# FORCE_CUDA=1 forces CUDAExtension (correct hipify source replacement); hipcc needs no GPU to compile.
RUN git clone --depth 1 -b v${TORCHVISION_VERSION} https://github.com/pytorch/vision.git /src/vision \
    && cd /src/vision && FORCE_CUDA=1 USE_ROCM=1 pip wheel --no-build-isolation --no-deps . -w /wheels \
    && rm -rf /src/vision

# --- aiter (gfx1201; kernels JIT at runtime, PREBUILD_KERNELS=0) ---
# PRETEND_VERSION: the checkout is shallow, so setuptools-scm cannot describe the tag and would fall
# back to a placeholder version; pin it to the tag being built.
RUN git clone --recursive --shallow-submodules -b v${AITER_VERSION} https://github.com/ROCm/aiter.git /src/aiter \
    && cd /src/aiter && GPU_ARCHS=${GFX_ARCH} PREBUILD_KERNELS=0 \
       SETUPTOOLS_SCM_PRETEND_VERSION=${AITER_VERSION} \
       pip wheel --no-build-isolation --no-deps . -w /wheels \
    && pip install /wheels/*aiter-*.whl && rm -rf /src/aiter

# --- vLLM, built against the torch above. use_existing_torch strips the torch/torchvision pins so
#     pip does not try to fetch them; the versions built above ARE the pinned ones, so this is now
#     just "use what was compiled above", not an override.
#     setuptools-rust is a pyproject build requirement that --no-build-isolation does not install.
#     VLLM_VERSION_OVERRIDE pins the reported version to the tag: the tree is dirty (use_existing_torch
#     rewrites the requirements files) and shallow, so setuptools-scm would otherwise stamp the wheel
#     with a guessed next-release dev version plus the build date. ---
# ARG at the point of use, not at the top of the stage: an ARG line is a cache-key instruction, so
# declaring it up there would make a vLLM bump rebuild torch, triton, torchvision and aiter too.
ARG VLLM_VERSION
RUN git clone --depth 1 -b v${VLLM_VERSION} https://github.com/vllm-project/vllm.git /src/vllm \
    && cd /src/vllm && python use_existing_torch.py \
    && pip install "setuptools-rust>=1.9.0" \
    && VLLM_TARGET_DEVICE=rocm VLLM_VERSION_OVERRIDE=${VLLM_VERSION} \
       pip wheel --no-build-isolation --no-deps . -w /wheels \
    && rm -rf /src/vllm
RUN ls -la /wheels

# --- rocm-bandwidth-test (the startup topology + bandwidth sweep) ---
# ARG is declared HERE, not in the stage's opening block: an ARG line is a cache-key instruction, so
# putting it up there would invalidate every layer below it -- including the PyTorch compile.
ARG RBT_VERSION
# Plain cmake against the HSA headers/libs the base image already ships; no extra build deps (the
# builder venv's pip cmake and the apt build-essential above cover it). The RUNPATH is REQUIRED:
# ROCm 7.14 keeps libhsa-runtime64.so under the versioned component dir (/opt/rocm/core-<ver>/lib,
# reachable via the `core` alternatives symlink), which is not on the loader's default search path,
# so an un-rpathed binary dies with "libhsa-runtime64.so.1: cannot open shared object file".
RUN git clone --depth 1 -b ${RBT_VERSION} https://github.com/ROCm/rocm_bandwidth_test.git /src/rbt \
    && cmake -S /src/rbt -B /src/rbt/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm \
         -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,/opt/rocm/core/lib:/opt/rocm/lib" \
    && cmake --build /src/rbt/build -j 16 \
    && mkdir -p /artifacts && cp /src/rbt/build/rocm-bandwidth-test /artifacts/ \
    && rm -rf /src/rbt

# =====================================================================================
# STAGE 2 rocmprune: cut the ROCm tree down to this image's single GPU architecture
# =====================================================================================
# ~19 GB of the base is device code for GPUs this image cannot run on, plus link-time-only
# archives. Pruning has to happen in a stage that the release stage COPYs FROM: deleting files
# in a layer stacked on the base reclaims nothing, it only writes whiteouts. See prune_rocm.sh
# for what is kept and why (the runtime still has to compile: AITER JITs kernels on first use).
FROM ${ROCM_BASE} AS rocmprune
ARG GFX_ARCH
COPY prune_rocm.sh /tmp/prune_rocm.sh
RUN bash /tmp/prune_rocm.sh ${GFX_ARCH} && rm -f /tmp/prune_rocm.sh

# =====================================================================================
# STAGE 3 assemble: install the wheels and apply the patches and kernels
# =====================================================================================
# Runs on the FULL base because it needs the toolchain (hipcc, headers, static archives) to
# compile the HIP kernels. Only the resulting /opt/vllm venv is carried into the release image.
FROM ${ROCM_BASE} AS assemble
ARG GFX_ARCH
ARG AITER_VERSION
ARG VLLM_VERSION
ARG TRANSFORMERS_VERSION
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12-venv git \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/vllm
RUN python3 -m venv /opt/vllm
ENV PATH=/opt/vllm/bin:$PATH
ENV SP=/opt/vllm/lib/python3.12/site-packages

# --- install the wheels ---
# torch/triton/vision/aiter with --no-deps so pip does not replace them; vLLM with its pure-python
# dependencies. amdsmi (the ROCm python bindings the base image ships) is required for vLLM's ROCm
# platform detection. The transformers pin goes in the SAME pip invocation as the vLLM wheel so the
# resolver sees it as a constraint -- installing it afterwards would first pull the newest release
# and then downgrade it, leaving both in the layer.
# Everything that comes from PyPI is pinned by constraints.txt (see its header; Renovate keeps it
# current, ci/check_constraints.py keeps it complete): without it a rebuild that misses the cache
# resolves vLLM's open ranges to whatever is newest that day.
COPY --from=builder /wheels /wheels
COPY constraints.txt /tmp/constraints.txt
RUN pip install --no-cache-dir -U pip wheel setuptools -c /tmp/constraints.txt \
 && pip install --no-cache-dir --no-deps \
      /wheels/torch-*.whl /wheels/triton-*.whl /wheels/torchvision-*.whl /wheels/*aiter-*.whl \
 && pip install --no-cache-dir -c /tmp/constraints.txt \
      /wheels/vllm-*.whl "transformers==${TRANSFORMERS_VERSION}" \
 && pip install --no-cache-dir -c /tmp/constraints.txt /opt/rocm/share/amd_smi pillow pybind11 \
 && rm -rf /wheels /root/.cache /tmp/constraints.txt

# RADIANCE_GFX_ARCH is what the gfx1201 patch and the banner read for the target arch (amdsmi's
# asic_info reports it empty on this card). It used to be called VLLM_ROCM_GCN_ARCH, which vLLM
# 0.26 flags as an unknown VLLM_* variable at startup; the old name is still honored.
ENV ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=${GFX_ARCH} RADIANCE_GFX_ARCH=${GFX_ARCH} \
    HIP_ARCHITECTURES=${GFX_ARCH} AMDGPU_TARGETS=${GFX_ARCH} GPU_ARCHS=${GFX_ARCH} \
    SAFETENSORS_FAST_GPU=1 TOKENIZERS_PARALLELISM=false TRITON_CACHE_AUTOTUNING=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- trim, strip and freeze the installed stack (the release image's cold venv layer) ---
# Everything below this point is radiance's own work on top of the stack; this step closes the
# part that changes only with a stack bump, and snapshots it for split_venv.py (see the venvsplit
# stage), which later ships the unchanged part as its own layer.
#   * triton's NVIDIA backend binaries (ptxas, cupti, ~360 MB): this triton only ever targets HIP.
#     The backend's Python modules stay -- triton imports every backend at load time.
#   * aiter's prebuilt assembly kernels for other archs (hsa/gfx942, gfx950, gfx1250, ~115 MB):
#     aiter looks them up under hsa/<device arch>, and there is none for gfx1201.
#   * debug symbols of the installed extensions (worth ~1 GB): release builds, but they still carry
#     .debug_* sections that nothing reads at runtime. The radiance kernels built further down
#     (R4D, the MXFP4 GEMM, the GDN decode kernel) are not stripped: tiny, and they carry device fatbins.
COPY split_venv.py /opt/split_venv.py
RUN set -eu; \
    rm -rf ${SP}/triton/backends/nvidia/bin ${SP}/triton/backends/nvidia/lib; \
    find ${SP}/aiter_meta/hsa -mindepth 1 -maxdepth 1 -type d -name 'gfx*' ! -name "${GFX_ARCH}" \
      -exec rm -rf {} +; \
    find /opt/vllm -type f -name '*.so*' -exec strip --strip-unneeded {} + 2>/dev/null || true; \
    find /opt/vllm -name '__pycache__' -type d -prune -exec rm -rf {} +; \
    /usr/bin/python3 /opt/split_venv.py snapshot /opt/vllm /opt/vllm.stable.json

# --- runtime modules and configs ---
# radiance_amdsmi.py and .pth: amdsmi init-order fix. amdsmi must init before HIP at site-init in
# every process, otherwise it enumerates 0 devices and platform detection fails.
COPY radiance_amdsmi.py radiance_amdsmi.pth \
     radiance_kernels.py radiance_vit_attn.py radiance_allreduce.py \
     radiance_draft.py radiance_draft_gpu.py radiance_drafthead.py radiance_gemm.py \
     radiance_r4d_attn.py radiance_gdn.py radiance_w4.py sly/mxfp4/radiance_mxfp4.py \
     sly/mxfp4/radiance_lmhead_fp8.py sly/mxfp4/radiance_lmhead_int4.py \
     sly/mxfp4/radiance_fused_norm.py sly/mxfp4/radiance_embed_int8.py sly/radiance_attn_decode.py \
     sly/radiance_attn_drafter.py sly/radiance_lookup_draft.py sly/gdn/radiance_gdn_decode.py \
     radiance_tp3pad.py radiance_nvfp4.py radiance_autoround.py radiance_escha.py \
     paroquant/radiance_paroquant.py paroquant/radiance_paroquant_mxfp4.py \
     sly/quant/radiance_quant_plugins.py sly/quant/radiance_quant_plugins.pth ${SP}/
COPY fp8-configs/ ${SP}/vllm/model_executor/layers/quantization/utils/configs/
COPY moe-configs/ ${SP}/vllm/model_executor/layers/fused_moe/configs/
# aiter's Triton GEMM-AFP4WFP4 (patch_quark_mxfp4.py's relaxed CDNA gate makes this reachable on
# gfx12x) hard-asserts on a missing DEFAULT.json -- there is no built-in fallback, and aiter ships
# no gfx1201 tuning for this GEMM family at all (only gfx950/gfx1250). The values here are aiter's
# own gfx950 and gfx1250 DEFAULT.json, which are byte-identical to each other across those two very
# different architectures -- evidence this is already aiter's generic/portable default rather than
# a per-arch tuning, so reusing it on gfx1201 is functionally correct, just unmeasured on this card.
# Real R9700 numbers are Task #7 (kernel-level MXFP4 profiling) work; until then this only affects
# the AITER fallback path for shapes the RadianceMxfp4W4A8LinearKernel hip kernel declines (M<=256,
# i.e. decode), since prefill's large-M shapes go through that kernel instead.
COPY sly/mxfp4-configs/ ${SP}/aiter/ops/triton/configs/

# --- gfx1201 fixes and tuned-kernel patches ---
# Each patch edits a vLLM (or aiter/triton) source file in place and checks for source drift before
# writing. patch_gdn_wmma covers the solve_tril triangular block-inverse only; the gated-delta-net
# gram cast is handled upstream since 0.26.0.
# patch_conv1d_blockn widens the gated-delta-net prefill conv1d channel block to a 16-byte-per-lane
# access; bit-identical, and it defuses a 2**14-byte row pitch the caller's split() view creates.
# patch_r4d is the whole libr4d integration in one patch, switchable at run time with
# RADIANCE_USE_R4D (see patch_r4d.py's own docstring for the three integration points).
# patch_dflash_fused_kv_fp8 lets that drafter be an fp8 checkpoint: the context-KV precompute
# fuses every layer's K/V projection by slicing the raw parameter, which is neither the right
# dtype nor the right row layout once the weights are quantized and preshuffled.
# patch_gdn_metadata cuts the per-step Python cost of building the gated-delta-net attention
# metadata: the per-request bookkeeping runs as one numpy pass over the same buffers, the arange
# and empty index become slices of cached buffers, and the block table is sliced rather than
# gathered when every sequence is a spec decode. Byte-identical output; RADIANCE_GDN_META=0 falls
# back to the stock path.
# patch_dflash_w4 marks the drafter's weight load so radiance_w4 can pack it to 4 bits -- the
# drafter's linears and the target's are indistinguishable inside a quant method's callback, and
# the drafter is loaded by exactly one call, so bracketing that call is the whole discriminator.
# Inert unless RADIANCE_FAST_DRAFT=1, the one switch over the whole tuned drafter stack: the draft
# pass falls 9.1% at a drafter batch of 64, and the acceptance question is settled -- acceptance on
# this stack is bimodal and the mode is drawn per compile, in the control arm too, so a single
# sample per arm reads a coin flip as a tax.
# RADIANCE_USE_R4D: the R4D attention backend enum, plus the gated-delta-net layer, where a whole
# step runs in five hand-written kernels (conv+prep+gating+cumsum, the K-gram with its triangular
# inverse, and the chunked scan on the prefill path; the conv update and the recurrent state
# update on the decode path), and the FLA chunk path still gets the fused scan for any step shape
# the layer hook declines. The two patches above tune that FLA path, which is what runs when
# RADIANCE_USE_R4D=0.
# DROPPED at the vLLM 0.29.0 bump, now native upstream (anchor-by-anchor verification in
# sly/README.md): patch_dflash2 (DFlash2 speculative decoding, vllm-project/vllm#52816, merged into
# vLLM itself ten days after 0.27.1 was tagged) and patch_dflash_base (the three DFlash correctness
# fixes DFlash2 depended on, including full cp_rank/cp_local_slot() context-parallelism support) --
# both now live in vllm/v1/worker/gpu/spec_decode/dflash/speculator.py. patch_radiance_fusion is
# also dropped: AiterRMSNormQuantFusionPass.is_rdna_aiter_enabled() (vllm/compilation/passes/fusion/
# rocm_aiter_fusion.py, vllm/_aiter_ops.py) now reaches the same fusion automatically.
# sly/patch_quark_mxfp4.py puts the RadianceMxfp4W4A8LinearKernel plugin (sly/mxfp4/radiance_mxfp4.py
# + the radiance_mxfp4_fp8.hip kernel built below) at the head of ROCm's MXFP4 kernel list, and
# separately relaxes aiter's CDNA-only MXFP4 gates so its own Triton gemm_afp4wfp4 path is reachable
# on gfx1201 -- RADIANCE_MXFP4=1 for the aiter path, RADIANCE_MXFP4_W4A8=1 for the hand-written
# kernel layered on top; see sly/README.md for why both are needed rather than either alone.
# sly/patch_short_prefill.py fixes a gated-delta-net metadata-classification bug where a 1-token
# prefill (the common case for a cache-hit continuation) is misclassified as a decode step.
# sly/patch_dflash_w4_packed.py lets the DFlash drafter be a compressed-tensors W4A16 checkpoint
# (qkv_proj has weight_packed, no raw .weight -- deferred + dequantised like the fp8 case);
# sly/patch_gdn_nonspec_mask.py defines non_spec_sequence_masks_cpu on patch_gdn_metadata's numpy
# path (UnboundLocalError at engine init as soon as --speculative-config is set). Both are
# prerequisites for DFlash2 on the MXFP4 target (vllm7), verified 2026-09-15.
# sly/patch_lmhead_fp8.py hooks radiance_lmhead_fp8.py into QuarkConfig.get_quant_method: with
# RADIANCE_LMHEAD_FP8=1 the (Quark-excluded, otherwise bf16) lm_head is quantised to fp8 per
# output channel after loading and applied via row-wise torch._scaled_mm -- halves the 2.54 GB
# of weight traffic that the target verify and the DFlash draft each pull per step.
# sly/patch_w4a16_tiles.py adds a gfx1201 per-shape tile table to rdna_hybrid_w4a16.py for the
# DFlash2 W4A16 drafter (the stock gfx12x heuristic was tuned on Llama-3.1-8B shapes and picks
# 16x16 tiles at M<=32 -- 2176 workgroups for the 34816-wide gate_up); measured with
# sly/bench_w4a16_tiles.py, RADIANCE_W4A16_TILES=0 falls back to the heuristic.
# sly/patch_lmhead_int4.py hooks radiance_lmhead_int4.py in front of the fp8 block: with
# RADIANCE_LMHEAD_INT4=1 the lm_head is quantised to int4 (group-128 bf16 scales, MSE clip
# search) after loading and applied on the drafter's W4A16 kernel path (656 MB per call instead
# of fp8's 1.27 GB; the tile table above carries the lm_head entries). Must run after
# patch_lmhead_fp8 (it anchors on that block).
# sly/patch_lmhead_int4_ct.py puts the same hook into CompressedTensorsConfig.get_quant_method, for a
# compressed-tensors W4A16 target (RedHatAI/Qwen3.8-27B-INT4, lm_head in `ignore` = bf16); the DFlash
# drafter's head has no quant_config and is shared, so only the target head changes.
# sly/patch_fused_norm_quant.py hooks radiance_fused_norm.py (0.1.6, RADIANCE_FUSED_NORM_QUANT=1,
# default off) into Qwen3.5/Qwen3-Next: the decoder add+rms_norm, the MLP silu*up and the GDN
# gated norm each end in the per-token fp8 quant (radiance_add_rms_quant / _silu_mul_quant /
# _gdn_norm_quant in the .hip below) and hand (q, scale) straight to the W4A8 GEMM's pq entry,
# plus the knob in vLLM's compile cache key. Anchors are the already-patched files, so it runs last.
# sly/patch_mamba_align_retire.py backports vllm#55450 (0.2.3): align-mode Mamba state retirement
# skips null gaps instead of stopping at them -- with async scheduling the stock code pinned one
# gated-delta-net block per group per prefill chunk until the request finished (258k prompt: KV
# pool exhausted, two self-preemptions).
# sly/patch_rocm_load_max_split.py lets Worker.load_model's max_split_size_mb:20 allocator scope run
# on ROCm (stock gates it on is_cuda()): without it, packed W4A16 weights were carved out of the
# freed 2.37 GiB bf16 embed/lm_head segments and pinned them (INT4 target k=7: 2.9 GiB stranded,
# 313k -> 386k KV tokens; MXFP4 prod: 376k -> 386k).
# patch_tp3_pad (ggz14) installs the three radiance_tp3pad.py hooks for TP=3 via zero-weight dummy
# heads; every hook returns immediately unless RADIANCE_TP_PAD=3, so TP 1/2 serves are unchanged.
# sly/patch_ar_knobs.py makes the TP=2 all-reduce size gate (RADIANCE_AR_MAX_KB) and the 6-bit
# wire geometry env-tunable; defaults are the shipped values.
# patch_nvfp4_mxfp4.py (ggz14, 2026-09-15) serves compressed-tensors NVFP4 checkpoints
# (unsloth/Qwen3.8-27B-NVFP4) by requantizing every linear to MXFP4 at load (radiance_nvfp4.py) and
# handing it to the same W4A8 kernel plugin the Quark checkpoint runs on. Inert unless
# RADIANCE_NVFP4_MXFP4=1; it only touches compressed-tensors scheme selection, never QuarkConfig.
COPY patch_*.py install_radiance_hooks.py _patchlib.py /opt/patches/
COPY sly/ /opt/patches/sly/
# PYTHONPATH=/opt/patches: `python sly/patch_quark_mxfp4.py` puts the SCRIPT's own directory
# (/opt/patches/sly), not cwd, at sys.path[0] -- without this, the sly/ entries' `from _patchlib
# import apply` would not find /opt/patches/_patchlib.py. The top-level entries already have
# /opt/patches as their own script directory, so this is a no-op for them.
RUN set -eu; cd /opt/patches; \
    for p in patch_gfx1201 patch_radiance_dispatch patch_skinny_gemm patch_unified_attention_lds \
             patch_gdn_wmma patch_preshuffle install_radiance_hooks \
             patch_unpad patch_mtp_mm_mask patch_mtp_loopbreak patch_qwen3_toolparse patch_from_json_filter \
             patch_dynamo_metrics patch_conv1d_blockn patch_r4d \
             patch_dflash_fused_kv_fp8 patch_dflash_w4 patch_gdn_metadata \
             sly/patch_quark_mxfp4 sly/patch_short_prefill \
             sly/patch_dflash_w4_packed sly/patch_gdn_nonspec_mask sly/patch_lmhead_fp8 \
             sly/patch_w4a16_tiles sly/patch_lmhead_int4 sly/patch_lmhead_int4_ct patch_nvfp4_mxfp4 \
             sly/patch_fused_norm_quant \
             sly/patch_kv_groups sly/patch_embed_int8 sly/patch_nvfp4_compile_key sly/patch_mamba_align_retire \
             sly/patch_rocm_load_max_split sly/patch_gdn_fused_decode \
             patch_tp3_pad sly/patch_ar_knobs patch_autoround patch_escha; do \
      echo "== applying $p =="; \
      PYTHONPATH=/opt/patches python "$p.py"; \
    done; \
    python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('${SP}/radiance_*.py')]; print('radiance modules parse OK')"

# --- R4D: the gfx1201 kernel library, cloned and compiled from source ---
# sly/r4d/r4d_extras_rx10.patch is ggz14's r4d_radiance_extras_rx10.patch (written against libr4d
# b9e42ab) rebased onto this pin: the kernel sources applied as-is, the four registry files
# (build.sh, r4d.h, r4d_module.hip, r4d_registry.hip) were merged by hand -- every conflict was
# additive, the pin's N-rank all-reduce family next to the extras. It adds the narrow-state (bf16 /
# fp16 SSM cache) gated-delta-net decode kernels that radiance_gdn.py binds when
# --mamba-ssm-cache-dtype is 16-bit, the lazy-snapshot GDN kernels (RADIANCE_GDN_LAZY), the fused
# GDN decode step (RADIANCE_GDN_FUSED_UPDATE), the 8-bit legs of the R4D prefill attention
# (R4D_ATTN_FP8, R4D backend only) and the TP=2 all-reduce with a fused decoder epilogue. ggz14's
# three-rank all-reduce is left out: it was written for the older two-rank radiance_allreduce.py,
# and this image ships StillDeadcode's N-rank module, which has no binding for it -- TP=3 all-reduces
# ride RCCL.
# One shared object holding every hand-written kernel this image runs: paged attention (prefill and
# decode, fp8 or bf16 KV), the fused gated-delta-net prefill scan, the TP=2 P2P all-reduce in both
# its exact and its 6-bit-packed form, the skinny bf16 GEMM, and (since this pin) the OCP-MXFP4 x
# fp8 skinny GEMM gemm_mxfp4a8_nt_m64 that sly/mxfp4/radiance_mxfp4.py's RADIANCE_MXFP4_R4D_DECODE_MAX_M
# path reads via `r4d.select("gemm_nt", ..., dtype="mxfp4a8")`. Built here rather than in the
# builder stage because it has to be compiled by the same hipcc the venv loads it against.
# ARGs are declared at the point of use: they are cache-key instructions, so putting them at the top
# of the stage would invalidate the wheel install above on every kernel bump.
ARG R4D_REPO
ARG R4D_VERSION
# R4D_VERSION is a commit SHA (see the ARG's own comment at the top of the file for why), so the
# clone can't use `git clone --depth 1 -b <ref>` -- that resolves tags and branches, not arbitrary
# SHAs. `git fetch <sha>` instead, confirmed against Codeberg's Gitea directly (a live
# `git fetch --depth 1 origin <sha>` against this exact repo returned the object, i.e.
# uploadpack.allowAnySHA1InWant is on) -- so this stays a shallow single-object fetch, not a full
# clone. This ALSO changes the shape of the correctness check below: the previous version compared
# `r4d.__version__` (from `#define R4D_VERSION "..."` in r4d.h) against the pinned tag, but that
# macro was NOT bumped by either of the two PRs merged into main after v0.5.0 -- confirmed via
# `git log -p -- r4d.h` against the live repo, it still reads "0.5.0" at this exact commit. Asserting
# a stale clone therefore now means asserting the CHECKED-OUT COMMIT equals the pin (what the
# original check was actually protecting against), not the self-reported semver string, which prints
# only as informational context alongside it.
RUN set -eu; mkdir -p /src/libr4d && cd /src/libr4d \
 && git init -q && git remote add origin ${R4D_REPO} \
 && git fetch --depth 1 origin ${R4D_VERSION} && git checkout -q FETCH_HEAD \
 && GOT=$(git rev-parse HEAD) \
 && [ "$GOT" = "${R4D_VERSION}" ] \
    || { echo "libr4d checked out $GOT, expected ${R4D_VERSION}" >&2; exit 1; } \
 && git apply /opt/patches/sly/r4d/r4d_extras_rx10.patch \
 && GFX_ARCH=${GFX_ARCH} OUT=${SP}/r4d.so ./build.sh \
 && python -c "import sys, torch, r4d; \
print('r4d commit', sys.argv[1], '(self-reported __version__', r4d.__version__ + ', not bumped ' \
      'upstream past 0.5.0 since the tag -- informational only) built:'); \
[print('   ', k['family'], k['name']) for k in r4d.kernels()]" "$GOT" \
 && cd / && rm -rf /src/libr4d

# --- RADIANCE MXFP4 W4A8: the hand-written fp8-WMMA GEMM kernel for gfx1201, compiled here for the
#     same reason as R4D above -- it has to link against the same hipcc/ROCm toolchain the venv
#     loads it against, and this is the only stage that has the full (unpruned) ROCm dev tree. ---
# sly/mxfp4/radiance_mxfp4_fp8.hip is a SINGLE-FILE pybind11 extension (no torch/ATen headers --
# every exported fn takes raw `uintptr_t` addresses, see its own PYBIND11_MODULE block), unlike
# R4D's multi-translation-unit build.sh, so it compiles with one hipcc invocation rather than a
# library build script. It has to be importable as the bare module name `radiance_mxfp4_fp8`
# (`import radiance_mxfp4_fp8 as _ext` in sly/mxfp4/radiance_mxfp4.py, copied to ${SP}/ above) --
# same trick as r4d.so above: CPython's default EXTENSION_SUFFIXES includes plain ".so", so an
# unadorned `<modname>.so` dropped straight into site-packages is importable with no build-tag
# renaming needed. Placed in ${SP} rather than /opt/patches so a stale copy in the patch tree can
# never shadow it (see radiance_mxfp4.py's own comment on the exact same risk for this file).
# torch is imported before the extension for the same reason as R4D's check above and the
# release-stage JIT probe below: a bare HIP extension has no ROCm entry in ld.so.conf and cannot
# resolve libamdhip64 on its own, so `import radiance_mxfp4_fp8` alone fails with
# "ImportError: libamdhip64.so.7: cannot open shared object file" even though the .so just linked
# fine -- torch's import is what actually pulls the runtime into the process.
RUN hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=${GFX_ARCH} \
      $(python -m pybind11 --includes) \
      /opt/patches/sly/mxfp4/radiance_mxfp4_fp8.hip -o ${SP}/radiance_mxfp4_fp8.so \
 && python -c "import torch, radiance_mxfp4_fp8 as m; print('radiance_mxfp4_fp8 built:', m.__file__)"

# sly/gdn/radiance_gdn_decode.hip: HIP port of vLLM's fused GDN MTP decode kernel (0.3.3, see the file
# header). Same single-file pybind11 build as above; registered as torch.ops._C.fused_gdn_decode_post_conv_mtp
# by radiance_gdn_decode.py only under RADIANCE_GDN_FUSED_DECODE=1.
RUN hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=${GFX_ARCH} \
      $(python -m pybind11 --includes) \
      /opt/patches/sly/gdn/radiance_gdn_decode.hip -o ${SP}/radiance_gdn_decode_ext.so \
 && python -c "import torch, radiance_gdn_decode_ext as m; print('radiance_gdn_decode_ext built:', m.__file__)"

# ggz14's three extra weight formats, each a single-file pybind11 extension built like the two
# above (their own flags: C++17, warnings off, as in run_autoround.sh / run_escha.sh /
# paroquant/run_paroquant.sh). The Python sides register a quantization config and only load when
# asked: RADIANCE_AUTOROUND=1 (patch_autoround), RADIANCE_ESCHA=1 (patch_escha),
# RADIANCE_PAROQUANT=1 (radiance_quant_plugins.pth). Dispatch is by the checkpoint's quant_method.
COPY radiance_autoround.hip radiance_autoround_kernels.h radiance_escha.hip /opt/quant/
COPY escha/escha_kernels.h escha/escha_act.h /opt/quant/escha/
COPY paroquant/radiance_paroquant.hip paroquant/par_kernels.h /opt/quant/
RUN cd /opt/quant \
 && for k in autoround escha paroquant; do \
      hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=${GFX_ARCH} $(python -m pybind11 --includes) \
        radiance_$k.hip -o ${SP}/radiance_${k}_kernel.so || exit 1; \
    done \
 && python -c "import torch, radiance_autoround_kernel, radiance_escha_kernel, radiance_paroquant_kernel; print('quant plugin kernels built')" \
 && cd / && rm -rf /opt/quant

# The installed extensions were stripped before the snapshot above; only bytecode is left to drop.
RUN find /opt/vllm -name '__pycache__' -type d -prune -exec rm -rf {} + || true

# =====================================================================================
# STAGE 3b venvsplit: the venv as two layers, so a release update pulls only its own part
# =====================================================================================
# /split/cold is the stack as installed above, minus whatever the patch chain rewrote: ~2.2 GB that
# is byte-identical from release to release until the stack is bumped. /split/hot is the rest:
# the patched vLLM and aiter trees, the few files patched elsewhere (e.g. torch/_dynamo/utils.py),
# the radiance modules and the HIP kernels. split_venv.py resets every mtime, so the cold layer's
# digest depends on file contents only: it repeats whenever the installed stack is the same, also
# when the steps after the wheel install rerun. A registry push then reports the layer as existing,
# and a `docker pull` of the next release skips it.
# The venv is only read here (copied, not moved: see split_venv.py's docstring).
FROM assemble AS venvsplit
RUN /usr/bin/python3 /opt/split_venv.py split /opt/vllm /opt/vllm.stable.json /split

# =====================================================================================
# STAGE 4 final: the release image -- a clean Ubuntu with only what is needed to serve
# =====================================================================================
# Built by COPYing an allowlist rather than by inheriting the ROCm base, which is what makes the
# prune above pay: the release image never contains the 19 GB tree, the build toolchain, the
# wheels, or the patch sources. Only the pruned ROCm, the venv, and the entrypoint come across.
FROM ${RELEASE_BASE} AS final
ARG GFX_ARCH
ARG AITER_VERSION
ARG VLLM_VERSION
ARG TRANSFORMERS_VERSION
ENV DEBIAN_FRONTEND=noninteractive
# ROCm 7.14 vendors its own libdrm / numa / elf / sqlite / zlib / zstd (the librocm_sysdeps_* set),
# so the release image needs very little from the distro:
#   python3.12     the interpreter the /opt/vllm venv was built against (Ubuntu 24.04 ships 3.12.3)
#   libnuma-dev    rocSHMEM dlopen()s the UNVERSIONED libnuma.so, which only the -dev package ships
#   numactl        optional --numa-bind;  curl  the compose healthcheck runs it inside the container
#   g++            NOT optional: AITER JIT-compiles its kernels on FIRST USE, inside this image, and
#                  hipcc needs the C++ standard headers (and the same g++ major torch was built
#                  with). Without it every JIT build dies with "Could not find standard C++ header
#                  'cmath'", aiter's flag probes all fail (including --offload-arch), and the engine
#                  crashes. The build-time probe below is what keeps this honest.
#   python3.12-dev Python.h, for the pybind11 modules aiter JIT-builds
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3.12 python3.12-dev g++ libnuma-dev numactl curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# /opt/rocm is a symlink farm pointing through /etc/alternatives into core-<ver>, so both have to
# come across or nothing resolves. The pruned tree is stable across radiance releases, which keeps
# it a cached layer users do not re-download for every version bump.
COPY --from=rocmprune /opt/rocm /opt/rocm
COPY --from=rocmprune /etc/alternatives /etc/alternatives
# Cold before hot: the hot layer fills in the paths the cold one leaves out (see venvsplit).
COPY --from=venvsplit /split/cold/ /opt/vllm/
COPY --from=venvsplit /split/hot/ /opt/vllm/
COPY --from=builder /artifacts/rocm-bandwidth-test /usr/local/bin/rocm-bandwidth-test

# RADIANCE_GFX_ARCH is what the gfx1201 patch and the banner read for the target arch (amdsmi's
# asic_info reports it empty on this card). It used to be called VLLM_ROCM_GCN_ARCH, which vLLM
# 0.26 flags as an unknown VLLM_* variable at startup; the old name is still honored.
ENV VIRTUAL_ENV=/opt/vllm \
    PATH=/opt/vllm/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd \
    VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=${GFX_ARCH} RADIANCE_GFX_ARCH=${GFX_ARCH} \
    HIP_ARCHITECTURES=${GFX_ARCH} AMDGPU_TARGETS=${GFX_ARCH} GPU_ARCHS=${GFX_ARCH} \
    SAFETENSORS_FAST_GPU=1 TOKENIZERS_PARALLELISM=false TRITON_CACHE_AUTOTUNING=1 \
    PYTHONDONTWRITEBYTECODE=1

# --- radiance feature flags (set any to 0 to fall back to stock). RADIANCE_USE_R4D is the master
#     switch for the hand-written gfx1201 kernel library: 0 takes it out of the picture entirely
#     (attention, the gated delta net, vision attention, the all-reduce and the skinny GEMM all
#     revert to the stock path) without a rebuild. RADIANCE_USE_R4D_AR and its _QUANT variant are
#     the two all-reduce behaviours worth switching independently, since one is bit-identical to
#     RCCL and the other is not. RADIANCE_RUN_BWTEST runs the bandwidth sweep at startup;
#     it is backgrounded and takes about a second, so it never delays the serve. ---
ENV RADIANCE_USE_R4D=1 \
    RADIANCE_USE_R4D_AR=1 RADIANCE_USE_R4D_AR_QUANT=1 \
    RADIANCE_PRESHUFFLE=1 RADIANCE_FUSE_RMS_QUANT=1 \
    RADIANCE_DYNAMIC_DRAFT=1 RADIANCE_DRAFT_SCHEDULE=1:8,2:7,4:6,8:5,16:4 RADIANCE_DRAFT_TAU=0.35 \
    RADIANCE_RUN_BWTEST=1

# Fail the build if the native stack does not import, or if a wheel reports a version that does not
# match the source it was built from (a silently mis-stamped wheel is how "aiter 0.0.0" shipped).
# Running this in the RELEASE stage also proves the allowlist above is complete: a library left
# behind by the prune or by the slim base shows up here as an ImportError, not in production.
# Kept GPU-free: no `import aiter` (it runs rocminfo) and no full `import vllm`; versions come from
# package metadata. R4D is imported after torch, which is what loads libamdhip64.
RUN WANT_VLLM=${VLLM_VERSION} WANT_AITER=${AITER_VERSION} WANT_TF=${TRANSFORMERS_VERSION} \
    python -c 'import os, torch, vllm._C, amdsmi, importlib.metadata as m; \
import r4d; \
v, a, t = m.version("vllm"), m.version("amd-aiter"), m.version("transformers"); \
assert v.startswith(os.environ["WANT_VLLM"]), "vllm wheel reports " + v + ", built tag is " + os.environ["WANT_VLLM"]; \
assert a.startswith(os.environ["WANT_AITER"]), "aiter wheel reports " + a + ", built tag is " + os.environ["WANT_AITER"]; \
assert t == os.environ["WANT_TF"], "transformers is " + t + ", pinned is " + os.environ["WANT_TF"]; \
print("stack OK | vllm", v, "| torch", torch.__version__, "| aiter", a, \
      "| torchvision", m.version("torchvision"), "| triton", m.version("triton"), \
      "| transformers", t, "| r4d", r4d.__version__)'

# The release image must still be able to COMPILE. AITER JIT-builds its kernels on first use, as a
# pybind11 HIP extension, so the shipped image needs hipcc AND the C++ standard headers AND Python.h.
# `hipcc --version` does not prove any of that -- it passes on an image whose JIT is broken, which is
# exactly how a slim release stage shipped with no libstdc++ headers. This mirrors aiter's real
# compile: build a pybind11 HIP module that includes <cmath>, then import it and call into it.
# torch is imported first because that is what pulls libamdhip64 into the process -- a bare HIP
# extension cannot resolve it on its own (no ROCm entry in ld.so.conf), here or in any prior release.
RUN printf '%s\n' \
      '#include <hip/hip_runtime.h>' \
      '#include <cmath>' \
      '#include <pybind11/pybind11.h>' \
      '__global__ void k(float* o) { o[threadIdx.x] = 1.0f; }' \
      'PYBIND11_MODULE(_jit_probe, m) { m.def("f", [](double x) { return std::sqrt(x); }); }' \
      > /tmp/_jit_probe.hip \
 && hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=${GFX_ARCH} \
      $(python -m pybind11 --includes) /tmp/_jit_probe.hip -o /tmp/_jit_probe.so \
 && python -c "import torch, sys; sys.path.insert(0, '/tmp'); import _jit_probe; assert _jit_probe.f(4.0) == 2.0" \
 && rm -f /tmp/_jit_probe.hip /tmp/_jit_probe.so \
 && echo "runtime JIT toolchain OK (hipcc + libstdc++ headers + Python.h + pybind11)"

ARG RADIANCE_VERSION=dev
ENV RADIANCE_VERSION=${RADIANCE_VERSION}
# The banner reads this file first: one source of truth for the version, so it reports what was
# built even when the image is built without --build-arg.
COPY VERSION /opt/radiance_version
# The chat template for Qwen 3.5 / 3.6 / 3.8: froggeric's fixed template (huggingface.co/froggeric/
# Qwen-Fixed-Chat-Templates, Apache-2.0, repo revision 855bffc). It fixes the official 3.8 template's
# xhigh-by-default reasoning, the blank <think></think> blocks it injects into chat history (a
# prefix-cache miss on every turn), crashes on JSON-string tool arguments and enable_thinking=false,
# and maps client effort aliases (high/max -> xhigh, none/off -> thinking off). Serve with
# `--chat-template /opt/qwen-fixed.jinja`; the path stays the same across template versions, the
# version is in the label io.slybase.radiance.chat-template and in the file's first line.
COPY qwen-fixed-v22.5.jinja /opt/qwen-fixed.jinja
COPY ci/check_chat_template.py /tmp/check_chat_template.py
RUN python /tmp/check_chat_template.py && rm -f /tmp/check_chat_template.py
COPY radiance_preamble.py /opt/radiance_preamble.py
COPY radiance_entrypoint.sh /opt/radiance_entrypoint.sh
RUN chmod +x /opt/radiance_entrypoint.sh
ENTRYPOINT ["/opt/radiance_entrypoint.sh"]

# --- image metadata: what this image is and where it came from ---
# `docker inspect --format '{{json .Config.Labels}}' <image>` shows them; build.yml sets the
# description and source on the registry manifest too, where ghcr.io reads them. Kept last:
# VCS_REF and BUILD_DATE change with every build, and a changed ARG invalidates every RUN below it.
# build.yml passes RADIANCE_VERSION (= VERSION), VCS_REF and BUILD_DATE; a local build says dev/unknown.
# ref.name only overrides the base image's own value ("ubuntu"), which would otherwise show through.
ARG ROCM_BASE
ARG RELEASE_BASE
ARG TORCH_VERSION
ARG TRITON_VERSION
ARG R4D_VERSION
ARG VCS_REF=unknown
ARG BUILD_DATE=unknown
LABEL org.opencontainers.image.title="vllm-sly-radiance" \
      org.opencontainers.image.description="vLLM ${VLLM_VERSION} for the AMD Radeon AI PRO R9700 (${GFX_ARCH}, RDNA4): MXFP4 (Quark) checkpoints with the W4A8 HIP GEMM, DFlash2 speculative decoding, libr4d kernels" \
      org.opencontainers.image.version="${RADIANCE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.source="https://github.com/SlyBase/vllm-sly-radiance" \
      org.opencontainers.image.url="https://github.com/SlyBase/vllm-sly-radiance" \
      org.opencontainers.image.documentation="https://github.com/SlyBase/vllm-sly-radiance#readme" \
      org.opencontainers.image.vendor="SlyBase" \
      org.opencontainers.image.base.name="${RELEASE_BASE}" \
      org.opencontainers.image.ref.name="${RADIANCE_VERSION}" \
      io.slybase.radiance.gfx-arch="${GFX_ARCH}" \
      io.slybase.radiance.rocm-base="${ROCM_BASE}" \
      io.slybase.radiance.vllm="${VLLM_VERSION}" \
      io.slybase.radiance.torch="${TORCH_VERSION}" \
      io.slybase.radiance.triton="${TRITON_VERSION}" \
      io.slybase.radiance.aiter="${AITER_VERSION}" \
      io.slybase.radiance.transformers="${TRANSFORMERS_VERSION}" \
      io.slybase.radiance.r4d="${R4D_VERSION}" \
      io.slybase.radiance.chat-template="qwen3.8-froggeric-v22.5 (/opt/qwen-fixed.jinja)"
