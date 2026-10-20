#!/usr/bin/env bash
# ci/build_so_nq.sh - CPU-only build of sly/mxfp4/radiance_mxfp4_fp8.hip from this branch (no GPU device).
#
# The .so carries RADIANCE_FUSED_NQ_REG (register-resident producer row slice), the set_nq_reg /
# set_nq_art A/B hooks and the NQ_ART workgroup-size variants; it is bind-mounted over the image's
# radiance_mxfp4_fp8.so by ci/prefill_ab.sh arms G/G0 and by ci/run_nq_ab.sh.
#
# Runs on the GPU host, inside the release image (~3 min, no --device):
#   OUT=/root/nongemm/so-night ci/build_so_nq.sh
# then mount $OUT/radiance_mxfp4_fp8.so (G_DIR=$OUT for ci/prefill_ab.sh).
set -euo pipefail
OUT=${OUT:-/root/nongemm/so-night}
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
IMG=${IMG:-vllm-sly-radiance:1.5.1-rocm10.1}
ARCH=${ARCH:-gfx1201}
mkdir -p "$OUT"
docker run --rm -v "$OUT":/b -v "$SRC":/s:ro --entrypoint bash "$IMG" -c "
set -e
hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=$ARCH \$(python -m pybind11 --includes) \
  /s/sly/mxfp4/radiance_mxfp4_fp8.hip -o /b/radiance_mxfp4_fp8.so
"
ls -l "$OUT/radiance_mxfp4_fp8.so"
md5sum "$OUT/radiance_mxfp4_fp8.so"
