#!/usr/bin/env bash
# Resource report (VGPR/SGPR/spills/LDS/occupancy) of one chunk_scan translation unit, CPU only.
# Usage (on the GPU host, no --device): ci/gdn_scan_resources.sh <r4d-src-dir> <unit.hip> [extra hipcc flags...]
# Prints the -Rpass-analysis=kernel-resource-usage remarks of the unit; redirect into
# ci/tests/fixtures/gdn_scan_<variant>.remark.txt to feed ci/tests/test_gdn_w3_budget.py.
set -euo pipefail
src=$1; unit=$2; shift 2
IMG=${IMG:-vllm-sly-radiance:1.5.1-rocm10.1}
docker run --rm --name "gdnscan-res-$$" -v "$src":/w -w /w --entrypoint bash "$IMG" -c \
  "hipcc -O3 -std=c++17 -fPIC --offload-arch=gfx1201 -Wno-unused-result -ffp-contract=off $* \
   -Rpass-analysis=kernel-resource-usage -c $unit -o /tmp/x.o 2>&1 | grep -E 'Function Name|SGPRs|VGPRs|ScratchSize|Occupancy|LDS Size|error' || true"
