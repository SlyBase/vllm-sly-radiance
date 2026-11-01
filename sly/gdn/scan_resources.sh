#!/usr/bin/env bash
# Regenerate the compiler resource reports the CPU test pins (sly/tests/fixtures/gdn_scan/*.remark.txt).
#
# Run on the build host, next to a checkout of libr4d with sly/r4d/r4d_gdn_scan_variants.patch applied:
#
#     ./sly/gdn/scan_resources.sh /path/to/libr4d
#
# No GPU is needed -- hipcc compiles for gfx1201 without one. Needs the ROCm image that builds r4d.so.
set -euo pipefail
SRC=${1:?usage: scan_resources.sh <libr4d checkout>}
IMG=${IMG:-ghcr.io/slybase/vllm-sly-radiance:1.6.0-rocm10.1}
# DEST=sly/tests/fixtures/gdn_scan_exact with r4d_gdn_exact_decay.patch applied on top makes that patch's fixtures.
DEST=$(cd "${DEST:-$(dirname "$0")/../tests/fixtures/gdn_scan}" && pwd)

cat > "$SRC/_res_probe.sh" <<'SH'
set -u
F=$1; EXTRA=${2:-}
hipcc -O3 -std=c++17 -fPIC --offload-arch=gfx1201 -Wno-unused-result -ffp-contract=off $EXTRA \
      -Rpass-analysis=kernel-resource-usage -c "$F" -o /tmp/res_probe.o 2>&1 \
 | grep -E "Function Name:|TotalSGPRs|VGPRs:|ScratchSize|Occupancy|LDS Size|Spill" \
 | sed 's/^[^:]*:314:[0-9]*: //'
SH

probe() {  # probe <out-name> <unit.hip> <extra flag>
  docker run --rm -v "$SRC":/w -w /w --entrypoint bash "$IMG" \
         -c "bash _res_probe.sh $2 '$3'" > "$DEST/$1.remark.txt"
}
probe stock r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip    -mcumode
probe s1    r4d_gdn_chunk_scan_k128_v128_c64_bf16_s1.hip ""
probe s2    r4d_gdn_chunk_scan_k128_v128_c64_bf16_s2.hip ""
probe s3    r4d_gdn_chunk_scan_k128_v128_c64_bf16_s3.hip ""
rm -f "$SRC/_res_probe.sh"
cat "$DEST"/stock.remark.txt "$DEST"/s3.remark.txt
