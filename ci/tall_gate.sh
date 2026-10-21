#!/usr/bin/env bash
# ci/tall_gate.sh - cell gate for the opt-in tall GEMM tile (RADIANCE_MXFP4_TALL), GPU host, inside the
# release image with the freshly built radiance_mxfp4_fp8.so bind-mounted over the image's. Does NOT
# acquire the GPU window: run it inside one (or on an idle GPU); it only takes /root/gpu.lock.
#
#   SO=/root/tall/out/radiance_mxfp4_fp8.so BENCH_DIR=/root/tall/src ci/tall_gate.sh
#
# Build the .so first (CPU only):  OUT=/root/tall/out IMG=ghcr.io/slybase/vllm-sly-radiance:1.6.0-rocm10.1 ci/build_so_nq.sh
# BENCH_DIR must hold bench_prefill_cells.py and tile_layout.py (sly/mxfp4/).
# Arms (one container each, same weights/seed):  tn2 = TN4_MIN_M huge; tn4 = TN4_MIN_M 513 (production);
#   tall = tn4 + RADIANCE_MXFP4_TALL=1 + TALL_MIN_M=0 (so the 1514 cell is exercised too).
# Phase 1 (DIGEST): --check at M=1514,2048,4089,4096, gate_up + in_proj; the DIGEST lines (sha256 of the bf16
#   output) of tall must equal tn2 (and tn4: same K order and WMMA chain per output, so expected equal).
# Phase 2 (ABBA): tall tn4 tn2 tn2 tn4 tall at M=1514,2048,4089; median us per cell over the 2 runs per arm.
set -uo pipefail
SO=${SO:-/root/tall/out/radiance_mxfp4_fp8.so}
BENCH_DIR=${BENCH_DIR:-/root/tall/src}
OUT=${OUT:-/root/tall/gate-$(date -u +%m%d-%H%M)}
IMG=${IMG:-ghcr.io/slybase/vllm-sly-radiance:1.6.0-rocm10.1}
ITERS=${ITERS:-100}; REPS=${REPS:-3}; TMO=${TMO:-900}
SP=/opt/vllm/lib/python3.12/site-packages
mkdir -p "$OUT"; cd "$OUT" || exit 1
COMMON="-e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 -e RADIANCE_MXFP4_DECODE_MAX_M=128 \
 -e RADIANCE_MXFP4_A_TILED_MIN_M=0 -e RADIANCE_FUSED_NORM_QUANT=1 -e RADIANCE_MXFP4_TILED_OPROJ=1 -e RADIANCE_MXFP4_WPERM=1 -e HIP_VISIBLE_DEVICES=0"
armenv() { case $1 in
  tn2)  echo "-e RADIANCE_MXFP4_TN4_MIN_M=99999999 -e RADIANCE_MXFP4_TALL=0";;
  tn4)  echo "-e RADIANCE_MXFP4_TN4_MIN_M=513 -e RADIANCE_MXFP4_TALL=0";;
  tall) echo "-e RADIANCE_MXFP4_TN4_MIN_M=513 -e RADIANCE_MXFP4_TALL=1 -e RADIANCE_MXFP4_TALL_MIN_M=0";;
esac; }
run() {   # run <arm> <tag> <bench args...>
  local arm=$1 tag=$2; shift 2
  flock -x /root/gpu.lock timeout "$TMO" docker run --rm --name "tallgate-$tag" \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 --security-opt seccomp=unconfined --ipc=host \
    $COMMON $(armenv "$arm") -v "$SO":$SP/radiance_mxfp4_fp8.so:ro \
    -v "$BENCH_DIR/bench_prefill_cells.py":/w/bench_prefill_cells.py:ro -v "$BENCH_DIR/tile_layout.py":/w/tile_layout.py:ro \
    -v "$OUT":/out --entrypoint python3 "$IMG" /w/bench_prefill_cells.py --shapes gate_up,in_proj "$@" > "$tag.log" 2>&1
  local rc=$?; echo "$tag rc=$rc"; return $rc
}
FAIL=0
grep -aq RADIANCE_MXFP4_TALL "$SO" || { echo "FAIL: $SO has no RADIANCE_MXFP4_TALL (wrong .so?)"; exit 1; }
echo "== phase 1: DIGEST bit identity (M 1514,2048,4089,4096)"
for arm in tn2 tn4 tall; do
  run $arm dig-$arm --check --ms 1514,2048,4089,4096 --iters 5 --reps 1 || FAIL=1
  grep -E '^DIGEST' dig-$arm.log | sort > dig-$arm.txt
done
NW=$(cat dig-*.log | grep -c WRONG); echo "WRONG lines: $NW"
echo "DIGEST lines per arm: $(wc -l < dig-tn2.txt) $(wc -l < dig-tn4.txt) $(wc -l < dig-tall.txt) (expect 8 each)"
# Never pass vacuously: two empty digest files diff clean, so demand the full line count per arm.
[ "$NW" = 0 ] || FAIL=1
for arm in tn2 tn4 tall; do [ "$(wc -l < dig-$arm.txt)" = 8 ] || { echo "FAIL: dig-$arm.txt not 8 DIGEST lines"; FAIL=1; }; done
if [ $FAIL = 0 ] && diff dig-tn2.txt dig-tall.txt; then echo "tall == tn2: IDENTICAL"; else echo "tall != tn2: DIFFERENT or INVALID"; FAIL=1; fi
[ $FAIL = 0 ] || { echo "GATE FAIL (phase 1); skipping timing"; exit 1; }
diff dig-tn4.txt dig-tall.txt > /dev/null && echo "tall == tn4: identical" || echo "tall != tn4: different (informational)"
echo "== phase 2: ABBA timing (M 1514,2048,4089)"
i=0; for arm in tall tn4 tn2 tn2 tn4 tall; do
  i=$((i+1)); run $arm t$i-$arm --ms 1514,2048,4089 --iters "$ITERS" --reps "$REPS" --csv /out/t$i-$arm.csv || { echo "GATE FAIL: t$i-$arm"; exit 1; }
done
python3 - <<'PY'
import csv, glob, re, statistics as st
d = {}
for f in sorted(glob.glob("t*-*.csv")):
    arm = re.match(r"t\d+-(\w+)\.csv", f).group(1)
    for r in csv.DictReader(open(f)):
        if r["arm"] == "atiled":
            d.setdefault((r["shape"], int(r["M"])), {}).setdefault(arm, []).append(float(r["us"]))
print(f"{'shape':8s} {'M':>5s} {'tall us':>9s} {'tn4 us':>9s} {'tn2 us':>9s} {'tall/tn4':>9s} {'tall/tn2':>9s}")
for (s, M), a in sorted(d.items()):
    if any(len(a.get(k, [])) != 2 for k in ("tall", "tn4", "tn2")):
        raise SystemExit(f"GATE FAIL: {s} M={M}: need 2 runs per arm, got { {k: len(v) for k, v in a.items()} }")
    t, f4, f2 = (st.mean(a[k]) for k in ("tall", "tn4", "tn2"))
    print(f"{s:8s} {M:5d} {t:9.1f} {f4:9.1f} {f2:9.1f} {t / f4:9.3f} {t / f2:9.3f}")
PY
