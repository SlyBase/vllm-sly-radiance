#!/usr/bin/env bash
# Greedy decode probe (ci/psw_gen.py) on the production launch line, one process per arm.
# PSW=0|2 TAG=<name> PSW_DIR=<dir with radiance_mxfp4_fp8.so + radiance_mxfp4.py> W=<work dir holding psw_gen.py>
# GT knobs (NQ_REG + SKIPPAD + TN4_MIN_M=513) on in both arms. Serialised on /root/gpu.lock, hard cap TMO s.
set -uo pipefail
W=${W:-/root/psw}
TAG=${TAG:-gen}
PSW=${PSW:-0}
PSW_DIR=${PSW_DIR:-$W/srv}
OUT=$W/out/$TAG
NAME=psw-$TAG
TMO=${TMO:-800}
UNIT=/etc/systemd/system/docker-vllm7.service
SP=/opt/vllm/lib/python3.12/site-packages
mkdir -p "$OUT"; cd "$OUT" || exit 1
log() { echo "$(date -u +%T) $*" | tee -a run.log; }
awk '/^ExecStart=/{f=1} f{print} /--enable-prefix-caching/{if(f) exit}' "$UNIT" | sed 's/^ExecStart=//' > launch_prod.txt
ENVX="-e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e REPS=${REPS:-2} -e MAXTOK=${MAXTOK:-128} \
-e RADIANCE_FUSED_NQ_REG=1 -e RADIANCE_MXFP4_AT_SKIP_PAD=1 -e RADIANCE_MXFP4_TN4_MIN_M=513 -e RADIANCE_MXFP4_AT_PSW=$PSW"
MNT="-v $W/gen:/g:ro -v $PSW_DIR/radiance_mxfp4_fp8.so:$SP/radiance_mxfp4_fp8.so:ro -v $PSW_DIR/radiance_mxfp4.py:$SP/radiance_mxfp4.py:ro"
sed -e "s/--rm --name vllm7-mxfp4/--rm --name $NAME/" \
    -e "s/-p 8000:8000 //" \
    -e "s|-v /root/swift-heretic-gptq:/models/swift-mxfp4:ro |-v /root/swift-heretic-gptq:/models/swift-mxfp4:ro $MNT $ENVX --entrypoint python3 |" \
    -e "s|^  vllm-sly-radiance:1.5.1-rocm10.1 |  vllm-sly-radiance:1.5.1-rocm10.1 /g/psw_gen.py |" launch_prod.txt > cmd.txt
grep -q "/g/psw_gen.py" cmd.txt && grep -q -- "--entrypoint python3" cmd.txt || { log "sed failed"; exit 1; }
[ "${DRY:-0}" = 1 ] && { cat cmd.txt; exit 0; }
log "wait lock (PSW=$PSW)"
flock -x /root/gpu.lock bash -c "echo \$(date -u +%T) start $NAME >> run.log; timeout $TMO bash cmd.txt > container.log 2>&1; echo \$(date -u +%T) exit \$? >> run.log; docker rm -f $NAME >/dev/null 2>&1"
grep -aE '^(GEN|GENSUM)|PSW mode 2|refused' container.log | tee -a run.log
