#!/usr/bin/env bash
# In-process producer A/B (ci/prefill_nq_ab.py) on the production launch line of the GPU host.
# TAG=<out subdir> LENS=1514,3520 ORDER=01100110 CHECK=0|1 SO=<dir with radiance_mxfp4_fp8.so>
# Serialised on /root/gpu.lock with the other GPU users; hard cap TMO seconds (default 700).
set -uo pipefail
W=${W:-/root/nongemm}
TAG=${TAG:-ab}
OUT=$W/out/$TAG
NAME=nongemm-$TAG
SO=${SO:-$W/src}
TMO=${TMO:-700}
UNIT=/etc/systemd/system/docker-vllm7.service
SP=/opt/vllm/lib/python3.12/site-packages
mkdir -p "$OUT"; cd "$OUT" || exit 1
log() { echo "$(date -u +%T) $*" | tee -a run.log; }
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1; touch DONE; }
trap cleanup EXIT
awk '/^ExecStart=/{f=1} f{print} /--enable-prefix-caching/{if(f) exit}' "$UNIT" | sed 's/^ExecStart=//' > launch_prod.txt
ENVX="-e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_CUSTOM_SCOPES_FOR_PROFILING=1 -e OUT=/w/out/$TAG \
-e LENS=${LENS:-1514,3520} -e ORDER=${ORDER:-01100110} -e NPROF=${NPROF:-4} -e CHECK=${CHECK:-0} ${EXTRA_ENV:-}"
MNT="-v $W:/w -v $SO/radiance_mxfp4_fp8.so:$SP/radiance_mxfp4_fp8.so:ro"
sed -e "s/--rm --name vllm7-mxfp4/--rm --name $NAME/" \
    -e "s/-p 8000:8000 //" \
    -e "s|-v /root/swift-heretic-gptq:/models/swift-mxfp4:ro |-v /root/swift-heretic-gptq:/models/swift-mxfp4:ro $MNT $ENVX --entrypoint python3 |" \
    -e "s|^  vllm-sly-radiance:1.5.1-rocm10.1 |  vllm-sly-radiance:1.5.1-rocm10.1 /w/prefill_nq_ab.py |" launch_prod.txt > cmd.txt
grep -q "/w/prefill_nq_ab.py" cmd.txt && grep -q -- "--entrypoint python3" cmd.txt || { log "sed failed"; exit 1; }
[ "${DRY:-0}" = 1 ] && { cat cmd.txt; exit 0; }
log "wait lock"
flock -x /root/gpu.lock bash -c "echo \$(date -u +%T) start $NAME >> run.log; timeout $TMO bash cmd.txt > container.log 2>&1; echo \$(date -u +%T) exit \$? >> run.log; docker rm -f $NAME >/dev/null 2>&1"
grep -E '^(SPLIT|NQCHECK)' container.log | tee -a run.log
