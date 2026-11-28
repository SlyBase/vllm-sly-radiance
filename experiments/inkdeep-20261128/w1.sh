#!/bin/bash
# Window 1: attribution arms for the 1.14.2 -> 1.15.1 deep-context decode drop.
# Detached on LXC 2408: nohup bash /root/inkdeep-20261128/w1.sh > /root/inkdeep-20261128/run.log 2>&1 &
# EXIT trap restores prod + releases the window on EVERY path. One acquire for the whole window.
set -u
W=/root/inkdeep-20261128; O=$W/out; GW=/usr/local/sbin/gpu-window
OWNER=daidalos-inkdeep; C=vllm-opt
CAP=/sys/class/drm/card0/device/hwmon/hwmon6/power1_cap
IMG151=vllm-sly-radiance:1.15.1-rocm10.1
IMG142=vllm-sly-radiance:1.14.2-rocm10.1
TC151=/root/vllm7-cache/tc-1.15.1
TC142=/root/vllm7-cache/torch_compile
DF2='{"method":"dflash","model":"/models/inkling-2-mini-dflash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto"}'
ACQUIRED=0
mkdir -p "$O"
log(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$W/progress.log"; }
ledger(){ echo "$*" >> "$W/ledger.jsonl"; }

restore(){
  log "RESTORE: prod + window release"
  docker rm -f "$C" >/dev/null 2>&1 || true
  rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/sem.mp-* 2>/dev/null || true
  systemctl start docker-vllm7.service
  local i
  for i in $(seq 90); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)" = 200 ] && break
    sleep 2
  done
  log "prod health: $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/health)"
  systemctl start gpu-window-guard.timer 2>/dev/null || true
  [ "$ACQUIRED" = 1 ] && "$GW" release >/dev/null 2>&1
  log "RESTORE done"
}
trap restore EXIT

ram_wait(){
  local a i
  for i in $(seq 40); do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    [ "$a" -ge 14000 ] && return 0
    sleep 3
  done
  log "RAM gate failed (MemAvailable ${a} MB < 14000)"
  return 1
}

wait_ready(){
  local i
  for i in $(seq 450); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/health)" = 200 ] && return 0
    [ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" = true ] || {
      log "container died during startup"; docker logs --tail 40 "$C" 2>&1 | tail -20; return 1; }
    sleep 2
  done
  return 1
}

# arm <name> <image> <compile-cache> <extra_env space-separated>
arm(){
  local name=$1 img=$2 tc=$3 extra=$4 o="$O/$1" env_args=() kv rc=0 t0
  mkdir -p "$o"; t0=$(date -u +%s)
  ram_wait || { ledger "{\"arm\":\"$name\",\"rc\":\"ram_gate\"}"; return 1; }
  docker rm -f "$C" >/dev/null 2>&1 || true
  rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/sem.mp-* 2>/dev/null || true
  for kv in $extra; do env_args+=(-e "$kv"); done
  log "=== ARM $name START img=$img env=[$extra] ==="
  docker run -d --name "$C" \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 --security-opt seccomp=unconfined --ipc=host \
    -e HIP_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e GPU_MAX_HW_QUEUES=1 "${env_args[@]}" \
    -p 8001:8000 \
    -v /root/hf-cache:/root/.cache/huggingface \
    -v /root/inkling2-mini:/model:ro \
    -v /root/dflash2-ink/model:/models/inkling-2-mini-dflash2:ro \
    -v /root/vllm7-cache/triton:/root/.triton \
    -v "$tc":/root/.cache/vllm/torch_compile_cache \
    -v /root/vllm7-cache/aiter:/root/.aiter \
    "$img" \
    --model /model --served-model-name inkling2-mini \
    --max-model-len 307200 --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 --port 8000 \
    --max-num-seqs 8 --no-async-scheduling --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0}' --reasoning-parser inkling \
    --enable-auto-tool-choice --tool-call-parser inkling \
    --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
    --speculative-config "$DF2" >/dev/null
  if wait_ready; then
    docker exec "$C" env | grep -E '^(RADIANCE|GPU_)' | sort > "$o/env-proof.txt"
    docker logs "$C" > "$o/boot.log" 2>&1
    grep -a -E "GPU KV cache size|radiance|HIP_CTRL|dflash|adaptive" "$o/boot.log" | head -n 30 > "$o/hooks.txt"
    python3 -I /root/ink-200/ttft/mia_decode.py http://127.0.0.1:8001 inkling2-mini "$o/mia.json" --tokens 400 --reps 3 > "$o/mia.txt" 2>&1 || rc=1
    python3 -I /root/defaults/ctxbench.py http://127.0.0.1:8001 "$o/ctx.json" > "$o/ctx.txt" 2>&1 || rc=1
    # validity: both result files must exist and parse
    python3 - "$o" <<'EOF' || rc=1
import json, sys, os
o = sys.argv[1]
for f in ("mia.json", "ctx.json"):
    p = os.path.join(o, f)
    assert os.path.getsize(p) > 50, f"{f} missing/empty"
    json.load(open(p))
print("results valid")
EOF
    cat "$o/mia.txt" 2>/dev/null | grep -a MIA
    cat "$o/ctx.txt" 2>/dev/null
  else
    rc=1
  fi
  docker logs "$C" > "$o/server.log" 2>&1 || true
  grep -a -E "GPU KV cache size|radiance|HIP_CTRL|dflash|adaptive" "$o/server.log" | head -n 40 > "$o/hooks.txt"
  docker rm -f "$C" >/dev/null 2>&1 || true
  ledger "{\"arm\":\"$name\",\"img\":\"$img\",\"env\":\"$extra\",\"rc\":$rc,\"secs\":$(( $(date -u +%s) - t0 ))}"
  log "=== ARM $name DONE rc=$rc ==="
  return $rc
}

# ---- preflight (no GPU) ----
[ "$(cat "$CAP")" = 210000000 ] || { log "ABORT: power cap not 210 W"; exit 1; }
for f in /root/defaults/ctxbench.py /root/ink-200/ttft/mia_decode.py "$W/bench_decode_attn.py"; do
  [ -f "$f" ] || { log "ABORT: missing $f"; exit 1; }
done
docker image inspect "$IMG151" >/dev/null 2>&1 || { log "ABORT: missing $IMG151"; exit 1; }
docker image inspect "$IMG142" >/dev/null 2>&1 || { log "ABORT: missing $IMG142"; exit 1; }
if [ "${PREFLIGHT:-0}" = 1 ]; then log "PREFLIGHT OK"; trap - EXIT; exit 0; fi

# ---- acquire once ----
"$GW" acquire "$OWNER" 7200
"$GW" status | grep -q "\"owner\": *\"$OWNER\"" || { log "ABORT: window not owned by us"; exit 1; }
ACQUIRED=1
systemctl stop gpu-window-guard.timer docker-vllm7.service
docker rm -f "$C" >/dev/null 2>&1 || true
for i in $(seq 30); do [ -z "$(docker ps -q)" ] && break; sleep 2; done
log "window acquired, prod stopped"

# ---- M0: tile microbench (no server) ----
mkdir -p "$O/M0_micro"
log "=== ARM M0_micro START (decode attn sweep) ==="
timeout 900 docker run --rm --name "$C" \
  --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 --security-opt seccomp=unconfined --ipc=host \
  -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=1 \
  -v /root/vllm7-cache/triton:/root/.triton \
  -v "$W/bench_decode_attn.py":/work/bench_decode_attn.py:ro \
  --entrypoint python3 "$IMG151" /work/bench_decode_attn.py --sweep > "$O/M0_micro/decode_attn.log" 2>&1
m0rc=$?; docker rm -f "$C" >/dev/null 2>&1 || true
ledger "{\"arm\":\"M0_micro\",\"rc\":$m0rc}"
log "=== ARM M0_micro DONE ==="

# ---- attribution arms: control first and last ----
arm C1  "$IMG151" "$TC151" ""
arm OLD "$IMG142" "$TC142" ""
arm T   "$IMG151" "$TC151" "RADIANCE_INKLING2_DECODE_BLOCK_N=64 RADIANCE_INKLING2_DECODE_WARPS=4 RADIANCE_INKLING2_LOCAL_BLOCK_N=128 RADIANCE_INKLING2_LOCAL_WARPS=4"
arm D   "$IMG151" "$TC151" "RADIANCE_DFLASH_MXFP4=0"
arm A   "$IMG151" "$TC151" "RADIANCE_ADAPTIVE_K= RADIANCE_INKLING2_ADAPTIVE_K="
arm C2  "$IMG151" "$TC151" ""
log "ALL DONE"
