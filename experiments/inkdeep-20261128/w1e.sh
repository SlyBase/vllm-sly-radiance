#!/bin/bash
# Window 1e: M1 micro-sweep + M2 ISA probe (image triton 3.8), then OLD4
# (1.14.2 control, fresh caches, logs captured even on death); if OLD4 dies,
# OLD5 retries with --gpu-memory-utilization 0.85 (documented deviation).
set -u
W=/root/inkdeep-20261128; GW=/usr/local/sbin/gpu-window
OWNER=daidalos-inkdeep; C=vllm-opt
CAP=/sys/class/drm/card0/device/hwmon/hwmon6/power1_cap
VRAM=/sys/class/drm/card0/device/mem_info_vram_used
IMG151=vllm-sly-radiance:1.15.1-rocm10.1
IMG142=vllm-sly-radiance:1.14.2-rocm10.1
DF2='{"method":"dflash","model":"/models/inkling-2-mini-dflash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto"}'
ACQUIRED=0
log(){ echo "$(date -u +%H:%M:%S) $*" >> "$W/progress.log"; }
restore(){
  docker rm -f "$C" >/dev/null 2>&1
  systemctl start docker-vllm7.service >/dev/null 2>&1
  systemctl start gpu-window-guard.timer >/dev/null 2>&1
  for i in $(seq 1 90); do
    h=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health || true)
    [ "$h" = "200" ] && break
    sleep 5
  done
  [ "$ACQUIRED" = "1" ] && "$GW" release >> "$W/progress.log" 2>&1
  log "W1E RESTORE: prod health=$h cap=$(cat $CAP 2>/dev/null)"
}
trap restore EXIT
PF=$W/driver.pid
if [ -f "$PF" ]; then
  old=$(cat "$PF" 2>/dev/null || echo 0)
  if [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null && grep -qa "w1" "/proc/$old/cmdline" 2>/dev/null; then
    log "ABORT: another driver alive (pid $old)"; exit 1
  fi
fi
echo $$ > "$PF"
"$GW" acquire "$OWNER" 5400
st=$("$GW" status)
echo "$st" | grep -q "\"owner\": \"$OWNER\"" || { log "ABORT: not owner: $st"; exit 1; }
ACQUIRED=1
systemctl stop gpu-window-guard.timer
systemctl stop docker-vllm7.service
docker rm -f "$C" >/dev/null 2>&1
for i in $(seq 1 36); do
  v=$(cat "$VRAM" 2>/dev/null || echo 999999999999)
  [ "$v" -le 1073741824 ] && break
  sleep 5
done
# --- M1 + M2 (entrypoint override: image ENTRYPOINT is vllm serve) ---
mkdir -p "$W/out/M1"
log "M1+M2 START cap=$(cat $CAP 2>/dev/null) vram=$(cat $VRAM)"
timeout 900 docker run --rm --name "$C" --device /dev/kfd --device /dev/dri \
  --group-add video --shm-size 8g --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=1 \
  --entrypoint /bin/bash -v "$W":/work "$IMG151" \
  -c "python3 -I /work/m1.py > /work/out/M1/m1.log 2>&1; python3 -I /work/m2.py > /work/out/M1/m2.log 2>&1"
log "M1+M2 DONE rc=$?"
docker rm -f "$C" >/dev/null 2>&1
# --- OLD4 / OLD5: 1.14.2 control, fresh caches, logs kept even on death ---
run_old(){
  local name=$1 util=$2 o=$W/out/$1
  mkdir -p "$o"
  for i in $(seq 1 24); do
    v=$(cat "$VRAM" 2>/dev/null || echo 999999999999)
    [ "$v" -le 1073741824 ] && break
    sleep 5
  done
  log "=== ARM $name START img=$IMG142 util=$util suites=mia,ctx caches=fresh ==="
  docker run -d --name "$C" \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 --security-opt seccomp=unconfined --ipc=host \
    -e HIP_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e GPU_MAX_HW_QUEUES=1 \
    -p 8001:8000 \
    -v /root/hf-cache:/root/.cache/huggingface \
    -v /root/inkling2-mini:/model:ro \
    -v /root/dflash2-ink/model:/models/inkling-2-mini-dflash2:ro \
    -v "$W/oldcache/triton":/root/.triton \
    -v "$W/oldcache/tc":/root/.cache/vllm/torch_compile_cache \
    -v "$W/oldcache/aiter":/root/.aiter \
    "$IMG142" \
    --model /model --served-model-name inkling2-mini \
    --max-model-len 307200 --gpu-memory-utilization "$util" --kv-cache-dtype fp8 --port 8000 \
    --max-num-seqs 8 --no-async-scheduling --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0}' --reasoning-parser inkling \
    --enable-auto-tool-choice --tool-call-parser inkling \
    --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
    --speculative-config "$DF2" >/dev/null
  local ready=0
  for i in $(seq 1 150); do
    if ! docker ps --format '{{.Names}}' | grep -q "^$C$"; then log "$name container died during startup"; break; fi
    h=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/health || true)
    if [ "$h" = "200" ]; then ready=1; break; fi
    sleep 5
  done
  local rc=1
  if [ "$ready" = "1" ]; then
    docker exec "$C" env | grep -E '^(RADIANCE|GPU_)' | sort > "$o/env-proof.txt"
    docker logs "$C" > "$o/boot.log" 2>&1
    grep -a -E "GPU KV cache size|radiance|HIP_CTRL|dflash|adaptive|cudagraph_mode" "$o/boot.log" | head -n 30 > "$o/hooks.txt"
    python3 -I /root/ink-200/ttft/mia_decode.py http://127.0.0.1:8001 inkling2-mini "$o/mia.json" --tokens 400 --reps 3 > "$o/mia.txt" 2>&1
    python3 -I /root/defaults/ctxbench.py http://127.0.0.1:8001 "$o/ctx.json" > "$o/ctx.txt" 2>&1
    [ -s "$o/mia.txt" ] && [ -s "$o/ctx.txt" ] && rc=0
  fi
  docker logs "$C" > "$o/server.log" 2>&1
  docker rm -f "$C" >/dev/null 2>&1
  echo "{\"arm\":\"$name\",\"rc\":$rc,\"ts\":\"$(date -u +%FT%TZ)\"}" >> "$W/ledger.jsonl"
  log "=== ARM $name DONE rc=$rc ==="
  return $rc
}
run_old OLD4 0.92 || run_old OLD5 0.85
exit 0
