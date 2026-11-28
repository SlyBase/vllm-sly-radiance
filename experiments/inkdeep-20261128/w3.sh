#!/bin/bash
# Window 3: sampled GSM8K-200 gate ONLY (temp 0.7, n=200, PAR=8) for PR #279.
# Fixes the window-2 harness bug: gsmq.py needs pandas -> run it with
# /root/lmeval-venv/bin/python (pandas 3.0.5 + pyarrow 25.0.1), not host python3.
# gsmq_ink.py = /root/pr-ab/gsmq.py with model field set to inkling2-mini.
# Arms: S0a = 1.15.1 control FIRST, S1 = candidate 1.15.2 (published bits,
# warm g1cache from window 2), S0b = 1.15.1 control LAST (noise floor).
# Same serve flags as window 2 (0.92 / 307200 / fp8 / k7 probabilistic /
# medium). 210 W untouched. Quality sampling: one warm boot per arm is enough
# (the second-start rule guards throughput numbers, not sampled accuracy).
set -u
W=/root/inkdeep-20261128; GW=/usr/local/sbin/gpu-window
OWNER=daidalos-inkdeep; C=vllm-opt
CAP=/sys/class/drm/card0/device/hwmon/hwmon6/power1_cap
VRAM=/sys/class/drm/card0/device/mem_info_vram_used
IMG151=vllm-sly-radiance:1.15.1-rocm10.1
IMGC=vllm-sly-radiance:1.15.2-rocm10.1
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
  log "W3 RESTORE: prod health=$h cap=$(cat $CAP 2>/dev/null)"
}
trap restore EXIT
PF=$W/driver.pid
if [ -f "$PF" ]; then
  old=$(cat "$PF" 2>/dev/null || echo 0)
  if [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null && grep -qa "w3" "/proc/$old/cmdline" 2>/dev/null; then
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
log "W3 START cap=$(cat $CAP 2>/dev/null) vram=$(cat $VRAM)"

wait_vram(){ for i in $(seq 1 24); do v=$(cat "$VRAM" 2>/dev/null || echo 999999999999); [ "$v" -le 1073741824 ] && break; sleep 5; done; }

boot(){
  local name=$1 img=$2 cd=$3 bw=${4:-150}
  mkdir -p "$cd/triton" "$cd/tc" "$cd/aiter"
  docker run -d --name "$C" \
    --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 --security-opt seccomp=unconfined --ipc=host \
    -e HIP_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e GPU_MAX_HW_QUEUES=1 \
    -p 8001:8000 \
    -v /root/hf-cache:/root/.cache/huggingface \
    -v /root/inkling2-mini:/model:ro \
    -v /root/dflash2-ink/model:/models/inkling-2-mini-dflash2:ro \
    -v "$cd/triton":/root/.triton \
    -v "$cd/tc":/root/.cache/vllm/torch_compile_cache \
    -v "$cd/aiter":/root/.aiter \
    "$img" \
    --model /model --served-model-name inkling2-mini \
    --max-model-len 307200 --gpu-memory-utilization 0.92 --kv-cache-dtype fp8 --port 8000 \
    --max-num-seqs 8 --no-async-scheduling --enable-prefix-caching \
    --limit-mm-per-prompt '{"image":0}' --reasoning-parser inkling \
    --enable-auto-tool-choice --tool-call-parser inkling \
    --default-chat-template-kwargs '{"reasoning_effort":"medium"}' \
    --speculative-config "$DF2" >/dev/null
  for i in $(seq 1 "$bw"); do
    if ! docker ps --format '{{.Names}}' | grep -q "^$C$"; then log "$name container died during startup"; return 1; fi
    h=$(curl -s -m 3 -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/health || true)
    [ "$h" = "200" ] && return 0
    sleep 5
  done
  log "$name startup timeout"; return 1
}

suites(){
  local name=$1 o=$W/out/$1
  mkdir -p "$o"
  docker exec "$C" env | grep -E '^(RADIANCE|GPU_)' | sort > "$o/env-proof.txt"
  docker logs "$C" > "$o/boot.log" 2>&1
  grep -a -E "GPU KV cache size|radiance|dflash|adaptive|cudagraph_mode|decode" "$o/boot.log" | head -n 30 > "$o/hooks.txt"
  log "$name gsmq sampled (venv python)"
  ( cd "$o" && URL=http://127.0.0.1:8001 PAR=8 TEMP=0.7 timeout 900 /root/lmeval-venv/bin/python "$W/gsmq_ink.py" "$name" 200 > "$o/gsmq.txt" 2>&1 )
  cp "$W/outq/$name.json" "$o/gsmq.json" 2>/dev/null
  tail -2 "$o/gsmq.txt" >> "$W/progress.log"
  curl -s http://127.0.0.1:8001/metrics | grep -E "^vllm:spec_decode_num_(drafts|accepted_tokens|draft_tokens)_total" > "$o/spec-metrics.txt"
}

finish_arm(){
  local name=$1 rc=$2 o=$W/out/$1
  docker logs "$C" > "$o/server.log" 2>&1
  docker rm -f "$C" >/dev/null 2>&1
  echo "{\"arm\":\"$name\",\"rc\":$rc,\"ts\":\"$(date -u +%FT%TZ)\",\"window\":3,\"suite\":\"sampled-gsm8k-200\"}" >> "$W/ledger.jsonl"
  log "=== ARM $name DONE rc=$rc ==="
  wait_vram
}

log "=== ARM S0a START img=$IMG151 ==="
if boot S0a "$IMG151" /root/vllm7-cache 150; then suites S0a; finish_arm S0a 0; else finish_arm S0a 1; fi

log "=== ARM S1 START img=$IMGC (warm g1cache) ==="
if boot S1 "$IMGC" "$W/g1cache" 150; then suites S1; finish_arm S1 0; else finish_arm S1 1; fi

log "=== ARM S0b START img=$IMG151 ==="
if boot S0b "$IMG151" /root/vllm7-cache 150; then suites S0b; finish_arm S0b 0; else finish_arm S0b 1; fi

log "W3 ALL DONE cap=$(cat $CAP 2>/dev/null)"