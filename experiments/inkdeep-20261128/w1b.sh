#!/bin/bash
# Window 2 / retry driver. Adds a VRAM gate (OLD died: C1's KV pool not yet freed at start).
# Usage: bash w1b.sh "NAME|IMAGE|COMPILE_CACHE|ENV_SPACE_SEP|SUITES" ...
# Suites: mia,ctx,lc64,lc150,gsm660 (comma-separated). Detached via setsid nohup.
set -u
W=/root/inkdeep-20261128; O=$W/out; GW=/usr/local/sbin/gpu-window
OWNER=daidalos-inkdeep; C=vllm-opt
CAP=/sys/class/drm/card0/device/hwmon/hwmon6/power1_cap
VRAM=/sys/class/drm/card0/device/mem_info_vram_used
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

res_wait(){
  local a v i
  for i in $(seq 40); do
    a=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    v=$(cat "$VRAM" 2>/dev/null || echo 0)
    [ "$a" -ge 14000 ] && [ "$v" -le 1073741824 ] && return 0
    sleep 3
  done
  log "resource gate failed (RAM ${a} MB, VRAM ${v} B)"
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

run_suites(){
  local suites=$1 o=$2 st
  for st in ${suites//,/ }; do
    case "$st" in
      mia) python3 -I /root/ink-200/ttft/mia_decode.py http://127.0.0.1:8001 inkling2-mini "$o/mia.json" --tokens 400 --reps 3 > "$o/mia.txt" 2>&1 ;;
      ctx) python3 -I /root/defaults/ctxbench.py http://127.0.0.1:8001 "$o/ctx.json" > "$o/ctx.txt" 2>&1 ;;
      lc64) python3 -I /root/mbt3/lc.py http://127.0.0.1:8001 "$o/lc64.json" 64000 100 > "$o/lc64.txt" 2>&1 ;;
      lc150) python3 -I /root/mbt3/lc.py http://127.0.0.1:8001 "$o/lc150.json" 150000 100 > "$o/lc150.txt" 2>&1 ;;
      gsm660)
        mkdir -p "$o/gsm"
        ( cd "$o" && HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 timeout 2400 /root/lmeval-venv/bin/lm_eval \
            --model local-chat-completions \
            --model_args "model=inkling2-mini,base_url=http://127.0.0.1:8001/v1/chat/completions,num_concurrent=16,max_retries=3,timeout=600" \
            --tasks gsm8k_cot_zeroshot --limit 660 --apply_chat_template \
            --gen_kwargs temperature=0,max_gen_toks=4096 --output_path "$o/gsm" --log_samples > "$o/gsm.log" 2>&1 )
        grep -a -E "exact_match" "$o/gsm.log" | tail -n 3
        ;;
    esac
  done
}

# arm spec: NAME|IMAGE|COMPILE_CACHE|ENV|SUITES
arm(){
  local spec=$1 name img tc extra suites o env_args=() kv rc=0 t0
  IFS='|' read -r name img tc extra suites <<< "$spec"
  o="$O/$name"; mkdir -p "$o"; t0=$(date -u +%s)
  res_wait || { ledger "{\"arm\":\"$name\",\"rc\":\"res_gate\"}"; return 1; }
  docker rm -f "$C" >/dev/null 2>&1 || true
  rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/sem.mp-* 2>/dev/null || true
  for kv in $extra; do env_args+=(-e "$kv"); done
  log "=== ARM $name START img=$img env=[$extra] suites=$suites ==="
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
    run_suites "$suites" "$o"
    python3 - "$o" "$suites" <<'EOF' || rc=1
import json, sys, os
o, suites = sys.argv[1], sys.argv[2].split(",")
need = {"mia": "mia.json", "ctx": "ctx.json", "lc64": "lc64.json", "lc150": "lc150.json"}
for s in suites:
    if s in need:
        p = os.path.join(o, need[s])
        assert os.path.getsize(p) > 50, f"{s} missing/empty"
        json.load(open(p))
    if s == "gsm660":
        assert os.path.getsize(os.path.join(o, "gsm.log")) > 100, "gsm.log missing"
print("results valid")
EOF
  else
    rc=1
  fi
  docker logs "$C" > "$o/server.log" 2>&1 || true
  grep -a -E "GPU KV cache size|radiance|HIP_CTRL|dflash|adaptive" "$o/server.log" | head -n 40 > "$o/hooks.txt"
  docker rm -f "$C" >/dev/null 2>&1 || true
  ledger "{\"arm\":\"$name\",\"img\":\"$img\",\"env\":\"$extra\",\"suites\":\"$suites\",\"rc\":$rc,\"secs\":$(( $(date -u +%s) - t0 ))}"
  log "=== ARM $name DONE rc=$rc ==="
  return $rc
}

[ "$(cat "$CAP")" = 210000000 ] || { log "ABORT: power cap not 210 W"; exit 1; }
[ $# -ge 1 ] || { log "ABORT: no arm specs"; exit 1; }
if [ "${PREFLIGHT:-0}" = 1 ]; then log "PREFLIGHT OK ($# arms)"; trap - EXIT; exit 0; fi

# Overlap guard: a previous driver's restore may still be running (gpu-window
# release blocks up to ~6 min while it restores prod). Never start on top of it.
PF=$W/driver.pid
if [ -f "$PF" ]; then
  old=$(cat "$PF" 2>/dev/null || echo 0)
  if [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null && grep -qa "w1" "/proc/$old/cmdline" 2>/dev/null; then
    log "ABORT: another driver alive (pid $old)"; exit 1
  fi
fi
echo $$ > "$PF"
"$GW" acquire "$OWNER" 10800
"$GW" status | grep -q "\"owner\": *\"$OWNER\"" || { log "ABORT: window not owned by us"; exit 1; }
ACQUIRED=1
systemctl stop gpu-window-guard.timer docker-vllm7.service
docker rm -f "$C" >/dev/null 2>&1 || true
for i in $(seq 30); do [ -z "$(docker ps -q)" ] && break; sleep 2; done
log "window acquired, prod stopped"

for spec in "$@"; do arm "$spec"; done
log "ALL DONE"
