#!/bin/bash
# Window 2: quality gate for the 1.15.2 decode-tile fix (PR #279).
# G0a = 1.15.1 control FIRST, G1 = candidate 1.15.2 (fresh compile, measure 2nd
# start), G0b = 1.15.1 control LAST (noise floor). Same serve flags as window 1
# (0.92 / 307200 / fp8 / k7 probabilistic / medium). 210 W untouched.
# Suites per arm: mia, ctx, lc64, lc150 (+4000-token fail rerun), sampled gsmq
# (temp 0.7, n=200), greedy GSM8K-660 (lm_eval, temp 0). G1 extra: conc probe.
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
  log "W2 RESTORE: prod health=$h cap=$(cat $CAP 2>/dev/null)"
}
trap restore EXIT
PF=$W/driver.pid
if [ -f "$PF" ]; then
  old=$(cat "$PF" 2>/dev/null || echo 0)
  if [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null && grep -qa "w2" "/proc/$old/cmdline" 2>/dev/null; then
    log "ABORT: another driver alive (pid $old)"; exit 1
  fi
fi
echo $$ > "$PF"
"$GW" acquire "$OWNER" 7200
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
log "W2 START cap=$(cat $CAP 2>/dev/null) vram=$(cat $VRAM)"

wait_vram(){ for i in $(seq 1 24); do v=$(cat "$VRAM" 2>/dev/null || echo 999999999999); [ "$v" -le 1073741824 ] && break; sleep 5; done; }

# boot NAME IMG CACHEDIR BOOTWAIT -> 0 if healthy
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
  log "$name mia"; timeout 600 python3 -I /root/ink-200/ttft/mia_decode.py http://127.0.0.1:8001 inkling2-mini "$o/mia.json" --tokens 400 --reps 3 > "$o/mia.txt" 2>&1
  log "$name ctx"; timeout 900 python3 -I /root/defaults/ctxbench.py http://127.0.0.1:8001 "$o/ctx.json" > "$o/ctx.txt" 2>&1
  log "$name lc64"; timeout 900 python3 -I /root/mbt3/lc.py http://127.0.0.1:8001 "$o/lc64.json" 64000 100 > "$o/lc64.txt" 2>&1
  ids=$(python3 -c "import json;print(','.join(str(x['id']) for x in json.load(open('$o/lc64.json'))['res'] if not x['ok']))" 2>/dev/null)
  [ -n "$ids" ] && { log "$name lc64 rerun ids=$ids"; LC_IDS="$ids" LC_MAXTOK=4000 timeout 900 python3 -I /root/mbt3/lc.py http://127.0.0.1:8001 "$o/lc64r.json" 64000 100 > "$o/lc64r.txt" 2>&1; }
  log "$name lc150"; timeout 1500 python3 -I /root/mbt3/lc.py http://127.0.0.1:8001 "$o/lc150.json" 150000 100 > "$o/lc150.txt" 2>&1
  ids=$(python3 -c "import json;print(','.join(str(x['id']) for x in json.load(open('$o/lc150.json'))['res'] if not x['ok']))" 2>/dev/null)
  [ -n "$ids" ] && { log "$name lc150 rerun ids=$ids"; LC_IDS="$ids" LC_MAXTOK=4000 timeout 1500 python3 -I /root/mbt3/lc.py http://127.0.0.1:8001 "$o/lc150r.json" 150000 100 > "$o/lc150r.txt" 2>&1; }
  log "$name gsmq sampled"; ( cd "$o" && URL=http://127.0.0.1:8001 PAR=8 TEMP=0.7 timeout 600 python3 /root/pr-ab/gsmq.py "$name" 200 > "$o/gsmq.txt" 2>&1 )
  log "$name gsm greedy 660"; mkdir -p "$o/gsm"
  ( cd "$o" && HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 timeout 2400 /root/lmeval-venv/bin/lm_eval --model local-chat-completions \
    --model_args model=inkling2-mini,base_url=http://127.0.0.1:8001/v1/chat/completions,num_concurrent=16,max_retries=3,timeout=600 \
    --tasks gsm8k_cot_zeroshot --limit 660 --apply_chat_template --gen_kwargs temperature=0,max_gen_toks=4096 \
    --output_path "$o/gsm" --log_samples > "$o/gsm.log" 2>&1 )
  grep -a -E "exact_match" "$o/gsm.log" | tail -3 | cut -c1-200 >> "$W/progress.log"
  curl -s http://127.0.0.1:8001/metrics | grep -E "^vllm:spec_decode_num_(drafts|accepted_tokens|draft_tokens)_total" > "$o/spec-metrics.txt"
}

finish_arm(){
  local name=$1 rc=$2 o=$W/out/$1
  docker logs "$C" > "$o/server.log" 2>&1
  docker rm -f "$C" >/dev/null 2>&1
  echo "{\"arm\":\"$name\",\"rc\":$rc,\"ts\":\"$(date -u +%FT%TZ)\"}" >> "$W/ledger.jsonl"
  log "=== ARM $name DONE rc=$rc ==="
  wait_vram
}

# --- G0a: 1.15.1 control first (warm prod caches, same image) ---
log "=== ARM G0a START img=$IMG151 ==="
if boot G0a "$IMG151" /root/vllm7-cache 150; then suites G0a; finish_arm G0a 0; else finish_arm G0a 1; fi

# --- G1: candidate, fresh caches; first start compiles, measure second ---
log "=== ARM G1 START img=$IMGC (warmup start) ==="
if boot G1w "$IMGC" "$W/g1cache" 240; then
  mkdir -p "$W/out/G1"
  docker logs "$C" > "$W/out/G1/boot-warmup.log" 2>&1
  docker rm -f "$C" >/dev/null 2>&1; wait_vram
  log "G1 warmup done; measure start"
  if boot G1 "$IMGC" "$W/g1cache" 150; then
    suites G1
    log "G1 conc probe"; timeout 600 python3 -I "$W/conc.py" http://127.0.0.1:8001 "$W/out/G1/conc.json" > "$W/out/G1/conc.txt" 2>&1
    finish_arm G1 0
  else finish_arm G1 1; fi
else
  mkdir -p "$W/out/G1"; docker logs "$C" > "$W/out/G1/boot-warmup.log" 2>&1
  finish_arm G1 1
fi
"$GW" acquire "$OWNER" 3600 >> "$W/progress.log" 2>&1 || true

# --- G0b: 1.15.1 control last (noise floor) ---
log "=== ARM G0b START img=$IMG151 ==="
if boot G0b "$IMG151" /root/vllm7-cache 150; then suites G0b; finish_arm G0b 0; else finish_arm G0b 1; fi

log "W2 ALL DONE cap=$(cat $CAP 2>/dev/null)"
exit 0