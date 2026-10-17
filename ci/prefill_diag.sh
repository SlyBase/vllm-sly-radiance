#!/usr/bin/env bash
# ci/prefill_diag.sh - one-window prefill diagnosis (PREFILL-4000 Messplan, step 2).
#
# Runs ON the GPU host (LXC 2408) with the card FREE: the orchestrator has stopped production
# (Radiance on LXC 2413) before; this script never stops or starts production and refuses to run
# unless /sys/class/drm/card*/device/mem_info_vram_used < 100 MB.
#
# Phases (restartable: SKIP_CELLS=1, SKIP_TRACE=1):
#   0  launcher reality   launch_reality.txt: effective backend / chunk / max-len from the launch line
#                         cut out of the production unit, plus what the server log says it really used
#   1  cells              bench_prefill_cells.py --check, then the full cell table, in a throw-away
#                         container with the unit's device flags -> cells.log, prefill_cells.csv
#   2  trace              the unit's launch line wrapped in `rocprofv3 --kernel-trace --stats -f csv json`
#                         (attach is broken, so the launcher is wrapped), warmup, then exactly one ~2k and
#                         one ~64k prompt (random token ids, max_tokens=1, TTFT = wall time), SIGTERM so
#                         the trace flushes, then ci/prefill_trace_split.py -> trace_split.txt/.json
#
# The launch line is NOT hand-written (same approach as ci/prefill_ab.sh): it is cut out of the unit
# and only the delta (rocprofv3 wrapper, -d/--name, trace mount, --max-model-len) is applied by sed;
# cmd_trace.diff shows the delta.
#
# Usage (copy ci/prefill_diag.sh, ci/prefill_trace_split.py, sly/mxfp4/bench_prefill_cells.py to /root/prefill-diag/):
#   DRY=1 bash /root/prefill-diag/prefill_diag.sh        # print every command, no GPU, no docker run
#   nohup bash /root/prefill-diag/prefill_diag.sh > /root/prefill-diag/nohup.out 2>&1 &    # real run
# Output: $OUT (default /root/prefill-diag-<date>): status.txt, DONE, run.log, launch_reality.txt,
# cells.log, prefill_cells.csv, trace/ (rocprofv3 output, bind mount), server.log, requests.tsv,
# trace_split.txt, trace_split.json, hw.csv.
#
# Rocprofv3 caveats (rocprofv3-profiling.md): ~56 us/kernel profiler overhead (TTFTs here are NOT
# production numbers, the class SHARES are what counts); vLLM exits slowly under rocprofv3 -> 300 s
# budget after SIGTERM, then docker kill; the trace dir is a bind mount so it survives the container.

set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
UNIT=${UNIT:-/etc/systemd/system/docker-vllm7.service}
OUT=${OUT:-/root/prefill-diag-$(date +%Y%m%d)}
NAME=${NAME:-prefill-diag}
PORT=${PORT:-8000}
MODEL=${MODEL:-slybase/Swift-1.5-Qwen3.8-27B-MXFP4-GPTQ}   # first --served-model-name of the unit
MAXLEN=${MAXLEN:-73728}            # trace run only (>= 64k + margin; keeps the KV pool and trace small)
LONG_TOKENS=${LONG_TOKENS:-65000}  # "64k" request, exact token count (prompt is a list of token ids)
SHORT_TOKENS=${SHORT_TOKENS:-2000}
IDLE_S=${IDLE_S:-6}                # GPU silence between requests; the parser cuts bursts at > 2 s
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-2400}
STOP_BUDGET_S=${STOP_BUDGET_S:-300}
DRY=${DRY:-0}
SKIP_CELLS=${SKIP_CELLS:-0}
SKIP_TRACE=${SKIP_TRACE:-0}
BENCH=${BENCH:-$HERE/bench_prefill_cells.py}
# EXTRA_CELLS: extra docker args for the cells container only (e.g. bind-mount a branch-built .so / python glue
# over the image copies, plus -e knobs), so the cells test the branch kernels, not the shipped ones.
EXTRA_CELLS=${EXTRA_CELLS:-}
SPLIT=${SPLIT:-$HERE/prefill_trace_split.py}

mkdir -p "$OUT" && cd "$OUT" || exit 1
OUT=$(pwd)
TRACE=$OUT/trace

log() { echo "$(date -u +%T) $*" | tee -a run.log; echo "$*" > status.txt; }
# run CMD...: DRY=1 prints, otherwise executes
run() { if [ "$DRY" = 1 ]; then echo "DRY: $*"; else "$@"; fi; }

cleanup() {
  [ "$DRY" = 1 ] || { docker rm -f "$NAME" >/dev/null 2>&1; docker rm -f "$NAME-cells" >/dev/null 2>&1; rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/sem.mp-*; }
  kill "${SPID:-}" 2>/dev/null
  touch "$OUT/DONE"
}
trap cleanup EXIT

# --- preflight ---------------------------------------------------------------------------------
VRAM=$(cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | sort -n | tail -1)
log "preflight: vram_used=${VRAM:-unreadable} B, DRY=$DRY"
if [ "$DRY" != 1 ] && { [ -z "$VRAM" ] || [ "$VRAM" -ge 100000000 ]; }; then
  log "ABORT: card not free (vram_used >= 100 MB or unreadable); production must be stopped by the caller"; exit 1
fi
HWMON=
for h in /sys/class/drm/card*/device/hwmon/hwmon*; do [ -r "$h/power1_cap" ] && HWMON=$h; done
[ -n "$HWMON" ] && log "power cap $(( $(cat "$HWMON/power1_cap") / 1000000 )) W (read only)"
sample() {
  while :; do
    echo "$(date -u +%T),$(cat "$HWMON/power1_average"),$(cat "$HWMON"/temp[123]_input | tr '\n' ','),$(cat "$HWMON/fan1_input")" >> hw.csv
    sleep 15
  done
}
if [ -n "$HWMON" ] && [ "$DRY" != 1 ]; then sample & SPID=$!; fi

# --- launch line from the unit (identical cut as prefill_ab.sh) ----------------------------------
awk '/^ExecStart=/{f=1} f{print} /--enable-prefix-caching/{if(f) exit}' "$UNIT" | sed 's/^ExecStart=//' > launch_prod.txt
grep -q -- '--model /models/swift-mxfp4' launch_prod.txt || { log "unit is not the swift-mxfp4 profile"; exit 1; }
IMG=$(grep -oE '^  vllm-sly-radiance:[^ ]+' launch_prod.txt | tr -d ' ')
[ -n "$IMG" ] || { log "no image line in launch_prod.txt"; exit 1; }
# device flags and env straight from the unit (never hand-written)
DEVFLAGS=$(grep -oE -- '--device=[^ ]+|--group-add [0-9]+|--security-opt [^ ]+' launch_prod.txt | tr '\n' ' ')
ENVFLAGS=$(grep -oE -- '-e [A-Za-z0-9_]+=[^ ]*' launch_prod.txt | tr '\n' ' ')
log "image $IMG; device flags: $DEVFLAGS"

# --- phase 0: launcher reality (static part) ------------------------------------------------------
val() { grep -oE -- "$1 [^ ]+" launch_prod.txt | head -1 | cut -d' ' -f2; }
{
  echo "# launcher reality, $(date -u +%FT%TZ)"
  echo "unit: $UNIT ($(grep -n -- '--attention-backend' "$UNIT" | head -1 | cut -d: -f1) = attention line)"
  echo "image: $IMG"
  echo "attention-backend:  $(val --attention-backend)   (spec drafter: $(val --speculative-config.attention_backend))"
  echo "max-num-batched-tokens (chunk): $(val --max-num-batched-tokens)"
  echo "max-model-len: $(val --max-model-len)   kv-cache-dtype: $(val --kv-cache-dtype)   max-num-seqs: $(val --max-num-seqs)"
  echo "RADIANCE_R4D_PREFILL_MIN_Q: $(grep -oE 'RADIANCE_R4D_PREFILL_MIN_Q=[^ ]+' launch_prod.txt || echo 'unset in unit -> image default 512 (README)')"
  echo "A_TILED_MIN_M: $(grep -oE 'RADIANCE_MXFP4_A_TILED_MIN_M=[^ ]+' launch_prod.txt || echo unset)"
  echo "note: production proper is Radiance on LXC 2413 (homelab lxc-radiance.yml radiance_serve_max_batched_tokens: 2048); this vLLM unit is only the launch-line template"
} > launch_reality.txt
cat launch_reality.txt | tee -a run.log

# --- phase 1: cells --------------------------------------------------------------------------------
if [ "$SKIP_CELLS" != 1 ]; then
  [ -r "$BENCH" ] || { log "bench script $BENCH not found (set BENCH=)"; exit 1; }
  log "phase 1: bench_prefill_cells --check + full table"
  # env from the unit; the bench docstring pins A_TILED_MIN_M=0 so both arms (rocBLAS-like vs a-tiled) are timed at every M
  run docker rm -f "$NAME-cells"
  # shellcheck disable=SC2086
  run docker run --rm --name "$NAME-cells" $DEVFLAGS --ipc=host $ENVFLAGS \
      -e RADIANCE_MXFP4_A_TILED_MIN_M=0 -e HIP_VISIBLE_DEVICES=0 $EXTRA_CELLS \
      -v "$BENCH":/w/bench_prefill_cells.py:ro -v "$OUT":/out --entrypoint python3 "$IMG" \
      /w/bench_prefill_cells.py --check --csv /out/prefill_cells.csv > cells.log 2>&1 \
    || log "cells: non-zero exit (see cells.log; WRONG cells = weights/WPERM mismatch)"
  [ "$DRY" = 1 ] && cat cells.log
  [ "$DRY" = 1 ] || grep -c WRONG cells.log | sed 's/^/cells: WRONG lines: /' | tee -a run.log
fi

# --- phase 2: rocprofv3 kernel trace -----------------------------------------------------------------
if [ "$SKIP_TRACE" != 1 ]; then
  mkdir -p "$TRACE"
  # delta: -d instead of --rm, own name, trace mount, rocprofv3 entrypoint + wrapper args, max-model-len
  sed -e "s|--rm --name vllm7-mxfp4|-d --name $NAME --entrypoint rocprofv3 -v $TRACE:/trace|" \
      -e "s|--max-model-len [0-9]*|--max-model-len $MAXLEN|" \
      -e "s|^\(  vllm-sly-radiance:[^ ]* \\\\\)\$|\1\n  --kernel-trace --stats -f csv json -d /trace -o p -- /opt/radiance_entrypoint.sh \\\\|" \
      launch_prod.txt > cmd_trace.txt
  diff <(tr -s ' \\\n' '\n\n\n' < launch_prod.txt) <(tr -s ' \\\n' '\n\n\n' < cmd_trace.txt) > cmd_trace.diff
  grep -q 'rocprofv3' cmd_trace.txt && grep -q -- '-- /opt/radiance_entrypoint.sh' cmd_trace.txt \
    || { log "wrapper sed did not apply (image line format changed?)"; exit 1; }
  log "phase 2: trace server start (delta in cmd_trace.diff)"
  if [ "$DRY" = 1 ]; then echo "DRY: launch line below"; cat cmd_trace.txt; echo "DRY: delta"; cat cmd_trace.diff; fi
  run docker rm -f "$NAME"
  [ "$DRY" = 1 ] || rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/sem.mp-*
  run bash cmd_trace.txt >/dev/null || { log "trace: docker run failed"; exit 1; }

  wait_healthy() {
    local t0=$SECONDS
    until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
      [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || return 1
      [ $((SECONDS - t0)) -gt "$STARTUP_TIMEOUT_S" ] && return 1
      sleep 5
    done
  }
  if [ "$DRY" = 1 ]; then
    echo "DRY: wait /health (<= ${STARTUP_TIMEOUT_S}s); requests: 2x ${SHORT_TOKENS}, 1x ${LONG_TOKENS} warmup, idle ${IDLE_S}s;"
    echo "DRY: measured: 1x ${SHORT_TOKENS}, idle, 1x ${LONG_TOKENS}, idle; docker kill -s TERM; wait <= ${STOP_BUDGET_S}s; docker kill"
  else
    wait_healthy || { docker logs "$NAME" > server.log 2>&1; log "trace: container died or no /health"; exit 1; }
    log "trace: healthy"
    : > requests.tsv; echo -e "tag\tprompt_tokens\twall_s(~TTFT, profiler on)" >> requests.tsv
    req() {  # req TAG N_TOKENS: random token-id prompt (unique), max_tokens=1
      python3 - "$PORT" "$MODEL" "$1" "$2" >> requests.tsv <<'PY'
import json, random, sys, time, urllib.request
port, model, tag, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ids = [random.randrange(1000, 100000) for _ in range(n)]
body = json.dumps({"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0}).encode()
t0 = time.time()
r = json.load(urllib.request.urlopen(urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions", body, {"Content-Type": "application/json"}), timeout=1200))
print(f"{tag}\t{r['usage']['prompt_tokens']}\t{time.time() - t0:.2f}")
PY
    }
    log "trace: warmup"
    req warm_short1 "$SHORT_TOKENS"; req warm_short2 "$SHORT_TOKENS"; req warm_long "$LONG_TOKENS"
    sleep "$IDLE_S"; log "trace: measured 2k";  req measured_2k "$SHORT_TOKENS"
    sleep "$IDLE_S"; log "trace: measured 64k"; req measured_64k "$LONG_TOKENS"
    sleep "$IDLE_S"
    cat requests.tsv | tee -a run.log
    log "trace: SIGTERM (clean flush), budget ${STOP_BUDGET_S}s"
    docker kill -s TERM "$NAME" >/dev/null
    t0=$SECONDS
    while [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] && [ $((SECONDS - t0)) -lt "$STOP_BUDGET_S" ]; do sleep 5; done
    docker logs "$NAME" > server.log 2>&1
    docker kill "$NAME" >/dev/null 2>&1   # no-op if already exited; trace is in the bind mount
    docker rm -f "$NAME" >/dev/null 2>&1
    rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/sem.mp-*
    log "trace: container gone after $((SECONDS - t0)) s; $(find "$TRACE" -name '*kernel_trace.csv' | wc -l) kernel_trace.csv"
    # launcher reality, dynamic part: what the server really chose
    { echo "--- from server.log (what really ran) ---"
      grep -iE "attention backend|attn_backend|R4D|chunked prefill|max_num_batched_tokens|max_seq_len|GPU KV cache size" server.log | head -20
    } | tee -a launch_reality.txt
    # 2k request = ceil(2000/chunk) forwards; 64k = ceil(65000/chunk)
    CH=$(val --max-num-batched-tokens)
    python3 "$SPLIT" "$TRACE" --segments 2 --labels 2k,64k --idle-gap-s 2 \
      --forwards "2k=$(( (SHORT_TOKENS + CH - 1) / CH )),64k=$(( (LONG_TOKENS + CH - 1) / CH ))" \
      --json trace_split.json > trace_split.txt 2>&1 || log "trace: parser failed (see trace_split.txt)"
    cat trace_split.txt | tee -a run.log
  fi
fi
log "ALL PHASES FINISHED"
