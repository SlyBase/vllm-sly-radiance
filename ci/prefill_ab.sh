#!/usr/bin/env bash
# ci/prefill_ab.sh - prefill A/B for the 4,000 tok/s goal (docs/PREFILL-4000.md).
#
# Runs ON the GPU host (LXC 2408) inside an acquired gpu-window with production stopped.
# The launch line is NOT hand-written: it is cut out of the production unit
# (docker-vllm7.service = profile swift-mxfp4, checkpoint /root/swift-heretic-gptq,
# compressed-tensors / mxfp4-pack-quantized, no --quantization flag) and only the arm's
# delta is applied with sed. A diff of the token streams is stored per arm, so the delta
# is auditable. Same approach as the README 1.5.1 reference run.
#
# Per arm: start (twice for non-control arms: the first start compiles fresh and shows a
# smaller KV pool), read "GPU KV cache size", then a BetterBench prefill-only sweep
# (target depths 2000 / 16000 / 21500 / 64000 = ~1.5k / 11.8k / 16k / 47k real prompt tokens, 2 warmup + 8 runs each, unique prompts, no prefix cache).
# A prefill-only sweep reads ~5 % higher than the prefill phase of a full run (BENCHMARKS.md),
# so compare arms only inside one window, against arm C of the same window.
#
# The script never touches the power cap; it records it (the production cap is 210 W).
#
# Usage:
#   OUT=/root/prefill-ab-$(date +%Y%m%d) ci/prefill_ab.sh              # all arms
#   ARMS="C A" OUT=... ci/prefill_ab.sh                                 # subset
#   SHORT=1 K2_SO=/root/k2-compile/c/radiance_mxfp4_fp8.so OUT=... ci/prefill_ab.sh
#                                                                       # short screening (below)
#
# SHORT=1 (screening mode): one start per arm (the compile cache must be warm, so the KV pool is
# only logged, never re-started for), prefill sweep only (no decode phase) at ~1.5k / 16k / 47k
# real prompt tokens, 3 runs per depth after 1 warmup, default arms ABBA = C K2 R5 C2. A gain below
# 5 % in this mode is unconfirmed until a full run (default mode) repeats it.
# Arms K2 and R5 are only valid with the 210 W / 300 W cap held constant across the window.
#
# QUANT_FLAG="compressed-tensors" adds `--quantization compressed-tensors` to EVERY arm. The default
# is empty because the production unit (homelab swift-mxfp4 profile) has no --quantization flag:
# vLLM reads compressed-tensors from the checkpoint's config.json, and an extra explicit flag would
# be a launch delta against production (and against the measured baselines).
#
# Build the K2 object CPU-only inside the release image (no GPU device needed, ~3 min):
#   docker run --rm -v /root/k2:/k2 --entrypoint bash vllm-sly-radiance:<tag> -c \
#     'hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=gfx1201 -DRADIANCE_GROUPM=16 \
#        $(python -m pybind11 --includes) /k2/radiance_mxfp4_fp8.hip -o /k2/radiance_mxfp4_fp8.so'
# with sly/mxfp4/radiance_mxfp4_fp8.hip of this branch copied to /root/k2 first; then K2_SO=/root/k2/radiance_mxfp4_fp8.so.
#
# Arms (default order: C A B M T0 K C2; SHORT: C K2 R5 C2; C2 = control again, drift check):
#   C   control, production 1.5.1 (262k, chunk 2048)
#   A   chunk 4096 at 262k
#   B   chunk 4096 at 131k (the README "less context" option)
#   M   RADIANCE_R4D_PREFILL_MIN_Q=256
#   T0  RADIANCE_MXFP4_A_TILED_MIN_M=0
#   K   4-bit KV (int4_per_token_head, TETRA target + drafter attention) + chunk 4096
#   K2  libr4d-style M-band raster grouping of the W4A8 prefill GEMM (RADIANCE_GROUPM=16 build of
#       radiance_mxfp4_fp8.so, bind-mounted over the image's .so; K2_SO=<path of that .so>)
#   R5  R4D_ATTN_FP8=3 (8-bit QK and PV legs of the libr4d prefill attention, R4D_HYBRID)
#   C2  control again
#
# Output in $OUT: <arm>.prefill.json (BetterBench), <arm>.kvlog, <arm>.diff (launch delta),
# <arm>.server.log, hw.csv (power / temps / fan every 15 s), summary.tsv, status.txt, DONE.

set -uo pipefail

UNIT=${UNIT:-/etc/systemd/system/docker-vllm7.service}
BB_DIR=${BB_DIR:-/root/betterbench}
OUT=${OUT:-/root/prefill-ab-$(date +%Y%m%d)}
NAME=${NAME:-prefill-ab}
PORT=${PORT:-8000}
MODEL=${MODEL:-slybase/Swift-1.5-Qwen3.8-27B-MXFP4-GPTQ}   # first --served-model-name of the unit
# BetterBench depths are TARGETS at ~4 chars/token; the server reports real prompt tokens, which come
# out ~25 % lower (2000 -> ~1.5k, 21500 -> ~16k, 64000 -> ~47k; 16000 -> 11.8k is the README "16k" row).
DEPTHS=${DEPTHS:-[2000,16000,21500,64000]}
SHORT=${SHORT:-0}
QUANT_FLAG=${QUANT_FLAG:-}
K2_SO=${K2_SO:-}
O_DIR=${O_DIR:-}
SP=${SP:-/opt/vllm/lib/python3.12/site-packages}   # site-packages inside the image
if [ "$SHORT" = 1 ]; then
  DEPTHS=${DEPTHS_SHORT:-[2000,21500,64000]}        # ~1.5k / 16k / 47k real prompt tokens
  RUNS=${RUNS:-3}; WARMUP=${WARMUP:-1}
  ARMS=${ARMS:-C K2 R5 C2}
else
  RUNS=${RUNS:-8}; WARMUP=${WARMUP:-2}
  ARMS=${ARMS:-C A B M T0 K C2}
fi
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-1500}

mkdir -p "$OUT"
cd "$OUT" || exit 1
OUT=$(pwd)

log() { echo "$(date -u +%T) $*" | tee -a run.log; echo "$*" > status.txt; }

# --- read-only hardware facts -------------------------------------------------
HWMON=
for h in /sys/class/drm/card*/device/hwmon/hwmon*; do
  [ -r "$h/power1_cap" ] && HWMON=$h
done
[ -n "$HWMON" ] || { echo "no GPU hwmon found" >&2; exit 1; }
CAP_W=$(( $(cat "$HWMON/power1_cap") / 1000000 ))
log "power cap ${CAP_W} W (read only; not changed by this script)"

sample() {
  while :; do
    echo "$(date -u +%T),$(cat "$HWMON/power1_average"),$(cat "$HWMON"/temp[123]_input | tr '\n' ','),$(cat "$HWMON/fan1_input")" >> hw.csv
    sleep 15
  done
}

cleanup() {
  kill "${SPID:-}" 2>/dev/null
  docker rm -f "$NAME" >/dev/null 2>&1
  rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/sem.mp-*
  touch "$OUT/DONE"
}
trap cleanup EXIT

# --- bench config: only the prefill depths and run counts differ from BetterBench's default -
python3 - "$BB_DIR/config/default.json" "$DEPTHS" "$RUNS" "$WARMUP" > bb_cfg.json <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1]))
cfg["prefill_depths"] = json.loads(sys.argv[2])
cfg["prefill_runs"] = int(sys.argv[3])
cfg["prefill_warmup"] = int(sys.argv[4])
print(json.dumps(cfg))
PY

# --- production launch line from the unit ------------------------------------
# From "ExecStart=" up to and including "--enable-prefix-caching" (the last flag of the unit).
awk '/^ExecStart=/{f=1} f{print} /--enable-prefix-caching/{if(f) exit}' "$UNIT" \
  | sed 's/^ExecStart=//' > launch_prod.txt
grep -q -- '--model /models/swift-mxfp4' launch_prod.txt || { log "unit is not the swift-mxfp4 profile"; exit 1; }
if grep -q -- '--quantization' launch_prod.txt; then log "unit carries --quantization; refusing"; exit 1; fi
if [ -n "$QUANT_FLAG" ]; then   # applied to every arm, so it never confounds an A/B
  sed -i "s|^  --model /models/swift-mxfp4 |  --quantization $QUANT_FLAG --model /models/swift-mxfp4 |" launch_prod.txt
  grep -q -- "--quantization $QUANT_FLAG" launch_prod.txt || { log "could not insert --quantization"; exit 1; }
fi

# arm_sed ARM -> sed expressions (one per line) that turn the production line into the arm's
arm_sed() {
  case $1 in
    C|C2) ;;
    A)  echo "s/--max-num-batched-tokens 2048/--max-num-batched-tokens 4096/" ;;
    B)  echo "s/--max-num-batched-tokens 2048/--max-num-batched-tokens 4096/"
        echo "s/--max-model-len 262144/--max-model-len 131072/" ;;
    M)  echo "s/-e GPU_MAX_HW_QUEUES=1 /-e GPU_MAX_HW_QUEUES=1 -e RADIANCE_R4D_PREFILL_MIN_Q=256 /" ;;
    T0) echo "s/RADIANCE_MXFP4_A_TILED_MIN_M=513/RADIANCE_MXFP4_A_TILED_MIN_M=0/" ;;
    K2) [ -r "$K2_SO" ] || { echo "K2 needs K2_SO=<radiance_mxfp4_fp8.so built with -DRADIANCE_GROUPM=16>" >&2; return 1; }
        echo "s|-v /root/swift-heretic-gptq:/models/swift-mxfp4:ro |-v /root/swift-heretic-gptq:/models/swift-mxfp4:ro -v $K2_SO:$SP/radiance_mxfp4_fp8.so:ro |" ;;
    O)  # tiled gdn_norm_quant (out_proj producer): .so built from this branch + patched python glue,
        # both bind-mounted over the image (O_DIR holds radiance_mxfp4_fp8.so and radiance_fused_norm.py)
        [ -r "$O_DIR/radiance_mxfp4_fp8.so" ] && [ -r "$O_DIR/radiance_fused_norm.py" ] || { echo "O needs O_DIR=<dir with radiance_mxfp4_fp8.so + radiance_fused_norm.py>" >&2; return 1; }
        echo "s|-v /root/swift-heretic-gptq:/models/swift-mxfp4:ro |-v /root/swift-heretic-gptq:/models/swift-mxfp4:ro -v $O_DIR/radiance_mxfp4_fp8.so:$SP/radiance_mxfp4_fp8.so:ro -v $O_DIR/radiance_fused_norm.py:$SP/radiance_fused_norm.py:ro -e RADIANCE_MXFP4_TILED_OPROJ=1 |" ;;
    O0) # control for O: same mounts, knob off (proves the new .so alone changes nothing)
        [ -r "$O_DIR/radiance_mxfp4_fp8.so" ] && [ -r "$O_DIR/radiance_fused_norm.py" ] || { echo "O0 needs O_DIR" >&2; return 1; }
        echo "s|-v /root/swift-heretic-gptq:/models/swift-mxfp4:ro |-v /root/swift-heretic-gptq:/models/swift-mxfp4:ro -v $O_DIR/radiance_mxfp4_fp8.so:$SP/radiance_mxfp4_fp8.so:ro -v $O_DIR/radiance_fused_norm.py:$SP/radiance_fused_norm.py:ro |" ;;
    R5) echo "s/-e GPU_MAX_HW_QUEUES=1 /-e GPU_MAX_HW_QUEUES=1 -e R4D_ATTN_FP8=3 /" ;;
    K)  echo "s/--max-num-batched-tokens 2048/--max-num-batched-tokens 4096/"
        echo "s/--kv-cache-dtype fp8/--kv-cache-dtype int4_per_token_head/"
        echo "s/--attention-backend R4D_HYBRID/--attention-backend TETRA/"
        echo "s/--speculative-config.attention_backend TRITON_ATTN/--speculative-config.attention_backend TETRA/" ;;
    *)  return 1 ;;
  esac
}

wait_healthy() {  # 0 = healthy, 1 = container died or timeout
  local t0=$SECONDS
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = true ] || return 1
    [ $((SECONDS - t0)) -gt "$STARTUP_TIMEOUT_S" ] && return 1
    sleep 5
  done
}

start_arm() {  # start_arm ARM -> container healthy
  local arm=$1 exprs=() e
  while IFS= read -r e; do [ -n "$e" ] && exprs+=(-e "$e"); done < <(arm_sed "$arm")
  # -d instead of --rm and our own name: a crashed candidate keeps its log
  sed -e "s/--rm --name vllm7-mxfp4/-d --name $NAME/" "${exprs[@]}" launch_prod.txt > "cmd_$arm.txt"
  diff <(tr -s ' \\\n' '\n\n\n' < launch_prod.txt) <(tr -s ' \\\n' '\n\n\n' < "cmd_$arm.txt") > "$arm.diff"
  docker rm -f "$NAME" >/dev/null 2>&1
  rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/sem.mp-*
  bash "cmd_$arm.txt" >/dev/null 2>&1 || { log "$arm: docker run failed"; return 1; }
  if ! wait_healthy; then
    docker logs "$NAME" > "$arm.server.log" 2>&1
    log "$arm: container died or no /health"; return 1
  fi
}

kv_line() { docker logs "$NAME" 2>&1 | grep -m1 "GPU KV cache size" | sed 's/.*GPU KV cache size/GPU KV cache size/'; }

run_arm() {
  local arm=$1 starts=2 n kv
  case $arm in C|C2) starts=1 ;; esac   # control: the production compile cache is warm
  [ "$SHORT" = 1 ] && starts=1          # screening: one start per arm
  for n in $(seq 1 "$starts"); do
    log "$arm: start $n/$starts"
    start_arm "$arm" || return 1
    log "$arm: healthy, $(kv_line)"
    if [ "$n" -lt "$starts" ]; then
      docker rm -f "$NAME" >/dev/null 2>&1; sleep 5
    fi
  done
  # A control pool off the 436-439k reference means a cold cache: one more start.
  [ "$SHORT" != 1 ] && case $arm in C|C2)
    # (capture first: `kv_line | grep -q` dies of SIGPIPE under pipefail and always reads "off")
    kv=$(kv_line)
    if ! grep -Eq '43[0-9],[0-9]{3} tokens' <<<"$kv"; then
      log "$arm: KV pool off the 436-439k reference (cold cache?), second start"
      docker rm -f "$NAME" >/dev/null 2>&1; sleep 5
      start_arm "$arm" || return 1
      log "$arm: healthy, $(kv_line)"
    fi ;;
  esac
  kv_line > "$arm.kvlog"
  if [ "$SHORT" = 1 ] && ! grep -Eq '43[0-9],[0-9]{3} tokens' "$arm.kvlog"; then
    log "$arm: KV pool off the 436-439k reference (cold cache or arm delta; read the kvlog)"
  fi
  log "$arm: BetterBench prefill sweep $DEPTHS"
  ( cd "$BB_DIR" && PYTHONPATH="$BB_DIR" python3 -u -c "
import sys
from betterbench.cli import main
sys.argv = ['bb', 'run', '--endpoint', 'http://127.0.0.1:$PORT/v1', '--model', '$MODEL',
            '--prefill', '--config', '$OUT/bb_cfg.json', '--out', '$OUT/$arm.prefill.json', '--no-html',
            '--note', 'arm=$arm', '--note', 'power=${CAP_W}W', '--note', 'window=prefill-ab']
main()
" > "$OUT/$arm.bb.log" 2>&1 ) || log "$arm: BetterBench exit non-zero (see $arm.bb.log)"
  if [ "$SHORT" = 1 ]; then   # decode guard: 3 sequential 256-token requests, not a BetterBench decode run
    log "$arm: decode sample"
    python3 - "$PORT" "$MODEL" > "$arm.decode.txt" 2>&1 <<'PY'
import json, sys, time, urllib.request
port, model = sys.argv[1:3]
for i in range(4):   # first request is a warmup and is not reported as a rate
    body = json.dumps({"model": model, "max_tokens": 256, "temperature": 0, "ignore_eos": True,
                       "messages": [{"role": "user", "content": "Write a long story about a lighthouse keeper."}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", body, {"Content-Type": "application/json"}), timeout=300))
    dt = time.time() - t0
    n = r["usage"]["completion_tokens"]
    print(f"run{i} {'warmup' if i == 0 else 'sample'} tokens={n} s={dt:.2f} tps={n / dt:.1f}")
PY
  fi
  docker logs "$NAME" > "$arm.server.log" 2>&1
  docker rm -f "$NAME" >/dev/null 2>&1
  rm -f /dev/shm/vllm_offload_*.mmap /dev/shm/sem.mp-*
  sleep 10
  log "$arm: stopped, vram_used=$(cat "$(dirname "$HWMON")/../mem_info_vram_used" 2>/dev/null)"
}

sample & SPID=$!
for arm in $ARMS; do
  arm_sed "$arm" >/dev/null || { log "unknown arm $arm"; continue; }
  run_arm "$arm" || log "$arm: FAILED"
done

# summary: median prefill tok/s per depth and arm
python3 - "$OUT" $ARMS > summary.tsv <<'PY'
import json, os, statistics, sys
out, arms = sys.argv[1], sys.argv[2:]
print("arm\tdepth_target\tprompt_tokens\tn\tpp_tps_median\tpp_tps_min\tpp_tps_max\tttft_ms_median")
for a in arms:
    p = os.path.join(out, a + ".prefill.json")
    if not os.path.exists(p):
        print(f"{a}\tNO RESULT")
        continue
    for row in json.load(open(p)).get("prefill", []):
        pp = row.get("pp_tps") or []
        if not pp:
            print(f"{a}\t{row.get('target_depth')}\t-\t0\t-\t-\t-\t-")
            continue
        pt = row.get("prompt_tokens") or [0]
        print(f"{a}\t{row['target_depth']}\t{int(statistics.median(pt))}\t{len(pp)}\t"
              f"{statistics.median(pp):.0f}\t{min(pp):.0f}\t{max(pp):.0f}\t{statistics.median(row['ttft_ms']):.0f}")
PY
log "ALL ARMS FINISHED"
