#!/usr/bin/env bash
# ci/prefill_ab.sh — prefill A/B for the 4,000 tok/s goal (docs/PREFILL-4000.md).
#
# Runs the production arguments (README quickstart) with one knob changed per arm,
# each arm twice (first start compiles, the second is measured — AGENTS.md), and
# one full BetterBench run per arm (the prefill phase of the full run, which reads
# ~5 % lower than a prefill-only sweep — compare within one window).
#
# Protocol (AGENTS.md): 300 W with the firmware fan curve, control arm first AND
# repeated last, only compare arms of the same window. The script does NOT touch
# the power cap: it reads it and warns. Set 300 W before, restore 210 W / 2,800 rpm
# after (LACT, on explicit order only).
#
# Usage on the GPU host (BetterBench installed per docs/BENCHMARKS.md):
#   IMAGE=ghcr.io/slybase/vllm-sly-radiance:1.5.1-rocm10.1 \
#   OUT=/root/betterbench/prefill-ab-$(date +%Y%m%d) ./ci/prefill_ab.sh
#
# Arms (order fixed: C A B M T0 K C):
#   C   control, production 1.5.1 (262k, chunk 2048)          — first and last
#   A   chunk 4096 at 262k                                     — +3.5..4.8 % (8k-64k, measured 131k)
#   B   chunk 4096 at 131k (the README "less context" option)  — same, pool -7 %
#   M   RADIANCE_R4D_PREFILL_MIN_Q=256                         — short-prompt TTFT probe
#   T0  RADIANCE_MXFP4_A_TILED_MIN_M=0                         — re-verify the +11-13 % tiled prefill
#   K   KVCACHE_DTYPE=kv4 + chunk 4096                         — TETRA, +3-4 % prefill at chunk >= 3456
#
# Output per arm: $OUT/<arm>.json (BetterBench), $OUT/<arm>.kvlog (KV pool lines).
# Every number produced here is a measurement; the plan marks what is still open.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
IMAGE=${IMAGE:-ghcr.io/slybase/vllm-sly-radiance:1.5.1-rocm10.1}
PORT=${PORT:-8000}
OUT=${OUT:-/root/betterbench/prefill-ab-$(date +%Y%m%d)}
BB_CONFIG=${BB_CONFIG:-$ROOT/ci/accept/configs/full.json}  # identical to betterbench config/default.json
MODEL=amd/Qwen3.8-27B-Quark-AWQ-MXFP4
DRAFTER=syvai/Qwen3.8-27B-DFlash2-W4A16
NAME=${NAME:-prefill-ab}
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-1800}

mkdir -p "$OUT"

# --- power: read-only check, the operator owns the cap ------------------------
cap_mw=0
for f in /sys/class/hwmon/hwmon*/power1_max; do
  [ -r "$f" ] && cap_mw=$(cat "$f" 2>/dev/null || echo 0) && break
done
if [ "$cap_mw" -lt 300000 ]; then
  echo "WARNING: power cap ${cap_mw} mW (< 300 W). A/B requires 300 W + firmware fan curve;" >&2
  echo "         210 W costs ~20 % prefill, so results at the current cap are not comparable." >&2
  echo "         Set the cap (LACT) before running, restore 210 W / 2,800 rpm afterwards." >&2
fi

# --- one arm: two starts (warm), one full BetterBench run ---------------------
# run_arm NAME MAXLEN CHUNK EXTRA_ENV...
run_arm() {
  local name=$1 max_len=$2 chunk=$3; shift 3
  local env_args=()
  env_args=(-e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=2 \
    -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
    -e RADIANCE_MXFP4_DECODE_MAX_M=128 -e RADIANCE_MXFP4_A_TILED_MIN_M=513 -e RADIANCE_MXFP4_WPERM=1 \
    -e RADIANCE_LMHEAD_INT4=1 -e RADIANCE_FUSED_NORM_QUANT=1 \
    -e RADIANCE_KV_GROUP_SIZE=8 -e RADIANCE_EMBED_INT8=1 -e RADIANCE_EMBED_BITS=4 \
    -e RADIANCE_MXFP4_WIDE_MAX_M=192 -e RADIANCE_ADAPTIVE_WIDTH=perseq \
    "$@")
  local serve_args=(
    --model "$MODEL" --quantization quark
    --max-model-len "$max_len" --gpu-memory-utilization 0.98
    --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16
    --speculative-config.method dflash --speculative-config.model "$DRAFTER"
    --speculative-config.num_speculative_tokens 7 \
    --speculative-config.draft_sample_method probabilistic \
    --speculative-config.attention_backend TRITON_ATTN
    --attention-backend R4D_HYBRID
    --max-num-seqs 8 --max-num-batched-tokens "$chunk"
    --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
    --compilation-config.cudagraph_capture_sizes '[8,16,24,32,40,48,56,64]'
    --enable-prefix-caching --skip-mm-profiling --enable-mm-embeds
    --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
    --chat-template /opt/qwen-fixed.jinja
    --default-chat-template-kwargs '{"reasoning_effort": "medium"}'
    --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}')

  docker rm -f "$NAME" >/dev/null 2>&1 || true
  for start in 1 2; do  # first start compiles; the second is the one measured
    echo "== $name: start $start/2 ($max_len / chunk $chunk)"
    docker run -d --name "$NAME" --network=host --ipc=host \
      --device /dev/kfd --device /dev/dri \
      --security-opt seccomp=unconfined \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      -v vllm-cache:/root/.cache/vllm -v triton-cache:/root/.triton -v aiter-cache:/root/.aiter \
      "${env_args[@]}" "$IMAGE" "${serve_args[@]}" >/dev/null
    local t0=$SECONDS
    until curl -sf "localhost:$PORT/health" >/dev/null 2>&1; do
      if [ $((SECONDS - t0)) -gt "$STARTUP_TIMEOUT_S" ]; then
        docker logs --tail 50 "$NAME" >&2
        docker rm -f "$NAME" >/dev/null
        echo "FAIL: $name did not become healthy in ${STARTUP_TIMEOUT_S} s" >&2
        exit 1
      fi
      sleep 10
    done
    if [ "$start" = 1 ]; then
      echo "== $name: warm restart (first start compiled)"
      docker stop "$NAME" >/dev/null
      docker rm "$NAME" >/dev/null
    fi
  done
  docker logs "$NAME" 2>/dev/null | grep -iE "kv cache|gpu blocks|available" > "$OUT/$name.kvlog" || true
  echo "== $name: BetterBench (full run, prefill phase is the read)"
  bb run --endpoint "http://localhost:$PORT/v1" --model "$MODEL" \
    --config "$BB_CONFIG" --concurrency --no-html --out "$OUT/$name.json" \
    --note arm="$name" --note image="$IMAGE" --note max_len="$max_len" --note chunk="$chunk"
  docker rm -f "$NAME" >/dev/null
}

# control first and last: only arms of the same window are comparable
run_arm C  262144 2048
run_arm A  262144 4096
run_arm B  131072 4096
run_arm M  262144 2048 -e RADIANCE_R4D_PREFILL_MIN_Q=256
run_arm T0 262144 2048 -e RADIANCE_MXFP4_A_TILED_MIN_M=0
run_arm K  262144 4096 -e KVCACHE_DTYPE=kv4
run_arm C2 262144 2048

echo "== done: $OUT"
echo "   prefill tok/s per depth: jq '.prefill // .runs' in each $OUT/*.json"
