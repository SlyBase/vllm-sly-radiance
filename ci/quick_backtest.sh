#!/usr/bin/env bash
# Backtest of the RELEASE tier (ci/quick_bench.py run --release: quick tier + 8 streams + GSM8K-50) against images that are already on host 51 and whose
# behaviour is KNOWN. Proves the tier flags every regression we have actually met. NOT run by CI:
# it needs the GPU. The caller holds the gpu-window and has stopped docker-vllm7 (same contract as the
# ink-200 arm scripts); this script starts and removes only its own container on :8001.
#
#   RECORD=1 ARMS="Q-GOOD I-GOOD" ci/quick_backtest.sh   # window 1: write the references from the two good arms
#   ci/quick_backtest.sh            # compare every arm against the committed references
#   ARMS="Q-GOOD I-INT4" ci/quick_backtest.sh   # a subset
#
# Arm          image                          expected   what it proves
#   Q-GOOD     1.13.4-rocm10.1 (Qwen)         PASS       reference arm (last release with Qwen args, QP2)
#   Q-GOOD2    1.14.0-rocm10.1 (Qwen)         PASS       new release must pass against the 1.13.4 reference
#   Q-EAGER    ci-262-ff1aff1 (Qwen)          FAIL hook  eager model (compilation mode NONE, no add_rms_quant)
#   I-GOOD     1.14.0-rocm10.1 (Inkling)      PASS       reference arm (released image)
#   I-CONC     ci-262-032eabe (Inkling)       FAIL       2-/4-stream collapse (c2/c1, c4/c1)
#   I-INT4     1.14.0 + CONV_W4=1             FAIL       int4 drafter (hook "conv projections int4 0", tok/update)
# Exit 0 only when every arm gave its expected verdict.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-/root/quick-bt/out}"
CACHE="${CACHE:-/root/quick-bt/cache}"
PORT=8001
C=quick-bt
ARMS="${ARMS:-Q-GOOD Q-GOOD2 Q-EAGER I-GOOD I-CONC I-INT4}"
mkdir -p "$OUT" "$CACHE"/{qwen,inkling}/{triton,torch_compile,aiter}

declare -A IMAGE=([Q-GOOD]=vllm-sly-radiance:1.13.4-rocm10.1 [Q-GOOD2]=vllm-sly-radiance:1.14.0-rocm10.1 [Q-EAGER]=vllm-sly-radiance:ci-262-ff1aff1
                  [I-GOOD]=vllm-sly-radiance:1.14.0-rocm10.1 [I-CONC]=vllm-sly-radiance:ci-262-032eabe
                  [I-INT4]=vllm-sly-radiance:1.14.0-rocm10.1)
declare -A EXPECT=([Q-GOOD]=PASS [Q-GOOD2]=PASS [Q-EAGER]=FAIL [I-GOOD]=PASS [I-CONC]=FAIL [I-INT4]=FAIL)

start_qwen() {  # production unit args of 1.13.4 (QP2/docker_cmd.txt); one -e per variable
  docker run -d --name "$C" --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    --security-opt seccomp=unconfined --ipc=host \
    -e HIP_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e GPU_MAX_HW_QUEUES=1 -e RADIANCE_MXFP4=1 \
    -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 -e RADIANCE_MXFP4_DECODE_MAX_M=128 \
    -e RADIANCE_MXFP4_SANITIZE=0 -e RADIANCE_LMHEAD_FP8=1 -e RADIANCE_LMHEAD_INT4=1 \
    -e RADIANCE_FUSED_NORM_QUANT=1 -e RADIANCE_KV_GROUP_SIZE=8 -e RADIANCE_EMBED_INT8=1 \
    -e RADIANCE_EMBED_BITS=4 -e RADIANCE_MXFP4_A_TILED_MIN_M=513 -e RADIANCE_MXFP4_TN4_MIN_M=513 \
    -e RADIANCE_FUSED_NQ_REG=1 -e RADIANCE_MXFP4_AT_SKIP_PAD=1 -e RADIANCE_MXFP4_TILED_OPROJ=1 \
    -e R4D_ATTN_FP8=3 -e RADIANCE_MXFP4_WPERM=1 -e RADIANCE_MXFP4_WIDE_MAX_M=192 \
    -e RADIANCE_ADAPTIVE_WIDTH=perseq -e RADIANCE_CACTUS_DELTA=0.5 \
    -e RADIANCE_MXFP4_BCQ=/models/swift-mxfp4/bcq_tables.pt \
    -e RADIANCE_LMHEAD_INT4_CODES=/models/swift-mxfp4/lm_head_int4.pt \
    -p "$PORT:8000" -v /root/hf-cache:/root/.cache/huggingface -v /root/mcb-ckpt-v2:/models/swift-mxfp4:ro \
    -v "$CACHE/qwen/triton:/root/.triton" -v "$CACHE/qwen/torch_compile:/root/.cache/vllm/torch_compile_cache" \
    -v "$CACHE/qwen/aiter:/root/.aiter" "$1" \
    --model /models/swift-mxfp4 --served-model-name qwen-quick --max-model-len 262144 \
    --gpu-memory-utilization 0.98 --kv-cache-dtype fp8 --port 8000 \
    --speculative-config.method dflash --speculative-config.model syvai/Qwen3.8-27B-DFlash2-W4A16 \
    --speculative-config.num_speculative_tokens 7 --speculative-config.draft_sample_method probabilistic \
    --speculative-config.attention_backend TRITON_ATTN \
    --override-generation-config '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}' \
    --max-num-seqs 8 --max-num-batched-tokens 4096 --mamba-ssm-cache-dtype bfloat16 \
    --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
    --compilation-config.cudagraph_capture_sizes '[8,16,24,32,40,48,56,64]' \
    --skip-mm-profiling --enable-mm-embeds --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
    --attention-backend R4D_HYBRID --enable-auto-tool-choice --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 --chat-template /opt/qwen-fixed.jinja \
    --default-chat-template-kwargs '{"reasoning_effort": "medium"}' --enable-prefix-caching \
    --no-enable-extensible-kv-cache --kv-cache-memory-bytes 15569256448 >/dev/null
}

start_inkling() {  # release arm of the ink-200 week (rel/out5/ON/args.json); $2 = optional extra "-e" pair
  docker run -d --name "$C" --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt label=disable --ipc=host \
    -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=1 -e RADIANCE_INKLING2_HIP_MOE=1 \
    -e RADIANCE_INKLING2_MXFP4_DENSE=1 -e HF_HUB_OFFLINE=1 ${2:+-e "$2"} \
    -p "$PORT:8000" -v /root/inkling2-mini:/model:ro -v /root/dflash2-ink/model:/models/dflash2:ro \
    -v "$CACHE/inkling/triton:/root/.triton" "$1" /model \
    --served-model-name inkling2-mini --gpu-memory-utilization 0.92 --limit-mm-per-prompt '{"image":0}' \
    --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
    --max-model-len 32768 --max-num-seqs 8 --no-enable-prefix-caching --no-async-scheduling \
    --speculative-config '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto"}' >/dev/null
}

wait_health() {
  for _ in $(seq 1 600); do
    [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health")" = 200 ] && return 0
    [ "$(docker inspect -f '{{.State.Running}}' "$C" 2>/dev/null)" = true ] || return 1
    sleep 2
  done
  return 1
}

declare -A GOT SECS
for arm in $ARMS; do
  profile=qwen; model=qwen-quick; [[ $arm == I-* ]] && { profile=inkling; model=inkling2-mini; }
  echo "=== $arm  image=${IMAGE[$arm]}  expect=${EXPECT[$arm]}"; T0=$SECONDS
  docker rm -f "$C" >/dev/null 2>&1
  if [ "$profile" = qwen ]; then start_qwen "${IMAGE[$arm]}"
  elif [ "$arm" = I-INT4 ]; then start_inkling "${IMAGE[$arm]}" RADIANCE_DFLASH_CONV_W4=1
  else start_inkling "${IMAGE[$arm]}"; fi
  if wait_health; then
    T1=$SECONDS
    [ "$arm" = I-INT4 ] && { docker exec "$C" env | grep -qx RADIANCE_DFLASH_CONV_W4=1 || echo "::error::env proof failed: RADIANCE_DFLASH_CONV_W4=1 not in container"; }
    docker logs "$C" > "$OUT/$arm.server.log" 2>&1
    rec=(); [ "${RECORD:-0}" = 1 ] && [ "${EXPECT[$arm]}" = PASS ] && rec=(--record)
    python3 "$HERE/quick_bench.py" run --release --profile "$profile" --base "http://127.0.0.1:$PORT" --model "$model" \
      --log "$OUT/$arm.server.log" --out "$OUT/$arm.json" "${rec[@]}" | tee "$OUT/$arm.txt"
    if [ "${PIPESTATUS[0]}" -eq 0 ]; then GOT[$arm]=PASS; else GOT[$arm]=FAIL; fi
    # a bad arm that "fails" only because the references are not recorded proves nothing
    grep -q "references not recorded" "$OUT/$arm.txt" && GOT[$arm]=UNRECORDED
  else
    docker logs "$C" > "$OUT/$arm.server.log" 2>&1
    GOT[$arm]=NOSTART   # a start failure is never the expected verdict, not even for a bad arm
  fi
  docker rm -f "$C" >/dev/null 2>&1
  SECS[$arm]="start $((${T1:-$SECONDS} - T0))s + tier $((SECONDS - ${T1:-$SECONDS}))s"
  echo "--- $arm ${SECS[$arm]}"
done

echo; echo "arm       expected  got"
bad=0
for arm in $ARMS; do
  mark=ok; [ "${GOT[$arm]}" = "${EXPECT[$arm]}" ] || { mark="WRONG"; bad=1; }
  printf '%-9s %-9s %-8s %-6s %s\n' "$arm" "${EXPECT[$arm]}" "${GOT[$arm]}" "$mark" "${SECS[$arm]}"
  grep -E '^(FAIL|::error::)' "$OUT/$arm.txt" 2>/dev/null | sed 's/^/            /' | head -8
done
exit $bad
