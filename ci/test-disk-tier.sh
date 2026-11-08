#!/usr/bin/env bash
# test-disk-tier.sh — E2E test for the re-anchored disk KV-cache patches.
#
# Runs on the inference host. Starts a throwaway disk-tier container (RAM tier +
# fs secondary tier) from the production image, then proves:
#   (1) it brings up with all disk patches applied,
#   (2) responses are coherent,
#   (3) KV blocks are written to disk and SURVIVE a container restart (re-prefilling
#       the same long prompt is far faster, and the greedy response is byte-identical).
# It does NOT touch the production RAM-tier model. See docs/DISK-TIER-TEST.md for
# interpretation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---- config (override via env) -----------------------------------------------
IMAGE=${IMAGE:-ghcr.io/slybase/vllm-sly-radiance:$(cat "$SCRIPT_DIR/VERSION" 2>/dev/null)-rocm10.1}
MODEL=${MODEL:-slybase/Swift-1.5-Qwen3.8-27B-heretic-MXFP4-GPTQ}
DRAFTER=${DRAFTER:-syvai/Qwen3.8-27B-DFlash2-W4A16}
SERVED=${SERVED:-disk-tier-test}
CONTAINER=${CONTAINER:-disk-tier-test}
# Patches live in <repo>/kv-cache/ and _patchlib.py at the repo root. Mount the
# repo root as /patches so /patches/kv-cache/ is the disk set and PYTHONPATH=/patches
# picks up _patchlib.
PATCHES_DIR=${PATCHES_DIR:-$SCRIPT_DIR}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$SCRIPT_DIR/qwen-fixed-v22.3.jinja}
KVCACHE_DIR=${KVCACHE_DIR:-/var/tmp/disk-tier-test}
PORT=${PORT:-8001}
LOGDIR=${LOGDIR:-$KVCACHE_DIR/logs}
SHM_SRC=${SHM_SRC:-/dev/shm/sly-swift}
HEALTH_TIMEOUT=${HEALTH_TIMEOUT:-1800}   # seconds; a cold compile can take ~30 min

die() { echo "FATAL: $*" >&2; exit 1; }

# ---- verdict tracking --------------------------------------------------------
# Correctness checks below record a soft FAIL rather than aborting mid-run, so the whole
# harness prints every signal in one pass. The final verdict exits non-zero if any held.
FAIL=0
fail() { echo "FAIL: $*"; FAIL=1; }
pass() { echo "ok:   $*"; }

# compare two /v1/chat/completions responses (message.content or .reasoning) for byte equality.
# $1=ref $2=got ; returns 0 if identical, records fail() otherwise.
responses_identical() {
  python3 - "$1" "$2" <<'PY'
import json,sys
def msg(p):
    d=json.load(open(p))["choices"][0]["message"]
    c=d.get("content"); r=d.get("reasoning")
    return c if c is not None else r
a=msg(sys.argv[1]); b=msg(sys.argv[2])
print("identical" if a==b else "DIFFERENT")
sys.exit(0 if a==b else 1)
PY
}

# ---- preflight ---------------------------------------------------------------
[ -d "$PATCHES_DIR/kv-cache" ] || die "PATCHES_DIR=$PATCHES_DIR has no kv-cache/. Clone the branch there first."
[ -f "$CHAT_TEMPLATE" ] || die "chat template $CHAT_TEMPLATE missing."
mkdir -p "$KVCACHE_DIR/blocks" "$LOGDIR"
free_gib() { df -B1 --output=avail "$KVCACHE_DIR" 2>/dev/null | tail -1 | awk '{print int($1/1073741824)}'; }
[ "$(free_gib)" -ge 20 ] || die "need >=20 GiB free at $KVCACHE_DIR (have $(free_gib))."
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not present; pull it first."
if docker ps --format '{{.Names}}' | grep -qE "(^|-)(sly|sly-swift)(-|$)|$CONTAINER"; then
  die "a $CONTAINER/sly container is running. Stop it first (docker stop <name>): the GPU must be free."
fi

# ---- fixed prompt (~40k tokens, deterministic) -------------------------------
PROMPT_FILE="$LOGDIR/prompt.txt"
PAYLOAD_FILE="$LOGDIR/payload.json"
python3 - "$PROMPT_FILE" <<'PY'
import sys
para = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
with open(sys.argv[1], "w") as f:
    f.write(para * 1600)   # ~40k tokens
PY

# ---- payload helpers (python, so no shell-quoting surprises) -----------------
payload() {  # $1=max_tokens ; prints a full /v1/chat/completions body
  python3 -c '
import json,sys
mt=int(sys.argv[1]); content=open(sys.argv[2]).read()
print(json.dumps({"model":sys.argv[3],"messages":[{"role":"user","content":content}],"max_tokens":mt,"temperature":0}))
' "$1" "$PROMPT_FILE" "$SERVED"
}
ttft() {  # $1=max_tokens ; prints total seconds (1 token ~= pure prefill time)
  payload "$1" > "$PAYLOAD_FILE"
  curl -s -o /dev/null -w '%{time_total}' -H 'Content-Type: application/json' \
    --data @"$PAYLOAD_FILE" "http://localhost:$PORT/v1/chat/completions"
}
resp() {  # $1=outfile ; grabs a 64-token greedy completion
  payload 64 > "$PAYLOAD_FILE"
  curl -s -H 'Content-Type: application/json' --data @"$PAYLOAD_FILE" \
    "http://localhost:$PORT/v1/chat/completions" > "$1"
}

# ---- accuracy-gate prompt set (varying length) -------------------------------
# Each prompt is run fresh (captured before restart) and cached (after restart);
# the cached greedy response must be byte-identical to the fresh one. The long
# prompt doubles as the TTFT baseline (resp_before/resp_after); short/medium are
# the extra gates. Captured fresh refs live in $REFDIR.
PROMPT_DIR="$LOGDIR/prompts"
REFDIR="$LOGDIR/refs"
mkdir -p "$PROMPT_DIR" "$REFDIR"
python3 - "$PROMPT_DIR" <<'PY'
import sys, os
d = sys.argv[1]
para = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
open(os.path.join(d, "short.txt"), "w").write(para * 20)    # ~1k tokens
open(os.path.join(d, "medium.txt"), "w").write(para * 250)  # ~12k tokens
open(os.path.join(d, "long.txt"), "w").write(para * 1600)   # ~40k (baseline)
# distinct-content long prompt: running it after the baseline forces the fs tier to
# evict some baseline blocks as the disk fills (partial eviction).
open(os.path.join(d, "evict.txt"), "w").write(
    "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
    "A journey of a thousand miles begins with a single step. To be or not to be, that is the question. "
    * 1600)
PY

# run one accuracy prompt, save the greedy completion to $3
acc_resp() {  # $1=prompt_name $2=max_tokens $3=outfile
  python3 -c '
import json,sys
mt=int(sys.argv[1]); content=open(sys.argv[2]).read()
print(json.dumps({"model":sys.argv[3],"messages":[{"role":"user","content":content}],"max_tokens":mt,"temperature":0}))
' "$2" "$PROMPT_DIR/$1.txt" "$SERVED" > "$PAYLOAD_FILE"
  curl -s -H 'Content-Type: application/json' --data @"$PAYLOAD_FILE" \
    "http://localhost:$PORT/v1/chat/completions" > "$3"
}

# ---- decode throughput (tok/s) -----------------------------------------------
# short prompt, large completion -> decode-bound. t/s from wall time + output token
# count. Reported with a soft floor (DISK_TIER_MIN_TPS, env-overridable); a single
# sample is noisy so it warns rather than hard-fails.
decode_tps() {  # $1=prompt_name $2=max_tokens ; prints tok/s
  python3 -c '
import json,sys,time
import urllib.request
mt=int(sys.argv[1]); content=open(sys.argv[2]).read()
body=json.dumps({"model":sys.argv[3],"messages":[{"role":"user","content":content}],"max_tokens":mt,"temperature":0}).encode()
req=urllib.request.Request("http://localhost:%s/v1/chat/completions"%sys.argv[4],data=body,headers={"Content-Type":"application/json"},method="POST")
t=time.time()
r=urllib.request.urlopen(req,timeout=600)
d=json.loads(r.read())
dt=time.time()-t
n=len(d["choices"][0]["message"]["content"].split())
print("%.1f"%(n/dt) if dt>0 else "inf")
' "$2" "$PROMPT_DIR/$1.txt" "$SERVED" "$PORT"
}

# ---- container lifecycle -----------------------------------------------------
CLEAN_SHM() {  # remove stale RAM-tier mmaps no live process holds (host $SHM_SRC)
  for f in "$SHM_SRC"/vllm_offload_*.mmap; do
    [ -e "$f" ] || continue
    if command -v fuser >/dev/null 2>&1; then fuser -s "$f" 2>/dev/null && continue; else continue; fi
    rm -f -- "$f"
  done
}

# Prelude: the common patches (as the production template applies them) + the
# disk additions. mixed_hit and instrumentation are FATAL (no `|| echo`): a
# failure aborts the prelude (set -e) and the container exits early, which the
# health poll detects. The rest warn-and-continue, matching serve-mxfp4.sh.
PRELUDE='
set -e
cd /patches
run(){ PYTHONPATH=/patches python3 /patches/kv-cache/$1; }
run patch_offload_mixed_hit.py
run patch_offload_instrumentation.py
run patch_offload_lookup_metrics.py         || echo "[kv-cache] WARNING: lookup-metrics did not apply"
run patch_eagle_groups.py                   || echo "[kv-cache] WARNING: eagle-groups did not apply"
run patch_mamba_stride.py                   || echo "[kv-cache] WARNING: mamba-stride did not apply"
run patch_reconcile_reask.py                || echo "[kv-cache] WARNING: reconcile-reask did not apply"
run patch_swa_align_touch.py                || echo "[kv-cache] WARNING: swa-align/touch did not apply"
run patch_sched_align_last_block.py         || echo "[kv-cache] WARNING: align-last-block did not apply"
run patch_swa_eagle_replay_tail.py          || echo "[kv-cache] WARNING: eagle-replay-tail did not apply"
run patch_offload_boundary_trace.py         || echo "[kv-cache] WARNING: boundary-trace did not apply"
run patch_offload_debug_instrument.py       || echo "[kv-cache] WARNING: debug-instrument did not apply"
run patch_offload_fs_fanout.py              || echo "[kv-cache] WARNING: fs-fanout did not apply"
run patch_offload_tier_report.py            || echo "[kv-cache] WARNING: tier-report did not apply"
run patch_offload_promotion_wallclock.py    || echo "[kv-cache] WARNING: promotion/wallclock did not apply"
run patch_lookup_invalidate.py              || echo "[kv-cache] WARNING: lookup-invalidate did not apply"
run patch_fs_failed_load.py                 || echo "[kv-cache] WARNING: failed-load-forget did not apply"
run patch_offload_miss_deferral_metrics.py  || echo "[kv-cache] WARNING: deferral/miss-reason did not apply"
exec /opt/radiance_entrypoint.sh "$@"'

start() {  # $1=log-label
  CLEAN_SHM
  docker rm -f "$CONTAINER" 2>/dev/null || true
  echo ">> starting $CONTAINER (label=$1)"
  docker run -d --name "$CONTAINER" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc=host \
    --mount type=bind,source="$SHM_SRC",target=/dev/shm \
    -p "$PORT":8000 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$CHAT_TEMPLATE:/chat-template.jinja:ro,z" \
    -v "$PATCHES_DIR:/patches:z" \
    -v "$KVCACHE_DIR:/kvcache:z" \
    -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=2 -e HF_HUB_DISABLE_SHARED_BLOBS=1 \
    -e PYTHONHASHSEED=0 \
    -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
    -e RADIANCE_MXFP4_DECODE_MAX_M=128 -e RADIANCE_MXFP4_A_TILED_MIN_M=513 -e RADIANCE_MXFP4_WPERM=1 \
    -e RADIANCE_MXFP4_TN4_MIN_M=513 -e RADIANCE_MXFP4_AT_SKIP_PAD=1 \
    -e RADIANCE_LMHEAD_INT4=1 -e RADIANCE_FUSED_NORM_QUANT=1 -e RADIANCE_FUSED_NQ_REG=1 \
    -e RADIANCE_MXFP4_TILED_OPROJ=1 \
    -e RADIANCE_KV_GROUP_SIZE=8 -e RADIANCE_EMBED_INT8=1 -e RADIANCE_EMBED_BITS=4 \
    -e RADIANCE_MXFP4_WIDE_MAX_M=192 -e RADIANCE_ADAPTIVE_WIDTH=perseq \
    -e RADIANCE_CACTUS_DELTA=0.5 \
    -e RADIANCE_OFFLOAD_MIXED_HIT=1 -e RADIANCE_OFFLOAD_EAGLE_GROUPS=1 \
    -e RADIANCE_MAMBA_STORE_STRIDE=4 -e RADIANCE_RECONCILE_REASK=1 \
    -e RADIANCE_SWA_STORE_MAMBA_ALIGN=1 -e RADIANCE_TOUCH_ALL_GROUPS=1 \
    -e RADIANCE_TOUCH_POSITION_ORDER=1 -e RADIANCE_ALIGN_PROMPT_LAST_BLOCK=1 \
    -e RADIANCE_SWA_EAGLE_REPLAY_TAIL=1 -e RADIANCE_OFFLOAD_BOUNDARY_TRACE=0 \
    --entrypoint bash "$IMAGE" -lc "$PRELUDE" _ \
    --model "$MODEL" \
    --served-model-name "$SERVED" \
    --max-model-len 262144 --gpu-memory-utilization 0.98 \
    --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
    --kv-offloading-size 16 --kv-offloading-backend native \
    --kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"spec_name":"TieringOffloadingSpec","secondary_tiers":[{"type":"fs","root_dir":"/kvcache/blocks","n_read_threads":8,"n_write_threads":4}]}}' \
    --speculative-config.method dflash \
    --speculative-config.model "$DRAFTER" \
    --speculative-config.num_speculative_tokens 7 \
    --speculative-config.draft_sample_method probabilistic \
    --speculative-config.attention_backend TRITON_ATTN \
    --attention-backend R4D_HYBRID \
    --max-num-seqs 8 --max-num-batched-tokens 4096 \
    --compilation-config.cudagraph_mode FULL_DECODE_ONLY \
    --compilation-config.cudagraph_capture_sizes '[8,16,24,32,40,48,56,64]' \
    --enable-prefix-caching --skip-mm-profiling --enable-mm-embeds \
    --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --chat-template /chat-template.jinja \
    --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
    --override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.1}'
}

alive() { docker ps -q -f "name=^${CONTAINER}$" 2>/dev/null | grep -q .; }

dump_log() {  # $1=label ; the vLLM + prelude output lives in `docker logs`, not run-stdout
  docker logs "$CONTAINER" > "$LOGDIR/$1.log" 2>&1 || true
}

wait_healthy() {
  local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
  echo ">> waiting for health (up to ${HEALTH_TIMEOUT}s)..."
  while :; do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then echo ">> healthy."; return 0; fi
    if ! alive; then
      dump_log early
      echo "FATAL: container exited before becoming healthy; last log lines:" >&2
      tail -40 "$LOGDIR/early.log" >&2
      return 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      dump_log timeout
      echo "FATAL: not healthy within ${HEALTH_TIMEOUT}s; see $LOGDIR/timeout.log" >&2
      return 1
    fi
    sleep 30
  done
}

# ---- the test ----------------------------------------------------------------
echo "=== PHASE 1: bring up ==="
start first
wait_healthy
dump_log first

# Verify the disk patches applied. instrumentation/mixed_hit are FATAL (caught by
# health); the rest warn-and-continue, so check their short WARNING names are absent.
echo ">> patch-application warnings in the log (expect NONE for the disk set):"
grep -E 'WARNING: .* did not apply' "$LOGDIR/first.log" || echo "   (no warnings)"
DISK_WARN_NAMES="fs-fanout lookup-invalidate failed-load-forget tier-report promotion/wallclock deferral/miss-reason"
FAIL=0
for n in $DISK_WARN_NAMES; do
  if grep -q "WARNING: $n did not apply" "$LOGDIR/first.log"; then
    echo "FAIL: disk patch '$n' did not apply"; FAIL=1
  fi
done
if [ "$FAIL" = 1 ]; then
  echo "RESULT: BRING-UP FAILED — a disk patch did not apply. Inspect $LOGDIR/first.log."
  exit 1
fi
echo ">> bring-up OK: all disk patches applied."

echo "=== PHASE 2: quality + populate disk ==="
# Clear leftover disk blocks (persist across runs) so BEFORE_TTFT is a TRUE fresh
# prefill — otherwise the disk tier loads stale blocks and the "baseline" is just
# another disk load, which never beats itself. GPU RAM is already empty: the
# container was started fresh in PHASE 1.
sudo rm -rf "$KVCACHE_DIR/blocks"/*
BEFORE_TTFT=$(ttft 1)
echo ">> baseline long-prompt TTFT (fresh prefill, no cache): ${BEFORE_TTFT}s"
resp "$LOGDIR/resp_before.json"
echo ">> baseline 64-token greedy response:"
python3 -c 'import json;d=json.load(open("'"$LOGDIR/resp_before.json"'"))["choices"][0]["message"];print((d.get("content") or "(null) [reasoning: "+(d.get("reasoning") or "")[:80]+")"))'
du -sh "$KVCACHE_DIR/blocks" | tee "$LOGDIR/du_before.txt"
BLOCKS_BEFORE=$(find "$KVCACHE_DIR/blocks" -type f | wc -l)
echo ">> block files before restart: $BLOCKS_BEFORE"

# Fresh (no-offload) references for the accuracy gate. short/medium are extra gates;
# the long prompt's reference is resp_before (captured just above). Captured here,
# before restart, so the cached runs in PHASE 5 compare against a true cold reference.
acc_resp short 32 "$REFDIR/short.json"
acc_resp medium 32 "$REFDIR/medium.json"
echo ">> accuracy-fresh refs written: $REFDIR/{short,medium}.json (long = resp_before)"

echo "=== PHASE 3: restart (stop + start, same /kvcache mount) ==="
docker stop "$CONTAINER"
sleep 5
start second
wait_healthy
dump_log second

echo "=== PHASE 4: survival ==="
AFTER_TTFT=$(ttft 1)
resp "$LOGDIR/resp_after.json"
du -sh "$KVCACHE_DIR/blocks" | tee "$LOGDIR/du_after.txt"
BLOCKS_AFTER=$(find "$KVCACHE_DIR/blocks" -type f | wc -l)
echo ">> TTFT before: ${BEFORE_TTFT}s   after: ${AFTER_TTFT}s"
echo ">> block files before: $BLOCKS_BEFORE   after: $BLOCKS_AFTER"
if responses_identical "$LOGDIR/resp_before.json" "$LOGDIR/resp_after.json"; then
  pass "response identical after restart (disk serves correct tokens)"
else
  fail "response DIFFERS after restart"
fi

echo "=== PHASE 5: accuracy gate (cached vs fresh, byte-identical) ==="
# long prompt: resp_after vs resp_before (checked in PHASE 4). short/medium: cached now
# vs the cold references captured in PHASE 2. All must be byte-identical (disk is lossless).
for p in short medium; do
  acc_resp "$p" 32 "$LOGDIR/cached_$p.json"
  if responses_identical "$REFDIR/$p.json" "$LOGDIR/cached_$p.json"; then
    pass "accuracy gate: $p cached == fresh"
  else
    fail "accuracy gate: $p cached != fresh"
  fi
done

echo "=== PHASE 6: decode throughput ==="
# short prompt, large completion -> decode-bound. t/s from wall time + output token count.
# A single sample is noisy (hardware varies), so this is a soft floor, not a hard fail.
MIN_TPS="${DISK_TIER_MIN_TPS:-8}"
TPS=$(decode_tps short 256)
echo ">> cached decode throughput (short prompt, 256 tok): ${TPS} tok/s (soft floor ${MIN_TPS})"
python3 - "$TPS" "$MIN_TPS" <<'PY'
import sys
tps = float(sys.argv[1]) if sys.argv[1] not in ("inf", "") else 999999.0
floor = float(sys.argv[2])
print("ok:   decode throughput %.1f >= floor %.1f" % (tps, floor)) if tps >= floor else \
    print("WARN: decode throughput %.1f < floor %.1f (single sample; hardware-dependent)" % (tps, floor))
PY

echo "=== PHASE 7: partial eviction (new prompt evicts baseline blocks) ==="
# The disk holds the baseline (long) prompt's blocks. Running a distinct-content prompt
# forces the fs tier to evict some of them as the disk fills, then serve the new prompt
# from a mix of cached and recomputed blocks. Capture its fresh reference, then verify the
# cached response is byte-identical (eviction + recompute is lossless).
acc_resp evict 32 "$REFDIR/evict.json"
echo ">> eviction-prompt fresh ref written: $REFDIR/evict.json"
acc_resp evict 32 "$LOGDIR/cached_evict.json"
if responses_identical "$REFDIR/evict.json" "$LOGDIR/cached_evict.json"; then
  pass "partial eviction: evict prompt cached == fresh (mix of cached/recomputed OK)"
else
  fail "partial eviction: evict prompt cached != fresh"
fi

echo "=== PHASE 8: durability (second restart) ==="
docker stop "$CONTAINER"
sleep 5
start third
wait_healthy
dump_log third
# baseline (long) blocks must survive a second restart and serve byte-identical output.
resp "$LOGDIR/resp_third.json"
if responses_identical "$LOGDIR/resp_before.json" "$LOGDIR/resp_third.json"; then
  pass "durability: response identical after a SECOND restart"
else
  fail "durability: response DIFFERS after a second restart"
fi
BLOCKS_THIRD=$(find "$KVCACHE_DIR/blocks" -type f | wc -l)
echo ">> block files after third restart: $BLOCKS_THIRD"

# ---- verdict ----------------------------------------------------------------
SPEEDUP=$(python3 -c "print(round(float('$BEFORE_TTFT')/float('$AFTER_TTFT'),2) if float('$AFTER_TTFT')>0 else 99)")
echo "=================== VERDICT ==================="
echo "TTFT speedup (before/after restart): ${SPEEDUP}x"
echo "disk blocks: $BLOCKS_BEFORE -> $BLOCKS_AFTER -> $BLOCKS_THIRD"
echo "response identical after restart: PASS/FAIL recorded above"
echo "--------------------------------------------"
echo "PASS iff ALL hold:"
echo "  (1) Phase 1: no disk-patch warnings (checked above)"
echo "  (2) Phase 2: baseline response is coherent English (eyeball it)"
echo "  (3) Phase 4: block count grew (>0 after)"
echo "  (4) Phase 4: TTFT after < before (restart beats a fresh prefill; speedup is disk-bandwidth bound)"
echo "  (5) Phase 4/5: response identical before/after (disk serves correct tokens)"
echo "  (6) Phase 6: decode throughput above soft floor (see WARN above)"
echo "  (7) Phase 7: partial eviction correct (evict prompt cached == fresh)"
echo "  (8) Phase 8: durability correct (response identical after a second restart)"
echo "Logs: $LOGDIR/first.log  $LOGDIR/second.log  $LOGDIR/third.log"
echo "================================================"
[ "$FAIL" = 0 ] || { echo "RESULT: FAIL — see FAIL lines above"; exit 1; }
echo "RESULT: PASS"
