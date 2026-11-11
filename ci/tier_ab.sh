#!/usr/bin/env bash
# ci/tier_ab.sh - RAM-only ARC-vs-LRU A/B for the KV eviction policy.
#
# Runs ON the GPU host inside an acquired gpu-window with production stopped. The GPU KV pool
# is SHRUNK (low --gpu-memory-utilization + --max-model-len) so the reused main offloads to the
# RAM tier, and the main is REPEATED (rounds) so ARC's frequent-set promotion has >=2 accesses
# to act on. L and A differ ONLY in KVCACHE_EVICT_POLICY; the comparison is L vs A within one
# window. C / C2 are the production control (no tier) at the start and end (drift check).
#
# Unlike ci/prefill_ab.sh (unique-prompt BetterBench sweep, which never touches the tier), the
# workload here is a repeated main prompt + bursts of distinct subagent prompts, and the verdict
# is read from the engine's native /metrics counters (no disk instrumentation in ram mode):
#   prompt_tokens_by_source{source=local_compute|local_cache_hit|external_kv_transfer}
#   kv_offload_tiering_chunk_hits{tier="0:primary"} / kv_offload_tiering_chunk_queries
#   per-resume main TTFT (wall time of each main re-send)
#
# The launch is the disk-tier harness's proven production arg set (ci/test-disk-tier.sh start()),
# in RAM mode: the same --kv-offloading-backend native + --kv-transfer-config (OffloadingConnector)
# pair disk mode always used, but WITHOUT the fs secondary tier (kv-cache/tools/tier-config.py adds
# it for disk only) and WITHOUT the disk-only patches/mount. The config string comes from that same
# helper serve-mxfp4.sh calls, so the launch exercises the launcher's code path.
#
# Arms (default order C L A C2; C2 = control again, drift check):
#   C   control, production (no tier, MAXLEN=262144 GPU_UTIL=0.98)
#   L   KVCACHE=ram + small GPU KV pool + KVCACHE_EVICT_POLICY=lru
#   A   KVCACHE=ram + small GPU KV pool + KVCACHE_EVICT_POLICY=arc
#   C2  control again
#
# Output in $OUT: <arm>.kvlog, <arm>.cmd (launch line), <arm>.metrics.before/after,
# <arm>.metrics.diff, <arm>.workload, <arm>.server.log, hw.csv (power/temp/fan every 15 s),
# summary.tsv, status.txt, DONE.
#
# Usage:
#   SMOKE=1 OUT=/root/tier-ab-$(date +%Y%m%d) ci/tier_ab.sh          # Test 1 smoke (ram boot + reload)
#   OUT=/root/tier-ab-$(date +%Y%m%d) ci/tier_ab.sh                  # full A/B (C L A C2)
#   ARMS="L A" RAM_GIB=1 MAXLEN=24576 GPU_UTIL=0.60 OUT=... ci/tier_ab.sh   # subset + re-size
#
# The script never touches the power cap; it records it (the A/B window is 300 W, restore 210 W
# after -- done by the human before/after, per AGENTS.md).
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE=${IMAGE:-ghcr.io/slybase/vllm-sly-radiance:$(cat "$SCRIPT_DIR/VERSION" 2>/dev/null)-rocm10.1}
MODEL=${MODEL:-slybase/Swift-1.5-Qwen3.8-27B-heretic-MXFP4-GPTQ}
DRAFTER=${DRAFTER:-syvai/Qwen3.8-27B-DFlash2-W4A16}
SERVED=${SERVED:-tier-ab}
CONTAINER=${CONTAINER:-tier-ab}
PATCHES_DIR=${PATCHES_DIR:-$SCRIPT_DIR}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-$SCRIPT_DIR/qwen-fixed-v22.3.jinja}
PORT=${PORT:-8001}
OUT=${OUT:-/root/tier-ab-$(date +%Y%m%d)}
NAME=${NAME:-tier-ab}
SMOKE=${SMOKE:-0}

# RAM-mode sizing (plan defaults; iterate after reading diagnostics, do not guess).
RAM_GIB=${RAM_GIB:-1}                 # RAM tier GiB (~40 KB/token -> RAM_GIB*1024/0.04 tokens)
MAXLEN=${MAXLEN:-24576}               # floor on the GPU KV pool (vLLM: pool >= max_model_len)
GPU_UTIL=${GPU_UTIL:-0.60}            # pool_GiB ~= util*32 - 17.3 (weights 14.79 + graph + init)
R=${R:-3}                             # flush rounds (each: big burst -> main), the discriminating samples
NSUB=${NSUB:-4}                       # distinct subagent prompts per FLUSH burst (sent CONCURRENTLY; gap volume must exceed the tier)
NSUB_FIRST=${NSUB_FIRST:-2}           # promote-phase subagents (small so main+burst < tier -> main reload hits -> T1->T2)
MAIN_TOK=${MAIN_TOK:-20000}           # main prompt target tokens (~25 tok/repetition)
SUB_TOK=${SUB_TOK:-10000}             # each subagent prompt target tokens
RETENTION=${RETENTION:-0}             # --prefix-cache-retention-interval: 0=boundary only (SSM not restorable mid-sequence -> reload recomputes -> ext_kv=0); >0=periodic Mamba checkpoints (multiple of the attention block size 880) so a mid-sequence reload CAN restore -> ext_kv>0 and the policy split becomes measurable
STARTUP_TIMEOUT_S=${STARTUP_TIMEOUT_S:-1800}
SHM_SRC=${SHM_SRC:-/dev/shm/sly-swift}   # host dir bind-mounted to the container's /dev/shm (the RAM tier lands here)

if [ "$SMOKE" = 1 ]; then ARMS=${ARMS:-L}; else ARMS=${ARMS:-C L A C2}; fi

mkdir -p "$OUT"; cd "$OUT" || exit 1; OUT=$(pwd)
log() { echo "$(date -u +%T) $*" | tee -a run.log; echo "$*" > status.txt; }

die() { echo "FATAL: $*" >&2; exit 1; }

# ---- preflight ---------------------------------------------------------------
[ -d "$PATCHES_DIR/kv-cache" ] || die "PATCHES_DIR=$PATCHES_DIR has no kv-cache/. Clone the branch there first."
[ -f "$CHAT_TEMPLATE" ] || die "chat template $CHAT_TEMPLATE missing."
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not present; pull it first."
running=$(docker ps --format '{{.Names}}')
if grep -qE "(^|-)(sly|sly-swift)(-|$)" <<<"$running" || grep -qF -- "$CONTAINER" <<<"$running"; then
  die "a $CONTAINER/sly container is running. Stop it first: the GPU must be free."
fi

# ---- read-only hardware facts + power/temp/fan sampler -----------------------
HWMON=
for h in /sys/class/drm/card*/device/hwmon/hwmon*; do
  [ -r "$h/power1_cap" ] && HWMON=$h
done
[ -n "$HWMON" ] || die "no GPU hwmon found"
CAP_W=$(( $(cat "$HWMON/power1_cap") / 1000000 ))
log "power cap ${CAP_W} W (read only; not changed by this script)"

CLEAN_SHM() {  # remove stale RAM-tier mmaps no live process holds (host $SHM_SRC, fuser/lsof decide)
  for f in "$SHM_SRC"/vllm_offload_*.mmap "$SHM_SRC"/sem.mp-*; do
    [ -e "$f" ] || continue
    if   command -v fuser >/dev/null 2>&1; then fuser -s "$f" 2>/dev/null && continue
    elif command -v lsof  >/dev/null 2>&1; then lsof -t -- "$f" >/dev/null 2>&1 && continue
    else continue; fi
    rm -f -- "$f"
  done
}

cleanup() {
  kill "${SPID:-}" "${LOGFOLLOWER:-}" 2>/dev/null
  docker rm -f "$CONTAINER" >/dev/null 2>&1
  CLEAN_SHM
  touch "$OUT/DONE"
}
trap cleanup EXIT
sample() {
  while :; do
    echo "$(date -u +%T),$(cat "$HWMON/power1_average"),$(cat "$HWMON"/temp[123]_input 2>/dev/null | tr '\n' ','),$(cat "$HWMON/fan1_input" 2>/dev/null)" >> hw.csv
    sleep 15
  done
}

# ---- ram-mode --kv-transfer-config (the launcher's own helper) ---------------
kv_transfer_config() {  # $1=policy -> the JSON serve-mxfp4.sh emits for KVCACHE=ram
  python3 "$PATCHES_DIR/kv-cache/tools/tier-config.py" ram "$1"
}

# ---- launch (harness production arg set, RAM mode) ---------------------------
# PRELUDE differs: control arms run the plain entrypoint (production, no tier); ram arms
# apply the common offload patch set serve-mxfp4.sh adds for KVCACHE=ram (mixed_hit FATAL,
# the rest warn-and-continue). No disk set, no instrumentation/lookup-metrics (disk-only).
PRELUDE_RAM='
set -e
cd /patches
run(){ PYTHONPATH=/patches python3 /patches/kv-cache/$1; }
run patch_offload_mixed_hit.py
run patch_eagle_groups.py            || echo "[kv-cache] WARNING: eagle-groups did not apply"
run patch_mamba_stride.py            || echo "[kv-cache] WARNING: mamba-stride did not apply"
run patch_reconcile_reask.py         || echo "[kv-cache] WARNING: reconcile-reask did not apply"
run patch_swa_align_touch.py         || echo "[kv-cache] WARNING: swa-align/touch did not apply"
run patch_sched_align_last_block.py  || echo "[kv-cache] WARNING: align-last-block did not apply"
run patch_swa_eagle_replay_tail.py   || echo "[kv-cache] WARNING: eagle-replay-tail did not apply"
run patch_offload_boundary_trace.py  || echo "[kv-cache] WARNING: boundary-trace did not apply"
exec /opt/radiance_entrypoint.sh "$@"'
PRELUDE_OFF='
exec /opt/radiance_entrypoint.sh "$@"'

start_arm() {  # $1=arm -> container healthy; builds the launch line into $OUT/<arm>.cmd
  local arm=$1 tier=() env_off=() prelude maxlen gpuutil policy cfg
  case $arm in
    C|C2) maxlen=262144; gpuutil=0.98; prelude=$PRELUDE_OFF ;;   # production, no tier
    L|A)  maxlen=$MAXLEN; gpuutil=$GPU_UTIL
          policy=$([ "$arm" = L ] && echo lru || echo arc)
          prelude=$PRELUDE_RAM
          cfg=$(kv_transfer_config "$policy") || die "kv_transfer_config($policy) failed: tier-config.py dead"
          tier=(--kv-offloading-size "$RAM_GIB" --kv-offloading-backend native --kv-transfer-config "$cfg")
          ret=(); [ "$RETENTION" -gt 0 ] && ret=(--prefix-cache-retention-interval "$RETENTION")
          env_off=(-e RADIANCE_OFFLOAD_MIXED_HIT=1 -e RADIANCE_OFFLOAD_EAGLE_GROUPS=1
                   -e RADIANCE_MAMBA_STORE_STRIDE=4 -e RADIANCE_RECONCILE_REASK=1
                   -e RADIANCE_SWA_STORE_MAMBA_ALIGN=1 -e RADIANCE_TOUCH_ALL_GROUPS=1
                   -e RADIANCE_TOUCH_POSITION_ORDER=1 -e RADIANCE_ALIGN_PROMPT_LAST_BLOCK=1
                   -e RADIANCE_SWA_EAGLE_REPLAY_TAIL=1 -e RADIANCE_OFFLOAD_BOUNDARY_TRACE=0) ;;
    *) die "unknown arm $arm" ;;
  esac
  CLEAN_SHM
  docker rm -f "$CONTAINER" 2>/dev/null || true
  docker run -d --name "$CONTAINER" \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt seccomp=unconfined --ipc=host \
    --mount type=bind,source="$SHM_SRC",target=/dev/shm \
    -p "$PORT":8000 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$CHAT_TEMPLATE:/chat-template.jinja:ro,z" \
    -v "$PATCHES_DIR:/patches:z" \
    -e HIP_VISIBLE_DEVICES=0 -e GPU_MAX_HW_QUEUES=2 -e HF_HUB_DISABLE_SHARED_BLOBS=1 \
    -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
    -e RADIANCE_MXFP4_DECODE_MAX_M=128 -e RADIANCE_MXFP4_A_TILED_MIN_M=513 -e RADIANCE_MXFP4_WPERM=1 \
    -e RADIANCE_MXFP4_TN4_MIN_M=513 -e RADIANCE_MXFP4_AT_SKIP_PAD=1 \
    -e RADIANCE_LMHEAD_INT4=1 -e RADIANCE_FUSED_NORM_QUANT=1 -e RADIANCE_FUSED_NQ_REG=1 \
    -e RADIANCE_MXFP4_TILED_OPROJ=1 \
    -e RADIANCE_KV_GROUP_SIZE=8 -e RADIANCE_EMBED_INT8=1 -e RADIANCE_EMBED_BITS=4 \
    -e RADIANCE_MXFP4_WIDE_MAX_M=192 -e RADIANCE_ADAPTIVE_WIDTH=perseq \
    -e RADIANCE_CACTUS_DELTA=0.5 \
    ${env_off[@]+"${env_off[@]}"} \
    --entrypoint bash "$IMAGE" -lc "$prelude" _ \
    --model "$MODEL" \
    --served-model-name "$SERVED" \
    --max-model-len "$maxlen" --gpu-memory-utilization "$gpuutil" \
    --kv-cache-dtype fp8 --mamba-ssm-cache-dtype bfloat16 \
    ${tier[@]+"${tier[@]}"} \
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
     ${ret[@]+"${ret[@]}"} \
     --limit-mm-per-prompt.image 0 --limit-mm-per-prompt.video 0 \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --chat-template /chat-template.jinja \
    --default-chat-template-kwargs '{"reasoning_effort": "medium"}' \
    --override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.1}' \
    > "$arm.cmd" 2>&1 || { log "$arm: docker run failed"; return 1; }
  kill "${LOGFOLLOWER:-}" 2>/dev/null   # reap the previous start's follower (it holds an fd on this log)
  docker logs -f "$CONTAINER" > "$OUT/$arm.server.log" 2>&1 & LOGFOLLOWER=$!
  if ! wait_healthy; then
    log "$arm: container died or no /health (see $arm.server.log)"; return 1
  fi
}

wait_healthy() {
  local t0=$SECONDS
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = true ] || return 1
    [ $((SECONDS - t0)) -gt "$STARTUP_TIMEOUT_S" ] && return 1
    sleep 5
  done
}

kv_line() { grep -m1 "GPU KV cache size" "$OUT/$1.server.log" 2>/dev/null | sed 's/.*GPU KV cache size/GPU KV cache size/'; }

# ---- /metrics scrape (native 0.32.0 counters) --------------------------------
scrape_metrics() {  # $1=file ; dumps the raw counter lines (no HELP/TYPE/_created gauges)
  curl -sf "http://127.0.0.1:$PORT/metrics" 2>/dev/null | grep -v '^#' | grep -v '_created' | grep -E \
    'prompt_tokens_by_source|kv_offload_tiering_chunk_(hits|queries)|external_prefix_cache_hits_total' \
    > "$1"
}
metrics_diff() {  # $1=before $2=after -> prints the deltas of the source + chunk counters
  python3 - "$1" "$2" <<'PY'
import sys
def load(p):
    d={}
    for ln in open(p):
        ln=ln.strip()
        if not ln: continue
        parts=ln.rsplit(' ',1)
        if len(parts)!=2: continue
        try: d[parts[0]]=float(parts[1])
        except ValueError: pass
    return d
b,a=load(sys.argv[1]),load(sys.argv[2])
keys=sorted(set(b)|set(a))
out=[]
for k in keys:
    d=a.get(k,0.0)-b.get(k,0.0)
    if any(t in k for t in ('local_compute','local_cache_hit','external_kv_transfer','chunk_hits','chunk_queries','external_prefix_cache_hits_total')):
        out.append((k,d))
for k,d in out: print(f"{k}  delta={d:+.0f}")
PY
}

# ---- promote-then-flush workload (main + subagent bursts), per arm -----------
# arc.py: a store is NOT a frequency hit (insertion_only_keys stays in T1); T2 promotion needs a
# genuine reload hit (reused_keys); evict() evicts T1 first while target_t1_size is 0. So the plain
# "main -> burst evicts main -> main" schedule never reaches the protected T2 and LRU==ARC (null).
# Discriminating schedule:
#   main(cold) -> small burst (main evicted from GPU -> stored T1; main+burst < tier so main stays)
#   -> main (reload HIT -> promoted T1->T2, main now "frequent")
#   -> R x (big distinct burst whose cumulative volume ALONE exceeds the tier, no main touch in the
#      gap -> LRU evicts main; ARC keeps it in T2) -> main (ARC reload hit vs LRU recompute).
# warm-up pass (discard), snapshot, measured pass, snapshot. Records per-resume main TTFT.
workload() {  # $1=arm
  python3 - "$PORT" "$SERVED" "$MAIN_TOK" "$SUB_TOK" "$R" "$NSUB" "$NSUB_FIRST" "$OUT/$1.workload" > "$OUT/$1.workload.out" 2>&1 <<'PY'
import json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
port, served = sys.argv[1], sys.argv[2]
main_tok, sub_tok, R, NSUB, NSUB_FIRST = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])
para = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. "
def para_n(t): return max(1, int(t/25))   # ~25 tokens/repetition (matches the harness fixture)
main = para * para_n(main_tok)
def send(content, mt=1):
    body = json.dumps({"model": served, "messages": [{"role": "user", "content": content}],
                       "max_tokens": mt, "temperature": 0}).encode()
    t = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", body,
        {"Content-Type": "application/json"}), timeout=600)
    json.loads(r.read())
    return time.time() - t
def burst(tag, n):  # n distinct one-shot subagent prompts for this burst, sent CONCURRENTLY
    subs = [("A journey of a thousand miles begins with a single step. To be or not to be, that is the question. "
             f" Burst {tag} variant {i}. " + para) * para_n(sub_tok) for i in range(n)]
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(send, subs))
print("== warm-up pass (discard: main, then one burst) ==")
send(main); burst("warm", NSUB)
print("== measured pass (promote-then-flush) ==")
tt = send(main)
print(f"main resume0 (cold) ttft={tt:.2f}s")
burst("promote", NSUB_FIRST)   # evict main from GPU -> store T1; keep main+burst < tier so main survives
tt = send(main)
print(f"main promote-hit ttft={tt:.2f}s (expect reload -> T2 promotion)")
for r in range(1, R+1):
    burst(f"flush{r}", NSUB)   # cumulative gap volume > tier -> LRU evicts main, ARC keeps T2
    tt = send(main)
    print(f"main flush{r} ttft={tt:.2f}s (ARC: reload hit; LRU: recompute)")
PY
}

run_arm() {  # $1=arm
  local arm=$1 starts=2 n kv
  case $arm in C|C2) starts=1 ;; esac          # control: production compile cache is warm; its kvlog line is the fresh-compile pool, NOT the 436,906 production pool - never quote it as such
  [ "$SMOKE" = 1 ] && starts=1                 # smoke: one boot is enough to prove the combo
  for n in $(seq 1 "$starts"); do
    log "$arm: start $n/$starts"
    start_arm "$arm" || return 1
    kv=$(kv_line "$arm"); log "$arm: healthy, ${kv:-no GPU KV cache size line}"
    if [ "$n" -lt "$starts" ]; then docker rm -f "$CONTAINER" >/dev/null 2>&1; sleep 5; fi
  done
  kv_line "$arm" > "$OUT/$arm.kvlog"
  # accuracy gate first (eviction policy must not change model outputs): GSM8K-200
  if [ "${GSM:-0}" = 1 ] && [ -x /root/gsm8k_any.sh ]; then
    log "$arm: gsm8k-200 accuracy gate"
    CONC=16 /root/gsm8k_any.sh "tierab-$arm-$$" "http://127.0.0.1:$PORT/v1" "$SERVED" > "$OUT/$arm.gsm8k.out" 2>&1
    cp "/root/out/gsm8k-tierab-$arm-$$.log" "$OUT/$arm.gsm8k.log" 2>/dev/null
    log "$arm: gsm8k done (see $arm.gsm8k.out)"
    echo ran > "$OUT/$arm.gate"
  else
    log "$arm: GSM8K gate skipped (GSM!=1 or /root/gsm8k_any.sh missing)"
    echo skipped > "$OUT/$arm.gate"
  fi
  log "$arm: scrape metrics (before)"
  scrape_metrics "$OUT/$arm.metrics.before"
  log "$arm: workload (repeated main + bursts)"
  workload "$arm"
  log "$arm: scrape metrics (after)"
  scrape_metrics "$OUT/$arm.metrics.after"
  metrics_diff "$OUT/$arm.metrics.before" "$OUT/$arm.metrics.after" > "$OUT/$arm.metrics.diff" 2>&1
  log "$arm: metrics diff -> $arm.metrics.diff"
  if [ "$arm" = L ] || [ "$arm" = A ]; then  # sanity: does the RAM reload path fire at all for this arm
    grep -qE "chunk_hits.*\+[1-9]" "$OUT/$arm.metrics.diff" \
      && log "$arm: reload path fires (chunk_hits>0)" \
      || log "$arm: WARN chunk_hits=0 -> RAM reload never fires (hybrid-model ram-mode reload limitation)"
  fi
  kv_line "$arm" >/dev/null 2>&1 || log "$arm: WARN no GPU KV cache size line in log"
  docker rm -f "$CONTAINER" >/dev/null 2>&1
  CLEAN_SHM
  sleep 10
  log "$arm: stopped, vram_used=$(cat "$(dirname "$HWMON")/../mem_info_vram_used" 2>/dev/null)"
}

sample & SPID=$!
for arm in $ARMS; do
  run_arm "$arm" || log "$arm: FAILED"
done

# ---- summary: per-arm GPU pool + primary-tier retention + recompute deltas ---
python3 - "$OUT" $ARMS > "$OUT/summary.tsv" <<'PY'
import os, re, sys
out, arms = sys.argv[1], sys.argv[2:]
print("arm\tgpu_kv_pool\texternal_kv_transfer_delta\tlocal_compute_delta\tchunk_hits_primary_delta\taccuracy_gate")
for a in arms:
    kv = ""
    p = os.path.join(out, a + ".kvlog")
    if os.path.exists(p):
        kv = open(p).read().strip()
    md = os.path.join(out, a + ".metrics.diff")
    ext = lc = hits = "n/a"
    if os.path.exists(md):
        for ln in open(md):
            if 'external_kv_transfer' in ln: ext = ln.rsplit('=',1)[-1].strip()
            if 'local_compute' in ln: lc = ln.rsplit('=',1)[-1].strip()
            if 'chunk_hits' in ln and 'primary' in ln: hits = ln.rsplit('=',1)[-1].strip()
    gate = "n/a"
    gp = os.path.join(out, a + ".gate")
    if os.path.exists(gp):
        gate = open(gp).read().strip()
    print(f"{a}\t{kv or 'n/a'}\t{ext}\t{lc}\t{hits}\t{gate}")
PY

# ---- verdict: L vs A within one window (the real A/B, not a single-arm sanity) --
python3 - "$OUT" > "$OUT/verdict.tsv" <<'PY' 2>&1 | tee -a "$OUT/run.log"
import os, sys, re
out = sys.argv[1]
def num(path):
    h = e = None
    p = os.path.join(out, path)
    if not os.path.exists(p):
        return None, None
    for ln in open(p):
        if 'chunk_hits' in ln and 'primary' in ln:
            h = int(re.sub(r'[^\d-]', '', ln.rsplit('=', 1)[-1]))
        if 'external_kv_transfer' in ln and 'source' in ln:
            e = int(re.sub(r'[^\d-]', '', ln.rsplit('=', 1)[-1]))
    return h, e
lh, le = num("L.metrics.diff")
ah, ae = num("A.metrics.diff")
if lh is None or ah is None:
    print("verdict\ta/b_incomplete\tL/A diff missing"); sys.exit()
ret = "arc_keeps_main" if ah > lh else ("lru_keeps_main" if lh > ah else "equal_hits_no_discrim")
tok = "tokens_differ_measurable" if le != ae else ("ext_kv_zero_no_effect" if (le or 0) == 0 else "both_convert_equally")
print(f"arm_hits\tL={lh}\tA={ah}\tretention={ret}")
print(f"ext_kv\tL={le}\tA={ae}\ttoken_effect={tok}")
print(f"decision\tsave={max(0, (ah-lh))} keep_lru_unless_token_effect=measurable")
PY
log "ALL ARMS FINISHED"
