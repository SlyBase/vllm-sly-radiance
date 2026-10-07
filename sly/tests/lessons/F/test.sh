#!/bin/bash
# sly/tests/lessons/F/test.sh -- vllm-sly-radiance 0.8.0-rc-rocm10.1 (torch 2.12 / ROCm 10.1) against 0.7.0-rocm10.0.
# Runs on 2408 as root; the GPU must be free (the main session owns the window: 300 W, firmware fan curve).
# Lives in /root/lessons/F/ next to greedy.py. Arms (A/B/A):
#   base (0.7.0) -> rc (10.1: cold start, then second start) -> base again
#   ARMS="base rc base"  default;  ARMS="rc" only the candidate (~25 min);  ARMS="rc2" skips the cold start
# Expected total ~50 min (the cold start of 10.1 compiles torch.compile / triton / aiter fresh, 10-20 min;
# every further start 3-6 min; ~6 min of probes per arm). The total is printed at the end.
#
# DECISION CRITERIA (against the control arms of the same window)
#   1 both starts of 10.1 reach /health, no Traceback / EngineCore death in the log
#   2 KV pool of the second start >= 430,433 tokens at util 0.98 (1.0.0 production number; the util 0.9655 arm shows 391,193)
#   3 idle CPU after the first requests < 150 % for the whole container. ROCm/ROCm#6406 (torch >= 2.12 spins a core in
#     libhsa-runtime64 after the first GPU op) is why the torch pin sat on 2.11; ROCm 10.1 should carry the fix
#   4 greedy: tok/upd within 1 % and ms/step within +-2 % of the control (the A/A spread is the noise floor); texts
#     identical or diverging only late (other libs: bit-exactness is not guaranteed, look at where it diverges)
#   5 TTFT and prefill 2k/8k within +-3 %; second-start time not worse than 0.7.0's by more than 20 %
# Anything else: stay on 0.7.0 / ROCm 10.0.
set -u
RC_IMAGE=${RC_IMAGE:-vllm-sly-radiance:0.8.0-rc-rocm10.1}
BASE_IMAGE=${BASE_IMAGE:-vllm-sly-radiance:0.7.0-rocm10.0}
ARMS=${ARMS:-"base rc base"}
F=/root/lessons/F
R=$F/results/$(date +%Y%m%d-%H%M)
mkdir -p "$R"
T0=$(date +%s)

# serve.sh with its own execstart: the rc image gets its own triton / torch_compile / aiter caches
# (the production ones hold aiter .so built by 10.0's hipcc and torch 2.11 compile artefacts).
mkdir -p $F/cache-rc/triton $F/cache-rc/torch_compile $F/cache-rc/aiter
sed 's#/root/vllm7-cache/#/root/lessons/F/cache-rc/#' /root/lessons/execstart.txt > $F/execstart-rc.txt
sed 's#/root/lessons/execstart.txt#${EXECSTART:-/root/lessons/execstart.txt}#' /root/lessons/serve.sh > $F/serve.sh
chmod +x $F/serve.sh

start() {  # start <label> <image> <execstart>
  local label=$1 image=$2 es=$3 t rc dt
  t=$(date +%s)
  EXECSTART=$es IMAGE=$image $F/serve.sh "$R/$label.log" > "$R/$label.start" 2>&1
  rc=$?
  dt=$(( $(date +%s) - t ))
  echo "$label: start rc=$rc wall=${dt}s $(grep -hE 'GPU KV cache size' "$R/$label.log" | tail -1 | sed 's/.*GPU KV/GPU KV/')" | tee -a "$R/summary.txt"
  echo "$label: traceback-lines $(grep -cE 'Traceback|EngineCore.*(died|failed)' "$R/$label.log")" | tee -a "$R/summary.txt"
  return $rc
}
cpu_idle() {  # container CPU after 20 s idle; ~0-50 % healthy, >= 100 % per spinning core (criterion 3)
  sleep 20
  local pcpu dstat
  pcpu=$(docker exec vllm-lessons ps -eo pcpu --no-headers 2>/dev/null | awk '{s+=$1} END {printf "%.0f", s}')
  dstat=$(docker stats --no-stream --format '{{.CPUPerc}}' vllm-lessons)
  echo "$1: idle CPU docker-stats=$dstat ps-sum(lifetime avg)=${pcpu}%" | tee -a "$R/summary.txt"
}
probes() {  # probes <label>
  local l=$1 U=http://127.0.0.1:8000
  python3 $F/greedy.py $U "$R/$l.greedy.json" | tee -a "$R/$l.probe.txt"
  cpu_idle "$l"
  python3 /root/lessons/accprobe3.py $U --engine vllm --mode greedy --reps 2 --label "$l" --json "$R/$l.acc.json" > "$R/$l.acc.txt" 2>&1
  grep -E '^== |ALL' "$R/$l.acc.txt" | sed "s/^/$l: /" | tee -a "$R/summary.txt"
  python3 /root/lessons/sweepmixed/ttft.py $U 10 72 110 256 2>&1 | grep TTFT | sed "s/^/$l: /" | tee -a "$R/summary.txt"
  python3 /root/lessons/prefbench.py $U --sizes 2048,8192 --reps 3 2>&1 | sed "s/^/$l: /" | tee -a "$R/summary.txt"
}
stop() { docker stop vllm-lessons >/dev/null 2>&1; docker rm -f vllm-lessons >/dev/null 2>&1; sleep 5; }

n=0
for arm in $ARMS; do
  n=$((n+1))
  case $arm in
    base) L=base$n; start $L $BASE_IMAGE /root/lessons/execstart.txt && probes $L; stop ;;
    rc)   L=rc-first; start $L $RC_IMAGE $F/execstart-rc.txt && cpu_idle $L; stop      # cold: compiles, smaller KV pool
          L=rc-second; start $L $RC_IMAGE $F/execstart-rc.txt && probes $L; stop ;;
    rc2)  L=rc-second; start $L $RC_IMAGE $F/execstart-rc.txt && probes $L; stop ;;
  esac
done

echo "=== greedy text comparison (control base1 -> candidate rc-second, then A/A)" | tee -a "$R/summary.txt"
if [ -f "$R/base1.greedy.json" ] && [ -f "$R/rc-second.greedy.json" ]; then
  python3 $F/greedy.py --cmp "$R/base1.greedy.json" "$R/rc-second.greedy.json" | tee -a "$R/summary.txt"
fi
if [ -f "$R/base1.greedy.json" ] && [ -f "$R/base3.greedy.json" ]; then
  python3 $F/greedy.py --cmp "$R/base1.greedy.json" "$R/base3.greedy.json" | sed 's/^/A\/A: /' | tee -a "$R/summary.txt"
fi
echo "=== total $(( ($(date +%s) - T0) / 60 )) min; results in $R (summary.txt). Apply criteria 1-5 from the header."
