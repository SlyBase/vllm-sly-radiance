#!/bin/bash
# H: R4D_HYBRID attention backend -- libr4d prefill for long prefill runs, ROCM_AITER_UNIFIED_ATTN for the rest.
# Arms (A/B/A, every arm measured on its SECOND start where a start compiles):
#   C1  control   prod launch (ROCM_AITER_UNIFIED_ATTN), compile cache warm -> one start
#   H0  hybrid    --attention-backend R4D_HYBRID, FIRST start (fresh compile ~6 min, only the routing log is read)
#   H   hybrid    second start -> full probe
#   C2  control   again
# optional (runtime env only, no recompile; set H_OPT="fp8 minq256" or either): hybrid + R4D_ATTN_FP8=3,
#   hybrid + RADIANCE_R4D_PREFILL_MIN_Q=256.
# Time: ~45 min (starts ~17 min: 3x ~3.5 + 1x ~6.5; probes 4 x ~8 min; summary instant). Each optional arm +9 min.
# Needs the GPU free (main session owns the window). Does NOT touch power or other units.
set -u
D=/root/lessons/H; R=$D/results-$(date +%m%d-%H%M); mkdir -p $R; L=/root/lessons; U=http://127.0.0.1:8000
IMG=${IMAGE:-vllm-sly-radiance:0.7.0-rocm10.0}; SP=/opt/vllm/lib/python3.12/site-packages
stop() { docker stop -t 60 vllm-lessons >/dev/null 2>&1; docker rm -f vllm-lessons >/dev/null 2>&1; }

# --- 0. CPU-only self test of the planner and the sub-block addressing (aborts on failure) ---
docker run --rm -e HIP_VISIBLE_DEVICES=-1 -v $D:/h --entrypoint bash $IMG -c 'cd /h && PYTHONPATH=/h python selftest_cpu.py' > $R/selftest.txt 2>&1 \
  || { cat $R/selftest.txt; echo "selftest failed"; exit 1; }
tail -1 $R/selftest.txt

# --- 1. the image's registry.py plus the R4D_HYBRID enum member (what patch_r4d.py does at build time) ---
docker run --rm --entrypoint cat $IMG $SP/vllm/v1/attention/backends/registry.py > $R/registry.py
grep -q R4D_HYBRID $R/registry.py || python3 - "$R/registry.py" <<'EOF'
import sys
p = sys.argv[1]; s = open(p).read()
a = '    R4D = "radiance_r4d_attn.R4DAttentionBackend"\n'
assert s.count(a) == 1, "R4D enum line not found"
open(p, "w").write(s.replace(a, a + '    R4D_HYBRID = "radiance_r4d_hybrid_attn.R4DHybridAttentionBackend"\n'))
EOF
HYB_MOUNTS="-v $D/radiance_r4d_hybrid_attn.py:$SP/radiance_r4d_hybrid_attn.py:ro -v $R/registry.py:$SP/vllm/v1/attention/backends/registry.py:ro"

arm() { # tag "extra docker args" "extra vllm args"
  local t=$1; echo "=== $t $(date +%T)"
  EXTRA="$2" ARGS_EXTRA="$3" $L/serve.sh $R/$t-serve.log || { echo "$t: start failed"; tail -25 $R/$t-serve.log; return 1; }
  grep -E "GPU KV cache size" $R/$t-serve.log | tail -1 | sed 's/.*\] //'
  grep -E "R4D_HYBRID|Using .* backend|block size to" $R/$t-serve.log | cut -c1-220 | head -6
}
probe() { # tag
  local t=$1
  python3 $D/probe_h.py $U $R/$t-probe.json > $R/$t-probe.txt 2>&1; tail -3 $R/$t-probe.txt          # cold cache first
  python3 $L/accprobe3.py $U --engine vllm --mode greedy --reps 1 --label $t --json $R/$t-acc.json > $R/$t-acc.txt 2>&1; grep -E "ALL" $R/$t-acc.txt | tail -1
  python3 $L/sweepmixed/ttft.py $U 8 72 256 904 2>&1 | tee $R/$t-ttft.txt | grep -i ttft
  python3 $L/prefbench.py $U --sizes 2048,8192,32768,65536 --reps 2 | tee $R/$t-pp.txt
  BBGAP_OUT=$R/bbgap python3 $L/bbgap2.py run --engine vllm --url $U --tag $t --per-cat 3 --reps 2 --conc 8 --conc-mode arrivals --no-single > $R/$t-c8.txt 2>&1
  grep -E "aggregate" $R/$t-c8.txt | head -1
  docker logs vllm-lessons 2>&1 | grep -E "R4D_HYBRID|HIP error|out of memory|Traceback" | sort | uniq -c | head -6 > $R/$t-routelog.txt; cat $R/$t-routelog.txt
}

arm C1 "" "" && probe C1; stop
arm H0 "$HYB_MOUNTS" "--attention-backend R4D_HYBRID"; stop               # fresh compile, discarded
arm H "$HYB_MOUNTS" "--attention-backend R4D_HYBRID" && probe H; stop
arm C2 "" "" && probe C2; stop
for o in ${H_OPT:-}; do
  case $o in
    fp8)     arm HF "$HYB_MOUNTS -e R4D_ATTN_FP8=3" "--attention-backend R4D_HYBRID" && probe HF; stop ;;
    minq256) arm HQ "$HYB_MOUNTS -e RADIANCE_R4D_PREFILL_MIN_Q=256" "--attention-backend R4D_HYBRID" && probe HQ; stop ;;
  esac
done

# --- summary ---
echo "=== summary  ($R)"
for t in C1 H C2 ${H_OPT:+HF HQ}; do
  [ -f $R/$t-acc.txt ] || continue
  echo "-- $t"
  grep KV $R/$t-serve.log | grep -o "GPU KV cache size: [0-9,]* tokens" | tail -1
  grep ALL $R/$t-acc.txt | tail -1
  grep -i ttft $R/$t-ttft.txt | tr '\n' ' ' | sed 's/  */ /g'; echo
  grep median $R/$t-pp.txt | sed 's/ runs.*//' | tr '\n' ' ' | sed 's/  */ /g'; echo
  grep aggregate $R/$t-c8.txt | head -1
done
echo "-- routing in H (must show the libr4d line, and no 'AITER only' fallback)"; cat $R/H-routelog.txt 2>/dev/null
echo "-- greedy text, SHORT prompts (below MIN_Q, decode+prefill path unchanged => must be identical)"
echo "C1 vs C2 (noise floor, must also be identical):"; python3 $D/compare_h.py $R/C1-probe.json $R/C2-probe.json | grep -E "short|SHORT"
echo "C1 vs H:"; python3 $D/compare_h.py $R/C1-probe.json $R/H-probe.json
cat <<'EOF'
decision (all against the mean of C1/C2, spread of C1 vs C2 is the noise):
  adopt R4D_HYBRID if
   (1) routing line present in H and no fallback line;
   (2) greedy step (ALL ms/step) and tok/upd of H equal control within the C1/C2 spread (decode path untouched), and TTFT 72/256 equal;
   (3) short-prompt greedy text identical to control (C1 vs H and C1 vs C2 both PASS);
   (4) prefill 32k/64k >= +10 % (R4D alone gave +9 % / +19 % at 32k / 64k), 2k/8k not worse, TTFT 904 not worse than +3 ms;
   (5) c8 arrivals not worse than control by more than the C1/C2 spread; KV pool equal to control;
   (6) long-prompt NLL delta small (|delta mean NLL| < 0.005, no systematic drift) -- then run the GSM8K 200 gate of AGENTS.md.
  fp8 arm (H_OPT=fp8): control already runs fp8 q + fp8 P (P.to(V.dtype) in the triton kernel), so R4D_ATTN_FP8=3 matches its
  precision class; adopt only if it adds >= 5 % at 32k/64k over H and NLL delta stays within the control's own spread.
EOF
