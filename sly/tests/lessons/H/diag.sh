#!/bin/bash
# diag.sh -- WHY is `--attention-backend R4D` decode 60.9 ms/step against 34.9 (AITER)? ~15 min, run
# BEFORE or after test.sh (the R4D compile cache from the G run is warm, so both starts are ~3.5 min).
# For each of R4D and AITER (control): start with the torch profiler, decode one request (256 tokens),
# profile the decode window, summarise the GPU trace. Read the two summaries side by side:
#   * R4D attention kernels dominate            -> the decode kernel / its launch shape is the cost
#   * GPU idle share much larger than control   -> CPU / sync bound (the step waits on the host)
#   * many more kernels per step than control   -> not graph-replayed (eager / piecewise decode)
set -u
D=/root/lessons/H; R=$D/diag-$(date +%m%d-%H%M); mkdir -p $R $D/prof; L=/root/lessons; U=http://127.0.0.1:8000
stop() { docker stop -t 60 vllm-lessons >/dev/null 2>&1; docker rm -f vllm-lessons >/dev/null 2>&1; }
PROF="--profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/prof\"}'"
for arm in R4D ROCM_AITER_UNIFIED_ATTN; do
  rm -rf $D/prof/*; echo "=== $arm $(date +%T)"
  ARGS_EXTRA="--attention-backend $arm $PROF" EXTRA="-v $D/prof:/prof" $L/serve.sh $R/$arm-serve.log || { echo "$arm start failed"; stop; continue; }
  grep -E "cudagraph|CUDAGraph|Capturing CUDA graphs \((FULL|PIECEWISE)\): 100%" $R/$arm-serve.log | cut -c1-200 | tail -6
  python3 $L/accprobe3.py $U --engine vllm --mode greedy --reps 1 --tokens 64 --label warm-$arm > /dev/null 2>&1
  curl -s -X POST $U/start_profile > /dev/null
  python3 $L/accprobe3.py $U --engine vllm --mode greedy --reps 1 --tokens 128 --label $arm --json $R/$arm-acc.json > $R/$arm-acc.txt 2>&1
  curl -s -X POST $U/stop_profile > /dev/null; sleep 25
  grep ALL $R/$arm-acc.txt | tail -1
  for f in $D/prof/*.json.gz; do echo "-- $f"; python3 $D/trace_sum.py "$f" 20 | tee -a $R/$arm-trace.txt; done
  stop
done
echo "results in $R"
