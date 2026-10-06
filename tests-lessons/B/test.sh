#!/bin/bash
# /root/lessons/B/test.sh -- package B: RADIANCE_MXFP4_GATED_FOLD (gate_up GEMM + SwiGLU + e4m3 quant fold).
# Host 2408, GPU free (main session owns the gpu-window / power setting). Files expected in /root/lessons/B:
#   radiance_mxfp4_fp8.hip radiance_fused_norm.py patch_gated_fold.py _patchlib.py check_gated.py
#   greedy_dump.py cmp_greedy.py   (this script is run from there; scp'd from tests-lessons/B + sly/)
#
# Phases (about 4 min kernel work + 3 server starts):
#   1. compile the extension (hipcc, no GPU needed)
#   2. isolated: fused vs unfused BYTES for every M in 8..64 on the production gate_up N/K, then a
#      DRAM-fed A/B/A microbench of the pair (decode GEMM + silu_mul_quant) vs (gated GEMM + quant rows)
#   3. server A/B/A on the production launch: A = knob unset, B = RADIANCE_MXFP4_GATED_FOLD=1, A again.
#      Per arm: accprobe3 --mode greedy (ms/step, tok/upd) and greedy_dump (completions, byte compare).
#      A and B mount the SAME code (patched qwen2_moe/envs, new radiance_fused_norm.py, new .so); the
#      only difference is the env knob, so A == the unchanged image behaviour and reuses its warm
#      compile cache. B compiles fresh once (knob is in the cache key); rerun with B2=1 to restart B
#      for a second-start measurement.
# Env: SKIP_SERVER=1 (phase 1+2 only)   ABA=0 (skip the closing control arm)   B2=1 (second B start)
#      REPS=2 TOKENS=192 (accprobe3 size)
#
# DECISION: (a) check_gated RESULT must be EXACT, (b) greedy single-stream completions identical
# (cmp_greedy), (c) log shows "mxfp4_gated_quant active" in B only, (d) B ms/step < mean(A1, A2) by more
# than the A1-A2 spread. Radiance measured -0.43 ms (-1.3%) per step single stream; expect the same
# order here, a bit less if the pair's second kernel keeps its launch (it does: see NOTES-B.md).
set -u
W=/root/lessons/B
IMG=${IMAGE:-vllm-sly-radiance:0.7.0-rocm10.0}
SP=/opt/vllm/lib/python3.12/site-packages
REPS=${REPS:-2}; TOKENS=${TOKENS:-192}
cd $W || exit 1
R=$W/results; mkdir -p $R

echo "=== 1. compile"
cat > $W/cc.sh <<'EOF'
cd /w && hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=gfx1201 $(python -m pybind11 --includes) radiance_mxfp4_fp8.hip -o radiance_mxfp4_fp8.so
EOF
docker run --rm -e HIP_VISIBLE_DEVICES=-1 -v $W:/w --entrypoint bash $IMG /w/cc.sh || { echo "COMPILE FAILED"; exit 1; }
ls -la $W/radiance_mxfp4_fp8.so

echo "=== 2. isolated exactness + microbench"
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES=0 -v $W:/w \
  -v $W/radiance_mxfp4_fp8.so:$SP/radiance_mxfp4_fp8.so:ro \
  --entrypoint python $IMG /w/check_gated.py 2>&1 | tee $R/check_gated.txt
CHK=${PIPESTATUS[0]}
echo "check_gated exit=$CHK"
[ "${SKIP_SERVER:-0}" = 1 ] && exit $CHK
[ $CHK -ne 0 ] && { echo "exactness FAILED: not starting servers"; exit 1; }

echo "=== 3. server A/B/A"
# patched vLLM files = the installed (src-070) ones + patch_gated_fold.py
mkdir -p $W/pl/vllm/model_executor/models
cp /root/src-070/vllm/envs.py $W/pl/vllm/envs.py
cp /root/src-070/vllm/model_executor/models/qwen2_moe.py $W/pl/vllm/model_executor/models/qwen2_moe.py
cat > $W/pl/run.py <<'EOF'
import runpy, sys, sysconfig
_o = sysconfig.get_paths
sysconfig.get_paths = lambda *a, **k: {**_o(*a, **k), "purelib": "/root/lessons/B/pl"}
sys.path.insert(0, "/root/lessons/B")
runpy.run_path("/root/lessons/B/patch_gated_fold.py", run_name="__main__")
EOF
python3 $W/pl/run.py || { echo "patch_gated_fold FAILED"; exit 1; }
MNT="-v $W/radiance_mxfp4_fp8.so:$SP/radiance_mxfp4_fp8.so:ro \
 -v $W/radiance_fused_norm.py:$SP/radiance_fused_norm.py:ro \
 -v $W/pl/vllm/model_executor/models/qwen2_moe.py:$SP/vllm/model_executor/models/qwen2_moe.py:ro \
 -v $W/pl/vllm/envs.py:$SP/vllm/envs.py:ro"

arm() {  # arm <label> <extra env>
  local L=$1 ENVX=$2
  echo "--- arm $L ($ENVX)"
  docker stop vllm-lessons >/dev/null 2>&1
  EXTRA="$MNT $ENVX" /root/lessons/serve.sh $R/serve-$L.log || { echo "arm $L: server failed"; tail -20 $R/serve-$L.log; return 1; }
  python3 /root/lessons/accprobe3.py http://127.0.0.1:8000 --mode greedy --reps $REPS --tokens $TOKENS --label $L \
      --json $R/acc-$L.json > $R/acc-$L.txt 2>&1
  grep -E "^  ALL" $R/acc-$L.txt | sed "s/^/[$L] /"
  python3 $W/greedy_dump.py http://127.0.0.1:8000 $R/greedy-$L.json
  echo "[$L] gated path in log: $(grep -c 'mxfp4_gated_quant active' $R/serve-$L.log) line(s)"
  grep -E "gated_fold=|mxfp4_gated_quant active|Traceback|Error" $R/serve-$L.log | head -5
}
arm A1 "" || exit 1
arm B "-e RADIANCE_MXFP4_GATED_FOLD=1" || exit 1
if [ "${B2:-0}" = 1 ]; then arm B2 "-e RADIANCE_MXFP4_GATED_FOLD=1" || exit 1; fi
if [ "${ABA:-1}" = 1 ]; then arm A2 "" || exit 1; fi
docker stop vllm-lessons >/dev/null 2>&1

echo "=== SUMMARY"
BARM=B; [ "${B2:-0}" = 1 ] && BARM=B2
python3 - <<EOF
import json, re
def ms(l):
    try:
        d = json.load(open("$R/acc-%s.json" % l))["modes"]["greedy"]["summary"]["ALL"]
        return d["ms_step"], d["sd_ms_step"], d["tok_upd"], d["cl_tok_s"]
    except Exception as e:
        return None
for l in ("A1", "$BARM", "A2"):
    v = ms(l)
    print("%-3s" % l, "n/a" if v is None else "ms/step %.3f +/- %.3f  tok/upd %.3f  client tok/s %.1f" % v)
a1, b, a2 = ms("A1"), ms("$BARM"), ms("A2")
if a1 and b:
    a = a1[0] if not a2 else 0.5 * (a1[0] + a2[0])
    spread = abs(a1[0] - a2[0]) if a2 else float("nan")
    print("B vs A: %+.3f ms/step (%+.2f%%), A1-A2 spread %.3f ms, Radiance reference -0.43 ms (-1.3%%)" % (b[0] - a, 100 * (b[0] - a) / a, spread))
    print("tok/upd A1 %.3f vs B %.3f (must match: greedy, identical math)" % (a1[2], b[2]))
EOF
echo "-- greedy outputs"
python3 $W/cmp_greedy.py $R/greedy-A1.json $R/greedy-$BARM.json
[ -f $R/greedy-A2.json ] && python3 $W/cmp_greedy.py $R/greedy-A1.json $R/greedy-A2.json
echo "-- isolated: $(grep RESULT $R/check_gated.txt)"
echo "DONE (results in $R)"
