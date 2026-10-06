#!/bin/bash
# /root/lessons/C/test.sh -- package C: int4 lm_head lean configs (RADIANCE_LMHEAD_INT4_LEAN). GPU must be free.
#   phase 0  (no GPU)  compile every config for gfx1201: VGPR / spill / scratch   ~1.5 min
#   phase 1  microbench control vs candidates, M = 8..64, exactness vs control   ~2 min
#   phase 2  server A (control, LEAN=0) -> B (LEAN=auto + winners)  [-> A2 with ABA=1]
#            per arm: accprobe3 greedy (ms/step, tok/upd) + c8 arrivals (bbgap2)    ~5 min per arm
# Total: ~10 min for A+B, ~15-16 min with ABA=1 (a third start).
# Decision: LEAN is worth shipping when (1) phase 1 shows an argmax/top-20-exact candidate >= 5 % faster at
# bucket 64 (the c8 verify and draft M), (2) B greedy tok/upd equals A within noise (same acceptance) and
# ms/step drops, (3) c8 arrivals tok/s of B > A (and A2 ~ A when run).  Radiance saw lm_head 4.52 -> 2.62 ms and
# step -5.2 % but its kernel spilled; this image's does not, so the expected gain is smaller (see NOTES-C.md).
set -u
D=/root/lessons/C
MOD=$D/radiance_lmhead_int4.py          # the changed module (copied next to this script)
SP=/opt/vllm/lib/python3.12/site-packages
IMG=${IMAGE:-vllm-sly-radiance:0.7.0-rocm10.0}
OUT=$D/results; mkdir -p $OUT
cd $D

echo "== phase 0: gfx1201 register / spill report (no GPU)"
docker run --rm -e HIP_VISIBLE_DEVICES=-1 -v $D:/w --entrypoint bash $IMG -c 'cd /w && python compile_cfgs.py 2>&1 | grep -v radiance.gemm' \
  | tee $OUT/compile.txt | awk '/production/ || /^M<=/ {print}'
echo "   (full candidate list: $OUT/compile.txt; lines with spill>0 must not be used)"

echo "== phase 1: microbench"
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 --security-opt seccomp=unconfined --ipc=host \
  -e HIP_VISIBLE_DEVICES=0 -v $D:/w -v /root/vllm7-cache/triton:/root/.triton --entrypoint bash $IMG \
  -c 'cd /w && python microbench.py 2>&1 | grep -v radiance.gemm' | tee $OUT/microbench.txt | grep -vE "^\s+M="
CFG=$(grep '^RECOMMENDED_LEAN_CFG=' $OUT/microbench.txt | sed 's/^RECOMMENDED_LEAN_CFG=//' | awk '{print $1}')
if [ -z "$CFG" ]; then
  echo "== no candidate >= 5 % faster than the stock tile table; skipping the server arms (control stays)."; exit 0
fi
echo "== winners: RADIANCE_LMHEAD_INT4_LEAN_CFG='$CFG'"

arm() {  # arm <label> <extra docker args>
  local L=$1; shift
  echo "== server arm $L"
  EXTRA="$*" /root/lessons/serve.sh $OUT/serve-$L.log || { echo "arm $L failed to start"; return 1; }
  python3 /root/lessons/accprobe3.py http://127.0.0.1:8000 --mode greedy --reps 2 --label $L --json $OUT/acc-$L.json > $OUT/acc-$L.txt 2>&1
  grep -E "^==|ALL" $OUT/acc-$L.txt
  BBGAP_OUT=$OUT/bbgap python3 /root/lessons/bbgap2.py run --engine vllm --url http://127.0.0.1:8000 --tag C-$L \
     --per-cat 3 --reps 2 --conc 8 --conc-mode arrivals --no-single > $OUT/c8-$L.txt 2>&1
  grep -E "conc 8|aggregate" $OUT/c8-$L.txt | head -3
  docker stop vllm-lessons >/dev/null 2>&1
}
MNT="-v $MOD:$SP/radiance_lmhead_int4.py:ro"
arm A  $MNT -e RADIANCE_LMHEAD_INT4_LEAN=0
arm B  $MNT -e RADIANCE_LMHEAD_INT4_LEAN=auto -e "'RADIANCE_LMHEAD_INT4_LEAN_CFG=$CFG'"
[ "${ABA:-0}" = 1 ] && arm A2 $MNT -e RADIANCE_LMHEAD_INT4_LEAN=0
grep -h "int4 lm_head lean" $OUT/serve-B.log | head -2
echo "== summary: compare ms/step + tok/upd (acc-*.txt) and c8 arrivals tok/s (c8-*.txt) of A vs B${ABA:+ vs A2}"
