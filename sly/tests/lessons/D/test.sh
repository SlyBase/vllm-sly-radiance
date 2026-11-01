#!/bin/bash
# D: adaptive verify width (RADIANCE_ADAPTIVE_WIDTH=0|uniform|perseq).  Needs the GPU free; start it yourself.
# Duration: 4 arms ~ 35-40 min (each arm = server start 3-4 min + ~5 min probes); QUICK=1 ~ 22 min;
#           ARMS="off perseq" QUICK=1 ~ 11 min (no A/B/A, noise unknown).
# Arms (ARMS="off uniform perseq off2" default; add "perseq-nograph" to prove the ggz14 diagnosis:
# per-request widths on the PIECEWISE route should lose against the graph-safe arms):
#   off      patched files, RADIANCE_ADAPTIVE_WIDTH=0 (inert; the control)
#   uniform  RADIANCE_ADAPTIVE_WIDTH=uniform  (one width per batch, uniform FULL graph per width)
#   perseq   RADIANCE_ADAPTIVE_WIDTH=perseq   (per-request widths, varlen FULL decode graphs)
#   perseq-nograph  perseq with RADIANCE_AW_VARLEN_GRAPH=0 (per-request widths on PIECEWISE)
#   off2     the control again
# Per arm: greedy single-stream (accprobe3: step ms, tok/upd) + output capture (greedy_eq.py),
#          8-way concurrent greedy capture, c8 arrivals (bbgap2), c4 steady (bbgap2),
#          the policy log (steps shortened) and the graph-route log (FULL vs PIECEWISE steps).
set -u
D=/root/lessons/D; L=/root/lessons; U=http://127.0.0.1:8000
IMAGE=${IMAGE:-vllm-sly-radiance:0.7.0-rocm10.0}
SP=/opt/vllm/lib/python3.12/site-packages
ARMS=${ARMS:-off uniform perseq off2}
QUICK=${QUICK:-0}
R=$D/results-$(date +%m%d-%H%M); mkdir -p $R $D/patched
PER_CAT=3; REPS=2; SREPS=3
[ "$QUICK" = 1 ] && { PER_CAT=2; REPS=1; SREPS=2; }

# ---- 0. patched copies of the five vLLM files, produced by the real patch inside the image (no GPU)
F="vllm/v1/core/sched/scheduler.py vllm/v1/core/sched/async_scheduler.py vllm/v1/worker/gpu/cudagraph_utils.py vllm/v1/worker/gpu/model_runner.py vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
if [ ! -f $D/patched/.done ] || [ $D/patch_adaptive_width.py -nt $D/patched/.done ] || [ $D/radiance_adaptive_width.py -nt $D/patched/.done ]; then
  docker run --rm -e HIP_VISIBLE_DEVICES=-1 -v $D:/w --entrypoint bash $IMAGE -c "set -e; cd /w; cp _patchlib.py /tmp/; \
    cp radiance_adaptive_width.py $SP/; PYTHONPATH=/tmp:/w python patch_adaptive_width.py; \
    for f in $F; do mkdir -p /w/patched/\$(dirname \$f); cp $SP/\$f /w/patched/\$f; done" || { echo "patch step failed"; exit 1; }
  touch $D/patched/.done
fi
MNT="-v $D/radiance_adaptive_width.py:$SP/radiance_adaptive_width.py:ro"
for f in $F; do MNT="$MNT -v $D/patched/$f:$SP/$f:ro"; done

stop() { docker stop -t 60 vllm-lessons >/dev/null 2>&1; docker rm -f vllm-lessons >/dev/null 2>&1; }

arm() { # $1 tag, $2 env knobs for the container
  local t=$1 envs=$2
  echo "=== arm $t  [$envs]  $(date +%T)"
  EXTRA="$MNT $envs -e RADIANCE_AW_GRAPH_STATS=1 -e RADIANCE_AW_LOG_EVERY=200" IMAGE=$IMAGE $L/serve.sh $R/$t-serve.log || { echo "$t: start failed"; return 1; }
  grep -E "GPU KV cache size|Graph capturing finished|extra FULL decode graphs" $R/$t-serve.log | sed 's/.*\] //' | cut -c1-200
  # greedy single stream: step ms + tok/upd, and the raw outputs
  python3 $L/accprobe3.py $U --engine vllm --mode greedy --reps 1 --label $t --json $R/$t-acc.json > $R/$t-acc.txt 2>&1
  grep -E "ALL" $R/$t-acc.txt | tail -1
  python3 $D/greedy_eq.py run $U $t $R/$t-eq.json
  # c8 arrivals, then c4 steady
  BBGAP_OUT=$R/bbgap python3 $L/bbgap2.py run --engine vllm --url $U --tag $t-c8 --per-cat $PER_CAT --reps $REPS --conc 8 --conc-mode arrivals --no-single > $R/$t-c8.txt 2>&1
  grep -E "conc 8|aggregate" $R/$t-c8.txt | head -3
  BBGAP_OUT=$R/bbgap python3 $L/bbgap2.py run --engine vllm --url $U --tag $t-c4 --per-cat $PER_CAT --reps 1 --conc 4 --conc-mode steady --steady-reps $SREPS --steady-warm 1 --no-single > $R/$t-c4.txt 2>&1
  grep -E "conc 4|aggregate|steady" $R/$t-c4.txt | head -3
  docker logs vllm-lessons 2>&1 | grep -E "radiance-aw" > $R/$t-aw.log
  echo "-- policy   : $(grep 'mode=' $R/$t-aw.log | tail -1 | cut -c1-260)"
  echo "-- graph    : $(grep 'graph route' $R/$t-aw.log | tail -1 | cut -c1-260)"
  docker logs vllm-lessons 2>&1 | grep -iE "out of memory|HIP error|Traceback|assert" | head -3
  stop
}

stop
for a in $ARMS; do
  case $a in
    off|off2) arm $a "-e RADIANCE_ADAPTIVE_WIDTH=0" ;;
    uniform)  arm $a "-e RADIANCE_ADAPTIVE_WIDTH=uniform" ;;
    perseq)   arm $a "-e RADIANCE_ADAPTIVE_WIDTH=perseq" ;;
    perseq-nograph) arm $a "-e RADIANCE_ADAPTIVE_WIDTH=perseq -e RADIANCE_AW_VARLEN_GRAPH=0" ;;
    *) echo "unknown arm $a" ;;
  esac
done

echo; echo "=================== SUMMARY ($R)"
REF=$R/off-eq.json
for a in $ARMS; do
  echo "--- $a"
  grep -E "ALL" $R/$a-acc.txt | tail -1 | sed 's/^/   greedy ALL: /'
  [ -f $R/$a-eq.json ] && [ "$a" != off ] && python3 $D/greedy_eq.py cmp $REF $R/$a-eq.json
  grep -E "aggregate" $R/$a-c8.txt | head -1 | sed 's/^/   c8 arrivals: /'
  grep -E "aggregate|steady" $R/$a-c4.txt | head -1 | sed 's/^/   c4 steady  : /'
  echo "   $(grep 'mode=' $R/$a-aw.log 2>/dev/null | tail -1 | cut -c1-230)"
  echo "   $(grep 'graph route' $R/$a-aw.log 2>/dev/null | tail -1 | cut -c1-230)"
done
cat <<'EOF'

DECISION (all must hold for an arm to be adopted; compare against the MEAN of off and off2):
  1. correctness: single-stream greedy outputs identical to off (PASS above); 8-way concurrent common
     prefix not worse than off-vs-off2 (the noise floor); no Traceback/assert/NaN in the arm's log.
  2. gain: c8 arrivals and c4 steady >= +3 % over mean(off, off2), and larger than |off - off2|.
     Radiance measured +9.8 % (arrivals) / +5.7 % (steady c8); this card's vLLM saves ~0.39 ms per row.
  3. mechanism: policy log shows steps shortened (the share of decisions with fewer rows, rows saved per
     step ~6-8 at c8) and the graph-route line shows decode-only steps still on FULL (uniform/perseq)
     as often as in off. If perseq-nograph runs and loses ~3 % while perseq (graphs) wins, ggz14's
     result is explained by the lost FULL graph.
  4. greedy single-stream step ms and tok/upd equal to off (nothing shrunk below 32 unshrunk rows).
  5. KV pool unchanged (436,097 tokens at util 0.98 + FULL_DECODE_ONLY): the extra FULL graphs (uniform: ~15, perseq: ~7) must not
     shrink it. If they do, restrict RADIANCE_AW_LENS (e.g. 3,5) or RADIANCE_AW_GRAPH_MIN_REQS=6.
EOF
