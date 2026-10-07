#!/bin/bash
# /root/lessons/A/test.sh [bench|server|all]     package A: wide decode band (RADIANCE_MXFP4_WIDE_MAX_M)
#
# Needs in /root/lessons/A/: radiance_mxfp4_fp8.so (built from sly/mxfp4/radiance_mxfp4_fp8.hip, see
# build.sh), radiance_mxfp4.py, bench_wide_cells.py, greedy_probe.py, test.sh.  GPU must be free.
#
#   1. KERNEL BENCH  (~4-5 min, container without the model): cur / wide / w1 / w2 / w4 / nt / tn4 arms over
#      the 7 production shapes at M = 16..512, DRAM-fed, with fp32-reference + cur-output exactness gate.
#   2. SERVER A/B/A  (~3 x (start + ~3 min probes)): control (knob dark), WIDE_MAX_M=256, control again.
#      Probes: greedy-equality (12 prompts 72..904 tokens), TTFT at 72/110/160/256/904, c8 arrivals (bbgap2).
#      Runs only if the bench gate passes (exactness PASS and wide >= 3 % better than cur at M=160 summed
#      over a step); FORCE=1 runs it anyway.  ARMS="A B A2" (default), ARMS="A B" for a shorter run.
#
# DECISION (printed at the end):
#   bench   exactness PASS; weighted per-step GEMM sum at M=160/192/256: wide <= -3 % vs cur; M <= 128 unchanged.
#   server  TTFT 160 and 256: B at least 3 % below mean(A, A2) and beyond the A-vs-A2 spread; TTFT 72/110 within
#           +-2 %; c8 arrivals B >= A/A2 minus their spread (no loss); greedy identical >= 10/12 and
#           max |dlogprob| first token < 0.02; log shows "wide decode band ... ON" and mhist decode_kernel=wide.
# Ship (default stays OFF) only if all hold; otherwise the knob stays a measured no-go in docs/NOT-ADOPTED.md.
set -u
MODE=${1:-all}
A=/root/lessons/A
IMG=${IMAGE:-vllm-sly-radiance:0.7.0-rocm10.0}
SP=/opt/vllm/lib/python3.12/site-packages
R=$A/results/$(date +%m%d-%H%M); mkdir -p $R
U=http://127.0.0.1:8000
ARMS=${ARMS:-"A B A2"}
say() { echo; echo "######## $*"; }

[ -f $A/radiance_mxfp4_fp8.so ] || { echo "missing $A/radiance_mxfp4_fp8.so (run $A/build.sh)"; exit 1; }
if curl -s -m 2 -o /dev/null $U/health || docker ps --format '{{.Names}}' | grep -q '^vllm-lessons$'; then
  echo "a server is already up on :8000 / vllm-lessons is running; stop it first"; exit 1; fi

bench() {
  say "1/2 kernel bench -> $R/bench.txt"
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add 44 --group-add 993 \
    --security-opt seccomp=unconfined --ipc=host -e HIP_VISIBLE_DEVICES=0 \
    -e RADIANCE_MXFP4=1 -e RADIANCE_MXFP4_W4A8=1 -e RADIANCE_MXFP4_W4A8_MIN_M=0 \
    -e RADIANCE_MXFP4_DECODE_MAX_M=128 -e RADIANCE_MXFP4_WPERM=1 -e RADIANCE_MXFP4_A_TILED_MIN_M=0 \
    -e PYTHONPATH=/w -v $A:/w -v $R:/out --entrypoint python3 $IMG \
    /w/bench_wide_cells.py --out /out 2>$R/bench.err | tee $R/bench.txt
  BENCH_RC=${PIPESTATUS[0]}
  grep -E "radiance.mxfp4\] kernel:" $R/bench.err | head -1     # which .so was loaded
  python3 - "$R/cells.csv" <<'EOF' | tee $R/gate.txt
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
L = {"gate_up": 64, "down": 64, "qkvz": 48, "in_proj_ba": 48, "out_proj": 48, "qkv": 16, "o_proj": 16}
def tot(arm, M):
    t = 0.0
    for s, l in L.items():
        r = [x for x in rows if x["arm"] == arm and x["shape"] == s and int(x["M"]) == M] or \
            [x for x in rows if x["arm"] == "cur" and x["shape"] == s and int(x["M"]) == M]
        t += float(r[0]["us"]) * l
    return t
g = {M: (tot("wide", M) - tot("cur", M)) / tot("cur", M) * 100 for M in (160, 192, 256)}
lo = max(abs((tot("wide", M) - tot("cur", M)) / tot("cur", M) * 100) for M in (16, 32, 48, 64, 96, 128))
print("GATE wide vs cur per-step GEMM sum: " + "  ".join(f"M={m}: {v:+.1f}%" for m, v in g.items())
      + f"   |delta| at M<=128 (should be ~0): {lo:.1f}%")
print("GATE_PASS" if g[160] <= -3.0 else "GATE_FAIL")
EOF
}

probes() {   # probes <label>
  local L=$1 O=$R/$1; mkdir -p $O
  python3 $A/greedy_probe.py $U $O/greedy.json > $O/greedy.txt 2>&1; cat $O/greedy.txt
  python3 /root/lessons/sweepmixed/ttft.py $U 8 72 110 160 256 904 2>&1 | tee $O/ttft.txt | grep TTFT
  BBGAP_OUT=$O/bbgap python3 /root/lessons/bbgap2.py run --engine vllm --url $U --tag $L --per-cat 3 --reps 2 \
    --conc 8 --conc-mode arrivals --no-single > $O/c8.txt 2>&1
  grep -E "conc 8|aggregate" $O/c8.txt | head -3
}

arm() {      # arm <label> <extra -e flags>
  local L=$1; shift
  say "server arm $L: $*"
  EXTRA="-e RADIANCE_MXFP4_MHIST=1 -v $A/radiance_mxfp4_fp8.so:$SP/radiance_mxfp4_fp8.so:ro -v $A/radiance_mxfp4.py:$SP/radiance_mxfp4.py:ro $*" \
    /root/lessons/serve.sh $R/serve-$L.log || { echo "arm $L failed to start"; docker stop vllm-lessons >/dev/null 2>&1; return 1; }
  grep -E "wide decode band|kernel: " $R/serve-$L.log | head -3
  probes $L
  grep -E "mhist.*decode_kernel=wide" $R/serve-$L.log | head -4
  echo "mhist M>128 lines: $(grep -cE 'mhist.* M=(1[3-9][0-9]|2[0-5][0-9]) ' $R/serve-$L.log)  wide: $(grep -c 'decode_kernel=wide' $R/serve-$L.log)"
  docker stop vllm-lessons >/dev/null 2>&1
}

server() {
  say "2/2 server A/B/A -> $R"
  for a in $ARMS; do
    case $a in
      B) arm B -e RADIANCE_MXFP4_WIDE_MAX_M=256 ;;
      *) arm $a ;;
    esac
  done
  python3 - $R $A <<'EOF'
import re, sys, glob, os
R, AD = sys.argv[1], sys.argv[2]
def ttft(l):
    d = {}
    for ln in open(f"{R}/{l}/ttft.txt") if os.path.exists(f"{R}/{l}/ttft.txt") else []:
        m = re.match(r"TTFT P=(\d+)\s+median\s+([\d.]+)", ln)
        if m: d[int(m[1])] = float(m[2])
    return d
T = {l: ttft(l) for l in ("A", "B", "A2")}
print("\n#### TTFT median ms          A        B       A2   B vs mean(A,A2)   A-A2 spread")
for P in (72, 110, 160, 256, 904):
    a, b, a2 = (T[l].get(P) for l in ("A", "B", "A2"))
    if None in (a, b):
        continue
    ref = (a + a2) / 2 if a2 else a
    print(f"  P={P:<4d}              {a:8.1f} {b:8.1f} {a2 if a2 else float('nan'):8.1f}   {(b - ref) / ref * 100:+8.1f}%   "
          f"{abs(a - a2) / ref * 100 if a2 else float('nan'):6.1f}%")
for l in ("A", "B", "A2"):
    p = f"{R}/{l}/c8.txt"
    if os.path.exists(p):
        print(f"c8 arrivals {l}: " + " | ".join(x.strip() for x in open(p) if re.search(r"conc 8|aggregate", x))[:300])
print()
for l in ("A2", "B"):
    if os.path.exists(f"{R}/{l}/greedy.json") and os.path.exists(f"{R}/A/greedy.json"):
        print(f"greedy A vs {l}: ", end="")
        os.system(f"python3 {AD}/greedy_probe.py cmp {R}/A/greedy.json {R}/{l}/greedy.json")
print("\nApply the DECISION block at the top of test.sh.")
EOF
}

case $MODE in
  bench)  bench ;;
  server) server ;;
  all)
    bench
    if grep -q GATE_PASS $R/gate.txt && [ "${BENCH_RC:-1}" = 0 ]; then server
    elif [ "${FORCE:-0}" = 1 ]; then echo "gate not passed, FORCE=1 -> server phase anyway"; server
    else echo; echo "bench gate not passed (exactness or <3 % gain at M=160): server phase skipped (FORCE=1 to run it)."; fi ;;
esac
echo; echo "results: $R"
