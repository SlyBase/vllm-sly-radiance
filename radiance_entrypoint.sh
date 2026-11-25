#!/usr/bin/env bash
# RADIANCE entrypoint: banner + environment preamble, then hand off to `vllm serve`.
# Runs ONCE in the container's main process (not per TP worker). Every check is best-effort:
# nothing here blocks the serve. Replaces the bare ENTRYPOINT ["vllm","serve"]; the container
# `command:` / `docker run` args after the image become "$@" and are passed through verbatim.
set -u

# Fast path: help/version queries skip the GPU preamble entirely.
case "${1:-}" in
  -h|--help|--version|--help-all) exec vllm serve "$@" ;;
esac

# ------------------------------------------------------------------ NUMA binding
# Optional: pin the whole vLLM fleet to the NUMA node(s) local to the visible GPUs. vLLM's
# TP workers inherit the parent's CPU-affinity mask + NUMA memory policy, so wrapping the
# final `vllm serve` in `numactl` binds them all. Enable with `--numa-bind[=SPEC]` (consumed
# here, never forwarded to vLLM) or RADIANCE_NUMA_BIND; the flag wins. Off by default; a no-op
# on single-node hosts. Needs the `numactl` binary and, under Docker's default seccomp,
# `--cap-add SYS_NICE` for the mempolicy syscalls.
#   SPEC: auto (detect GPU-local node) | <nodes> | bind=<nodes> | interleave[=<nodes>]
#         | preferred=<node> | none
_numa_spec=""
_args=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --numa-bind)   _numa_spec="auto"; shift ;;
    --numa-bind=*) _numa_spec="${1#*=}"; shift ;;
    *)             _args+=("$1"); shift ;;
  esac
done
set -- ${_args[@]+"${_args[@]}"}
[ -z "$_numa_spec" ] && _numa_spec="${RADIANCE_NUMA_BIND:-}"

# NUMA node ids local to the *visible* AMD GPUs, PCI-bus-ordered to match HIP enumeration.
_numa_gpu_nodes() {
  local vis n; local -a ordered=() sel=() nodes=()
  while IFS= read -r n; do ordered+=("$n"); done < <(
    for d in /sys/class/drm/renderD*; do
      [ "$(cat "$d/device/vendor" 2>/dev/null)" = "0x1002" ] || continue
      printf '%s\t%s\n' "$(basename "$(readlink -f "$d/device")")" "$d"
    done | sort | cut -f2-)
  [ "${#ordered[@]}" -gt 0 ] || return 0
  vis="${HIP_VISIBLE_DEVICES:-${ROCR_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}}"
  if [[ "$vis" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    local -a idx; IFS=',' read -ra idx <<<"$vis"
    for n in "${idx[@]}"; do [ "$n" -lt "${#ordered[@]}" ] && sel+=("${ordered[$n]}"); done
  else
    sel=("${ordered[@]}")
  fi
  for d in "${sel[@]}"; do
    n=$(cat "$d/device/numa_node" 2>/dev/null); [[ "$n" =~ ^[0-9]+$ ]] && nodes+=("$n")
  done
  [ "${#nodes[@]}" -gt 0 ] && printf '%s\n' "${nodes[@]}" | sort -un
}

NUMACTL=()
if [ -n "$_numa_spec" ]; then
  export RADIANCE_NUMA_BIND="$_numa_spec"      # so the preamble reports it (flag or env)
  case "${_numa_spec,,}" in
    none|off|"") ;;
    *)
      if ! command -v numactl >/dev/null 2>&1; then
        echo "[radiance] WARN --numa-bind='$_numa_spec' but numactl not found; launching unbound" >&2
      else
        case "${_numa_spec,,}" in
          auto)
            mapfile -t _n < <(_numa_gpu_nodes)
            if   [ "${#_n[@]}" -eq 0 ]; then echo "[radiance] --numa-bind=auto: no GPU-local NUMA node (non-NUMA host?); unbound" >&2
            elif [ "${#_n[@]}" -eq 1 ]; then NUMACTL=(--cpunodebind="${_n[0]}" --membind="${_n[0]}")
            else _s=$(IFS=,; echo "${_n[*]}"); NUMACTL=(--cpunodebind="$_s" --localalloc); fi ;;
          all|interleave) NUMACTL=(--interleave=all) ;;
          interleave=*)   NUMACTL=(--interleave="${_numa_spec#*=}") ;;
          preferred=*)    NUMACTL=(--preferred="${_numa_spec#*=}") ;;
          bind=*)         _s="${_numa_spec#*=}"; NUMACTL=(--cpunodebind="$_s" --membind="$_s") ;;
          [0-9]*)         NUMACTL=(--cpunodebind="$_numa_spec" --membind="$_numa_spec") ;;
          *) echo "[radiance] WARN unrecognised --numa-bind='$_numa_spec' (want: auto|<nodes>|bind=<nodes>|interleave[=nodes]|preferred=node|none); unbound" >&2 ;;
        esac
      fi ;;
  esac
fi
[ "${#NUMACTL[@]}" -gt 0 ] && export RADIANCE_NUMA_ACTIVE="numactl ${NUMACTL[*]}"

# Inkling-2-Mini decode defaults: drafter shortlist + adaptive k keep the greedy output (coda NVFP4 below does not;
# docs/INKLING2-MINI.md). NSD is NOT a default: W3 2026-11-25 changed the greedy text (gate 2/4, degraded).
# The knobs are module-level reads in every engine process, before the model is known, so the model name in
# the served target model (see _target) decides here.
# Not with the DFlash2 drafter (`"method":"dflash"` in --speculative-config, any JSON spacing): MEASURED 221.46 vs
# 223.96 tok/s (-1 %, 4.86 vs 5.13 tokens per update). DSpark or no speculative config keeps them.
# `${VAR-default}` (no colon): any explicit value, even an empty one, wins. Other models: untouched.
# Coda experts NVFP4 (L20-23, MXFP8->NVFP4 at load) is a LOSSY default, user-approved on a measured gate:
# KLD 0.0033 nats vs 0.010 gate, dPPL +0.26 %, +3.9 % tok/s (2026-11-25). Any drafter; `=0` or empty turns it off.
_dflash='"method"[[:space:]]*:[[:space:]]*"dflash"'
# Target model = `--model X` / `--model=X` (last wins) or else a first positional; the drafter path, the tokenizer
# and --served-model-name are labels, not the target (a Qwen target with an `inkling` drafter dir is not Inkling).
# Inkling if (a) the target name says so, (b) a local target's config.json model_type/architectures says so (the
# documented `/model` mount; real value `inkling2_loop`), or (c) `--reasoning-parser inkling` (Inkling-only parser).
_target=""; [ "${1:-}" != "" ] && [ "${1:0:1}" != "-" ] && _target="$1"
_parser=""
for ((_i=1; _i<=$#; _i++)); do
  case "${!_i}" in
    --model=*) _target="${!_i#--model=}" ;;
    --model) _j=$((_i+1)); _target="${!_j-}" ;;
    --reasoning-parser=*) _parser="${!_i#--reasoning-parser=}" ;;
    --reasoning-parser) _j=$((_i+1)); _parser="${!_j-}" ;;
  esac
done
_cfg=""
[ -f "$_target/config.json" ] && _cfg=$(python3 -I -c 'import json,sys;c=json.load(open(sys.argv[1]));print(c.get("model_type"),c.get("architectures"))' "$_target/config.json" 2>/dev/null)
case "${_target,,} ${_cfg,,} ${_parser,,}" in
  *inkling*)
    export RADIANCE_INKLING2_CODA_FP4="${RADIANCE_INKLING2_CODA_FP4-1}"
    [[ "${*,,}" =~ $_dflash ]] || \
    export RADIANCE_INKLING2_DRAFT_SHORTLIST="${RADIANCE_INKLING2_DRAFT_SHORTLIST-4096}" \
      RADIANCE_INKLING2_ADAPTIVE_K="${RADIANCE_INKLING2_ADAPTIVE_K-3,5,7}" ;;
esac

# Topology + bandwidth sweep, kicked off in the BACKGROUND so it never delays startup.
# rocm-bandwidth-test (no args) prints: device list, inter-device access (P2P) matrix, NUMA
# distance, and uni + bidirectional peak copy bandwidth (GB/s) for every agent pair (d2d, d2h,
# h2d). It surfaces a few seconds into vLLM's startup logs. ON by default (the image ships the
# binary and sets RADIANCE_RUN_BWTEST=1; it costs about a second); set RADIANCE_RUN_BWTEST=0 to
# skip it.
if [ "${RADIANCE_RUN_BWTEST:-0}" = "1" ] && ! command -v rocm-bandwidth-test >/dev/null 2>&1; then
  echo "[radiance] WARN RADIANCE_RUN_BWTEST=1 but rocm-bandwidth-test is not on PATH; skipping the sweep" >&2
fi
if [ "${RADIANCE_RUN_BWTEST:-0}" = "1" ] && command -v rocm-bandwidth-test >/dev/null 2>&1; then
  (
    report=$(timeout 150 rocm-bandwidth-test 2>&1) \
      || report="${report}"$'\n'"(rocm-bandwidth-test exited non-zero / timed out)"
    # same colour opt-out the banner honours, so a log scraper gets plain text everywhere
    _c=$'\033[1;38;5;39m'; _r=$'\033[0m'
    if [ -n "${NO_COLOR:-}" ] || [ "${RADIANCE_BANNER_PLAIN:-0}" = "1" ]; then _c=""; _r=""; fi
    printf '\n%s┌─[ RADIANCE · GPU TOPOLOGY & BANDWIDTH (rocm-bandwidth-test) ]%s%s\n%s\n%s└─[ end bandwidth report ]%s%s\n' \
      "$_c" "$(printf '─%.0s' {1..2})" "$_r" "$report" "$_c" "$(printf '─%.0s' {1..30})" "$_r"
  ) &
fi

# Synchronous banner + arch / P2P / optimizations / versions checks. The serve's own args are
# passed through so the banner can report the attention backend that was actually selected.
python /opt/radiance_preamble.py "$@" || true

# Hand off. exec so vLLM becomes PID 1 and receives signals directly; the backgrounded
# bandwidth job keeps its inherited stdout and prints its report when it finishes. When
# --numa-bind resolved a binding, wrap the exec in numactl (TP workers inherit it).
if [ "${#NUMACTL[@]}" -gt 0 ]; then
  exec numactl "${NUMACTL[@]}" vllm serve "$@"
fi
exec vllm serve "$@"
