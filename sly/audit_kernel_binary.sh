#!/usr/bin/env bash
# audit_kernel_binary.sh -- RDNA4 (gfx1201) code-object efficiency audit for the
# radiance HIP kernels.
#
# Takes built extension .so files (or raw .hsaco/.co code objects, or
# directories containing them) and prints one row per device kernel with the
# numbers the RDNA4 efficiency budgets are stated against:
#
#   .vgpr_count / .vgpr_spill_count   budget: 0 spills, ~<=213 VGPR review band
#   .wavefront_size                   budget: wave32 everywhere
#   .group_segment_fixed_size         LDS bytes per block
#   ds_* instruction census           LDS traffic in the hot loop
#   branch census (s_branch/s_cbranch_*), s_barrier count, scratch_* count
#
# No GPU is required: this works on build artifacts alone. hipcc needs no GPU
# either, so a CI job can do: hipcc ... file.hip -o file.so; this file.so.
#
# Device code inside a hipcc-built .so lives in an embedded HIP fatbin, so
# code objects are extracted first, in this order:
#   1. rocm-obj-extract        (any ROCm >= 6; extracts every target)
#   2. clang-offload-bundler   (ships with hipcc; --list then --unbundle)
# Raw .hsaco/.co inputs skip extraction and are analyzed directly.
#
# Analysis tools (searched in AUDIT_LLVM_BIN, PATH, /opt/rocm/llvm/bin,
# /opt/rocm/bin, /usr/lib/llvm-*/bin):
#   llvm-readelf --notes   -> per-kernel amdgcn metadata (vgpr, spills,
#                             wavefront size, group segment, ...)
#   llvm-objdump  -d       -> ds_*/branch/barrier/scratch census
# Missing tools degrade gracefully: no readelf -> '?' metadata columns; no
# objdump -> '?' census columns; neither -> the file is reported as not
# analyzable (exit 2 if nothing at all could be analyzed).
#
# Verdicts are advisory unless --strict is given:
#   WAVE64   wavefront_size != 32      (RDNA4 budget wants wave32)
#   SPILL    vgpr_spill_count > N      (--spill-max, default 0)
#   VGPR     vgpr_count > N            (--vgpr-max, default 213)
#
# Usage:
#   audit_kernel_binary.sh [options] FILE|DIR ...
#
# Options:
#   --list          only list code-object targets and kernel names
#   --json          one JSON object per kernel (JSONL) instead of a table
#   --strict        exit 1 if any kernel trips a budget verdict
#   --all-syms      also report disassembled symbols that are not kernels
#   --vgpr-max N    VGPR review band (default 213)
#   --spill-max N   allowed spill count (default 0)
#   --wave N        required wavefront size (default 32)
#   --target GLOB    only analyze targets matching GLOB (default '*')
#   --help          this text
#
# Examples:
#   # inside the runtime image (hipcc's llvm tools live in /opt/rocm/llvm/bin):
#   sly/audit_kernel_binary.sh --list \
#       /opt/vllm/lib/python3.12/site-packages/radiance_mxfp4_fp8.so
#   sly/audit_kernel_binary.sh \
#       /opt/vllm/lib/python3.12/site-packages/radiance_*_kernel.so
#
#   # no container handy? build a .so first (no GPU needed), then audit it:
#   hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=gfx1201 \
#       $(python3 -m pybind11 --includes) \
#       sly/mxfp4/radiance_mxfp4_fp8.hip -o /tmp/radiance_mxfp4_fp8.so
#   sly/audit_kernel_binary.sh /tmp/radiance_mxfp4_fp8.so
#
# Exit codes: 0 findings printed (warnings allowed); 1 budget violations with
# --strict; 2 usage error or nothing analyzable.

set -u -o pipefail

PROG="audit_kernel_binary.sh"

# RDNA4 budget defaults
VGPR_MAX=213
SPILL_MAX=0
WAVE_REQ=32
MODE="table"        # table | list | json
STRICT=0
ALLSYMS=0
TARGET_GLOB="*"

READELF=""
OBJDUMP=""
EXTRACT=""
BUNDLER=""

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/audk.XXXXXX")" || exit 2
trap 'rm -rf "$TMPD"' EXIT
CO_LIST="$TMPD/cos.txt"
META="$TMPD/meta.tsv"
CENSUS="$TMPD/census.tsv"
ROWS="$TMPD/rows.txt"

VIOL=0
ROWS_EMITTED=0
LIST_EMITTED=0
FILES_FAILED=0

usage() {
  awk 'NR > 1 && /^set / {exit} NR > 1 {sub(/^# ?/, ""); print}' "$0"
}

warn() { printf '%s: %s\n' "$PROG" "$*" >&2; }

# ---------------------------------------------------------------- arguments

FILES=()
need_value() { [ $# -ge 2 ] || { warn "option $1 needs a value"; exit 2; }; }
while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h)   usage; exit 0 ;;
    --list)      MODE=list ;;
    --json)      MODE=json ;;
    --strict)    STRICT=1 ;;
    --all-syms)  ALLSYMS=1 ;;
    --vgpr-max)  need_value "$@"; VGPR_MAX=$2; shift ;;
    --spill-max) need_value "$@"; SPILL_MAX=$2; shift ;;
    --wave)      need_value "$@"; WAVE_REQ=$2; shift ;;
    --target)    need_value "$@"; TARGET_GLOB=$2; shift ;;
    --)          shift; while [ $# -gt 0 ]; do FILES+=("$1"); shift; done; break ;;
    -*)          warn "unknown option: $1"; usage >&2; exit 2 ;;
    *)           FILES+=("$1") ;;
  esac
  shift
done
if [ "${#FILES[@]}" -eq 0 ]; then
  usage >&2
  exit 2
fi

# ---------------------------------------------------------------- tool probe

find_tool() {
  local base="$1" cand d v
  local search="${AUDIT_LLVM_BIN:-}:${PATH}:/opt/rocm/llvm/bin:/opt/rocm/bin:/usr/lib/llvm-20/bin:/usr/lib/llvm-19/bin:/usr/lib/llvm-18/bin:/usr/lib/llvm-17/bin:/usr/lib/llvm-16/bin:/usr/lib/llvm-15/bin"
  for v in 20 19 18 17 16 15 14; do
    cand="$(IFS=:; for d in $search; do
      [ -n "$d" ] && [ -x "$d/${base}-${v}" ] && { printf '%s' "$d/${base}-${v}"; break; }
    done; true)"
    [ -n "$cand" ] && { printf '%s\n' "$cand"; return 0; }
  done
  cand="$(IFS=:; for d in $search; do
    [ -n "$d" ] && [ -x "$d/$base" ] && { printf '%s' "$d/$base"; break; }
  done; true)"
  [ -n "$cand" ] && { printf '%s\n' "$cand"; return 0; }
  return 1
}

READELF="$(find_tool llvm-readelf || true)"
OBJDUMP="$(find_tool llvm-objdump || true)"
EXTRACT="$(find_tool rocm-obj-extract || true)"
BUNDLER="$(find_tool clang-offload-bundler || true)"

if [ -z "$READELF" ] && [ -z "$OBJDUMP" ]; then
  warn "no llvm tools found (tried AUDIT_LLVM_BIN, PATH, /opt/rocm/llvm/bin, /usr/lib/llvm-*/bin)"
  warn "install rocm-llvm / llvm-XX-tools, or point AUDIT_LLVM_BIN at a bin dir"
fi

# ---------------------------------------------------------------- parsers
# metadata -> TSV: name sym vgpr spill sgpr agpr wave gseg pseg kseg maxwg
# Handles both code-object v4/v5 msgpack dumps (.name, .vgpr_count, ...;
# .wavefront_size is log2 there) and older v3 YAML (Name:, ...).
META_AWK='
function flushrec() {
  if (name != "" && (vgpr != "" || gseg != "")) {
    if (wave != "" && wave <= 7) wave = 2 ^ wave
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
      name, sym, vgpr, spill, sgpr, agpr, wave, gseg, pseg, kseg, maxwg
  }
  name = sym = vgpr = spill = sgpr = agpr = wave = gseg = pseg = kseg = maxwg = ""
}
/^[[:space:]]*-[[:space:]]*[.A-Za-z_][A-Za-z0-9_.]*[[:space:]]*:/ { flushrec() }
{
  if ($0 !~ /^[[:space:]]*-?[[:space:]]*[.]?[A-Za-z_][A-Za-z0-9_]*[[:space:]]*:/) next
  t = $0
  sub(/^[[:space:]]*-?[[:space:]]*/, "", t)
  p = index(t, ":")
  if (p < 2) next
  k = tolower(substr(t, 1, p - 1))
  sub(/^[.]*/, "", k)
  v = substr(t, p + 1)
  gsub(/\047/, "", v)
  gsub(/"/, "", v)
  sub(/^[[:space:]]+/, "", v)
  sub(/[[:space:]]+$/, "", v)
  if      (k == "name")                       name  = v
  else if (k == "symbol")                     sym   = v
  else if (k == "vgpr_count")                 vgpr  = v
  else if (k == "vgpr_spill_count")           spill = v
  else if (k == "sgpr_count")                 sgpr  = v
  else if (k == "agpr_count")                 agpr  = v
  else if (k == "wavefront_size")             wave  = v
  else if (k == "group_segment_fixed_size")   gseg  = v
  else if (k == "private_segment_fixed_size") pseg  = v
  else if (k == "kernarg_segment_size")       kseg  = v
  else if (k == "max_flat_workgroup_size")    maxwg = v
  else if (k == "vgprcount")                vgpr  = v
  else if (k == "vgprspillcount")           spill = v
  else if (k == "sgprcount")                sgpr  = v
  else if (k == "agprcount")                agpr  = v
  else if (k == "wavefrontsize")           wave  = v
  else if (k == "groupsegmentfixedsize")    gseg  = v
  else if (k == "privatesegmentfixedsize")  pseg  = v
  else if (k == "kernargsegmentsize")       kseg  = v
  else if (k == "maxflatworkgroupsize")     maxwg = v
}
END { flushrec() }
'

# disassembly -> TSV: label ds_count branch_count barrier_count scratch_count
CENSUS_AWK='
function flushrec() {
  if (lbl != "") printf "%s\t%s\t%s\t%s\t%s\n", lbl, ds, br, bar, scr
  lbl = ""; ds = br = bar = scr = 0
}
/^[[:space:]]*[0-9a-fA-F]+ <.*>:[[:space:]]*$/ {
  flushrec()
  lbl = $0
  sub(/^[[:space:]]*[0-9a-fA-F]+ </, "", lbl)
  sub(/>:[[:space:]]*$/, "", lbl)
  next
}
{
  p = index($0, ":")
  if (p < 2) next
  pre = substr($0, 1, p - 1)
  sub(/^[[:space:]]+/, "", pre)
  if (pre !~ /^[0-9a-fA-F]+$/) next
  m = substr($0, p + 1)
  sub(/^[[:space:]]+/, "", m)
  while (m ~ /^[0-9a-fA-F][0-9a-fA-F][[:space:]]/)   # raw bytes, if not suppressed
    sub(/^[0-9a-fA-F][0-9a-fA-F][[:space:]]/, "", m)
  if      (m ~ /^ds_/)                       ds++
  else if (m ~ /^s_cbranch/)                  br++
  else if (m ~ /^s_branch([[:space:]]|$)/)    br++
  else if (m ~ /^s_barrier/)                  bar++
  else if (m ~ /^scratch_/)                   scr++
}
END { flushrec() }
'

# join census + metadata -> table or JSONL rows
JOIN_AWK='
function val(v) { return (v == "" ? "-" : v) }
function verdict(vg, sp, wv,    s) {
  s = "ok"
  if (wv != "-" && (wv + 0) != WAVE_REQ) s = "WAVE" wv
  if (sp != "-" && (sp + 0) > SPILL_MAX) s = s (s == "ok" ? "" : "+") "SPILL" sp
  if (vg != "-" && (vg + 0) > VGPR_MAX) s = s (s == "ok" ? "" : "+") "VGPR" vg
  if (s != "ok") viol++
  return s
}
NR==FNR { c_ds[$1] = $2; c_br[$1] = $3; c_bar[$1] = $4; c_scr[$1] = $5; seen[$1] = 1; next }
{
  nm = $1; key = $2
  if (key == "-") key = nm
  sub(/\.kd$/, "", key)
  ds = br = bar = scr = "-"
  if (key in c_ds) {
    ds = c_ds[key]; br = c_br[key]; bar = c_bar[key]; scr = c_scr[key]; emitted[key] = 1
  } else if (nm in c_ds) {
    ds = c_ds[nm]; br = c_br[nm]; bar = c_bar[nm]; scr = c_scr[nm]; emitted[nm] = 1
  } else {
    for (l in c_ds) {                        # demangled label: "<name>(<...>)"
      if (substr(l, 1, length(nm)) == nm) {
        c = substr(l, length(nm) + 1, 1)
        if (c == "" || c == "<" || c == "(") {
          ds = c_ds[l]; br = c_br[l]; bar = c_bar[l]; scr = c_scr[l]; emitted[l] = 1
          break
        }
      }
    }
  }
  vd = verdict(val($3), val($4), val($7))
  if (MODE == "json") {
    printf "{\"src\":\"%s\",\"arch\":\"%s\",\"kernel\":\"%s\",\"vgpr\":%s,\"vgpr_spill\":%s,\"sgpr\":%s,\"agpr\":%s,\"wavefront\":%s,\"group_seg\":%s,\"priv_seg\":%s,\"kernarg\":%s,\"wg_max\":%s,\"ds\":%s,\"branch\":%s,\"s_barrier\":%s,\"scratch\":%s,\"verdict\":\"%s\"}\n", \
      SRC, ARCH, nm, jv($3), jv($4), jv($5), jv($6), jv($7), jv($8), jv($9), jv($10), jv($11), jv(ds), jv(br), jv(bar), jv(scr), vd
  } else {
    printf "%-44s %5s %5s %5s %5s %4s %8s %8s %8s %5s %5s %6s %5s %6s  %s\n", \
      substr(nm, 1, 44), val($3), val($4), val($5), val($6), val($7), \
      val($8), val($9), val($10), val($11), val(ds), val(br), val(bar), val(scr), vd
  }
}
END {
  if (ALLSYMS)                                 # disassembled non-kernel symbols
    for (l in seen)
      if (!(l in emitted)) {
        if (MODE == "json")
          printf "{\"src\":\"%s\",\"arch\":\"%s\",\"kernel\":\"%s\",\"vgpr\":null,\"vgpr_spill\":null,\"sgpr\":null,\"agpr\":null,\"wavefront\":null,\"group_seg\":null,\"priv_seg\":null,\"kernarg\":null,\"wg_max\":null,\"ds\":%s,\"branch\":%s,\"s_barrier\":%s,\"scratch\":%s,\"verdict\":\"sym\"}\n", \
            SRC, ARCH, l, c_ds[l], c_br[l], c_bar[l], c_scr[l]
        else
          printf "%-44s %5s %5s %5s %5s %4s %8s %8s %8s %5s %5s %6s %5s %6s  sym\n", \
            substr(l, 1, 44), "-", "-", "-", "-", "-", "-", "-", "-", "-", \
            c_ds[l], c_br[l], c_bar[l], c_scr[l]
      }
  if (viol > 0) printf "__VIOLATIONS__\t%d\n", viol
}
function jv(v) { return (v == "" || v == "-") ? "null" : v }
'

# ---------------------------------------------------------------- helpers

meta_of() { "$READELF" --notes "$1" 2>/dev/null | awk "$META_AWK"; }

arch_of() {
  if [ -n "$READELF" ]; then
    "$READELF" --notes "$1" 2>/dev/null | grep -oE 'gfx[0-9][0-9a-zA-Z:+.-]*' | head -1
  fi
}

census_of() {
  if [ -n "$OBJDUMP" ]; then
    "$OBJDUMP" -d --no-show-raw-insn "$1" 2>/dev/null | awk "$CENSUS_AWK"
  fi
}

print_table_header() {
  printf '%-44s %5s %5s %5s %5s %4s %8s %8s %8s %5s %5s %6s %5s %6s  %s\n' \
    "kernel" "vgpr" "spill" "sgpr" "agpr" "wave" "groupseg" "privseg" \
    "kernarg" "wgmax" "ds_*" "branch" "s_bar" "scratch" "verdict"
}

# extract_cos <file> <workdir> -- appends code-object paths to $CO_LIST
extract_cos() {
  local f="$1" w="$2" t co d
  if [ -n "$EXTRACT" ]; then
    mkdir -p "$w"
    "$EXTRACT" -d "$w" "$f" >/dev/null 2>&1 || "$EXTRACT" -o "$w" "$f" >/dev/null 2>&1 || true
    if ! find "$w" -type f -size +0 -print 2>/dev/null | grep -q .; then
      (cd "$w" && "$EXTRACT" "$f" >/dev/null 2>&1) || true
    fi
    while IFS= read -r -d '' co; do
      case "$(basename "$co")" in $TARGET_GLOB) printf '%s\n' "$co" >> "$CO_LIST" ;; esac
    done < <(find "$w" -type f -size +0 -print0 2>/dev/null)
    if [ -s "$CO_LIST" ]; then return 0; fi
  fi
  if [ -n "$BUNDLER" ]; then
    for t in hipo ho; do
      d="$("$BUNDLER" -type=$t -inputs="$f" -list 2>/dev/null | grep -E '^hip' | head -50 || true)"
      [ -n "$d" ] || continue
      for t in $d; do
        case "$t" in $TARGET_GLOB) ;; *) continue ;; esac
        co="$w/$(printf '%s' "$t" | tr -c 'A-Za-z0-9_.-' '_').co"
        if "$BUNDLER" -type=$t -inputs="$f" -outputs="$co" -targets="$t" -unbundle \
            >/dev/null 2>&1 && [ -s "$co" ]; then
          printf '%s\n' "$co" >> "$CO_LIST"
        fi
      done
      [ -s "$CO_LIST" ] && return 0
    done
  fi
  return 1
}

# analyze_code_object <co> <source-file-label>
analyze_code_object() {
  local co="$1" src="$2" arch allsyms v
  arch="$(arch_of "$co")"
  [ -n "$arch" ] || arch="unknown"

  : > "$META"
  if [ -n "$READELF" ]; then meta_of "$co" > "$META"; fi

  if [ "$MODE" = "list" ]; then
    printf '%s (%s):\n' "$src" "$arch"
    if [ -s "$META" ]; then
      awk -F'\t' '{printf "    %s\n", $1}' "$META"
      LIST_EMITTED=$((LIST_EMITTED + $(awk 'END{print NR}' "$META")))
    else
      printf '    (no kernel metadata; llvm-readelf could not parse this)\n'
    fi
    return 0
  fi

  census_of "$co" > "$CENSUS"

  if [ ! -s "$META" ] && [ ! -s "$CENSUS" ]; then
    warn "no analyzable content in $co (tools missing or not a code object)"
    return 1
  fi

  allsyms=$ALLSYMS
  [ ! -s "$META" ] && allsyms=1      # metadata unavailable: keep the census

  if [ "$MODE" = "table" ]; then
    if [ "$co" = "$src" ]; then
      printf '\n=== %s (%s) ===\n' "$src" "$arch"
    else
      printf '\n=== %s -> %s (%s) ===\n' "$src" "$co" "$arch"
    fi
    print_table_header
  fi

  awk -F'\t' -v MODE="$MODE" -v SRC="$src" -v ARCH="$arch" -v VGPR_MAX="$VGPR_MAX" \
      -v SPILL_MAX="$SPILL_MAX" -v WAVE_REQ="$WAVE_REQ" -v ALLSYMS="$allsyms" \
      "$JOIN_AWK" "$CENSUS" "$META" > "$ROWS"

  v=0
  if grep -q '^__VIOLATIONS__' "$ROWS" 2>/dev/null; then
    v="$(grep '^__VIOLATIONS__' "$ROWS" | cut -f2)"
    grep -v '^__VIOLATIONS__' "$ROWS" > "$ROWS.n" && mv "$ROWS.n" "$ROWS"
  fi
  VIOL=$((VIOL + ${v:-0}))
  n=$(grep -cve '^$' "$ROWS" 2>/dev/null || echo 0)
  ROWS_EMITTED=$((ROWS_EMITTED + n))
  cat "$ROWS"
  return 0
}

# process_file <file-or-dir>
process_file() {
  local f="$1" n w
  if [ -d "$f" ]; then
    : > "$CO_LIST"
    find "$f" -maxdepth 1 -type f \( -name '*.so' -o -name '*.hsaco' -o -name '*.co' \) \
      | sort > "$CO_LIST"
    if [ ! -s "$CO_LIST" ]; then
      warn "no .so/.hsaco/.co directly under $f"
      FILES_FAILED=$((FILES_FAILED + 1))
      return
    fi
    while IFS= read -r f; do process_file "$f"; done < "$CO_LIST"
    return
  fi
  if [ ! -f "$f" ]; then
    warn "not found: $f"
    FILES_FAILED=$((FILES_FAILED + 1))
    return
  fi

  # the input may already be a raw code object
  n=0
  if [ -n "$READELF" ]; then
    n="$("$READELF" --notes "$f" 2>/dev/null | awk "$META_AWK" | grep -c . || true)"
  fi
  case "$f" in
    *.co|*.hsaco) n=1 ;;   # trust the extension; readelf absence should not skip it
  esac
  if [ "${n:-0}" -gt 0 ]; then
    analyze_code_object "$f" "$f" || FILES_FAILED=$((FILES_FAILED + 1))
    return
  fi

  # embedded fatbin -> extract
  w="$TMPD/x$(printf '%s' "$f" | tr -c 'A-Za-z0-9' '_')"
  : > "$CO_LIST"
  if extract_cos "$f" "$w" && [ -s "$CO_LIST" ]; then
    while IFS= read -r co; do
      analyze_code_object "$co" "$f" || FILES_FAILED=$((FILES_FAILED + 1))
    done < "$CO_LIST"
  else
    warn "no extractable device code in $f"
    warn "  need rocm-obj-extract or clang-offload-bundler, or pass a .hsaco/.co"
    FILES_FAILED=$((FILES_FAILED + 1))
  fi
}

# ---------------------------------------------------------------- main

for f in "${FILES[@]}"; do
  process_file "$f"
done

if [ "$ROWS_EMITTED" -eq 0 ] && [ "$LIST_EMITTED" -eq 0 ]; then
  warn "nothing analyzed (no tools, no device code, or bad input paths)"
  exit 2
fi
if [ "$STRICT" -eq 1 ] && [ "$VIOL" -gt 0 ]; then
  warn "$VIOL budget verdict(s) tripped (--strict)"
  exit 1
fi
exit 0
