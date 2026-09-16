#!/usr/bin/env bash
# CI dry run of the Dockerfile's patch loop against the pinned upstream sources, without ROCm,
# without building a wheel and without a GPU.
#
# The patches only ever rewrite Python source files under `sysconfig.get_paths()["purelib"]`
# (see _patchlib.apply), so a venv whose site-packages holds the *source trees* of vllm, aiter,
# transformers, torch/_dynamo and triton at the pinned tags is enough to prove that every
# anchor still matches. The loop itself comes from the Dockerfile (ci/patch_list.py), in the
# Dockerfile's order, so anchor dependencies between patches are exercised exactly as in the
# image build. Two passes: pass 1 must apply every patch (OK), pass 2 must be all NOOP
# (idempotence, which the image build relies on when a stage is re-run).
#
# Usage: ci/patch_dryrun.sh            (work dir .dryrun/, reused between runs)
#        DRYRUN_WORK=/tmp/x ci/patch_dryrun.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
WORK=${DRYRUN_WORK:-$ROOT/.dryrun}
PY=${PYTHON:-python3}
eval "$("$PY" "$ROOT/ci/read_pins.py" --shell)"
mkdir -p "$WORK/src"

# --- pinned source trees (sparse, blob-filtered: only the paths the patches can touch) ---
sparse() { # <name> <url> <ref> <sparse path>...
  local name=$1 url=$2 ref=$3 dir
  shift 3
  dir="$WORK/src/$name"
  if [ -f "$dir/.ok" ] && [ "$(cat "$dir/.ok")" = "$ref" ]; then
    echo "== $name @ $ref (cached)"; return
  fi
  rm -rf "$dir"
  echo "== $name @ $ref"
  git clone --quiet --filter=blob:none --no-checkout --depth 1 --branch "$ref" "$url" "$dir"
  git -C "$dir" sparse-checkout set --no-cone "$@"
  git -C "$dir" checkout --quiet
  echo "$ref" > "$dir/.ok"
}
sparse vllm         https://github.com/vllm-project/vllm.git      "v$VLLM_VERSION"         '/vllm/'
sparse aiter        https://github.com/ROCm/aiter.git             "v$AITER_VERSION"        '/aiter/'
sparse transformers https://github.com/huggingface/transformers.git "v$TRANSFORMERS_VERSION" '/src/transformers/'
sparse pytorch      https://github.com/pytorch/pytorch.git        "v$TORCH_VERSION"        '/torch/_dynamo/'
sparse triton       https://github.com/triton-lang/triton.git     "v$TRITON_VERSION"       '/python/triton/' '/third_party/amd/backend/'

# --- fresh venv whose purelib is assembled from those trees (the patches resolve their targets
#     through sysconfig, exactly like in the image's /opt/vllm venv) ---
rm -rf "$WORK/venv"
"$PY" -m venv "$WORK/venv"
VPY="$WORK/venv/bin/python"
SP=$("$VPY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
echo "== purelib $SP"
cp -r "$WORK/src/vllm/vllm"                   "$SP/vllm"
cp -r "$WORK/src/aiter/aiter"                 "$SP/aiter"
cp -r "$WORK/src/transformers/src/transformers" "$SP/transformers"
mkdir -p "$SP/torch" && cp -r "$WORK/src/pytorch/torch/_dynamo" "$SP/torch/_dynamo"
cp -r "$WORK/src/triton/python/triton"        "$SP/triton"
mkdir -p "$SP/triton/backends/amd" && cp -r "$WORK/src/triton/third_party/amd/backend/." "$SP/triton/backends/amd/"
# the assemble stage COPYs these before the loop; mirrored so a patch may anchor on them
cp "$ROOT"/radiance_*.py "$ROOT"/sly/mxfp4/radiance_*.py "$SP/"
mkdir -p "$SP/aiter/ops/triton/configs" && cp -r "$ROOT/sly/mxfp4-configs/." "$SP/aiter/ops/triton/configs/"

# --- skip allowlist ---
declare -A SKIP
while read -r name reason; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  SKIP["$name"]="$reason"
done < "$ROOT/ci/patch_dryrun_skip.txt"

mapfile -t PATCHES < <("$PY" "$ROOT/ci/patch_list.py" --lines)
for k in "${!SKIP[@]}"; do
  printf '%s\n' "${PATCHES[@]}" | grep -qx "$k" || { echo "FAIL: skip entry '$k' is not in the Dockerfile loop"; exit 1; }
done

run_pass() { # <pass no> -> log file
  local pass=$1 log="$WORK/pass$1.log"
  : > "$log"
  for p in "${PATCHES[@]}"; do
    if [[ -n "${SKIP[$p]:-}" ]]; then
      echo "SKIP $p: ${SKIP[$p]}" | tee -a "$log"; continue
    fi
    echo "== pass $pass: $p ==" | tee -a "$log"
    (cd "$ROOT" && PYTHONPATH="$ROOT" "$VPY" "$p.py") 2>&1 | tee -a "$log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "FAIL: $p exited non-zero in pass $pass"; exit 1; }
  done
}

run_pass 1
if grep -q '^  NOOP ' "$WORK/pass1.log"; then
  echo "::warning::pass 1 had NOOP hunks -- a sentinel already matches the pinned upstream (patch obsolete or absorbed upstream?):"
  grep '^  NOOP ' "$WORK/pass1.log"
fi
run_pass 2
if grep -q '^  OK ' "$WORK/pass2.log"; then
  echo "FAIL: pass 2 applied hunks again (patch not idempotent):"; grep '^  OK ' "$WORK/pass2.log"; exit 1
fi

# same final check as the Dockerfile loop: every radiance module still parses
"$VPY" -c "import ast,glob,sys; fs=glob.glob('$SP/radiance_*.py'); [ast.parse(open(f).read()) for f in fs]; print(f'radiance modules parse OK ({len(fs)})')"
echo "patch dry run OK: $(grep -c '^  OK ' "$WORK/pass1.log") hunks applied, pass 2 all NOOP, $(grep -c '^SKIP ' "$WORK/pass1.log" || true) skipped"
