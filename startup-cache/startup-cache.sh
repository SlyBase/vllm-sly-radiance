# startup-cache/startup-cache.sh -- build-cache identity, JIT-cache mounts and stale-tree pruning, for any launcher.
#
# SOURCE it once, after IMAGE is set and before anything is built or keyed:
#
#   STARTUP_CACHE=${STARTUP_CACHE:-$(dirname "$(realpath -m "$0")")/startup-cache}
#   . "$STARTUP_CACHE/startup-cache.sh"
#
# and add "${STARTUP_CACHE_RUN_ARGS[@]}" to the container run command. See README.md for the
# optional container-side piece (the build-once .so, hip-so-cache.sh).
#
# The one rule: every cache key is a fingerprint of every upstream input the artifact was built
# from, so any change MISSES (always safe: the slow path rebuilds) instead of silently reusing a
# stale build. This file is the only definition of the image/arch half of that fingerprint.
#
# Needs:  IMAGE. RUNTIME is auto-detected if unset, and the image is pulled if missing.
#         die() is used if the launcher has one.
# Sets:
#   IMG_TAG IMG_ID IMG_KEY  image tag (for humans) + image ID (the invalidation signal). A re-pull
#                           under the same tag keeps the tag but changes the ID. If inspect fails
#                           the key is the tag alone -- a different key, so a miss, never a false hit.
#   ARCH                    GPU target the kernels are compiled for, read from the KFD topology.
#                           ARCH=<gfx...> overrides.
#   STARTUP_CACHE_KEY       "$IMG_KEY-$ARCH" -- put it in every build-cache key or directory name.
#   STARTUP_CACHE_RUN_ARGS  container run args: mounts this dir at /startup-cache, passes the key
#                           and arch (LAUNCH_IMG_KEY, LAUNCH_GFX_ARCH).
# JIT caches under the container's HOME (call once the launcher's cache dir is set):
#   startup_cache_jit_mounts <dir>  persists comgr, tvm-ffi and tilelang under <dir>, which should
#                                   already carry STARTUP_CACHE_KEY (the launcher's /cache dir).
#   STARTUP_CACHE_CONTAINER_HOME    the image's HOME (default /root, the radiance image).
# Stale-tree pruning (call once the launcher's cache dir is set):
#   startup_cache_reap <dir> <prefix>  runs cache-reap.sh over the sibling trees "<prefix>"*; <dir>
#                                   is the live tree and must sit under <prefix>, or nothing runs.

STARTUP_CACHE_DIR="$(dirname "$(realpath -m "${BASH_SOURCE[0]}")")"
_sc_die() {
  if declare -F die >/dev/null; then die "$@"; fi
  echo "[startup-cache] ERROR: $1" >&2; shift
  for l in "$@"; do echo "  $l" >&2; done
  exit 1
}
[ -n "${IMAGE:-}" ] || _sc_die "startup-cache.sh needs IMAGE set before it is sourced"
if [ -z "${RUNTIME:-}" ]; then
  if   command -v podman >/dev/null 2>&1; then RUNTIME=podman
  elif command -v docker >/dev/null 2>&1; then RUNTIME=docker
  else _sc_die "no container runtime found (podman or docker)"; fi
fi

# ---------------------------------------------------------------- image identity
# The image must be present for its ID to be read. Without this, a launcher whose `run` does the
# pull would key its first boot on the tag alone and its second on the ID: two cold compiles.
if ! "$RUNTIME" image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[startup-cache] pulling $IMAGE (its ID keys every build cache)"
  "$RUNTIME" pull "$IMAGE" >&2 || echo "[startup-cache] pull failed -- keying on the tag alone this boot"
fi
# shellcheck disable=SC2001
IMG_TAG=$(echo "$IMAGE" | sed 's|.*/||; s|[^A-Za-z0-9._-]|-|g')
IMG_ID=$("$RUNTIME" image inspect --format '{{.Id}}' "$IMAGE" 2>/dev/null | sed 's/^sha256://' | cut -c1-12) || true
if [ -n "$IMG_ID" ]; then IMG_KEY="$IMG_TAG-$IMG_ID"; else IMG_KEY="$IMG_TAG"; fi

# ---------------------------------------------------------------- GPU arch
if [ -z "${ARCH:-}" ]; then
  # gfx_target_version is major*10000 + minor*100 + stepping (120001 -> gfx1201); CPU nodes
  # report 0. A host with a second GPU family (an iGPU) is narrowed to gfx12; anything still
  # ambiguous has to be named.
  _sc_archs=$(awk '$1 == "gfx_target_version" && $2 > 0 {
                     v = $2; printf "gfx%d%x%x\n", int(v / 10000), int(v / 100) % 100, v % 100 }' \
                /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null | sort -u) || true
  if [ "$(printf '%s\n' "$_sc_archs" | grep -c .)" -gt 1 ]; then
    _sc_archs=$(printf '%s\n' "$_sc_archs" | grep '^gfx12') || true
  fi
  case "$(printf '%s\n' "$_sc_archs" | grep -c .)" in
    1) ARCH=$_sc_archs ;;
    0) _sc_die "could not read the GPU arch from /sys/class/kfd/kfd/topology" \
               "set it by hand: ARCH=gfx1201 (R9700), or ARCH=gfx1200 for the other RDNA4 parts" ;;
    *) _sc_die "more than one gfx12 GPU arch on this host: $(echo $_sc_archs)" \
               "set the one to serve on: ARCH=<gfx...>" ;;
  esac
fi
STARTUP_CACHE_KEY="$IMG_KEY-$ARCH"

STARTUP_CACHE_RUN_ARGS=(-v "$STARTUP_CACHE_DIR":/startup-cache:ro,z
                        -e LAUNCH_IMG_KEY="$IMG_KEY" -e LAUNCH_GFX_ARCH="$ARCH")

# ---------------------------------------------------------------- JIT caches under HOME
# Three JIT caches default to the container's HOME, not to any of the *_CACHE_DIR variables the
# launchers already point at /cache, so a --rm container throws them away on every boot:
#   comgr     ~/.cache/comgr     ROCm code-object cache (GPU code compiled while the engine starts)
#   tvm-ffi   ~/.cache/tvm-ffi   the torch DLPack addon tvm-ffi compiles from C++ on import (~22 s
#                                of every boot, measured -- it was the largest avoidable cost left)
#   tilelang  ~/.tilelang        tilelang's kernel cache
# Bind-mounting them from a dir that already carries STARTUP_CACHE_KEY gives them the same
# invalidation as everything else: a new image or arch gets a fresh, empty set. comgr names its
# entries by content hash and the tvm-ffi addon carries the torch version in its file name, so a
# stale entry cannot be picked up by mistake inside one key either.
startup_cache_jit_mounts() {
  local dir home="${STARTUP_CACHE_CONTAINER_HOME:-/root}"
  dir="$(realpath -m "$1")"              # container runtimes want an absolute host path
  [ -n "${DRY_RUN:-}" ] || mkdir -p "$dir"/{comgr,tvm-ffi,tilelang}   # DRY_RUN creates nothing
  STARTUP_CACHE_RUN_ARGS+=(-v "$dir/comgr":"$home/.cache/comgr":z
                           -v "$dir/tvm-ffi":"$home/.cache/tvm-ffi":z
                           -v "$dir/tilelang":"$home/.tilelang":z)
}

# ---------------------------------------------------------------- stale cache trees
# Each image, arch or flag change gets a new keyed tree, and nothing else removes the old ones.
# This runs cache-reap.sh over the siblings "<prefix>"* before the boot: only when the disk is
# above 65% used, never the live tree, never one launched or read in the last 24 h (the rule is in
# cache-reap.sh's header). It stamps the live tree first, so a tree any launcher is using is
# protected even on a filesystem mounted noatime. A live dir outside <prefix> (a custom CACHE)
# skips pruning: its siblings are not ours to judge. Output goes to the boot log as [cache-reap]
# lines; a DRY_RUN only reports. A slow or failed run never stops the boot.
startup_cache_reap() {
  local live base mode=--apply
  live="$(realpath -m "$1")"
  base="$2"
  case "$base" in              # absolute, keeping a trailing / (it decides what the siblings are)
    /*) ;;
    */) base="$(realpath -m "$base")/";;
    *)  base="$(realpath -m "$base")";;
  esac
  case "$live" in "$base"?*) ;; *)
    echo "[cache-reap] $live is not under $base -- custom cache dir, not pruning" >&2; return 0;;
  esac
  if [ -n "${DRY_RUN:-}" ]; then
    mode=--dry-run
  else
    mkdir -p "$live"
    printf 'ts=%s\nhost=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${HOSTNAME:-}" > "$live/.last-boot-ok" 2>/dev/null || true
  fi
  timeout 20 "$STARTUP_CACHE_DIR/cache-reap.sh" --live "$live" --base "$base" "$mode" 2>&1 \
    | sed 's/^cache-reap: /[cache-reap] /' >&2 || true
}
