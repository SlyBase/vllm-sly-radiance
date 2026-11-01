# startup-cache

Build caches for the container launchers, keyed so that they invalidate by themselves. Every cache
key is a fingerprint of everything its artifact was built from: image **ID** (not tag), GPU arch and,
for the `.so`, source and flags. When any of those changes the key changes and the cache misses.
A miss is always safe: the slow path rebuilds.

Opt-in in `serve-mxfp4.sh`: `STARTUP_CACHE=1` (default `0`, which leaves the generated command
unchanged).

| File | Side | What it does |
|---|---|---|
| `startup-cache.sh` | host, sourced | image ID + arch → `STARTUP_CACHE_KEY`; builds the container run args; persists the JIT caches that live under the container's HOME; prunes stale cache trees |
| `hip-so-cache.sh` | container | compiles a `.hip` to a `.so` once per key |
| `cache-reap.sh` | host | removes stale sibling cache trees when the disk fills; run for you at launch |

## What `STARTUP_CACHE=1` changes in `serve-mxfp4.sh`

1. **The cache dir carries the key.** `$CACHE` becomes `…-<image tag>-<image id>-<arch>`. Today it is
   named by hand (`…-093…`), so a re-pulled image under the same tag silently reuses graphs compiled
   for the old one; now it misses and rebuilds.
2. **Three JIT caches that live under the container's HOME are persisted** instead of being thrown
   away with the `--rm` container: `~/.cache/comgr` (ROCm code objects), `~/.cache/tvm-ffi` (the torch
   DLPack addon tvm-ffi compiles from C++ on import, ~22 s a boot) and `~/.tilelang`.
3. **`radiance_mxfp4_fp8.so` is compiled once per key** (`hip-so-cache.sh`) instead of by `hipcc` on
   every boot.
4. **Stale trees are pruned** (below).

## What it saves

Measured on the author's launcher (one R9700, radiance 0.9.3, Qwen3.8-27B MXFP4 + DFlash2 ×7), by
phase, from log timestamps, so treat the numbers as approximate:

| Cache | Saved per boot |
|---|---|
| comgr + tvm-ffi + tilelang mounts, measured as an A/B with every other cache warm | 189 s → 151 s (almost all of it the tvm-ffi addon) |
| `radiance_mxfp4_fp8.so` (`hip-so-cache.sh`) | ~30 s |

The keyed `$CACHE` itself saves nothing on a second boot (the dir was already persistent); it is
insurance against reusing a stale graph after an image or arch change. A cached boot still spends
fixed cost: Python imports in vLLM's processes, weights (~20–40 s) and loading the compiled graphs.

## Wiring another launcher

**1. Host, once `IMAGE` is set, before anything is built or keyed** (it pulls the image if missing):

```bash
STARTUP_CACHE=${STARTUP_CACHE:-$(dirname "$(realpath -m "$0")")/startup-cache}
. "$STARTUP_CACHE/startup-cache.sh"
```

Put `$STARTUP_CACHE_KEY` into every cache directory or key the launcher owns, and add
`"${STARTUP_CACHE_RUN_ARGS[@]}"` to the `podman run` / `docker run` command. Then, once the launcher's own
cache dir is final:

```bash
startup_cache_jit_mounts "$CACHE"                         # comgr, tvm-ffi, tilelang
startup_cache_reap "$CACHE" "$HOME/.my-launcher-cache-"    # stale sibling trees
```

Set `STARTUP_CACHE_CONTAINER_HOME` if the image's HOME is not `/root`.

**2. Container, for a `.hip` compiled at boot** (optional):

```bash
bash /startup-cache/hip-so-cache.sh foo.hip "$SP"/foo.so -O3 -fPIC -shared $(python3 -m pybind11 --includes)
```

The key is the source, every `#include "..."` next to it (one level), the flag list including the
arch, and the image ID. The cache lives at `${HIP_SO_CACHE:-/cache/hipso}`; a `.so` is published by
rename, so a boot killed mid-build never leaves a truncated file under a key.

## Pruning stale trees

Every new image, card or flag combination gets a new keyed tree, and the old ones stay behind: on the
box this was written on, nine stale trees held 17 GiB. `startup_cache_reap <dir> <prefix>` runs
`cache-reap.sh` over the siblings `<prefix>*` before each boot. The rule, in order:

1. Nothing happens below **65% disk use**. The trigger is space, not age.
2. The live tree is never touched, and neither is a tree with the same flag suffix.
3. A tree with a **stamp** (`.last-boot-ok`) from the last 24 h is kept. Every launch stamps its tree.
4. Otherwise a tree goes only if **no file in it was read in the last 24 h** (the stamp is left out of
   that check: reading it refreshes its own access time).
5. Whole trees, oldest first, at most 8 a run, stopping as soon as the disk is back under 65%.

Getting it wrong costs one slow boot (everything recompiles), never a wrong answer. Its result is in
the boot log as `[cache-reap]` lines. A custom `CACHE` outside the prefix is never pruned, and neither
are its neighbours. To see what it would do: `startup-cache/cache-reap.sh --live <tree> --base <prefix>`
(a dry run unless you add `--apply`). Knobs: `CACHE_REAP_TARGET_PCT` (65), `CACHE_REAP_MIN_AGE_HOURS`
(24), `CACHE_REAP_MAX_DELETE` (8).

## Scope

Any vLLM launcher that runs in a container (podman/docker) on AMD/ROCm. Containers only (the identity
is the image ID); AMD only (the arch comes from `/sys/class/kfd`; `ARCH=<gfx...>` overrides it); needs
bash and GNU `realpath`.

## Limits

A key only covers inputs it can see. If a build step silently does nothing, the key still says
"configured" and not "happened". The libr4d builds under `~/.cache/radiance-libr4d/` are not covered.
