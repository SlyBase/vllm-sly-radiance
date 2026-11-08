# Disk KV-cache tier: end-to-end test harness

`ci/test-disk-tier.sh` proves the re-anchored disk-mode patches (see
[`../CHANGELOG.md`](../CHANGELOG.md) → `## [Unreleased]`) actually offload to disk and survive a
restart. It runs on the inference host against the production image, on a throwaway container that
does **not** touch the serving RAM-tier model. See `AGENTS.md` (`Opt-in attention back ends`) for
why an opt-in back end needs this: a GPU start finds a broken tier too late.

## What it checks

The disk tier is a two-level cache: a small RAM tier for the hot tail of a request, and an fs
secondary tier (`--kv-offloading-size 16`, `native` backend, `TieringOffloadingSpec` with a single
`fs` tier rooted at `/kvcache/blocks`) for everything the RAM tier evicts. The test drives a set of
prompts of varying length through eight phases:

1. **Bring up.** Start the container with the prelude (the common patches, as the production
   template applies them, then the seven disk additions). Instrumentation and mixed-hit are FATAL —
   a failure aborts the prelude and the container exits before health, which the poll detects. The
   rest warn-and-continue, so the test also greps the log for any `WARNING: … did not apply` for the
   disk set.
2. **Baseline + populate.** Fire the long prompt once, measure time-to-first-token, grab a 64-token
   greedy response, and record the block count. Also capture the fresh (no-offload) references for
   the accuracy gate (below).
3. **Survival.** Stop the container, start it again with the **same** `/kvcache` mount, and repeat.
4. **Accuracy gate.** For every prompt (short, medium, long), the cached greedy response after the
   restart must be byte-identical to its fresh reference from PHASE 2. This is the AGENTS.md accuracy
   gate — the disk tier serves tokens from blocks; any difference means the round-trip was lossy.
5. **Throughput.** Measure decode throughput (tok/s) on a decode-bound request and report it against
   a soft floor. The disk tier does not change warm decode, so this is a sanity signal, not a hard
   gate (a single sample is hardware-dependent).
6. **Partial eviction.** Run a distinct-content prompt after the disk already holds the baseline's
   blocks. Filling the disk forces the fs tier to evict some baseline blocks and serve the new prompt
   from a mix of cached and recomputed blocks; the cached response must be byte-identical to its
   fresh reference. This exercises the eviction + recompute path.
7. **Durability.** A second restart must leave the baseline blocks surviving and serving byte-identical
   output — the disk tier is meant to persist across container lifecycles, not just one.
8. **Verdict.** Prints all signals and exits non-zero if any correctness check recorded a FAIL.

A passing run satisfies **all** of:

- **Bring-up:** no `WARNING: <disk-patch> did not apply` in the log. Any of the seven missing means
  the tier is silently a no-op (it looks enabled but writes nothing to disk).
- **Baseline response is coherent English.** A no-op tier still produces a correct response (the
  model just never offloads), so correctness alone does not prove offloading — but an incoherent
  response means the patches broke something.
- **Block count grew** after the first prompt (disk blocks were written).
- **TTFT after restart < before.** The first start compiles the CUDA graph fresh and starts with a
  smaller KV pool; the second start reuses the compiled graph and the warm disk blocks. If the tier
  works, the second prefill loads the already-written blocks instead of recomputing them, so it is
  faster. The speedup is disk-bandwidth bound (see "Speedup" above): on the fast SSD it is ~5×, on
  a spinning disk much smaller. The criterion is "restart beats a fresh prefill", not a fixed
  multiplier — any speedup > 1× means the tier offloaded and is helping.
- **Response identical before/after.** The greedy completion must be byte-identical across the
  restart. The disk tier serves tokens from blocks; if the response changed, the blocks did not
  round-trip correctly.
- **Accuracy gate (per prompt).** Short, medium and long prompts each run fresh (captured in
  PHASE 2) and cached (after restart); every cached greedy response must be byte-identical to its
  fresh reference. The long prompt's reference is the baseline response; short/medium are extra
  gates across a range of cache states.
- **Throughput.** Decode tok/s is reported against a soft floor (`DISK_TIER_MIN_TPS`, default 8).
  A single sample is noisy, so a below-floor number warns rather than fails.
- **Partial eviction.** A distinct prompt run after the disk is already populated forces the fs tier
  to evict some baseline blocks; its cached response must be byte-identical to its fresh reference
  (eviction + recompute is lossless).
- **Durability.** A second restart must leave the baseline blocks surviving and serving byte-identical
  output.

## Interpreting the output

```
TTFT speedup (before/after restart): 4.7x
disk blocks: 120 -> 134 -> 134
response identical after restart: PASS
ok:     accuracy gate: short cached == fresh
ok:     accuracy gate: medium cached == fresh
ok:     decode throughput 11.4 >= floor 8.0
ok:     partial eviction: evict prompt cached == fresh
ok:     durability: response identical after a SECOND restart
```

The harness records each correctness check as `ok:` or `FAIL:` and exits non-zero if any check
failed, so a red line in the log is a hard failure. The throughput line warns (`WARN:`) rather than
fails — a single sample is hardware-dependent.

- **Speedup.** The long prompt is ~40k tokens of pure prefill; TTFT ≈ prefill cost. The baseline
  recomputes all 40k tokens from scratch; the restart loads the already-written blocks and skips the
  recomputation. The speedup is disk-bandwidth bound: on the fast SSD this was measured on it was
  ~5× (the blocks read in well under a second). On a spinning disk the same blocks take far longer
  to read, so expect a much smaller speedup — the signal is "restart beats a fresh prefill", not a
  fixed multiplier. A much larger number usually means the baseline was polluted (a stale block was
  loaded — see below); a number near 1× means the tier did not offload.
- **Block count.** `blocks` are the fs-tier storage units. They should be > 0 after the prompt and
  grow with prompt length. They persist across runs (bind mount), which is both the feature (survival)
  and the trap (see below).
- **Response.** The reasoning model returns `content: null`; the answer lives in `reasoning`, so the
  harness compares `reasoning` when `content` is null. Identical = the round-trip is lossless.

## Why PHASE 2 clears the disk blocks

The block files persist across runs (they live on the host under `KVCACHE_DIR/blocks`). If a
previous run left blocks there, the baseline prompt would load them and the "fresh prefill" would
just be another disk load — which never beats itself. The harness `sudo rm -rf`s the blocks before
the baseline so the number is a true cold prefill. GPU RAM is already empty because the container
was started fresh in PHASE 1.

## Running it

```bash
# on the inference host (pve.local)
scp ci/test-disk-tier.sh pve.local:~/test-disk-tier.sh
ssh pve.local 'bash ~/test-disk-tier.sh'
```

The first start compiles fresh and shows a smaller KV pool; the harness accounts for this by
comparing the *second* start (warm graph + warm disk) against the *first* (cold). The health poll
waits up to `HEALTH_TIMEOUT` (default 1800 s) for a cold compile.

Overrides (all via env): `IMAGE`, `MODEL`, `DRAFTER`, `PATCHES_DIR`, `CHAT_TEMPLATE`, `KVCACHE_DIR`,
`PORT`, `LOGDIR`, `HEALTH_TIMEOUT`, `DISK_TIER_MIN_TPS`. `PATCHES_DIR` defaults to the repo root so `/patches/kv-cache/`
is the disk set; `KVCACHE_DIR` defaults to `/var/tmp/disk-tier-test`; `DISK_TIER_MIN_TPS` sets the
soft decode-throughput floor (default 8 tok/s).

## Troubleshooting

- **`FATAL: image … not present; pull it first.`** The image is not on this host. `docker pull` it.
- **`FATAL: a disk-tier-test/sly container is running.`** Stop it first; the GPU must be free.
- **`FATAL: container exited before becoming healthy`** — the prelude failed. A FATAL patch
  (instrumentation, mixed-hit) aborts the container; the log is dumped to `$LOGDIR/early.log`. A
  non-FATAL patch that did not apply shows as `WARNING: … did not apply` and is caught in bring-up.
- **`FATAL: not healthy within …s`** — a cold compile on a busy GPU. Increase `HEALTH_TIMEOUT` or
  run when the GPU is idle.
- **Bring-up fails (a disk patch did not apply)** — the branch is stale. Re-pull the branch; the
  seven patches are anchored to vLLM 0.32.0.
- **Speedup near 1× but blocks grew** — the tier offloaded but did not help on the second start.
  Usually the baseline was polluted by stale blocks; the harness clears them, so this means the
  prompt is too short to fill the tier or the fs backend is not engaged.
