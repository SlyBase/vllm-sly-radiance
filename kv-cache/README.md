# Opt-in KV-cache offload (off by default)

Adds a second-tier prefix cache behind the GPU prefix cache for **Qwen3.8-27B**
(Gated-DeltaNet hybrid, MXFP4 weights, FP8 KV) on one **AMD Radeon AI PRO R9700**
(gfx1201 / RDNA4, 32 GB, TP=1). Served by vLLM 0.27.1 / radiance 0.9.3.

This is **opt-in and off by default.** With `KVCACHE` unset (or `off`), `serve-mxfp4.sh`
generates a serve command that is byte-identical to the unmodified script — nothing is
mounted, patched, or passed to the engine.

## The knob

`serve-mxfp4.sh` reads one variable (and two sizing options):

```bash
KVCACHE=off    # default; also the state when KVCACHE is unset. Stock behaviour.
KVCACHE=ram    # GPU -> RAM tier in /dev/shm + the 6 "always" behavioural patches.
KVCACHE=disk   # GPU -> RAM -> disk + the full 15-patch instrumented set.

KVCACHE_RAM_GIB=16   # size of the RAM tier, GiB (KVCACHE=ram or disk)
KVCACHE_DISK=/kvcache   # host path of a dedicated filesystem (KVCACHE=disk only)
```

## The two options

### GPU -> RAM (the RAM option)

A RAM tier in `/dev/shm` behind the GPU prefix cache, plus the 6 "always" behavioural
patches (mixed-hit, eagle-groups, mamba-stride, reconcile-reask, swa-align/touch,
align-last-block). Needs `/dev/shm` big enough for `KVCACHE_RAM_GIB`. Runs with the
stock LRU policy and prompt-only offload (vLLM defaults).

### GPU -> RAM -> disk

The same RAM tier with a filesystem secondary tier behind it, plus the full 15-patch
instrumented set. Needs `KVCACHE_DISK` (a dedicated filesystem, mounted read-write into
the container) **and `kvcache-reap.sh`** — the fs tier writes and never deletes on its own.
Give the tier its own filesystem: the reaper keeps that filesystem at or below 70% by deleting
the oldest blocks, so anything else stored on it is paid for with cache.

```bash
sudo cp kv-cache/kvcache-reap.sh /usr/local/bin/
sudo cp kv-cache/kvcache-reap.{service,timer} /etc/systemd/system/
sudoedit /etc/systemd/system/kvcache-reap.service     # set KVCACHE_ROOT=<KVCACHE_DISK>/blocks
sudo systemctl daemon-reload && sudo systemctl enable --now kvcache-reap.timer
```

## vLLM 0.30 and later (image 0.4.0+)

The patch set was written against vLLM 0.27.1. vLLM 0.30 rebuilt the offload scheduler and
took over part of what the patches did, so on such an image a patch reports `SKIP` for the
hunks that have no target any more and applies the rest. `SKIP` is not a failure; the
launcher's fatal patches exit 0 on it. What each patch does there:

| patch | on vLLM >= 0.30 | why |
|---|---|---|
| mixed-hit | kill switch applies; lookback + assertion **skipped** | `update_state_after_alloc` loads from the first fresh block at or above the request's local boundary, and the boundary assertion is gone -- the load never asks for chunks the lookup did not confirm |
| eagle-groups | applies (new shape) | upstream annotates on every grouping path, but reaches the positional "last registered layer" rule only for DeepSeek V4 and otherwise treats *every* group as non-draft; the knob re-enables the rule, and the boot line `draft attention groups [8] detected` stays the self-check |
| mamba-stride | **skipped** | Mamba "align" groups are stored per retained checkpoint through the manager's boundary hand-off, not per chunk; `--prefix-cache-retention-interval` (default 0) keeps only the replay boundary and shared-prefix junctions, on the GPU and in the tier. A positive multiple of the block size adds periodic checkpoints -- the stride, without the dead zone |
| reconcile-reask | applies | the `hit_diverged` fallback is unchanged upstream and still never re-asks the tier |
| swa-align / touch | swa-align **skipped**, touch-all applies | store reachability is judged on the absolute grid with an explicit store horizon (`reachable_block_mask`); `_touch` is unchanged upstream |
| align-last-block | applies | `_mamba_block_aligned_split` is unchanged upstream |
| eagle-replay-tail (new, 0.30+ only) | applies | with an eagle drafter the scheduler commits the Mamba boundary state one block before the aligned prompt end, but the drafter group's retained tail ends *at* the aligned end (plus the never-storable partial block), so the lookup finds no complete window there and falls back to the previous turn's junction. The patch keeps one more drafter block per reachable boundary (`RADIANCE_SWA_EAGLE_REPLAY_TAIL=1`, a superset of the stock mask) |
| boundary-trace (new, 0.30+ only) | applies, silent | `RADIANCE_OFFLOAD_BOUNDARY_TRACE=1` logs every Mamba boundary-state hand-off and its fate (stored / dropped and why); diagnostic only |

vLLM 0.31 (the pin on `main` since 1.1.0) differs again in three places, all carried: the
annotator call on the uniform path is multi-line with `use_trailing_layer_fallback`
(eagle-groups, third shape), `_touch` is gone because recency is tracked per request (touch-all
skips), and `_make_boundary_key` takes the request context (boundary-trace, second shape).

Each skip is decided by a marker string the newer tree has and the older one does not
(`_patchlib.skip_if_upstream`), so the same files keep applying on a 0.27.1 image.
`ci/patch_dryrun.sh` runs the ram-mode patch list against the pinned vLLM after the Dockerfile
loop, so a vLLM bump that breaks an anchor fails `ci` instead of printing a WARNING at boot.
`RADIANCE_MAMBA_STORE_STRIDE` and `RADIANCE_SWA_STORE_MAMBA_ALIGN` are read only by the
skipped hunks and do nothing on 0.30+. `kvwatch.py` and `turnbench.py` read vLLM's stock
metrics and work unchanged; set `TIERBENCH_API_KEY` when the server runs with `VLLM_API_KEY`.

Knobs the two new patches add (the launcher sets them for `KVCACHE=ram|disk`):

| knob | default | launcher | what |
|---|---|---|---|
| `RADIANCE_SWA_EAGLE_REPLAY_TAIL` | unset = upstream mask | `1` | one more sliding-window block per reachable boundary under eagle; without it the tier hit lands two turns back (table below) |
| `RADIANCE_OFFLOAD_BOUNDARY_TRACE` | unset = silent | passed through, `0` | one INFO line per Mamba boundary-state hand-off; diagnostic |

Measured 2026-10-08, one R9700, radiance 1.0.0 (vLLM 0.30.0), production launch + vision,
`KVCACHE=ram` (16 GiB), turnbench 3 sessions x 7 turns, step 18k (~398k tokens together against a
331k-token GPU pool, so turns 6-7 come back from the tier):

| | tier hit lands at | tokens recomputed per tier turn | tier turn wall |
|---|---|--:|--:|
| 0.30 stock lookup (eagle group annotated) | prompt end of the turn *before* the previous one | 37-56k | 29-41 s |
| + eagle-replay-tail | one block before the previous turn's prompt end (= where a GPU hit lands) | 20-21k | 17-20 s |
| eagle group NOT annotated (upstream default) | 4-5 turns back | 39-91k | 32-59 s |

Repeated on image 1.1.1 (vLLM 0.31.0, same launch, second start, KV pool 334,651): tier hits
at 94,160 / 112,640 / 117,920 / 114,400, 19-21k tokens recomputed per tier turn, tier turn wall
16.7-19.8 s against 19.6-22.1 s for the GPU-hit turns of the same run; cold twins of the tier turns
first divergent token 63 / 19 / 2 / none, max |dlogprob| 0.13-0.18; HEALTH pass; no engine errors.

The eagle annotation therefore stays on. Cold-twin comparison of the run with eagle-replay-tail
(bf16 SSM cache, so not the bit-exact setting of the fp32 gate above): the four tier turns
C6/A7/B7/C7 against their cold twins -- first divergent token 52 / none / 3 / none (A7 and C7
token-identical), max |dlogprob| 0.24 / 0.33 / 0.06 / 0.16. The GPU-hit turns of the same run
drift the same way (first divergent token 0-276, max |dlogprob| 0.04-0.43), and the run without
the patch gave the same positions for the tier turns (52 / none / 3 / 1), i.e. the tier adds no
drift of its own; bit-exactness cannot be shown on this build because the GPU path is not exact
either. HEALTH passes in both runs: no empty replies, no loops, no cut-offs in either phase.
Single-stream decode with the tier on is unchanged (code 127-137 tok/s, prose 59 tok/s, same
prompts as without).

## What it does on this build

Measured with `./serve-tp1.sh` defaults (fp16 ssm cache, MAXSEQS 3, 220k context, 248,235-token
GPU pool), changing only `KVCACHE` (`KVCACHE_RAM_GIB=16`). The workload is `turnbench.py`: three
agent-style sessions growing to ~110k tokens each, served round-robin, so together they outgrow
the GPU pool from turn 4 on. Times are whole requests (prefill plus 320 generated tokens); "no
cache" is the same request replayed with a fresh cache salt. Brackets: tokens served from the tier.

| turn | prompt | no cache | `KVCACHE=ram` | `KVCACHE=disk` |
|---|--:|--:|--:|--:|
| B4 | 65,981 | 30.7 s | 12.8 s (49k) | 12.8 s (49k) |
| A5 | 80,895 | 37.8 s | 15.1 s (63k) | 15.2 s (63k) |
| C5 | 81,756 | 38.8 s | 13.2 s (67k) | 13.3 s (67k) |
| A6 | 95,502 | 46.7 s | 15.5 s (77k) | 15.5 s (77k) |
| B6 | 95,278 | 46.9 s | 21.6 s (67k) | 16.6 s (77k) |
| C6 | 96,116 | 47.2 s | 27.9 s (53k) | 18.0 s (77k) |
| A7 | 110,063 | 51.1 s | 44.2 s (21k) | 14.9 s (92k) |
| B7 | 108,878 | 51.8 s | 42.4 s (28k) | 16.5 s (92k) |
| C7 | 110,553 | 52.9 s | 49.6 s (11k) | 15.5 s (95k) |

A 16 GiB RAM tier carries the sessions until they outgrow it (turn 6 here); the disk tier keeps
every later turn near 15 s. All 21 turns served cleanly in both modes.

## Correctness gate

`turnbench.py` runs 3 sessions x 7 turns and compares every cached turn
token-for-token and logprob-for-logprob against a cold twin. It is the gate for
bit-identical KV-cache serving (the focus of this patch set; peak speed was not chased).
It compares bit-for-bit, so run it with an fp32 ssm cache.

## Watching it work

`watch -n 5 python3 kv-cache/kvwatch.py` shows GPU and offload-tier hit rates (lifetime and
since the last refresh), bytes the tier loaded and stored, and deferred lookups. It reads only
metrics stock vLLM exports, from `http://127.0.0.1:$PORT/metrics` (`KVWATCH_METRICS` overrides).
Read the "this refresh" column; lifetime rates on a long-lived server barely move.

Example, one R9700 after about five days of agent coding traffic (GPU -> RAM -> disk):

```
KV CACHE  qwen3.8-27b-vllm     17:49:04   delta: first sample
  running 0   waiting 0    pool   0.0%  preemptions 248

PREFIX CACHE            lifetime                   this refresh
  GPU             84.9%  (105.3 M / 124.0 M)      --    idle
  offload tier    47.5%  (8.9 M / 18.7 M)      --    idle

OFFLOAD TIER
  xfer buffers   0.00% busy  read 0.00%  write 0.00%   (in-flight, NOT tier fill)
  load     331.8 GB in   28.09s   ~11.8 GB/s     idle
  store    400.2 GB in   36.78s   ~10.9 GB/s     idle
  deferred lookups  n=467    mean wait 0.123s
```

The offload row counts only what the GPU cache missed: of the 18.7 M prompt tokens that fell
through to the tier, 8.9 M came back from RAM or disk instead of being prefilled again.

## Sizing

With a 16-bit ssm cache the attention block is 880 tokens rather than 1,648, so each offloaded
token carries twice as many Mamba states. Size `KVCACHE_RAM_GIB` from the token count you want
to hold.

## Files

- `patch_*.py` — the 15 consolidated patches (applied at boot from this directory, which
  rides the repo's existing `/patches` mount).
- `turnbench.py` (+ its deps `tierbench.py`, `equivbench.py`) — the correctness gate.
- `kvwatch.py` — live hit rates and tier traffic.
- `kvcache-reap.sh` + `.service`/`.timer` — eviction for the disk tier.

Evidence for these claims comes from radiance 0.9.3 / vLLM 0.27.1 on one R9700.
