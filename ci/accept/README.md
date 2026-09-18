# Acceptance gate (`ci/accept`)

Third CI tier, on top of `ci.yml` (lint/patch-dryrun/consistency) and `build.yml` (image build +
CPU import smoke test): **does a new image still behave like the one in production?**

`ci.yml` and `build.yml` prove that the patches apply and that the image imports. Neither can see
a regression that only shows up on the GPU — a lost KV pool, a kernel that silently fell back to
AITER, a drafter that stopped accepting tokens, a reasoning parser that broke. That is what this
gate measures, always against the numbers the production image produced.

## The one rule this is built around

There is exactly one R9700 in LXC 2408 and it runs production (`docker-vllm7.service`). The gate
therefore never touches the GPU directly. It runs on the CPU-only CI LXC and asks
`/usr/local/sbin/gpu-window` in 2408 — reachable only through a forced-command SSH login as
`gpuwin` — to hand the GPU over and, unconditionally, to give it back:

```
acquire → start candidate → measure → release (stop candidate, start vllm7, verify /health=200)
```

`release` runs in a `finally`, a failed release fails the gate, and `gpu-window-guard.timer` in
2408 restores production on its own if a window ever outlives its TTL (crashed runner, cancelled
job, network loss). The candidate runs with the arguments of the *live* production unit
(`systemctl cat/show docker-vllm7.service`), with only the image, container name and compile-cache
directory substituted — so the gate cannot drift away from what actually runs.

## Running it

Automatically: `build.yml` succeeds → `accept.yml` queues → the job waits in the **`gpu-window`**
environment until a reviewer approves it. Nothing happens to production before that click.

By hand: *Actions → accept → Run workflow* (image, profile, `fast`/`full`, `force`,
`record_baseline`). Or directly from the CI LXC:

```bash
python3 ci/accept/accept.py \
  --profile ci/accept/profiles/vllm7-mxfp4.json \
  --image vllm-sly-radiance:0.2.5-rocm10.0 \
  --mode fast --out out/accept-0.2.5
```

`--dry-run` prints the `gpu-window` calls without running anything.

## Modes

| | `fast` (~25 min) | `full` (~70 min) |
| --- | --- | --- |
| when | every build | major/minor VERSION bump, or by hand |
| BetterBench | conc 1, 8 · 24 requests | conc 1, 2, 4, 8, 16 · 48 requests |
| GSM8K | — | 200 items |
| everything else | yes | yes |

## What is checked

| check | source | fails when |
| --- | --- | --- |
| startup | `gpu-window start` | server never reports `/health=200` or no KV pool line |
| log markers | container log | a required `[radiance]` line is missing or a forbidden pattern appears |
| KV pool | `GPU KV cache size: N tokens` | `< 98 %` of the baseline |
| smoke | 4 chat completions | wrong answer, no reasoning content, no tool call, needle not found |
| throughput | BetterBench `aggregate_tps` | `< 95 %` of the baseline at any level |
| tokens/step | `/metrics` delta over the benchmark | more than 0.1 below the baseline (drafter regression) |
| preemptions | `vllm:num_preemptions_total` delta | `> 0` |
| GSM8K | lm-eval | more than 0.05 below the baseline (`full` only) |
| production restore | `gpu-window release` + `curl /health` | production is not back up |

Startup time is a soft check (reported, never blocking): the first start after a compile-cache
invalidation legitimately takes twice as long.

`known_warnings` in the profile suppresses the noise that production has carried since 0.2.3
(`install_attn_config_hook failed`, TritonBundler cubin, dlpack JIT) — it must stay an explicit
list so a *new* warning still trips the forbidden patterns.

## Files

```
profiles/vllm7-mxfp4.json   production MXFP4 + DFlash2 k=7: markers, smoke cases, thresholds
profiles/vllm5-int4.json    the INT4 fallback service (same LXC, disabled unit)
baselines/*.json            the numbers of the image currently in production
configs/{fast,full}.json    BetterBench configs (full == betterbench config/default.json)
accept.py                   orchestrator: window, checks, verdict, report.json + report.md
```

Baselines carry mode-independent values (`kv_tokens`, `gsm8k`, `mean_tokens_per_step`) and
per-mode throughput, because `aggregate_tps` depends on `concurrency_requests`. An empty or `null`
entry is skipped, never failed — so a fresh baseline degrades to "measure and report" instead of
blocking. Calibrate with one run against the image that is already in production:

```bash
python3 ci/accept/accept.py --profile ci/accept/profiles/vllm7-mxfp4.json \
  --image vllm-sly-radiance:0.2.3-rocm10.0 --mode fast --record-baseline --out out/calib
```

**After a rollout, move the baseline forward** (commit the `--record-baseline` diff, or edit the
file by hand): the baseline is "what production does today", not "the best we ever measured".

## The `gpu-window` contract

`accept.py` speaks to exactly one root-owned script in 2408 (homelab repo:
`infrastructure/proxmox/ansible/playbooks/files/gpu-window`). Subcommands, one JSON object on the
last stdout line:

| command | does | returns |
| --- | --- | --- |
| `status` | lock state, running unit, health, requests in flight | `{locked, owner, age_s, service, health, busy}` |
| `acquire <owner> <ttl_s> [force]` | takes `/root/gpu-window.lock.d`, silences the vLLM alerts; **refuses** while requests are running/waiting or `POST /v1/` appeared in the last minutes unless `force` | `{owner, ttl_s, previous_service, silences}` |
| `start <image> <service> [load_format]` | stops the production unit, starts the candidate container from the live unit's arguments, double-starts on an invalidated compile cache, waits for `/health` | `{container, image, kv_tokens, startup_s, starts, load_format}` |
| `logs [n]` | candidate container log | raw text |
| `release` | stops the candidate, starts the production unit, waits for `/health=200`, winds the silence down | `{service, health, kv_tokens}` |
| `guard` | timer-driven: releases a window past its TTL | `{action, ...}` |

The runner user never gets `docker` or `systemctl` rights of its own — only this script, through a
sudoers whitelist and an SSH forced command.

### Alerts during a window

A window takes port 8000 down for 13–25 minutes and `VllmDown` has `for: 10m`, so every single
acceptance run used to page. `acquire` therefore creates two Alertmanager silences — `job="vllm"`
for `VllmDown`/`TargetDown`, and `alertname=~"Vllm.*"` for the two alerts built on `vllm:*`
recording rules, which drop the `job` label — and `release` winds them down to `now + 300 s` so the
firing alert can resolve quietly instead of notifying on the way out. If the production unit does
**not** come back healthy, the silences are expired immediately: that is no longer planned
downtime. All of it is best-effort; an unreachable Alertmanager never fails a run.

### `load_format`

The one engine argument the runner may set, and it is checked against a literal allowlist
(`auto`, `safetensors`, `runai_streamer`, `fastsafetensors`) in both the forced command and the
helper. It exists because ~109 s of every start is weight loading at ~197 MB/s against a measured
~420 MB/s ceiling, and the multi-threaded loaders in the image are the cheapest thing to try
against that. It changes only how weights get into memory, never what the engine computes.
A gate run leaves it unset and thus tests exactly what production runs — pass it only for an
experiment inside an already-open window, where the marginal cost of one more start is ~5 min
against the window's ~15 min fixed block.
