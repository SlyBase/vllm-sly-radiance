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
  --image vllm-sly-radiance:0.4.5-rocm10.0 \
  --mode fast --owner manual --out out/accept-manual
```

### While production is busy

If `acquire` is refused because production is serving traffic (or a window is
already held), `accept.py` does not fail the gate. It backs off (default 90 s,
`ACCEPT_ACQUIRE_DELAY`) and retries until either the window opens or the total
patience runs out (default 60 min, `ACCEPT_ACQUIRE_WAIT`). `--force` is **not**
escalated automatically — it remains an explicit opt-in for planned maintenance
windows. Any other refusal (foreign lock owner, bad TTL, …) still fails the
gate immediately.

`--dry-run` prints the `gpu-window` calls without running anything.

## After the gate: release

A release is an annotated tag `v<VERSION>` on a commit main already carries; `build.yml` publishes every
`v*` tag to ghcr.io. `ci/release_tag.sh` cuts it, and there are two ways in:

| gate | what happens | who |
| --- | --- | --- |
| **green** | the `release` job of `accept.yml` tags the gated commit and the tag starts `build` | nobody, it just runs through |
| **red** | nothing is released; the report is on the run page | you decide |
| red, but the image is good anyway | Actions → **release** → *Run workflow* (`reason` is mandatory and goes into the tag message) | a human with write access |

```bash
gh workflow run release.yml -f reason="kv baseline stale: unit moved to --max-num-batched-tokens 8192, all else green"
# optional: -f version=0.2.7 -f sha=<commit>   (default: VERSION and the tip of main)
```

Only the main build of `VERSION` (or a by-hand run of the production profile with the default image on
main) releases automatically; calibration runs (`record_baseline`) and other profiles never do. The tag
is pushed with the sync App's token, because a tag pushed with `GITHUB_TOKEN` would not start `build`. The
script refuses a commit that is not on main, a VERSION that differs from the tag, and moving an existing
tag; the same tag on the same commit is a no-op, so re-running is safe.

The manual path is a dispatch and not an environment approval on the red gate on purpose: an environment
that does not exist yet, or has lost its required reviewers, runs without asking, i.e. the release would
fail open.

## Modes

| | `fast` (~22 min) | `full` (~27 min) |
| --- | --- | --- |
| when | every build | major/minor VERSION bump, or by hand |
| BetterBench | conc 1, 8 · 24 requests | conc 1, 2, 4, 8 · 48 requests |
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
| throughput per category | BetterBench `single_stream` medians | any category `< 95 %` of the baseline (`category_decode_tps_min_ratio`) |
| chat replay | 3-turn conversation at 4k/8k/16k, cold prefix cache | decode t/s `< 95 %` of the baseline (`chat_tps_min_ratio`); hard in `full`, soft in `fast` |
| tokens/step | `/metrics` delta over the benchmark | more than 0.1 below the baseline (drafter regression) |
| preemptions | `vllm:num_preemptions_total` delta | `> 0` |
| GSM8K | lm-eval | more than 0.05 below the baseline (`full` only) |
| production restore | `gpu-window release` + `curl /health` | production is not back up |

### Why per category and not only weighted (1.9.0, k=15)

The weighted decode median is a MIXTURE over `configs/*.json` weights. On 1.9.0 the DFlash2
depth change to k=15 raised it +10.3 % and the release shipped with it — while `prose` lost
7.2 %, `chat` 8.6 %, and a real chat lost half its throughput (1k context: 72 → 34 t/s, 16k:
28 → 21 t/s, measured 2026-10-26, same image, step time 39 ms in both arms). The mixture
carries 55 % weight on code/json/file_edit, so a category can lose without moving the number.

Two consequences, both in the gate now:

* **every category is its own check.** A lever that trades prose for code has to say so in the
  PR and re-calibrate the category that loses (threshold + baseline in the same PR).
* **the gate measures the profile the weights cannot describe.** BetterBench single-stream
  prompts are 60–120 tokens; the chat replay phase walks 4k → 16k with a cold prefix cache and
  reports decode-only t/s (TTFT excluded, thinking off).

`ci/accept/prod_mix.py` attacks the same problem from the data side: it reads the *production*
counters (per-position draft acceptance → the depth that still pays, plus the concurrency mix)
and prints the JSON to paste into a PR body. It states what it cannot do: **content mix
(code/prose/chat) is not in any metric** — the weights stay a declared assumption in
`configs/*.json`, and the per-category checks are what keep that assumption honest.

```bash
python3 ci/accept/prod_mix.py --prometheus http://192.168.178.21:30090   # live Prometheus
python3 ci/accept/prod_mix.py --metrics-file /tmp/prod-metrics.txt --json
```

Startup time is a soft check (reported, never blocking): the first start after a compile-cache
invalidation legitimately takes twice as long.

`known_warnings` in the profile suppresses the noise that production carries (TritonBundler cubin,
dlpack JIT) — it must stay an explicit list so a *new* warning still trips the forbidden patterns.
`install_attn_config_hook failed` used to be on it (0.2.3–0.2.6): it was the AttributeError of the decode
attention tune against aiter 0.1.21's config-table API, i.e. a dead feature filed as noise. It is off
the list since 0.2.7 (`sly/radiance_attn_decode.py`), and the log line `[radiance] decode attn tune
installed` is a *required* marker instead, so the tune silently not installing fails the gate.

The drafter tune (0.2.8, `sly/radiance_attn_drafter.py`) is guarded the same way: `drafter attn tune installed` is
required on both profiles, and on the production profile also `drafter attn plan:`, which is logged only when the
tuned launch takes the drafter's real call during the DFlash CUDA-graph capture. That second marker is the one that
matters: the first version of the gate declined the drafter's call (it is `causal=False`) and everything else looked
fine. `drafter attn tune fell back` (an exception in the tuned path) is a forbidden pattern.

The prompt-lookup override of the DFlash draft (0.2.9, `sly/radiance_lookup_draft.py`) is guarded the same way:
`[radiance] lookup draft: installed` is required on both profiles (it is logged at start by every process that
hooks the V2 runner's DFlash2 speculator, so a vLLM that changed under it fails the gate instead of silently
losing the feature), and `lookup draft: off after an error` / `not hooked` are forbidden. The `active` line is
logged at the first real decode step and is not required: the gate's smoke requests are short.

## Files

```
profiles/vllm7-mxfp4.json   production MXFP4 + DFlash2 k=7: markers, smoke cases, thresholds
profiles/vllm5-int4.json    the INT4 fallback service (same LXC, disabled unit)
profiles/vllm7-kv4.json     KV4/TETRA start smoke (needs a unit carrying launch_args; baselines/vllm7-kv4.json = KV pool floor)
baselines/*.json            the numbers of the image currently in production
configs/{fast,full}.json    BetterBench configs (full == betterbench config/default.json)
accept.py                   orchestrator: window, checks, verdict, report.json + report.md
../release_tag.sh           cuts the v<VERSION> tag (used by accept.yml's release job and release.yml)
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

A window takes port 8000 down for 20-30 minutes and `VllmDown` has `for: 10m`, so every single
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
