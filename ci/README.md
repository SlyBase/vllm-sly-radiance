# CI regression gates (`ci/`)

Each gate below exists because a regression of the 1.13.x / Inkling-200 weeks passed every older check.
The checkers are pure Python in `ci/quick_bench.py` (stdlib only) with unit tests in
`ci/tests/test_quick_bench.py` (run by the `resolve` job: `python3 -m unittest discover -s ci/tests`).
The fixtures in `ci/tests/fixtures/gates` are short excerpts of real host-51 runs: QP2 = 1.13.4 (good Qwen),
QC2 = eager Qwen (bad), QCL = the fix, `inkling-good` = the Inkling release arm.

## The gates

| # | Gate | Fails when | Where it runs |
|---|------|-----------|---------------|
| 1 | **Hook proof** (`hook_problems`) | a required log line is missing or a forbidden one is present. Qwen: `[radiance.fused_norm] add_rms_quant active`, compilation mode not `NONE`. Inkling: `RADIANCE_CODA_FP4_PROOF n=1 mxfp8->nvfp4`, `Capturing model for DFlash2 speculator`, `drafter conv projections int4 0`, `cudagraph_capture_sizes` covering 16..64 | every tier (accept gate, quick, release) |
| 2 | **Greedy vs last release** (`greedy_problems`) | one of 4 greedy prompts (T0, seed 1, 192 tokens) differs from `ci/accept/references/quick-<profile>-greedy.json`, unless `ci/accept/references/greedy_allow.json` lists that prompt index with a reason | release tier (`greedy_allow.json`), `fast`/`full`, quick |
| 3 | **Concurrency shape** (`concurrency_problems`) | aggregate tok/s is not rising 1 -> 2 -> 4 -> 8, or 2 streams < 1.2x one stream (BetterBench result; today 1.31x) / < 1.05x in the quick/release probe (8 x 128 tokens; measured 1.14-1.20 good, 0.68 collapsed) | release tier (1/2/4/8 probe), `fast`/`full` (levels in the BetterBench result) |
| 4 | **Drafter floor** (`floor_problems`) | median tok/update (code) of the BetterBench decode result < floor. Qwen 5.2 (`profiles/vllm7-mxfp4.json: tok_per_update_floor`; QP2 measured 5.5), Inkling 4.8 (`quick-inkling.json`; D4 5.13, the int4 drafter 3.65) | release tier (`greedy_allow.json`), `fast`/`full`, quick |
| 5 | **`sly/inkling2/tests` on CPU** | any test fails | `ci.yml` job `inkling2-tests` (Triton interpreter, torch-cpu, the pinned vLLM source). GPU-only tests skip themselves; two files that need triton / a dev tree are named in the pytest header by `sly/inkling2/tests/conftest.py`. The job prints `N passed, M skipped` |
| 6 | **Harness lint** | a script passes several `KEY=VAL` in one `-e` (`docker run -e "A=1 B=2"` sets ONE variable) | `ci/check_consistency.py` (`consistency` job, all tracked `*.sh`) |
| 7 | **Runner hygiene** | -- | `clean workspace` step before `actions/checkout` on every self-hosted job (`build.yml`, `accept.yml`, `accept-night.yml`): deletes the workspace, with a root `docker run ... find -delete` fallback for root-owned `__pycache__` |

Gates 1-4 sit in `accept.py` as hard checks in the report table (`gate_checks`), so a red gate means no release
(`release` job, `accept-night`). They also work standalone on artifacts:

```bash
python3 ci/quick_bench.py hooks server.log --profile qwen        # or inkling
python3 ci/quick_bench.py conc bb-c1.json bb-c2.json bb-c4.json bb-c8.json
python3 ci/quick_bench.py floor bb-decode.json --floor 4.8
```

### Updating a reference

* **Greedy** (`quick-<profile>-greedy.json`): the output of the last *released* image. After a release, record on
  the released image (`quick_bench.py run ... --record`, below). A bump that legitimately changes the text
  (new torch, new kernel with another reduction order) does not touch the reference; list the prompt in
  `greedy_allow.json` -- `{"qwen": {"2": "torch 2.15 changes the reduction order, answers stay correct"}}` -- in the
  same PR. An empty reason does not count. The allow-list entry is removed when the reference is re-recorded.
* **Thresholds / floors**: edit `ci/accept/references/quick-<profile>.json` (`tok_per_update.code`,
  `c2_over_c1_min`, `c4_over_c1_min`, `thresholds`) or `tok_per_update_floor` in the accept profile. `null` =
  not calibrated yet, skipped (never failed). `--record` writes floors at 95 % (tok/update) and 90 % (ratios)
  of the measurement.

## GPU tiers

| Tier | What | Time (after health) | Runs |
|------|------|------|------|
| **quick** | `accept.py --mode quick` (or `quick_bench.py run` on any live server): hook proof on the server log, greedy gate, decode (Mia method: prose + code, 400 tokens, `ignore_eos`, thinking on, T0.7 top_k 20, 5 reps, median) with tok/update from `/metrics` deltas, prefill 8k x 2 (token-id prompts, TTFT -> pp/s), concurrency 1/2/4 streams (8 short requests each). A `null` reference entry is skipped. | ~5 min | PR images, by hand: Actions -> accept -> mode `quick`. Never tags |
| **release** | `accept.py --mode release` = quick + 8 streams + **GSM8K-50** (first 50 items of `references/gsm8k50.jsonl`, greedy T=0, thinking on, 4096 max tokens; floor = reference correct - 2 items) + the drafter floor. **A `null` reference FAILS the tier** | ~10 min per profile | every main build; gates the tag; **both profiles**: `vllm7-mxfp4` (Qwen) and `inkling` |
| **full** | `accept.py --mode full` (and the legacy `fast`): smoke, chat replay, full BetterBench sweep, spec counters, lm-eval GSM8K, baselines in `ci/accept/baselines`; prefill sweep / HumanEval belong here too | 22-30 min | optional, manual (`accept.yml` mode `full`) and the legacy night gate; **not required for tagging** |

Verdict against `ci/accept/references/quick-<profile>.json`: decode -2 %, prefill -3 % (measured noise of the Mia
method <= 0.2 %), tok/update >= floor, c2/c4/c8-over-c1 >= floor, GSM8K >= reference - 2. Hook proof comes first:
a dead hook makes the numbers meaningless.

### Fail closed until the references are recorded

In release mode `unrecorded_problems` checks the reference before any probe runs: every decode value, `prefill_pps`,
`tok_per_update`, `c2/c4/c8_over_c1_min`, `gsm8k_correct` and the greedy reference must be set, otherwise the tier
fails with `references not recorded, run ci/quick_backtest.sh RECORD=1 (missing: ...)` and the report row
`references recorded` is red. The committed seed files are deliberately incomplete, so **no release passes until the
first recording window has run**. (`--record`, only on a known-good image, is the one way past the guard.)

### Release chain

`build` green on main -> `accept.yml` (`plan` -> `gate` matrix over **vllm7-mxfp4, inkling**, one GPU window and one
`gpu-window` approval each, sequential) -> each leg sets the commit status `release-tier/<profile>` -> the `release`
job (needs every leg) re-reads both statuses with `quick_bench.py release-check` and only then tags. `release.yml`
(manual) checks the same two statuses; `skip_tier_check` is the audited override (the reason goes into the tag
message). The night gate (`accept-night.yml`) still runs the old fast/full gate and passes `skip_tier_check=true`
until it is moved to the release tier (TODO in the workflow).

**Inkling leg without a helper change.** `gpu-window` only knows the prod units, so the `inkling` profile
(`"window": "local-docker"`) runs inside its own window like our hand chains: `acquire` (helper lock) ->
`systemctl stop docker-vllm7.service` -> `docker run` of the release-arm container on :8001 (`inkling_docker_argv`
in `accept.py`, hand-run twin `start_inkling` in `ci/quick_backtest.sh`, a unit test keeps them in step) -> health ->
release tier -> `docker rm -f` -> `systemctl start docker-vllm7.service` -> helper `release` (checks `/health`).
`LocalDockerWindow.cleanup()` is idempotent and runs from `release()`, an `atexit` hook and SIGTERM/SIGINT (a cancelled
job); the order and every failure path are unit-tested against a mocked docker/systemctl
(`ci/tests/test_local_window.py`). A SIGKILL is left to `gpu-window-guard.timer`. Because it starts docker itself, this
leg runs on the GPU host runner (`rocm-build` label, LXC 2408); the runner user needs docker, permission to
stop/start `docker-vllm7.service` and read access to the model dirs (repo variables `ACCEPT_INKLING_MODEL`,
`ACCEPT_INKLING_DRAFTER`, `ACCEPT_INKLING_CACHE`, defaults `/root/inkling2-mini`, `/root/dflash2-ink/model`,
`/root/accept-inkling-cache`). That is unverified until the first run. Each profile keeps its own window and its own
`gpu-window` approval.

**`skip_tier_check` (audited override).** `release.yml` refuses to tag unless both `release-tier/*` statuses are
green on the commit; `skip_tier_check=true` overrides that, the mandatory `reason` goes into the tag message and the
run summary says `release tier: skipped`. The night gate uses it today (TODO in `accept-night.yml`: move it to the
release tier after the backtest passed, then drop the flag). Do not use it to get past a red release tier without
writing why.

### Backtest and recording (one GPU window each, not run yet)

`ci/quick_backtest.sh` runs the release tier on known images, one after the other, and compares the verdict with
what we know:

| Arm | Image | Expected |
|-----|-------|----------|
| Q-GOOD | `1.13.4-rocm10.1` (Qwen) | PASS (reference) |
| Q-EAGER | `ci-262-ff1aff1` (Qwen) | FAIL, hook proof (eager) |
| I-GOOD | `ci-262-ff1aff1` (Inkling) | PASS (reference) |
| I-CONC | `ci-262-032eabe` (Inkling) | FAIL, c2/c1 / c4/c1 |
| I-INT4 | `ci-262-ff1aff1` + `RADIANCE_DFLASH_CONV_W4=1` | FAIL, hook + tok/update |

```bash
# on host 51 with the gpu-window held and docker-vllm7 stopped, repo checkout at ci/regression-gates:
RECORD=1 ARMS="Q-GOOD I-GOOD" ci/quick_backtest.sh   # 1. record the references (~25 min); commit them
ci/quick_backtest.sh                                  # 2. all five arms against them (~60 min)
```

Exit 0 only if every arm gave the expected verdict; a start failure counts as wrong, and so does a bad arm that
failed only because a reference is still unrecorded (`UNRECORDED`). If `I-CONC` passes, the c2/c4 floors are too
low: raise them in `quick-inkling.json` and repeat that arm. Greedy reference = output of the last *released*
image; re-record after each release.
