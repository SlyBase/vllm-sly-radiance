# CI regression gates (`ci/`)

Each gate below exists because a regression of the 1.13.x / Inkling-200 weeks passed every older check.
The checkers are pure Python in `ci/quick_bench.py` (stdlib only) with unit tests in
`ci/tests/test_quick_bench.py` (run by the `resolve` job: `python3 -m unittest discover -s ci/tests`).
The fixtures in `ci/tests/fixtures/gates` are short excerpts of real host-51 runs: QP2 = 1.13.4 (good Qwen),
QC2 = eager Qwen (bad), QCL = the fix, `inkling-good` = the Inkling release arm.

## The gates

| # | Gate | Fails when | Where it runs |
|---|------|-----------|---------------|
| 1 | **Hook proof** (`hook_problems`) | a required log line is missing or a forbidden one is present. Qwen: `[radiance.fused_norm] add_rms_quant active`, compilation mode not `NONE`. Inkling: `RADIANCE_CODA_FP4_PROOF n=1 mxfp8->nvfp4`, `Capturing model for DFlash2 speculator`, `drafter conv projections int4 0`, `cudagraph_capture_sizes` covering 16..64 | accept gate (all modes) + quick tier |
| 2 | **Greedy vs last release** (`greedy_problems`) | one of 4 greedy prompts (T0, seed 1, 192 tokens) differs from `ci/accept/references/quick-<profile>-greedy.json`, unless `ci/accept/references/greedy_allow.json` lists that prompt index with a reason | accept gate `fast`/`full`, quick tier |
| 3 | **Concurrency shape** (`concurrency_problems`) | aggregate tok/s is not rising 1 -> 2 -> 4 -> 8, or 2 streams < 1.2x one stream (today 1.31x; tighten later) | accept gate `fast`/`full` (levels present in the BetterBench result) |
| 4 | **Drafter floor** (`floor_problems`) | median tok/update (code) of the BetterBench decode result < floor. Qwen 5.2 (`profiles/vllm7-mxfp4.json: tok_per_update_floor`; QP2 measured 5.5), Inkling 4.8 (`quick-inkling.json`; D4 5.13, the int4 drafter 3.65) | accept gate `fast`/`full`, quick tier |
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

| Tier | What | Time | When |
|------|------|------|------|
| **quick** | `accept.py --mode quick` (or `quick_bench.py run` on any live server): hook proof on the server log, greedy gate, decode (Mia method: prose + code, 400 tokens, `ignore_eos`, thinking on, T0.7 top_k 20, 5 reps, median) with tok/update from `/metrics` deltas, prefill 8k x 2 (token-id prompts, TTFT -> pp/s), concurrency 1/2/4 streams (8 short requests each) | ~5 min after health | PR builds (label `build`), by hand: Actions -> accept -> mode `quick` with the PR image |
| **release** | `accept.py --mode fast` / `full`: smoke, chat replay, BetterBench sweep, spec counters, GSM8K (full), plus gates 1-4 | 22-30 min | main builds / accept-night / manual |

Verdict of the quick tier against `ci/accept/references/quick-<profile>.json`: decode -2 %, prefill -3 %
(measured noise of the Mia method <= 0.2 %), tok/update >= floor, c2/c1 and c4/c1 >= floor. A `null` reference is
skipped. Hook proof fails first -- a dead hook makes the numbers meaningless. In `accept.py` the quick mode never
releases (`release` job: `inputs.mode != 'quick'`).

Not wired yet: the automatic start on a PR build. `accept.yml` only plans for builds of `main` (the PR image is a
local tag on host 51 and the gate waits for an approval in `gpu-window`). Until that is decided the quick tier is
a manual dispatch with `image=vllm-sly-radiance:ci-<pr>-<sha>`.

### Backtest (one GPU window, not run yet)

`ci/quick_backtest.sh` starts known images one after the other, runs the quick tier and compares the verdict
with what we know:

| Arm | Image | Expected |
|-----|-------|----------|
| Q-GOOD | `1.13.4-rocm10.1` (Qwen) | PASS (reference) |
| Q-EAGER | `ci-262-ff1aff1` (Qwen) | FAIL, hook proof (eager) |
| I-GOOD | `ci-262-ff1aff1` (Inkling) | PASS (reference) |
| I-CONC | `ci-262-032eabe` (Inkling) | FAIL, c2/c1 / c4/c1 |
| I-INT4 | `ci-262-ff1aff1` + `RADIANCE_DFLASH_CONV_W4=1` | FAIL, hook + tok/update |

```bash
# on host 51 with the gpu-window held and docker-vllm7 stopped, repo checkout at ci/regression-gates:
RECORD=1 ARMS="Q-GOOD I-GOOD" ci/quick_backtest.sh   # records the two references (~15 min)
ci/quick_backtest.sh                                  # all five arms against them (~30 min)
```

Exit 0 only if every arm gave the expected verdict; a start failure counts as wrong. If `I-CONC` passes, the c2/c4
floors are too low: raise them in `quick-inkling.json` and repeat that arm. Commit the recorded references.
