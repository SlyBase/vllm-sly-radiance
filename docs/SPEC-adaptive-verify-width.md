# Spec: Adaptive verify width for DFlash2 (native vLLM 0.34 mechanism)

Status: DRAFT — awaiting owner approval. No code until approved.
Supersedes the "block-drafter redesign" hypothesis of dynab-20261119 (the MTP
controller port stays CLOSED; this is a different mechanism).

ASSUMPTIONS I'M MAKING:
1. Goal is decode throughput on the production config (Swift-1.5-27B MXFP4,
   DFlash2 k=7 probabilistic, TRITON_ATTN drafter, 210 W), lossless outputs.
2. Upstream-native code beats custom code: we enable and audit what vLLM 0.34
   already ships instead of writing a new controller (ponytail rung 4).
3. The vendored DFlash2 backport (`dflash2/`, `patch_dflash2.py`, a 0.27.1-era
   snapshot of upstream PR #52816) can be converged onto native 0.34 DFlash2
   unless the audit finds a vendored-only essential.
4. Measurement follows AGENTS.md: same-window interleaved A/B, per-category
   table, GSM8K, KV-pool invariant; a tie closes the project (YAGNI guard).
Correct me now or I'll proceed with these.

## Objective

Trim low-confidence draft tails per request so the target verifies fewer wasted
positions. Expected gain sits mainly at concurrency (verify batch = B*(k+1)
tokens; trimming shrinks it) — at conc 1 the win may be a tie, which is an
acceptable outcome that closes the project.

**Key finding (pinned v0.34.0 tree):** the mechanism already exists upstream:

- `SpeculativeConfig.enable_adaptive_verification` (config/speculative.py:633,
  default False; docstring says "dspark only", but the wiring is generic and
  DFlash2 calls it explicitly — no hard method restriction found).
- `OnlineAcceptanceEstimator` (spec_decode/acceptance_estimator.py): per-position
  acceptance prediction from the draft distribution's log-odds, logistic with
  online calibration from the target's real verdicts (`observe_verification`).
- `AdaptiveVerificationManager` (spec_decode/adaptive_verification.py): survival
  probabilities → per-step draft-token budget across requests (Triton
  `_assign_draft_token_budget_kernel`), variable-width verification;
  `resolve_adaptive_cudagraph_mode` keeps FULL_DECODE_ONLY (our prod mode).
- Native DFlash2 feeds it the realized selector scores:
  `dflash2/speculator.py:254` `_maybe_predict_acceptance(candidate_sampler.scores…)`.

**Why we don't have it:** our image installs the vendored `dflash2/speculator.py`
OVER the native one (patch_dflash2.py seam 6). The vendored snapshot predates the
adaptive branch and has drifted from the 0.34 sampling contract:

| aspect | vendored (ours) | native 0.34 |
|---|---|---|
| Gumbel noise key | raw `seed`, position | `spec_step_seed(seed, step)`, `IS_DRAFTING=True` |
| rejection sampler contract | — | 0.34 rejection_sampler.py:275 uses `spec_step_seed(seed, col)` |
| draft-logits cache | row = req_state | `draft_logits_row()` + row_mapping (sharded-safe) |
| adaptive verification | absent | wired |

The noise-key mismatch means draft and target no longer draw the same noise —
still lossless (ratio test), but the consistent-noise rejection path is degraded.
Converging to native fixes that independently of adaptive verification.

Native 0.34 also ships `qwen3_dflash2.py` and the registry entry
(`registry.py:688`) — the entire backport is upstream now.

## Options

- **A (recommended):** converge onto native 0.34 DFlash2 (retire vendored files
  and the patch seams that are native now), enable the flag in the unit, measure.
- **B:** keep the vendored speculator, port the adaptive branch + noise contract
  into it. More code, deeper divergence; only if the audit finds a vendored-only
  essential that native lacks.
- **C:** custom redesign. YAGNI until A and B are proven insufficient.

Phase 0 decides A vs B. No custom kernel work is planned.

## Commands

```
# static / CPU (repo root)
python3 ci/check_consistency.py --base origin/main
ci/patch_dryrun.sh
python3 -m unittest discover -s ci/tests
python3 -m pytest sly/tests/ -q

# GPU window (LXC 2408, prodtest kit — see skill reference vllm-0.34-upgrade.md)
/opt/data/stack245/prodtest/prodtest.sh swap      # acquire window, image swap, markers+smoke
/opt/data/stack245/prodtest/prodtest.sh battery   # BB decode-only, GSM8K-200, loops, metrics
/opt/data/stack245/prodtest/prodtest.sh finish KEEP|REVERT

# gate / release (after a win)
gh workflow run accept-night.yml -f pr=main -f dry_run=false -f run_now=true   # owner-approved
```

## Project structure (files touched, per phase)

```
Phase 0 (audit, no image change):
  docs/TECHNICAL.md                  audit note: seam-by-seam native-vs-vendored verdict
Phase 1+ (option A):
  dflash2/speculator.py              retire (native replaces it)
  dflash2/qwen3_dflash2.py           retire or diff-merge if vendored-only essentials
  patch_dflash2.py                   shrink to the seams 0.34 still needs (likely ~0)
  Dockerfile                         patch loop / COPY adjustments
  sly/tests/test_dflash2_native_cpu.py   noise-contract + regression tests (TDD)
  homelab: docker-vllm7.service.j2   --speculative-config.enable_adaptive_verification true
  homelab: group_vars/lxc_vllm7.yml  the variable + comment
Phase 3 (only on a win):
  CHANGELOG.md [Unreleased]          Added + Measured per-category table
  ci/accept/profiles/vllm7-mxfp4.json  required marker for the enabled feature
  README.md                          options table row (the flag, default, why)
```

## Code style

Repo conventions (AGENTS.md, sly/README.md): idempotent anchored patches via
`_patchlib.apply`, CPU tests with tensors that raise on device→host reads, one
knob with a documented default. The knob here is vLLM's own — no new radiance env
var. Unit change shape:

```jinja
  --speculative-config.draft_sample_method {{ vllm_speculative_config_draft_sample_method }} \
{% if vllm_speculative_config_adaptive_verification | default(false) | bool %}
  --speculative-config.enable_adaptive_verification true \
{% endif %}
```

## Testing strategy

- **CPU (TDD, red first):**
  - noise contract: draft walk and rejection sampler derive the same draw for
    (seed, step, position) — the regression the vendored override causes today;
  - seam audit tests: registry resolves DFlash2DraftModel natively; causality
    reads `is_causal`; draft_logits cache is fp32/-inf with row_mapping;
  - estimator shape test: predict() on synthetic K-candidate scores produces
    probabilities in [0,1], step() folds verdicts without host reads.
- **GPU (prodtest window):** startup markers + smoke, KV pool == 425532, estimator
  log line present, no forbidden patterns; seeded greedy replay identity
  (losslessness: draft strategy must not change outputs).
- **A/B battery (AGENTS.md):** one window, control first and repeated last,
  interleaved ×2, 300 W firmware fan curve for the A/B then restore 210 W;
  rank by step gap or ≥128 BetterBench runs per arm; per-category decode table
  + chat replay 4k/8k/16k; GSM8K-200; acceptance rate + tokens/step counters.
- **Gate:** full accept-night gate green before any release (per-category ratios
  and chat replay enforced mechanically).

## Boundaries

- **Always:** TDD; AGENTS.md measurement rules; restore 210 W and production after
  every window; per-category numbers for anything that moves tokens/update;
  homelab changes via repo + GitOps.
- **Ask first:** retiring the vendored backport (architecture decision); the
  homelab unit flag merge; each GPU window; 300 W A/B; any upstream PR.
- **Never:** revive the radiance_draft MTP controller (closed by owner decision);
  edit the rejection sampler; ship without per-category numbers; ad-hoc
  pct/kubectl as persistent state; merge this spec's implementation before the
  owner approves the spec.

## Success criteria (proposed thresholds — owner confirms)

1. Weighted decode **≥ +2 %** vs control, sign-stable across ≥2 interleaved pairs
   in one window (quote the spread; dynab lesson: one pair is never a result).
2. No category worse than **−2 %**; chat replay ≥ 0.98x at 4k/8k/16k.
3. GSM8K-200 ≥ baseline − 0.01; seeded greedy replay identical (lossless).
4. KV pool stays 425532; startup ≤ 1.5x baseline; production restored 200.
5. If 1–4 fail or tie: NOT-ADOPTED.md entry with the data, project closed.

## Risks

- **Noise contract / acceptance:** converging changes the draft sampling key;
  acceptance could move either way. Phase 0 reads the 0.34 rejection path;
  Phase 1 measures acceptance before the A/B.
- **Backend compatibility:** the manager raises for attention backends that
  cannot trim query lengths on device. Audit target + drafter backends
  (TRITON_ATTN drafter; target backend per unit) before the window.
- **"dspark only" docstring:** may hide an unsupported dflash assumption;
  Phase 1 smoke is the decider, cheap.
- **k=7 is shallow:** trimming saves little at conc 1; a conc-1-only tie is a
  legitimate kill. Concurrency levels 2/4/8 carry the hypothesis.
- **Estimator cold start:** generic coefficients, online fit converges within a
  refit or two — measure warm, second start.

## Open questions

1. Win threshold: is ≥ +2 % weighted with the −2 % category floor right, or does
   the owner want a stricter/looser bar?
2. If the audit finds vendored-only essentials: option B (port) or drop them?
3. After a win: flag default-on in the prod unit, or opt-in per deployment?
4. Window budget: 2–3 GPU windows for the interleaved A/B — acceptable?
