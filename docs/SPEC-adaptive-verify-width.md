# Spec: Adaptive verify width for DFlash2 (native vLLM 0.34 mechanism)

Status: CLOSED 2026-11-20 — the A/B failed the win bar (weighted −3.8 %, no trimming at
concurrency 1). Results and the revisit conditions: docs/NOT-ADOPTED.md, entry avab-20261120.
Phase 0 audit + Phase 1/2 executed as specified; no code shipped. Original spec below.
Supersedes the "block-drafter redesign" hypothesis of dynab-20261119 (the MTP
controller port stays CLOSED; this is a different mechanism).

ASSUMPTIONS (confirmed by approval):
1. Goal is decode throughput on the production config (Swift-1.5-27B MXFP4,
   DFlash2 k=7 probabilistic, TRITON_ATTN drafter, R4D_HYBRID target, 210 W),
   lossless outputs.
2. Upstream-native code beats custom code: enable and measure what vLLM 0.34
   already ships (ponytail rung 4).
3. Measurement follows AGENTS.md: same-window interleaved A/B, per-category
   table, GSM8K, KV-pool invariant; a tie closes the project (YAGNI guard).

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
  online calibration from the target's real verdicts (`observe_verification`,
  wired generically at spec_decode/speculator.py:681).
- `AdaptiveVerificationManager` (spec_decode/adaptive_verification.py): survival
  probabilities → per-step draft-token budget across requests (Triton
  `_assign_draft_token_budget_kernel`), variable-width verification;
  `resolve_adaptive_cudagraph_mode` keeps FULL_DECODE_ONLY (our prod mode).
- Native DFlash2 feeds it the realized selector scores:
  `dflash2/speculator.py:254` `_maybe_predict_acceptance(candidate_sampler.scores…)`.

**Why it is off (corrected by the Phase-0 audit):** ONLY the config default.
The earlier draft of this spec assumed the vendored 0.27.1 backport overwrites
the native speculator — that is FALSE on the current stack:

- `patch_dflash2` was DROPPED from the Dockerfile loop at the vLLM 0.29.0 bump
  (ci/unused_patches.txt: "native upstream since the vLLM 0.29.0 bump; kept for
  the 0.27.x lineage"); its `install_modules()` additionally NOOP-guards against
  overwriting upstream files.
- The running 1.13.4 image carries the NATIVE speculator: CPU-only grep in the
  image finds `enable_adaptive_verification` and `IS_DRAFTING` in
  site-packages/.../dflash2/speculator.py (2 matches).
- The native walk keys Gumbel noise by `spec_step_seed(seed, step)` — the same
  key the 0.34 rejection sampler uses (rejection_sampler.py:275). The noise
  contract is consistent; the vendored raw-seed drift exists only in the inert
  lineage files, not in the image.

So the experiment is **config-only**: set the flag, measure. No vLLM-repo code
change is needed to run it.

## Phase 0 audit results (DONE, 2026-11-20, static + CPU-only image grep)

| check | result | evidence (pinned v0.34.0) |
|---|---|---|
| native DFlash2 in the image | YES | image grep: enable_adaptive_verification + IS_DRAFTING present |
| backport inert on 0.34 | YES | Dockerfile comment (dropped at 0.29), unused_patches.txt, NOOP guard |
| seams native (causality, registry, top-k, gumbel, draft_logits_spec) | YES | qwen3_dflash.py:88, registry.py:688, logits_processor.py:265, gumbel.py:75/172 |
| noise contract draft↔rejection | consistent | spec_step_seed in walk + rejection_sampler.py:275 |
| estimator online calibration wired | YES | speculator.py:681 observe_verification(idx_mapping, num_sampled, num_rejected) |
| target backend R4D_HYBRID | compatible | subclasses RocmAiterUnifiedAttentionBackend; base default `not is_ssm()` = True (backend.py:220); decode consumes device query_start_loc (rocm_aiter_unified_attn.py:305); hybrid inherits supports_* on purpose |
| GDN layers (Swift is attention+GDN hybrid) | compatible | gdn_attn.py:89 returns True (decode kernels take device offsets); mamba2 True unless use_replayssm |
| FULL_DECODE_ONLY | preserved | resolve_adaptive_cudagraph_mode keeps FULL_DECODE_ONLY |
| use_local_argmax_reduction conflict | none | False in prod (config raises only if both set) |
| "dspark only" docstring | no hard restriction | only the local-argmax conflict + backend check; Phase-1 smoke is the empirical decider |

Remaining unknowns are empirical (estimator fit on our model, actual throughput
effect) → Phase 1/2.

## Options

- **A (recommended, audit-confirmed):** config-only enablement + measurement.
- **B:** backend/estimator adaptation code — only if the Phase-1 smoke or the
  A/B shows a concrete incompatibility the audit missed.
- **C:** custom redesign. YAGNI unless A and B are proven insufficient.

## Commands

```
# static / CPU (repo root)
python3 ci/check_consistency.py --base origin/main
python3 -m unittest discover -s ci/tests

# GPU window (LXC 2408, prodtest kit — skill reference vllm-0.34-upgrade.md)
/opt/data/stack245/prodtest/prodtest.sh swap      # acquire window, image/unit swap, markers+smoke
/opt/data/stack245/prodtest/prodtest.sh battery   # BB decode-only, GSM8K-200, loops, metrics
/opt/data/stack245/prodtest/prodtest.sh finish KEEP|REVERT

# gate / release (only after a win, owner-approved window)
gh workflow run accept-night.yml -f pr=main -f dry_run=false -f run_now=true
```

## Project structure (files touched, per phase)

```
Phase 1 (test window, no persistent change):
  host unit docker-vllm7.service   manual ExecStart edit inside the window:
                                   --speculative-config.enable_adaptive_verification true
                                   (prodtest flow keeps the host unit in sync; reverted on REVERT)
Phase 2 (A/B): none (same image, flag on/off arms)
Phase 3, only on a win:
  homelab: docker-vllm7.service.j2   flag line behind a variable (GitOps PR)
  homelab: group_vars/lxc_vllm7.yml  the variable + comment
  repo CHANGELOG.md [Unreleased]     Added + Measured per-category table (release PR)
  repo ci/accept/profiles/vllm7-mxfp4.json  required marker for the enabled feature
  repo README.md                     options table row (the flag, default, why)
On a tie/loss:
  repo docs/NOT-ADOPTED.md           entry with the A/B data, project closed
```

## Code style

No new radiance env var — the knob is vLLM's own. Persistent unit change shape:

```jinja
  --speculative-config.draft_sample_method {{ vllm_speculative_config_draft_sample_method }} \
{% if vllm_speculative_config_adaptive_verification | default(false) | bool %}
  --speculative-config.enable_adaptive_verification true \
{% endif %}
```

## Testing strategy

- **Phase 1 smoke (GPU window):** startup markers + smoke 4/4, KV pool == 425532,
  estimator/manager log lines present, no forbidden patterns, no backend
  ValueError; seeded greedy replay identity (losslessness: the draft strategy
  must not change outputs).
- **A/B battery (AGENTS.md):** one window, control first and repeated last,
  interleaved ×2, 300 W firmware fan curve for the A/B then restore 210 W;
  rank by step gap or ≥128 BetterBench runs per arm; per-category decode table
  + chat replay 4k/8k/16k; GSM8K-200; acceptance rate + tokens/step counters.
- **Gate:** full accept-night gate green before any release (per-category ratios
  and chat replay enforced mechanically).
- CPU/TDD applies only if option B becomes necessary (concrete incompatibility).

## Boundaries

- **Always:** AGENTS.md measurement rules; restore 210 W and production after
  every window; per-category numbers for anything that moves tokens/update;
  homelab changes via repo + GitOps; TDD for any code.
- **Ask first:** each GPU window; the homelab unit flag merge; 300 W A/B;
  any upstream PR.
- **Never:** revive the radiance_draft MTP controller (closed by owner decision);
  edit the rejection sampler; ship without per-category numbers; ad-hoc
  pct/kubectl as persistent state; delete the inert lineage backport (still
  serves the 0.27.x stack line).

## Success criteria (approved defaults)

1. Weighted decode **≥ +2 %** vs control, sign-stable across ≥2 interleaved pairs
   in one window (quote the spread; dynab lesson: one pair is never a result).
2. No category worse than **−2 %**; chat replay ≥ 0.98x at 4k/8k/16k.
3. GSM8K-200 ≥ baseline − 0.01; seeded greedy replay identical (lossless).
4. KV pool stays 425532; startup ≤ 1.5x baseline; production restored 200.
5. If 1–4 fail or tie: NOT-ADOPTED.md entry with the data, project closed.

## Risks (post-audit)

- **"dspark only" docstring:** may hide an unsupported dflash assumption not
  visible statically; Phase-1 smoke is the cheap decider.
- **k=7 is shallow:** trimming saves little at conc 1; a conc-1-only tie is a
  legitimate kill. Concurrency levels 2/4/8 carry the hypothesis.
- **Estimator cold start:** generic coefficients, online fit converges within a
  refit or two — measure warm, second start.
- **Varlen cudagraph replay:** the second manager check
  (get_varlen_cudagraph_unsupported_backend) must pass for R4D_HYBRID under
  FULL_DECODE_ONLY; smoke decides.
- **GDN state planning under trimmed batches:** statically cleared
  (gdn_attn supports mismatch), empirically confirmed by GSM8K + greedy replay.

## Open questions

1. After a win: flag default-on in the prod unit, or opt-in per deployment?
2. Window budget: 2–3 GPU windows for the interleaved A/B — acceptable?
3. Flag-on smoke may reveal the estimator needs a model-specific refit window
   (how many warm steps before the A/B arms start) — decide from Phase-1 logs.
