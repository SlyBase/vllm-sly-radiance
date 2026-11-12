# Decision: Cactus stability/quality default (Issue #164, PR #239)

- **Date:** 2026-11-12
- **Status:** Accepted
- **Deciders:** owner (Kore/Timon), measured by Daidalos
- **Supersedes:** the "GPU A/B pending, δ=0 the safe setting" guidance in the 1.9.0–1.11.0 README

## Context

Cactus speculative acceptance (`RADIANCE_CACTUS_DELTA > 0`) raises tokens/step but can lock long sampled
generations into verbatim repetition loops (#164). v1.9.0 shipped a length **fade** (delta → 0 over
4096→16384 output tokens) plus a length-independent **cap** (`CAP_R=0.5`) as the default mitigation.
PR #239 adds an opt-in **escape mixture** (`RADIANCE_CACTUS_ESCAPE`, eps): with probability eps the exact
stock acceptance test runs instead of Cactus, guaranteeing a non-zero escape chance on every token —
including the absorbing corner (`q >= 1/(1+2*delta)`, `p <= h`) where fade and cap provide none.

A GPU measurement campaign (2026-11-11/12, R9700 210 W) established the facts below.

## Measured evidence

Loop rate on the #164 repro prompt (16–32 sampled runs/arm, rep32 metric, 8 parallel):

| Rule | Loops | Acceptance | tok/s | Note |
|---|---|---|---|---|
| Stock (delta 0) | 2/16 | 0.532 | 355 | exact; still loops |
| Raw Cactus 0.5 (no fade/cap) | 11/32 (34 %) | 0.670 | 443 | Swift; unbounded drift |
| **Fade+Cap (default)** | **1/16 (6 %)** | 0.604 | 398 (+12 %) | stock-level loops, ~49 % of gain |
| Cap alone | 5/16 (31 %) | 0.653 | 436 | no loop reduction (p=0.55) |

AMD Quark stock-model control (same prompt, simulated MXFP4, within-model):

| Rule | Loops |
|---|---|
| Stock (delta 0) | 0/8 |
| Raw Cactus 0.5 | 12/16 (75 %) |
| Raw Cactus + eps 0.25 | 9/16 (56 %, p=0.46 vs raw) |
| Raw Cactus + eps 0.5 | 4/16 (25 %, **p=0.012** vs raw) |
| Fade+Cap | 0/8 |

PR #239 impact A/B (Swift, release image vs PR image, 16 seeds): eps=0 is bit-exact with no measurable
overhead (BB decode 140.1 vs 141.3 t/s); eps=0.25 on the fade+cap default gives no measurable loop
reduction (1/16 vs 2/16, p≈1.0) at ~2–4 % cost. GSM8K-200 sampled (temp 0.7) 0.95–0.97 across all arms.

**Key structural fact:** the fade makes `req_delta = 0` for outputs beyond `FADE_END` (16384 tokens), i.e.
**long outputs are generated bit-exact stock** — drift saturates at 16384 tokens instead of growing
without bound (raw Cactus). This bounds the long-output drift by construction.

## Decision

1. **Keep fade+cap as the default** (`DELTA=0.5 FADE_START=4096 FADE_END=16384 CAP_R=0.5`). It is the
   measured sweet spot: stock-level loop rate, +12 % speed, drift bounded by the fade. It dominates stock
   (same loops, more speed) and is far safer than raw Cactus.
2. **Merge PR #239 as opt-in, `ESCAPE` default 0.** The escape mixture has a demonstrated use case
   (mitigates raw Cactus looping, eps=0.5: 75 %→25 %, p=0.012) and a proven guarantee (CPU + GPU), but adds
   no measurable benefit on top of the already-protected fade+cap default. Ship the knob; do not enable it
   by default.
3. **Document the stability ladder** (README): default fade+cap (balance) → +ESCAPE=0.25 (lock-in
   guarantee, ~2–4 % cost) → DELTA=0 (max exactness, zero drift, −12 % speed).
4. **Version 1.11.1** carries the escape knob + guidance (1.11.0 was already released without it).

## What this does NOT claim

- **No acceptance rule gives 0 % loops.** Stock itself loops (2/16). A hard termination guarantee needs a
  server-side repetition detector that aborts/restarts the request (model-independent) — tracked as a
  separate feature, not part of this decision.
- **Long-output drift is not cleanly quantified empirically.** The true-token dNLL measure is
  content-confounded (different rules generate different text); `kl_score.py` is prefill scoring and inert
  for acceptance rules. The drift is bounded by the fade (exact stock >16384 tokens) and theoretically >0
  below that, but its exact magnitude on long outputs is unmeasured. A kernel-trace per-position KL would
  quantify it; deferred unless production quality complaints arise.
- Loop rates are from one adversarial prompt at n=16–32; small effects (eps=0.25) are not resolvable.

## Revisit triggers

- Production quality complaints on long outputs → port the kernel-trace per-position KL and quantify drift.
- A workload that loops under fade+cap → evaluate +ESCAPE or a server-side loop detector.
- A larger multi-prompt study before any change to the default delta.

## References

- `docs/ANALYSIS-164-cactus-looping.md` — full analysis + attempt history.
- Measurement reports (agent host `/opt/data/analysis/`): `REPORT-cactus-239-gpu-ab-20261112.md`,
  `REPORT-quark-attractor-control-20261112.md`, `REPORT-pr-impact-ab-20261112.md`,
  `REPORT-quark-escape-vs-pure-cactus-20261112.md`, `REPORT-dnll-drift-20261112.md`.
- Raw data: `/opt/data/research/issue164-escape/` (and `quark/`, `pr-ab/`, `dnll/`).