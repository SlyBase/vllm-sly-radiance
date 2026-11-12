# Analysis: Cactus repetition lock-in (#164) — why fade + cap is not enough, and an escape floor that holds everywhere

Issue: [#164](https://github.com/SlyBase/vllm-sly-radiance/issues/164) · Branch: `fix/164-cactus-escape-mixture` · Date: 2026-11-11
Status: **CPU-verified** (patch wiring, unit tests, Triton-interpreter kernel check, closed-form simulation). GPU A/B is planned, not run — see [GPU test plan](#8-gpu-test-plan-after-the-card-is-free).

## 1. Summary

Cactus accepts a drafted token when `h(n) > u·p(n)` with `h(n) = γ* = min(q(n) + √(2δ·q(n)(1−q(n))), 1)`.
Whenever `h(n) ≥ p(n)` the acceptance probability is **exactly 1**: the target's own sampling step is
skipped and the drafter's token is emitted unconditionally. In a repetition loop the target becomes
very confident in the loop token (`q → 1`) and the drafter predicts it confidently too (`p → 1`).
That corner is **absorbing**: with `δ = 0.5`, every state with `q ≥ 0.5` and `p ≤ h` has zero escape
probability, so the loop can never end by sampling. The KL divergence budget does not see this —
the corner has KL ≈ 0.001 nats, far below any budget, while escape is exactly zero.

The 1.9.0 mitigations attack the corner only partially:

- **Cap** (`h ≤ 1 − r(1−q)`) guarantees escape `r(1−q)` **only at `p = 1`**. For a probabilistic
  drafter with `p ≤ 1 − r(1−q)` (e.g. `q = 0.90, p = 0.92, r = 0.5` → `h = 0.95 ≥ p`) acceptance is
  still 1. The production drafter is `draft_sample_method probabilistic`, i.e. exactly this case.
- **Fade** is length-based, not state-based. A loop that locks in before 4096 output tokens runs at
  full `δ` while it locks; after 16384 tokens the rule is stock, but by then `q → 1` makes even the
  stock escape `1 − q/p` tiny, and the loop keeps reinforcing `q`.

The fix in this branch is an **escape mixture**: with probability `ε` the exact stock test runs
instead of the Cactus test (same uniform `u`, one comparison against the mixed threshold).
Per-step acceptance becomes the convex mix `(1−ε)·min(1, h/p) + ε·min(1, q/p)`, which gives two
exact guarantees at **every** `(q, p)`, every length, independent of drafter confidence:

1. output distribution `= (1−ε)·out_cactus + ε·q` (verified to 2e-16 TV),
2. escape probability `≥ ε ·` stock escape (verified to 9e-17).

`ε = 0` (default) and `δ = 0` stay bit-exact to today; `ε = 1` is bit-exact stock. The mixture also
shrinks the per-step KL drift by convexity: `KL(out_mix, q) ≤ (1−ε)·KL(out_cactus, q)`.
Measured acceptance cost on benign high-agreement states: −0.006 at `ε = 0.1`, −0.016 at `ε = 0.25`.

What it does **not** fix: loops the stock rule itself enters (2/16 runs on the #164 repro prompt).
The guarantee is "never more absorbing than `ε`-scaled stock", not "no loops ever".

## 2. What has been tried (verified timeline)

Sources: git tags `v1.5.1`/`v1.9.0`, `CHANGELOG.md`, releases API, all 28 issue comments.

| When | Version | Change | Evidence / outcome |
|---|---|---|---|
| 10-17 | 1.5.1 | Cactus knob `δ`, default 0.5, `γ*` with clip, **no cap, no fade** (commits `5134f5f`, `0042495`) | EcceGratum: loops, rolled back to 1.3.2; issue body: 151K-token run, 98.6–99.9 % acceptance, never ends; `δ = 0` completes |
| 10-21 | 1.6.0 | vLLM 0.31→0.32 re-anchor, Cactus rule unchanged | orochifj controlled A/B `δ = 0` vs `0.5`, same server/task: lock-in only at 0.5 |
| 10-23/24 | 1.8.0/1.8.1 | — | goim01: loops on 3 models at xhigh, "guaranteed on long prompts"; `δ = 0` fixes all; **NSD θ = 0.95 loops too**; anknetau: `δ = 0` fixes but "removes most of the speedups" |
| 10-25 | — | **Streak / lock-in guard** (`2fcfa2a`, branch `fix/164-cactus-lockin-guard`): after 8 all-accept verify steps, 32 steps on exact stock | commit message already names the corner: "Cactus lifts h to 1 for q >= 1/(1+2δ), so inside a repetition the target never rejects and the stock rule's escape (residual) is gone". **GPU: did not help, 2/3 loops with and without** (`944a9fa`): "loops form at partial acceptance and, once deep, the target itself has q ~ 1" |
| 10-25 | — | Guard **replaced by length fade** (`944a9fa`), then **cap** (`9d77308`), merged as fade+cap (`06ce04b`) | measured on the #164 request, 16–32 seeds: δ 0.5 loops 11/32, stock 2/16, Cactus only in first 4096 output tokens 0/16 |
| 10-25 | — | slydlake: "NSD is even worse when it comes to looping"; hipfire ships Cactus opt-in only; "the looping comes from the drift, which accumulates over the sequence" | issue comments |
| 10-26 | 1.9.0 | **Length fade** 4096→16384 + **cap r = 0.5** ships | slydlake sweep (README): loops 11/32 → cap alone 5/16 → fade+cap **1/16**; hard cut at 4096: 0/16; stock itself: 2/16. orochifj: request.json seed 1, 32K budget → no lock-in, tool call at 14,530 tokens |
| 10-26/27 | — | **Context fade over absolute position** (PR #207, branch `fix/cactus-fade-by-position`, **closed unmerged** 10-27) | production evidence in the PR: Swift-1.5 session, ~170k context, agent turn locked into a reasoning loop after **~3k output tokens**, per-position acceptance 1.0 for the last ~300 tokens, full δ 0.5, **cap active and not enough**. Closed because defaults were provisional pending a GPU loop replay |
| 10-27 | — | slydlake: preserve-thinking + Cactus raises loop probability | issue comment |
| 10-28 | — | powerleizhou: **1.9.0 still locks in at xhigh** (fade+cap, one arm FADE_END=8192); `δ = 0` resolves. slybase-bot log analysis: acceptance climbs 3.45 → 6.94 → 7.95 into the cut; 15 windows ≥ 7.95 *without* a cut → high acceptance alone is not sufficient | issue comments + attachments |
| 10-29 | 1.9.1 | Chat-template/truncation fix (post-compression stream cuts — a **separate** bug) | powerleizhou 10-30: truncation fix verified, 6/6 clean stops; goim01 10-31: ~400M tokens, 3 agents, no loops |
| 11-03 | 1.9.1 | — | EcceGratum: "still present to a lesser degree"; anknetau: "unusable at anything over CACTUS 0. Tried 0.1, 0.25 and 0.5" |
| 11-07 | 1.10.0 | Cactus rule unchanged | EcceGratum 11-09: **still loops on 1.10.0**; `δ = 0` + retry does not loop; easy repro: multi-turn story with choices, verbose parts |
| 11-10 | 1.11.0 | lm_head hot rows etc., Cactus rule unchanged | current main |

Reading: every mitigation so far tunes **how much** Cactus is applied — a streak detector (guard),
the output length (fade), the absolute position (ctx fade, unmerged), or the threshold height (cap).
None guarantees that an escape **exists** in the locked-in state at every token. The guard failed
because it waited for an all-accept streak that partial-acceptance loops never produce; the fades
miss early-onset loops (the #207 production turn locked at ~3k output tokens, before the 4096 fade
start); the cap misses probabilistic drafters (§4.2). Users still hit the absorbing corner on
1.10.0/1.11.0. NSD (accept more inside the nucleus, same residual) makes escape worse, consistent
with goim01's and slydlake's reports.

## 3. The rule, exactly

**Repo (v1.5.1 and main, verified from the tags)** — only the acceptance threshold changes; the
residual/recovery distribution stays stock (`norm(max(q − p, 0))`):

```python
q_n = tl.exp(target_logprob)
h_n = tl.minimum(tl.maximum(q_n + tl.sqrt(2.0 * req_delta * q_n * (1.0 - q_n)), 0.0), 1.0)
# 1.9.0+ only:
h_n = tl.minimum(h_n, 1.0 - cactus_cap_r * (1.0 - q_n))
log_h_n = tl.where(req_delta == 0.0, target_logprob, tl.log(h_n))
accepted = log_h_n > tl.log(u) + draft_logprob
```

**Paper** ([arXiv 2604.04987](https://arxiv.org/abs/2604.04987), Eq. 15/16, Corollary 5): the same
`γ*`, but `h` is a **full distribution** — `h(n) = γ*`, `h(i≠n) = (1−γ*)/(1−q(n))·q(i)` — and the
recovery distribution is `g = (h − p·φ) / E_p[1−φ]` with `φ = min(1, h/p)`, so the per-step output
equals `h` exactly and `D_f(h‖q) ≤ δ` by construction (Theorem 3).

Two deviations of the repo from the paper, both verified by enumeration (`sly/tests/cactus_escape_sim.py`, part B):

1. **Threshold-only**: the repo keeps the stock residual instead of `g`. The per-step output KL
   profile is nevertheless almost identical to the paper-faithful construction (mean 0.379 vs 0.378
   nats, p99 0.880 vs 0.878, max 1.272 both, over 20 000 random `(q,p)` pairs at δ = 0.5).
2. **`γ*` is the paper's small-δ approximation** (Eq. 15 is labelled "approximated solution"; the
   exact binary optimum solves `q·f(γ/q) + (1−q)·f((1−γ)/(1−q)) = δ`). At δ = 0.5 the tail exceeds
   the budget: 53 % of random pairs have output KL > Γ(0.5) = ln 2 ≈ 0.347 nats, max 1.27. This is
   true for the paper's own construction too — it is not a repo bug, but it means "divergence stays
   bounded by delta" is a worst-case statement about `h`, not a per-step guarantee about the output.

Neither deviation is the loop cause. The loop cause is structural and shared with the paper:

## 4. Why it locks in

### 4.1 The zero-escape corner

Acceptance probability is `a(n) = min(1, h(n)/p(n))`. Whenever `h(n) ≥ p(n)`, `a(n) = 1`: the
uniform is never consulted, the target's residual is never sampled, the drafter's token is emitted
unconditionally. For `γ*`:

```
γ* ≥ 1  ⟺  q + √(2δq(1−q)) ≥ 1  ⟺  q ≥ 1/(1+2δ)
```

| δ | zero escape for q ≥ |
|---|---|
| 0.1 | 0.833 |
| 0.25 | 0.667 |
| 0.5 | **0.500** |

(Verified identical in `v1.5.1` and main — the formula never changed; 1.9.0 added cap + fade on top.)
At the image default δ = 0.5, **any state where the target gives the loop token ≥ 50 % and the
drafter gives it ≤ h is absorbing**. A verbatim repetition loop sits exactly there: q ≈ 0.9–0.999,
p ≈ 0.9–1. The issue's 99 % acceptance (≈7.9/8 tokens per step) is the signature: almost every draft
token is accepted unconditionally.

### 4.2 The cap does not cover the probabilistic drafter

With cap `r`, `h = min(γ*, 1 − r(1−q))`. At `p = 1` this gives escape `1 − h = r(1−q) > 0` — half the
stock escape at `r = 0.5`. But for `p < 1`:

```
escape = 0  ⟺  p ≤ 1 − r(1−q)
```

At `q = 0.90, r = 0.5`: every `p ≤ 0.95` is still absorbing. At `q = 0.98, p = 0.99`: `h = 0.99 ≥ p`,
absorbing. The production configuration is `draft_sample_method probabilistic` — p is close to 1 but
not 1, which is precisely the uncovered band. Simulation part D (exact enumeration):

| state (δ=0.5, cap r=0.5) | KL(out_cactus, q) | escape/step cactus | stock | mix ε=0.25 | E[steps to escape] cactus / stock / mix |
|---|---|---|---|---|---|
| q=0.999, p=1.00, no cap | 0.00100 nats | **0** | 1.0e-3 | 2.5e-4 | **∞** / 1000 / 4000 |
| q=0.999, p=1.00, cap | 0.00015 nats | 5.0e-4 | 1.0e-3 | 6.25e-4 | 2000 / 1000 / 1600 |
| q=0.90, p=0.92, cap | 0.00237 nats | **0** | 2.2e-2 | 5.4e-3 | **∞** / 46 / 184 |
| q=0.98, p=0.99, cap | 0.00312 nats | **0** | 1.0e-2 | 2.5e-3 | **∞** / 99 / 396 |

### 4.3 KL budget ≠ escape

The table above is the mathematical core of #164: the absorbing corner has KL **0.001–0.003 nats**,
two orders of magnitude below Γ(0.5) ≈ 0.347. A divergence budget controls average drift; it says
nothing about the *support* of the output distribution. `out = δ_loop-token` vs `q = 0.999` on that
token is a tiny KL — and zero escape. Any mitigation that only shrinks δ (fade) or lowers h (cap)
still leaves a region where `h ≥ p`; the loop lives in that region. **The quantity to bound is the
escape probability, not the divergence.**

### 4.4 Self-reinforcement, and why fade arrives too late

Inside a verbatim loop, each emitted repetition makes the target more confident in the next one:
`q` rises toward 1. Stock escape `1 − min(1, q/p)` shrinks as `q → p`; Cactus escape is already 0.
This is a positive feedback: drift → higher q → less escape → more drift. slydlake's "drift
accumulates over the sequence" and the bot's log analysis (acceptance climbing 3.45 → 6.94 → 7.95
exactly at the cut) are this feedback measured.

Fade lowers δ linearly from 4096 to 16384 output tokens. Two gaps:

1. **Onset**: loops that start before 4096 tokens lock in at full δ. The 1.9.0 sweep's "all loops
   after 7.7k tokens" is corpus-specific: PR #207 documents a production agent turn (~170k context)
   that locked after ~3k output tokens with acceptance 1.0 for the last ~300 tokens, cap active.
   Agent turns are new requests with short outputs on long contexts — the output-length fade never
   starts there.
2. **Late rescue**: after 16384 tokens the rule is stock, but a locked-in loop has q ≈ 0.999+, so
   stock escape is ≈ 1e-3/step — ~1000 steps (minutes) to break, while q keeps rising. Users stop
   the run first. And the stock rule itself loops 2/16 on this prompt: fade can at best reach the
   stock baseline, which is not loop-free.

### 4.5 Why NSD is not the answer

NSD ([arXiv 2610.07822](https://arxiv.org/abs/2610.07822)) *adds* acceptance (draft inside the target
nucleus) with the same stock residual. More acceptance at unchanged residual means less residual
sampling — escape can only shrink. Consistent with goim01 (NSD θ=0.95 loops) and slydlake ("NSD is
even worse"). The two knobs are mutually exclusive in the patch for this reason.

### 4.6 Why the streak guard failed, and what that implies

The 10-25 lock-in guard (`2fcfa2a`) switched a request to exact stock for 32 steps after 8
consecutive all-accept verify steps. On GPU it changed nothing (2/3 loops with and without,
`944a9fa`), for two reasons that the commit itself names:

1. **Loops form at partial acceptance.** The guard's trigger is an *all-accept streak*. With a
   probabilistic drafter, the loop token is accepted unconditionally (`h ≥ p`) while the other
   k−1 draft tokens still get rejected now and then — the streak never reaches 8, the guard never
   fires, and the loop token keeps being emitted with probability 1.
2. **Once deep, stock itself has q ~ 1.** A late stock window faces escape `1 − q/p ≈ 1e-3`; the
   loop has already ratcheted q up (§4.4).

Both failures point at the same design rule: the escape must be **per-token and unconditional**,
not streak-triggered and not deferred. A per-token floor `a(n) ≤ 1 − ε·(1 − min(1, q/p))` acts from
the first loop token, at any acceptance pattern, before q can ratchet — which is exactly the
mixture in §5. It also explains why the guard's diagnosis was right but its mechanism could not
work: detecting the corner is not the same as having an escape in it.

## 5. The fix: escape mixture (`RADIANCE_CACTUS_ESCAPE`)

### 5.1 Mechanism

One extra kernel argument `cactus_escape` (env `RADIANCE_CACTUS_ESCAPE`, default `0`, validated to
`[0,1]` at startup). The acceptance test becomes:

```python
acc_cactus = log_h_n > tl.log(u) + draft_logprob          # today's test, unchanged
a_h = tl.exp(tl.minimum(log_h_n - draft_logprob, 0.0))    # min(1, h/p)
a_q = tl.exp(tl.minimum(target_logprob - draft_logprob, 0.0))  # min(1, q/p)
acc_mix = u < (1.0 - cactus_escape) * a_h + cactus_escape * a_q
acc_stock = target_logprob > tl.log(u) + draft_logprob
accepted = tl.where(cactus_escape == 1.0, acc_stock,
                    tl.where((cactus_escape > 0.0) & (req_delta > 0.0),
                             acc_mix, acc_cactus))
```

Comparing the same uniform against the mixed threshold is distributionally identical to "with
probability ε run the exact stock test, else the Cactus test". The residual path is untouched, so
with `out_c` / `out_s` the per-step outputs of the two rules:

```
out_mix = p·a_mix + (1 − Σ p·a_mix)·residual
        = (1−ε)·out_c + ε·out_s
        = (1−ε)·out_c + ε·q          (Leviathan identity: out_s = q)
```

**Why mix the acceptance probabilities and not the thresholds.** The seemingly equivalent
shortcut `h_mix = (1−ε)·h + ε·q` compared against the same `u·p` does **not** have these
guarantees: `min(1, ·)` is not linear, and in the band `q < p < h` the mixed threshold can still
sit above `p` (δ=0.5, q=0.9, p=0.95, ε=0.25: `h_mix = 0.975 ≥ p` → accept with probability 1),
so the escape floor is violated exactly where the loop sits. Enumeration over 20000 random
(q, p, ε, cap) pairs: threshold mixing breaks the escape floor on 5389 pairs and the output
identity (TV up to 0.23); the shipped probability mix has 0 violations. The mix must happen
**after** the clip, on the acceptance probabilities — do not "simplify" the kernel to a
threshold mix.

Scope is inherited from the original Cactus patch (v1.5.1 release notes): only the **non-greedy
Leviathan branch** of the V2 rejection kernel is touched; greedy, synthetic-rate and
block-verification paths stay untouched, and the V1 sampler is not used by this image's production
config. Greedy requests are unaffected by any of this.

### 5.2 Guarantees (all verified by enumeration, part A, 4000 random pairs)

| Property | Statement | Max error |
|---|---|---|
| Stock identity | `out_s = q` | TV 1.7e-16 |
| Mixture identity | `out_mix = (1−ε)·out_c + ε·q` | TV 1.8e-16 |
| **Escape floor** | `1 − a_mix(n) ≥ ε·(1 − a_stock(n))` for every `(q,p)` | violation 9e-17 |
| Drift | `KL(out_mix, q) ≤ (1−ε)·KL(out_c, q)` (convexity of KL) | — |
| Bit-exactness | `ε=0` → today's comparison; `δ=0` → stock; `ε=1` → stock log test | interpreter: exact |

The escape floor holds **at every (q, p), every output length, for any drafter confidence** — it
does not depend on p ≈ 1 (cap) or on the request being long (fade). Expected time to escape a loop
is at most `1/ε` × the stock rule's: a loop stock breaks in ~100 steps breaks in ≤ ~400 at ε = 0.25.
Where stock itself always accepts (`q ≥ p`) the mixture also always accepts — it does not invent
escape where the target model itself has none (§4.4, stock's own 2/16).

### 5.3 Cost

Acceptance loss is `ε·(a_c − a_s)` per token, zero when the rules agree (q ≈ p, the common case).
Simulation on 20 000 benign high-agreement pairs: mean −0.0062 (ε=0.1), −0.0155 (ε=0.25); max
−0.035. Against the 1.9.0 sweep's fade+cap acceptance 0.604 (stock 0.532), ε = 0.25 costs ≈ 0.015 →
≈ 0.589, still ≈ +10 % over stock (estimate, to be confirmed on GPU). The kernel computes two extra
`exp` per token even at ε = 0; the result path is bit-exact, the ALU cost is negligible in a
memory-bound kernel.

### 5.4 Interaction with fade and cap

Orthogonal and composable: fade still lowers δ with length, cap still lowers h, the mixture floors
escape regardless. The interpreter check covers fade+escape together. Recommended starting point
for agent workloads: keep fade+cap defaults, add `ε = 0.25`; the GPU plan (§8) measures whether
ε = 0.1 suffices.

### 5.5 Upgrade path

The acceptance hunk matches three shapes (stock, 1.5.1–1.8.1, 1.9.0–1.11.0 fade), so a launcher
re-run on an older image upgrades to the escape mixture in one pass. Verified: old-patch → new-patch
produces a kernel byte-identical to a fresh apply, second run is all-NOOP.

## 6. Verification done (CPU only, GPU was busy until ~20:00)

Environment: Python 3.13 venv, torch CPU, triton (interpreter mode), pinned vLLM **v0.32.0** sources
(`rejection_sampler_utils.py`, `rejection_sampler.py`) — the same files the Dockerfile patches.

| Check | Result |
|---|---|
| Fresh apply on pinned source (14 hunks) | PASS |
| Idempotence (second apply) | all NOOP |
| Upgrade: v1.11.0 patch → new patch | kernel byte-identical to fresh apply, then NOOP |
| `ast.parse` of patched files | OK |
| `sly/tests/test_cactus_escape_cpu.py` (8 tests: identities, escape floor, corner cases, patch wiring, env validation, interpreter launch) | **8/8 PASS** |
| `sly/tests/cactus_escape_kernel_check.py` (Triton interpreter, patched V2 kernel vs float64 reference: ε=0/0.25/0.5/1, cap on/off, fade+escape, δ=0) | **PASS**, ε=1 and δ=0 bit-exact stock |
| Regression `sly/tests/test_cactus_cpu.py` (structure + fade launcher) | OK |
| Regression `sly/tests/cactus_fade_kernel_check.py` against the new patch | PASS (fade/cap unchanged at ε=0) |
| `sly/tests/cactus_escape_sim.py` parts A–D (closed-form enumeration + Markov chain) | identities ≤ 2e-16; tables in §4 |

CI: `pytest sly/tests -q` collects the new test file automatically; without triton the interpreter
test skips cleanly — the math and patch-wiring tests run everywhere. `ci/patch_dryrun.sh` needs no
change (no new patch file).

**Re-verification (2026-11-11, independent pass over `925e6c4`, CPU only):** full CI-equivalent
suite `pytest sly/tests` **275 passed / 21 skipped**, `ci/tests` 36 OK, `check_consistency.py` OK;
simulation re-run byte-identical to `cactus_escape_sim_results.txt`; fresh/idempotent apply and
both upgrade paths (1.5.1-era shape 2, fade-era shape 3) byte-identical to a fresh apply;
escape-floor sweep over 20000 random (q, p, ε, cap) pairs: 0 violations for the shipped
probability mix. The commit message and the CHANGELOG/README wording originally described a
*threshold* mix `h_mix = (1−ε)h + εq` and quoted simulation numbers from the wrong table cell
(83.8 % is the cap column at q=0.999, p=0.9999, not stock at q=0.99); both corrected — the kernel
itself was already the probability mix and needed no change.

Markov-chain evidence (3-cycle attractor, T=400, 400 runs, lock-in % / mean escape step):

| q_c | p_c | stock | cactus δ=.5 | +cap r=.5 | mix ε=.10 | mix ε=.25 |
|---|---|---|---|---|---|---|
| 0.90 | 1.00 | 0 % / 10 | **100 % / ∞** | 0 % / 18 | 0 % / 17 | 0 % / 15 |
| 0.99 | 1.00 | 2 % / 92 | **100 % / ∞** | 11 % / 141 | 9 % / 137 | 7.5 % / 123 |
| 0.90 | 0.99 | 0 % / 10 | 1.8 % / 94 | 0 % / 20 | 0 % / 17 | 0 % / 15 |
| 0.99 | 0.999 | 2.5 % / 89 | **68.5 % / 199** | 12 % / 143 | 8 % / 140 | 6.8 % / 129 |
| 0.999 | 0.9999 | 69 % / 199 | 96.5 % / 198 | 84 % / 193 | 82 % / 191 | 79 % / 198 |
| 0.80 | 0.99 | 0 % / 5 | 1.8 % / 94 | 0 % / 10 | 0 % / 9 | 0 % / 8 |

The last row is the model-inherent regime (§4.4): even stock locks in 69 %; no acceptance rule fixes
that, and the mixture honestly stays near stock instead of pretending to.

## 7. Answers to the issue's questions

1. **Seen with δ > 0?** Yes — reproduced analytically and in simulation; the absorbing corner is a
   property of the rule, not of a specific image. 1.5.1 through 1.11.0 share the same `γ*`.
2. **Delta sweep with long-generation check?** The GPU plan (§8) does exactly this: several sampled
   runs at high reasoning effort, large max_tokens, counting runs that end at max_tokens with
   > 95 % acceptance, plus the escape-rate metric.
3. **README note for agent workloads?** Added in this branch: long agent / long-reasoning workloads
   should use `RADIANCE_CACTUS_DELTA=0` (exact) or, once GPU-validated, `RADIANCE_CACTUS_ESCAPE≥0.25`
   (keeps most of the gain with a guaranteed escape floor). The README's former claim that the cap
   makes zero-escape pinning impossible is corrected — it holds only at p ≈ 1.

## 8. GPU test plan (after the card is free)

Preconditions: R9700 free, image built from this branch (`ci` dry-run + wheel as usual), model
`amd/Qwen3.8-27B-Quark-AWQ-MXFP4` + DFlash2 W4A16 k=7 probabilistic, production launch args from the
issue (fp8 KV, 204800 ctx, T=1.0/top_p=0.95/top_k=20), `request.json` from orochifj's 10-25 comment
as the primary repro, EcceGratum's multi-turn-story prompt as the secondary.

**Arms** (same image, env-only differences, restart between arms):

| Arm | δ | fade | cap r | ε |
|---|---|---|---|---|
| A stock | 0 | — | — | — |
| B today | 0.5 | 4096→16384 | 0.5 | 0 |
| C escape low | 0.5 | 4096→16384 | 0.5 | 0.10 |
| D escape mid | 0.5 | 4096→16384 | 0.5 | 0.25 |
| E escape high | 0.5 | 4096→16384 | 0.5 | 0.50 |
| F no-fade escape | 0.5 | off (END=0) | 0.5 | 0.25 |

F isolates the mixture from the fade: if F ≈ D on loops, the escape floor works even for
early-onset loops (the powerleizhou/EcceGratum case).

**Protocol per arm**: 16 sampled runs of request.json (seed varied, max_tokens 32768,
reasoning_effort xhigh) + 8 runs of the story prompt; output logging ON; record per-run
`vllm:spec_decode_num_accepted_tokens_total / num_draft_tokens_total` in 30 s windows, finish
reason, total tokens, wall time; detect verbatim loops (n-gram repetition ≥ 3 cycles over ≥ 200
tokens) in the logged output. Plus one BetterBench 0.4.0 default-config run per arm for the
acceptance/tok-s cost, and GSM8K-200 at temp 0.7 for quality (as in the 1.9.0 sweep).

**Metrics & success criteria**:

1. Loop rate: D ≤ stock's 2/16 (target 0–1/16); B reproduced as reference (~1–5/16).
2. No run ends at max_tokens with > 95 % acceptance in D/E (the issue's stuck signature).
3. Acceptance: D ≥ 0.57 (≥ +7 % over stock 0.532); tok/s D ≥ 380 (≥ +7 % over stock 354.8).
4. GSM8K-200: D ≥ 0.95 (1.9.0 fade+cap was 0.960).
5. Escape-floor sanity: in logged loop episodes, D shows residual-sampling events (acceptance
   dips below 100 %) where B shows flat ~99 % — the mechanism visibly working.

**Decision rule**: if D meets 1–4, propose image default `RADIANCE_CACTUS_ESCAPE=0.25` (release PR,
VERSION bump by `ci/cut_release.py`); if only E meets them, default 0.5 with the acceptance cost
documented; if none meet 1, keep default 0 and ship the knob + README guidance only.

**Rollback**: env-only knob, default 0 = bit-exact today; no image rebuild needed to disable.

**Cost estimate**: 6 arms × (24 long runs × ~3–15 min + BetterBench ~20 min + GSMQ ~10 min) ≈ 6–10 h
of card time; can be split across nights, arms are independent.

## 9. Recommendation

- Ship the knob now (default 0, zero risk), with the corrected README and this analysis.
- For long agent / long-reasoning workloads today: `RADIANCE_CACTUS_DELTA=0` remains the only
  *exact* option; `RADIANCE_CACTUS_ESCAPE=0.25` is the mathematically sound middle ground
  (escape floor at every state, ≤ 1/ε × stock escape time, ~1.5 points acceptance cost) — promote
  it to the recommended agent setting after the GPU A/B.
- Do not raise δ above 0.5; the zero-escape threshold `1/(1+2δ)` falls with δ (0.5 → q ≥ 0.5,
  1.0 → q ≥ 0.33), enlarging the absorbing region.
- The paper's Γ(δ) budget is not a loop guarantee (§4.3); future acceptance-rule work should state
  an escape floor explicitly, e.g. `a(n) ≤ 1 − ε·(1 − min(1, q/p))`, which is exactly what this
  mixture implements.

## 10. Sources

- Issue #164 body + all 28 comments (orochifj A/B 10-21, request.json 10-25, goim01 NSD 10-25,
  powerleizhou 1.9.0 repro 10-28 + bot log analysis 10-28, EcceGratum 1.10.0 11-09, anknetau 11-03).
- Releases v1.5.1 (10-17) … v1.11.0 (11-10); git tags v1.5.1, v1.9.0 (formula verification).
- Commit history of `sly/patch_cactus.py`: `5134f5f` (knob), `0042495` (default 0.5), `2fcfa2a`
  (lock-in guard, branch `fix/164-cactus-lockin-guard`), `944a9fa` (guard → fade, GPU failure
  numbers), `9d77308` (cap), `06ce04b` (fade+cap; the commit title references #196, but that PR
  object is not resolvable in this repo — the measured numbers are quoted from the v1.9.0 release
  notes), `1a7c9fa` (ctx fade, PR #207, closed
  unmerged, branch `fix/cactus-fade-by-position`).
- `sly/patch_cactus.py`, `sly/patch_nsd.py`, `CHANGELOG.md`, README 1.9.0 sweep table.
- Cactus: [arXiv 2604.04987](https://arxiv.org/abs/2604.04987) (Eq. 15/16, Theorem 3, App. A.3;
  ICLR 2026, Hao & Mou). NSD: [arXiv 2610.07822](https://arxiv.org/abs/2610.07822).
  Leviathan et al., speculative sampling (stock identity). Vivien's provably-optimal lossy variant
  blog (cross-check of the acceptance/recovery algebra).
- Pinned vLLM v0.32.0 `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` (V2 kernel).
- This branch: `sly/tests/cactus_escape_sim.py` (+ `cactus_escape_sim_results.txt`),
  `sly/tests/test_cactus_escape_cpu.py`, `sly/tests/cactus_escape_kernel_check.py`.