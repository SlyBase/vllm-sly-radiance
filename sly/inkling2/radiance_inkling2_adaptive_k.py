"""Adaptive DSpark verify window policy (Inkling-2-Mini, patch_inkling2_adaptive_k).

Pure functions, no torch / no vLLM: the scheduler-side EMA and the k chooser.
The DSpark drafter's checkpoint block size fixes the MAX window at k = 7; a shorter
window (k in {3, 5}) is verified exactly the same way (rejection sampling is exact
for any k), it just reads fewer distinct MoE experts and less dense attention per
step. The chooser trades that step saving against the lost acceptance headroom.

Signal: accepted DRAFTS per step (num_accepted = generated - 1 bonus). Measured at
the full window k=7: prose ~1.5, chat ~1.7, reasoning ~2.2 (low cluster) vs code
~2.8, math ~2.7, json ~3.2, file_edit ~3.5, summarization ~3.4 (high cluster).

CENSORING: at k < 7 only the best k positions are drafted, so accepted-drafts is a
censored LOWER bound -- a high-acceptance workload run at k=3 reads ~2 and can never
signal that it would accept ~3.5 at k=7. A plain EMA therefore locks k low once a
low-acceptance stretch pulls it down. Fix: keep the estimate FULL-WINDOW only (update
it just when k == max) and periodically PROBE the full window to refresh it; right
after a probe, jump straight to the estimate's k (the probe sample is unbiased).

Hysteresis bands (accepted-drafts at the full window): 2.45..2.6 for 5<->7 and
1.8..2.0 for 3<->5, plus a cooldown, keep a borderline workload from ping-ponging.
"""

# Cooldown in steps after a k switch before another switch is allowed.
COOLDOWN = 10
# EMA coefficient per full-window observation (half-life ~6.6 probes/steps).
ALPHA = 0.1
# Probe the full window every N steps while k < max, to de-censor the estimate.
PROBE_INTERVAL = 32

_K3_TO_K5 = 2.0   # k=3 rises to 5 above this
_K5_TO_K3 = 1.8   # k=5 drops to 3 below this
_K5_TO_K7 = 2.6   # k=5 rises to 7 above this
_K7_TO_K5 = 2.45  # k=7 drops to 5 below this


def update_ema(old, accepted, alpha=ALPHA):
    """Exponential moving average of accepted drafts per step."""
    return accepted if old is None else alpha * accepted + (1.0 - alpha) * old


def decide_k(ema_accepted, current_k, steps_since_switch):
    """k for the next step given the full-window EMA, current k and steps since switch.

    Returns one of {3, 5, 7}; a no-op while inside the cooldown.
    """
    if steps_since_switch < COOLDOWN:
        return current_k
    if current_k == 3:
        return 5 if ema_accepted > _K3_TO_K5 else 3
    if current_k == 5:
        if ema_accepted > _K5_TO_K7:
            return 7
        if ema_accepted < _K5_TO_K3:
            return 3
        return 5
    # current_k == 7
    return 5 if ema_accepted < _K7_TO_K5 else 7


def target_k(ema_accepted):
    """The k the full-window estimate points at (region, no hysteresis). Used for the
    post-probe jump, where the sample is unbiased so we can land directly."""
    if ema_accepted < _K5_TO_K3:
        return 3
    if ema_accepted < _K7_TO_K5:
        return 5
    return 7


class State:
    """Running k choice: full-window acceptance EMA + hysteresis + cooldown + probe."""

    def __init__(self, ks):
        self.ks = tuple(ks)
        self.max_k = max(self.ks)
        self.k = self.max_k  # cold start: the full window
        self.ema = None      # FULL-WINDOW acceptance estimate (updated only at k == max)
        self._steps_since_switch = COOLDOWN
        self._steps_since_probe = 0
        self._probe_pending = False

    def observe(self, accepted, drafted):
        # Only full-window observations are unbiased; censored k<max samples would
        # drag the estimate down and lock k low.
        if drafted >= self.max_k:
            self.ema = update_ema(self.ema, accepted)

    def next_k(self):
        if self.ema is None or len(self.ks) == 1:
            return self.k
        # Right after a probe the estimate is unbiased: land directly on its k.
        if self._probe_pending:
            self._probe_pending = False
            t = min(self.ks, key=lambda k: (abs(k - target_k(self.ema)), k))
            if t != self.k:
                self.k = t
                self._steps_since_switch = 0
            self._steps_since_switch += 1
            return self.k
        self._steps_since_probe += 1
        if self.k < self.max_k and self._steps_since_probe >= PROBE_INTERVAL:
            self._steps_since_probe = 0
            self._probe_pending = True
            self.k = self.max_k
            self._steps_since_switch = 0
            return self.k
        new = decide_k(self.ema, self.k, self._steps_since_switch)
        if new != self.k:
            self.k = new
            self._steps_since_switch = 0
        self._steps_since_switch += 1
        return self.k


def state_from_env():
    """State for RADIANCE_ADAPTIVE_K (legacy name RADIANCE_INKLING2_ADAPTIVE_K), or None (off).

    The value is a comma list of allowed k's, each in {3, 5, 7}, at most three
    entries, e.g. "3,5,7" (adaptive) or "5" (fixed). Anything else is off: the
    DSpark checkpoint only graphs the 4/6/8 verify rows, so no other k exists.
    """
    import os
    import sys

    raw = os.environ.get("RADIANCE_ADAPTIVE_K") or os.environ.get("RADIANCE_INKLING2_ADAPTIVE_K", "")
    ks = []
    for part in raw.split(","):
        part = part.strip()
        if part not in {"3", "5", "7"}:
            return None
        ks.append(int(part))
    if not ks or len(ks) > 3 or len(set(ks)) != len(ks):
        return None
    print(f"[radiance.adaptive_k] active: ks={tuple(ks)}", file=sys.stderr, flush=True)
    return State(ks)
