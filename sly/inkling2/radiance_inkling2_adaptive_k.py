"""Adaptive DSpark verify window policy (Inkling-2-Mini, patch_inkling2_adaptive).

Pure functions, no torch / no vLLM: the scheduler-side EMA and the k chooser.
The DSpark drafter's checkpoint block size fixes the MAX window at k = 7; a shorter
window (k in {3, 5}) is verified exactly the same way (rejection sampling is exact
for any k), it just reads fewer distinct MoE experts and less dense attention per
step. The chooser trades that step saving against the lost acceptance headroom,
from a running per-step acceptance estimate.

Hysteresis (per current k, EMA of accepted tokens per step):
  k = 3: rise to 5 when ema >  3.0
  k = 5: rise to 7 when ema >  4.0, drop to 3 when ema <  2.8
  k = 7: drop to 5 when ema <  3.4
The bands (3.4..4.0 for 5<->7, 2.8..3.0 for 3<->5) keep a borderline workload from
ping-ponging; the cooldown does the rest. The down-from-7 threshold (3.4) sits above
the low-acceptance cluster (prose/chat/reasoning ~2.5-3.2) and below the high one
(code/math/json/file_edit ~3.8-4.6); the down-from-5 threshold (2.8) is above the
lowest acceptance (~2.5) so k=3 is actually reachable, not dead code.
"""

# Cooldown in steps after a k switch before another switch is allowed.
COOLDOWN = 10
# EMA coefficient per step (half-life ~6.6 steps).
ALPHA = 0.1

_K3_TO_K5 = 3.0   # k=3 rises to 5 above this
_K5_TO_K3 = 2.8   # k=5 drops to 3 below this
_K5_TO_K7 = 4.0   # k=5 rises to 7 above this
_K7_TO_K5 = 3.4   # k=7 drops to 5 below this


def update_ema(old, accepted, alpha=ALPHA):
    """Exponential moving average of accepted tokens per step."""
    return accepted if old is None else alpha * accepted + (1.0 - alpha) * old


def decide_k(ema_accepted, current_k, steps_since_switch):
    """k for the next step given the EMA, the current k and steps since its switch.

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


class State:
    """Running k choice: one EMA over accepted tokens per step + hysteresis + cooldown."""

    def __init__(self, ks):
        self.ks = tuple(ks)
        self.k = max(self.ks)  # cold start: the full window
        self.ema = None
        self._steps_since_switch = COOLDOWN  # first decision may switch (no wasted cooldown)

    def observe(self, accepted):
        self.ema = update_ema(self.ema, accepted)

    def next_k(self):
        if self.ema is None or len(self.ks) == 1:
            return self.k
        new = decide_k(self.ema, self.k, self._steps_since_switch)
        if new != self.k:
            self.k = new
            self._steps_since_switch = 0
        self._steps_since_switch += 1
        return self.k


def state_from_env():
    """State for RADIANCE_INKLING2_ADAPTIVE_K, or None (off).

    The value is a comma list of allowed k's, each in {3, 5, 7}, at most three
    entries, e.g. "3,5,7" (adaptive) or "5" (fixed). Anything else is off: the
    DSpark checkpoint only graphs the 4/6/8 verify rows, so no other k exists.
    """
    import os

    raw = os.environ.get("RADIANCE_INKLING2_ADAPTIVE_K", "")
    ks = []
    for part in raw.split(","):
        part = part.strip()
        if part not in {"3", "5", "7"}:
            return None
        ks.append(int(part))
    if not ks or len(ks) > 3 or len(set(ks)) != len(ks):
        return None
    return State(ks)
