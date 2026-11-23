"""sly/inkling2/radiance_inkling2_adaptive_k.py: adaptive DSpark verify window policy.

Pure-function tests: decide_k (hysteresis + cooldown) and update_ema.
No GPU, no vLLM import.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radiance_inkling2_adaptive_k import State, decide_k, state_from_env, update_ema


# --------------------------------------------------------------------------- decide_k

def test_low_acceptance_goes_to_k3():
    """EMA below 2.0 with k=5: drop to k=3."""
    assert decide_k(ema_accepted=1.5, current_k=5, steps_since_switch=20) == 3


def test_medium_acceptance_stays_k5():
    """EMA at the 2.8 boundary with k=5: no drop (strict <)."""
    assert decide_k(ema_accepted=2.8, current_k=5, steps_since_switch=20) == 5


def test_high_acceptance_goes_to_k7():
    """EMA above 4.0 with k=5: rise to k=7."""
    assert decide_k(ema_accepted=4.5, current_k=5, steps_since_switch=20) == 7


def test_k7_drops_to_k5_on_low_ema():
    """EMA below 3.4 with k=7: drop to k=5."""
    assert decide_k(ema_accepted=2.5, current_k=7, steps_since_switch=20) == 5


def test_k7_stays_on_high_ema():
    """EMA above 3.4 with k=7: no change."""
    assert decide_k(ema_accepted=4.2, current_k=7, steps_since_switch=20) == 7


def test_k3_rises_to_k5_on_high_ema():
    """EMA above 3.0 with k=3: rise to k=5."""
    assert decide_k(ema_accepted=3.5, current_k=3, steps_since_switch=20) == 5


def test_k3_stays_on_low_ema():
    """EMA below 2.0 with k=3: no change."""
    assert decide_k(ema_accepted=1.0, current_k=3, steps_since_switch=20) == 3


# ------------------------------------------------------------------- hysteresis band

def test_hysteresis_k5_to_k7_needs_ema_above_4():
    """EMA in (3.5, 4.0] with k=5: no switch (hysteresis band)."""
    assert decide_k(ema_accepted=3.8, current_k=5, steps_since_switch=20) == 5


def test_hysteresis_k5_to_k3_needs_ema_below_2p8():
    """EMA in [2.8, 4.0] with k=5: no switch (hysteresis band)."""
    assert decide_k(ema_accepted=2.9, current_k=5, steps_since_switch=20) == 5


def test_hysteresis_k7_to_k5_needs_ema_below_3p4():
    """EMA in [3.4, ...) with k=7: no switch (hysteresis band)."""
    assert decide_k(ema_accepted=3.5, current_k=7, steps_since_switch=20) == 7


def test_hysteresis_k3_to_k5_needs_ema_above_3():
    """EMA in [2.0, 3.0) with k=3: no switch (hysteresis band)."""
    assert decide_k(ema_accepted=2.8, current_k=3, steps_since_switch=20) == 3


# ------------------------------------------------- cluster calibration (FACTS values)

def test_low_cluster_drops_from_k7():
    """prose/chat/reasoning acceptance (~2.5-3.2) all drop k=7 -> k=5."""
    for ema in (2.52, 2.67, 3.16):
        assert decide_k(ema, 7, 20) == 5, ema


def test_high_cluster_stays_k7():
    """code/math/json/file_edit acceptance (~3.8-4.6) keep k=7."""
    for ema in (3.79, 3.97, 4.45, 4.63):
        assert decide_k(ema, 7, 20) == 7, ema


def test_prose_chat_reach_k3():
    """The lowest acceptance (prose/chat ~2.5-2.7) drops k=5 -> k=3 (k3 reachable)."""
    for ema in (2.52, 2.67):
        assert decide_k(ema, 5, 20) == 3, ema


def test_reasoning_stays_k5():
    """reasoning (~3.16) drops to k=5 but not k=3."""
    assert decide_k(3.16, 5, 20) == 5


# ------------------------------------------------------------------------ cooldown

def test_cooldown_prevents_immediate_switch():
    """Within cooldown: no switch even if EMA crosses threshold."""
    assert decide_k(ema_accepted=1.0, current_k=5, steps_since_switch=3) == 5
    assert decide_k(ema_accepted=5.0, current_k=3, steps_since_switch=5) == 3


def test_cooldown_boundary():
    """At exactly the cooldown step: switch is allowed."""
    # COOLDOWN = 10, so steps_since_switch == 10 is the first switch-eligible step
    assert decide_k(ema_accepted=1.0, current_k=5, steps_since_switch=10) == 3


# ----------------------------------------------------------------------- update_ema

def test_ema_init():
    """First observation: EMA = value."""
    assert update_ema(None, 3.5) == 3.5


def test_ema_update():
    """Subsequent observations: EMA = alpha * new + (1 - alpha) * old."""
    ema = update_ema(3.0, 5.0, alpha=0.1)
    assert abs(ema - (0.1 * 5.0 + 0.9 * 3.0)) < 1e-9
    assert abs(ema - 3.2) < 1e-9


def test_ema_converges():
    """Repeated observations converge to the observed value."""
    ema = None
    for _ in range(200):
        ema = update_ema(ema, 4.0)
    assert abs(ema - 4.0) < 0.01


# --------------------------------------------------------------------- env + state

def test_parse_env_off_by_default(monkeypatch):
    monkeypatch.delenv("RADIANCE_INKLING2_ADAPTIVE_K", raising=False)
    assert state_from_env() is None


def test_parse_env_adaptive_set(monkeypatch):
    monkeypatch.setenv("RADIANCE_INKLING2_ADAPTIVE_K", "3,5,7")
    st = state_from_env()
    assert st is not None
    assert st.ks == (3, 5, 7)
    assert st.k == 7  # starts at the max window


def test_parse_env_fixed_k(monkeypatch):
    monkeypatch.setenv("RADIANCE_INKLING2_ADAPTIVE_K", "5")
    st = state_from_env()
    assert st is not None and st.ks == (5,)
    # fixed: never changes
    for _ in range(50):
        st.observe(1.0)
        assert st.next_k() == 5


def test_parse_env_invalid_is_off(monkeypatch):
    for bad in ("", "2", "3,4", "9", "3,5,7,9", "abc"):
        monkeypatch.setenv("RADIANCE_INKLING2_ADAPTIVE_K", bad)
        assert state_from_env() is None, bad


def test_state_cold_start_stays_at_max():
    st = State((3, 5, 7))
    assert st.next_k() == 7  # no observations yet: stay at the max window


def test_state_switches_with_cold_and_warm_ema():
    st = State((3, 5, 7))
    # low acceptance: 7 -> 5 after cooldown, 5 -> 3 after cooldown
    for _ in range(60):
        st.observe(1.5)
        st.next_k()
    assert st.k == 3
    # high acceptance: back up
    for _ in range(200):
        st.observe(5.0)
        st.next_k()
    assert st.k == 7


def test_state_no_oscillation():
    """EMA pinned near the 5->7 threshold: k must not flip every step."""
    st = State((3, 5, 7))
    for _ in range(50):
        st.observe(4.0)
        st.next_k()
    flips = 0
    prev = st.k
    for _ in range(100):
        st.observe(4.0)
        k = st.next_k()
        flips += k != prev
        prev = k
    assert flips == 0


# -------------------------------------------------------------------------- mutants

def test_mutants_killed():
    """Every constant mutation must change a pinned behaviour (mutant killed)."""
    import radiance_inkling2_adaptive_k as m

    cases = [
        # (mutant: attr -> value, original behaviour the mutant must break).
        # Probes sit just inside the no-switch side of each threshold, so a shift
        # of the threshold across the probe flips the decision.
        ("_K5_TO_K7", 2.9, lambda: m.decide_k(3.9, 5, 20) == 5),   # 3.9<4.0 stays; mutant 2.9 -> rises
        ("_K5_TO_K3", 3.8, lambda: m.decide_k(2.9, 5, 20) == 5),   # 2.9>2.8 stays; mutant 3.8 -> drops
        ("_K7_TO_K5", 4.4, lambda: m.decide_k(3.5, 7, 20) == 7),   # 3.5>3.4 stays; mutant 4.4 -> drops
        ("_K3_TO_K5", 1.9, lambda: m.decide_k(2.9, 3, 20) == 3),   # 2.9<3.0 stays; mutant 1.9 -> rises
        ("COOLDOWN", 5, lambda: m.decide_k(1.0, 5, 7) == 5),       # cooldown blocks; mutant 5 -> switches
    ]
    for attr, val, check in cases:
        orig = getattr(m, attr)
        setattr(m, attr, val)
        try:
            assert not check(), f"mutant {attr}={val} survived: pinned behaviour unchanged"
        finally:
            setattr(m, attr, orig)

    # alpha mutant: the EMA update value must move
    orig = m.ALPHA
    m.ALPHA = 0.5
    try:
        assert abs(m.update_ema(3.0, 5.0, alpha=m.ALPHA) - 4.0) < 1e-9
        assert abs(m.update_ema(3.0, 5.0, alpha=m.ALPHA) - 3.2) > 1e-9
    finally:
        m.ALPHA = orig
