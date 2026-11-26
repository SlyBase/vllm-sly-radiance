"""sly/inkling2/radiance_inkling2_adaptive_k.py: adaptive DSpark verify window policy.

Pure-function tests: decide_k / target_k (hysteresis + cooldown), update_ema, and the
State's full-window de-censoring (probe). No GPU, no vLLM import.

Signal = accepted DRAFTS per step (tpu - 1 bonus). Full-window k=7 values:
prose ~1.5, chat ~1.7, reasoning ~2.2 (low) vs code ~2.8, json ~3.2, file_edit ~3.5 (high).
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from radiance_inkling2_adaptive_k import (
    State, decide_k, state_from_env, target_k, update_ema,
)


# --------------------------------------------------------------------------- decide_k

def test_low_acceptance_goes_to_k3():
    assert decide_k(ema_accepted=1.5, current_k=5, steps_since_switch=20) == 3


def test_medium_acceptance_stays_k5():
    assert decide_k(ema_accepted=2.0, current_k=5, steps_since_switch=20) == 5


def test_high_acceptance_goes_to_k7():
    assert decide_k(ema_accepted=2.8, current_k=5, steps_since_switch=20) == 7


def test_k7_drops_to_k5_on_low_ema():
    assert decide_k(ema_accepted=2.0, current_k=7, steps_since_switch=20) == 5


def test_k7_stays_on_high_ema():
    assert decide_k(ema_accepted=2.8, current_k=7, steps_since_switch=20) == 7


def test_k3_rises_to_k5_on_high_ema():
    assert decide_k(ema_accepted=2.2, current_k=3, steps_since_switch=20) == 5


def test_k3_stays_on_low_ema():
    assert decide_k(ema_accepted=1.5, current_k=3, steps_since_switch=20) == 3


# ------------------------------------------------------------------- hysteresis band

def test_hysteresis_k5_to_k7():
    assert decide_k(ema_accepted=2.5, current_k=5, steps_since_switch=20) == 5


def test_hysteresis_k5_to_k3():
    assert decide_k(ema_accepted=1.9, current_k=5, steps_since_switch=20) == 5


def test_hysteresis_k7_to_k5():
    assert decide_k(ema_accepted=2.5, current_k=7, steps_since_switch=20) == 7


def test_hysteresis_k3_to_k5():
    assert decide_k(ema_accepted=1.9, current_k=3, steps_since_switch=20) == 3


# ------------------------------------------------- cluster calibration (accepted-drafts)

def test_low_cluster_drops_from_k7():
    for ema in (1.52, 1.67, 2.16):  # prose / chat / reasoning
        assert decide_k(ema, 7, 20) == 5, ema


def test_high_cluster_stays_k7():
    for ema in (2.79, 2.74, 3.23, 3.49, 3.44):  # code/math/json/file_edit/summ
        assert decide_k(ema, 7, 20) == 7, ema


def test_prose_chat_reach_k3():
    for ema in (1.52, 1.67):
        assert decide_k(ema, 5, 20) == 3, ema


def test_reasoning_stays_k5():
    assert decide_k(2.16, 5, 20) == 5


def test_target_k_regions():
    assert target_k(1.5) == 3
    assert target_k(2.16) == 5
    assert target_k(2.79) == 7


# ------------------------------------------------------------------------ cooldown

def test_cooldown_prevents_immediate_switch():
    assert decide_k(ema_accepted=1.0, current_k=5, steps_since_switch=3) == 5
    assert decide_k(ema_accepted=3.0, current_k=3, steps_since_switch=5) == 3


def test_cooldown_boundary():
    assert decide_k(ema_accepted=1.0, current_k=5, steps_since_switch=10) == 3


# ----------------------------------------------------------------------- update_ema

def test_ema_init():
    assert update_ema(None, 3.5) == 3.5


def test_ema_update():
    ema = update_ema(3.0, 5.0, alpha=0.1)
    assert abs(ema - 3.2) < 1e-9


def test_ema_converges():
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
    assert st is not None and st.ks == (3, 5, 7) and st.k == 7


def test_parse_env_fixed_k(monkeypatch):
    monkeypatch.setenv("RADIANCE_INKLING2_ADAPTIVE_K", "5")
    st = state_from_env()
    assert st is not None and st.ks == (5,)
    for _ in range(50):
        st.observe(1.0, 5)
        assert st.next_k() == 5


def test_parse_env_invalid_is_off(monkeypatch):
    for bad in ("", "2", "3,4", "9", "3,5,7,9", "abc"):
        monkeypatch.setenv("RADIANCE_INKLING2_ADAPTIVE_K", bad)
        assert state_from_env() is None, bad


def test_state_cold_start_stays_at_max():
    st = State((3, 5, 7))
    assert st.next_k() == 7  # no full-window observation yet


def test_ema_only_updates_at_full_window():
    st = State((3, 5, 7))
    st.observe(1.0, 3)  # censored k=3 sample: must NOT move the estimate
    assert st.ema is None
    st.observe(2.8, 7)  # full window: updates
    assert st.ema == 2.8


# ------------------------------------------------------------------- de-censoring

def test_decensor_high_acceptance_recovers_k7():
    """After a low stretch locks k=3, a high-acceptance workload must climb back to 7
    via full-window probes (the censored k=3 samples alone could never signal it)."""
    st = State((3, 5, 7))
    for _ in range(80):
        d = st.k
        st.observe(1.5, d)
        st.next_k()
    assert st.k == 3
    for _ in range(400):
        drafted = st.k
        acc = 3.5 if drafted >= 7 else min(3.5, drafted * 0.75)
        st.observe(acc, drafted)
        st.next_k()
    assert st.k == 7


def test_decensor_low_acceptance_stays_k3():
    """Probes must not wrongly raise a genuinely low-acceptance workload."""
    st = State((3, 5, 7))
    for _ in range(400):
        drafted = st.k
        acc = 1.5 if drafted >= 7 else drafted * 0.5
        st.observe(acc, drafted)
        st.next_k()
    assert st.k == 3


def test_state_no_oscillation():
    st = State((3, 5, 7))
    for _ in range(60):
        st.observe(2.5, 7)
        st.next_k()
    flips = 0
    prev = st.k
    for _ in range(120):
        st.observe(2.5, 7)
        k = st.next_k()
        flips += k != prev
        prev = k
    assert flips == 0


# -------------------------------------------------------------------------- mutants

def test_mutants_killed():
    import radiance_inkling2_adaptive_k as m

    cases = [
        ("_K5_TO_K7", 1.5, lambda: m.decide_k(2.5, 5, 20) == 5),
        ("_K5_TO_K3", 2.8, lambda: m.decide_k(1.9, 5, 20) == 5),
        ("_K7_TO_K5", 3.4, lambda: m.decide_k(2.5, 7, 20) == 7),
        ("_K3_TO_K5", 0.9, lambda: m.decide_k(1.9, 3, 20) == 3),
        ("COOLDOWN", 5, lambda: m.decide_k(1.0, 5, 7) == 5),
    ]
    for attr, val, check in cases:
        orig = getattr(m, attr)
        setattr(m, attr, val)
        try:
            assert not check(), f"mutant {attr}={val} survived"
        finally:
            setattr(m, attr, orig)

    orig = m.ALPHA
    m.ALPHA = 0.5
    try:
        assert abs(m.update_ema(3.0, 5.0, alpha=m.ALPHA) - 4.0) < 1e-9
    finally:
        m.ALPHA = orig

    # probe-interval mutant: a much longer interval delays the k=7 recovery
    orig = m.PROBE_INTERVAL
    m.PROBE_INTERVAL = 10000
    try:
        st = m.State((3, 5, 7))
        for _ in range(80):
            d = st.k
            st.observe(1.5, d); st.next_k()
        for _ in range(100):  # < interval: no probe fires, k stays 3
            d = st.k
            st.observe(3.5 if d >= 7 else d * 0.75, d); st.next_k()
        assert st.k == 3, "probe-interval mutant survived (recovered without a probe)"
    finally:
        m.PROBE_INTERVAL = orig
