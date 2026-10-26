"""CPU tests for the acceptance gate's per-category and chat-replay checks.

The 1.9.0 release shipped a spec-depth change that won +10.3 % on the weighted
median while losing every prose category and halving chat throughput. These
tests pin the two checks that would have caught it: per-category decode and the
multi-turn chat replay. No GPU, no network -- pure functions only.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ACCEPT_DIR = Path(__file__).resolve().parents[2] / "ci" / "accept"


def _load(name: str):
    path = ACCEPT_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"gate_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


accept = _load("accept")
prod_mix = _load("prod_mix")

THRESHOLDS = {
    "kv_tokens_min_ratio": 0.98,
    "aggregate_tps_min_ratio": 0.95,
    "category_decode_tps_min_ratio": 0.95,
    "chat_tps_min_ratio": 0.95,
    "acceptance_rate_min_delta": -0.02,
    "mean_tokens_per_step_min_delta": -0.3,
    "gsm8k_min_delta": -0.05,
    "startup_s_max_ratio": 1.5,
    "preemptions_max": 0,
    "aggregate_tps_min_ratio_conc_1": 0.85,
}


def _entry(decode_tps, tokens_per_update=3.0, ok=True):
    return {"ok": ok, "decode_tps": decode_tps, "tokens_per_update": tokens_per_update}


def _results(**over):
    res = {
        "start": {"kv_tokens": 400000, "startup_s": 200},
        "logs": {"ok": True, "missing": [], "forbidden": []},
        "smoke": {"skipped": "dry-run"},
        "spec": {"acceptance_rate": 0.5, "mean_tokens_per_step": 4.0, "preemptions": 0},
    }
    res.update(over)
    return res


def _baseline(**over):
    base = {
        "kv_tokens": 400000,
        "startup_s": 200,
        "aggregate_tps": {"1": 100.0},
        "acceptance_rate": 0.5,
        "mean_tokens_per_step": 4.0,
    }
    base.update(over)
    return base


def _by_name(checks):
    return {c["name"]: c for c in checks}


def test_category_medians_are_medians_and_skip_failed_runs():
    data = {"single_stream": {
        "prose": [_entry(70.0), _entry(80.0), _entry(90.0)],
        "chat": [_entry(10.0, ok=False), _entry(60.0), _entry(40.0)],
        "empty": [_entry(0.0, ok=False)],
    }}
    tps, tpu = accept.category_medians(data)
    assert tps["prose"] == 80.0
    assert tps["chat"] == 50.0          # failed run ignored, even count -> mean of middle
    assert "empty" not in tps
    assert tpu["prose"] == 3.0


def test_losing_prose_category_fails_even_when_the_mix_wins():
    """The 1.9.0 k=15 shape: weighted up, prose and chat down."""
    results = _results(
        betterbench={
            "aggregate_tps": {"1": 110.0},          # +10 % -- would pass alone
            "categories_decode_tps": {
                "code": 197.2, "json": 257.6, "prose": 74.4, "chat": 130.9,
            },
        },
        chat={"ok": True, "levels": {
            "4000": {"decode_tps": 34.0}, "16000": {"decode_tps": 21.0}}},
    )
    baseline = _baseline(
        categories_decode_tps={
            "code": 173.9, "json": 197.9, "prose": 80.2, "chat": 143.2},
        chat_decode_tps={"4000": 72.0, "16000": 29.0},
    )
    checks = _by_name(accept.compare(results, baseline, THRESHOLDS, "full"))
    assert checks["throughput conc 1"]["ok"]
    assert checks["throughput json"]["ok"]
    assert not checks["throughput prose"]["ok"]
    assert not checks["throughput chat"]["ok"]
    assert checks["throughput prose"]["hard"]
    assert not checks["chat 4k decode"]["ok"]
    assert checks["chat 4k decode"]["hard"]
    assert not checks["chat 16k decode"]["ok"]


def test_chat_check_is_soft_in_fast_mode():
    results = _results(
        betterbench={"aggregate_tps": {"1": 100.0}, "categories_decode_tps": {}},
        chat={"ok": True, "levels": {"4000": {"decode_tps": 40.0}}},
    )
    baseline = _baseline(chat_decode_tps={"4000": 72.0})
    checks = _by_name(accept.compare(results, baseline, THRESHOLDS, "fast"))
    assert not checks["chat 4k decode"]["ok"]
    assert not checks["chat 4k decode"]["hard"]


def test_untouched_platform_is_still_a_pass():
    results = _results(
        betterbench={
            "aggregate_tps": {"1": 100.0},
            "categories_decode_tps": {"prose": 80.0, "code": 174.0},
        },
        chat={"ok": True, "levels": {"4000": {"decode_tps": 75.0}}},
    )
    baseline = _baseline(
        categories_decode_tps={"prose": 80.2, "code": 173.9},
        chat_decode_tps={"4000": 72.0},
    )
    checks = accept.compare(results, baseline, THRESHOLDS, "full")
    assert [c["name"] for c in checks if not c["ok"] and c["hard"]] == []


def test_missing_category_baseline_is_reported_not_silently_passed():
    results = _results(
        betterbench={"aggregate_tps": {"1": 100.0},
                     "categories_decode_tps": {"prose": 80.0}})
    checks = _by_name(accept.compare(results, _baseline(), THRESHOLDS, "fast"))
    assert checks["throughput per category"]["ok"] is None      # soft, explicit
    assert "no category medians" in checks["throughput per category"]["actual"]


def test_build_chat_grows_with_depth_and_ends_with_the_question():
    small = accept.build_chat(4000, 3)
    big = accept.build_chat(16000, 3)
    assert small[-1]["role"] == "user"
    assert len("".join(m["content"] for m in small)) < len("".join(m["content"] for m in big))
    assert len(small) == 7                     # 3 user/assistant pairs + the question


def test_draft_profile_marginal_and_depth_recommendation():
    # 4 positions, drafted tokens = 1000 -> marginals 0.40, 0.20, 0.10, 0.02
    per_pos = {0: 400.0, 1: 200.0, 2: 100.0, 3: 20.0}
    prof = prod_mix.draft_profile(per_pos, 1000.0)
    assert prof[0]["marginal"] == pytest.approx(0.40)
    assert prof[3]["marginal"] == pytest.approx(0.02)
    assert prof[0]["cumulative_tokens_per_update"] == pytest.approx(1.40)
    rec = prod_mix.recommend_depth(prof, min_marginal=0.08)
    assert rec["depth"] == 3                   # position 3 (0.02) does not pay
    assert rec["tokens_per_update"] == pytest.approx(1.70)


def test_metrics_text_parse_feeds_profile_and_mix():
    text = "\n".join([
        "# HELP vllm:spec_decode_num_draft_tokens_total drafts",
        'vllm:spec_decode_num_draft_tokens_total{model_name="m"} 2000.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="0"} 800.0',
        'vllm:spec_decode_num_accepted_tokens_per_pos_total{model_name="m",position="1"} 200.0',
        'vllm:num_requests_running{model_name="m"} 1.0',
        'vllm:num_requests_running{model_name="m"} 6.0',
    ])
    parsed = prod_mix.parse_metrics_text(text)
    assert parsed["counters"]["vllm:spec_decode_num_draft_tokens_total"] == 2000.0
    assert parsed["per_pos"] == {0: 800.0, 1: 200.0}
    mix, total = prod_mix.concurrency_mix(parsed["running_samples"])
    assert total == 2 and mix["1"] == 0.5 and mix["5-8"] == 0.5
    prof = prod_mix.draft_profile(parsed["per_pos"],
                                  parsed["counters"]["vllm:spec_decode_num_draft_tokens_total"])
    assert prof[0]["marginal"] == pytest.approx(0.40)


def test_profile_declares_the_new_thresholds_and_chat_block():
    profile = json.loads((ACCEPT_DIR / "profiles" / "vllm7-mxfp4.json").read_text())
    assert profile["thresholds"]["category_decode_tps_min_ratio"] >= 0.9
    assert profile["thresholds"]["chat_tps_min_ratio"] >= 0.9
    replay = profile["chat_replay"]
    assert replay["depths"] == sorted(replay["depths"])
    assert replay["depths"][0] >= 4000
    assert replay["fast_levels"] >= 1
