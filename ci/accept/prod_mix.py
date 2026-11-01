#!/usr/bin/env python3
"""Derive the gate's draft-depth and concurrency assumptions from PRODUCTION.

Why this exists
---------------
The gate compares a candidate against a baseline, and the baseline's "weighted
decode" is a fixed mixture (configs/full.json: code 0.30, reasoning 0.20, prose
0.15, json 0.15, file_edit 0.10, summarization 0.10). That mixture is a DECISION,
not a measurement: on 1.9.0 a lever won +10.3 % weighted while losing every prose
category, and the release notes only carried the weighted number.

What production can actually answer, and what it cannot
------------------------------------------------------
The served vLLM exposes counters, not request content. So:

* MEASURABLE -- how deep the drafter should go: the per-position acceptance
  counters (`vllm:spec_decode_num_accepted_tokens_per_pos_total`) divided by the
  drafted-token counter give the marginal tokens per draft position ON THE REAL
  LOAD. A position below `--min-marginal` barely pays for its forward pass.
* MEASURABLE -- the concurrency mix: samples of `vllm:num_requests_running`.
  This is the number behind "~80 % of busy time is a single request".
* NOT MEASURABLE -- the content mix (code vs prose vs chat). No metric carries
  it. Do not pretend otherwise: the weights stay a declared assumption in
  configs/full.json, and the per-category gate (`category_decode_tps_min_ratio`)
  is what stops a declared mix from hiding a losing category.

Usage
-----
    python3 ci/accept/prod_mix.py --prometheus http://192.168.178.21:30090
    python3 ci/accept/prod_mix.py --metrics-file /tmp/prod-metrics.txt
    python3 ci/accept/prod_mix.py --prometheus URL --json

The output is meant to be pasted into a PR body when a spec-depth or an
acceptance-rule change is proposed -- not to be applied automatically.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

SAMPLE_RE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+\d.eE]+)$")
POS_RE = re.compile(r'position="(?P<pos>\d+)"')

# Counters we read. `drafted` is the denominator of the acceptance rate.
COUNTERS = (
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_per_pos_total",
    "vllm:generation_tokens_total",
    "vllm:iteration_tokens_total_count",
)


def fetch(url: str, query: str, timeout: int = 30) -> str:
    """Prometheus instant query, returns the raw JSON text."""
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/v1/query?query={urllib.parse.quote(query)}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode()


def fetch_metric(url: str, metric: str) -> float:
    """Sum one counter over all series via Prometheus."""
    body = json.loads(fetch(url, f"sum({metric})"))
    results = body.get("data", {}).get("result") or []
    return float(results[0]["value"][1]) if results else 0.0


def parse_metrics_text(text: str) -> dict:
    """Parse a raw /metrics scrape into the counters this script needs."""
    out: dict = {"counters": {}, "per_pos": {}, "running_samples": []}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SAMPLE_RE.match(line)
        if not m:
            continue
        name = m.group("name")
        value = float(m.group("value"))
        if name not in COUNTERS:
            if name == "vllm:num_requests_running":
                out["running_samples"].append(int(value))
            continue
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos_total":
            pos = POS_RE.search(m.group("labels") or "")
            if pos:
                key = int(pos.group("pos"))
                out["per_pos"][key] = out["per_pos"].get(key, 0.0) + value
        else:
            out["counters"][name] = out["counters"].get(name, 0.0) + value
    return out


def draft_profile(per_pos: dict, drafted: float) -> dict:
    """Marginal tokens per draft position, and the depth where it stops paying.

    `per_pos[i]` counts accepted tokens at position i, `drafted` counts all
    drafted tokens (drafts x depth). The marginal contribution of position i is
    accepted_i / drafted = tokens added per engine step by that position alone.
    Cumulative sum + 1 is tokens/update, i.e. what the throughput is made of.
    """
    if drafted <= 0 or not per_pos:
        return {}
    positions = sorted(per_pos)
    marg = {p: per_pos[p] / drafted for p in positions}
    cum, rows = 1.0, {}
    for p in positions:
        cum += marg[p]
        rows[p] = {"marginal": round(marg[p], 4), "cumulative_tokens_per_update": round(cum, 3)}
    return rows


def concurrency_mix(samples: list[int]) -> tuple[dict, int]:
    """Share of samples by running-request bucket."""
    if not samples:
        return {}, 0
    buckets = {"1": 0, "2-4": 0, "5-8": 0, ">8": 0}
    for n in samples:
        if n <= 1:
            buckets["1"] += 1
        elif n <= 4:
            buckets["2-4"] += 1
        elif n <= 8:
            buckets["5-8"] += 1
        else:
            buckets[">8"] += 1
    total = len(samples)
    return {k: round(v / total, 3) for k, v in buckets.items()}, total


def recommend_depth(profile: dict, min_marginal: float) -> dict:
    """Deepest position whose marginal acceptance still clears `min_marginal`."""
    paying = [p for p, row in profile.items() if row["marginal"] >= min_marginal]
    if not paying:
        return {"depth": 0, "reason": f"no position above {min_marginal} marginal"}
    depth = max(paying) + 1                      # positions are 0-based
    cum = profile[max(paying)]["cumulative_tokens_per_update"]
    return {"depth": depth, "tokens_per_update": cum,
            "reason": f"positions 0..{max(paying)} each add >= {min_marginal} tokens/step"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prometheus", help="base URL of a Prometheus, e.g. http://192.168.178.21:30090")
    src.add_argument("--metrics-file", help="a saved /metrics scrape of the served engine")
    ap.add_argument("--min-marginal", type=float, default=0.08,
                    help="a draft position below this many tokens/step does not pay (default 0.08)")
    ap.add_argument("--json", action="store_true", help="machine-readable output only")
    args = ap.parse_args()

    if args.prometheus:
        per_pos = {}
        # Prometheus carries the per-position counter as one series per position.
        body = json.loads(fetch(args.prometheus,
                                "vllm:spec_decode_num_accepted_tokens_per_pos_total"))
        for series in body.get("data", {}).get("result") or []:
            pos = series.get("metric", {}).get("position")
            if pos is not None:
                per_pos[int(pos)] = per_pos.get(int(pos), 0.0) + float(series["value"][1])
        counters = {m: fetch_metric(args.prometheus, m) for m in
                    ("vllm:spec_decode_num_draft_tokens_total",)}
        samples = []
        running = json.loads(fetch(args.prometheus, "vllm:num_requests_running"))
        for series in running.get("data", {}).get("result") or []:
            try:
                samples.append(int(float(series["value"][1])))
            except (KeyError, ValueError):
                pass
    else:
        text = Path(args.metrics_file).read_text()
        parsed = parse_metrics_text(text)
        per_pos = parsed["per_pos"]
        counters = parsed["counters"]
        samples = parsed["running_samples"]

    drafted = counters.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    prof = draft_profile(per_pos, drafted)
    rec = recommend_depth(prof, args.min_marginal) if prof else {}
    mix, n_samples = concurrency_mix(samples)

    out = {
        "source": args.prometheus or args.metrics_file,
        "drafted_tokens": drafted,
        "per_position": prof,
        "depth_recommendation": rec,
        "concurrency_mix": mix,
        "concurrency_samples": n_samples,
        "not_measurable": ("content mix (code/prose/chat): no metric carries it. "
                           "Keep it a declared assumption in ci/accept/configs/*.json; "
                           "the per-category gate is what keeps it honest."),
    }
    if args.json:
        print(json.dumps(out, indent=2))
        return 0

    print("Draft depth from production counters")
    print(f"  source: {out['source']}  (drafted tokens: {drafted:,.0f})")
    if prof:
        print(f"  {'pos':>4} {'marginal tok/step':>18} {'cumulative':>11}")
        for pos, row in prof.items():
            print(f"  {pos:>4} {row['marginal']:>18.4f} {row['cumulative_tokens_per_update']:>11.3f}")
    if rec:
        print(f"  => depth {rec.get('depth')}: {rec.get('reason')}")
        print(f"     tokens/update there: {rec.get('tokens_per_update')}")
    if mix:
        print("\nConcurrency mix (samples of num_requests_running)")
        for bucket, share in mix.items():
            print(f"  {bucket:>4} requests: {share * 100:5.1f} %")
        if n_samples < 10:
            print(f"  ({n_samples} sample(s) -- a single scrape is a snapshot, "
                  "use --prometheus for a real mix)")
    print(f"\nNot measurable: {out['not_measurable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
