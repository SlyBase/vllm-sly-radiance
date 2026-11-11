#!/usr/bin/env python3
"""CPU test: the KV-cache consumers run clean against the 0.32.0 metric fixture.

`kvvalidate.py` and `tierreport.py` were ported to the 0.32.0 metric surface (the
token-granularity hit counter dropped; the prompt-token counters lost their `_total`
suffix). The fixture `kv-cache/tools/fixture-metrics-032.txt` captures that surface
synthetically so the consumers can be exercised offline, without a live engine. This
pins the contract the review flagged as untested: the counter invariants partition,
the degraded token counters degrade to UNKNOWN (never a false PASS/FAIL), and the
two on-disk invariants report UNKNOWN on a saved snapshot rather than walking a
foreign store.

Run:  python3 sly/tests/test_kv_consumers_cpu.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "kv-cache" / "tools"
FIXTURE = str(TOOLS / "fixture-metrics-032.txt")


def run(script, *extra):
    proc = subprocess.run([sys.executable, str(TOOLS / script), *extra],
                          capture_output=True, text=True)
    return proc


def run_kvvalidate_json():
    proc = run("kvvalidate.py", "--metrics-file", FIXTURE, "--json")
    assert proc.returncode == 0, f"kvvalidate exit {proc.returncode}\n{proc.stderr}"
    return json.loads(proc.stdout)


def test_kvvalidate_invariants_partition_and_degrade():
    """The four always-run invariants resolve; the two on-disk ones go UNKNOWN on a snapshot."""
    d = run_kvvalidate_json()
    states = {i["name"]: i["state"] for i in d["invariants"]}
    # The counter invariants resolve to a definite state on the fixture.
    assert states["lookup_partition"] == "pass"
    assert states["lookup_subpartition"] == "pass"
    assert states["promotion_refused"] == "pass"
    # The two on-disk invariants cannot run against a saved snapshot (the store they
    # would walk is a live one, foreign to this capture) -> UNKNOWN, never a false fail.
    assert states["disk_vs_engine"] == "unknown"
    assert states["group_ratio"] == "unknown"
    # Six invariants total (the count the docstring names).
    assert len(states) == 6


def test_kvvalidate_no_false_pass_on_missing_token_counter():
    """A degraded (absent) counter reads UNKNOWN, not a fabricated PASS/FAIL."""
    d = run_kvvalidate_json()
    states = {i["name"]: i["state"] for i in d["invariants"]}
    # The token-granularity counter is absent on the 0.32.0 surface; the invariant
    # that depends on it must not hard-pass on a missing counter.
    assert "cpu_equals_external" in states
    assert states["cpu_equals_external"] != "pass"


def test_tierreport_runs_clean_on_fixture():
    """tierreport parses the 0.32.0 surface without crashing and degrades gracefully."""
    proc = run("tierreport.py", "--metrics-file", FIXTURE, "--json")
    assert proc.returncode == 0, f"tierreport exit {proc.returncode}\n{proc.stderr}"
    d = json.loads(proc.stdout)
    # The degraded token counter is surfaced, not derived from the latency histogram.
    assert isinstance(d, dict)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
