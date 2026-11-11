#!/usr/bin/env python3
"""Shared helpers for the KV-cache effectiveness consumers (kvvalidate, tierreport).

Only the genuinely byte-identical bits live here. The two parsers are NOT
merged: they diverge on purpose (kvvalidate's is label-superset tolerant,
tierreport's is label-preserving and drives the per-tier report), so merging
them would couple two behaviors that must evolve independently.
"""

import urllib.request


def fetch_metrics(url, timeout=30):
    """GET the Prometheus /metrics endpoint (read-only) and return the body."""
    req = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")
