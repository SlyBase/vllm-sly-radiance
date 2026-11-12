#!/usr/bin/env python3
"""tier-config.py -- emit the --kv-transfer-config JSON for one KVCACHE mode.

usage: tier-config.py <ram|disk> [eviction_policy]

kv-cache/tier_config.json is the mode-neutral baseline (CPU primary tier only).
kv-cache/fs_tier.json is the fs secondary tier disk mode adds; disk appends it to
kv_connector_extra_config as a one-element secondary_tiers list.

An omitted or empty eviction_policy keeps the file's value. Only lru and arc are
accepted: TieringOffloadingSpec treats any other string as an out-of-tree CachePolicy
class name and fails deep in the engine, so a typo has to die at launch, not in the log.
"""
import json
import sys
from pathlib import Path

KV_CACHE_DIR = Path(__file__).resolve().parent.parent


def main(argv):
    if len(argv) not in (2, 3):
        sys.exit(f"usage: {Path(argv[0]).name} <ram|disk> [eviction_policy]")
    mode, policy = argv[1], argv[2] if len(argv) == 3 else ""
    if mode not in ("ram", "disk"):
        sys.exit(f"mode must be ram or disk (got: {mode})")
    cfg = json.loads((KV_CACHE_DIR / "tier_config.json").read_text())
    extra = cfg["kv_connector_extra_config"]
    if mode == "disk":
        extra["secondary_tiers"] = [json.loads((KV_CACHE_DIR / "fs_tier.json").read_text())]
    if policy:
        if policy not in ("lru", "arc"):
            sys.exit(f"eviction_policy must be lru or arc (got: {policy})")
        extra["eviction_policy"] = policy
    print(json.dumps(cfg, separators=(",", ":")))


if __name__ == "__main__":
    main(sys.argv)
