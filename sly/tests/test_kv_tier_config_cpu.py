#!/usr/bin/env python3
"""CPU test: `kv-cache/tools/tier-config.py` emits the `--kv-transfer-config` for both modes.

The launcher used to inline this config as a `python3 -c` one-liner and `ci/tier_ab.sh` carried a
copy of it, so the gate and the launcher could drift apart invisibly (the throughput sweep never
sees a wrong tier config). Both now call the helper, so they emit the same bytes, and a policy
outside `lru`/`arc` has to fail at launch rather than as an out-of-tree `CachePolicy` import deep
in vLLM.

Run:  python3 sly/tests/test_kv_tier_config_cpu.py
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = str(ROOT / "kv-cache" / "tools" / "tier-config.py")
BASE = json.loads((ROOT / "kv-cache" / "tier_config.json").read_text())
FS_TIER = json.loads((ROOT / "kv-cache" / "fs_tier.json").read_text())


def emit(mode, policy=""):
    proc = subprocess.run(["python3", HELPER, mode, policy], capture_output=True, text=True)
    return None if proc.returncode else json.loads(proc.stdout)


def test_ram_is_the_base_config_alone():
    """ram emits the mode-neutral baseline with no fs secondary tier."""
    cfg = emit("ram", "lru")
    assert cfg["kv_connector"] == BASE["kv_connector"]
    assert "secondary_tiers" not in cfg["kv_connector_extra_config"]
    assert cfg["kv_connector_extra_config"]["spec_name"] == "TieringOffloadingSpec"


def test_disk_adds_the_fs_tier_from_its_own_file():
    """disk emits the baseline plus the fs tier read from kv-cache/fs_tier.json."""
    cfg = emit("disk", "lru")
    assert cfg["kv_connector_extra_config"]["secondary_tiers"] == [FS_TIER]
    assert FS_TIER["root_dir"] == "/kvcache/blocks"


def test_policy_reaches_both_modes():
    """arc lands in the RAM tier in ram AND disk mode (the disk primary tier is the RAM tier)."""
    for mode in ("ram", "disk"):
        assert emit(mode, "arc")["kv_connector_extra_config"]["eviction_policy"] == "arc"
        assert emit(mode)["kv_connector_extra_config"]["eviction_policy"] == "lru"


def test_bad_policy_dies_at_launch():
    """A typo is rejected by the helper, not passed to vLLM as a CachePolicy class name."""
    for mode in ("ram", "disk"):
        assert emit(mode, "arcc") is None, f"{mode} accepted the bad policy arcc"


def test_every_caller_uses_the_helper():
    """The launcher and both harnesses call the helper; no stale config file name is left."""
    for rel in ("serve-mxfp4.sh", "ci/test-disk-tier.sh", "ci/tier_ab.sh"):
        text = (ROOT / rel).read_text()
        assert "kv-cache/tools/tier-config.py" in text, f"{rel} does not call the helper"
        assert "disk_tier_config.json" not in text, f"{rel} still names the old config file"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
