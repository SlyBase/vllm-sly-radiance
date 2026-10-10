#!/usr/bin/env python3
"""Trace every Mamba boundary-state hand-off the offload scheduler sees (vLLM >= 0.30).

WHY
vLLM 0.30 stores a Mamba "align" group's state per retained checkpoint: the KV cache manager
offers (group, block, boundary) triples each step, and `_build_aligned_boundary_store_jobs`
turns the usable ones into store jobs. Which boundaries arrive, and which are dropped on the
way, is invisible in the log -- yet it decides where the next turn's tier hit can land.
turnbench 2026-10-08 (R9700, 1.0.0, KVCACHE=ram): every tier hit landed on the prompt end of
the turn BEFORE the previous one, i.e. the previous turn's prompt-end state never reached
the tier. This trace shows the hand-offs so that can be read off instead of guessed.

WHAT
One INFO line per offered boundary with the decision:
  [radiance] boundary req=<id> g=<group> boundary=<tokens> block=<id> -> stored | <reason>
Reasons: no_req_status, null_block, past_max(<max>), misaligned(<tokens_per_chunk>),
alloc_failure, already_present. Gate: RADIANCE_OFFLOAD_BOUNDARY_TRACE=1 (unset = silent,
the hunk is a no-op). Diagnostic only; it changes no decision.

Applies only where the function exists (vLLM >= 0.30); older trees report SKIP.
"""
import os
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _patchlib import apply, apply_any  # noqa: E402

VLLM = Path(os.environ.get("RADIANCE_VLLM_DIR", sysconfig.get_paths()["purelib"] + "/vllm"))
OFFSCHED = VLLM / "distributed" / "kv_transfer" / "kv_connector" / "v1" / "offloading" / "scheduler.py"

FLAG_ANCHOR = "\nclass SchedulerOffloadConfig(NamedTuple):\n"
FLAG_NEW = (
    "\n# radiance boundary-trace: log every Mamba boundary-state hand-off and its fate.\n"
    '_RADIANCE_BOUNDARY_TRACE = os.environ.get("RADIANCE_OFFLOAD_BOUNDARY_TRACE", "0") == "1"\n'
    "\n"
    "\n"
    "def _radiance_boundary_log(req_id, group_idx, boundary, block_id, fate):\n"
    "    if _RADIANCE_BOUNDARY_TRACE:\n"
    "        logger.info(\n"
    '            "[radiance] boundary req=%s g=%d boundary=%d block=%d -> %s",\n'
    "            req_id, group_idx, boundary, block_id, fate,\n"
    "        )\n"
) + FLAG_ANCHOR

LOOP_ANCHOR = (
    "        for req_id, entries in handoffs.items():\n"
    "            req_status = self._req_status.get(req_id)\n"
    "            if req_status is None:\n"
    "                continue\n"
    "            req = req_status.req\n"
    "            max_boundary = self._calc_num_offloadable_tokens(req_status, req.num_tokens)\n"
    "            for group_idx, block_id, boundary in entries:\n"
    "                config_idx = config_idx_by_group.get(group_idx)\n"
    "                if config_idx is None:\n"
    "                    continue\n"
    "                group_config = self.config.kv_group_configs[config_idx]\n"
    "                if (\n"
    "                    block_id == 0\n"
    "                    or boundary > max_boundary\n"
    "                    or boundary % group_config.tokens_per_chunk != 0\n"
    "                ):\n"
    "                    continue\n"
    "\n"
    "                key = self._make_boundary_key(req, group_idx, boundary)\n"
    "                store_output = self.manager.prepare_store([key], req_status.req_context)\n"
    "                if store_output is None:\n"
    "                    self._connector_stats.increase_counter(\n"
    "                        _ConnectorMetricName.ALLOCATION_FAILURE\n"
    "                    )\n"
    "                    continue\n"
    "                if not store_output.keys_to_store:\n"
    "                    continue\n"
)
LOOP_NEW = (
    "        for req_id, entries in handoffs.items():\n"
    "            req_status = self._req_status.get(req_id)\n"
    "            if req_status is None:\n"
    "                for group_idx, block_id, boundary in entries:  # radiance boundary-trace\n"
    '                    _radiance_boundary_log(req_id, group_idx, boundary, block_id, "no_req_status")\n'
    "                continue\n"
    "            req = req_status.req\n"
    "            max_boundary = self._calc_num_offloadable_tokens(req_status, req.num_tokens)\n"
    "            for group_idx, block_id, boundary in entries:\n"
    "                config_idx = config_idx_by_group.get(group_idx)\n"
    "                if config_idx is None:\n"
    "                    continue\n"
    "                group_config = self.config.kv_group_configs[config_idx]\n"
    "                if (\n"
    "                    block_id == 0\n"
    "                    or boundary > max_boundary\n"
    "                    or boundary % group_config.tokens_per_chunk != 0\n"
    "                ):\n"
    "                    _radiance_boundary_log(  # radiance boundary-trace\n"
    "                        req_id, group_idx, boundary, block_id,\n"
    '                        "null_block" if block_id == 0\n'
    '                        else "past_max(%d)" % max_boundary if boundary > max_boundary\n'
    '                        else "misaligned(%d)" % group_config.tokens_per_chunk,\n'
    "                    )\n"
    "                    continue\n"
    "\n"
    "                key = self._make_boundary_key(req, group_idx, boundary)\n"
    "                store_output = self.manager.prepare_store([key], req_status.req_context)\n"
    "                if store_output is None:\n"
    "                    self._connector_stats.increase_counter(\n"
    "                        _ConnectorMetricName.ALLOCATION_FAILURE\n"
    "                    )\n"
    '                    _radiance_boundary_log(req_id, group_idx, boundary, block_id, "alloc_failure")\n'
    "                    continue\n"
    "                if not store_output.keys_to_store:\n"
    '                    _radiance_boundary_log(req_id, group_idx, boundary, block_id, "already_present")\n'
    "                    continue\n"
    '                _radiance_boundary_log(req_id, group_idx, boundary, block_id, "stored")\n'
)


def main() -> None:
    src = OFFSCHED.read_text() if OFFSCHED.exists() else ""
    if "def _build_aligned_boundary_store_jobs" not in src:
        print("  SKIP  boundary-trace: no _build_aligned_boundary_store_jobs (vLLM < 0.30)")
        return
    if "\nimport os\n" not in src:
        # stock 0.30 has no module-level os; mixed-hit adds it too, order-independent
        OFFSCHED.write_text(
            src.replace("\nlogger = init_logger(__name__)", "\nimport os\n\nlogger = init_logger(__name__)", 1)
        )
        print("  OK    boundary-trace: added missing 'import os'")
    apply(OFFSCHED, FLAG_ANCHOR, FLAG_NEW, "_RADIANCE_BOUNDARY_TRACE", "boundary-trace: flag + logger")
    # vLLM 0.31 passes the request context into _make_boundary_key (key positions for
    # per-request recency); the loop is otherwise the same.
    key_030 = "                key = self._make_boundary_key(req, group_idx, boundary)\n"
    key_031 = (
        "                key = self._make_boundary_key(\n"
        "                    req, group_idx, boundary, req_status.req_context\n"
        "                )\n"
    )
    assert key_030 in LOOP_ANCHOR and key_030 in LOOP_NEW
    apply_any(
        OFFSCHED,
        [
            (LOOP_ANCHOR, LOOP_NEW),
            (LOOP_ANCHOR.replace(key_030, key_031), LOOP_NEW.replace(key_030, key_031)),
        ],
        '"no_req_status")',
        "boundary-trace: hand-off fates",
    )


if __name__ == "__main__":
    main()
