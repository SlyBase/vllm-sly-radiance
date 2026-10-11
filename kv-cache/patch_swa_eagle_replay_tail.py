#!/usr/bin/env python3
"""Keep one more drafter block at the replay boundary so the next turn hits where Mamba has a state.

THE MISS THIS FIXES (turnbench 2026-10-08, R9700, radiance 1.0.0 / vLLM 0.30, KVCACHE=ram, eagle
group annotated, block size 896): every tier hit landed on the prompt end of the turn BEFORE the
previous one, 37-56k tokens of prefill per tier turn instead of ~20k. patch_offload_boundary_trace.py
shows why. Session A, turn 6, prompt 113,886 tokens = 127 full blocks + a 94-token tail:

  1. With an Eagle-style drafter the scheduler's block-aligned prefill split stops one block
     early (`_mamba_block_aligned_split`: last_cache_position -= block_size), so the Mamba
     state the request leaves behind sits at block 125 (ending at 112,896), not at the aligned
     prompt end 113,792 (block 126). The trace shows exactly one boundary per request at
     aligned_end - 896, never one at aligned_end.
  2. The drafter group (sliding window of 2 blocks, under eagle block drop) keeps its replay-
     boundary tail from `SlidingWindowManager.reachable_block_mask`: `need` = 3 blocks ending at
     `aligned // block_size + shift` with shift = 1 under eagle -- blocks 125, 126 and 127,
     where 127 is the partial prompt tail that is never storable.
  3. The next turn's lookup needs window + 1 = 3 consecutive drafter chunks ending one past the
     block where Mamba has its state: 124, 125, 126. Block 124 was never stored, so the run is 2
     and `_sliding_window_lookup` walks back to the next place both groups line up: the
     shared-prefix junction of the previous turn, i.e. the prompt end of the turn before it.
     Measured: A7 hit at 94,976 (= A5's aligned prompt end) instead of 112,896.

WHAT THIS CHANGES
At each reachable boundary the sliding-window tail starts one block earlier when `use_eagle` is
set, i.e. it keeps `need + 1` blocks ending at the same place. `use_eagle` is the engine-wide
flag (eagle block drop on), the same one stock's `shift` keys on -- so this widens the tail of
every sliding-window group under eagle, not only the drafter's; on this model the drafter is the
only such group. A superset of the stock mask: one extra sliding-window block (~a chunk of
drafter KV) per retained boundary, on the GPU and in the tier, and nothing is dropped that stock
kept. The Mamba mask is untouched. Measured with it: A7 hit at 112,896, 19,970 tokens recomputed.

Why widen the tail rather than move the boundary: moving `reachable_boundaries` one block
earlier under eagle would also be consistent, for the Mamba mask too, but it changes which
block stock keeps and can lose a hit for a group that does line up at `aligned`; one block more
cannot. The upstream-shaped fix is the moved boundary with a unit test on both masks.

GATE: RADIANCE_SWA_EAGLE_REPLAY_TAIL (unset/0 = upstream, 1 = the wider tail). vLLM >= 0.30
only (the mask does not exist before); older trees report SKIP.
"""
import os
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, "/patches")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _patchlib import apply  # noqa: E402

VLLM = Path(os.environ.get("RADIANCE_VLLM_DIR", sysconfig.get_paths()["purelib"] + "/vllm"))
STKM = VLLM / "v1" / "core" / "single_type_kv_cache_manager.py"

ANCHOR = (
    "        if retention_interval is not None:\n"
    "            for boundary_tokens in reachable_boundaries:\n"
    "                aligned = boundary_tokens // alignment_tokens * alignment_tokens\n"
    "                end = aligned // block_size + shift\n"
    "                for j in range(max(start_block, end - need), min(end_block, end)):\n"
    "                    mask[j - start_block] = True\n"
)
NEW = (
    "        if retention_interval is not None:\n"
    "            # radiance eagle-replay-tail: under eagle block drop the scheduler commits the\n"
    "            # Mamba boundary state one block before the aligned prompt end, so the hit the\n"
    "            # next turn can land on needs this window to end one block earlier too. Keep\n"
    "            # need + 1 blocks (a superset of the stock tail; use_eagle is engine-wide).\n"
    "            # Kill switch: RADIANCE_SWA_EAGLE_REPLAY_TAIL unset/0.\n"
    "            _rad_extra = int(\n"
    '                use_eagle and os.environ.get("RADIANCE_SWA_EAGLE_REPLAY_TAIL", "0") == "1"\n'
    "            )\n"
    "            for boundary_tokens in reachable_boundaries:\n"
    "                aligned = boundary_tokens // alignment_tokens * alignment_tokens\n"
    "                end = aligned // block_size + shift\n"
    "                for j in range(\n"
    "                    max(start_block, end - need - _rad_extra), min(end_block, end)\n"
    "                ):\n"
    "                    mask[j - start_block] = True\n"
)


def main() -> None:
    src = STKM.read_text() if STKM.exists() else ""
    if "def reachable_block_mask" not in src:
        print("  SKIP  eagle-replay-tail: no reachable_block_mask (vLLM < 0.30)")
        return
    if "\nimport os\n" not in src:
        first_import = src.index("\nimport ")
        STKM.write_text(src[:first_import] + "\nimport os" + src[first_import:])
        print("  OK    eagle-replay-tail: added missing 'import os'")
    apply(STKM, ANCHOR, NEW, "radiance eagle-replay-tail", "eagle-replay-tail: wider SWA tail at reachable boundaries")


if __name__ == "__main__":
    main()
