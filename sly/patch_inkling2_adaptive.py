#!/usr/bin/env python3
"""Adaptive DSpark verify window for Inkling-2-Mini (RADIANCE_INKLING2_ADAPTIVE_K).

The DSpark drafter's checkpoint block size fixes the MAX verify window at k = 7
(a verify step is M = 8 rows). A shorter window (k in {3, 5}) verifies exactly
the same way -- rejection sampling is exact for any k -- it just reads fewer
distinct MoE experts and less dense attention per step, so the step is cheaper.
This patch lets the scheduler pick k in {3, 5, 7} per step from a running
acceptance estimate (EMA of accepted tokens per step, hysteresis + cooldown),
using the native per-step `num_spec_tokens_to_schedule` seam. It is a NOOP on a
vLLM without `models/inkling2`.

The k override runs AFTER the native `dynamic_sd_lookup` block (which is batch-
size keyed and off here) so the acceptance-based k wins. The draft step still
runs at its fixed 8-row block; only the verify width (k + 1 rows) changes, and
the runner already masks `draft_tokens[:, k:] = -1` from the same value, so the
draft and verify widths stay in lockstep.

Inert unless RADIANCE_INKLING2_ADAPTIVE_K is set to a comma list of k's, each in
{3, 5, 7}, at most three entries (e.g. "3,5,7" adaptive, "5" fixed).
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
ROOT = SP / "vllm/models/inkling2"
SCHED = SP / "vllm/v1/core/sched/scheduler.py"
MR = SP / "vllm/v1/worker/gpu/model_runner.py"

OBSERVE_ANCHOR = (
    "                num_rejected = len(scheduled_spec_token_ids) - num_accepted\n"
)
OBSERVE_NEW = OBSERVE_ANCHOR + "                self._rad_ik_observe(num_draft_tokens, num_accepted)\n"

K_ANCHOR = (
    "        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:\n"
    "            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
    "                len(num_scheduled_tokens)\n"
    "            ]\n"
)
K_NEW = K_ANCHOR + "        num_spec_tokens_to_schedule = self._rad_ik_k(num_spec_tokens_to_schedule)\n"

# The target's FULL decode cudagraph is captured only for the config verify width
# (q = k+1 = 8). A shorter window (q = 4 / 6) would find no FULL graph and fall to
# the ~2x slower PIECEWISE route. Capture the adaptive widths too, so every allowed
# k replays a FULL graph. Inert unless the knob lists those k's.
MR_ANCHOR = (
    "            decode_query_lens=self.vllm_config.get_decode_cudagraph_query_lens(\n"
    "                self.decode_query_len\n"
    "            ),\n"
)
MR_NEW = (
    "            decode_query_lens=_rad_ik_query_lens(\n"
    "                self.vllm_config, self.decode_query_len\n"
    "            ),\n"
)
MR_TAIL = '''

def _rad_ik_query_lens(vllm_config, decode_query_len):
    """Target decode query lens plus the adaptive verify widths (k+1), so a shorter
    window still replays a FULL cudagraph instead of falling to PIECEWISE."""
    lens = vllm_config.get_decode_cudagraph_query_lens(decode_query_len)
    import os
    raw = os.environ.get("RADIANCE_INKLING2_ADAPTIVE_K", "")
    ks = {int(p) for p in raw.split(",") if p.strip() in ("3", "5", "7")}
    if ks:
        cap = max(lens.values()) if lens else vllm_config.scheduler_config.max_num_seqs
        for k in ks:
            lens.setdefault(k + 1, cap)
    return lens
'''

TAIL = '''
# ---- RADIANCE adaptive DSpark verify window (patch_inkling2_adaptive.py) ----
try:
    from radiance_inkling2_adaptive_k import state_from_env as _rad_ik_state_from_env
except Exception:  # module missing: stay inert
    _rad_ik_state_from_env = None

_RAD_IK_STATE = _rad_ik_state_from_env() if _rad_ik_state_from_env is not None else None


def _rad_ik_observe(self, num_draft_tokens, num_accepted):
    """Feed one (drafted, accepted) spec step into the running acceptance EMA."""
    if _RAD_IK_STATE is not None:
        _RAD_IK_STATE.observe(num_accepted)


def _rad_ik_k(self, k):
    """Per-step verify window k from the running acceptance estimate (inert if off)."""
    if _RAD_IK_STATE is None:
        return k
    return _RAD_IK_STATE.next_k()


Scheduler._rad_ik_observe = _rad_ik_observe
Scheduler._rad_ik_k = _rad_ik_k
'''


def main():
    if not ROOT.is_dir():
        print("  NOOP  inkling2 adaptive-k: this vLLM has no models/inkling2")
        return
    apply(SCHED, OBSERVE_ANCHOR, OBSERVE_NEW, "self._rad_ik_observe(", "inkling2 adaptive-k observe hook")
    apply(SCHED, K_ANCHOR, K_NEW, "self._rad_ik_k(num_spec_tokens_to_schedule)", "inkling2 adaptive-k override")
    # append the helper methods once (idempotent on the sentinel)
    s = SCHED.read_text()
    if "def _rad_ik_k" not in s:
        s = s.rstrip("\n") + "\n" + TAIL
        import ast
        ast.parse(s)
        SCHED.write_text(s)
        print("  OK    inkling2 adaptive-k scheduler helpers")
    else:
        print("  NOOP  inkling2 adaptive-k scheduler helpers already applied")
    # capture the adaptive verify widths as FULL decode graphs (target runner)
    apply(MR, MR_ANCHOR, MR_NEW, "_rad_ik_query_lens(", "inkling2 adaptive-k multi-width capture")
    m = MR.read_text()
    if "def _rad_ik_query_lens" not in m:
        m = m.rstrip("\n") + "\n" + MR_TAIL
        import ast
        ast.parse(m)
        MR.write_text(m)
        print("  OK    inkling2 adaptive-k query-lens helper")
    else:
        print("  NOOP  inkling2 adaptive-k query-lens helper already applied")


if __name__ == "__main__":
    main()
