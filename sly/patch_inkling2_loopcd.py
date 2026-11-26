#!/usr/bin/env python3
"""LoopCD-Hidden for Inkling-2-Mini (arxiv 2610.02185), opt-in source patch.

Contrasts the final loop pass's hidden state against the first pass's before
the coda layers (h' = h_R + omega * (h_R - h_1)), then the existing coda +
lm_head run once -- zero extra output pass.

Opt-in: RADIANCE_INKLING2_LOOPCD=1, strength RADIANCE_INKLING2_LOOPCD_OMEGA
(default 0.5), pass-count override RADIANCE_INKLING2_LOOPS (e.g. 2 to run
3 -> 2 looped passes; DSpark capture visit ids are remapped to match).
RADIANCE_INKLING2_LOOP_MODE=late_entry runs the LAST n passes instead of the
first n (arxiv 2611.24576, training-free; see radiance_inkling2_loopcd.py).
NOOP on a vLLM without models/inkling2.

Patches vllm/models/inkling2/nvidia/loop.py (Inkling2LoopModel.forward +
one new helper method). Idempotent: re-running is a NOOP.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
ROOT = SP / "vllm/models/inkling2"
LOOP = ROOT / "nvidia" / "loop.py"

# ---- import the radiance logic (guarded: inert if the module is absent)
IMPORT_ANCHOR = "from .ops.loop import blend_rms_norm, pair_rms_norm\n"
IMPORT_NEW = IMPORT_ANCHOR + (
    "\ntry:\n"
    "    from radiance_inkling2_loopcd import (\n"
    "        apply_hidden as _rad_apply_hidden,\n"
    "        aux_capture_ids as _rad_aux_capture_ids,\n"
    "        blends as _rad_blends,\n"
    "        enabled as _rad_loopcd_enabled,\n"
    "        first_pass as _rad_first_pass,\n"
    "        loops_override as _rad_loopcd_loops,\n"
    "        omega as _rad_loopcd_omega,\n"
    "        pass_range as _rad_pass_range,\n"
    "        select_bank as _rad_select_bank,\n"
    "    )\nexcept Exception:  # module missing: stay inert\n"
    "    _rad_apply_hidden = None\n"
    "    def _rad_aux_capture_ids(ids, *_):\n"
    "        return tuple(ids)\n"
    "    def _rad_loopcd_enabled():\n"
    "        return False\n"
    "    def _rad_loopcd_loops():\n"
    "        return None\n"
    "    def _rad_loopcd_omega():\n"
    "        return 0.0\n"
    "    def _rad_first_pass(passes, new_passes):\n"
    "        return 0\n"
    "    def _rad_pass_range(passes, new_passes):\n"
    "        return range(min(new_passes, passes))\n"
    "    def _rad_blends(pass_idx, first):\n"
    "        return pass_idx > 0\n"
    "    def _rad_select_bank(bank, looped, pass_idx, first, prelude_partial):\n"
    "        return bank if looped else None\n"
)

# ---- honor RADIANCE_INKLING2_LOOPS (clamp pass count downward)
NPASSES_ANCHOR = (
    "        num_passes = self.config.num_loop_passes\n"
    "        stack = inputs_embeds.new_empty(\n"
)
NPASSES_NEW = (
    "        num_passes = self.config.num_loop_passes\n"
    "        _rad_loops = _rad_loopcd_loops()\n"
    "        if _rad_loops is not None:\n"
    "            num_passes = min(_rad_loops, num_passes)\n"
    "        _rad_first = _rad_first_pass(self.config.num_loop_passes, num_passes)\n"
    "        # DSpark capture visit ids follow the shortened loop (see radiance module)\n"
    "        _rad_aux = _rad_aux_capture_ids(\n"
    "            self.aux_hidden_state_layers, core.start, len(core),\n"
    "            self.config.num_loop_passes, num_passes,\n"
    "        )\n"
    "        stack = inputs_embeds.new_empty(\n"
)

# ---- Late Entry hooks: pass range, message blending, first-pass bank
VISITS_ANCHOR = "        visits += [(i, p) for p in range(num_passes) for i in core]\n"
VISITS_NEW = (
    "        visits += [\n"
    "            (i, p)\n"
    "            for p in _rad_pass_range(self.config.num_loop_passes, num_passes)\n"
    "            for i in core\n"
    "        ]\n"
)
LOOPED_ANCHOR = (
    "            looped = pass_idx > 0\n"
    "            if i == core.start and not looped:\n"
)
LOOPED_NEW = (
    "            looped = _rad_blends(pass_idx, _rad_first)\n"
    "            if i == core.start and pass_idx == _rad_first:\n"
)
NEXT_NORM_ANCHOR = "                next_norm = None if next_pass else self.layers[next_i].attn_norm\n"
NEXT_NORM_NEW = (
    "                next_norm = (\n"
    "                    None\n"
    "                    if _rad_blends(next_pass, _rad_first)\n"
    "                    else self.layers[next_i].attn_norm\n"
    "                )\n"
)
BANK_ANCHOR = "                bank if looped else None,\n"
BANK_NEW = (
    "                _rad_select_bank(bank, looped, pass_idx, _rad_first, prelude_partial),\n"
)

# ---- capture on the remapped ids (a duplicate id yields a duplicate feature)
CAPTURE_ANCHOR = (
    "            if visit in self.aux_hidden_state_layers:\n"
    "                aux_hidden_states.append(stack[num_blocks].clone())\n"
)
CAPTURE_NEW = (
    "            for _ in range(_rad_aux.count(visit)):\n"
    "                aux_hidden_states.append(stack[num_blocks].clone())\n"
)

# ---- capture h_1 after pass 0, inject h' before the coda, per forward call
INJECT_ANCHOR = "            )\n        assert inputs is not None\n"
INJECT_NEW = (
    "            )\n"
    "            # ---- RADIANCE LoopCD-Hidden (patch_inkling2_loopcd.py) ----\n"
    "            if _rad_loopcd_enabled():\n"
    "                self._rad_h1 = None\n"
    "            inputs = self._rad_loopcd_step(\n"
    "                i, pass_idx - _rad_first, num_passes, inputs, stack, num_blocks\n"
    "            )\n"
    "        assert inputs is not None\n"
)

# ---- the helper method appended after forward
METHOD_TAIL = '''
def _rad_loopcd_step(
    self, i, pass_idx, num_passes, inputs, stack, num_blocks
):
    """LoopCD-Hidden: capture h_1 after pass 0, inject h' before the coda.

    h' = h_R + omega * (h_R - h_1) replaces the last pass's state so the
    coda + lm_head run once on the guided state (zero extra output pass).
    """
    if not _rad_loopcd_enabled():
        return inputs
    core = self.loop_core
    if not core or i != core.stop - 1:
        return inputs
    if pass_idx == 0:
        self._rad_h1 = inputs[0].clone()
        return inputs
    if pass_idx == num_passes - 1 and self._rad_h1 is not None:
        h_prime = _rad_apply_hidden(
            inputs[0], self._rad_h1, _rad_loopcd_omega()
        )
        if h_prime is not inputs[0]:
            stack[num_blocks].copy_(h_prime)
            inputs = (h_prime,) + inputs[1:]
    return inputs


Inkling2LoopModel._rad_loopcd_step = _rad_loopcd_step
'''


def main():
    if not ROOT.is_dir():
        print(f"  NOOP  inkling2 loopcd: this vLLM has no models/inkling2 ({ROOT})")
        return
    if not LOOP.is_file():
        print(f"  NOOP  inkling2 loopcd: {LOOP} missing")
        return
    apply(LOOP, IMPORT_ANCHOR, IMPORT_NEW, "_rad_apply_hidden", "inkling2 loopcd import")
    apply(LOOP, NPASSES_ANCHOR, NPASSES_NEW, "_rad_loops = _rad_loopcd_loops()", "inkling2 loopcd loops override")
    apply(LOOP, CAPTURE_ANCHOR, CAPTURE_NEW, "_rad_aux.count(visit)", "inkling2 loopcd capture-id remap")
    apply(LOOP, VISITS_ANCHOR, VISITS_NEW, "for p in _rad_pass_range(", "inkling2 late-entry pass range")
    apply(LOOP, LOOPED_ANCHOR, LOOPED_NEW, "_rad_blends(pass_idx, _rad_first)", "inkling2 late-entry blend flag")
    apply(LOOP, NEXT_NORM_ANCHOR, NEXT_NORM_NEW, "_rad_blends(next_pass", "inkling2 late-entry next norm")
    apply(LOOP, BANK_ANCHOR, BANK_NEW, "pass_idx, _rad_first, prelude_partial", "inkling2 late-entry bank")
    apply(LOOP, INJECT_ANCHOR, INJECT_NEW, "# ---- RADIANCE LoopCD-Hidden", "inkling2 loopcd contrast injection")
    s = LOOP.read_text()
    if "def _rad_loopcd_step" not in s:
        s = s.rstrip("\n") + "\n" + METHOD_TAIL
        import ast
        ast.parse(s)
        LOOP.write_text(s)
        print("  OK    inkling2 loopcd helper method")
    else:
        print("  NOOP  inkling2 loopcd helper method already applied")


if __name__ == "__main__":
    main()
