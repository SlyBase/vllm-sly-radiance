#!/usr/bin/env python3
"""GPU-less check of radiance_lookup_draft: the reference semantics (longest suffix match, ties, cap, the
enter / stay / hot decision), which speculators get hooked, when the override runs and what happens on an error.

torch, triton and vllm are stubbed and the override (`_override`) is replaced by a recorder, so this proves the
hook and the decision rules -- not the kernels, which sly/bench_lookup_draft.py --check compares against the same
reference on the card.

  python3 sly/check_lookup_draft.py
"""
import importlib
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("FAIL", msg)


def stub_env(gfx12=True, dflash2=True):
    """Fresh stub modules; returns (RunnerCls, Dflash2Cls, OtherCls)."""
    for name in [n for n in sys.modules if n.startswith(("vllm", "triton", "radiance_lookup_draft"))]:
        del sys.modules[name]
    tr = types.ModuleType("triton")
    tr.jit = lambda f: f
    tr.next_power_of_2 = lambda n: 1 << max(0, (n - 1).bit_length())
    tl = types.ModuleType("triton.language")
    tl.constexpr = object
    tr.language = tl
    sys.modules.update({"triton": tr, "triton.language": tl})

    class DFlash2Speculator:
        def __init__(self):
            self.calls = []

        def propose(self, *a, **k):
            self.calls.append((a, k))
            return "drafts"

    class OtherSpeculator:
        def propose(self, *a, **k):
            return "other"

    class GPUModelRunner:
        made = 0

        def __init__(self, kind="dflash2"):
            self.req_states = types.SimpleNamespace(tag="req_states")
            self.speculator = {"dflash2": DFlash2Speculator, "other": OtherSpeculator, "none": lambda: None}[kind]()
            GPUModelRunner.made += 1

    rocm = types.ModuleType("vllm.platforms.rocm")
    rocm.on_gfx12x = lambda: gfx12
    mods = {"vllm": types.ModuleType("vllm"), "vllm.platforms": types.ModuleType("vllm.platforms"),
            "vllm.platforms.rocm": rocm, "vllm.v1": types.ModuleType("vllm.v1"),
            "vllm.v1.worker": types.ModuleType("vllm.v1.worker"),
            "vllm.v1.worker.gpu": types.ModuleType("vllm.v1.worker.gpu"),
            "vllm.v1.worker.gpu.spec_decode": types.ModuleType("vllm.v1.worker.gpu.spec_decode"),
            "vllm.v1.worker.gpu.spec_decode.dflash2": types.ModuleType("vllm.v1.worker.gpu.spec_decode.dflash2")}
    mr = types.ModuleType("vllm.v1.worker.gpu.model_runner")
    mr.GPUModelRunner = GPUModelRunner
    mods["vllm.v1.worker.gpu.model_runner"] = mr
    if dflash2:
        sp = types.ModuleType("vllm.v1.worker.gpu.spec_decode.dflash2.speculator")
        sp.DFlash2Speculator = DFlash2Speculator
        mods["vllm.v1.worker.gpu.spec_decode.dflash2.speculator"] = sp
    sys.modules.update(mods)
    return GPUModelRunner, DFlash2Speculator, OtherSpeculator


def load(gfx12=True, dflash2=True, **env):
    for k in [k for k in os.environ if k.startswith("RADIANCE_LOOKUP_")]:
        del os.environ[k]
    os.environ.update({f"RADIANCE_LOOKUP_{k}": v for k, v in env.items()})
    runner, dsp, other = stub_env(gfx12, dflash2)
    return importlib.import_module("radiance_lookup_draft"), runner, dsp, other


def reference_semantics():
    R, *_ = load(DRAFT="1")
    rb = R.ref_best_match

    a, b, c, d, e = 1, 2, 3, 4, 5
    # suffix "b c d" occurred once before, followed by e
    ids = [a, b, c, d, e, 9, 8, b, c, d]
    check(rb(ids, len(ids), 3) == (3, 4), f"single earlier occurrence: {rb(ids, len(ids), 3)}")
    check(rb(ids, len(ids), 4) == (0, 0), "a match shorter than minl is no match")
    # the longer match wins over the more recent shorter one
    ids = [7, b, c, d, 0, 0, c, d, 6, b, c, d]
    check(rb(ids, len(ids), 2) == (3, 4), f"longest wins over most recent: {rb(ids, len(ids), 2)}")
    # a tie goes to the most recent occurrence
    ids = [b, c, d, 1, b, c, d, 2, b, c, d]
    check(rb(ids, len(ids), 3) == (3, 7), f"tie -> most recent: {rb(ids, len(ids), 3)}")
    # the match length is capped
    ids = list(range(100, 140)) * 2
    check(rb(ids, len(ids), 3) == (24, 40), f"cap: {rb(ids, len(ids), 3)}")
    # no self match: a history without repeats has none
    check(rb([1, 2, 3, 4, 5, 6], 6, 1) == (0, 0), "no repeats, no match")
    # a run of one token matches itself one position back, with a one-token continuation
    ids = [5, 5, 5, 5, 5, 5]
    check(rb(ids, 6, 3) == (5, 5), f"run of one token: {rb(ids, 6, 3)}")
    # only the first `length` tokens count (the rest of the buffer is stale)
    ids = [b, c, d, 1, b, c, d, 99, 99, 99]
    check(rb(ids, 7, 3) == (3, 3), f"length limits the history: {rb(ids, 7, 3)}")
    # min_dist: a source closer than that is not a candidate, a farther one is taken even if a nearer match is longer
    ids = [b, c, d, 1, 2, 3, 4, 5, 6, 7, 8, 9, b, c, d]                 # earlier occurrence continues at index 3
    check(rb(ids, len(ids), 3, min_dist=12) == (3, 3), f"source exactly min_dist back: {rb(ids, len(ids), 3, min_dist=12)}")
    check(rb(ids, len(ids), 3, min_dist=13) == (0, 0), "source closer than min_dist: no candidate")
    ids = [7, b, c, d, 0, 0, c, d, 6, b, c, d]
    check(rb(ids, len(ids), 2, min_dist=8) == (3, 4), f"min_dist keeps the far match: {rb(ids, len(ids), 2, min_dist=8)}")
    check(rb(ids, len(ids), 2, min_dist=9) == (0, 0), f"min_dist 9 leaves only the too-close ones: {rb(ids, len(ids), 2, min_dist=9)}")
    check(rb(ids, len(ids), 2, min_dist=0) == (3, 4) and rb(ids, len(ids), 2) == (3, 4), "min_dist 0 = no restriction")

    dec = R.ref_decide
    K = 7
    check(dec(8, 100, 200, K, 0, -1, 4) == (True, 7, False), "match >= enter starts lookup mode")
    check(dec(7, 100, 200, K, 0, -1, 4)[0] is False, "match < enter, not hot: DFlash")
    check(dec(3, 100, 200, K, 1, 190, 10)[0:1] == (True,) and dec(3, 100, 200, K, 1, 190, 10)[2] is True,
          "hot (last lookup step accepted 9 >= 3): stay on a short match")
    check(dec(3, 100, 200, K, 1, 196, 4)[0] is True, "accepted 3 (= HOT) is hot")
    check(dec(3, 100, 200, K, 1, 197, 3)[0] is False, "accepted 2 is not hot")
    check(dec(3, 100, 200, K, 0, 190, 10)[0] is False, "the last step was DFlash's: not hot")
    check(dec(3, 100, 200, K, 1, 190, 10)[0] is True and dec(3, 100, 200, K, 1, 190, 9)[0] is False,
          "the same step with a broken history (190 + 9 != 200) is not hot")
    check(dec(3, 100, 200, K, 1, 150, 10)[0] is False, "history not continuous (slot reused): not hot")
    check(dec(0, 0, 200, K, 1, 190, 10)[0] is False, "no match is never taken")
    check(dec(9, 197, 200, K, 0, -1, 4)[1] == 3, "continuation clamps to the known tokens")
    check(dec(9, 100, 200, K, 0, -1, 4, enter=10)[0] is False, "ENTER knob")


def hook_behaviour():
    # RADIANCE_LOOKUP_DRAFT=0: nothing wrapped
    R, Runner, DSp, Other = load(DRAFT="0")
    orig_init = Runner.__init__
    check(R.install() is True and Runner.__init__ is orig_init, "DRAFT=0 must not wrap the runner")

    # on by default: wraps once, hooks DFlash2 only
    R, Runner, DSp, Other = load()
    seen = []
    R._override = lambda sp, rs, ib, ns: seen.append((sp, rs, ib, ns))
    check(R.install() is True and getattr(Runner.__init__, "_radiance_lookup", False), "install() must wrap __init__")
    wrapped = Runner.__init__
    check(R.install() is True and Runner.__init__ is wrapped, "install() is not idempotent")

    r = Runner("dflash2")
    sp = r.speculator
    ib, ns = object(), object()
    check(sp.propose(ib, "attn", "slots", "h", None, ns, "rej", "last", "pref", "t", "seeds", dp_sync=None) == "drafts",
          "the hooked propose must return the original result")
    check(len(seen) == 1 and seen[0][1] is r.req_states and seen[0][2] is ib and seen[0][3] is ns,
          "positional call: override gets req_states, input_batch, num_sampled")
    sp.propose(input_batch=ib, attn_metadata=1, slot_mappings=2, last_hidden_states=3, aux_hidden_states=4,
               num_sampled=ns, num_rejected=5, last_sampled=6, next_prefill_tokens=7, temperature=8, seeds=9)
    check(len(seen) == 2 and seen[1][2] is ib and seen[1][3] is ns, "keyword call")
    sp.propose(input_batch=ib, num_sampled=ns, dummy_run=True)
    check(len(seen) == 2, "a dummy run (warm-up, graph capture, profiling) must not be overridden")
    check(len(sp.calls) == 3, "the original propose runs every time")

    check(Runner("other").speculator.propose(ib, "a", "s", "h", None, ns) == "other" and len(seen) == 2,
          "other speculators stay untouched")
    check(Runner("none").speculator is None, "no speculator: nothing to hook")

    # the switch file: the override runs only while it exists
    import tempfile
    sw = os.path.join(tempfile.mkdtemp(), "on")
    R, Runner, DSp, Other = load(DRAFT="1", SWITCH=sw)
    seen2 = []
    R._override = lambda sp, rs, ib, ns: seen2.append(1)
    R.install()
    sp = Runner("dflash2").speculator
    sp.propose(1, 2, 3, 4, 5, 6)
    check(not seen2, "switch file absent: no override")
    open(sw, "w").close()
    sp.propose(1, 2, 3, 4, 5, 6)
    check(len(seen2) == 1, "switch file present: override")
    os.remove(sw)
    sp.propose(1, 2, 3, 4, 5, 6)
    check(len(seen2) == 1, "switch file removed again: no override")

    # an error turns the override off for the process; the result is still the graph's draft
    R, Runner, DSp, Other = load(DRAFT="1")

    def boom(*a):
        raise RuntimeError("boom")
    R._override = boom
    R.install()
    sp = Runner("dflash2").speculator
    check(sp.propose(1, 2, 3, 4, 5, 6) == "drafts" and R._state["failed"], "an error must fall back to the graph's draft")
    R._override = lambda *a: (_ for _ in ()).throw(AssertionError("must not run again"))
    check(sp.propose(1, 2, 3, 4, 5, 6) == "drafts", "after a failure the override is not tried again")

    # knobs and platforms
    for env, why in (({"ENTER": "2", "STAY": "3"}, "ENTER < STAY"), ({"ENTER": "30"}, "ENTER above the match cap"),
                     ({"HOT": "0"}, "HOT 0")):
        R, Runner, DSp, Other = load(DRAFT="1", **env)
        orig_init = Runner.__init__
        check(R.install() is True and Runner.__init__ is orig_init, f"{why} must not install")
    R, Runner, DSp, Other = load(gfx12=False, DRAFT="1")
    orig_init = Runner.__init__
    check(R.install() is True and Runner.__init__ is orig_init, "not gfx12x: not hooked")
    R, Runner, DSp, Other = load(dflash2=False, DRAFT="1")
    check(R.install() is False, "a vLLM without the DFlash2 speculator reports itself as not applicable")


def main():
    reference_semantics()
    hook_behaviour()
    if failures:
        print(f"FAIL: {len(failures)} finding(s)")
        sys.exit(1)
    print("radiance_lookup_draft check OK")


if __name__ == "__main__":
    main()
