"""LoopCD-Hidden for Inkling-2-Mini (arxiv 2610.02185), pure logic.

Contrasts the final loop pass's hidden state against the first pass's,
*before* the coda layers, then the existing coda + lm_head run once:

    h' = h_R + omega * (h_R - h_1)      (zero extra output pass)
    z'   = lm_head(coda(h'))

Fixed guidance strength (omega in [0.2, 0.3] for generation per paper §5.3;
default 0.5). No torch import: the arithmetic is duck-typed over tensors, so
this module is CPU-testable without vLLM or a real model.

Env knobs:
  RADIANCE_INKLING2_LOOPCD=1        enable (default off)
  RADIANCE_INKLING2_LOOPCD_OMEGA    contrast strength (default 0.5)
  RADIANCE_INKLING2_LOOPS           loop pass count override (default: the
                                    checkpoint's; e.g. 2 to halve 3->2)
  RADIANCE_INKLING2_LOOP_MODE       early_exit (default: run the FIRST n passes)
                                    | late_entry (run the LAST n passes, arxiv
                                    2611.24576, training-free: no fine-tune, no
                                    Wake verification, lossy)
  RADIANCE_INKLING2_LE_BANK         none (default) | prelude: what the first
                                    executed pass reads as loop message; none =
                                    no message, prelude = the prelude state
  RADIANCE_INKLING2_LE_MSG_SCALE    scale of the prelude message (default 1.0)
"""
import os

LOOPCD_ENV = "RADIANCE_INKLING2_LOOPCD"
OMEGA_ENV = "RADIANCE_INKLING2_LOOPCD_OMEGA"
LOOPS_ENV = "RADIANCE_INKLING2_LOOPS"
MODE_ENV = "RADIANCE_INKLING2_LOOP_MODE"
BANK_ENV = "RADIANCE_INKLING2_LE_BANK"
SCALE_ENV = "RADIANCE_INKLING2_LE_MSG_SCALE"

DEFAULT_OMEGA = 0.5


def enabled() -> bool:
    """LoopCD-Hidden on iff RADIANCE_INKLING2_LOOPCD=1."""
    return os.environ.get(LOOPCD_ENV, "") == "1"


def omega() -> float:
    """Contrast strength omega; falls back to 0.5 on garbage."""
    try:
        return float(os.environ.get(OMEGA_ENV, str(DEFAULT_OMEGA)))
    except (TypeError, ValueError):
        return DEFAULT_OMEGA


def loops_override() -> int | None:
    """Requested loop pass count (RADIANCE_INKLING2_LOOPS), else None.

    None means: serve the checkpoint's num_loop_passes. A value clamps the
    pass count downward at forward time (never upward, to stay within the
    allocated per-pass KV/conv caches).
    """
    raw = os.environ.get(LOOPS_ENV, "").strip()
    if not raw:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n >= 1 else None


def loop_mode() -> str:
    """early_exit (stock) or late_entry; anything else falls back to early_exit."""
    mode = os.environ.get(MODE_ENV, "").strip()
    return mode if mode == "late_entry" else "early_exit"


def _bank_mode() -> str:
    return "prelude" if os.environ.get(BANK_ENV, "").strip() == "prelude" else "none"


def msg_scale() -> float:
    """Scale of the prelude stand-in message; 1.0 on garbage."""
    try:
        return float(os.environ.get(SCALE_ENV, "1.0"))
    except (TypeError, ValueError):
        return 1.0


def first_pass(passes: int, new_passes: int) -> int:
    """Index of the first executed core pass: 0, or passes-new under late_entry."""
    if loop_mode() != "late_entry" or new_passes >= passes:
        return 0
    return passes - new_passes


def pass_range(passes: int, new_passes: int) -> range:
    """Checkpoint pass indices to run (each keeps its own per-pass caches)."""
    first = first_pass(passes, new_passes)
    return range(first, passes) if first else range(min(new_passes, passes))


def blends(pass_idx: int, first: int) -> bool:
    """Does this pass blend a loop message into its sublayer inputs?

    Stock: every pass after the first. Late entry: every pass after the first
    executed one; the first executed pass blends only with LE_BANK=prelude.
    """
    if pass_idx > first:
        return True
    return pass_idx == first > 0 and _bank_mode() == "prelude"


def entry_bank(prelude_partial, scale: float):
    """Message bank for a skipped-in first pass: the prelude state, twice."""
    return (prelude_partial * scale).unsqueeze(0).expand(2, -1, -1)


def select_bank(bank, looped: bool, pass_idx: int, first: int, prelude_partial):
    """Message bank of this pass (None = no message blending)."""
    if not looped:
        return None
    if pass_idx == first:
        return entry_bank(prelude_partial, msg_scale())
    return bank


def apply_hidden(h_R, h_1, omega: float):
    """LoopCD-Hidden combination h' = h_R + omega * (h_R - h_1).

    omega == 0 or h_1 is None -> h_R (identity, unguided). Otherwise a new
    tensor; neither input is mutated.
    """
    if omega == 0.0 or h_1 is None:
        return h_R
    return h_R + omega * (h_R - h_1)


def remap_capture_ids(
    ids, prelude: int, core: int, passes: int, new_passes: int, late: bool = False
):
    """Remap DSpark capture visit ids when the loop runs `new_passes` < `passes`.

    late=True (Late Entry): the LAST passes run, so ids in a kept pass keep
    their pass and shift down by the skipped visits; ids in a skipped pass map
    to the same layer of the first kept pass.

    Visits: prelude [0, prelude), core `passes` x `core`, then the coda.
    Prelude ids and ids in a kept pass stay; ids in a dropped pass map to the
    same layer of the last kept pass; coda ids and the final-output sentinel
    (== total visits) shift down by the removed visits. Identity if
    new_passes >= passes. Duplicates are kept (one feature per id).
    """
    if new_passes >= passes:
        return tuple(ids)
    if late:
        return _remap_late(ids, prelude, core, passes, new_passes)
    kept_end = prelude + core * new_passes
    core_end = prelude + core * passes
    shift = (passes - new_passes) * core
    out = []
    for v in ids:
        if v < kept_end:
            out.append(v)
        elif v < core_end:
            out.append(kept_end - core + (v - prelude) % core)
        else:
            out.append(v - shift)
    return tuple(out)


def _remap_late(ids, prelude: int, core: int, passes: int, new_passes: int):
    skipped = (passes - new_passes) * core
    out = []
    for v in ids:
        if v < prelude:
            out.append(v)
        elif v < prelude + skipped:  # skipped pass -> same layer, first kept pass
            out.append(prelude + (v - prelude) % core)
        else:  # kept pass or coda/sentinel: shift down by the skipped visits
            out.append(v - skipped)
    return tuple(out)


_LOGGED = False


def aux_capture_ids(ids, prelude: int, core: int, passes: int, new_passes: int):
    """Sorted, remapped capture ids for the forward; logs one proof line."""
    global _LOGGED
    old = tuple(sorted(ids))
    late = loop_mode() == "late_entry"
    new = remap_capture_ids(old, prelude, core, passes, new_passes, late=late)
    if not _LOGGED:
        _LOGGED = True
        on = enabled()
        print(
            f"RADIANCE_LOOPCD_PROOF loops={min(new_passes, passes)}/{passes} "
            f"loopcd={int(on)} omega={omega() if on else 0.0} "
            f"capture_ids={old}->{new} "
            f"mode={loop_mode()} first={first_pass(passes, new_passes)} "
            f"bank={_bank_mode()} msg_scale={msg_scale()}",
            flush=True,
        )
    return new
