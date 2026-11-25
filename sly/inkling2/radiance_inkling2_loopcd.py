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
"""
import os

LOOPCD_ENV = "RADIANCE_INKLING2_LOOPCD"
OMEGA_ENV = "RADIANCE_INKLING2_LOOPCD_OMEGA"
LOOPS_ENV = "RADIANCE_INKLING2_LOOPS"

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


def apply_hidden(h_R, h_1, omega: float):
    """LoopCD-Hidden combination h' = h_R + omega * (h_R - h_1).

    omega == 0 or h_1 is None -> h_R (identity, unguided). Otherwise a new
    tensor; neither input is mutated.
    """
    if omega == 0.0 or h_1 is None:
        return h_R
    return h_R + omega * (h_R - h_1)


def remap_capture_ids(ids, prelude: int, core: int, passes: int, new_passes: int):
    """Remap DSpark capture visit ids when the loop runs `new_passes` < `passes`.

    Visits: prelude [0, prelude), core `passes` x `core`, then the coda.
    Prelude ids and ids in a kept pass stay; ids in a dropped pass map to the
    same layer of the last kept pass; coda ids and the final-output sentinel
    (== total visits) shift down by the removed visits. Identity if
    new_passes >= passes. Duplicates are kept (one feature per id).
    """
    if new_passes >= passes:
        return tuple(ids)
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


_LOGGED = False


def aux_capture_ids(ids, prelude: int, core: int, passes: int, new_passes: int):
    """Sorted, remapped capture ids for the forward; logs one proof line."""
    global _LOGGED
    old = tuple(sorted(ids))
    new = remap_capture_ids(old, prelude, core, passes, new_passes)
    if not _LOGGED:
        _LOGGED = True
        on = enabled()
        print(
            f"RADIANCE_LOOPCD_PROOF loops={min(new_passes, passes)}/{passes} "
            f"loopcd={int(on)} omega={omega() if on else 0.0} "
            f"capture_ids={old}->{new}",
            flush=True,
        )
    return new
