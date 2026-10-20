#!/usr/bin/env python3
"""CPU: RADIANCE_GDN_SCAN_V2 picks the v2 chunk_scan entry point.

radiance_gdn binds its kernels at import, so each case sets the env, installs a fake `r4d`
(select() answers with a name, like the real registry) and imports a fresh copy of the module.
Default is the stock kernel: with the knob unset the module must not even ask for the variant.

Lives under sly/tests (not ci/tests) because it imports radiance_gdn, which imports torch: the
ci/tests job runs on a bare Python without torch, `pytest sly/tests` does not.

Run:  python3 sly/tests/test_gdn_scan_v2_select_cpu.py     or   pytest sly/tests
"""
import contextlib
import importlib
import io
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STOCK, V2 = "gdn_chunk_scan_k128_v128_c64_bf16", "gdn_chunk_scan_k128_v128_c64_bf16_v2"
KNOBS = ("RADIANCE_GDN_SCAN_V2", "RADIANCE_GDN_LAZY", "RADIANCE_GDN_FUSED_UPDATE")


def fake_r4d(has_v2):
    m = types.ModuleType("r4d")
    m.GDN_HEAD_K, m.GDN_HEAD_V, m.GDN_CHUNK = 128, 128, 64
    asked = []

    def select(op, **kw):
        asked.append((op, kw))
        if op != "gdn_chunk_scan":
            return f"{op}_stub"
        # the real registry ignores keys a kernel has no constraint on: a build without v2 answers STOCK
        return V2 if has_v2 and kw.get("variant") == "v2" else STOCK
    m.select, m.asked = select, asked
    for name in (STOCK, V2, "gdn_conv_prep_stub", "gdn_kkt_solve_stub", "gdn_conv_update_stub",
                 "gdn_recurrent_update_stub", "gdn_fused_update_stub", "gdn_gated_rmsnorm_stub"):
        setattr(m, name, (lambda n: (lambda *a: n))(name))
        getattr(m, name).__name__ = name
    return m


@contextlib.contextmanager
def loaded(has_v2, knob):
    """Import a fresh radiance_gdn with this (knob, registry) pair; restore everything on exit."""
    env, path = dict(os.environ), list(sys.path)
    had_r4d = sys.modules.get("r4d")
    err = io.StringIO()
    try:
        sys.modules["r4d"] = fake_r4d(has_v2)
        for k in KNOBS:
            os.environ.pop(k, None)
        if knob is not None:
            os.environ["RADIANCE_GDN_SCAN_V2"] = knob
        os.environ["RADIANCE_USE_R4D"] = "1"
        sys.path.insert(0, str(ROOT))
        sys.modules.pop("radiance_gdn", None)
        with contextlib.redirect_stderr(err):
            mod = importlib.import_module("radiance_gdn")
        yield mod, err
    finally:
        os.environ.clear()                    # env is a mapping, not a list: restore in place
        os.environ.update(env)
        sys.path[:] = path
        sys.modules.pop("radiance_gdn", None)
        if had_r4d is None:
            sys.modules.pop("r4d", None)
        else:
            sys.modules["r4d"] = had_r4d


def test_knob_unset_is_stock_and_never_asks_for_v2():
    with loaded(True, None) as (g, _):
        assert g._CHUNK_SCAN.__name__ == STOCK
        assert all("variant" not in kw for _, kw in sys.modules["r4d"].asked)


def test_knob_zero_is_stock():
    with loaded(True, "0") as (g, _):
        assert g._CHUNK_SCAN.__name__ == STOCK


def test_knob_on_with_v2_picks_v2():
    with loaded(True, "1") as (g, _):
        assert g._CHUNK_SCAN.__name__ == V2


def test_knob_on_without_v2_keeps_stock_and_says_so_once():
    with loaded(False, "1") as (g, err):
        assert g._CHUNK_SCAN.__name__ == STOCK
        assert err.getvalue().count("RADIANCE_GDN_SCAN_V2=1") == 1
        assert "v2" in err.getvalue()


def main():
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:                                  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all tests passed" if not fails else f"{fails} test(s) failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
