"""CPU: RADIANCE_GDN_SCAN_V2 picks the v2 chunk_scan entry point.

radiance_gdn binds its kernels at import, so each case sets the env, installs a fake `r4d`
(select() answers with a name, like the real registry) and imports a fresh copy of the module.
Default is the stock kernel: with the knob unset the module must not even ask for the variant.
"""
import importlib
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
STOCK, V2 = "gdn_chunk_scan_k128_v128_c64_bf16", "gdn_chunk_scan_k128_v128_c64_bf16_v2"


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


def load(monkeypatch, has_v2, knob):
    monkeypatch.setitem(sys.modules, "r4d", fake_r4d(has_v2))
    for k in ("RADIANCE_GDN_SCAN_V2", "RADIANCE_GDN_LAZY", "RADIANCE_GDN_FUSED_UPDATE"):
        monkeypatch.delenv(k, raising=False)
    if knob is not None:
        monkeypatch.setenv("RADIANCE_GDN_SCAN_V2", knob)
    monkeypatch.setenv("RADIANCE_USE_R4D", "1")
    monkeypatch.syspath_prepend(str(ROOT))
    sys.modules.pop("radiance_gdn", None)
    return importlib.import_module("radiance_gdn")


def test_knob_unset_is_stock_and_never_asks_for_v2(monkeypatch):
    g = load(monkeypatch, True, None)
    assert g._CHUNK_SCAN.__name__ == STOCK
    assert all("variant" not in kw for _, kw in sys.modules["r4d"].asked)


def test_knob_zero_is_stock(monkeypatch):
    assert load(monkeypatch, True, "0")._CHUNK_SCAN.__name__ == STOCK


def test_knob_on_with_v2_picks_v2(monkeypatch):
    assert load(monkeypatch, True, "1")._CHUNK_SCAN.__name__ == V2


def test_knob_on_without_v2_keeps_stock_and_says_so_once(monkeypatch, capsys):
    g = load(monkeypatch, False, "1")
    assert g._CHUNK_SCAN.__name__ == STOCK
    err = capsys.readouterr().err
    assert err.count("RADIANCE_GDN_SCAN_V2=1") == 1 and "v2" in err
