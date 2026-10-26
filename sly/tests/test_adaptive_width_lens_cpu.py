#!/usr/bin/env python3
"""Default verify-width ladder of the adaptive width (sly/radiance_adaptive_width.py).

The ladder must reach the draft depth: with k=15 and the old default {1,3,5,7} a request could only
verify 1, 3, 5, 7 or all 15 drafts. The default is the odd widths below the depth, then the depth,
so k=7 keeps exactly the ladder it shipped with (decisions bit-identical) and k=11/k=15 get the
in-between steps without an env override. No torch, no GPU.

Run:  python3 -m pytest sly/tests/test_adaptive_width_lens_cpu.py -q
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sly"))


def _aw(monkeypatch, lens=None):
    monkeypatch.setenv("RADIANCE_ADAPTIVE_WIDTH", "perseq")
    if lens is None:
        monkeypatch.delenv("RADIANCE_AW_LENS", raising=False)
    else:
        monkeypatch.setenv("RADIANCE_AW_LENS", lens)
    import radiance_adaptive_width as m
    return importlib.reload(m)


def test_k7_default_ladder_is_unchanged(monkeypatch):
    m = _aw(monkeypatch)
    assert m.AdaptiveWidth(7).lens == [1, 3, 5, 7]


def test_k11_default_ladder_reaches_the_depth(monkeypatch):
    m = _aw(monkeypatch)
    assert m.AdaptiveWidth(11).lens == [1, 3, 5, 7, 9, 11]


def test_k15_default_ladder_reaches_the_depth(monkeypatch):
    m = _aw(monkeypatch)
    assert m.AdaptiveWidth(15).lens == [1, 3, 5, 7, 9, 11, 13, 15]


def test_env_override_still_wins(monkeypatch):
    m = _aw(monkeypatch, "3,5")
    assert m.AdaptiveWidth(15).lens == [3, 5, 15]


def test_invalid_env_falls_back_to_the_default(monkeypatch):
    m = _aw(monkeypatch, "x,y")
    assert m.AdaptiveWidth(15).lens == [1, 3, 5, 7, 9, 11, 13, 15]
