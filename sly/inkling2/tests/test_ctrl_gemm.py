"""Loop message controllers on the batched HIP GEMV (RADIANCE_INKLING2_HIP_CTRL, default off).

The loop blend runs two torch.bmm per sublayer visit (down: [S,T,2D] x [S,r,2D]^T -> bf16 [S,T,r], then
[gain; forget]: [S,T,r] x [S,2D,r]^T -> fp32 [S,T,2D]). hipBLASLt takes 12 / 6-8 us per call on 0.6-1.8 MB of
weights (0.92 ms per verify step, 96 calls). `controller_gemms` runs the same two products on
radiance_inkling2_gemv.gemv_bf16_batched (stream index = grid.y, one launch each) and returns the fp32
gain/forget block, or None where the caller keeps torch.bmm.

CPU: the dispatcher against a fake extension that EXECUTES the call (it reads the raw pointers, strides and
batch strides it is handed, in fp64) -- so pointer / stride / batch arithmetic is checked against torch.bmm,
not just recorded. The kernel itself is covered by test_gemv_batched_hip.py (GPU).
"""
import ctypes
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import radiance_inkling2_gemv_hip as disp  # noqa: E402

SRC_PATH = HERE.parent / "radiance_inkling2_gemv_hip.py"
D2, R = 4608, 64  # 2 * hidden, gate_rank of Inkling-2-Mini


def _span(ptr, count, dtype):
    size = torch.empty((), dtype=dtype).element_size()
    return torch.frombuffer((ctypes.c_char * (count * size)).from_address(ptr), dtype=dtype)


class ExecExt:
    """gemv_bf16_batched that runs: y[b] = x[b] @ w[b]^T in fp64, rounded to the output dtype."""

    def __init__(self):
        self.calls = []

    def gemv_bf16_batched(self, x, w, y, nb, m, n, k, sx, sw, sy, bx, bw, by, out32, stream, cfg):
        self.calls.append((x, w, y, nb, m, n, k, sx, sw, sy, bx, bw, by, out32, stream, cfg))
        odt = torch.float32 if out32 else torch.bfloat16
        xs = _span(x, (nb - 1) * bx + (m - 1) * sx + k, torch.bfloat16)
        ws = _span(w, (nb - 1) * bw + (n - 1) * sw + k, torch.bfloat16)
        ys = _span(y, (nb - 1) * by + (m - 1) * sy + n, odt)
        for b in range(nb):
            xb = xs[b * bx:].as_strided((m, k), (sx, 1))
            wb = ws[b * bw:].as_strided((n, k), (sw, 1))
            ys[b * by:].as_strided((m, n), (sy, 1)).copy_((xb.double() @ wb.double().T).to(odt))


@pytest.fixture()
def on(monkeypatch):
    fake = ExecExt()
    monkeypatch.setattr(disp, "_ext", fake)
    monkeypatch.setattr(disp, "HIP_CTRL", True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: type("S", (), {"cuda_stream": 7})())
    return fake


def _inputs(s, t, seed=0):
    g = torch.Generator().manual_seed(seed)
    # the layer's stream-stacked buffers, sliced the way blend() does
    down = (torch.randn(4, R, D2, generator=g) * 0.05).bfloat16()
    gf = (torch.randn(4, D2, R, generator=g) * 0.2).bfloat16()
    pair = torch.randn(s, t, D2, generator=g).bfloat16()
    return pair, down[:s], gf[:s]


def _torch_path(pair, down, gf):
    # fp64 reference: the bf16 rounding of `hidden` is the only step where a different fp32 summation order
    # (hipBLASLt vs the kernel) can move a value, by one bf16 ulp, in a few elements
    hidden = torch.bmm(pair.double(), down.double().transpose(1, 2)).bfloat16()
    return torch.bmm(hidden.double(), gf.double().transpose(1, 2)).float()


@pytest.mark.parametrize("s", [1, 3, 4])
@pytest.mark.parametrize("t", range(1, 9))
def test_matches_the_two_bmm_for_every_row_count(on, s, t):
    pair, down, gf = _inputs(s, t, seed=s * 10 + t)
    out = disp.controller_gemms(pair, down, gf)
    assert out.dtype == torch.float32 and tuple(out.shape) == (s, t, D2)
    torch.testing.assert_close(out, _torch_path(pair, down, gf), rtol=2e-3, atol=2e-3)


def test_two_launches_with_batch_strides(on):
    pair, down, gf = _inputs(3, 8)
    disp.controller_gemms(pair, down, gf)
    a, b = on.calls
    # x w y nb M N K SX SW SY BX BW BY out32 stream cfg
    assert a[3:14] == (3, 8, R, D2, D2, D2, R, 8 * D2, R * D2, 8 * R, 0) and a[14:] == (7, -1)
    assert b[3:14] == (3, 8, D2, R, R, R, D2, 8 * R, D2 * R, 8 * D2, 1)
    assert a[2] == b[0]  # the bf16 hidden of the first launch is the x of the second


def test_row_slice_of_the_stacked_weights_is_routed(on):
    pair, down, gf = _inputs(4, 8)
    out = disp.controller_gemms(pair[3:], down[3:], gf[3:])  # the mlp stream: streams = slice(3, 4)
    torch.testing.assert_close(out, _torch_path(pair[3:], down[3:], gf[3:]), rtol=2e-3, atol=2e-3)


def test_default_off_and_knob(monkeypatch):
    assert disp.HIP_CTRL is False
    for val, want in ((None, False), ("0", False), ("1", True)):
        if val is None:
            monkeypatch.delenv("RADIANCE_INKLING2_HIP_CTRL", raising=False)
        else:
            monkeypatch.setenv("RADIANCE_INKLING2_HIP_CTRL", val)
        spec = importlib.util.spec_from_file_location("disp_ctrl_copy", SRC_PATH)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        assert m.HIP_CTRL is want


@pytest.mark.parametrize("what", ["knob_off", "no_ext", "old_ext", "rows>16", "fp32_pair", "strided_pair",
                                  "rank%8", "empty"])
def test_declines_outside_its_contract(on, monkeypatch, what):
    pair, down, gf = _inputs(3, 8)
    if what == "knob_off":
        monkeypatch.setattr(disp, "HIP_CTRL", False)
    elif what == "no_ext":
        monkeypatch.setattr(disp, "_ext", None)
    elif what == "old_ext":
        monkeypatch.setattr(disp, "_ext", object())  # a build without gemv_bf16_batched
    elif what == "rows>16":
        pair = torch.randn(3, 17, D2).bfloat16()
    elif what == "fp32_pair":
        pair = pair.float()
    elif what == "strided_pair":
        pair = torch.randn(3, 8, 2 * D2).bfloat16()[..., ::2]
    elif what == "rank%8":
        down, gf = torch.randn(3, 60, D2).bfloat16(), torch.randn(3, D2, 60).bfloat16()  # dense, r % 8 == 4
    elif what == "empty":
        pair = pair[:, :0]
    assert disp.controller_gemms(pair, down, gf) is None
    assert on.calls == []


# ---------------------------------------------------------------- proof lines: a silent knob cannot be told from a dead one
def test_proof_line_on_first_batched_launch_only(on, capsys):
    disp._LOGGED.clear()
    pair, down, gf = _inputs(3, 8)
    disp.controller_gemms(pair, down, gf)
    disp.controller_gemms(pair, down, gf)
    out = capsys.readouterr().out
    assert out.count("[radiance.inkling2] HIP_CTRL active") == 1


def test_fallback_reason_is_logged_once_when_the_knob_is_on(on, monkeypatch, capsys):
    disp._LOGGED.clear()
    monkeypatch.setattr(disp, "_ext", None)
    pair, down, gf = _inputs(3, 8)
    assert disp.controller_gemms(pair, down, gf) is None
    assert disp.controller_gemms(pair, down, gf) is None
    out = capsys.readouterr().out
    assert out.count("[radiance.inkling2] HIP_CTRL=1 but") == 1 and "extension" in out


def test_no_log_when_the_knob_is_off(monkeypatch, capsys):
    disp._LOGGED.clear()
    monkeypatch.setattr(disp, "HIP_CTRL", False)
    pair, down, gf = _inputs(3, 8)
    assert disp.controller_gemms(pair, down, gf) is None
    assert capsys.readouterr().out == ""
