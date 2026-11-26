"""radiance_inkling2_gemv.hip `gemv_bf16_batched` (loop controllers, RADIANCE_INKLING2_HIP_CTRL) -- GPU window.

Skips without a GPU or the compiled module (CPU coverage of the dispatcher: test_ctrl_gemm.py).
  * exact: small integers (bf16-exact, fp32 sums exact in any order), bit for bit, the two controller shapes
    (down: N = 64, K = 4608; gain/forget: N = 4608, K = 64) x batch 1..4 x M 1..8 x bf16 / fp32 output
  * numeric: random floats, the real two-stage chain vs fp64 (hidden rounded to bf16 as in the model)
  * batch isolation: NaN-poisoned neighbour streams do not leak, nothing is written past the batch span
  * nb = 1 equals plain gemv_bf16; graph capture; guards throw
Mutants (mutate_gemv_hip.py bz_*) must turn this file red.
"""
import pytest
import torch

import test_gemv_hip as base

D2, R = 4608, 64


@pytest.fixture()
def ext():
    e = base._gpu_ext()
    if not hasattr(e, "gemv_bf16_batched"):
        pytest.fail("build has no gemv_bf16_batched")
    return e


def _ints(g, *shape, hi=3):
    return torch.randint(-hi, hi + 1, shape, device="cuda", generator=g).to(torch.bfloat16)


def _launch(ext, x, w, y, nb, m, n, k, *, out32, cfg=-1, sy=None):
    sy = n if sy is None else sy
    ext.gemv_bf16_batched(x.data_ptr(), w.data_ptr(), y.data_ptr(), nb, m, n, k, k, k, sy,
                          x[0].numel(), w[0].numel(), y[0].numel(), int(out32),
                          int(torch.cuda.current_stream().cuda_stream), cfg)
    torch.cuda.synchronize()


@pytest.mark.parametrize("out32", [False, True])
@pytest.mark.parametrize("m", range(1, 9))
@pytest.mark.parametrize("nb", [1, 2, 3, 4])
@pytest.mark.parametrize("n,k", [(R, D2), (D2, R)])
def test_exact_on_the_controller_shapes(ext, n, k, nb, m, out32):
    g = torch.Generator(device="cuda").manual_seed(n + nb * 7 + m)
    x, w = _ints(g, nb, m, k), _ints(g, nb, n, k)
    y = torch.full((nb, m, n), float("nan"), device="cuda", dtype=torch.float32 if out32 else torch.bfloat16)
    _launch(ext, x, w, y, nb, m, n, k, out32=out32)
    ref = torch.bmm(x.double(), w.double().transpose(1, 2)).to(y.dtype)
    assert torch.equal(y, ref)


@pytest.mark.parametrize("cfg", [0, 1, 2])
def test_every_config_exact(ext, cfg):
    g = torch.Generator(device="cuda").manual_seed(cfg)
    x, w = _ints(g, 3, 8, D2), _ints(g, 3, R, D2)
    y = torch.empty(3, 8, R, device="cuda", dtype=torch.bfloat16)
    _launch(ext, x, w, y, 3, 8, R, D2, out32=False, cfg=cfg)
    assert torch.equal(y, torch.bmm(x.double(), w.double().transpose(1, 2)).to(torch.bfloat16))


@pytest.mark.parametrize("s,t", [(3, 8), (1, 8), (3, 1), (1, 5)])
def test_two_stage_chain_numeric(ext, s, t):
    g = torch.Generator(device="cuda").manual_seed(s * 10 + t)
    pair = torch.randn(s, t, D2, device="cuda", generator=g).bfloat16()
    down = (torch.randn(s, R, D2, device="cuda", generator=g) * 0.05).bfloat16()
    gf = (torch.randn(s, D2, R, device="cuda", generator=g) * 0.2).bfloat16()
    hidden = torch.empty(s, t, R, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(s, t, D2, device="cuda", dtype=torch.float32)
    _launch(ext, pair, down, hidden, s, t, R, D2, out32=False)
    _launch(ext, hidden, gf, out, s, t, D2, R, out32=True)
    h_ref = torch.bmm(pair.double(), down.double().transpose(1, 2))
    # the kernel's hidden is the fp64 value rounded to bf16 up to 1 ulp (fp32 summation order)
    torch.testing.assert_close(hidden.double(), h_ref, rtol=2 ** -7, atol=1e-3)
    ref = torch.bmm(hidden.double(), gf.double().transpose(1, 2))  # stage 2 on the kernel's own hidden
    torch.testing.assert_close(out.double(), ref, rtol=2e-5, atol=2e-4)
    # the old path (torch.bmm, hipBLASLt) differs by the same 1-ulp hidden rounding only
    old = torch.bmm(torch.bmm(pair, down.transpose(1, 2)), gf.transpose(1, 2), out_dtype=torch.float32)
    torch.testing.assert_close(out, old, rtol=1e-2, atol=1e-2)


def test_streams_do_not_leak_into_each_other(ext):
    g = torch.Generator(device="cuda").manual_seed(9)
    x, w = _ints(g, 3, 8, D2), _ints(g, 3, R, D2)
    x[0], w[2] = float("nan"), float("nan")
    y = torch.zeros(3, 8, R, device="cuda", dtype=torch.float32)
    _launch(ext, x, w, y, 3, 8, R, D2, out32=True)
    ref = torch.bmm(x[1:2].double(), w[1:2].double().transpose(1, 2)).float()
    assert torch.equal(y[1:2], ref) and torch.isnan(y[0]).all() and torch.isnan(y[2]).all()


def test_nothing_is_written_past_the_batch_span(ext):
    g = torch.Generator(device="cuda").manual_seed(4)
    nb, m, n, k = 3, 8, 4608, 64
    x, w = _ints(g, nb, m, k), _ints(g, nb, n, k)
    buf = torch.full((nb * m * n + 4096,), 12345.0, device="cuda", dtype=torch.float32)
    y = buf[:nb * m * n].view(nb, m, n)
    _launch(ext, x, w, y, nb, m, n, k, out32=True)
    assert torch.equal(buf[nb * m * n:], torch.full_like(buf[nb * m * n:], 12345.0))


def test_batch_of_one_equals_plain_gemv_bf16(ext):
    g = torch.Generator(device="cuda").manual_seed(2)
    x, w = _ints(g, 1, 8, 2304), _ints(g, 1, 2048, 2304)
    a = torch.empty(1, 8, 2048, device="cuda", dtype=torch.bfloat16)
    b = torch.empty(8, 2048, device="cuda", dtype=torch.bfloat16)
    _launch(ext, x, w, a, 1, 8, 2048, 2304, out32=False)
    ext.gemv_bf16(x.data_ptr(), w.data_ptr(), 0, b.data_ptr(), 8, 2048, 2304, 2304, 2304, 2048, 0,
                  int(torch.cuda.current_stream().cuda_stream), -1)
    torch.cuda.synchronize()
    assert torch.equal(a[0], b)


def test_graph_capture(ext):
    g0 = torch.Generator(device="cuda").manual_seed(3)
    x, w = _ints(g0, 3, 8, D2), _ints(g0, 3, R, D2)
    y = torch.empty(3, 8, R, device="cuda", dtype=torch.bfloat16)
    gs = torch.cuda.Stream()

    def launch():
        ext.gemv_bf16_batched(x.data_ptr(), w.data_ptr(), y.data_ptr(), 3, 8, R, D2, D2, D2, R,
                              8 * D2, R * D2, 8 * R, 0, int(gs.cuda_stream), -1)

    with torch.cuda.stream(gs):
        launch()
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=gs):
        launch()
    x.copy_(_ints(g0, 3, 8, D2))
    gr.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, torch.bmm(x.double(), w.double().transpose(1, 2)).to(torch.bfloat16))


@pytest.mark.parametrize("bad", ["nb0", "nb65", "m17", "bx%8", "bw%8", "negative", "sy<n", "walign", "cfg"])
def test_guards_throw_instead_of_faulting(ext, bad):
    x = torch.zeros(2, 4, 4608, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(2, 64, 4608, device="cuda", dtype=torch.bfloat16)
    y = torch.zeros(2, 4, 64, device="cuda", dtype=torch.bfloat16)
    a = dict(x=x.data_ptr(), w=w.data_ptr(), nb=2, m=4, sy=64, bx=4 * 4608, bw=64 * 4608, by=256, cfg=-1)
    a.update({"nb0": dict(nb=0), "nb65": dict(nb=65), "m17": dict(m=17), "bx%8": dict(bx=4 * 4608 + 1),
              "bw%8": dict(bw=64 * 4608 + 4), "negative": dict(by=-8), "sy<n": dict(sy=32),
              "walign": dict(w=w.data_ptr() + 2), "cfg": dict(cfg=3)}[bad])
    with pytest.raises((ValueError, TypeError)):
        ext.gemv_bf16_batched(a["x"], a["w"], y.data_ptr(), a["nb"], a["m"], 64, 4608, 4608, 4608, a["sy"],
                              a["bx"], a["bw"], a["by"], 0, int(torch.cuda.current_stream().cuda_stream),
                              a["cfg"])
