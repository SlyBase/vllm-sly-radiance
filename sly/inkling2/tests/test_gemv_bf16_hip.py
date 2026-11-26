"""radiance_inkling2_gemv.hip `gemv_bf16` + the default bf16 route of radiance_inkling2_gemv_hip.py.

Lossless low-M GEMV on the model's own bf16 weights: y[M,N] = x[M,K] bf16 @ W[N,K]^T bf16 (+ bias),
fp32 accumulation, bf16 or fp32 output. Replaces vLLM's Triton `_gemv_kernel` (which pads M to 16 rows)
on ROCm for M 1..16, default on, RADIANCE_INKLING2_HIP_GEMV=0 = Triton.

CPU part: routing decisions with a fake extension, the knob, fallbacks, interplay with the opt-in
int8 path, mutation checks of the dispatcher and the anchors of the kernel mutants.
GPU part (skips without a GPU or the compiled module): the kernel against independent references:
  * exact: small integers (bf16-exact, fp32 sums exact in any order) compared bit for bit, every M
    1..16 x every launch config, odd N, K tails, weight / x / y row strides with NaN-poisoned
    padding and sentinel-guarded output padding, bias, fp32 and bf16 (round to nearest even) output;
  * numeric: random floats vs an fp64 reference with a sum|terms| error bound on the real shapes;
  * NaN propagation, graph capture.
RADIANCE_GEMV_BUILD=<dir> asserts which build (RED stub / mutant runs) the process imported.
"""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import radiance_inkling2_gemv_hip as disp  # noqa: E402
import test_gemv_hip as base  # noqa: E402  (shared extension loader and real shape list)

BF16_NCFG = 3  # launch configs (T row tiles per wave, W K-splitting waves per block)
SRC_PATH = HERE.parent / "radiance_inkling2_gemv_hip.py"


@pytest.fixture()
def ext():
    e = base._gpu_ext()
    if not hasattr(e, "gemv_bf16"):
        pytest.fail("build has no gemv_bf16")
    return e


# ---------------------------------------------------------------- GPU helpers


def _ints(g, *shape, hi=3):
    """Small integers: exact in bf16, every partial sum exact in fp32 -> order independent."""
    return torch.randint(-hi, hi + 1, shape, device="cuda", generator=g).to(torch.bfloat16)


def _call(ext, x, w, b, y, *, cfg=-1):
    m, k = x.shape
    ext.gemv_bf16(x.data_ptr(), w.data_ptr(), b.data_ptr() if b is not None else 0, y.data_ptr(),
                  m, w.shape[0], k, x.stride(0), w.stride(0), y.stride(0),
                  int(y.dtype == torch.float32), int(torch.cuda.current_stream().cuda_stream),
                  cfg)
    torch.cuda.synchronize()


def _guarded(shape_m, n, dtype, pad=64):
    """Output view [m, n] of a buffer with `pad` extra columns and one extra row, all 7.0."""
    buf = torch.full((shape_m + 1, n + pad), 7.0, device="cuda", dtype=dtype)
    return buf, buf[:shape_m, :n]


def _untouched(buf, y_view, n, pad=64):
    assert (buf[:, n:] == 7.0).all(), "wrote outside the N columns"
    assert (buf[y_view.shape[0]:] == 7.0).all(), "wrote outside the M rows"


def _run_exact(ext, m, n, k, *, sx=0, sw=0, out32=True, bias=False, cfg=-1, seed=0):
    """Integer data, strided x / W (NaN in the padding), guarded y: result must match bit for bit."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    xb = torch.full((m, max(sx, k)), float("nan"), device="cuda", dtype=torch.bfloat16)
    xb[:, :k] = _ints(g, m, k)
    x = xb[:, :k]
    wb = torch.full((n + 8, max(sw, k)), float("nan"), device="cuda", dtype=torch.bfloat16)
    wb[:n, :k] = _ints(g, n, k)
    w = wb[:n, :k]
    b = _ints(g, n) if bias else None
    dt = torch.float32 if out32 else torch.bfloat16
    buf, y = _guarded(m, n, dt)
    _call(ext, x, w, b, y, cfg=cfg)
    ref = x.double() @ w.double().t()
    if b is not None:
        ref = ref + b.double()
    ref = ref.float().to(dt)
    assert torch.equal(y, ref), f"max |diff| {(y.float() - ref.float()).abs().max().item()}"
    _untouched(buf, y, n)


def _ref_bound(x, w, bias, rows=16384):
    """fp64 reference and sum|terms| error bound, chunked over N (the unembed is 3.7 GB in fp64)."""
    refs, bounds = [], []
    for i in range(0, w.shape[0], rows):
        wc = w[i:i + rows].double()
        r, bd = x.double() @ wc.t(), x.abs().double() @ wc.abs().t()
        if bias is not None:
            r, bd = r + bias[i:i + rows].double(), bd + bias[i:i + rows].abs().double()
        refs.append(r)
        bounds.append(bd)
    return torch.cat(refs, 1), torch.cat(bounds, 1)


def _run_float(ext, m, n, k, *, out32=False, bias=False, cfg=-1, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) / k**0.5
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g) if bias else None
    dt = torch.float32 if out32 else torch.bfloat16
    y = torch.empty(m, n, device="cuda", dtype=dt)
    _call(ext, x, w, b, y, cfg=cfg)
    ref, bound = _ref_bound(x, w, b)
    err = (y.double() - ref).abs()
    # fp32 accumulation error <= ~K * 2^-24 of sum|terms| (1e-5 is 10x that bound and still 100x
    # below the error of any indexing bug); bf16 output adds half an ulp of the result.
    tol = 1e-5 * bound + (0 if out32 else 2.0**-8 * ref.abs())
    assert bool((err <= tol).all()), f"worst err/tol {(err / tol.clamp_min(1e-30)).max().item():.3g}"


# ---------------------------------------------------------------- GPU: kernel


def test_loaded_build_is_the_requested_one(ext):
    want = os.environ.get("RADIANCE_GEMV_BUILD")
    if want:
        assert os.path.realpath(os.path.dirname(ext.__file__)) == os.path.realpath(want)


def test_config_table(ext):
    assert ext.BF16_NCFG == BF16_NCFG and len(ext.BF16_CFGS) == BF16_NCFG
    for t, w in ext.BF16_CFGS:
        assert t == 1 and w in (2, 4, 8)


@pytest.mark.parametrize("cfg", range(BF16_NCFG))
@pytest.mark.parametrize("m", range(1, 17))
def test_every_m_every_config_exact(ext, cfg, m):
    _run_exact(ext, m, 2000, 2312, cfg=cfg)  # K tail of 8 past 2304 (9 x 256)


@pytest.mark.parametrize("m", range(1, 17))
def test_every_m_auto_config_exact(ext, m):
    _run_exact(ext, m, 2304, 3072)
    _run_exact(ext, m, 8600, 2304, out32=False)


@pytest.mark.parametrize("m", [1, 8, 16])
@pytest.mark.parametrize("n,k", base.REAL_SHAPES)
def test_real_shapes_numeric(ext, m, n, k):
    _run_float(ext, m, n, k, out32=(n == 201024), bias=(n == 768))


@pytest.mark.parametrize("m", [1, 3, 16])
@pytest.mark.parametrize("n", [1, 5, 23, 63, 1023, 8601])
def test_odd_n_exact(ext, m, n):
    _run_exact(ext, m, n, 2304, out32=False)


@pytest.mark.parametrize("cfg", range(BF16_NCFG))
@pytest.mark.parametrize("n", [1, 7, 33])  # fewer rows than one block, rows past N never stored
def test_tiny_n_every_config(ext, cfg, n):
    _run_exact(ext, 8, n, 2304, cfg=cfg)


@pytest.mark.parametrize("m", [1, 8, 16])
@pytest.mark.parametrize("k", [8, 16, 264, 2312, 2560, 4104, 9224])  # %8 == 0, not a multiple of 256
def test_k_tail_exact(ext, m, k):
    _run_exact(ext, m, 1030, k, sw=k + 64)


@pytest.mark.parametrize("m", [1, 5, 16])
def test_x_weight_and_y_strides_exact(ext, m):
    _run_exact(ext, m, 2304, 2304, sx=4096, sw=2304 + 128)
    _run_exact(ext, m, 1001, 3072, sx=3072 + 8, sw=4096, out32=False)


@pytest.mark.parametrize("m", [1, 7, 16])
def test_bias_and_output_dtypes_exact(ext, m):
    for out32 in (True, False):
        _run_exact(ext, m, 1024, 2304, bias=True, out32=out32)
        _run_exact(ext, m, 4097, 2312, bias=True, out32=out32, sx=2560)


def test_bf16_output_is_round_to_nearest_even(ext):
    """x . w = 257 (tie between 256 and 258 in bf16) and 259 (tie, other direction): RNE -> 256, 260."""
    k = 2304
    x = torch.zeros(1, k, device="cuda", dtype=torch.bfloat16)
    x[0, :2] = 1
    w = torch.zeros(4, k, device="cuda", dtype=torch.bfloat16)
    w[0, 0], w[0, 1] = 128, 129  # 257
    w[1, 0], w[1, 1] = 130, 129  # 259
    w[2, 0], w[2, 1] = -128, -129  # -257
    w[3, 0], w[3, 1] = 128, 127  # 255 (representable)
    y = torch.empty(1, 4, device="cuda", dtype=torch.bfloat16)
    _call(ext, x, w, None, y)
    assert y.float().tolist() == [[256.0, 260.0, -256.0, 255.0]]


def test_all_bf16_weight_values_and_signs_exact(ext):
    """Every finite bf16 pattern of a 256-wide exponent/mantissa sweep reaches the right lane."""
    k, n = 2304, 256
    codes = torch.arange(n, device="cuda")
    w = torch.zeros(n, k, device="cuda", dtype=torch.bfloat16)
    vals = ((codes - 128) / 16.0).to(torch.bfloat16)  # exact in bf16
    w[codes, (codes * 9) % k] = vals  # every position inside a 16 B lane quantum
    x = torch.ones(1, k, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(1, n, device="cuda", dtype=torch.float32)
    _call(ext, x, w, None, y)
    assert torch.equal(y[0], vals.float())


@pytest.mark.parametrize("m", [1, 8, 16])
def test_nan_and_inf_propagate_per_row(ext, m):
    n, k = 1024, 2304
    x = torch.ones(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.ones(n, k, device="cuda", dtype=torch.bfloat16)
    x[m - 1, 1000] = float("nan")
    x[0, 5] = float("inf")
    for dt in (torch.float32, torch.bfloat16):
        y = torch.empty(m, n, device="cuda", dtype=dt)
        _call(ext, x, w, None, y)
        assert torch.isnan(y[m - 1]).all()
        if m > 1:
            assert torch.isinf(y[0]).all() and torch.isfinite(y[1:m - 1]).all()


def test_graph_capture(ext):
    m, n, k = 8, 2304, 3072
    g0 = torch.Generator(device="cuda").manual_seed(3)
    x, w = _ints(g0, m, k), _ints(g0, n, k)
    x2, w2 = _ints(g0, 3, 2304), _ints(g0, 777, 2304)
    y1 = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    y2 = torch.empty(3, 777, device="cuda", dtype=torch.float32)
    gs = torch.cuda.Stream()  # capture + launch on a dedicated non-default stream

    def launch(sptr):
        ext.gemv_bf16(x.data_ptr(), w.data_ptr(), 0, y1.data_ptr(), m, n, k, k, k, n, 0, sptr, -1)
        ext.gemv_bf16(x2.data_ptr(), w2.data_ptr(), 0, y2.data_ptr(), 3, 777, 2304, 2304, 2304, 777, 1,
                      sptr, -1)

    for _ in range(3):
        launch(int(gs.cuda_stream))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=gs):
        launch(int(gs.cuda_stream))
    x.copy_(_ints(g0, m, k))
    x2.copy_(_ints(g0, 3, 2304))
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y1, (x.double() @ w.double().t()).float().to(torch.bfloat16))
    assert torch.equal(y2, (x2.double() @ w2.double().t()).float())


@pytest.mark.parametrize("bad", ["m0", "m17", "k%8", "sw%8", "sx%8", "xalign", "walign", "cfg"])
def test_guards_throw_instead_of_faulting(ext, bad):
    k = 2312 if bad != "k%8" else 2308
    x = torch.zeros(2, 2336, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(64, 2336, device="cuda", dtype=torch.bfloat16)
    y = torch.zeros(2, 64, device="cuda", dtype=torch.bfloat16)
    a = dict(x=x.data_ptr(), w=w.data_ptr(), b=0, y=y.data_ptr(), m=2, n=64, k=k, sx=2336, sw=2336,
             sy=64, o32=0, cfg=-1)
    if bad == "m0":
        a["m"] = 0
    elif bad == "m17":
        a["m"] = 17
    elif bad == "sw%8":
        a["sw"] = 2335
    elif bad == "sx%8":
        a["sx"] = 2332
    elif bad == "xalign":
        a["x"] += 2
    elif bad == "walign":
        a["w"] += 2
    elif bad == "cfg":
        a["cfg"] = BF16_NCFG
    with pytest.raises((ValueError, TypeError)):
        ext.gemv_bf16(a["x"], a["w"], a["b"], a["y"], a["m"], a["n"], a["k"], a["sx"], a["sw"],
                      a["sy"], a["o32"], int(torch.cuda.current_stream().cuda_stream), a["cfg"])


def test_dispatch_end_to_end_default_on(ext, monkeypatch):
    monkeypatch.setattr(disp, "_ext", ext)
    monkeypatch.setattr(disp, "W8", False)
    monkeypatch.setattr(disp, "HIP_BF16", True)
    g = torch.Generator(device="cuda").manual_seed(5)
    x, w = _ints(g, 8, 2304), _ints(g, 2048, 2304)
    out = torch.empty(8, 2048, device="cuda", dtype=torch.bfloat16)
    assert disp.hip_gemv(x, w, None, out) is True
    torch.cuda.synchronize()
    assert torch.equal(out, (x.double() @ w.double().t()).float().to(torch.bfloat16))


# ---------------------------------------------------------------- CPU: dispatcher


class FakeExt:
    NCFG = base.NCFG
    BF16_NCFG = BF16_NCFG

    def __init__(self):
        self.q8, self.bf16 = [], []

    def gemv_q8(self, *args):
        self.q8.append(args)

    def gemv_bf16(self, *args):
        self.bf16.append(args)


@pytest.fixture()
def routed(monkeypatch):
    """Default config: bf16 route on, int8 opt-in off, fake extension, CPU tensors."""
    fake = FakeExt()
    state = {"capturing": False}
    monkeypatch.setattr(disp, "_ext", fake)
    monkeypatch.setattr(disp, "W8", False)
    monkeypatch.setattr(disp, "HIP_BF16", True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: type("S", (), {"cuda_stream": 7})())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state["capturing"])
    disp._Q8_CACHE.clear()
    yield fake, state
    disp._Q8_CACHE.clear()


def _t(m, n, k=2304, dtype=torch.bfloat16):
    return torch.randn(m, k, dtype=dtype), torch.randn(n, k, dtype=dtype), torch.empty(m, n, dtype=dtype)


def test_bf16_route_is_default_on():
    src = SRC_PATH.read_text()
    assert 'os.environ.get("RADIANCE_INKLING2_HIP_GEMV", "1")' in src
    assert disp.W8 is False  # the int8 opt-in stays opt-in


@pytest.mark.parametrize("val,want", [(None, True), ("1", True), ("0", False)])
def test_knob_reads_the_environment(monkeypatch, val, want):
    if val is None:
        monkeypatch.delenv("RADIANCE_INKLING2_HIP_GEMV", raising=False)
    else:
        monkeypatch.setenv("RADIANCE_INKLING2_HIP_GEMV", val)
    spec = importlib.util.spec_from_file_location("disp_env_copy", SRC_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.HIP_BF16 is want


def test_routes_bf16_weight_with_strides_and_pointers(routed):
    fake, _ = routed
    x, w, out = _t(8, 4096)
    assert disp.hip_gemv(x, w, None, out) is True
    (call,) = fake.bf16
    assert call[:4] == (x.data_ptr(), w.data_ptr(), 0, out.data_ptr())
    assert call[4:11] == (8, 4096, 2304, 2304, 2304, 4096, 0)  # M N K SX SW SY out32
    assert call[11] == 7 and call[12] == -1  # stream, cfg auto
    assert fake.q8 == [] and disp._Q8_CACHE == {}  # the weight is read as it is: no copy, no scratch


@pytest.mark.parametrize("m", range(1, 17))
def test_every_m_1_to_16_is_routed(routed, m):
    fake, _ = routed
    x, w, out = _t(m, 4096)
    assert disp.hip_gemv(x, w, None, out) is True
    assert len(fake.bf16) == 1 and fake.bf16[0][4] == m


def test_m_above_the_kernel_range_stays_on_triton(routed):
    fake, _ = routed
    x, w, out = _t(17, 4096)
    assert disp.hip_gemv(x, w, None, out) is False and fake.bf16 == []


def test_fp32_out_bias_and_strided_views_are_passed(routed):
    fake, _ = routed
    x, w, _out = _t(4, 4096)
    bias = torch.randn(4096, dtype=torch.bfloat16)
    out = torch.empty(4, 4096, dtype=torch.float32)
    assert disp.hip_gemv(x, w, bias, out) is True
    a = fake.bf16[0]
    assert a[2] == bias.data_ptr() and a[10] == 1
    big = torch.randn(512, 2560, dtype=torch.bfloat16)
    wv = big[:, :2304]  # column slice: row stride 2560 (a prefix of a wider tensor)
    xv = torch.randn(4, 4096, dtype=torch.bfloat16)[:, :2304]
    o = torch.empty(4, 512, dtype=torch.bfloat16)
    assert disp.hip_gemv(xv, wv, None, o) is True
    b = fake.bf16[1]
    assert (b[3], b[7], b[8], b[9]) == (o.data_ptr(), 4096, 2560, 512)  # y ptr, SX, SW, SY


def test_row_prefix_slice_of_a_fused_weight_is_routed(routed):
    fake, _ = routed
    x, w, _ = _t(8, 8600)
    out = torch.empty(8, 6552, dtype=torch.bfloat16)
    assert disp.hip_gemv(x, w[:6552], None, out) is True  # global layers: q-side of the fused qkvr
    assert fake.bf16[0][5] == 6552 and fake.bf16[0][1] == w.data_ptr()


@pytest.mark.parametrize("n", [1, 128, 768, 1024])
def test_small_and_odd_row_counts_are_routed(routed, n):  # router 128, rnn_proj 768, k/v 1024: all faster than Triton
    fake, _ = routed
    x, w, out = _t(8, n)
    assert disp.hip_gemv(x, w, None, out) is True and len(fake.bf16) == 1


@pytest.mark.parametrize("what", ["k%8", "fp32_x", "fp16_w", "xstride", "xalign", "walign", "wstride",
                                  "w_transposed", "out_dtype", "out_col_stride", "out_shape",
                                  "bias_dtype", "bias_stride", "k_mismatch"])
def test_unsupported_inputs_fall_back(routed, what):
    fake, _ = routed
    x, w, out = _t(8, 4096)
    bias = None
    if what == "k%8":
        x, w = x[:, :2300], w[:, :2300]  # K = 2300 (% 8 == 4, % 4 == 0) with row strides 2304: only K says no
    elif what == "fp32_x":
        x = x.float()
    elif what == "fp16_w":
        w = w.half()
    elif what == "xstride":
        x = torch.randn(8, 2304 + 4, dtype=torch.bfloat16)[:, :2304]  # row stride % 8 != 0
    elif what == "xalign":
        x = torch.randn(8 * 2304 + 8, dtype=torch.bfloat16)[1:1 + 8 * 2304].view(8, 2304)  # +2 B
    elif what == "walign":
        w = torch.randn(4096 * 2304 + 8, dtype=torch.bfloat16)[1:1 + 4096 * 2304].view(4096, 2304)
    elif what == "wstride":
        w = torch.randn(4096, 2304 + 4, dtype=torch.bfloat16)[:, :2304]  # row stride % 8 != 0
    elif what == "w_transposed":
        w = torch.randn(2304, 4096, dtype=torch.bfloat16).t()  # stride(-1) != 1
    elif what == "out_dtype":
        out = out.half()
    elif what == "out_col_stride":
        out = torch.empty(4096, 8, dtype=torch.bfloat16).t()
    elif what == "out_shape":
        out = torch.empty(8, 4095, dtype=torch.bfloat16)
    elif what == "bias_dtype":
        bias = torch.randn(4096)
    elif what == "bias_stride":
        bias = torch.randn(8192, dtype=torch.bfloat16)[::2]
    elif what == "k_mismatch":
        w = w[:, :2296]
    assert disp.hip_gemv(x, w, bias, out) is False
    assert fake.bf16 == [] and fake.q8 == []


def test_knob_off_means_triton(routed, monkeypatch):
    fake, _ = routed
    monkeypatch.setattr(disp, "HIP_BF16", False)
    x, w, out = _t(8, 4096)
    assert disp.hip_gemv(x, w, None, out) is False and fake.bf16 == []


def test_no_extension_means_triton(routed, monkeypatch):
    monkeypatch.setattr(disp, "_ext", None)
    x, w, out = _t(8, 4096)
    assert disp.hip_gemv(x, w, None, out) is False


def test_graph_capture_routes_without_any_conversion(routed):
    fake, state = routed
    state["capturing"] = True
    x, w, out = _t(8, 4096)
    assert disp.hip_gemv(x, w, None, out) is True  # no scratch, no copy: capture safe by design
    assert len(fake.bf16) == 1 and disp._Q8_CACHE == {}


def test_int8_opt_in_takes_every_m_and_bf16_takes_the_rest(routed, monkeypatch):
    fake, state = routed
    monkeypatch.setattr(disp, "W8", True)
    x, w, out = _t(2, 4096)
    assert disp.hip_gemv(x, w, None, out) is True
    assert len(fake.q8) == 1 and fake.bf16 == []  # W8: the int8 copy
    x, w2, out = _t(8, 4096)
    assert disp.hip_gemv(x, w2, None, out) is True
    assert len(fake.q8) == 2 and fake.bf16 == []  # M 5..16 (a spec-verify step) too
    x, w2, out = _t(2, 128)
    assert disp.hip_gemv(x, w2, None, out) is True
    assert len(fake.q8) == 2 and len(fake.bf16) == 1  # a tensor too small for int8: bf16 HIP, not Triton
    assert (w2.data_ptr(), 2304) not in disp._Q8_CACHE  # and no conversion for a call it would not serve
    x, w3, out = _t(2, 4096)
    state["capturing"] = True
    assert disp.hip_gemv(x, w3, None, out) is True  # unseen weight under capture: bf16 HIP, not Triton
    assert len(fake.bf16) == 2 and (w3.data_ptr(), 2304) not in disp._Q8_CACHE


# ---------------------------------------------------------------- CPU: mutation checks
# Each mutation goes into a COPY in tmp_path; the real file is never touched, and after the loop
# we assert it is byte-identical (restore contract).


def _mutated(tmp_path, old, new):
    src = SRC_PATH.read_text()
    assert src.count(old) == 1, f"mutation anchor not unique: {old!r}"
    p = tmp_path / "mutant.py"
    p.write_text(src.replace(old, new, 1))
    spec = importlib.util.spec_from_file_location("mutant_bf16_disp", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _arm(mod, monkeypatch, capturing=False):
    fake = FakeExt()
    monkeypatch.setattr(mod, "_ext", fake)
    monkeypatch.setattr(mod, "W8", False)
    monkeypatch.setattr(mod, "HIP_BF16", True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: type("S", (), {"cuda_stream": 7})())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    mod._Q8_CACHE.clear()
    return fake


def test_dispatcher_mutations_are_caught(tmp_path, monkeypatch):
    real = SRC_PATH.read_bytes()

    def refuses(mod, mutate=lambda x, w, o: (x, w, o), **kw):
        _arm(mod, monkeypatch, **kw)
        x, w, o = mutate(*_t(8, 4096))
        return mod.hip_gemv(x, w, None, o) is False

    def m_range(mod):
        _arm(mod, monkeypatch)
        x, w, o = _t(17, 4096)
        assert mod.hip_gemv(x, w, None, o) is False, "M 17 routed"

    def m_zero_rows(mod):
        _arm(mod, monkeypatch)
        x, w, o = torch.empty(0, 2304, dtype=torch.bfloat16), torch.randn(4096, 2304, dtype=torch.bfloat16), \
            torch.empty(0, 4096, dtype=torch.bfloat16)
        assert mod.hip_gemv(x, w, None, o) is False, "M 0 routed"

    def k8(mod):
        assert refuses(mod, lambda x, w, o: (x[:, :2300], w[:, :2300], o)), "K%8"

    def xalign(mod):
        assert refuses(mod, lambda x, w, o: (torch.randn(8 * 2304 + 8, dtype=torch.bfloat16)[1:1 + 8 * 2304]
                                             .view(8, 2304), w, o)), "x misaligned routed"

    def walign(mod):
        assert refuses(mod, lambda x, w, o: (x, torch.randn(4096 * 2304 + 8, dtype=torch.bfloat16)
                                             [1:1 + 4096 * 2304].view(4096, 2304), o)), "w misaligned routed"

    def wstride(mod):
        assert refuses(mod, lambda x, w, o: (x, torch.randn(4096, 2308, dtype=torch.bfloat16)[:, :2304], o)), \
            "W row stride % 8 routed"

    def wdtype(mod):
        assert refuses(mod, lambda x, w, o: (x, w.float(), o)), "fp32 weight routed"

    def odtype(mod):
        assert refuses(mod, lambda x, w, o: (x, w, o.half())), "fp16 out routed"

    def oshape(mod):
        assert refuses(mod, lambda x, w, o: (x, w, torch.empty(8, 4095, dtype=torch.bfloat16))), "out shape"

    def knob(mod):
        _arm(mod, monkeypatch)
        monkeypatch.setattr(mod, "HIP_BF16", False)
        x, w, o = _t(8, 4096)
        assert mod.hip_gemv(x, w, None, o) is False, "knob ignored"

    def capture(mod):
        fake = _arm(mod, monkeypatch, capturing=True)
        x, w, o = _t(8, 4096)
        assert mod.hip_gemv(x, w, None, o) is True and len(fake.bf16) == 1, "capture blocks the route"

    def args(mod):
        fake = _arm(mod, monkeypatch)
        x = torch.randn(8, 2560, dtype=torch.bfloat16)[:, :2304]
        w = torch.randn(4096, 2312, dtype=torch.bfloat16)[:, :2304]
        o = torch.empty(8, 4104, dtype=torch.bfloat16)[:, :4096]
        assert mod.hip_gemv(x, w, None, o) is True
        a = fake.bf16[0]
        assert a[7:10] == (2560, 2312, 4104), "strides"
        assert a[4:7] == (8, 4096, 2304), "M N K"
        assert a[11] == 7, "stream"
        assert a[10] == 0, "out32 flag"

    def out32(mod):
        fake = _arm(mod, monkeypatch)
        x, w, _ = _t(8, 4096)
        mod.hip_gemv(x, w, None, torch.empty(8, 4096))
        assert fake.bf16[0][10] == 1, "fp32 out flag"

    def bias(mod):
        fake = _arm(mod, monkeypatch)
        x, w, o = _t(8, 4096)
        b = torch.randn(4096, dtype=torch.bfloat16)
        mod.hip_gemv(x, w, b, o)
        assert fake.bf16[0][2] == b.data_ptr(), "bias pointer"

    def q8_needs_opt_in(mod):
        fake = _arm(mod, monkeypatch)  # W8 off
        x, w, o = _t(2, 4096)
        mod.hip_gemv(x, w, None, o)
        assert fake.q8 == [] and len(fake.bf16) == 1, "int8 path used without RADIANCE_INKLING2_W8"

    mutations = [
        ("0 < m <= MAX_M", "m <= 99", m_range),
        ("0 < m <= MAX_M", "m <= MAX_M", m_zero_rows),
        ("k % 8 == 0", "k % 4 == 0", k8),
        ("_aligned(x2)", "True", xalign),
        ("_aligned(weight)", "True", walign),
        ("weight.stride(0) % 8 == 0", "True", wstride),
        ("x2.dtype == weight.dtype == torch.bfloat16", "x2.dtype == torch.bfloat16", wdtype),
        ("out.dtype in (torch.bfloat16, torch.float32)\n        and tuple", "True\n        and tuple", odtype),
        ("tuple(out.shape) == (m, n)", "True", oshape),
        ("return HIP_BF16 and _hip_bf16(", "return _hip_bf16(", knob),
        ("m, n, k, x2.stride(0), weight.stride(0), out.stride(0),", "m, n, k, k, k, n,", args),
        ("int(out.dtype == torch.float32), torch.cuda", "0, torch.cuda", out32),
        ("weight.data_ptr(), bias.data_ptr() if bias is not None else 0,", "weight.data_ptr(), 0,", bias),
        ("if _want_q8(weight) and _hip_q8(", "if _hip_q8(", q8_needs_opt_in),
    ]
    caught = 0
    for old, new, check in mutations:
        mod = _mutated(tmp_path, old, new)
        try:
            check(mod)
        except (AssertionError, RuntimeError, TypeError):
            caught += 1
        else:
            raise AssertionError(f"mutation survived: {old!r} -> {new!r}")
    assert caught == len(mutations)
    assert SRC_PATH.read_bytes() == real  # restored


def test_kernel_mutants_apply_to_the_real_source(tmp_path):
    """The GPU-window mutants of the bf16 kernel come from exact, unique anchors."""
    import mutate_gemv_hip as mg

    path = HERE.parent / "radiance_inkling2_gemv.hip"
    src = path.read_text()
    mg.write_mutants(path, tmp_path)
    bf = {k: v for k, v in mg.MUTATIONS.items() if k.startswith("bf_")}
    assert len(bf) >= 14
    for name, (old, new) in bf.items():
        assert src.count(old) == 1, name
        assert old != new
        assert (tmp_path / f"m_{name}.hip").read_text() == src.replace(old, new, 1)


def test_kernel_source_has_the_bf16_entry_point():
    src = (HERE.parent / "radiance_inkling2_gemv.hip").read_text()
    assert 'm.def("gemv_bf16"' in src and 'm.def("gemv_q8"' in src  # one .so, both entries
    assert "__builtin_nontemporal_load" in src  # weights are read once: stream them
    assert "__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12" in src  # all M rows per weight load on the matrix unit
