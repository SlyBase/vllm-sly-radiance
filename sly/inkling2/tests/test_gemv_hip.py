"""sly/inkling2/radiance_inkling2_gemv.hip + radiance_inkling2_gemv_hip.py: int8 (Q8_0) low-M GEMV.

Weight format of the opt-in RADIANCE_INKLING2_W8 path: int8 with one fp32 scale per 32 consecutive
K values (symmetric, scale = amax / 127, round to nearest) = ggml Q8_0 with an fp32 scale,
1.125 B/weight; activations stay bf16 -> fp32. The kernel reads a TILED copy (pack_q8: 16 weight rows
x 64 K per tile group, laid out in the order one wave loads it, see the .hip header); the plain
[N, K] int8 + [N, K/32] scale pair (to_q8) is the reference format of these tests.

CPU part: Q8_0 conversion math, tile layout against an independent emulation of the kernel's
loads, routing decisions (fake extension), fallback without the extension, the gemv.py patch hook
(against a copy of the real tree) and mutation checks of the dispatcher and of the kernel source.
GPU part: int8 kernel vs an independent torch fp32 reference (and bit-exact on integer data), every
M 1..16, every launch config, real shapes, odd N, tiny N, K tails, x strides, bias, fp32 out, all
256 int8 codes, per-block scales, graph capture. Skips cleanly without a GPU or the compiled module
(CI is CPU only).

RADIANCE_GEMV_BUILD=<dir> asserts which build (RED stub / mutant runs) the process imported.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
# Walk up to find the repo root (holds sly/patch_inkling2_gfx12.py):
# local: sly/inkling2/tests -> repo root; GPU window: /root/ink-gemv/tests -> no repo (CPU tests skip).
REPO = None
for _p in [HERE, *HERE.parents]:
    if (_p / "sly/patch_inkling2_gfx12.py").exists():
        REPO = _p
        break
if REPO is None:
    REPO = HERE.parent / "repo"
sys.path.insert(0, str(HERE.parent))  # sly/inkling2: the wrapper under test
sys.path.insert(0, str(HERE))  # the kernel mutant generator

import radiance_inkling2_gemv_hip as disp  # noqa: E402

BLK = 32  # K values per scale
# Real Inkling-2-Mini decode shapes (N, K): fused local qkvr, global q-side, k/v, wo, unembed,
# DSpark qkv / kv_ctx / gate_up / down, markov rnn_proj (bias, fp32 out).
REAL_SHAPES = [
    (8600, 2304), (6552, 2304), (1024, 2304), (2304, 3072), (201024, 2304),
    (5120, 2304), (2048, 2304), (18432, 2304), (2304, 9216), (2304, 256), (768, 5120),
]
NCFG = 12  # launch configs of the kernel (T row tiles per wave, W K-splitting waves, D groups in flight)


def _load_ext():
    # The GPU window puts ONE build (real, stub or a mutant) first on PYTHONPATH, each as
    # <dir>/radiance_inkling2_gemv.so, and names its dir in RADIANCE_GEMV_BUILD. (Loading a second
    # .so by path under the same name in one process silently returns the first: single-phase
    # extension modules are cached by name.)
    try:
        import radiance_inkling2_gemv as m
        return m
    except ImportError:
        return None


def _gpu_ext():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    ext = _load_ext()
    if ext is None:
        pytest.skip("radiance_inkling2_gemv module not built")
    return ext


@pytest.fixture()
def ext():
    return _gpu_ext()


def _deq(wq, scale):
    """Independent dequantisation: int8 * the fp32 scale of its 32-value block."""
    return wq.float() * scale.repeat_interleave(BLK, dim=1)


def _ref(x, deq, bias, out_dtype):
    r = x.float() @ deq.t()
    if bias is not None:
        r = r + bias.float()
    return r.to(out_dtype)


def _check(y, ref, out_dtype):
    if out_dtype == torch.float32:
        torch.testing.assert_close(y.float(), ref.float(), rtol=2e-5, atol=2e-4)
    else:  # bf16 output rounding: 1 ulp (2**-8) margin over the fp32 accumulation order
        torch.testing.assert_close(y.float(), ref.float(), rtol=2**-7, atol=1e-3)


def _call(ext, x, wq, scale, b, y, *, cfg=-1):
    m, k = x.shape
    n = wq.shape[0]
    qt, st = disp.pack_q8(wq, scale)
    stream = torch.cuda.current_stream().cuda_stream
    ext.gemv_q8(x.data_ptr(), qt.data_ptr(), st.data_ptr(),
                b.data_ptr() if b is not None else 0, y.data_ptr(),
                m, n, k, x.stride(0), y.stride(0), int(y.dtype == torch.float32), qt.shape[0],
                int(stream), cfg)
    torch.cuda.synchronize()


def _run(ext, m, n, k, *, bias=False, out32=False, sx=None, cfg=-1, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, sx or k, device="cuda", dtype=torch.bfloat16, generator=g)[:, :k]
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) / k**0.5
    wq, scale = disp.to_q8(w)
    b = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=g) if bias else None
    dt = torch.float32 if out32 else torch.bfloat16
    y = torch.empty(m, n, device="cuda", dtype=dt)
    _call(ext, x, wq, scale, b, y, cfg=cfg)
    _check(y, _ref(x, _deq(wq, scale), b, dt), dt)
    return y


# ---------------------------------------------------------------- GPU: kernel


def test_loaded_build_is_the_requested_one(ext):
    want = os.environ.get("RADIANCE_GEMV_BUILD")
    if want:
        assert os.path.realpath(os.path.dirname(ext.__file__)) == os.path.realpath(want)


def test_ncfg_matches(ext):
    assert ext.NCFG == NCFG


@pytest.mark.parametrize("m", list(range(1, 17)))
def test_every_m(ext, m):
    _run(ext, m, 2304, 3072)


@pytest.mark.parametrize("cfg", range(NCFG))
@pytest.mark.parametrize("m", [1, 3, 8, 16])
def test_every_config(ext, cfg, m):
    _run(ext, m, 2000, 2304, cfg=cfg)


@pytest.mark.parametrize("m", [1, 16])
@pytest.mark.parametrize("n,k", REAL_SHAPES)
def test_real_shapes(ext, m, n, k):
    _run(ext, m, n, k, out32=(n == 201024), bias=(n == 768))


@pytest.mark.parametrize("m", [1, 3, 8, 16])
@pytest.mark.parametrize("n", [1, 15, 17, 23, 1023, 8601])
def test_odd_n(ext, m, n):
    _run(ext, m, n, 2304)


@pytest.mark.parametrize("cfg", range(NCFG))
@pytest.mark.parametrize("n", [1, 7, 33])  # fewer rows than one block, rows past N never stored
def test_tiny_n_every_config(ext, cfg, n):
    _run_exact(ext, 8, n, 2304, cfg=cfg)


@pytest.mark.parametrize("m", [1, 16])
@pytest.mark.parametrize("n,k", [(24, 2304), (1, 256), (64, 2304), (100, 9216)])
def test_tiny_n(ext, m, n, k):
    _run(ext, m, n, k)


@pytest.mark.parametrize("m", [1, 16])
@pytest.mark.parametrize("k", [2368, 192, 64, 4160])  # %64 == 0 but an odd number of 64 K groups / not %128
def test_k_tail(ext, m, k):
    _run(ext, m, 1024, k)


def _guarded(m, n, dt, pad=64):
    buf = torch.full((m + 1, n + pad), 7.0, device="cuda", dtype=dt)
    return buf, buf[:m, :n]


def _run_exact(ext, m, n, k, *, out32=True, bias=False, sx=0, cfg=-1, seed=0):
    """Small integer codes, power-of-two block scales, small integer x: every product and partial sum
    is exact in fp32 in ANY order, so the result must equal the double reference bit for bit. x rows
    have a NaN-poisoned stride padding, y a sentinel border (nothing may be written outside [m, n])."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    xb = torch.full((m, max(sx, k)), float("nan"), device="cuda", dtype=torch.bfloat16)
    xb[:, :k] = torch.randint(-3, 4, (m, k), device="cuda", generator=g).to(torch.bfloat16)
    x = xb[:, :k]
    wq = torch.randint(-3, 4, (n, k), device="cuda", generator=g, dtype=torch.int8)
    scale = 2.0 ** torch.randint(-2, 3, (n, k // BLK), device="cuda", generator=g).float()
    b = torch.randint(-3, 4, (n,), device="cuda", generator=g).to(torch.bfloat16) if bias else None
    dt = torch.float32 if out32 else torch.bfloat16
    buf, y = _guarded(m, n, dt)
    _call(ext, x, wq, scale, b, y, cfg=cfg)
    ref = x.double() @ _deq(wq, scale).double().t()
    if b is not None:
        ref = ref + b.double()
    ref = ref.float().to(dt)
    assert torch.equal(y, ref), f"max |diff| {(y.float() - ref.float()).abs().max().item()}"
    assert (buf[:, n:] == 7.0).all() and (buf[m:] == 7.0).all(), "wrote outside [m, n]"


@pytest.mark.parametrize("cfg", range(NCFG))
@pytest.mark.parametrize("m", range(1, 17))
def test_every_m_every_config_exact(ext, cfg, m):
    _run_exact(ext, m, 2000, 2368, cfg=cfg)


@pytest.mark.parametrize("m", range(1, 17))
def test_every_m_auto_config_exact(ext, m):
    _run_exact(ext, m, 2304, 3072)
    _run_exact(ext, m, 8600, 2304, out32=False)


@pytest.mark.parametrize("m", [1, 8, 16])
def test_bias_and_strides_exact(ext, m):
    _run_exact(ext, m, 1001, 2304, bias=True, sx=4096)
    _run_exact(ext, m, 4097, 2368, bias=True, out32=False, sx=2368 + 8)


@pytest.mark.parametrize("m", [1, 5, 16])
def test_x_row_stride(ext, m):
    _run(ext, m, 2304, 2304, sx=4096)


@pytest.mark.parametrize("m", [1, 16])
def test_bias_and_fp32_out(ext, m):
    _run(ext, m, 1024, 2304, bias=True)
    _run(ext, m, 1024, 2304, bias=True, out32=True, sx=2560)
    _run(ext, m, 4096, 2304, bias=True)
    _run(ext, m, 4096, 2304, bias=True, out32=True)


def test_all_int8_codes_exact(ext):
    """Row i holds code i-128 at one column and zero elsewhere, scale 1: y = the code exactly."""
    n, k = 256, 2304
    wq = torch.zeros(n, k, dtype=torch.int8)
    codes = torch.arange(n)
    wq[codes, (codes * 9) % k] = (codes - 128).to(torch.int8)  # every byte position of the 32 B span
    wq = wq.cuda()
    scale = torch.ones(n, k // BLK, device="cuda")
    x = torch.ones(1, k, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(1, n, device="cuda", dtype=torch.float32)
    _call(ext, x, wq, scale, None, y)
    torch.testing.assert_close(y[0], wq.float().sum(dim=1), rtol=0, atol=0)


def test_scale_is_per_32_block_not_per_row(ext):
    n, k = 64, 2304
    g = torch.Generator(device="cuda").manual_seed(1)
    wq = torch.randint(-127, 128, (n, k), device="cuda", dtype=torch.int8, generator=g)
    scale = torch.rand(n, k // BLK, device="cuda", generator=g) * 100 + 0.01  # varies per block
    x = torch.randn(2, k, device="cuda", dtype=torch.bfloat16, generator=g)
    y = torch.empty(2, n, device="cuda", dtype=torch.float32)
    _call(ext, x, wq, scale, None, y)
    deq = _deq(wq, scale)
    ref = x.double() @ deq.double().t()
    bound = x.abs().double() @ deq.abs().double().t()  # fp32 summation error scales with sum|terms|
    assert ((y.double() - ref).abs() <= 1e-5 * bound).all()


def test_auto_config_follows_the_measured_table(ext):
    """Rotated-pool sweep (bench_gemv.py --q8 --cfgs, M 1 / 8 / 16 alike): T=1 W=4 D=2 (cfg 1) for every shape but
    the 6552 / 8600-row qkv (cfg 0, W=2) and the unembed (cfg 6, T=2 W=2)."""
    table = {(1024, 2304): 1, (2048, 2304): 1, (2304, 3072): 1, (2304, 9216): 1, (5120, 2304): 1,
             (6552, 2304): 0, (8600, 2304): 0, (18432, 2304): 1, (201024, 2304): 6}
    for (n, k), cfg in table.items():
        for m in (1, 8, 16):
            assert ext.auto_cfg_q8(m, n, k) == cfg, (m, n, k)


def test_guards_throw_instead_of_faulting(ext):
    m, n, k = 2, 64, 2304
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    qt, st = disp.pack_q8(*disp.to_q8(torch.randn(n, k, device="cuda", dtype=torch.bfloat16)))
    y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream

    def call(**kw):
        a = dict(x=x.data_ptr(), q=qt.data_ptr(), s=st.data_ptr(), bias=0, y=y.data_ptr(), M=m, N=n,
                 K=k, SX=k, SY=n, out32=0, ntiles=qt.shape[0], stream=stream, cfg=-1)
        a.update(kw)
        ext.gemv_q8(**a)

    call()  # the arguments themselves are valid
    for bad in (dict(M=0), dict(M=17), dict(N=0), dict(K=2272), dict(K=32), dict(SX=k - 8), dict(SX=k + 4),
                dict(x=x.data_ptr() + 2), dict(q=qt.data_ptr() + 8), dict(s=st.data_ptr() + 2),
                dict(ntiles=qt.shape[0] - 1), dict(N=n + 1), dict(cfg=NCFG), dict(cfg=-2)):
        with pytest.raises((ValueError, RuntimeError)):
            call(**bad)
    torch.cuda.synchronize()


# ---------------------------------------------------------------- GPU: graph capture


def test_graph_capture(ext):
    m, n, k = 4, 2304, 3072
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k**0.5
    wq, scale = disp.to_q8(w)
    nsm, nsk = 64, 2304  # a tiny shape too
    wq2, scale2 = disp.to_q8(torch.randn(nsm, nsk, device="cuda", dtype=torch.bfloat16))
    qt, st = disp.pack_q8(wq, scale)
    qt2, st2 = disp.pack_q8(wq2, scale2)
    y1 = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    y2 = torch.empty(m, nsm, device="cuda", dtype=torch.float32)
    x2 = torch.randn(m, nsk, device="cuda", dtype=torch.bfloat16)
    gs = torch.cuda.Stream()  # capture + launch on a dedicated non-default stream

    def launch(sptr: int) -> None:
        ext.gemv_q8(x.data_ptr(), qt.data_ptr(), st.data_ptr(), 0, y1.data_ptr(),
                    m, n, k, k, n, 0, qt.shape[0], sptr, -1)
        ext.gemv_q8(x2.data_ptr(), qt2.data_ptr(), st2.data_ptr(), 0, y2.data_ptr(),
                    m, nsm, nsk, nsk, nsm, 1, qt2.shape[0], sptr, -1)

    for _ in range(3):  # eager warmup on the capture stream
        launch(int(gs.cuda_stream))
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=gs):
        launch(int(gs.cuda_stream))
    x.copy_(torch.randn_like(x))
    x2.copy_(torch.randn_like(x2))
    g.replay()
    torch.cuda.synchronize()
    _check(y1, _ref(x, _deq(wq, scale), None, torch.bfloat16), torch.bfloat16)
    _check(y2, _ref(x2, _deq(wq2, scale2), None, torch.float32), torch.float32)


# ---------------------------------------------------------------- GPU: dispatch end to end


def test_dispatch_end_to_end(ext, monkeypatch):
    monkeypatch.setattr(disp, "_ext", ext)
    monkeypatch.setattr(disp, "W8", True)
    disp._Q8_CACHE.clear()
    x = torch.randn(2, 2304, device="cuda", dtype=torch.bfloat16)
    big = torch.randn(2048, 2304, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(2, 2048, device="cuda", dtype=torch.bfloat16)
    assert disp.hip_gemv(x, big, None, out) is True
    torch.cuda.synchronize()
    wq, sc = disp.to_q8(big)
    _check(out, _ref(x, _deq(wq, sc), None, torch.bfloat16), torch.bfloat16)
    disp._Q8_CACHE.clear()


# ---------------------------------------------------------------- CPU: Q8_0 conversion


def _blocks(t):  # [N, K] -> [N, K/32, 32]
    return t.reshape(t.shape[0], -1, BLK)


def test_q8_format_scale_is_block_amax_over_127():
    w = torch.randn(64, 2304, dtype=torch.bfloat16) * torch.linspace(0.01, 8, 64)[:, None]
    wq, scale = disp.to_q8(w)
    assert wq.dtype == torch.int8 and wq.shape == w.shape
    assert scale.dtype == torch.float32 and scale.shape == (64, 2304 // BLK)
    wf = _blocks(w.float())
    torch.testing.assert_close(scale, wf.abs().amax(dim=2) / 127.0, rtol=0, atol=0)
    assert int(wq.abs().max()) == 127  # the block amax maps to +-127
    assert (_blocks(wq).abs().amax(dim=2) == 127).all()  # in EVERY block (no zero block here)
    deq = _blocks(wq).float() * scale[..., None]
    assert ((deq - wf).abs() <= scale[..., None] * (0.5 + 1e-3)).all()  # round to nearest


def test_q8_scale_varies_inside_a_row():
    w = torch.ones(1, 128, dtype=torch.bfloat16)
    w[0, 32:64] = 100.0
    wq, scale = disp.to_q8(w)
    torch.testing.assert_close(scale[0], torch.tensor([1.0, 100.0, 1.0, 1.0]) / 127.0, rtol=0, atol=0)
    assert (wq[0, :32] == 127).all() and (wq[0, 32:64] == 127).all()


def test_q8_conversion_is_chunked_but_identical(monkeypatch):
    w = torch.randn(50, 256, dtype=torch.bfloat16) * torch.linspace(0.1, 5, 50)[:, None]
    ref = disp.to_q8(w)
    monkeypatch.setattr(disp, "_CONV_ROWS", 7)  # 50 rows -> 8 chunks, the last one partial
    wq, scale = disp.to_q8(w)
    assert wq.equal(ref[0]) and scale.equal(ref[1])
    wf = _blocks(w.float())
    s = wf.abs().amax(dim=2) / 127.0  # independent whole-tensor formula
    assert scale.equal(s)
    assert wq.equal((wf / s[..., None]).round().clamp(-127, 127).to(torch.int8).reshape(50, 256))


def test_q8_zero_blocks_and_extremes():
    w = torch.zeros(4, 64, dtype=torch.bfloat16)
    w[1, 0] = 100.0
    w[2] = -0.5
    w[3, 32:] = 3e-30  # tiny but non-zero block
    wq, scale = disp.to_q8(w)
    assert torch.isfinite(scale).all() and (scale >= 0).all()
    deq = _blocks(wq).float() * scale[..., None]
    assert deq[0].abs().max() == 0 and (scale[0] == 0).all()  # all-zero block: scale 0, q 0
    assert (deq[1, 1] == 0).all()
    torch.testing.assert_close(deq[1].reshape(-1), w[1].float(), rtol=2**-7, atol=1e-6)
    torch.testing.assert_close(deq[2].reshape(-1), w[2].float(), rtol=2**-7, atol=1e-6)
    torch.testing.assert_close(deq[3].reshape(-1), w[3].float(), rtol=2**-7, atol=1e-35)


def test_q8_cache_reuses_and_dedups_prefix():
    disp._Q8_CACHE.clear()
    w = torch.randn(8600, 2304, dtype=torch.bfloat16)
    a = disp._q8_for(w)
    b = disp._q8_for(w)
    assert a[0].data_ptr() == b[0].data_ptr()  # converted once
    assert a[0].shape == (538, 36, 1024) and a[1].shape == (538, 36, 32)
    assert len(disp._Q8_CACHE) == 1
    q = disp._q8_for(w[:6552])  # row-0 prefix slice shares the base pointer: 6552 rows = 410 tiles
    assert q[0].data_ptr() == a[0].data_ptr() and q[0].shape[0] == 410
    assert q[1].data_ptr() == a[1].data_ptr() and q[1].shape == (410, 36, 32)
    assert len(disp._Q8_CACHE) == 1
    other = torch.randn(8600, 2304, dtype=torch.bfloat16)
    disp._q8_for(other)
    assert len(disp._Q8_CACHE) == 2
    disp._Q8_CACHE.clear()


def test_conversion_is_announced_once_per_weight(capsys):
    """The server log is the proof an arm really ran int8: one line per converted weight, none for a cache hit."""
    disp._Q8_CACHE.clear()
    w = torch.randn(64, 2304, dtype=torch.bfloat16)
    disp._q8_for(w)
    disp._q8_for(w)
    out = capsys.readouterr().out
    assert out.count("radiance_inkling2: int8 copy") == 1 and "64x2304" in out
    disp._Q8_CACHE.clear()


def test_q8_memory_cost_is_1_125_bytes_per_weight():
    w = torch.randn(1024, 2304, dtype=torch.bfloat16)
    wq, scale = disp.to_q8(w)
    assert (wq.numel() * wq.element_size() + scale.numel() * scale.element_size()) / w.numel() == 1.125


def test_pack_layout_matches_the_addressing_of_the_kernel():
    """Independent emulation of what each lane of a wave loads (see the .hip header): lane = 16 h + c
    reads 16 B at ((tile * KG + g) * 1024 + b * 512 + lane * 16) = weight row 16 tile + c, K = 64 g +
    32 b + 16 h ..+15, and 8 B of scale at (tile * KG + g) * 32 floats + c * 2 = that row's blocks 2g, 2g+1."""
    n, k = 40, 192  # 3 tiles (the last one holds 8 real rows), 3 groups
    g = torch.Generator().manual_seed(3)
    wq = torch.randint(-127, 128, (n, k), dtype=torch.int8, generator=g)
    scale = torch.rand(n, k // BLK, generator=g) + 0.5
    qt, st = disp.pack_q8(wq, scale)
    kg = k // 64
    assert qt.dtype == torch.int8 and qt.shape == (3, kg, 1024)
    assert st.dtype == torch.float32 and st.shape == (3, kg, 32)
    qf, sf = qt.reshape(-1), st.reshape(-1)
    for tile in range(3):
        for gi in range(kg):
            for b in range(2):
                for lane in range(32):
                    c, h = lane % 16, lane // 16
                    row = tile * 16 + c
                    off = (tile * kg + gi) * 1024 + b * 512 + lane * 16
                    want = wq[row, gi * 64 + b * 32 + 16 * h:][:16] if row < n else torch.zeros(16, dtype=torch.int8)
                    assert qf[off:off + 16].equal(want), (tile, gi, b, lane)
            for c in range(16):
                row = tile * 16 + c
                off = (tile * kg + gi) * 32 + c * 2
                want = scale[row, 2 * gi:2 * gi + 2] if row < n else torch.zeros(2)
                assert sf[off:off + 2].equal(want), (tile, gi, c)


def test_pack_is_1_125_bytes_per_weight_without_padding_rows():
    qt, st = disp.pack_q8(*disp.to_q8(torch.randn(1024, 2304, dtype=torch.bfloat16)))
    assert (qt.numel() + st.numel() * 4) / (1024 * 2304) == 1.125


@pytest.mark.parametrize("n,k", [(16, 63), (16, 100), (0, 64)])
def test_pack_rejects_what_the_kernel_cannot_read(n, k):
    with pytest.raises(AssertionError):
        disp.pack_q8(torch.zeros(n, k, dtype=torch.int8), torch.zeros(n, max(k // 32, 1)))


def test_hip_source_constants_match_the_python_format():
    src = (HERE.parent / "radiance_inkling2_gemv.hip").read_text()
    for name, val in (("Q_GROUP", disp.Q8_GROUP), ("Q_TILE", disp.Q8_TILE), ("Q_GROUP_BYTES", disp.Q8_GROUP * disp.Q8_TILE)):
        assert f"#define {name} {val}" in src, name


class FakeExt:
    NCFG = NCFG

    def __init__(self):
        self.calls = []

    def gemv_q8(self, *args):
        self.calls.append(args)


@pytest.fixture()
def routed(monkeypatch):
    """W8 on, fake extension, CPU tensors: the dispatcher decisions without a GPU."""
    fake = FakeExt()
    state = {"capturing": False}
    monkeypatch.setattr(disp, "_ext", fake)
    monkeypatch.setattr(disp, "W8", True)
    monkeypatch.setattr(disp, "HIP_BF16", False)  # these tests isolate the int8 route
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: type("S", (), {"cuda_stream": 7})())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: state["capturing"])
    disp._Q8_CACHE.clear()
    yield fake, state
    disp._Q8_CACHE.clear()


def _t(m, n, k=2304, dtype=torch.bfloat16):
    return torch.randn(m, k, dtype=dtype), torch.randn(n, k, dtype=dtype), torch.empty(m, n, dtype=dtype)


def test_w8_defaults_off_and_the_int8_route_needs_it(monkeypatch):
    fake = FakeExt()
    monkeypatch.setattr(disp, "_ext", fake)
    monkeypatch.setattr(disp, "W8", False)
    monkeypatch.setattr(disp, "HIP_BF16", False)  # the bf16 route has its own tests (test_gemv_bf16_hip.py)
    x, w, out = _t(1, 4096)
    assert disp.hip_gemv(x, w, None, out) is False
    assert fake.calls == [] and not hasattr(disp, "ENABLED")  # int8 knob: W8
    src = (HERE.parent / "radiance_inkling2_gemv.hip").read_text()
    assert "gemv_q8" in src


def test_routes_big_weight_to_q8_with_scale_and_strides(routed):
    fake, _ = routed
    x, w, out = _t(2, 4096)
    assert disp.hip_gemv(x, w, None, out) is True
    (xp, wp, sp, bp, yp, m, n, k, sx, sy, o32, nt, stream, cfg), = fake.calls
    wq, sc = disp.to_q8(w)
    assert (m, n, k, sx, sy, o32, bp, nt, stream, cfg) == (2, 4096, 2304, 2304, 4096, 0, 0, 256, 7, -1)
    assert (xp, yp) == (x.data_ptr(), out.data_ptr())
    ent = disp._Q8_CACHE[(w.data_ptr(), 2304)]
    assert wp == ent[1].data_ptr() and sp == ent[2].data_ptr()  # the cached int8 copy, not bf16
    qt, st = disp.pack_q8(wq, sc)  # the tiled copy of the plain conversion
    assert ent[1].equal(qt) and ent[2].equal(st)


def test_fp32_out_and_bias_are_passed(routed):
    fake, _ = routed
    x, w, _out = _t(1, 4096)
    bias = torch.randn(4096, dtype=torch.bfloat16)
    out = torch.empty(1, 4096, dtype=torch.float32)
    assert disp.hip_gemv(x, w, bias, out) is True
    a = fake.calls[0]
    assert a[3] == bias.data_ptr() and a[10] == 1


@pytest.mark.parametrize("m", range(1, 18))
def test_every_spec_verify_width_is_routed(routed, m):
    """DSpark verify feeds k + 1 = 8 rows (any k up to 15 must stay on the int8 kernel); 17+ is prefill."""
    fake, _ = routed
    x, w, out = _t(m, 4096)
    assert disp.hip_gemv(x, w, None, out) is (m <= 16)
    assert len(fake.calls) == (1 if m <= 16 else 0)
    if m > 16:
        assert disp._Q8_CACHE == {}  # no conversion for a call the kernel cannot take


@pytest.mark.parametrize("n", [128, 1024])
def test_small_tensors_and_router_stay_on_triton(routed, n):
    fake, _ = routed
    x, w, out = _t(1, n)  # MoE router 128 x 2304; k/v 1024 x 2304: launch bound, 1.05x at best
    assert disp.hip_gemv(x, w, None, out) is False
    assert fake.calls == [] and disp._Q8_CACHE == {}
    x, w, out = _t(1, 2048)  # DSpark kv_ctx 2048 x 2304 is big enough
    assert disp.hip_gemv(x, w, None, out) is True


@pytest.mark.parametrize("what", ["k%64", "fp32_x", "xstride", "xalign", "out_dtype"])
def test_unsupported_inputs_fall_back(routed, what):
    fake, _ = routed
    x, w, out = _t(2, 4096)
    if what == "k%64":
        x, w, out = _t(2, 4096, k=2336)  # 2336 % 32 == 0, % 64 == 32
    elif what == "fp32_x":
        x = x.float()
    elif what == "xstride":
        x = torch.randn(2, 2304 + 4, dtype=torch.bfloat16)[:, :2304]  # row stride % 8 != 0
    elif what == "xalign":
        x = torch.randn(2 * 2304 + 8, dtype=torch.bfloat16)[1:1 + 2 * 2304].view(2, 2304)  # +2 B
    elif what == "out_dtype":
        out = out.half()
    assert disp.hip_gemv(x, w, None, out) is False
    assert fake.calls == []


def test_graph_capture_never_converts(routed):
    fake, state = routed
    x, w, out = _t(1, 4096)
    state["capturing"] = True
    assert disp.hip_gemv(x, w, None, out) is False  # unseen weight under capture: Triton
    assert disp._Q8_CACHE == {} and fake.calls == []
    state["capturing"] = False
    assert disp.hip_gemv(x, w, None, out) is True  # eager call converts (warmup before capture)
    state["capturing"] = True
    assert disp.hip_gemv(x, w, None, out) is True  # captured call reuses the cached copy
    assert len(fake.calls) == 2 and fake.calls[0][1] == fake.calls[1][1]


def test_prefix_slice_reuses_the_conversion(routed):
    fake, _ = routed
    x, w, out = _t(1, 8600)
    assert disp.hip_gemv(x, w, None, out) is True
    out2 = torch.empty(1, 6552, dtype=torch.bfloat16)
    assert disp.hip_gemv(x, w[:6552], None, out2) is True  # global layers: q-side row prefix
    assert len(disp._Q8_CACHE) == 1 and fake.calls[1][1] == fake.calls[0][1]
    assert fake.calls[1][6] == 6552 and fake.calls[1][11] == 410  # 6552 rows = 409.5 tiles: 410 of 538


def test_head_stays_bf16_without_w8_head(routed, monkeypatch):
    """RADIANCE_INKLING2_W8_HEAD=0 (fidelity arm 'q8nohead'): a vocab-sized projection is not quantized."""
    fake, _ = routed
    x, head, out = _t(2, 201024)
    assert disp.hip_gemv(x, head, None, out) is True  # default W8_HEAD=1: the head goes int8
    monkeypatch.setattr(disp, "W8_HEAD", False)
    fake.calls.clear()
    assert disp.hip_gemv(x, head, None, out) is False and fake.calls == []
    x, w, out = _t(2, 18432)
    assert disp.hip_gemv(x, w, None, out) is True  # every other projection stays int8


def test_w8_head_defaults_on():
    assert disp.W8_HEAD is True  # RADIANCE_INKLING2_W8_HEAD unset


def test_w8_gating():
    big = torch.randn(2048, 2304, dtype=torch.bfloat16)
    router = torch.randn(128, 2304, dtype=torch.bfloat16)
    assert disp._want_q8(big) is False  # W8 defaults off
    disp.W8 = True
    try:
        assert disp._want_q8(big) is True
        assert disp._want_q8(router) is False  # tiny tensors stay bf16
        assert disp._want_q8(big.float()) is False
    finally:
        disp.W8 = False


def test_fallback_without_extension(monkeypatch):
    monkeypatch.setattr(disp, "_ext", None)
    monkeypatch.setattr(disp, "W8", True)
    x, w, out = _t(1, 4096)
    assert disp.hip_gemv(x, w, None, out) is False


# ---------------------------------------------------------------- CPU: patch hook


@pytest.fixture()
def tree(tmp_path):
    src = os.environ.get("INKLING_VLLM_SRC")
    if not src or not Path(src, "models/inkling2").exists():
        pytest.skip("INKLING_VLLM_SRC not set")
    shutil.copytree(src, tmp_path / "vllm", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def _patch(target):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(target), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(REPO / "sly/patch_inkling2_gfx12.py")],
                          capture_output=True, text=True, env=env)


def test_patch_routes_gemv_to_hip(tree):
    r = _patch(tree)
    assert r.returncode == 0, r.stdout + r.stderr
    s = (tree / "vllm/models/inkling2/nvidia/ops/gemv.py").read_text()
    assert "from radiance_inkling2_gemv_hip import hip_gemv" in s
    assert "if hip_gemv(x2, weight, bias, out):" in s
    assert "torch.version.hip" in s
    # the Triton kernel stays as the fallback
    assert "_gemv_kernel[(triton.cdiv(n, _BN),)](" in s


def test_patch_gemv_hook_idempotent(tree):
    import ast
    assert _patch(tree).returncode == 0
    p = tree / "vllm/models/inkling2/nvidia/ops/gemv.py"
    before = p.read_text()
    ast.parse(before)
    r = _patch(tree)
    assert r.returncode == 0
    assert "NOOP  inkling2 gemv" in r.stdout
    assert p.read_text() == before


# ---------------------------------------------------------------- CPU: mutation checks
# Each mutation goes into a COPY in tmp_path; the real file is never touched. After the check we
# assert the real file is byte-identical to what it was before (restore contract).


def _mutated_module(tmp_path, old, new):
    src = (HERE.parent / "radiance_inkling2_gemv_hip.py").read_text()
    assert src.count(old) == 1, f"mutation anchor not unique: {old!r}"
    # one file per mutation: a rewritten mutant.py of the same size inside the same second would load the
    # previous mutant's cached .pyc (source mtime + size are the cache key) and "survive"
    mut = tmp_path / f"mutant_{abs(hash((old, new)))}.py"
    mut.write_text(src.replace(old, new, 1))
    spec = importlib.util.spec_from_file_location("mutant_gemv_disp", mut)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _arm(m, monkeypatch, capturing=False):
    """Same fake-extension setup as the `routed` fixture, on a mutant module."""
    fake = FakeExt()
    monkeypatch.setattr(m, "_ext", fake)
    monkeypatch.setattr(m, "W8", True)
    monkeypatch.setattr(m, "HIP_BF16", False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: type("S", (), {"cuda_stream": 7})())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    m._Q8_CACHE.clear()
    return fake


def test_mutations_are_caught(tmp_path, monkeypatch):
    real = (HERE.parent / "radiance_inkling2_gemv_hip.py").read_bytes()
    w = torch.randn(64, 2304, dtype=torch.bfloat16)

    def check_scale(m):
        torch.testing.assert_close(m.to_q8(w)[1], _blocks(w.float()).abs().amax(dim=2) / 127.0,
                                   rtol=0, atol=0)

    def check_threshold(m):
        m.W8 = True
        assert not m._want_q8(torch.randn(128, 2304, dtype=torch.bfloat16)), "router quantized"

    def check_round(m):
        wf = _blocks(w.float())
        wq, sc = m.to_q8(w)
        deq = _blocks(wq).float() * sc[..., None]
        assert ((deq - wf).abs() <= sc[..., None] * (0.5 + 1e-3)).all(), "not round-to-nearest"

    def check_scale_blocks(m):
        _, sc = m.to_q8(w)
        assert sc.shape == (64, 2304 // BLK), "scale is not per 32 values"

    def check_m(m):
        fake = _arm(m, monkeypatch)
        x, big, out = _t(m.MAX_M + 1, 4096)
        assert m.hip_gemv(x, big, None, out) is False, "M above the kernel range routed"

    def check_head(m):
        fake = _arm(m, monkeypatch)
        m.W8_HEAD = False
        x, head, out = _t(2, 201024)
        assert m.hip_gemv(x, head, None, out) is False, "head quantized with W8_HEAD=0"

    def check_capture(m):
        fake = _arm(m, monkeypatch, capturing=True)
        x, big, out = _t(1, 4096)
        assert m.hip_gemv(x, big, None, out) is False and not m._Q8_CACHE, "converted under capture"

    def check_k(m):
        _arm(m, monkeypatch)
        x, big, out = _t(2, 4096, k=2336)
        assert m.hip_gemv(x, big, None, out) is False, "K % 64 != 0 routed"

    def check_align(m):
        _arm(m, monkeypatch)
        x = torch.randn(2 * 2304 + 8, dtype=torch.bfloat16)[1:1 + 2 * 2304].view(2, 2304)
        _, big, out = _t(2, 4096)
        assert m.hip_gemv(x, big, None, out) is False, "misaligned x routed"

    def check_bf16_weight_only(m):
        _arm(m, monkeypatch)
        x, big, out = _t(1, 4096)
        assert m.hip_gemv(x, big.float(), None, out) is False, "fp32 weight routed"

    mutations = [
        ("/ 127.0", "/ 63.0", check_scale),    # wrong int8 max -> wrong scale
        ("Q8_MIN_NUMEL = 3 << 20", "Q8_MIN_NUMEL = 0", check_threshold),  # tiny tensors quantized
        (".round()", ".trunc()", check_round),  # round to nearest, not toward zero
        ("Q8_BLOCK = 32", "Q8_BLOCK = 64", check_scale_blocks),
        ("1 <= m <= MAX_M  # the kernel's M range: a spec-verify step (k + 1 rows) is routed whole",
         "1 <= m  # the kernel's M range: a spec-verify step (k + 1 rows) is routed whole", check_m),  # the kernel takes M <= 16
        ("and (W8_HEAD or weight.shape[0] < Q8_HEAD_MIN_N)", "", check_head),
        ("elif not torch.cuda.is_current_stream_capturing():", "elif True:", check_capture),
        ("k % 64 == 0", "k % 32 == 0", check_k),  # the kernel reads whole 64 K tile groups
        ("x2.data_ptr() % 16 == 0", "True", check_align),  # 16 B vector x staging
        ("weight.dtype == torch.bfloat16 and", "", check_bf16_weight_only),
    ]
    caught = 0
    for old, new, check in mutations:
        m = _mutated_module(tmp_path, old, new)
        try:
            check(m)
        except (AssertionError, RuntimeError):  # RuntimeError: the mutant crashed on the input
            caught += 1
        else:
            raise AssertionError(f"mutation survived: {old!r} -> {new!r}")
    assert caught == len(mutations)
    assert (HERE.parent / "radiance_inkling2_gemv_hip.py").read_bytes() == real  # restored


def test_kernel_mutants_apply_to_the_real_source(tmp_path):
    """The GPU-window mutants (each must turn a test red) come from exact, unique anchors."""
    import mutate_gemv_hip as mg

    src = (HERE.parent / "radiance_inkling2_gemv.hip").read_text()
    names = mg.write_mutants(HERE.parent / "radiance_inkling2_gemv.hip", tmp_path)
    assert len(names) == len(mg.MUTATIONS) >= 10
    for name, (old, new) in mg.MUTATIONS.items():
        assert src.count(old) == 1, name
        assert (tmp_path / f"m_{name}.hip").read_text() == src.replace(old, new, 1)
        assert old != new


def test_patch_reapplies_after_the_hook_was_removed(tree):
    """Sentinel check: a tree whose dispatch was deleted is patched again (not a false NOOP)."""
    assert _patch(tree).returncode == 0
    p = tree / "vllm/models/inkling2/nvidia/ops/gemv.py"
    s = p.read_text()
    block = ("    if m and torch.version.hip:\n"
             "        from radiance_inkling2_gemv_hip import hip_gemv\n\n"
             "        if hip_gemv(x2, weight, bias, out):\n"
             "            return out.view(*lead, n)\n")
    assert block in s
    p.write_text(s.replace(block, "", 1))
    r = _patch(tree)
    assert r.returncode == 0
    assert "OK    inkling2 gemv" in r.stdout
    assert p.read_text() == s  # byte-identical restore by the patch itself