"""sly/inkling2/radiance_inkling2_mx4.hip: MXFP4 (e2m1 + E8M0 per 32) W4A16 low-M GEMV on the matrix unit.

GPU tests (skip cleanly without a GPU or the compiled module; CI is CPU only): the kernel against an independent
torch fp32 reference built from the format spec, every M 1..16, every launch config, real shapes, odd N, tiny N,
K tails, x strides, fp32 out, every e2m1 code at every nibble position, per-block scales, guards, graph capture,
and the prefill dequant kernel (bit exact). The tile layout comes from radiance_inkling2_mxfp4dense.pack_mx4
(CPU-tested in test_mx4_layout.py).

RADIANCE_MX4_BUILD=<dir> asserts which build (RED stub / mutant runs) the process imported: single-phase
extension modules are cached by NAME, so every build lives in its own dir and each pytest process gets one.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # sly/inkling2

import radiance_inkling2_mxfp4dense as MD  # noqa: E402

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
NCFG = 12  # launch configs (T row tiles per wave, W K-splitting waves per block, D weight groups in flight)
# Real Inkling-2-Mini decode shapes (N, K): fused qkvr and slices, wo, unembed, DSpark qkv / kv_ctx / gate_up / down.
REAL_SHAPES = [
    (8600, 2304), (6552, 2304), (1024, 2304), (3072, 2304), (2304, 3072), (201024, 2304),
    (5120, 2304), (2048, 2304), (18432, 2304), (2304, 9216),
]


def _ext():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    try:
        import radiance_inkling2_mx4 as m
    except ImportError:
        pytest.skip("radiance_inkling2_mx4 module not built")
    return m


@pytest.fixture()
def ext():
    return _ext()


def ref_dequant(packed, scale):
    """Independent of the kernel and of pack_mx4: packed [N, K/2] (low nibble = even k, bit 3 = sign), scale
    [K/32, N] E8M0 (2^(e-127), byte 0 = factor 0) -> fp32 [N, K]."""
    n, kh = packed.shape
    c = torch.stack([packed & 0x0F, packed >> 4], -1).reshape(n, kh * 2).long()
    tab = torch.tensor(E2M1, device=packed.device)
    val = torch.where(c >= 8, -tab[c & 7], tab[c & 7])
    s = scale.T.float()
    s = torch.where(s == 0, 0.0, torch.exp2(s - 127.0)).repeat_interleave(32, dim=1)
    return val * s


def rand_mx(n, k, g, lo=-2, hi=2):
    packed = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8, generator=g)
    scale = (127 + torch.randint(lo, hi + 1, (k // 32, n), device="cuda", generator=g)).to(torch.uint8)
    return packed, scale


def _call(ext, x, packed, scale, y, *, cfg=-1):
    m, k = x.shape
    n = packed.shape[0]
    qt, st = MD.pack_mx4(packed, scale)
    ext.gemv_mx4(x.data_ptr(), qt.data_ptr(), st.data_ptr(), y.data_ptr(), m, n, k, x.stride(0), y.stride(0),
                 int(y.dtype == torch.float32), qt.shape[0], torch.cuda.current_stream().cuda_stream, cfg)
    torch.cuda.synchronize()


def _check(y, ref, dt):
    if dt == torch.float32:
        torch.testing.assert_close(y.float(), ref.float(), rtol=2e-5, atol=2e-4)
    else:  # bf16 rounding of the output: 1 ulp margin over the fp32 accumulation order
        torch.testing.assert_close(y.float(), ref.float(), rtol=2**-7, atol=1e-3)


def _run(ext, m, n, k, *, out32=False, sx=None, cfg=-1, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, sx or k, device="cuda", dtype=torch.bfloat16, generator=g)[:, :k]
    packed, scale = rand_mx(n, k, g, -9, -6)  # |w| ~ 6 * 2^-7 .. : keeps the sums O(1)
    dt = torch.float32 if out32 else torch.bfloat16
    y = torch.empty(m, n, device="cuda", dtype=dt)
    _call(ext, x, packed, scale, y, cfg=cfg)
    _check(y, (x.float() @ ref_dequant(packed, scale).t()).to(dt), dt)


def _guarded(m, n, dt, pad=64):
    buf = torch.full((m + 1, n + pad), 7.0, device="cuda", dtype=dt)
    return buf, buf[:m, :n]


def _run_exact(ext, m, n, k, *, out32=True, sx=0, cfg=-1, seed=0):
    """Small integer x, any e2m1 codes, power-of-two scales in 2^-2..2^2: every product and partial sum is exact in
    fp32 in ANY order, so the result equals the double reference bit for bit. x rows have a NaN-poisoned stride
    padding, y a sentinel border (nothing may be written outside [m, n])."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    xb = torch.full((m, max(sx, k)), float("nan"), device="cuda", dtype=torch.bfloat16)
    xb[:, :k] = torch.randint(-3, 4, (m, k), device="cuda", generator=g).to(torch.bfloat16)
    x = xb[:, :k]
    packed, scale = rand_mx(n, k, g)
    dt = torch.float32 if out32 else torch.bfloat16
    buf, y = _guarded(m, n, dt)
    _call(ext, x, packed, scale, y, cfg=cfg)
    ref = (x.double() @ ref_dequant(packed, scale).double().t()).float().to(dt)
    assert torch.equal(y, ref), f"max |diff| {(y.float() - ref.float()).abs().max().item()}"
    assert (buf[:, n:] == 7.0).all() and (buf[m:] == 7.0).all(), "wrote outside [m, n]"


# ---------------------------------------------------------------- GPU: kernel


def test_loaded_build_is_the_requested_one(ext):
    want = os.environ.get("RADIANCE_MX4_BUILD")
    if want:
        assert os.path.realpath(os.path.dirname(ext.__file__)) == os.path.realpath(want)


def test_ncfg_matches(ext):
    assert ext.NCFG == NCFG and len(ext.CFGS) == NCFG


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
    _run(ext, m, n, k, out32=(n == 201024))


@pytest.mark.parametrize("m", [1, 3, 8, 16])
@pytest.mark.parametrize("n", [1, 15, 17, 23, 1023, 8601])
def test_odd_n(ext, m, n):
    _run(ext, m, n, 2304)


@pytest.mark.parametrize("cfg", range(NCFG))
@pytest.mark.parametrize("n", [1, 7, 33])  # fewer rows than one block, rows past N never stored
def test_tiny_n_every_config(ext, cfg, n):
    _run_exact(ext, 8, n, 2304, cfg=cfg)


@pytest.mark.parametrize("m", [1, 16])
@pytest.mark.parametrize("k", [2368, 192, 64, 4160])  # %64 == 0 but an odd number of 64 K groups / not %128
def test_k_tail(ext, m, k):
    _run(ext, m, 1024, k)


@pytest.mark.parametrize("cfg", range(NCFG))
@pytest.mark.parametrize("m", range(1, 17))
def test_every_m_every_config_exact(ext, cfg, m):
    _run_exact(ext, m, 2000, 2368, cfg=cfg)


@pytest.mark.parametrize("m", range(1, 17))
def test_every_m_auto_config_exact(ext, m):
    _run_exact(ext, m, 2304, 3072)
    _run_exact(ext, m, 8600, 2304, out32=False)


@pytest.mark.parametrize("m", [1, 8, 16])
def test_strides_exact(ext, m):
    _run_exact(ext, m, 1001, 2304, sx=4096)
    _run_exact(ext, m, 4097, 2368, out32=False, sx=2368 + 8)


@pytest.mark.parametrize("m", [1, 5, 16])
def test_x_row_stride(ext, m):
    _run(ext, m, 2304, 2304, sx=4096)


@pytest.mark.parametrize("m", [1, 16])
def test_fp32_out(ext, m):
    _run(ext, m, 1024, 2304, out32=True)
    _run(ext, m, 4096, 2304, out32=True, sx=2560)


@pytest.mark.parametrize("pos", range(8))
def test_every_code_at_every_nibble_position_exact(ext, pos):
    """Row r holds code (r % 16) at K = 8 * (r // 16) + pos of group 0..: y = that e2m1 value, scale 1. Together
    with x = 1 this pins slot <-> nibble <-> K for all 16 codes (sign included) and every position of the dword."""
    n, k = 256, 2304
    codes = torch.zeros(n, k, dtype=torch.uint8)
    r = torch.arange(n)
    codes[r, (r // 16) * 8 + pos + 64 * (r % 7)] = (r % 16).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).cuda()
    scale = torch.full((k // 32, n), 127, dtype=torch.uint8, device="cuda")
    x = torch.ones(1, k, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(1, n, device="cuda", dtype=torch.float32)
    _call(ext, x, packed, scale, y)
    tab = torch.tensor(E2M1 + [-v for v in E2M1], device="cuda")
    torch.testing.assert_close(y[0], tab[(r % 16).cuda()], rtol=0, atol=0)


def test_scale_is_per_32_block_not_per_row(ext):
    n, k = 64, 2304
    g = torch.Generator(device="cuda").manual_seed(1)
    packed, scale = rand_mx(n, k, g, -6, 6)  # varies per block
    x = torch.randn(2, k, device="cuda", dtype=torch.bfloat16, generator=g)
    y = torch.empty(2, n, device="cuda", dtype=torch.float32)
    _call(ext, x, packed, scale, y)
    deq = ref_dequant(packed, scale)
    ref = x.double() @ deq.double().t()
    bound = x.abs().double() @ deq.abs().double().t()  # fp32 summation error scales with sum|terms|
    assert ((y.double() - ref).abs() <= 1e-5 * bound).all()


def test_scale_byte_zero_is_a_zero_factor(ext):
    n, k = 32, 256
    g = torch.Generator(device="cuda").manual_seed(2)
    packed, scale = rand_mx(n, k, g)
    scale[3, 5] = 0  # block 3 of row 5 contributes nothing
    x = torch.randn(1, k, device="cuda", dtype=torch.bfloat16, generator=g)
    y = torch.empty(1, n, device="cuda", dtype=torch.float32)
    _call(ext, x, packed, scale, y)
    _check(y, x.float() @ ref_dequant(packed, scale).t(), torch.float32)


def test_auto_config_follows_the_measured_table(ext):
    """Rotated-pool graph sweep (bench_mx4.py --cfgs, M 1 / 8 / 16 alike, GB/s of codes + scales): cfg 2 (T=1 W=8 D=4)
    for the 384 / 1024-row k/v (N < 2048), cfg 1 (T=1 W=4 D=4) for 2k-4k and 8.6k rows, cfg 7 (T=2 W=4 D=4) for the 6552-row
    q side, cfg 11 (T=2 W=8 D=8) for the unembed."""
    table = {(384, 2304): 2, (1024, 2304): 2, (2048, 2304): 1, (2304, 3072): 1, (3072, 2304): 1, (5120, 2304): 1,
             (6552, 2304): 7, (8600, 2304): 1, (18432, 2304): 1, (2304, 9216): 1, (201024, 2304): 11}
    for (n, k), cfg in table.items():
        for m in (1, 8, 16):
            assert ext.auto_cfg_mx4(m, n, k) == cfg, (m, n, k)


def test_guards_throw_instead_of_faulting(ext):
    m, n, k = 2, 64, 2304
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    qt, st = MD.pack_mx4(*rand_mx(n, k, g))
    y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream

    def call(**kw):
        a = dict(x=x.data_ptr(), q=qt.data_ptr(), s=st.data_ptr(), y=y.data_ptr(), M=m, N=n, K=k, SX=k, SY=n,
                 out32=0, ntiles=qt.shape[0], stream=stream, cfg=-1)
        a.update(kw)
        ext.gemv_mx4(**a)

    call()  # the arguments themselves are valid
    for bad in (dict(M=0), dict(M=17), dict(N=0), dict(K=2272), dict(K=32), dict(SX=k - 8), dict(SX=k + 4),
                dict(SY=n - 1), dict(x=x.data_ptr() + 2), dict(q=qt.data_ptr() + 8), dict(s=st.data_ptr() + 8),
                dict(ntiles=qt.shape[0] - 1), dict(N=n + 1), dict(cfg=NCFG), dict(cfg=-2)):
        with pytest.raises((ValueError, RuntimeError)):
            call(**bad)
    torch.cuda.synchronize()


# ---------------------------------------------------------------- GPU: prefill dequant kernel


@pytest.mark.parametrize("n,k", [(8600, 2304), (37, 192), (16, 64), (2304, 3072), (1, 64)])
def test_dequant_is_bit_exact(ext, n, k):
    g = torch.Generator(device="cuda").manual_seed(4)
    packed, scale = rand_mx(n, k, g, -9, 3)
    qt, st = MD.pack_mx4(packed, scale)
    y = torch.full((n, k), 5.0, device="cuda", dtype=torch.bfloat16)
    ext.dequant_mx4(qt.data_ptr(), st.data_ptr(), y.data_ptr(), n, k, qt.shape[0],
                    torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    assert torch.equal(y, ref_dequant(packed, scale).to(torch.bfloat16))


def test_dequant_guards(ext):
    n, k = 40, 128
    qt, st = MD.pack_mx4(*rand_mx(n, k, torch.Generator(device="cuda").manual_seed(5)))
    y = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.current_stream().cuda_stream
    for bad in (dict(N=0), dict(K=100), dict(ntiles=qt.shape[0] - 1), dict(y=y.data_ptr() + 2)):
        a = dict(q=qt.data_ptr(), s=st.data_ptr(), y=y.data_ptr(), N=n, K=k, ntiles=qt.shape[0], stream=stream)
        a.update(bad)
        with pytest.raises((ValueError, RuntimeError)):
            ext.dequant_mx4(**a)


# ---------------------------------------------------------------- GPU: graph capture


def test_graph_capture(ext):
    m, n, k = 4, 2304, 3072
    g = torch.Generator(device="cuda").manual_seed(6)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    pk, sc = rand_mx(n, k, g, -9, -6)
    nsm, nsk = 64, 2304  # a tiny shape too
    pk2, sc2 = rand_mx(nsm, nsk, g, -9, -6)
    qt, st = MD.pack_mx4(pk, sc)
    qt2, st2 = MD.pack_mx4(pk2, sc2)
    y1 = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    y2 = torch.empty(m, nsm, device="cuda", dtype=torch.float32)
    x2 = torch.randn(m, nsk, device="cuda", dtype=torch.bfloat16)
    gs = torch.cuda.Stream()  # capture + launch on a dedicated non-default stream

    def launch(sptr: int) -> None:
        ext.gemv_mx4(x.data_ptr(), qt.data_ptr(), st.data_ptr(), y1.data_ptr(), m, n, k, k, n, 0, qt.shape[0],
                     sptr, -1)
        ext.gemv_mx4(x2.data_ptr(), qt2.data_ptr(), st2.data_ptr(), y2.data_ptr(), m, nsm, nsk, nsk, nsm, 1,
                     qt2.shape[0], sptr, -1)

    for _ in range(3):  # eager warmup on the capture stream
        launch(int(gs.cuda_stream))
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=gs):
        launch(int(gs.cuda_stream))
    x.copy_(torch.randn_like(x))
    x2.copy_(torch.randn_like(x2))
    graph.replay()
    torch.cuda.synchronize()
    _check(y1, (x.float() @ ref_dequant(pk, sc).t()).to(torch.bfloat16), torch.bfloat16)
    _check(y2, x2.float() @ ref_dequant(pk2, sc2).t(), torch.float32)


# ---------------------------------------------------------------- GPU: the module route end to end


def test_module_route_matches_the_dequant_dot_for_every_decode_m_and_prefill(ext):
    import importlib

    for k_ in [k for k in os.environ if k.startswith("RADIANCE_INKLING2_MXFP4_DENSE")]:
        os.environ.pop(k_)
    os.environ["RADIANCE_INKLING2_MXFP4_DENSE"] = "1"
    os.environ["RADIANCE_INKLING2_MXFP4_DENSE_MODE"] = "gemv"
    sys.modules.pop("radiance_inkling2_mxfp4dense", None)
    md = importlib.import_module("radiance_inkling2_mxfp4dense")
    assert md._resolve_mx() is not None  # what _ready() does at load
    g = torch.Generator(device="cuda").manual_seed(7)
    w = (torch.randn(3000, 2304, device="cuda", dtype=torch.bfloat16, generator=g) / 48)
    p = torch.nn.Parameter(w.clone(), requires_grad=False)
    md.register(p, [(0, 1000), (1000, 777)])
    packed, scale, _ = md.quantize_mxfp4(w)
    deq = ref_dequant(packed, scale)
    for mm in (1, 7, 16, 17, 40):
        x = torch.randn(mm, 2304, device="cuda", dtype=torch.bfloat16, generator=g)
        for r0, nn in ((0, 3000), (0, 1000), (1000, 777)):
            y = md.mxfp4_dense_linear(x, p.data[r0:r0 + nn])
            _check(y, (x.float() @ deq[r0:r0 + nn].t()).to(torch.bfloat16), torch.bfloat16)
