"""CPU tests for the MXFP4 dense route on the HIP W4A16 GEMV (MODE=gemv): entries, routing, knobs.

The kernel itself is GPU-only (tests/test_gemv_mx4_hip.py); here a fake extension implements its contract
through the raw pointers the module passes (tiles are undone with an inverse written from the .hip header,
independent of pack_mx4) so routing, strides, chunking, slices and memory behaviour are real code paths.
"""
import ctypes
import importlib
import os
import sys
import types

import pytest
import torch

MODNAME = "radiance_inkling2_mxfp4dense"
K, QSIDE, QROWS = 128, 149, 152
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
POS_OF_SLOT = [0, 2, 4, 6, 1, 3, 5, 7]  # slot s of an operand <- nibble POS_OF_SLOT[s] of its code dword


def load(**env):
    for k in [k for k in os.environ if k.startswith("RADIANCE_INKLING2_MXFP4_DENSE")]:
        os.environ.pop(k)
    os.environ.update(env)
    sys.modules.pop(MODNAME, None)
    return importlib.import_module(MODNAME)


def view(ptr, dtype, shape, strides=None):
    """A CPU tensor over raw memory (the kernel's view of an argument)."""
    n = 1
    for s in shape:
        n *= s
    esz = torch.empty((), dtype=dtype).element_size()
    span = n * esz if strides is None else (sum((d - 1) * st for d, st in zip(shape, strides)) + 1) * esz
    buf = (ctypes.c_uint8 * span).from_address(ptr)
    t = torch.frombuffer(buf, dtype=torch.uint8).view(dtype)
    return t.reshape(shape) if strides is None else t.as_strided(shape, strides)


def untile(qt, st, n, k):
    """tiles (u8 [NT, K/64, 512], u8 [NT, K/64, 32]) -> fp32 [N, K] dequantized weights, from the header's layout."""
    nt, kg = qt.shape[0], k // 64
    b = qt.reshape(nt, kg, 2, 16, 2, 2, 4)  # [t, g, h, c, b, j, byte]
    nibs = torch.stack([b & 0x0F, b >> 4], -1).reshape(nt, kg, 2, 16, 2, 2, 8)  # nibble 2i = byte i low
    code = nibs[..., POS_OF_SLOT]  # slot s <- nibble POS_OF_SLOT[s]
    code = code.permute(0, 3, 1, 4, 2, 5, 6).reshape(nt * 16, k).long()  # [t,c,g,b,h,j,slot] -> rows x K
    mag = torch.tensor(E2M1)[code & 7]
    val = torch.where(code >= 8, -mag, mag)
    sc = st.reshape(nt, kg, 16, 2).permute(0, 2, 1, 3).reshape(nt * 16, k // 32)
    sc = torch.where(sc == 0, 0.0, torch.exp2(sc.float() - 127.0)).repeat_interleave(32, dim=1)
    return (val * sc)[:n]


class FakeMx4:
    """radiance_inkling2_mx4 on the CPU: same argument contract, same guards the tests rely on."""
    NCFG = 12

    def __init__(self):
        self.gemv_calls, self.dq_calls = [], []

    def gemv_mx4(self, x, q, s, y, M, N, K, SX, SY, out32, ntiles, stream, cfg=-1):
        assert 1 <= M <= 16 and K % 64 == 0 and SX % 8 == 0 and SX >= K and x % 16 == 0 and q % 16 == 0
        assert ntiles >= -(-N // 16)
        self.gemv_calls.append(dict(M=M, N=N, K=K, SX=SX, SY=SY, out32=out32, ntiles=ntiles, cfg=cfg))
        xt = view(x, torch.bfloat16, (M, K), (SX, 1)).float()
        w = untile(view(q, torch.uint8, (ntiles, K // 64, 512)), view(s, torch.uint8, (ntiles, K // 64, 32)), N, K)
        yv = (xt @ w.t())
        dt = torch.float32 if out32 else torch.bfloat16
        view(y, dt, (M, N), (SY, 1)).copy_(yv.to(dt))

    def dequant_mx4(self, q, s, y, N, K, ntiles, stream):
        self.dq_calls.append(dict(N=N, K=K, ntiles=ntiles))
        w = untile(view(q, torch.uint8, (ntiles, K // 64, 512)), view(s, torch.uint8, (ntiles, K // 64, 32)), N, K)
        view(y, torch.bfloat16, (N, K)).copy_(w.to(torch.bfloat16))


def mx4(**env):
    env.setdefault("RADIANCE_INKLING2_MXFP4_DENSE_MODE", "gemv")
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MIN_N="32", **env)
    m._mx = FakeMx4()
    return m


def expected(m, x, w):
    packed, scale, _ = m.quantize_mxfp4(w.clone())
    return (x.float() @ m.dequant_mxfp4(packed, scale).float().t()).to(torch.bfloat16)


def reg(m, w, rows=(), free=True):
    p = torch.nn.Parameter(w.clone(), requires_grad=False)
    m.register(p, rows, free)
    return p


def test_gemv_entry_keeps_only_the_tiles():
    m = mx4()
    torch.manual_seed(0)
    p = reg(m, torch.randn(200, K, dtype=torch.bfloat16))
    (ent,) = m._CACHE.values()
    assert set(ent) == {"qt", "st"}  # no [N, K/2] copy, no wref: the tiles are the whole 4-bit weight
    assert tuple(ent["qt"].shape) == (13, 2, 512) and tuple(ent["st"].shape) == (13, 2, 32)
    assert p.data.stride() == (1, 0)  # bf16 released


def test_decode_rows_go_to_one_gemv_launch_and_equal_the_dequant_dot():
    m = mx4()
    torch.manual_seed(1)
    w = torch.randn(200, K, dtype=torch.bfloat16)
    p = reg(m, w)
    for mm in (1, 5, 16):
        x = torch.randn(mm, K, dtype=torch.bfloat16)
        y = m.mxfp4_dense_linear(x, p)
        assert tuple(y.shape) == (mm, 200) and y.dtype == torch.bfloat16
        torch.testing.assert_close(y.float(), expected(m, x, w).float(), rtol=2**-7, atol=1e-2)
        c = m._mx.gemv_calls[-1]
        assert (c["M"], c["N"], c["K"], c["SX"], c["SY"], c["out32"], c["ntiles"]) == (mm, 200, K, K, 200, 0, 13)
    assert len(m._mx.gemv_calls) == 3 and not m._mx.dq_calls


def test_batched_leading_dims_and_bias():
    m = mx4()
    torch.manual_seed(2)
    w = torch.randn(64, K, dtype=torch.bfloat16)
    p = reg(m, w)
    x = torch.randn(2, 3, K, dtype=torch.bfloat16)
    bias = torch.randn(64, dtype=torch.bfloat16)
    y = m.mxfp4_dense_linear(x, p, bias=bias)
    assert tuple(y.shape) == (2, 3, 64)
    exp = (expected(m, x.reshape(6, K), w).float() + bias.float()).reshape(2, 3, 64)
    torch.testing.assert_close(y.float(), exp, rtol=2**-7, atol=1e-2)
    out = torch.empty(2, 3, 64, dtype=torch.bfloat16)
    assert m.mxfp4_dense_linear(x, p, out) is out


def test_strided_and_misaligned_activations_reach_the_kernel_legally():
    m = mx4()
    torch.manual_seed(3)
    w = torch.randn(64, K, dtype=torch.bfloat16)
    p = reg(m, w)
    wide = torch.randn(4, K + 8, dtype=torch.bfloat16)
    x = wide[:, :K]  # row stride 136: legal, no copy
    m.mxfp4_dense_linear(x, p)
    assert m._mx.gemv_calls[-1]["SX"] == K + 8
    odd = torch.randn(4 * K + 1, dtype=torch.bfloat16)[1:].reshape(4, K)  # data_ptr % 16 != 0
    y = m.mxfp4_dense_linear(odd, p)  # FakeMx4 asserts x % 16 == 0: the module must have copied
    torch.testing.assert_close(y.float(), expected(m, odd, w).float(), rtol=2**-7, atol=1e-2)
    ragged = torch.randn(4, K + 3, dtype=torch.bfloat16)[:, :K]  # row stride % 8 != 0 -> copy
    y = m.mxfp4_dense_linear(ragged, p)
    torch.testing.assert_close(y.float(), expected(m, ragged, w).float(), rtol=2**-7, atol=1e-2)


def test_prefill_rows_dequantize_to_a_bf16_gemm_not_a_launch_per_16_rows():
    m = mx4()
    torch.manual_seed(4)
    w = torch.randn(200, K, dtype=torch.bfloat16)
    p = reg(m, w)
    x = torch.randn(40, K, dtype=torch.bfloat16)
    y = m.mxfp4_dense_linear(x, p)
    assert not m._mx.gemv_calls and m._mx.dq_calls == [dict(N=200, K=K, ntiles=13)]
    packed, scale, _ = m.quantize_mxfp4(w)
    exp = torch.nn.functional.linear(x, m.dequant_mxfp4(packed, scale))
    torch.testing.assert_close(y.float(), exp.float(), rtol=2**-7, atol=1e-2)


def test_head_class_weights_chunk_every_m_through_the_gemv_and_can_return_fp32():
    m = mx4(RADIANCE_INKLING2_MXFP4_DENSE_HEAD_MIN_N="256")
    torch.manual_seed(5)
    w = torch.randn(512, K, dtype=torch.bfloat16)
    p = reg(m, w, free=False)
    x = torch.randn(40, K, dtype=torch.bfloat16)
    y = m.mxfp4_dense_head(p, x, torch.float32)
    assert y.dtype == torch.float32 and tuple(y.shape) == (40, 512)
    assert [c["M"] for c in m._mx.gemv_calls] == [16, 16, 8] and not m._mx.dq_calls
    assert all(c["out32"] == 1 and c["SY"] == 512 for c in m._mx.gemv_calls)
    torch.testing.assert_close(y, expected(m, x, w).float(), rtol=2**-7, atol=1e-2)


def test_row_slices_share_the_tiles_when_tile_aligned_else_get_their_own_copy():
    m = mx4()
    torch.manual_seed(6)
    w = torch.randn(256, K, dtype=torch.bfloat16)
    rows = ((0, 152), (32, 48), (149, 40), (8, 40))  # prefix, aligned middle, unaligned, half-tile offset
    p = reg(m, w, rows)
    ents = {k: v for k, v in m._CACHE.items()}
    main = ents[m._key(p.data)]
    pre, mid, odd, half = (ents[m._key(p.data[r0:r0 + n])] for r0, n in rows)
    assert pre["qt"].data_ptr() == main["qt"].data_ptr() and tuple(pre["qt"].shape)[0] == 10
    assert mid["qt"].data_ptr() == main["qt"].data_ptr() + 2 * 2 * 512  # tile 2, KG = 2 groups of 512 B
    assert odd["qt"].data_ptr() not in range(main["qt"].data_ptr(), main["qt"].data_ptr() + main["qt"].numel())
    assert half["qt"].data_ptr() not in range(main["qt"].data_ptr(), main["qt"].data_ptr() + main["qt"].numel())
    x = torch.randn(3, K, dtype=torch.bfloat16)
    for r0, n in rows:
        y = m.mxfp4_dense_linear(x, p.data[r0:r0 + n])
        torch.testing.assert_close(y.float(), expected(m, x, w[r0:r0 + n]).float(), rtol=2**-7, atol=1e-2)


def test_gemv_mode_never_resolves_the_w4a8_op_or_its_scratch(monkeypatch):
    m = mx4()
    m.DECODE_MAX_M = 16
    monkeypatch.setattr(m, "_resolve_op", lambda: pytest.fail("W4A8 op resolved in gemv mode"))
    monkeypatch.setattr(m, "_setup_scratch", lambda d: pytest.fail("decode scratch allocated in gemv mode"))
    from test_mxfp4dense import FakeTarget  # noqa: E402

    model = FakeTarget()
    m.process_target(model)
    assert m._CACHE


def test_missing_extension_falls_back_to_the_w4a8_route(monkeypatch, capsys):
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MIN_N="32")  # default mode
    monkeypatch.setitem(sys.modules, "radiance_inkling2_mx4", None)  # import raises ImportError
    m._linear_op = lambda *a: None  # the W4A8 op is "available"
    from test_mxfp4dense import FakeTarget  # noqa: E402

    m.MODE = "gemv"
    assert m._ready(FakeTarget()) is True
    assert m.MODE == "w4a8" and "radiance_inkling2_mx4" in capsys.readouterr().err
    m2 = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MIN_N="32")
    m2.MODE = "gemv"
    m2._mx = False
    m2._linear_op = False
    assert m2._ready(FakeTarget()) is False  # neither route: bf16 stays


def test_modes_are_the_documented_three():
    m = mx4(RADIANCE_INKLING2_MXFP4_DENSE_MODE="bogus")
    with pytest.raises(ValueError, match="MODE"):
        m._ready(types.SimpleNamespace(parameters=lambda: iter([torch.zeros(1)])))


@pytest.mark.parametrize("mode", ["w4a8", "w4a16"])
def test_other_modes_keep_the_old_entries(mode):
    m = load(RADIANCE_INKLING2_MXFP4_DENSE="1", RADIANCE_INKLING2_MXFP4_DENSE_MIN_N="32",
             RADIANCE_INKLING2_MXFP4_DENSE_MODE=mode)
    p = reg(m, torch.randn(64, K, dtype=torch.bfloat16))
    (ent,) = m._CACHE.values()
    assert {"packed", "scale", "wref"} <= set(ent) and "qt" not in ent


def test_load_prints_one_proof_line_with_mode_and_bytes(capsys):
    from test_mxfp4dense import FakeTarget  # noqa: E402

    m = mx4()
    m.process_target(FakeTarget())
    err = capsys.readouterr().err
    assert "[radiance.inkling2.mxfp4dense] MODE=gemv" in err and "weights" in err and "MiB" in err
    assert err.count("MODE=gemv") == 1
    seen = {}
    for e in m._CACHE.values():  # aligned slices are views of the same tiles: count each storage once
        seen.setdefault(e["qt"].data_ptr(), e)
    nbytes = sum(t.numel() * t.element_size() for e in seen.values() for t in e.values())
    assert f"{m._STATS[0]} weights registered after the target, {nbytes >> 20} MiB" in err
    assert m._STATS[1] == nbytes  # bytes of the tiles only, counted once per weight (the slices share them)
