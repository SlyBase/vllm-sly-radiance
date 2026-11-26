"""Host side of the HIP MoE expert GEMMs (CPU only, no GPU, no built module).

The wrapper in radiance_inkling2_moe.py decides WHETHER the HIP kernel runs (tile == 8, K within
the kernel's register budget, env knob) and checks every pointer extent before the launch: a GPU
memory fault has hard-reset the Proxmox host before, so the launch geometry is emulated here in
plain Python and compared with the tensor sizes.  The kernel itself is tested in test_moe_hip.py.
"""
import importlib
import sys
import types

import pytest
import torch

import test_moe_hip as gpu_tests

make_case = gpu_tests.make_case


class FakeHip:
    """Stand-in for the compiled module: records the launch, writes nothing."""

    def __init__(self):
        self.calls = []

    def nvfp4_gemm(self, *a):
        self.calls.append(("nvfp4", a))

    def mxfp8_gemm(self, *a):
        self.calls.append(("mxfp8", a))


@pytest.fixture()
def m(monkeypatch):
    import radiance_inkling2_moe as mod
    fake = FakeHip()
    monkeypatch.setattr(mod, "_hip", fake)
    monkeypatch.setattr(mod, "_stream", lambda: 0)
    return mod, fake


def nvfp4_args(c):
    return (c["act"], c["act_scale"], c["row_scale"], c["weight"], c["weight_scale"],
            c["expert_scale"], c["tiles"], c["plan"])


def mxfp8_args(c):
    return (c["act"], c["act_scale"], c["weight"], c["weight_scale"], c["expert_scale"],
            c["tiles"], c["plan"])


# ---- dispatch -------------------------------------------------------------------------------

def test_nvfp4_tile8_launches_hip_with_the_plan_geometry(m):
    mod, fake = m
    c = make_case("nvfp4", experts=8, n=64, k=256, tile=8, rows_per=[1, 2, 1, 1, 3, 1, 1, 1], seed=1)
    out = mod.nvfp4_expert_gemm(*nvfp4_args(c))
    (kind, a), = fake.calls
    assert kind == "nvfp4"
    ptrs = [c[k].data_ptr() for k in ("act", "act_scale", "row_scale", "weight", "weight_scale",
                                      "expert_scale")]
    assert list(a[:6]) == ptrs and a[6] == out.data_ptr() and a[7] == c["tiles"].data_ptr()
    # max_ctas, experts, N, K, tile, lds rows (max_ctas 8 = one token -> one row per CTA), stream
    assert a[8:] == (8, 8, 64, 256, 8, 1, 0)
    assert out.shape == (64, 64) and out.dtype == torch.bfloat16


def test_mxfp8_tile8_launches_hip_with_the_plan_geometry(m):
    mod, fake = m
    c = make_case("mxfp8", experts=4, n=64, k=256, tile=8, rows_per=[1] * 8, seed=2,
                  expert_ids=[0, 1, 2, 3, 0, 1, 2, 3])
    out = mod.mxfp8_expert_gemm(*mxfp8_args(c))
    (kind, a), = fake.calls
    ptrs = [c[k].data_ptr() for k in ("act", "act_scale", "weight", "weight_scale", "expert_scale")]
    assert kind == "mxfp8" and list(a[:5]) == ptrs
    assert a[5] == out.data_ptr() and a[6] == c["tiles"].data_ptr()
    assert a[7:] == (8, 4, 64, 256, 8, 1, 0)


@pytest.mark.parametrize("max_ctas,lds_rows", [(8, 1), (16, 2), (32, 4), (64, 8), (128, 8), (1, 1),
                                               (9, 2), (13, 2), (29, 4)])  # never round down
def test_lds_rows_follow_the_token_bound(m, max_ctas, lds_rows):
    """A CTA is one expert, an expert gets <= 1 row per token, tokens = max_ctas / top-k."""
    mod, _ = m
    assert mod._lds_rows(max_ctas) == lds_rows


def covered(tiles, max_ctas, tile):
    """(expert, row) pairs a launch computes: CTA c holds rows tile*c .. min(row_end, tile*c + tile)."""
    out = set()
    for c in range(int(tiles[2 * max_ctas])):
        lo = c * tile
        out |= {(int(tiles[c]), r) for r in range(lo, min(int(tiles[max_ctas + c]), lo + tile))}
    return out


@pytest.mark.parametrize("tile", [16, 24, 32, 64])
def test_tile_view_splits_a_multi_stream_tile_into_tile8_ctas(m, tile):
    """2-8 streams of spec-verify rows plan tile 16..64: the same rows, as tile-8 CTAs of the same expert."""
    mod, _ = m
    rows_per = [1, 5, tile, 3, tile - 1, 2]
    c = make_case("nvfp4", experts=6, n=64, k=128, tile=tile, rows_per=rows_per, count=5, seed=3)
    plan8, tiles8 = mod.hip_tile_view(c["plan"], c["tiles"])
    f = tile // 8
    assert (plan8.tile, plan8.max_ctas, plan8.rows) == (8, len(rows_per) * f, c["plan"].rows)
    assert tiles8.dtype == torch.int32 and tiles8.numel() == c["tiles"].numel() + 2 * len(rows_per) * (f - 1)
    assert covered(tiles8, plan8.max_ctas, 8) == covered(c["tiles"], c["plan"].max_ctas, tile)
    assert tiles8[2 * plan8.max_ctas + 1:].tolist() == c["tiles"][2 * len(rows_per) + 1:].tolist()  # padded rows (+ total): unchanged


def test_tile8_view_is_the_identity(m):
    mod, _ = m
    c = make_case("nvfp4", experts=2, n=64, k=128, tile=8, rows_per=[1, 5], seed=3)
    plan8, tiles8 = mod.hip_tile_view(c["plan"], c["tiles"])
    assert plan8 is c["plan"] and tiles8 is c["tiles"]


@pytest.mark.parametrize("tile", [16, 32])
@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_multi_stream_tiles_launch_the_hip_kernel_as_tile8(m, tile, fmt):
    mod, fake = m
    c = make_case(fmt, experts=2, n=64, k=256, tile=tile, rows_per=[1, 5], seed=3)
    out = (mod.nvfp4_expert_gemm if fmt == "nvfp4" else mod.mxfp8_expert_gemm)(
        *(nvfp4_args(c) if fmt == "nvfp4" else mxfp8_args(c)))
    (kind, a), = fake.calls
    f = tile // 8
    # ..., tiles ptr, max_ctas, experts, N, K, tile, lds rows, stream
    assert kind == fmt and a[-7:-3] == (2 * f, 2, 64, 256) and a[-3] == 8
    assert out.shape == (c["plan"].rows, 64)


def test_tiles_from_128_up_and_the_off_switch_stay_on_triton(m, monkeypatch):
    mod, fake = m
    assert not mod._hip_fits(types.SimpleNamespace(tile=128, max_ctas=8, rows=1024), 256, True)
    monkeypatch.setattr(mod, "_SPLIT", False)
    c = make_case("nvfp4", experts=2, n=64, k=128, tile=16, rows_per=[1, 5], seed=3)
    out = mod.nvfp4_expert_gemm(*nvfp4_args(c), dot_dtype=mod.tl.float32)
    assert fake.calls == []
    ref, valid = gpu_tests.reference(c), gpu_tests.valid_mask(c)
    torch.testing.assert_close(out[valid].float(), ref[valid], rtol=1e-2, atol=1e-2)  # bf16 output


def test_split_env_knob(monkeypatch):
    fake = types.ModuleType("radiance_inkling2_moe_hip")
    monkeypatch.setitem(sys.modules, "radiance_inkling2_moe_hip", fake)
    import radiance_inkling2_moe as mod
    try:
        monkeypatch.delenv("RADIANCE_INKLING2_HIP_MOE_SPLIT", raising=False)
        assert importlib.reload(mod)._SPLIT is True                    # default on
        monkeypatch.setenv("RADIANCE_INKLING2_HIP_MOE_SPLIT", "0")
        assert importlib.reload(mod)._SPLIT is False
    finally:
        monkeypatch.undo()
        importlib.reload(mod)


def test_k_beyond_the_kernel_register_budget_stays_on_triton(m):
    mod, fake = m
    c = make_case("nvfp4", experts=1, n=32, k=2368, tile=8, rows_per=[1], seed=4)
    mod.nvfp4_expert_gemm(*nvfp4_args(c), dot_dtype=mod.tl.float32)
    assert fake.calls == []


def test_env_knob_switches_hip_off(monkeypatch):
    fake = types.ModuleType("radiance_inkling2_moe_hip")
    monkeypatch.setitem(sys.modules, "radiance_inkling2_moe_hip", fake)
    import radiance_inkling2_moe as mod
    try:
        monkeypatch.delenv("RADIANCE_INKLING2_HIP_MOE", raising=False)
        assert importlib.reload(mod)._hip is fake                      # default on
        monkeypatch.setenv("RADIANCE_INKLING2_HIP_MOE", "0")
        assert importlib.reload(mod)._hip is None                      # '0' = Triton
    finally:
        monkeypatch.undo()
        importlib.reload(mod)


# ---- pointer extents --------------------------------------------------------------------------

SHAPES = [(n, k) for n, k in ((2304, 2304), (2304, 1152), (48, 192), (32, 64))]


@pytest.mark.parametrize("nvfp4", [True, False])
@pytest.mark.parametrize("n,k", SHAPES)
@pytest.mark.parametrize("max_ctas", [8, 29, 64])
def test_extents_of_the_real_shapes_fit_their_buffers(m, nvfp4, n, k, max_ctas):
    mod, _ = m
    if not nvfp4 and k % 128:
        pytest.skip("MXFP8 needs K % 128 == 0")
    e = 128
    ext = mod.launch_extents(nvfp4, rows=8 * max_ctas, max_ctas=max_ctas, experts=e, n=n, k=k)
    # bytes the kernel may touch of each buffer, never beyond its exact size
    wb = e * n * (k // 2 if nvfp4 else k)
    sb = e * n * (k // 16 if nvfp4 else k // 32)
    ab = 8 * max_ctas * (k // 2 if nvfp4 else k)
    asb = 8 * max_ctas * (k // 16 if nvfp4 else k // 32)
    assert ext["weight"] == wb and ext["weight_scale"] == sb       # the whole expert pool is read
    assert ext["act"] == ab and ext["act_scale"] == asb              # every row, every group
    assert ext["out"] == 8 * max_ctas * n * 2
    assert ext["tiles"] == (2 * max_ctas + 1) * 4
    assert ext["expert_scale"] == e * 4


def test_extent_emulation_sees_a_one_row_overshoot(m):
    """Mutation guard for the emulation itself: with N rounded up the weight pool is overrun."""
    mod, _ = m
    ok = mod.launch_extents(True, rows=64, max_ctas=8, experts=4, n=64, k=256)
    over = mod.launch_extents(True, rows=64, max_ctas=8, experts=4, n=64 + 1, k=256)
    assert over["weight"] > ok["weight"]


def test_wrapper_refuses_a_short_weight_scale_pool_before_the_launch(m):
    mod, fake = m
    c = make_case("nvfp4", experts=2, n=64, k=256, tile=8, rows_per=[1, 1], seed=5)
    short = c["weight_scale"][:, :-1].contiguous()     # one row of N missing
    with pytest.raises(ValueError, match="weight_scale"):
        mod.nvfp4_expert_gemm(c["act"], c["act_scale"], c["row_scale"], c["weight"], short,
                              c["expert_scale"], c["tiles"], c["plan"])
    assert fake.calls == []


def test_wrapper_refuses_fewer_act_rows_than_the_plan(m):
    mod, fake = m
    c = make_case("nvfp4", experts=2, n=64, k=256, tile=8, rows_per=[1, 1], seed=5)
    with pytest.raises(ValueError, match="act"):
        mod.nvfp4_expert_gemm(c["act"][:-8], c["act_scale"], c["row_scale"], c["weight"],
                              c["weight_scale"], c["expert_scale"], c["tiles"], c["plan"])
    assert fake.calls == []


def test_wrapper_refuses_an_undersized_act_scale(m):
    mod, fake = m
    c = make_case("mxfp8", experts=2, n=64, k=256, tile=8, rows_per=[1, 1], seed=6)
    with pytest.raises(ValueError, match="act_scale"):
        mod.mxfp8_expert_gemm(c["act"], c["act_scale"][:-4], c["weight"], c["weight_scale"],
                              c["expert_scale"], c["tiles"], c["plan"])
    assert fake.calls == []


def test_debug_mode_checks_the_tile_table(m, monkeypatch):
    """Needs the tiles on the host: expert ids inside the pool, row_end sane, count <= max_ctas."""
    mod, fake = m
    monkeypatch.setenv("RADIANCE_INKLING2_HIP_DEBUG", "1")
    c = make_case("nvfp4", experts=2, n=64, k=256, tile=8, rows_per=[1, 1], seed=7)
    mod.nvfp4_expert_gemm(*nvfp4_args(c))
    assert len(fake.calls) == 1
    bad = c["tiles"].clone()
    bad[0] = 2                                       # expert 2 of 2
    with pytest.raises(ValueError, match="expert"):
        mod.nvfp4_expert_gemm(c["act"], c["act_scale"], c["row_scale"], c["weight"],
                              c["weight_scale"], c["expert_scale"], bad, c["plan"])
    assert len(fake.calls) == 1
    bad = c["tiles"].clone()
    bad[2 * 2] = 3                                   # CTA count > max_ctas
    with pytest.raises(ValueError, match="count"):
        mod.nvfp4_expert_gemm(c["act"], c["act_scale"], c["row_scale"], c["weight"],
                              c["weight_scale"], c["expert_scale"], bad, c["plan"])


@pytest.mark.parametrize("field,bad,match", [
    ("tiles", lambda t: t.to(torch.int64), "tiles"),
    ("expert_scale", lambda t: t.to(torch.bfloat16), "expert_scale"),
    ("row_scale", lambda t: t.double(), "row_scale"),
    ("weight", lambda t: t.transpose(1, 2).contiguous().transpose(1, 2), "weight"),
])
def test_wrapper_refuses_a_wrong_dtype_or_layout(m, field, bad, match):
    """The kernel reinterprets raw pointers: dtype / layout are checked, not trusted."""
    mod, fake = m
    c = make_case("nvfp4", experts=2, n=64, k=256, tile=8, rows_per=[1, 1], seed=8)
    c[field] = bad(c[field])
    with pytest.raises(ValueError, match=match):
        mod.nvfp4_expert_gemm(*nvfp4_args(c))
    assert fake.calls == []


# ---- the patched vLLM tree routes MXFP8 experts to the wrapper -----------------------------------

def test_patched_bmm_routes_mxfp8_to_the_wrapper_on_rocm_only(tmp_path, monkeypatch):
    """Real patched moe_sm120.bmm: on ROCm the MXFP8 branch hands (act, act_scale, weight,
    weight_scale, ones, tiles, plan) to radiance_inkling2_moe.mxfp8_expert_gemm and returns its
    result; off ROCm it never imports it."""
    mod, restored = gpu_tests._patched_moe_sm120(tmp_path)
    seen = []
    fake = types.ModuleType("radiance_inkling2_moe")
    fake.mxfp8_expert_gemm = lambda *a: seen.append(a) or "routed"
    try:
        c = make_case("mxfp8", experts=2, n=64, k=256, tile=8, rows_per=[1, 1], seed=9)
        args = (c["act"], c["act_scale"], None, c["weight"], c["weight_scale"], c["expert_scale"],
                c["tiles"], c["plan"], 0)
        monkeypatch.setitem(sys.modules, "radiance_inkling2_moe", fake)
        monkeypatch.setattr(torch.version, "hip", "7.0")
        assert mod.bmm(*args) == "routed"
        (got,) = seen
        assert [id(x) for x in got] == [id(x) for x in (c["act"], c["act_scale"], c["weight"],
                                                         c["weight_scale"], c["expert_scale"],
                                                         c["tiles"], c["plan"])]
        monkeypatch.setattr(torch.version, "hip", None)
        seen.clear()
        mod._mxfp8_gemm = type("Stub", (), {"__getitem__": lambda self, g: (lambda *a, **k: None)})()
        # ^ CUDA path: vLLM's Triton kernel only
        mod.bmm(*args)
        assert seen == []
    finally:
        gpu_tests._restore(restored)


def test_hip_needs_k_multiples_the_scale_layout_allows(m):
    mod, _ = m
    plan = types.SimpleNamespace(tile=8, max_ctas=8, rows=64)
    assert mod._hip_fits(plan, 2304, True) and mod._hip_fits(plan, 2304, False)
    assert mod._hip_fits(plan, 64, True) and not mod._hip_fits(plan, 32, True)      # K % 64 (R8c4)
    assert mod._hip_fits(plan, 128, False) and not mod._hip_fits(plan, 192, False)  # K % 128
    assert not mod._hip_fits(plan, 2368, True)                                       # K <= 2304
    assert mod._hip_fits(types.SimpleNamespace(tile=16, max_ctas=8, rows=128), 256, True)   # split into tile 8
    assert not mod._hip_fits(types.SimpleNamespace(tile=20, max_ctas=8, rows=160), 256, True)  # not a multiple of 8
    assert not mod._hip_fits(types.SimpleNamespace(tile=128, max_ctas=8, rows=1024), 256, True)  # prefill


def test_wrapper_refuses_a_misaligned_activation(m):
    mod, fake = m
    c = make_case("nvfp4", experts=2, n=64, k=256, tile=8, rows_per=[1, 1], seed=10)
    flat = torch.cat([torch.zeros(1, dtype=torch.uint8), c["act"].reshape(-1)])
    c["act"] = flat[1:].reshape(c["act"].shape)       # 1 byte off a 16-byte boundary
    with pytest.raises(ValueError, match="aligned"):
        mod.nvfp4_expert_gemm(*nvfp4_args(c))
    assert fake.calls == []


@pytest.mark.parametrize("field,dtype", [("act", torch.float16), ("weight", torch.int16),
                                         ("weight_scale", torch.float32), ("act_scale", torch.int32)])
def test_wrapper_refuses_a_wider_item_type_even_if_the_bytes_suffice(m, field, dtype):
    """A 2- or 4-byte tensor of the same shape is big enough in bytes, but the kernel would read it as bytes."""
    mod, fake = m
    c = make_case("nvfp4", experts=2, n=64, k=256, tile=8, rows_per=[1, 1], seed=11)
    c[field] = torch.zeros(c[field].shape, dtype=dtype)
    with pytest.raises(ValueError, match=field):
        mod.nvfp4_expert_gemm(*nvfp4_args(c))
    assert fake.calls == []
