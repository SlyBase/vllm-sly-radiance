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


@pytest.mark.parametrize("tile", [16, 32])
def test_other_tiles_stay_on_triton(m, tile):
    mod, fake = m
    c = make_case("nvfp4", experts=2, n=64, k=128, tile=tile, rows_per=[1, 5], seed=3)
    out = mod.nvfp4_expert_gemm(*nvfp4_args(c), dot_dtype=mod.tl.float32)
    assert fake.calls == []
    ref, valid = gpu_tests.reference(c), gpu_tests.valid_mask(c)
    torch.testing.assert_close(out[valid].float(), ref[valid], rtol=1e-2, atol=1e-2)  # bf16 output


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
    assert not mod._hip_fits(types.SimpleNamespace(tile=16, max_ctas=8, rows=128), 256, True)


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
