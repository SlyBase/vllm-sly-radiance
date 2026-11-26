"""Prefill NVFP4 expert GEMM v2: cheap e2m1 decode, activations dequantised once, nibbles split.

Same oracle as test_nvfp4_expert_gemm (dequantise in torch, fp32 matmul, outer scales). The
interpreter mis-computes bf16 dots, so the kernels run with dot_dtype=float32 (same values).
"""
import types

import pytest
import torch
import triton
import triton.language as tl

import test_nvfp4_expert_gemm as base

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
ENV = "RADIANCE_INKLING2_MOE_PREFILL_V2"


@triton.jit
def _decode_all(Code, Out, SAFE: tl.constexpr):
    from radiance_inkling2_moe import _fp4_val

    i = tl.arange(0, 16)
    tl.store(Out + i, _fp4_val(tl.load(Code + i), SAFE))


@pytest.mark.parametrize("safe", [False, True])
def test_fp4_val_decodes_all_16_codes(safe):
    code = torch.arange(16, dtype=torch.int32)
    out = torch.empty(16)
    _decode_all[(1,)](code, out, SAFE=safe)
    want = torch.tensor(E2M1 + [-v for v in E2M1])
    assert torch.equal(out, want)


def run_v2(c, safe=False, **launch):  # launch may carry v2_gs=True (scale loaded once per 16-k group)
    from radiance_inkling2_moe import nvfp4_expert_gemm

    return nvfp4_expert_gemm(c["act"], c["act_scale"], c["row_scale"], c["weight"], c["weight_scale"],
                             c["expert_scale"], c["tiles"], c["plan"], dot_dtype=tl.float32,
                             v2=True, v2_safe=safe, **launch)


def compare(c, out):
    ref = base.reference(c)
    valid = torch.zeros(c["plan"].rows, dtype=torch.bool)
    for t in range(c["plan"].max_ctas):
        valid[t * c["plan"].tile: min(int(c["row_end"][t]), (t + 1) * c["plan"].tile)] = True
    assert ref[valid].abs().max() > 0
    torch.testing.assert_close(out[valid].float(), ref[valid], rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("safe", [False, True])
def test_v2_matches_reference(safe):
    c = base.make_case(experts=2, n=48, k=192, tile=16, seed=3)  # N ragged vs BN, K = 3 x BK 64
    compare(c, run_v2(c, safe, block_m=16, block_n=32, block_k=64))


def test_v2_group_scale_variant_matches_reference():
    c = base.make_case(experts=2, n=48, k=192, tile=16, seed=11)
    compare(c, run_v2(c, False, block_m=16, block_n=32, block_k=64, v2_gs=True))


def test_v2_two_row_blocks_per_tile():
    c = base.make_case(experts=2, n=32, k=128, tile=32, seed=7)  # tile 32 = 2 x BM 16, short last tile
    compare(c, run_v2(c, False, block_m=16, block_n=32, block_k=64))


def test_v2_one_hot_activation_selects_one_weight_column():
    c = base.make_case(experts=2, n=32, k=128, tile=16, seed=1)
    c["act"].zero_()
    c["act"][:, 1] = 0x04  # column 2 (odd/even split must not move it): code 4 = 2.0 in the low nibble of byte 1
    compare(c, run_v2(c, False, block_m=16, block_n=32, block_k=64))


def test_v2_leaves_rows_past_row_end_alone_and_skips_idle_tiles():
    c = base.make_case(experts=2, n=32, k=128, tile=16, seed=2)
    c["tiles"][-1] = 1  # one active CTA: tile 1 must not be touched
    out = run_v2(c, False, block_m=16, block_n=32, block_k=64)
    ref = base.reference({**c, "plan": types.SimpleNamespace(**{**vars(c["plan"]), "max_ctas": 1})})
    torch.testing.assert_close(out[:16].float(), ref[:16], rtol=2e-2, atol=2e-2)


def test_v2_falls_back_when_k_is_not_a_multiple_of_bk():
    c = base.make_case(experts=2, n=32, k=192, tile=16, seed=4)  # 192 % 128 != 0 -> v1 kernel
    compare(c, run_v2(c, False, block_m=16, block_n=32, block_k=128))


def test_v2_default_is_on_for_prefill_tiles_only(monkeypatch):
    import radiance_inkling2_moe as m

    monkeypatch.delenv(ENV, raising=False)
    assert m.prefill_v2(128) and m.prefill_v2(256)
    assert not m.prefill_v2(64) and not m.prefill_v2(16)  # decode batches (tile < 128) skip the extra launch
    monkeypatch.setenv(ENV, "0")
    assert not m.prefill_v2(128)
    monkeypatch.setenv(ENV, "1")
    assert m.prefill_v2(16)


class Boom:
    """Stands in for the v1 kernel: any launch fails the test."""

    def __getitem__(self, grid):
        raise AssertionError("v1 kernel launched")


def test_v2_path_never_launches_v1(monkeypatch):
    import radiance_inkling2_moe as m

    monkeypatch.setattr(m, "_expert_gemm", Boom())
    c = base.make_case(experts=2, n=32, k=128, tile=16, seed=5)
    compare(c, run_v2(c, False, block_m=16, block_n=32, block_k=64))


def test_v2_off_stays_on_v1(monkeypatch):
    import radiance_inkling2_moe as m

    monkeypatch.setenv(ENV, "0")
    monkeypatch.setattr(m, "_expert_gemm_v2", Boom())
    c = base.make_case(experts=2, n=32, k=128, tile=16, seed=5)
    base.run_and_compare(c)


class Spy:
    def __init__(self):
        self.kw = None

    def __getitem__(self, grid):
        def launch(*args, **kw):
            self.kw = kw
        return launch


@pytest.mark.parametrize("gs", [False, True])
def test_group_scale_flag_reaches_the_kernel(monkeypatch, gs):
    import radiance_inkling2_moe as m

    spy = Spy()
    monkeypatch.setattr(m, "_expert_gemm_v2", spy)
    c = base.make_case(experts=2, n=32, k=128, tile=16, seed=5)
    run_v2(c, False, block_m=16, block_n=32, block_k=64, v2_gs=gs)
    assert spy.kw["GS"] is gs


def test_default_tile_128_launches_v2_with_the_measured_config(monkeypatch):
    """No env, tile 128: v2 with 128x128x32, 4 warps, 2 stages, per-group scale (mb3: 9.14 ms vs 19.7 ms v1 table)."""
    import radiance_inkling2_moe as m

    monkeypatch.delenv(ENV, raising=False)
    monkeypatch.delenv("RADIANCE_INKLING2_MOE_PREFILL_CFG", raising=False)
    monkeypatch.setattr(m, "_hip", None)
    spy, boom = Spy(), Boom()
    monkeypatch.setattr(m, "_expert_gemm_v2", spy)
    monkeypatch.setattr(m, "_dequant_act", Spy())
    monkeypatch.setattr(m, "_expert_gemm", boom)
    plan = types.SimpleNamespace(tile=128, max_ctas=4, rows=512)
    z = lambda *sh, dt=torch.uint8: torch.zeros(*sh, dtype=dt)
    m.nvfp4_expert_gemm(z(512, 1152), z(1), z(512, dt=torch.float32), z(1, 2304, 1152), z(1, 2304, 144),
                        z(1, dt=torch.float32), z(9, dt=torch.int32), plan)
    kw = spy.kw
    assert (kw["BM"], kw["BN"], kw["BK"], kw["num_warps"], kw["num_stages"], kw["GS"]) == (128, 128, 32, 4, 2, True)
