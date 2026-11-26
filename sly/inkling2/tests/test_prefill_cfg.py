"""Prefill launch config of the Triton expert GEMMs (tile > 8): default, JSON override, validation.

Inkling's expert GEMMs do not read vLLM's fused_moe config files (E=..,N=..,device_name=...json); the
launch shape is code. This is the equivalent: one place that maps (format, token tile) to the Triton
launch config, with a JSON table (RADIANCE_INKLING2_MOE_PREFILL_CFG) that a tuning sweep writes.
"""
import json
import types

import pytest
import torch

import test_nvfp4_expert_gemm as base

ENV = "RADIANCE_INKLING2_MOE_PREFILL_CFG"


class Spy:
    """Stands in for a @triton.jit kernel: records grid + launch kwargs instead of running."""

    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kw):
            self.calls.append((grid, kw))
        return launch


def launch_nvfp4(monkeypatch, tile, n=2304, k=2304, experts=1):
    import radiance_inkling2_moe as m

    spy = Spy()
    monkeypatch.setattr(m, "_expert_gemm", spy)
    monkeypatch.setattr(m, "_hip", None)
    monkeypatch.setenv("RADIANCE_INKLING2_MOE_PREFILL_V2", "0")  # these tests are about the v1 launch table
    plan = types.SimpleNamespace(tile=tile, max_ctas=4, rows=tile * 4)
    z = lambda *s, dt=torch.uint8: torch.zeros(*s, dtype=dt)
    m.nvfp4_expert_gemm(z(plan.rows, k // 2), z(1), z(plan.rows, dt=torch.float32),
                        z(experts, n, k // 2), z(experts, n, k // 16), z(experts, dt=torch.float32),
                        z(2 * plan.max_ctas + 1, dt=torch.int32), plan)
    assert len(spy.calls) == 1
    return spy.calls[0]


def test_default_is_the_tuned_table_for_tile_128(monkeypatch):
    """No env: the built-in table (PT arm, BM128/BN128/BK64, +64..71 % prefill, dNLL 0.0000)."""
    monkeypatch.delenv(ENV, raising=False)
    grid, kw = launch_nvfp4(monkeypatch, tile=128)
    assert (kw["BM"], kw["BN"], kw["BK"], kw["num_warps"], kw["num_stages"]) == (128, 128, 64, 4, 2)
    assert grid == (4, 1, 18)  # max_ctas, cdiv(128, 128), cdiv(2304, 128)


@pytest.mark.parametrize("off", ["0", "off", ""])
def test_env_off_is_the_old_hardcoded_tile(monkeypatch, off):
    monkeypatch.setenv(ENV, off)
    grid, kw = launch_nvfp4(monkeypatch, tile=128)
    assert (kw["BM"], kw["BN"], kw["BK"]) == (32, 64, 128)
    assert "num_warps" not in kw and "num_stages" not in kw
    assert grid == (4, 4, 36)  # max_ctas, cdiv(128, 32), cdiv(2304, 64)


def test_builtin_table_passes_the_validator(monkeypatch, tmp_path):
    import radiance_inkling2_moe as m

    cfg = tmp_path / "t.json"
    cfg.write_text(json.dumps(m._DEFAULT_TABLE))
    assert m._cfg_table(str(cfg)) == m._DEFAULT_TABLE


def test_json_table_overrides_the_tile_it_names(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"nvfp4": {"128": dict(block_m=64, block_n=128, block_k=64,
                                                      num_warps=8, num_stages=2)}}))
    monkeypatch.setenv(ENV, str(cfg))
    grid, kw = launch_nvfp4(monkeypatch, tile=128)
    assert (kw["BM"], kw["BN"], kw["BK"], kw["num_warps"], kw["num_stages"]) == (64, 128, 64, 8, 2)
    assert grid == (4, 2, 18)
    _, other = launch_nvfp4(monkeypatch, tile=64)  # a tile the file does not name keeps the default
    assert (other["BM"], other["BN"], other["BK"]) == (32, 64, 128)


@pytest.mark.parametrize("bad", [dict(block_m=48), dict(block_k=24), dict(num_warps=0), dict(bogus=1)])
def test_invalid_table_entry_is_refused(monkeypatch, tmp_path, bad):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"nvfp4": {"128": bad}}))
    monkeypatch.setenv(ENV, str(cfg))
    with pytest.raises(ValueError):
        launch_nvfp4(monkeypatch, tile=128)


def test_override_does_not_change_the_result(monkeypatch, tmp_path):
    import triton.language as tl
    from radiance_inkling2_moe import nvfp4_expert_gemm

    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"nvfp4": {"16": dict(block_m=16, block_n=32, block_k=64)}}))
    monkeypatch.setenv(ENV, str(cfg))
    c = base.make_case(experts=2, n=48, k=192, tile=16, seed=5)
    out = nvfp4_expert_gemm(c["act"], c["act_scale"], c["row_scale"], c["weight"], c["weight_scale"],
                            c["expert_scale"], c["tiles"], c["plan"], dot_dtype=tl.float32)
    ref = base.reference(c)
    valid = torch.zeros(c["plan"].rows, dtype=torch.bool)
    for t in range(c["plan"].max_ctas):
        valid[t * 16: min(int(c["row_end"][t]), (t + 1) * 16)] = True
    assert ref[valid].abs().max() > 0
    torch.testing.assert_close(out[valid].float(), ref[valid], rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("tile,hip,expect", [
    (16, True, dict(block_n=64, block_t=16, num_warps=4)),
    (128, True, dict(block_n=64, block_t=16, num_warps=4)),     # gfx12: 64 KB LDS shrink
    (16, False, dict(block_n=128, block_t=16, num_warps=4)),
    (64, False, dict(block_n=128, block_t=64, num_warps=4)),
    (256, False, dict(block_n=256, block_t=128, num_warps=8)),  # vLLM's own sm120 tiles
])
def test_mxfp8_default_launch_is_the_historical_one(monkeypatch, tile, hip, expect):
    import radiance_inkling2_moe as m

    monkeypatch.setenv(ENV, "0")
    got = m.mxfp8_launch(tile, hip)
    assert {k: got[k] for k in expect} == expect
    assert (got["block_k"], got["num_stages"]) == (128, 3)


def test_mxfp8_json_table_overrides_the_tile_it_names(monkeypatch, tmp_path):
    import radiance_inkling2_moe as m

    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"mxfp8": {"128": dict(block_n=128, block_t=32, num_warps=8, num_stages=2)}}))
    monkeypatch.setenv(ENV, str(cfg))
    got = m.mxfp8_launch(128, True)
    assert (got["block_n"], got["block_t"], got["num_warps"], got["num_stages"]) == (128, 32, 8, 2)
    assert m.mxfp8_launch(64, True)["block_t"] == 16  # untouched tile
