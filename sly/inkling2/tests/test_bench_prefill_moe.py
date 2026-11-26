"""bench_prefill_moe.py: the parts that are not a GPU timing (routing layout, ranking, table file)."""
import json
import math

import pytest

import bench_prefill_moe as b


@pytest.mark.parametrize("tokens,tile", [(4096, 128), (700, 64), (9, 16), (16384, 256)])
def test_routing_layout_matches_the_plan_the_kernels_get(tokens, tile):
    r = b.make_routing(tokens, topk=8, experts=128, tile=tile, seed=1)
    rows = tokens * 8
    assert r.rows_real == rows
    # expert_plan: min(E, rows) + (rows - min(E, rows)) // tile CTAs is an upper bound of the real count
    assert r.count <= r.max_ctas == min(128, rows) + (rows - min(128, rows)) // tile
    assert len(r.tiles) == 2 * r.max_ctas + 2 and r.tiles[2 * r.max_ctas] == r.count
    ends = r.tiles[r.max_ctas: 2 * r.max_ctas]
    live = [e - t * tile for t, e in enumerate(ends[: r.count])]
    assert sum(live) == rows and all(1 <= v <= tile for v in live)       # every routed row once, in a tile
    ids = r.tiles[: r.count]
    for e in set(ids):                                                    # an expert's CTAs are consecutive
        pos = [i for i, x in enumerate(ids) if x == e]
        assert pos == list(range(pos[0], pos[0] + len(pos)))
        assert all(v == tile for v in [live[i] for i in pos[:-1]])        # only its last CTA is partial


def test_routing_is_reproducible_and_spreads_over_the_experts():
    a, c = b.make_routing(4096, 8, 128, 128, seed=3), b.make_routing(4096, 8, 128, 128, seed=3)
    assert a.tiles == c.tiles
    assert len(set(a.tiles[: a.count])) == 128                            # 256 rows/expert on average: all busy


def test_flops_count_real_rows_only():
    assert b.flops(rows=10, n=4, k=8) == 2 * 10 * 4 * 8


def test_pick_best_sums_the_projections_and_skips_configs_that_failed_on_one():
    a, c, d = dict(block_m=32), dict(block_m=64), dict(block_m=128)
    results = [(a, [10.0, 10.0]), (c, [4.0, 12.0]), (d, [1.0, None])]     # d did not run on the second shape
    cfg, total = b.pick_best(results)
    assert cfg == c and total == 16.0


def test_table_merge_keeps_other_tiles_and_formats(tmp_path):
    path = tmp_path / "t.json"
    b.merge_table(path, "nvfp4", 128, dict(block_m=64, num_warps=8))
    b.merge_table(path, "mxfp8", 128, dict(block_n=64))
    b.merge_table(path, "nvfp4", 256, dict(block_m=128))
    b.merge_table(path, "nvfp4", 128, dict(block_m=32))                   # replaces only its own entry
    assert json.loads(path.read_text()) == {"nvfp4": {"128": {"block_m": 32}, "256": {"block_m": 128}},
                                            "mxfp8": {"128": {"block_n": 64}}}


def test_the_written_table_is_accepted_by_the_kernel_module(tmp_path, monkeypatch):
    import radiance_inkling2_moe as m

    path = tmp_path / "t.json"
    b.merge_table(path, "nvfp4", 128, dict(block_m=64, block_n=128, block_k=64, num_warps=8, num_stages=2))
    monkeypatch.setenv(m._CFG_ENV, str(path))
    assert m.prefill_cfg("nvfp4", 128)["block_m"] == 64


def test_cpu_smoke_runs_the_whole_sweep_on_a_tiny_shape(tmp_path):
    out = tmp_path / "t.json"
    res = b.main(["--cpu-smoke", "--table-out", str(out)])
    assert res["nvfp4"]["best"] and res["nvfp4"]["default_us"] > 0
    assert json.loads(out.read_text())["nvfp4"]
    assert math.isfinite(res["nvfp4"]["max_rel_vs_default"]) and res["nvfp4"]["max_rel_vs_default"] < 2e-2
