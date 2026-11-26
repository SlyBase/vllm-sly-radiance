"""HIP MoE expert GEMMs (radiance_inkling2_moe_hip): GPU correctness vs an independent torch
fp32 reference built from the raw nibbles / bytes (never via the kernel or its Python wrapper).

Skips cleanly without a GPU or without the built module (CI is CPU-only).  On a GPU host run with
TRITON_INTERPRET=0 so the Triton fallback arms compile for real:
    INKLING_VLLM_SRC=<site-packages>/vllm TRITON_INTERPRET=0 python -m pytest -q tests/test_moe_hip.py
The kernel launches with `hip.<fmt>_gemm(ptrs..., max_ctas, experts, N, K, tile, lds_rows, stream)`;
the output buffer is caller owned, with a NaN canary of 8 extra rows behind it.

Reference semantics (upstream ops/moe_sm120.py):
  NVFP4: bf16 [plan.rows, N] = (dequant(act) @ dequant(W[e])^T) * expert_scale[e] * row_scale[row]
  MXFP8: same without row_scale; e8m0 decode with the exponent==0 special case (2^-127).
Rows past a tile's row_end stay UNWRITTEN; tiles >= the CTA count do nothing.
"""
import os
import shutil
import sys
from pathlib import Path

import pytest
import torch
import types

import test_nvfp4_expert_gemm as base  # torch-only dequant / r8c4 helpers, independent of the kernel

E2M1 = base.E2M1


def e8m0_scale(u8):
    """e8m0 bytes -> fp32, including upstream's exponent==0 -> 2^-127 (0x00400000) special case."""
    e = u8.int()
    bits = torch.where(e == 0, torch.tensor(0x00400000, dtype=torch.int32), e << 23)
    return bits.view(torch.float32)


def deq_mxfp8(vals, scales):
    """[R, K] e4m3 bytes + [R, K/32] e8m0 -> [R, K] fp32."""
    return vals.view(torch.float8_e4m3fn).float() * e8m0_scale(scales).repeat_interleave(32, 1)


def r8c4_gen(scales):
    """Row-major [R, C] scales -> the flat R8c4 buffer vLLM's quantize kernels emit (TILE < 128)."""
    return base.r8c4(scales)


def make_case(fmt, experts=8, n=64, k=128, tile=8, rows_per=(1,), seed=0, count=None,
              expert_ids=None, zero_exp=False):
    """One MoE GEMM case.  `rows_per` = valid rows per CTA (cycled); `count` = active CTA count
    (default: all); `expert_ids` overrides the per-CTA expert (multiple tiles per expert)."""
    gen = torch.Generator().manual_seed(seed)
    max_ctas = len(rows_per) if isinstance(rows_per, (list, tuple)) else experts
    if isinstance(rows_per, int):
        rows_per = [rows_per] * max_ctas
    rows_per = list(rows_per)
    max_ctas = len(rows_per)
    count = max_ctas if count is None else count
    rows = tile * max_ctas
    if fmt == "nvfp4":
        w, ws = zip(*[base.rand_operand(n, k, gen) for _ in range(experts)])
        weight, wscale = torch.stack(w), torch.stack(ws)
        a, a_s = base.rand_operand(rows, k, gen)
        act, act_scale = a, r8c4_gen(a_s)
        row_scale = torch.rand(rows, generator=gen) + 0.5
    else:
        # e4m3 codes without NaN (0x7F / 0xFF); scales e8m0 around 1.0
        weight = torch.randint(0, 120, (experts, n, k), dtype=torch.uint8, generator=gen).view(torch.float8_e4m3fn)
        wscale = torch.randint(120, 135, (experts, n, k // 32), dtype=torch.uint8, generator=gen)
        act = torch.randint(0, 120, (rows, k), dtype=torch.uint8, generator=gen).view(torch.float8_e4m3fn)
        a_s = torch.randint(120, 135, (rows, k // 32), dtype=torch.uint8, generator=gen)
        if zero_exp:
            # ALL weight scale bytes 0 (-> 2^-127), activations 1.0 with scale 1.0: every product
            # is 2^-127, the sum ~ K * 2^-127 is a representable bf16 normal.  Decoding byte 0 as
            # 0.0 instead would zero the whole output (caught with atol=0 below).
            wscale.zero_()
            a_s.fill_(127)
            weight.fill_(0x38)  # e4m3 1.0
            act.fill_(0x38)
        act_scale = r8c4_gen(a_s)
        row_scale = None
    expert_scale = torch.rand(experts, generator=gen) + 0.5
    ids = list(range(max_ctas)) if expert_ids is None else list(expert_ids)
    row_end = [t * tile + rows_per[t] for t in range(max_ctas)]
    tiles = torch.tensor(ids + row_end + [count] + [sum(rows_per[:count])], dtype=torch.int32)
    # stale but plausible entries past the active count (a real plan buffer keeps the last step's
    # ids): a kernel without the count guard would compute and write these rows
    for t in range(count, max_ctas):
        tiles[t] = t % experts
        tiles[max_ctas + t] = t * tile + 2
    plan = types.SimpleNamespace(tile=tile, max_ctas=max_ctas, rows=rows)
    return dict(fmt=fmt, act=act, act_scale=act_scale, a_scale_rm=a_s, row_scale=row_scale,
                weight=weight, weight_scale=wscale, expert_scale=expert_scale, tiles=tiles,
                plan=plan, n=n, k=k, count=count)


def reference(c):
    """Independent fp32 reference from raw bytes; masked rows stay 0 (never written)."""
    plan = c["plan"]
    out = torch.zeros(plan.rows, c["n"])
    if c["fmt"] == "nvfp4":
        a = base.dequant(c["act"], c["a_scale_rm"])
    else:
        a = deq_mxfp8(c["act"], c["a_scale_rm"])
    for t in range(c["count"]):
        e = int(c["tiles"][t])
        if not 0 <= e < c["weight"].shape[0]:
            continue  # the kernel skips a CTA whose expert is outside the pool
        lo = t * plan.tile
        hi = min(int(c["tiles"][plan.max_ctas + t]), lo + plan.tile)
        if c["fmt"] == "nvfp4":
            w = base.dequant(c["weight"][e], c["weight_scale"][e])
        else:
            w = deq_mxfp8(c["weight"][e], c["weight_scale"][e])
        r = (a[lo:hi] @ w.T) * c["expert_scale"][e]
        if c["row_scale"] is not None:
            r = r * c["row_scale"][lo:hi, None]
        out[lo:hi] = r
    return out


def valid_mask(c):
    plan = c["plan"]
    valid = torch.zeros(plan.rows, dtype=torch.bool)
    for t in range(c["count"]):
        if not 0 <= int(c["tiles"][t]) < c["weight"].shape[0]:
            continue
        lo = t * plan.tile
        valid[lo:min(int(c["tiles"][plan.max_ctas + t]), lo + plan.tile)] = True
    return valid


@pytest.fixture(scope="module")
def hip():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    try:
        import radiance_inkling2_moe_hip as m
    except ImportError:
        pytest.skip("radiance_inkling2_moe_hip not built")
    return m


def to_gpu(c):
    return {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in c.items()}


CANARY_ROWS = 8


def run_hip(hip, c, lds_rows=8):
    """Low-level call with a caller-owned output buffer, so unwritten rows are observable.  Eight
    NaN rows follow the output: a kernel that writes past its last row is caught."""
    g = to_gpu(c)
    plan = g["plan"]
    buf = torch.full((plan.rows + CANARY_ROWS, g["n"]), float("nan"), dtype=torch.bfloat16,
                     device="cuda")
    out = buf[:plan.rows]
    stream = torch.cuda.current_stream().cuda_stream
    experts = g["weight"].shape[0]
    if g["fmt"] == "nvfp4":
        hip.nvfp4_gemm(g["act"].data_ptr(), g["act_scale"].data_ptr(), g["row_scale"].data_ptr(),
                       g["weight"].data_ptr(), g["weight_scale"].data_ptr(),
                       g["expert_scale"].data_ptr(), out.data_ptr(), g["tiles"].data_ptr(),
                       plan.max_ctas, experts, g["n"], g["k"], plan.tile, lds_rows, stream)
    else:
        hip.mxfp8_gemm(g["act"].data_ptr(), g["act_scale"].data_ptr(), g["weight"].data_ptr(),
                       g["weight_scale"].data_ptr(), g["expert_scale"].data_ptr(),
                       out.data_ptr(), g["tiles"].data_ptr(), plan.max_ctas, experts, g["n"],
                       g["k"], plan.tile, lds_rows, stream)
    torch.cuda.synchronize()
    assert torch.isnan(buf[plan.rows:]).all(), "the kernel wrote behind the output buffer"
    return out.cpu()


def check(hip, c, tight=False, lds_rows=8):
    out = run_hip(hip, c, lds_rows)
    ref = reference(c)
    valid = valid_mask(c)
    assert ref[valid].abs().max() > 0  # the comparison must not be vacuous
    # bf16 output; tight (atol=0) for the 2^-127-scale case, where atol would swallow everything
    # bf16 output: <= 0.4 % from rounding; a dropped scale group moves the sum by ~8 %
    torch.testing.assert_close(out[valid].float(), ref[valid], rtol=1e-2,
                               atol=0.0 if tight else 1e-2)
    assert torch.isnan(out[~valid]).all(), "rows past row_end / past the CTA count were written"


# ---- NVFP4 ---------------------------------------------------------------------------------

def test_00_smallest_cases_first(hip):
    """One block, one row: if the kernel faults, it faults here and not on a 2304 x 2304 case."""
    check(hip, make_case("nvfp4", experts=1, n=32, k=64, tile=8, rows_per=[1], seed=0))
    check(hip, make_case("mxfp8", experts=1, n=32, k=128, tile=8, rows_per=[1], seed=0))


def test_nvfp4_decode_one_row_per_expert(hip):
    check(hip, make_case("nvfp4", experts=8, n=2304, k=2304, tile=8, rows_per=1, seed=1))


def test_nvfp4_w2_shape(hip):
    check(hip, make_case("nvfp4", experts=8, n=2304, k=1152, tile=8, rows_per=1, seed=2))


@pytest.mark.parametrize("rows", [2, 4, 8])
def test_nvfp4_spec_verify_rows_per_expert(hip, rows):
    check(hip, make_case("nvfp4", experts=8, n=2304, k=2304, tile=8, rows_per=rows, seed=3))


def test_nvfp4_spec_verify_spread_16_ctas(hip):
    check(hip, make_case("nvfp4", n=2304, k=2304, tile=8, rows_per=[1] * 16, seed=4))


def test_nvfp4_masked_rows_stay_unwritten(hip):
    check(hip, make_case("nvfp4", n=128, k=256, tile=8, rows_per=[1, 3, 0, 8, 5, 2, 7, 4], seed=5))


def test_nvfp4_multiple_tiles_per_expert(hip):
    check(hip, make_case("nvfp4", experts=4, n=128, k=256, tile=8,
                         rows_per=[2, 3, 1, 4], expert_ids=[3, 0, 3, 1], seed=6))


def test_nvfp4_empty_ctas_do_nothing(hip):
    check(hip, make_case("nvfp4", n=128, k=256, tile=8, rows_per=[2, 1, 8, 3, 1, 1, 2, 1],
                         count=3, seed=7))


def test_nvfp4_ragged_n_and_k(hip):
    check(hip, make_case("nvfp4", experts=3, n=48, k=192, tile=8, rows_per=[1, 2, 3], seed=8))


def test_nvfp4_measured_spec_unions(hip):
    """2 / 4 / 8 tokens use ~13 / 20 / 29 distinct experts of 16 / 32 / 64 launched CTAs."""
    for tokens, ctas in ((2, 13), (4, 20), (8, 29)):
        n_cta = 8 * tokens
        rows = [(2 if i < (8 * tokens - ctas) else 1) if tokens < 8 else (3 if i < 6 else 2)
                for i in range(ctas)]
        rows += [0] * (n_cta - ctas)
        check(hip, make_case("nvfp4", experts=32, n=2304, k=2304, tile=8, rows_per=rows,
                             expert_ids=[i % 32 for i in range(n_cta)], count=ctas, seed=40 + tokens),
              lds_rows=max(1, tokens))


def test_nvfp4_more_rows_than_lds_rows_takes_extra_passes(hip):
    check(hip, make_case("nvfp4", n=128, k=256, tile=8, rows_per=[8, 5, 3, 1, 2, 6, 7, 4], seed=41),
          lds_rows=1)
    check(hip, make_case("nvfp4", n=128, k=256, tile=8, rows_per=[8, 5, 3, 1, 2, 6, 7, 4], seed=42),
          lds_rows=3)


def test_nvfp4_expert_outside_the_pool_is_skipped(hip):
    """A bad id inside the CTA count must not index out of the pool: the CTA writes nothing."""
    check(hip, make_case("nvfp4", experts=4, n=128, k=256, tile=8, rows_per=[2, 3, 1, 4],
                         expert_ids=[3, 77, -1, 1], seed=43))


def test_nvfp4_k_limits(hip):
    """K = 2304 (18 steps per lane, the maximum), 64 (one step, three idle lanes), 1152."""
    for k in (2304, 1152, 64):
        check(hip, make_case("nvfp4", experts=3, n=72, k=k, tile=8, rows_per=[1, 2, 3], seed=44))


def test_hip_rejects_what_it_cannot_run(hip):
    base = dict(experts=2, n=64, rows_per=[1, 1])
    for bad in (dict(tile=16, k=256), dict(tile=8, k=2368)):
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            run_hip(hip, make_case("nvfp4", **base, **bad))
    for lds_rows in (0, 9):
        with pytest.raises((ValueError, RuntimeError, TypeError)):
            run_hip(hip, make_case("nvfp4", tile=8, k=256, **base), lds_rows=lds_rows)


def test_geometry_matches_the_python_emulation(hip):
    import radiance_inkling2_moe as m
    assert dict(hip.geometry()) == m.GEOMETRY


# ---- MXFP8 ---------------------------------------------------------------------------------

def test_mxfp8_decode_one_row_per_expert(hip):
    check(hip, make_case("mxfp8", experts=8, n=2304, k=2304, tile=8, rows_per=1, seed=11))


def test_mxfp8_w2_shape(hip):
    check(hip, make_case("mxfp8", experts=8, n=2304, k=1152, tile=8, rows_per=1, seed=12))


@pytest.mark.parametrize("rows", [2, 8])
def test_mxfp8_spec_verify_rows_per_expert(hip, rows):
    check(hip, make_case("mxfp8", experts=8, n=2304, k=2304, tile=8, rows_per=rows, seed=13))


def test_mxfp8_masked_rows_and_empty_ctas(hip):
    check(hip, make_case("mxfp8", n=128, k=256, tile=8, rows_per=[1, 3, 0, 8, 5, 2, 7, 4],
                         count=6, seed=15))


def test_mxfp8_multiple_tiles_per_expert(hip):
    check(hip, make_case("mxfp8", experts=4, n=128, k=256, tile=8,
                         rows_per=[2, 3, 1, 4], expert_ids=[2, 2, 0, 1], seed=16))


def test_mxfp8_e8m0_zero_exponent_special_case(hip):
    """exponent byte 0 decodes to 2^-127 (0x00400000), not 0.0 (upstream _dequant_e4m3)."""
    check(hip, make_case("mxfp8", experts=3, n=128, k=256, tile=8, rows_per=[1, 2, 3],
                         seed=17, zero_exp=True), tight=True)


def test_mxfp8_ragged_n(hip):
    check(hip, make_case("mxfp8", experts=3, n=48, k=384, tile=8, rows_per=[1, 5, 8], seed=18))


def test_mxfp8_measured_spec_unions(hip):
    for tokens, ctas in ((2, 13), (4, 20), (8, 29)):
        n_cta = 8 * tokens
        rows = [(2 if i < (8 * tokens - ctas) else 1) if tokens < 8 else (3 if i < 6 else 2)
                for i in range(ctas)]
        rows += [0] * (n_cta - ctas)
        check(hip, make_case("mxfp8", experts=32, n=2304, k=2304, tile=8, rows_per=rows,
                             expert_ids=[i % 32 for i in range(n_cta)], count=ctas, seed=50 + tokens),
              lds_rows=max(1, tokens))


def test_mxfp8_more_rows_than_lds_rows_takes_extra_passes(hip):
    check(hip, make_case("mxfp8", n=128, k=256, tile=8, rows_per=[8, 5, 3, 1, 2, 6, 7, 4], seed=51),
          lds_rows=1)


def test_mxfp8_expert_outside_the_pool_is_skipped(hip):
    check(hip, make_case("mxfp8", experts=4, n=128, k=256, tile=8, rows_per=[2, 3, 1, 4],
                         expert_ids=[3, 77, -1, 1], seed=52))


def test_mxfp8_k_limits(hip):
    for k in (2304, 1152, 128):
        check(hip, make_case("mxfp8", experts=3, n=72, k=k, tile=8, rows_per=[1, 2, 3], seed=53))


# ---- Python dispatch ------------------------------------------------------------------------

def test_wrapper_uses_hip_and_matches_reference(hip):
    import radiance_inkling2_moe as m
    assert m._hip is hip
    c = make_case("nvfp4", experts=8, n=2304, k=2304, tile=8, rows_per=1, seed=21)
    g = to_gpu(c)
    out = m.nvfp4_expert_gemm(g["act"], g["act_scale"], g["row_scale"], g["weight"],
                              g["weight_scale"], g["expert_scale"], g["tiles"], g["plan"])
    torch.cuda.synchronize()
    ref, valid = reference(c), valid_mask(c)
    torch.testing.assert_close(out.cpu()[valid].float(), ref[valid], rtol=1e-2, atol=1e-2)
    c8 = make_case("mxfp8", experts=8, n=2304, k=2304, tile=8, rows_per=2, seed=22)
    g8 = to_gpu(c8)
    out8 = m.mxfp8_expert_gemm(g8["act"], g8["act_scale"], g8["weight"], g8["weight_scale"],
                               g8["expert_scale"], g8["tiles"], g8["plan"])
    torch.cuda.synchronize()
    ref8, valid8 = reference(c8), valid_mask(c8)
    torch.testing.assert_close(out8.cpu()[valid8].float(), ref8[valid8], rtol=1e-2, atol=1e-2)


def _wrapper_call(m, g):
    if g["fmt"] == "nvfp4":
        return m.nvfp4_expert_gemm(g["act"], g["act_scale"], g["row_scale"], g["weight"],
                                   g["weight_scale"], g["expert_scale"], g["tiles"], g["plan"])
    return m.mxfp8_expert_gemm(g["act"], g["act_scale"], g["weight"], g["weight_scale"],
                               g["expert_scale"], g["tiles"], g["plan"])


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
def test_wrapper_is_graph_capturable_and_follows_the_device_tile_table(hip, fmt):
    """Decode runs in a CUDA graph: the kernel must capture (no host sync / alloc inside) and the
    CTA -> expert map is read from the DEVICE tile table at replay time, not frozen at capture."""
    import radiance_inkling2_moe as m
    a = make_case(fmt, experts=8, n=128, k=256, tile=8, rows_per=[1, 2, 1, 3], expert_ids=[0, 1, 2, 3],
                  seed=60)
    b = make_case(fmt, experts=8, n=128, k=256, tile=8, rows_per=[2, 1, 3, 1], expert_ids=[7, 6, 5, 4],
                  seed=60)  # same weights / acts (same seed), another routing
    ga = to_gpu(a)
    for k in ("weight", "weight_scale", "act", "act_scale", "expert_scale"):
        assert torch.equal(a[k].view(torch.uint8) if a[k].dtype == torch.float8_e4m3fn else a[k],
                           b[k].view(torch.uint8) if b[k].dtype == torch.float8_e4m3fn else b[k])
    _wrapper_call(m, ga)  # warm up outside the capture
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _wrapper_call(m, ga)
    for case in (a, b, a):
        ga["tiles"].copy_(case["tiles"])  # new routing, same graph
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        ref, valid = reference(case), valid_mask(case)
        torch.testing.assert_close(out.cpu()[valid].float(), ref[valid], rtol=1e-2, atol=1e-2)
        assert torch.isnan(out.cpu()[~valid]).all()


def test_wrapper_debug_mode_on_the_gpu(hip, monkeypatch):
    import radiance_inkling2_moe as m
    monkeypatch.setenv("RADIANCE_INKLING2_HIP_DEBUG", "1")
    c = make_case("nvfp4", experts=4, n=128, k=256, tile=8, rows_per=[1, 2, 1, 3], seed=61)
    g = to_gpu(c)
    out = _wrapper_call(m, g)
    torch.cuda.synchronize()
    ref, valid = reference(c), valid_mask(c)
    torch.testing.assert_close(out.cpu()[valid].float(), ref[valid], rtol=1e-2, atol=1e-2)
    g["tiles"][1] = 4  # expert 4 of 4: refused on the host, nothing is launched
    with pytest.raises(ValueError, match="expert"):
        _wrapper_call(m, g)


def test_wrapper_dispatch_by_tile(hip, monkeypatch):
    """tile 8 -> HIP; every other tile stays on Triton.  Count calls through the real module."""
    import radiance_inkling2_moe as m
    calls = []
    real = m._hip.nvfp4_gemm

    def spy(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(m._hip, "nvfp4_gemm", spy)
    c = make_case("nvfp4", experts=2, n=128, k=256, tile=8, rows_per=1, seed=23)
    g = to_gpu(c)
    m.nvfp4_expert_gemm(g["act"], g["act_scale"], g["row_scale"], g["weight"],
                        g["weight_scale"], g["expert_scale"], g["tiles"], g["plan"])
    assert len(calls) == 1
    c32 = make_case("nvfp4", experts=2, n=128, k=256, tile=32, rows_per=[1, 2], seed=24)
    g32 = to_gpu(c32)
    out = m.nvfp4_expert_gemm(g32["act"], g32["act_scale"], g32["row_scale"], g32["weight"],
                              g32["weight_scale"], g32["expert_scale"], g32["tiles"], g32["plan"])
    torch.cuda.synchronize()
    assert len(calls) == 1  # tile 32 did NOT take the HIP path
    ref, valid = reference(c32), valid_mask(c32)
    torch.testing.assert_close(out.cpu()[valid].float(), ref[valid], rtol=1e-2, atol=1e-2)


def test_env_off_falls_back_to_triton(hip, monkeypatch):
    """RADIANCE_INKLING2_HIP_MOE=0 (module attr _hip=None) -> the existing Triton kernel."""
    import radiance_inkling2_moe as m
    monkeypatch.setattr(m, "_hip", None)
    # (the MXFP8 Triton arm is exercised on the GPU by bench_experts.py, which compares its output
    #  with the HIP kernel's at the real shapes; its import needs vLLM's real package, not the shim)
    c = make_case("nvfp4", experts=4, n=128, k=256, tile=8, rows_per=[1, 2, 3, 4], seed=25)
    g = to_gpu(c)
    out = m.nvfp4_expert_gemm(g["act"], g["act_scale"], g["row_scale"], g["weight"],
                              g["weight_scale"], g["expert_scale"], g["tiles"], g["plan"])
    torch.cuda.synchronize()
    ref, valid = reference(c), valid_mask(c)
    torch.testing.assert_close(out.cpu()[valid].float(), ref[valid], rtol=1e-2, atol=1e-2)


# ---- CPU: the MXFP8 Triton fallback (no HIP module needed) ------------------------------------

def _patched_moe_sm120(tmp_path):
    """Inject the PATCHED vLLM tree's moe_sm120 under its importable name, with the two
    interpreter-only adaptations (bf16 dot/store mis-computed by the Triton CPU interpreter;
    values stay fp32-exact there).  Returns the module plus restorers for sys.modules."""
    import importlib.util

    import test_patch_gfx12 as pt

    src = os.environ.get("INKLING_VLLM_SRC")
    if not src or not Path(src, "models/inkling2").exists():
        pytest.skip("INKLING_VLLM_SRC not set")
    root = Path(src)
    shutil.copytree(root, tmp_path / "vllm", ignore=shutil.ignore_patterns("__pycache__"))
    r = pt.run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    tree_root = tmp_path / "vllm"

    def shim_init(dotted, init_file):
        mod_ = types.ModuleType(dotted)
        mod_.__file__ = str(init_file)
        mod_.__path__ = [str(init_file.parent)]
        exec(init_file.read_text(), mod_.__dict__)
        sys.modules[dotted] = mod_
        return mod_

    shim_init("vllm.models.inkling2.nvidia.ops",
              tree_root / "models/inkling2/nvidia/ops/__init__.py")
    plt = types.ModuleType("vllm.platforms")
    plt.__path__ = [str(tree_root / "platforms")]
    old_platforms = sys.modules.get("vllm.platforms")
    sys.modules["vllm.platforms"] = plt
    old_ops = sys.modules.get("vllm.models.inkling2.nvidia.ops")

    name = "vllm.models.inkling2.nvidia.ops.moe_sm120"
    spec = importlib.util.spec_from_file_location(
        name, tree_root / "models/inkling2/nvidia/ops/moe_sm120.py")
    mod = importlib.util.module_from_spec(spec)
    old = sys.modules.get(name)
    sys.modules[name] = mod
    src_text = Path(spec.origin).read_text() if spec.origin else ""
    old_ret = "return (values.to(tl.float32) * scales).to(tl.bfloat16)"
    assert src_text.count(old_ret) == 1
    old_store = "        acc.to(tl.bfloat16),\n"
    assert src_text.count(old_store) == 1
    exec(compile(src_text.replace(old_ret, "return values.to(tl.float32) * scales")
                            .replace(old_store, "        acc,\n"),
                 spec.origin, "exec"), mod.__dict__)
    return mod, (old, old_ops, old_platforms, name)


def _restore(restored):
    old, old_ops, old_platforms, name = restored
    if old is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = old
    if old_ops is None:
        sys.modules.pop("vllm.models.inkling2.nvidia.ops", None)
    else:
        sys.modules["vllm.models.inkling2.nvidia.ops"] = old_ops
    if old_platforms is None:
        sys.modules.pop("vllm.platforms", None)
    else:
        sys.modules["vllm.platforms"] = old_platforms


@pytest.mark.parametrize("rocm", [False, True])
def test_mxfp8_fallback_forwards_tensors_and_the_platform_tiles_cpu(tmp_path, monkeypatch, rocm):
    """Without the HIP module, mxfp8_expert_gemm forwards the right tensors, grid and launch
    arguments to vLLM's _mxfp8_gemm (BN 128 on CUDA, 64 on ROCm: 64 KB LDS), and its output matches
    the independent fp32 reference computed from the same tensors.  Stub kernel: no Triton launch."""
    mod, restored = _patched_moe_sm120(tmp_path)
    captured = {}

    def stub(*a, **kw):
        captured["args"] = a
        captured["kw"] = dict(kw)
        ref, valid = reference(c), valid_mask(c)
        a[5][valid] = ref[valid].to(torch.bfloat16)  # write into the wrapper's output buffer

    class StubKernel:
        def __getitem__(self, grid):
            captured["grid"] = tuple(grid)
            return stub

    mod._mxfp8_gemm = StubKernel()
    sys.modules["vllm.models.inkling2.nvidia.ops.moe_sm120"] = mod
    try:
        import radiance_inkling2_moe as m
        monkeypatch.setattr(m, "_hip", None)
        monkeypatch.setattr(torch.version, "hip", "7.0" if rocm else None)
        c = make_case("mxfp8", experts=3, n=64, k=256, tile=8, rows_per=[1, 2, 3], seed=31)
        out = m.mxfp8_expert_gemm(c["act"], c["act_scale"], c["weight"], c["weight_scale"],
                                  c["expert_scale"], c["tiles"], c["plan"])
        ref, valid = reference(c), valid_mask(c)
        assert ref[valid].abs().max() > 0
        bn = 64 if rocm else 128
        assert captured["grid"] == (-(-64 // bn), 3)   # N blocks, 3 CTAs x cdiv(8, 16) = 1
        assert captured["kw"] == {
            "MAX_CTAS": 3, "N": 64, "K": 256, "TILE": 8, "BN": bn, "BT": 16, "BK": 128,
            "num_warps": 4, "num_stages": 3,
        }
        act, act_scale, weight, weight_scale, expert_scale, got_out, tiles = captured["args"]
        u8 = lambda t: t.view(torch.uint8)
        assert u8(act).equal(u8(c["act"])) and u8(weight).equal(u8(c["weight"]))
        assert torch.equal(expert_scale, c["expert_scale"]) and torch.equal(tiles, c["tiles"])
        assert got_out.data_ptr() == out.data_ptr()  # the wrapper's buffer is used
        torch.testing.assert_close(out[valid].float(), ref[valid], rtol=1e-2, atol=1e-2)
    finally:
        _restore(restored)


@pytest.mark.parametrize("fmt", ["nvfp4", "mxfp8"])
@pytest.mark.parametrize("tile", [16, 32, 64])
def test_split_big_tile_matches_the_big_tile_reference(hip, fmt, tile):
    """2-8 streams plan tile 16..64: the kernel runs them as tile-8 CTAs (`hip_tile_view`); the result is the
    big tile's reference, rows past a row end stay unwritten."""
    import radiance_inkling2_moe as mod
    rows_per = [1, 5, tile, 3, tile - 1, 2, 9, 8][:8]
    c = make_case(fmt, experts=8, n=2304 if tile == 16 else 256, k=2304 if tile == 16 else 256, tile=tile,
                  rows_per=rows_per, count=7, seed=11)
    plan8, tiles8 = mod.hip_tile_view(c["plan"], c["tiles"])
    out = run_hip(hip, dict(c, plan=plan8, tiles=tiles8))
    ref, valid = reference(c), valid_mask(c)
    assert ref[valid].abs().max() > 0
    torch.testing.assert_close(out[valid].float(), ref[valid], rtol=1e-2, atol=1e-2)
    assert torch.isnan(out[~valid]).all()
