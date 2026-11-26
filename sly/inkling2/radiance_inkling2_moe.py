"""Inkling 2 expert GEMMs for gfx12 (RDNA4): a hand-written HIP kernel for the decode / spec-verify
tile (plan.tile == 8) and vLLM's Triton kernels (with an NVFP4 dequant GEMM, gfx12 has no FP4 MMA)
for everything else.

Layouts are vLLM's (models/inkling2/nvidia/ops/moe_sm120.py): packed uint8 with the even column in
the low nibble, e4m3 weight scales row-major [E, N, K/16] (NVFP4), e8m0 scales per 32 columns
[E, N, K/32] (MXFP8), activation scales in the tile-padded R8c4 layout (`_scale_offsets`),
`tiles` = [expert per CTA | row end per CTA | CTA count | padded rows].  Output bf16
[plan.rows, N]; rows past a tile's `row_end` are left unwritten; tiles >= the CTA count do nothing.

HIP path (radiance_inkling2_moe.hip): env RADIANCE_INKLING2_HIP_MOE default on, '0' = Triton.
RADIANCE_INKLING2_HIP_DEBUG=1 also reads the tile table back (host sync, never while a graph is
captured) and rejects ids the kernel would index out of the weight pool.
"""
import functools
import json
import os

import torch
from vllm.models.inkling2.nvidia.ops.block_scales import _scale_offsets
from vllm.triton_utils import tl, triton

try:
    import radiance_inkling2_moe_hip as _hip
except ImportError:
    _hip = None
if os.environ.get("RADIANCE_INKLING2_HIP_MOE") == "0":
    _hip = None

_HIP_TILE = 8  # decode and spec-verify steps (<= 8 tokens); bigger prefill tiles stay on Triton
_TOPK = 8      # experts per token: an expert (= one CTA) gets at most one row per token
# Geometry of radiance_inkling2_moe.hip (test_moe_hip.py asserts it equals hip.geometry()):
# BN weight rows per block, LPR lanes per row, a lane streams `quantum` k-values per step and at
# most NJ_MAX steps (K <= K_MAX), a CTA holds at most MAX_ROWS rows.
GEOMETRY = dict(bn=32, lpr=8, quantum=32, nj_max=9, max_rows=8, k_max=2304)


# Prefill / extend tiles (plan.tile > 8) run the Triton kernels below. Inkling does not read vLLM's
# fused_moe config files; this table is their equivalent: {"nvfp4"|"mxfp8": {"<tile>": {key: value}}},
# a JSON file named by RADIANCE_INKLING2_MOE_PREFILL_CFG (written by bench_prefill_moe.py --table-out).
# Without an entry the launch is the historical one.
_CFG_ENV = "RADIANCE_INKLING2_MOE_PREFILL_CFG"
_V2_ENV = "RADIANCE_INKLING2_MOE_PREFILL_V2"
_CFG_KEYS = dict(nvfp4=("block_m", "block_n", "block_k", "num_warps", "num_stages"),
                 mxfp8=("block_n", "block_t", "block_k", "num_warps", "num_stages"))
_POW2 = ("block_m", "block_n", "block_t", "block_k", "num_warps")


@functools.lru_cache(maxsize=None)
def _cfg_table(path):
    if not path:
        return {}
    with open(path) as f:
        table = json.load(f)
    for fmt, tiles in table.items():
        if fmt not in _CFG_KEYS:
            raise ValueError(f"{path}: unknown format {fmt!r}")
        for tile, cfg in tiles.items():
            if set(cfg) - set(_CFG_KEYS[fmt]):
                raise ValueError(f"{path}: {fmt}/{tile}: unknown keys {sorted(set(cfg) - set(_CFG_KEYS[fmt]))}")
            for key, v in cfg.items():
                if not isinstance(v, int) or v < 1 or (key in _POW2 and v & (v - 1)):
                    raise ValueError(f"{path}: {fmt}/{tile}: {key}={v!r} (power of 2 / positive int)")
            if cfg.get("block_k", 128) < 32:
                raise ValueError(f"{path}: {fmt}/{tile}: block_k < 32")
    return table


def prefill_cfg(fmt, tile):
    """Table entry for (format, token tile), {} = launch defaults."""
    return _cfg_table(os.environ.get(_CFG_ENV, "")).get(fmt, {}).get(str(tile), {})


def _stream():
    return torch.cuda.current_stream().cuda_stream


def _lds_rows(max_ctas):
    """Act rows the kernel keeps in LDS at once (a CTA with more rows takes extra passes)."""
    return max(1, min(GEOMETRY["max_rows"], -(-max_ctas // _TOPK)))


def _hip_fits(plan, k, nvfp4):
    return (_hip is not None and plan.tile == _HIP_TILE and k <= GEOMETRY["k_max"]
            and k % (64 if nvfp4 else 128) == 0)


@functools.lru_cache(maxsize=None)
def launch_extents(nvfp4, rows, max_ctas, experts, n, k):
    """Bytes of each buffer the HIP kernel can touch, by emulating its index math over the whole
    launch (every block / wave / lane / step; every CTA with the worst case of 8 rows and the last
    expert).  Mirrors radiance_inkling2_moe.hip: change both together."""
    import numpy as np

    g = GEOMETRY
    assert rows == _HIP_TILE * max_ctas and k % g["quantum"] == 0
    nq = k // g["quantum"]
    qb, sb = (16, 2) if nvfp4 else (32, 1)         # weight / act bytes and scale bytes per quantum
    cols = nq * sb                                  # act scale columns per row
    nj = -(-nq // g["lpr"])
    assert nj <= g["nj_max"]
    rows_per_wave = 32 // g["lpr"]
    nb, waves = -(-n // g["bn"]), g["bn"] // rows_per_wave
    blk = np.arange(nb)[:, None, None, None]
    wave = np.arange(waves)[None, :, None, None]
    lane = np.arange(32)[None, None, :, None]
    j = np.arange(nj)[None, None, None, :]
    nn = blk * g["bn"] + wave * rows_per_wave + lane // g["lpr"]
    q = j * g["lpr"] + (lane & (g["lpr"] - 1))
    live = (nn < n) & (q < nq)
    wrow = (experts - 1) * n + nn
    ext = {
        "weight": int(((wrow * nq * qb + q * qb + qb)[live]).max()),
        "weight_scale": int(((wrow * nq * sb + q * sb + sb)[live]).max()),
    }
    row = np.arange(rows)[:, None]
    qa = np.arange(nq)[None, :]
    first = qa * sb                                 # first scale group of the quantum
    off = ((row >> 3) * (cols // 4) + first // 4) * 32 + (row & 7) * 4 + first % 4
    ext["act"] = int((row * nq * qb + qa * qb + qb).max())
    ext["act_scale"] = int((off + sb).max())
    ext["out"] = int(((rows - 1) * n + int(np.broadcast_to(nn, live.shape)[live].max()) + 1) * 2)
    ext["row_scale"] = rows * 4 if nvfp4 else 0
    ext["expert_scale"] = experts * 4
    ext["tiles"] = (2 * max_ctas + 1) * 4
    return ext


_ITEMSIZE = dict(act=1, act_scale=1, weight=1, weight_scale=1, row_scale=4, expert_scale=4, tiles=4)
_DTYPE = dict(row_scale=torch.float32, expert_scale=torch.float32, tiles=torch.int32)


def _check_launch(nvfp4, plan, experts, n, k, **tensors):
    """Refuse a launch whose index math leaves a buffer (a GPU fault can reset the host)."""
    ext = launch_extents(nvfp4, plan.rows, plan.max_ctas, experts, n, k)
    for name, t in tensors.items():
        if not t.is_contiguous():
            raise ValueError(f"{name}: not contiguous")
        if t.element_size() != _ITEMSIZE[name]:
            raise ValueError(f"{name}: dtype {t.dtype} (the kernel reads {_ITEMSIZE[name]}-byte items)")
        if name in _DTYPE and t.dtype != _DTYPE[name]:
            raise ValueError(f"{name}: dtype {t.dtype}, the kernel reads {_DTYPE[name]}")
    for name, t in tensors.items():
        if ext[name] and t.numel() * t.element_size() < ext[name]:
            raise ValueError(f"{name}: {t.numel() * t.element_size()} bytes, kernel reads/writes "
                             f"{ext[name]} (N={n} K={k} CTAs={plan.max_ctas} experts={experts})")
    for name in ("act", "weight"):
        if tensors[name].data_ptr() % 16:
            raise ValueError(f"{name}: pointer not 16-byte aligned")
    if os.environ.get("RADIANCE_INKLING2_HIP_DEBUG") == "1" and not (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()):
        t = tensors["tiles"].cpu().tolist()
        count = t[2 * plan.max_ctas]
        if not 0 <= count <= plan.max_ctas:
            raise ValueError(f"tiles: CTA count {count} outside [0, {plan.max_ctas}]")
        for cta in range(count):
            if not 0 <= t[cta] < experts:
                raise ValueError(f"tiles: expert {t[cta]} of CTA {cta} outside [0, {experts})")


@triton.jit
def _e2m1(code):
    """4-bit e2m1 code (sign in bit 3) -> fp32."""
    mag = code & 7
    exp = (mag >> 1).to(tl.float32)
    man = (mag & 1).to(tl.float32)
    val = tl.where(mag < 2, man * 0.5, (1.0 + man * 0.5) * tl.exp2(exp - 1.0))
    return tl.where((code >> 3) == 1, -val, val)


@triton.jit
def _dequant_rows(packed, scales, DT: tl.constexpr):
    """[R, K/2] uint8 + [R, K/16] fp32 scales -> [R, K] in DT (exact in bf16)."""
    r: tl.constexpr = packed.shape[0]
    h: tl.constexpr = packed.shape[1]
    g: tl.constexpr = scales.shape[1]
    p = packed.to(tl.int32)
    vals = tl.reshape(tl.join(_e2m1(p & 0xF), _e2m1(p >> 4)), (r, h * 2))
    s = tl.reshape(tl.broadcast_to(scales[:, :, None], (r, g, 16)), (r, g * 16))
    return (vals * s).to(DT)


@triton.jit
def _expert_gemm(
    Act, ActScale, RowScale, Weight, WeightScale, ExpertScale, Out, Tiles,
    MAX_CTAS: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TILE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DT: tl.constexpr,
):
    tile = tl.program_id(0)
    if tile >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    row_end = tl.load(Tiles + MAX_CTAS + tile)
    rows = tile * TILE + tl.program_id(1) * BM + tl.arange(0, BM)
    valid = rows < tl.minimum(row_end, (tile + 1) * TILE)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k in range(tl.cdiv(K, BK)):
        packed = k * (BK // 2) + tl.arange(0, BK // 2)
        groups = k * (BK // 16) + tl.arange(0, BK // 16)
        a = tl.load(Act + rows[:, None].to(tl.int64) * (K // 2) + packed[None, :],
                    valid[:, None] & (packed[None, :] < K // 2), other=0)
        a_scale = tl.load(ActScale + _scale_offsets(rows[:, None], groups[None, :], K // 16, TILE),
                          valid[:, None] & (groups[None, :] < K // 16), other=0.0)
        w = tl.load(Weight + (expert * N + cols[:, None]) * (K // 2) + packed[None, :],
                    (cols[:, None] < N) & (packed[None, :] < K // 2), other=0)
        w_scale = tl.load(WeightScale + (expert * N + cols[:, None]) * (K // 16) + groups[None, :],
                          (cols[:, None] < N) & (groups[None, :] < K // 16), other=0.0)
        acc = tl.dot(_dequant_rows(a, a_scale.to(tl.float32), DT),
                     tl.trans(_dequant_rows(w, w_scale.to(tl.float32), DT)), acc)
    acc *= tl.load(ExpertScale + expert)
    acc *= tl.load(RowScale + rows, valid, other=0)[:, None]
    tl.store(Out + rows[:, None].to(tl.int64) * N + cols[None, :], acc,
             valid[:, None] & (cols[None, :] < N))


@triton.jit
def _fp4_val(code, SAFE: tl.constexpr):
    """e2m1 code (int32, sign in bit 3) -> fp32, bit tricks instead of exp2. The 3 magnitude bits
    placed at the fp32 exponent/mantissa-MSB position read as (denormal) 0, 0.5 * 2^-126 and
    (1 + m/2) * 2^(e-127); times 2^126 that is the e2m1 value. SAFE avoids the denormal operand
    (a flush-to-zero multiply would lose 0.5)."""
    mag = code & 7
    if SAFE:
        v = tl.where(mag >= 2, ((mag << 22) + 0x3F000000).to(tl.float32, bitcast=True),
                     mag.to(tl.float32) * 0.5)
        return tl.where((code & 8) != 0, -v, v)
    bits = (mag << 22) | ((code & 8) << 28)
    return bits.to(tl.float32, bitcast=True) * 8.507059173023462e37  # 2^126


@triton.jit
def _dequant_act(Act, ActScale, Tiles, Out, MAX_CTAS: tl.constexpr, K: tl.constexpr,
                 TILE: tl.constexpr, BK: tl.constexpr, BR: tl.constexpr, DT: tl.constexpr,
                 SAFE: tl.constexpr):
    """Routed activation rows -> DT `[rows, K]`, each BK block stored as [even k | odd k], once per
    row instead of once per output column block."""
    tile = tl.program_id(0)
    if tile >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    row_end = tl.load(Tiles + MAX_CTAS + tile)
    rows = tile * TILE + tl.program_id(1) * BR + tl.arange(0, BR)
    valid = rows < tl.minimum(row_end, (tile + 1) * TILE)
    half: tl.constexpr = BK // 2
    j = tl.arange(0, half)
    for k in range(K // BK):
        p = tl.load(Act + rows[:, None].to(tl.int64) * (K // 2) + k * half + j[None, :],
                    valid[:, None], other=0).to(tl.int32)
        s = tl.load(ActScale + _scale_offsets(rows[:, None], (k * (BK // 16) + j // 8)[None, :], K // 16, TILE),
                    valid[:, None], other=0.0).to(tl.float32)
        out = Out + rows[:, None].to(tl.int64) * K + k * BK + j[None, :]
        tl.store(out, (_fp4_val(p & 15, SAFE) * s).to(DT), valid[:, None])
        tl.store(out + half, (_fp4_val(p >> 4, SAFE) * s).to(DT), valid[:, None])


@triton.jit
def _expert_gemm_v2(
    ActDQ, RowScale, Weight, WeightScale, ExpertScale, Out, Tiles,
    MAX_CTAS: tl.constexpr, N: tl.constexpr, K: tl.constexpr, TILE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, DT: tl.constexpr, SAFE: tl.constexpr,
):
    """`_expert_gemm` with the activation already dequantised (`_dequant_act`) and the weight nibbles
    kept apart: even-k and odd-k halves are two dots (a sum over k does not care about the order), so no
    interleave/shuffle. K % BK == 0."""
    tile = tl.program_id(2)  # N blocks run fastest: the CTAs of one tile share its activation rows in L2
    if tile >= tl.load(Tiles + 2 * MAX_CTAS):
        return
    expert = tl.load(Tiles + tile).to(tl.int64)
    row_end = tl.load(Tiles + MAX_CTAS + tile)
    rows = tile * TILE + tl.program_id(1) * BM + tl.arange(0, BM)
    valid = rows < tl.minimum(row_end, (tile + 1) * TILE)
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    cmask = cols < N
    half: tl.constexpr = BK // 2
    j = tl.arange(0, half)
    acc = tl.zeros((BM, BN), tl.float32)
    a_ptr = ActDQ + rows[:, None].to(tl.int64) * K + j[None, :]
    w_row = (expert * N + cols)[:, None]
    for k in range(K // BK):
        a_lo = tl.load(a_ptr + k * BK, valid[:, None], other=0.0)
        a_hi = tl.load(a_ptr + k * BK + half, valid[:, None], other=0.0)
        w = tl.load(Weight + w_row * (K // 2) + k * half + j[None, :], cmask[:, None], other=0).to(tl.int32)
        s = tl.load(WeightScale + w_row * (K // 16) + k * (BK // 16) + (j // 8)[None, :],
                    cmask[:, None], other=0.0).to(tl.float32)
        acc = tl.dot(a_lo, tl.trans((_fp4_val(w & 15, SAFE) * s).to(DT)), acc)
        acc = tl.dot(a_hi, tl.trans((_fp4_val(w >> 4, SAFE) * s).to(DT)), acc)
    acc *= tl.load(ExpertScale + expert)
    acc *= tl.load(RowScale + rows, valid, other=0)[:, None]
    tl.store(Out + rows[:, None].to(tl.int64) * N + cols[None, :], acc, valid[:, None] & cmask[None, :])


def prefill_v2():
    """RADIANCE_INKLING2_MOE_PREFILL_V2=1: the v2 prefill kernels (see `_expert_gemm_v2`)."""
    return os.environ.get(_V2_ENV, "0") == "1"


def nvfp4_expert_gemm(act, act_scale, row_scale, weight, weight_scale, expert_scale, tiles, plan,
                      block_m=None, block_n=None, block_k=None, dot_dtype=tl.bfloat16,
                      v2=None, v2_safe=False, **launch):
    """bf16 `[plan.rows, N]` = routed rows x NVFP4 expert weights `[E, N, K/2]` (rows past a tile's
    `row_end` are left unwritten).  `launch`: extra Triton launch args (num_warps, num_stages). `dot_dtype`: bf16 is exact for these products; the Triton CPU
    interpreter mis-computes bf16 dots, so the tests pass float32 (same values)."""
    n, k = weight.shape[1], weight.shape[2] * 2
    if _hip_fits(plan, k, True):
        _check_launch(True, plan, weight.shape[0], n, k, act=act, act_scale=act_scale,
                      row_scale=row_scale, weight=weight, weight_scale=weight_scale,
                      expert_scale=expert_scale, tiles=tiles)
        out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
        _hip.nvfp4_gemm(
            act.data_ptr(), act_scale.data_ptr(), row_scale.data_ptr(),
            weight.data_ptr(), weight_scale.data_ptr(), expert_scale.data_ptr(),
            out.data_ptr(), tiles.data_ptr(),
            plan.max_ctas, weight.shape[0], n, k, plan.tile, _lds_rows(plan.max_ctas), _stream())
        return out
    cfg = prefill_cfg("nvfp4", plan.tile)
    bm = block_m or cfg.get("block_m") or min(32, triton.next_power_of_2(plan.tile))  # arange needs a
    bn = block_n or cfg.get("block_n") or min(64, triton.next_power_of_2(n))          # power of 2; the
    bk = block_k or cfg.get("block_k") or min(128, triton.next_power_of_2(k))         # masks cover the overhang
    launch = {**{key: cfg[key] for key in ("num_warps", "num_stages") if key in cfg}, **launch}
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    if (prefill_v2() if v2 is None else v2) and k % bk == 0 and bk >= 32:
        dq = torch.empty((plan.rows, k), dtype=torch.float32 if dot_dtype == tl.float32 else torch.bfloat16,
                         device=act.device)
        br = min(16, bm)
        _dequant_act[(plan.max_ctas, triton.cdiv(plan.tile, br))](
            act, act_scale, tiles, dq, MAX_CTAS=plan.max_ctas, K=k, TILE=plan.tile, BK=bk, BR=br,
            DT=dot_dtype, SAFE=v2_safe)
        _expert_gemm_v2[(triton.cdiv(n, bn), triton.cdiv(plan.tile, bm), plan.max_ctas)](
            dq, row_scale, weight, weight_scale, expert_scale, out, tiles,
            MAX_CTAS=plan.max_ctas, N=n, K=k, TILE=plan.tile, BM=bm, BN=bn, BK=bk, DT=dot_dtype,
            SAFE=v2_safe, **launch)
        return out
    _expert_gemm[(plan.max_ctas, triton.cdiv(plan.tile, bm), triton.cdiv(n, bn))](
        act, act_scale, row_scale, weight, weight_scale, expert_scale, out, tiles,
        MAX_CTAS=plan.max_ctas, N=n, K=k, TILE=plan.tile, BM=bm, BN=bn, BK=bk, DT=dot_dtype, **launch)
    return out


def mxfp8_launch(tile, hip):
    """Triton launch config of vLLM's `_mxfp8_gemm`: vLLM's sm120 tiles (shrunk to the 64 KB LDS of
    gfx12 on ROCm), then the prefill table entry for this token tile."""
    if tile <= 16:
        block_n, block_t, num_warps = 128, 16, 4
    elif tile <= 64:
        block_n, block_t, num_warps = 128, 64, 4
    else:
        block_n, block_t, num_warps = 256, 128, 8
    if hip:
        block_n, block_t, num_warps = min(block_n, 64), min(block_t, 16), 4
    c = dict(block_n=block_n, block_t=block_t, block_k=128, num_warps=num_warps, num_stages=3)
    c.update(prefill_cfg("mxfp8", tile))
    return c


def mxfp8_expert_gemm(act, act_scale, weight, weight_scale, expert_scale, tiles, plan):
    """bf16 `[plan.rows, N]` = MXFP8 expert rows x weights * expert_scale (no row scale). HIP kernel
    for decode / spec tiles (plan.tile == 8) unless RADIANCE_INKLING2_HIP_MOE=0; otherwise vLLM's
    Triton `_mxfp8_gemm` with the ROCm tile shrink, identical to what ops/moe_sm120.py's `bmm()`
    launches."""
    n, k = weight.shape[1], weight.shape[2]
    if _hip_fits(plan, k, False):
        _check_launch(False, plan, weight.shape[0], n, k, act=act, act_scale=act_scale,
                      weight=weight, weight_scale=weight_scale, expert_scale=expert_scale,
                      tiles=tiles)
        out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
        _hip.mxfp8_gemm(
            act.data_ptr(), act_scale.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(),
            expert_scale.data_ptr(), out.data_ptr(), tiles.data_ptr(),
            plan.max_ctas, weight.shape[0], n, k, plan.tile, _lds_rows(plan.max_ctas), _stream())
        return out
    from vllm.models.inkling2.nvidia.ops.moe_sm120 import _mxfp8_gemm
    c = mxfp8_launch(plan.tile, bool(torch.version.hip))
    out = torch.empty((plan.rows, n), dtype=torch.bfloat16, device=act.device)
    _mxfp8_gemm[
        (triton.cdiv(n, c["block_n"]), plan.max_ctas * triton.cdiv(plan.tile, c["block_t"]))
    ](
        act, act_scale, weight, weight_scale, expert_scale, out, tiles,
        MAX_CTAS=plan.max_ctas, N=n, K=k, TILE=plan.tile, BN=c["block_n"], BT=c["block_t"],
        BK=c["block_k"], num_warps=c["num_warps"], num_stages=c["num_stages"],
    )
    return out
