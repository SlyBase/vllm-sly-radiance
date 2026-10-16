"""KV4H store: rotate, quantize and write K/V into the HIP-kernel layout (radiance_kv4h_format).

One Triton program per 16 (token, kv head) rows; the rotation is a bf16 dot with the +-1 matrix
inside the kernel (fp32 accumulation), so a store call is one launch (~40 us at 8 tokens; the
torch rotation + per-token kernel cost ~230 us per layer and step). Held to
radiance_kv4h_format.encode_block up to the rotation's summation order (sly/tests/kv4/test_kv4h.py:
bytes identical but for rare 1-LSB code flips).

Cache: uint8 (num_blocks, num_kv_heads, block_size, 2 * TPT), AITER's shape; the (block, head)
region of block_size * 2 * TPT bytes is laid out as the format module describes for the head
size (256 or 128; the last two dims are only a byte count to this kernel). slot_mapping: int64
(num_tokens,), -1 = skip.
"""
import os

import torch

from vllm.triton_utils import tl, triton

import radiance_kv4_format as F
import radiance_kv4h_format as H

_FP8_TL = tl.float8e4nv


@triton.jit
def _round_up_e4m3(sc):
    """radiance_kv4_format.round_up_e4m3 in Triton (integer bit math, see there)."""
    big = tl.minimum(tl.maximum(sc, 0.015625), 448.0)
    b = big.to(tl.int32, bitcast=True)
    t = b & -1048576
    t = tl.where((b & 1048575) != 0, t + 1048576, t)
    normal = tl.minimum(t.to(tl.float32, bitcast=True), 448.0)
    small = tl.maximum(tl.ceil(sc * 512.0), 1.0) * 0.001953125
    return tl.where(sc < 0.015625, small, normal)


@triton.jit
def _e4m3_step_down(s):
    """radiance_kv4h_format.e4m3_step_down in Triton."""
    normal = (s.to(tl.int32, bitcast=True) - 1048576).to(tl.float32, bitcast=True)
    sub = tl.maximum(s - 0.001953125, 0.001953125)
    return tl.where(s > 0.015625, normal, sub)


@triton.jit
def _best_scale(x3, clip, QMAX: tl.constexpr, QMIN: tl.constexpr, STEPS: tl.constexpr):
    """x3 (B, G, N): per (row, group) the e4m3 scale with the smallest squared error among the
    absmax scale and the STEPS - 1 e4m3 values below it (radiance_kv4h_format._scales)."""
    s = _round_up_e4m3(tl.max(tl.abs(x3), axis=2) * clip / QMAX)
    best = s
    if STEPS > 1:
        q = tl.minimum(tl.maximum(tl.floor(x3 / s[:, :, None] + 0.5), QMIN), QMAX)
        d = q * s[:, :, None] - x3
        best_err = tl.sum(d * d, axis=2)
        for _ in tl.static_range(STEPS - 1):
            s = _e4m3_step_down(s)
            q = tl.minimum(tl.maximum(tl.floor(x3 / s[:, :, None] + 0.5), QMIN), QMAX)
            d = q * s[:, :, None] - x3
            err = tl.sum(d * d, axis=2)
            better = err < best_err
            best = tl.where(better, s, best)
            best_err = tl.where(better, err, best_err)
    return best


@triton.jit
def _wht_rows(x, B: tl.constexpr, N: tl.constexpr, LOG2: tl.constexpr):
    """Unnormalized Walsh-Hadamard transform of every row of a (B, N) register tensor: x @ H with
    the Sylvester H, as LOG2 butterfly stages (pairs (i, i + h) -> (a + b, a - b)). x @ R for the
    rotation R = diag(D) @ H is _wht_rows(x * D); y @ R^T is _wht_rows(y) * D."""
    for st in tl.static_range(LOG2):
        x = _wht_stage_rows(x, B, N, 1 << st)
    return x


@triton.jit
def _wht_stage_rows(x, B: tl.constexpr, N: tl.constexpr, HALF: tl.constexpr):
    x4 = tl.reshape(x, (B, N // (2 * HALF), 2, HALF))
    x4 = tl.permute(x4, (0, 1, 3, 2))
    a, b = tl.split(x4)
    y4 = tl.join(a + b, a - b)
    y4 = tl.permute(y4, (0, 1, 3, 2))
    return tl.reshape(y4, (B, N))


@triton.jit
def _kv4h_store_kernel(
    k_ptr, v_ptr, r_ptr, cache_ptr, slot_ptr, n_rows, num_kv_heads, clip,
    stride_kt: tl.int64, stride_kh: tl.int64,
    stride_vt: tl.int64, stride_vh: tl.int64,
    stride_cb: tl.int64, stride_ch: tl.int64,
    block_size: tl.int32,
    K_REGION: tl.int64,          # block_size * TPT
    QMAX: tl.constexpr, QMIN: tl.constexpr, STEPS: tl.constexpr,
    HEAD: tl.constexpr, NG: tl.constexpr, GROUP: tl.constexpr, TG: tl.constexpr,
    PACKED: tl.constexpr, REG: tl.constexpr, K_AREA: tl.constexpr, V_SCALES: tl.constexpr,
    BLOCK: tl.constexpr, KC: tl.constexpr, WHT: tl.constexpr, LOG2: tl.constexpr,
):
    """One program per BLOCK rows, row = token * num_kv_heads + head. The rotation is a bf16 dot
    with the +-1 matrix R (fp32 accumulation) inside the kernel: one launch per store call."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inb = rows < n_rows
    tok = rows // num_kv_heads
    head = rows % num_kv_heads
    slot = tl.load(slot_ptr + tok, mask=inb, other=-1)
    live = inb & (slot >= 0)
    cols = tl.arange(0, HEAD)
    kbase = k_ptr + tok * stride_kt + head * stride_kh
    vbase = v_ptr + tok * stride_vt + head * stride_vh
    if WHT:   # r_ptr: the sign vector D (fp32, HEAD)
        d = tl.load(r_ptr + cols)
        xk = tl.load(kbase[:, None] + cols[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        xv = tl.load(vbase[:, None] + cols[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        kacc = _wht_rows(xk * d[None, :], BLOCK, HEAD, LOG2)
        vacc = _wht_rows(xv * d[None, :], BLOCK, HEAD, LOG2)
    else:     # r_ptr: the rotation matrix R (bf16, HEAD x HEAD)
        kacc = tl.zeros((BLOCK, HEAD), dtype=tl.float32)
        vacc = tl.zeros((BLOCK, HEAD), dtype=tl.float32)
        for k0 in range(0, HEAD, KC):
            kk = k0 + tl.arange(0, KC)
            r = tl.load(r_ptr + kk[:, None] * HEAD + cols[None, :])
            xk = tl.load(kbase[:, None] + kk[None, :], mask=live[:, None], other=0.0)
            xv = tl.load(vbase[:, None] + kk[None, :], mask=live[:, None], other=0.0)
            kacc += tl.dot(xk.to(tl.bfloat16), r)
            vacc += tl.dot(xv.to(tl.bfloat16), r)

    blk = slot // block_size
    t = slot % block_size
    grp = t // TG
    tl_ = t % TG
    region = cache_ptr + blk.to(tl.int64) * stride_cb + head.to(tl.int64) * stride_ch
    pk = tl.arange(0, PACKED)

    # ---- K: NG group scales, codes QMIN..7, interleaved nibbles (dims 2p, 2p + 1 share byte p) ----
    x2 = tl.reshape(kacc, (BLOCK, NG, GROUP))
    sf = _best_scale(x2, clip, QMAX, QMIN, STEPS)                                        # (BLOCK, NG)
    q = tl.minimum(tl.maximum(tl.floor(x2 / sf[:, :, None] + 0.5), QMIN), QMAX).to(tl.int32)
    ev, od = tl.split(tl.reshape(q, (BLOCK, PACKED, 2)))
    packed = ((ev & 0xF) | ((od & 0xF) << 4)).to(tl.uint8)
    g = tl.arange(0, NG)
    tl.store(region[:, None] + grp[:, None] * REG + g[None, :] * TG + tl_[:, None],
             sf.to(_FP8_TL).to(tl.uint8, bitcast=True), mask=live[:, None])
    tl.store(region[:, None] + grp[:, None] * REG + K_AREA + tl_[:, None] * PACKED + pk[None, :],
             packed, mask=live[:, None])

    # ---- V: one scale, codes, dim-major bytes (d, d + 128) at d * 16 + t ----
    vs = tl.reshape(_best_scale(tl.reshape(vacc, (BLOCK, 1, HEAD)), clip, QMAX, QMIN, STEPS), (BLOCK,))
    qv = tl.minimum(tl.maximum(tl.floor(vacc / vs[:, None] + 0.5), QMIN), QMAX).to(tl.int32)
    lo, hi = tl.split(tl.permute(tl.reshape(qv, (BLOCK, 2, PACKED)), (0, 2, 1)))   # dims d, d + PACKED
    vpacked = ((lo & 0xF) | ((hi & 0xF) << 4)).to(tl.uint8)
    vreg = region + K_REGION + grp * REG
    tl.store(vreg + tl_, vs.to(_FP8_TL).to(tl.uint8, bitcast=True), mask=live)
    tl.store(vreg[:, None] + V_SCALES + pk[None, :] * TG + tl_[:, None], vpacked, mask=live[:, None])


_R: dict = {}


def _rmat(device, head):
    key = (str(device), head)
    if key not in _R:
        _R[key] = F.rotation_matrix(device, torch.float32, head).to(torch.bfloat16).contiguous()
    return _R[key]


def _dvec(device, head):
    key = ("d", str(device), head)
    if key not in _R:
        _R[key] = F.signs(device, head).to(torch.float32).contiguous()
    return _R[key]


def _env_int(name, default):
    v = os.environ.get(name, "")
    return int(v) if v else default


# knobs: the rotation as a register butterfly ('wht') or a bf16 dot with the +-1 matrix ('dot'),
# rows per program, warps
ROT = os.environ.get("RADIANCE_KV4H_STORE_ROT", "") or "dot"     # measured: dot 45 us / wht 54 us at 2048 tokens
_BLOCK = _env_int("RADIANCE_KV4H_STORE_BLOCK", 16)
_WARPS = _env_int("RADIANCE_KV4H_STORE_WARPS", 4)


def kv4h_store(key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor,
               slot_mapping: torch.Tensor, k_group: int | None = None,
               scale_steps: int | None = None, qmin: int | None = None, clip: float | None = None) -> None:
    """k_group: K dims per scale (None = RADIANCE_KV4H_K_GROUP, 256); the kernel must be built for it.
    scale_steps / qmin / clip: scale search, lowest code, first scale = amax * clip / 7 (None =
    RADIANCE_KV4H_SCALE_STEPS / _QMIN / _SCALE_CLIP,
    radiance_kv4h_format); the attention kernel reads either."""
    steps = H.SCALE_STEPS if scale_steps is None else scale_steps
    qmin = H.QMIN if qmin is None else qmin
    clip = H.SCALE_CLIP if clip is None else clip
    assert steps >= 1 and qmin in (-7, -8) and 0.5 <= clip <= 1.0, (steps, qmin, clip)
    L = H.layout(key.shape[-1], min(k_group or H.GROUP, key.shape[-1]))
    assert value.shape[-1] == L.HEAD
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == 2 * L.TPT
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    num_kv_heads = key.shape[1]
    # the token count is the slot mapping's (vLLM's reshape_and_cache does the same): key/value
    # may carry padding rows past it, never fewer rows
    num_tokens = slot_mapping.shape[0]
    assert key.shape[0] >= num_tokens and value.shape[0] >= num_tokens, (key.shape, slot_mapping.shape)
    assert kv_cache.shape[1] == num_kv_heads and value.shape[1] == num_kv_heads, (kv_cache.shape, key.shape)
    assert kv_cache.stride(3) == 1 and kv_cache.stride(2) == 2 * L.TPT, kv_cache.stride()
    block_size = kv_cache.shape[2]
    assert block_size % L.TG == 0
    rows = num_tokens * num_kv_heads
    if rows == 0:
        return
    if key.dtype != torch.bfloat16:
        key = key.to(torch.bfloat16)
    if value.dtype != torch.bfloat16:
        value = value.to(torch.bfloat16)
    wht = ROT == "wht"
    _kv4h_store_kernel[(triton.cdiv(rows, _BLOCK),)](
        key, value, (_dvec if wht else _rmat)(key.device, L.HEAD), kv_cache, slot_mapping, rows, num_kv_heads, float(clip),
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), block_size, L.k_region_bytes(block_size),
        QMAX=L.QMAX, QMIN=float(qmin), STEPS=steps, HEAD=L.HEAD, NG=L.NG, GROUP=L.GROUP, TG=L.TG, PACKED=L.PACKED,
        REG=L.REG, K_AREA=L.K_AREA, V_SCALES=L.V_SCALES, BLOCK=_BLOCK, KC=64, WHT=wht,
        LOG2=L.HEAD.bit_length() - 1, num_warps=_WARPS,
    )
