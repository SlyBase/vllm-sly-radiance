"""KV4 store: rotate, quantize, pack and write K/V into the 4-bit paged cache.

`kv4_store(key, value, kv_cache, slot_mapping, codebook)` is the backend's `do_kv_cache_update`.
The rotation is one fp32 matmul over the chunk (torch); the Triton kernel then runs one program
per (token, kv head) and writes the 272-byte cell: group scales, packed nibbles, K side then V side.
It is held to `radiance_kv4_format.encode_content` bit for bit (sly/tests/kv4/test_kv4_kernels.py),
so the arithmetic here mirrors the reference: amax per group, scale rounded UP to e4m3, index by
floor(v + 0.5) (uniform), 15 threshold compares (lloyd) or sign + 7 magnitude compares (geo). The scale rounding is integer bit
math on the fp32 pattern in both places, so no conversion rounding mode can make them disagree.

Layout in: key, value (num_tokens, num_kv_heads, 256) any float dtype, already rope'd.
Cache: uint8 (num_blocks, num_kv_heads, block_size, 272), AITER's content-packed layout.
slot_mapping: int64 (num_tokens,), -1 = skip (vLLM pads with it).
"""
import os

import torch

from vllm.triton_utils import tl, triton

import radiance_kv4_format as F

# RADIANCE_KV4_DUMP=<dir>: save the first RADIANCE_KV4_DUMP_N (default 64) store calls' unquantized
# K/V (bf16, after rope) as <dir>/kv_<n>.pt -- the real-activation input for the format
# comparison (sly/kv4/eval_format.py). Off when unset.
_DUMP_DIR = os.environ.get("RADIANCE_KV4_DUMP")
if not _DUMP_DIR and os.path.exists("/patches/sly/kv4/DUMP"):   # serve scripts cannot pass -e; a marker file in the mount
    _DUMP_DIR = open("/patches/sly/kv4/DUMP").read().strip() or None
_DUMP_N = int(os.environ.get("RADIANCE_KV4_DUMP_N", "64"))
_dump_count = 0
_dump_skip = int(os.environ.get("RADIANCE_KV4_DUMP_SKIP", "400"))   # profiling + graph-capture calls carry dummy inputs

_FP8_TL = tl.float8e4nv  # OCP e4m3, what torch.float8_e4m3fn is on gfx12


@triton.jit
def _quantize_side(
    x_ptr,            # fp32 (HEAD,) rotated values of this (token, head)
    out_ptr,          # uint8 cell base + side offset
    thr_ptr,          # fp32 (15,) codebook thresholds (lloyd only)
    C_MAX: tl.constexpr,
    CODEBOOK: tl.constexpr,     # 0 = uniform, 1 = lloyd, 2 = geo (radiance_kv4_format.CODEBOOKS)
    HEAD: tl.constexpr,
    NGROUP: tl.constexpr,
    GROUP: tl.constexpr,
    PACKED: tl.constexpr,
    SCALE_BYTES: tl.constexpr,
):
    x = tl.load(x_ptr + tl.arange(0, HEAD)).to(tl.float32)
    x2 = tl.reshape(x, (NGROUP, GROUP))
    amax = tl.max(tl.abs(x2), axis=1)                      # (NGROUP,)
    # scale = round_up_e4m3(amax / c_max), the reference's integer bit math (see there)
    sc = amax / C_MAX
    big = tl.minimum(tl.maximum(sc, 0.015625), 448.0)
    b = big.to(tl.int32, bitcast=True)
    t = b & -1048576                                       # ~((1 << 20) - 1)
    t = tl.where((b & 1048575) != 0, t + 1048576, t)
    normal = tl.minimum(t.to(tl.float32, bitcast=True), 448.0)
    small = tl.maximum(tl.ceil(sc * 512.0), 1.0) * 0.001953125
    sf = tl.where(sc < 0.015625, small, normal)
    v = x2 / sf[:, None]
    if CODEBOOK == 0:      # uniform: round to the half-integer grid
        idx = tl.minimum(tl.maximum(tl.floor(v + 8.0), 0.0), 15.0).to(tl.int32)
    elif CODEBOOK == 2:    # geo: sign bit + nearest magnitude (7 thresholds)
        a = tl.abs(v)
        idx = tl.where(v < 0, 8, 0)
        for j in tl.static_range(7):
            idx += (a > tl.load(thr_ptr + j)).to(tl.int32)
    else:                  # lloyd: nearest of the sorted centroids (15 thresholds)
        idx = tl.zeros((NGROUP, GROUP), dtype=tl.int32)
        for j in tl.static_range(15):
            idx += (v > tl.load(thr_ptr + j)).to(tl.int32)
    lo, hi = tl.split(tl.permute(tl.reshape(idx, (2, PACKED)), (1, 0)))   # dim p -> lo, p + 128 -> hi
    packed = (lo | (hi << 4)).to(tl.uint8)
    tl.store(out_ptr + tl.arange(0, NGROUP), sf.to(_FP8_TL).to(tl.uint8, bitcast=True))
    tl.store(out_ptr + SCALE_BYTES + tl.arange(0, PACKED), packed)


@triton.jit
def _kv4_store_kernel(
    k_ptr, v_ptr, cache_ptr, slot_ptr, thr_k_ptr, thr_v_ptr,
    stride_kt: tl.int64, stride_kh: tl.int64,
    stride_vt: tl.int64, stride_vh: tl.int64,
    stride_cb: tl.int64, stride_ch: tl.int64, stride_cs: tl.int64,
    block_size: tl.int32,
    C_MAX_K: tl.constexpr, C_MAX_V: tl.constexpr,
    CODEBOOK_K: tl.constexpr, CODEBOOK_V: tl.constexpr,
    HEAD: tl.constexpr, NGROUP_K: tl.constexpr, NGROUP_V: tl.constexpr,
    PACKED: tl.constexpr, SIDE: tl.constexpr, SCALE_BYTES: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    slot = tl.load(slot_ptr + tok)
    if slot < 0:
        return
    blk = slot // block_size
    s = slot % block_size
    cell = cache_ptr + blk * stride_cb + head * stride_ch + s * stride_cs
    _quantize_side(k_ptr + tok * stride_kt + head * stride_kh, cell, thr_k_ptr,
                   C_MAX_K, CODEBOOK_K, HEAD, NGROUP_K, HEAD // NGROUP_K, PACKED, SCALE_BYTES)
    _quantize_side(v_ptr + tok * stride_vt + head * stride_vh, cell + SIDE, thr_v_ptr,
                   C_MAX_V, CODEBOOK_V, HEAD, NGROUP_V, HEAD // NGROUP_V, PACKED, SCALE_BYTES)


_THR: dict = {}


def _thr(name: str, device) -> torch.Tensor:
    key = (name, str(device))
    if key not in _THR:
        t = F.thresholds(name) if name != "uniform" else torch.zeros(1)
        _THR[key] = t.to(device).contiguous()
    return _THR[key]


def kv4_store(key: torch.Tensor, value: torch.Tensor, kv_cache: torch.Tensor,
              slot_mapping: torch.Tensor, codebooks=(F.CODEBOOK_K, F.CODEBOOK_V)) -> None:
    """Quantize key/value (num_tokens, num_kv_heads, 256) into kv_cache at slot_mapping. The
    codebooks default to the kernel's (K uniform, V geo); the tests also run the reference ones."""
    assert key.shape[-1] == F.HEAD and value.shape[-1] == F.HEAD
    assert kv_cache.dtype == torch.uint8 and kv_cache.shape[-1] == F.CONTENT
    num_tokens, num_kv_heads, _ = key.shape
    if num_tokens == 0:
        return
    global _dump_count, _dump_skip
    if _DUMP_DIR and _dump_count < _DUMP_N and num_tokens >= 64 and not torch.cuda.is_current_stream_capturing():
        if _dump_skip > 0:
            _dump_skip -= 1
            return _store(key, value, kv_cache, slot_mapping, codebooks)
        os.makedirs(_DUMP_DIR, exist_ok=True)
        torch.save({"k": key.detach().to(torch.bfloat16).cpu(), "v": value.detach().to(torch.bfloat16).cpu()},
                   os.path.join(_DUMP_DIR, f"kv_{_dump_count:03d}.pt"))
        _dump_count += 1
    _store(key, value, kv_cache, slot_mapping, codebooks)


def _store(key, value, kv_cache, slot_mapping, codebooks):
    num_tokens, num_kv_heads, _ = key.shape
    cb_k, cb_v = codebooks
    k_rot = F.rotate(key.reshape(-1, F.HEAD).to(torch.float32)).reshape(num_tokens, num_kv_heads, F.HEAD)
    v_rot = F.rotate(value.reshape(-1, F.HEAD).to(torch.float32)).reshape(num_tokens, num_kv_heads, F.HEAD)
    _kv4_store_kernel[(num_tokens, num_kv_heads)](
        k_rot, v_rot, kv_cache, slot_mapping, _thr(cb_k, key.device), _thr(cb_v, key.device),
        k_rot.stride(0), k_rot.stride(1), v_rot.stride(0), v_rot.stride(1),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        kv_cache.shape[2],
        C_MAX_K=F.c_max(cb_k), C_MAX_V=F.c_max(cb_v),
        CODEBOOK_K=F.CODEBOOKS.index(cb_k), CODEBOOK_V=F.CODEBOOKS.index(cb_v),
        HEAD=F.HEAD, NGROUP_K=F.NGROUP_K, NGROUP_V=F.NGROUP_V, PACKED=F.PACKED, SIDE=F.SIDE,
        SCALE_BYTES=F.SCALE_BYTES,
        num_warps=2,
    )
