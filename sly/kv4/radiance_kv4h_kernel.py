"""KV4H attention launcher: Triton query rotation + int8 quantization, the HIP int4-WMMA kernel
(radiance_kv4h.hip) for the split-KV decode / verify path, a Triton segment merge that also
un-rotates into vLLM's output (or only merges, when the caller wants the rotated result),
AITER's config tables for the split count. Builds the extension with the image's hipcc on first use (no torch headers, ~20 s) and
caches the .so under RADIANCE_KV4H_BUILD_DIR (default /tmp/radiance_kv4h).

Every launch is the 3D form: one program per (64-row query block, kv head, segment); with one
segment the kernel normalizes and writes the output itself. Prefill chunks run the same kernel
with as many query blocks as the chunk has (block_q = 64 // nqpkv query tokens per block).
Head 256 (the target) and head 128 with a sliding window (the DFlash drafter) are both served.
"""
import fcntl
import hashlib
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl

import aiter.ops.triton.attention.unified_attention as UA
import aiter.ops.triton.utils.unified_attention_utils as UU
from aiter.ops.triton.utils.types import e4m3_dtype

import radiance_kv4_format as F
import radiance_kv4h_format as H
from radiance_kv4h_store import _wht_rows

_EXT: dict = {}
_ROT: dict = {}
LAST_LAUNCH: dict = {}


def _env_int(name, default):
    v = os.environ.get(name, "")
    return int(v) if v else default


# launch knobs (env defaults; every one is also an argument of kv4h_attention and a bench flag)
TILE_DECODE = _env_int("RADIANCE_KV4H_TILE", 32)            # tokens per tile, decode / verify launches
TILE_PREFILL = _env_int("RADIANCE_KV4H_PREFILL_TILE", 32)   # ... prefill chunks (max query len > 16)
ROW_TILES = _env_int("RADIANCE_KV4H_ROWTILES", 4)           # 16-row tiles per program (2 | 4)
Q_PLANES = _env_int("RADIANCE_KV4H_QPLANES", 1)             # 1: int4 query (default since the gate: GSM8K 0.80), 2: int8 as two planes
K_GROUP = _env_int("RADIANCE_KV4H_K_GROUP", 256)            # K dims per scale (format knob: store + kernel); 256 = one per token
SEGMENTS = _env_int("RADIANCE_KV4H_SEGMENTS", 0)            # split count (0 = AITER's table / the fork's tune)
WINDOW_TPS = _env_int("RADIANCE_KV4H_WINDOW_TPS", 8)        # tiles per segment cap for windowed calls
REDUCE_WARPS = _env_int("RADIANCE_KV4H_REDUCE_WARPS", 0)    # merge kernel warps (0 = auto: by segments x head)
P_REF = _env_int("RADIANCE_KV4H_PREF", 1)                   # P's uint8 reference: 1 the tile's max (default: KL vs bf16
#                                                             0.0170 -> 0.0139), 0 the running max (KV4H_PREF)


def _ext(row_tiles=None, q_planes=None, k_groups=None, p_ref=None):
    """The extension built for (row tiles, query planes, K scale groups, P reference); one .so per
    combination, built with hipcc on first use and cached under RADIANCE_KV4H_BUILD_DIR."""
    key = (row_tiles or ROW_TILES, q_planes or Q_PLANES, k_groups or H.HEAD // min(K_GROUP, H.HEAD),
           P_REF if p_ref is None else p_ref)
    if key in _EXT:
        return _EXT[key]
    nrt, qp, ns, pref = key
    assert nrt in (2, 4) and qp in (1, 2) and ns in (1, 2, 4) and pref in (0, 1), key
    src = Path(__file__).resolve().parent / "radiance_kv4h.hip"
    if not src.exists():  # installed copy in site-packages: the .hip sits next to it
        src = Path(__file__).resolve().with_suffix(".hip")
    bdir = Path(os.environ.get("RADIANCE_KV4H_BUILD_DIR", "/tmp/radiance_kv4h"))
    bdir.mkdir(parents=True, exist_ok=True)
    tag = f"r{nrt}q{qp}s{ns}" + (f"p{pref}" if pref else "")
    modname = f"radiance_kv4h_{tag}"
    so = bdir / f"{modname}.so"
    stamp = bdir / f"{modname}.stamp"
    cflags = os.environ.get("RADIANCE_KV4H_CFLAGS", "").split()
    cflags += [f"-DKV4H_NRT={nrt}", f"-DKV4H_QP={qp}", f"-DKV4H_NS={ns}", f"-DKV4H_PREF={pref}",
               f"-DKV4H_MODNAME={modname}"]
    # keyed on the source CONTENT (install_kv4 re-copies the file at every container start, so an
    # mtime key never hits a persistent build dir) and the flags; built under a lock into a temp
    # file and renamed, so two processes never load a half-written .so
    key_s = f"{hashlib.sha1(src.read_bytes()).hexdigest()}:{' '.join(cflags)}"
    if not so.exists() or not stamp.exists() or stamp.read_text() != key_s:
        with open(bdir / f"{modname}.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if not so.exists() or not stamp.exists() or stamp.read_text() != key_s:
                inc = subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True).split()
                tmp = so.with_name(f"{modname}.{os.getpid()}.tmp.so")
                cmd = ["hipcc", "-O3", "-fPIC", "-shared", "-std=c++20", "--offload-arch=gfx1201", *inc, *cflags,
                       str(src), "-o", str(tmp)]
                subprocess.run(cmd, check=True)
                os.replace(tmp, so)
                stamp.write_text(key_s)
    spec = importlib.util.spec_from_file_location(modname, so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _EXT[key] = mod
    return mod


def _rot(device, head):
    """(R, R^T) as bf16, cached per device and head: R = diag(D) @ H with +-1 entries (exact in bf16)."""
    key = (str(device), head)
    if key not in _ROT:
        r = F.rotation_matrix(device, torch.float32, head)
        _ROT[key] = (r.to(torch.bfloat16).contiguous(), r.t().to(torch.bfloat16).contiguous())
    return _ROT[key]


@triton.jit
def _rotate_rows(x_ptr, r_ptr, rows, mask, stride_x, HEAD: tl.constexpr, KC: tl.constexpr):
    """(x[rows] @ R) in fp32 for a block of rows; x bf16, R bf16 (HEAD, HEAD) row-major."""
    cols = tl.arange(0, HEAD)
    acc = tl.zeros((rows.shape[0], HEAD), dtype=tl.float32)
    for k0 in range(0, HEAD, KC):
        kk = k0 + tl.arange(0, KC)
        x = tl.load(x_ptr + rows[:, None] * stride_x + kk[None, :], mask=mask[:, None], other=0.0)
        r = tl.load(r_ptr + kk[:, None] * HEAD + cols[None, :])
        acc += tl.dot(x.to(tl.bfloat16), r)
    return acc


@triton.jit
def _quantize_q_kernel(x_ptr, r_ptr, q8_ptr, ds_ptr, n_rows, stride_x, qmax,
                       BLOCK: tl.constexpr, HEAD: tl.constexpr, KC: tl.constexpr,
                       WHT: tl.constexpr, LOG2: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < n_rows
    if WHT:   # r_ptr: D (fp32)
        cols = tl.arange(0, HEAD)
        x = tl.load(x_ptr + rows[:, None] * stride_x + cols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        acc = _wht_rows(x * tl.load(r_ptr + cols)[None, :], BLOCK, HEAD, LOG2)
    else:
        acc = _rotate_rows(x_ptr, r_ptr, rows, mask, stride_x, HEAD, KC)
    amax = tl.maximum(tl.max(tl.abs(acc), axis=1), 1e-12)
    desc = amax / qmax
    v = acc / desc[:, None]
    q = tl.where(v >= 0, tl.floor(v + 0.5), -tl.floor(-v + 0.5))          # round half away from zero
    q = tl.minimum(tl.maximum(q, -qmax), qmax)
    cols = tl.arange(0, HEAD)
    tl.store(q8_ptr + rows[:, None] * HEAD + cols[None, :], q.to(tl.int8), mask=mask[:, None])
    tl.store(ds_ptr + rows, desc, mask=mask)


@triton.jit
def _unrotate_kernel(y_ptr, rt_ptr, out_ptr, n_rows, n_heads, stride_y, stride_o0, stride_o1,
                     BLOCK: tl.constexpr, HEAD: tl.constexpr, KC: tl.constexpr,
                     WHT: tl.constexpr, LOG2: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < n_rows
    if WHT:   # rt_ptr: D (fp32); y @ R^T = WHT(y) * D
        cols = tl.arange(0, HEAD)
        y = tl.load(y_ptr + rows[:, None] * stride_y + cols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
        acc = _wht_rows(y, BLOCK, HEAD, LOG2) * (tl.load(rt_ptr + cols)[None, :] * (1.0 / HEAD))
    else:
        acc = _rotate_rows(y_ptr, rt_ptr, rows, mask, stride_y, HEAD, KC) * (1.0 / HEAD)
    tok = rows // n_heads
    head = rows % n_heads
    cols = tl.arange(0, HEAD)
    tl.store(out_ptr + (tok * stride_o0 + head * stride_o1)[:, None] + cols[None, :],
             acc.to(tl.bfloat16), mask=mask[:, None])


_QBLOCK = _env_int("RADIANCE_KV4H_Q_BLOCK", 32)          # rows per program, q quantization / un-rotation
_QWARPS = _env_int("RADIANCE_KV4H_Q_WARPS", 4)
ROT = os.environ.get("RADIANCE_KV4H_ROT", "") or "wht"    # q / un-rotation: butterfly 'wht' | bf16 dot 'dot'
#                         (2048 tokens: q 123 vs 215 us, un-rotation 45 vs 178 us; decode shapes equal)
_SIGNS: dict = {}


def _signs(device, head):
    """D (+-1 as fp32) for the merge's un-rotation."""
    key = (str(device), head)
    if key not in _SIGNS:
        _SIGNS[key] = F.signs(device, head).to(torch.float32).contiguous()
    return _SIGNS[key]


@triton.jit
def _wht_stage(x, N: tl.constexpr, HALF: tl.constexpr):
    """One Walsh-Hadamard butterfly stage on a 1-D register tensor: pairs (i, i + HALF) within
    blocks of 2 * HALF become (a + b, a - b)."""
    x3 = tl.reshape(x, (N // (2 * HALF), 2, HALF))
    x3 = tl.permute(x3, (0, 2, 1))
    a, b = tl.split(x3)
    y3 = tl.join(a + b, a - b)
    y3 = tl.permute(y3, (0, 2, 1))
    return tl.reshape(y3, (N,))


@triton.jit
def _reduce_unrotate_kernel(so_ptr, sm_ptr, ss_ptr, sign_ptr, out_ptr, n_rows, n_heads,
                            stride_o0, stride_o1,
                            NSEG: tl.constexpr, HEAD: tl.constexpr, LOG2: tl.constexpr,
                            UNROT: tl.constexpr = True):
    """Merge the split-KV partials of one (token, head) row -- segm_out [row][seg][HEAD] fp32,
    segm_max / segm_expsum [row][seg] in the exp2 domain, empty segments carry (-inf, 0, 0) --
    and un-rotate the result ((y @ H) * D / HEAD, H as a Walsh-Hadamard butterfly, no matmul)
    straight into vLLM's (strided) bf16 output. One program per row, like AITER's reduce."""
    row = tl.program_id(0).to(tl.int64)
    segs = tl.arange(0, NSEG)
    m = tl.load(sm_ptr + row * NSEG + segs)
    lsum = tl.load(ss_ptr + row * NSEG + segs)
    M = tl.max(m, axis=0)
    M = tl.where(M == float("-inf"), 0.0, M)
    w = tl.exp2(m - M)
    L = tl.sum(w * lsum, axis=0)
    inv = tl.where(L > 0, 1.0 / L, 0.0)
    cols = tl.arange(0, HEAD)
    acc = tl.sum(tl.load(so_ptr + (row * NSEG + segs)[:, None] * HEAD + cols[None, :]) * w[:, None], axis=0)
    y = acc * inv
    if UNROT:   # without: the rotated result (the caller un-rotates)
        y = y * (1.0 / HEAD)
        for st in tl.static_range(LOG2):
            y = _wht_stage(y, HEAD, 1 << st)
        y = y * tl.load(sign_ptr + cols)
    tok = row // n_heads
    head = row % n_heads
    tl.store(out_ptr + tok * stride_o0 + head * stride_o1 + cols, y.to(tl.bfloat16))


def reduce_unrotate(segm_out, segm_max, segm_sum, output, warps=None, unrotate=True):
    """-> output (num_tokens, num_q_heads, head) bf16, any strides with a unit last stride.
    unrotate=False: merge only (the rotated result).
    warps: the merge program's warps (None: RADIANCE_KV4H_REDUCE_WARPS, 0 = auto: one warp per
    2k fp32 partial values, 2..8 -- a row at 64 segments x 256 dims is 16k values)."""
    n, h, nseg, head = segm_out.shape
    assert output.dtype == torch.bfloat16 and output.shape == (n, h, head) and output.stride(2) == 1
    rows = n * h
    w = warps or REDUCE_WARPS or max(2, min(8, (nseg * head) // 2048))
    if rows:
        _reduce_unrotate_kernel[(rows,)](
            segm_out, segm_max, segm_sum, _signs(segm_out.device, head), output, rows, h,
            output.stride(0), output.stride(1), NSEG=nseg, HEAD=head, LOG2=head.bit_length() - 1,
            UNROT=bool(unrotate), num_warps=w)
    return output


def quantize_query(query: torch.Tensor, qmax: float = 127.0):
    """(num_tokens, num_q_heads, head) bf16 -> (int8 rotated (same shape), fp32 descale
    [num_tokens, num_q_heads]); one Triton launch. qmax 127: int8 (two int4 planes in the kernel),
    7: int4 (one plane)."""
    n, h, head = query.shape
    x = query.reshape(-1, head)
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    rows = n * h
    q8 = torch.empty(n, h, head, dtype=torch.int8, device=query.device)
    desc = torch.empty(n, h, dtype=torch.float32, device=query.device)
    if rows:
        wht = ROT == "wht"
        r = _signs(query.device, head) if wht else _rot(query.device, head)[0]
        _quantize_q_kernel[(triton.cdiv(rows, _QBLOCK),)](
            x, r, q8, desc, rows, x.stride(0), float(qmax), BLOCK=_QBLOCK, HEAD=head, KC=64,
            WHT=wht, LOG2=head.bit_length() - 1, num_warps=_QWARPS)
    return q8, desc


def unrotate_output(out_rot: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
    """Inverse rotation of the kernel's output, written straight into `output` (bf16, any
    (num_tokens, num_q_heads, head) strides) or into a new tensor."""
    n, h, head = out_rot.shape
    y = out_rot.reshape(-1, head)
    if y.dtype != torch.bfloat16:
        y = y.to(torch.bfloat16)
    if output is None:
        output = torch.empty(n, h, head, dtype=torch.bfloat16, device=out_rot.device)
    assert output.dtype == torch.bfloat16 and output.shape == (n, h, head) and output.stride(2) == 1
    rows = n * h
    if rows:
        wht = ROT == "wht"
        rt = _signs(out_rot.device, head) if wht else _rot(out_rot.device, head)[1]
        _unrotate_kernel[(triton.cdiv(rows, _QBLOCK),)](
            y, rt, output, rows, h, y.stride(0), output.stride(0), output.stride(1),
            BLOCK=_QBLOCK, HEAD=head, KC=64, WHT=wht, LOG2=head.bit_length() - 1, num_warps=_QWARPS)
    return output


@triton.jit
def _rotate_bf16_kernel(x_ptr, d_ptr, out_ptr, n_rows, stride_x,
                        BLOCK: tl.constexpr, HEAD: tl.constexpr, LOG2: tl.constexpr):
    """rotate(x) = (x * D) @ H for a block of rows, bf16 in, bf16 out (butterfly, fp32 inside)."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = rows < n_rows
    cols = tl.arange(0, HEAD)
    x = tl.load(x_ptr + rows[:, None] * stride_x + cols[None, :], mask=mask[:, None], other=0.0).to(tl.float32)
    acc = _wht_rows(x * tl.load(d_ptr + cols)[None, :], BLOCK, HEAD, LOG2)
    tl.store(out_ptr + rows[:, None] * HEAD + cols[None, :], acc.to(tl.bfloat16), mask=mask[:, None])


def rotate_query(query: torch.Tensor) -> torch.Tensor:
    """(num_tokens, num_q_heads, head) bf16 -> rotate(query) as a contiguous bf16 tensor; one Triton launch.
    The int4 libr4d prefill route hands this to the kernel's bf16-query path, which takes it to e4m3 itself."""
    n, h, head = query.shape
    x = query.reshape(-1, head)
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    out = torch.empty(n, h, head, dtype=torch.bfloat16, device=query.device)
    rows = n * h
    if rows:
        _rotate_bf16_kernel[(triton.cdiv(rows, _QBLOCK),)](
            x, _signs(query.device, head), out, rows, x.stride(0),
            BLOCK=_QBLOCK, HEAD=head, LOG2=head.bit_length() - 1, num_warps=_QWARPS)
    return out


def quantize_query_torch(query: torch.Tensor, qmax: float = 127.0):
    """The torch reference of quantize_query (tests)."""
    n, h, head = query.shape
    q_rot = F.rotate(query.reshape(-1, head).to(torch.float32)).reshape(n, h, head)
    amax = q_rot.abs().amax(-1).clamp_min(1e-12)
    descale = amax / qmax
    q8 = torch.round(q_rot / descale[..., None]).clamp(-qmax, qmax).to(torch.int8).contiguous()
    return q8, descale.contiguous()


def unrotate_output_torch(out_rot: torch.Tensor) -> torch.Tensor:
    n, h, head = out_rot.shape
    return F.unrotate(out_rot.reshape(-1, head).to(torch.float32)).reshape(n, h, head)


_TUNE: dict | None = None
_ENV_SET = {k: (k in os.environ and os.environ[k] != "") for k in
            ("RADIANCE_KV4H_TILE", "RADIANCE_KV4H_PREFILL_TILE", "RADIANCE_KV4H_ROWTILES", "RADIANCE_KV4H_SEGMENTS")}


def _tune_table():
    """The knob table from tune_kv4h.py --emit-table: RADIANCE_KV4H_TUNE=<json>, 'off' = none,
    unset or empty = the kv4h_tune.json next to this module (install_kv4 copies it there). The
    choice is logged once to stderr."""
    global _TUNE
    if _TUNE is None:
        src = os.environ.get("RADIANCE_KV4H_TUNE", "")
        if not src:
            cand = Path(__file__).resolve().with_name("kv4h_tune.json")
            src = str(cand) if cand.exists() else ""
        if src and src != "off":
            import json
            _TUNE = json.load(open(src))["table"]
        else:
            _TUNE = {}
        print(f"[tetra] tune table: {src or 'none found'}" + (f" ({', '.join(sorted(_TUNE))})" if _TUNE else ""),
              file=sys.stderr, flush=True)
    return _TUNE


def _nearest(keys, value):
    ks = sorted(int(k) for k in keys)
    return str(min(ks, key=lambda k: abs(math.log(k) - math.log(max(value, 1)))))


def tuned_knobs(q_planes, k_group, head, window, max_seqlen_q, max_seqlen_k, num_seqs):
    """(tile, row_tiles, num_segments) from the table for this launch, or None."""
    tab = _tune_table()
    mode = "drafter" if (head == 128 and window > 0) else "decode1" if int(max_seqlen_q) == 1 else \
        "verify" if int(max_seqlen_q) <= 16 else "prefill"
    t = tab.get(f"{q_planes}:{k_group}")
    if mode == "drafter" and not (t and t.get("drafter")):      # the drafter was swept at K group 128
        t = tab.get(f"{q_planes}:128")
    if not t:
        return None
    m = t.get(mode) or t.get("verify")
    if not m:
        return None
    e = m[_nearest(m.keys(), int(max_seqlen_k) if mode == "prefill" else int(num_seqs))]
    return e["tile"], e["row_tiles"], e["num_segments"], e.get("reduce_warps")


def kv4h_attention(query, kv_cache, out, cu_seqlens_q, max_seqlen_q, seqused_k,
                   max_seqlen_k, softmax_scale, block_table, num_segments=None, causal=True,
                   window=0, output=None, tile=None, row_tiles=None, q_planes=None, k_group=None,
                   window_tps=None, reduce_warps=None, p_ref=None):
    """query: bf16 (num_tokens, num_q_heads, head); rotated and quantized to int8 here.
    out: bf16 scratch of the same shape (the rotated result when the launch has one segment).
    output: when given, the un-rotated result is written there (any strides) -- fused into the
    segment merge for split launches -- and returned; without it the caller un-rotates `out`.
    window > 0: sliding window of that many keys (vLLM's convention: key t of query at qabs is
    allowed iff qabs - t < window, and t - qabs < window too when non-causal)."""
    num_tokens, num_q_heads, head = query.shape
    k_group = min(k_group or K_GROUP, head)      # a 256-dim scale group is one per token at head 128
    L = H.layout(head, k_group)
    q_planes = q_planes or Q_PLANES
    num_seqs = int(seqused_k.shape[0])
    window = int(window or 0)
    # launch knobs: explicit argument > env (if set) > tune table > built-in default
    tuned = tuned_knobs(q_planes, k_group, head, window, max_seqlen_q, max_seqlen_k, num_seqs)
    prefill = int(max_seqlen_q) > 16
    if tile is None:
        tile_env = _ENV_SET["RADIANCE_KV4H_PREFILL_TILE" if prefill else "RADIANCE_KV4H_TILE"]
        tile = (TILE_PREFILL if prefill else TILE_DECODE) if (tile_env or not tuned) else tuned[0]
    if row_tiles is None:
        row_tiles = ROW_TILES if (_ENV_SET["RADIANCE_KV4H_ROWTILES"] or not tuned) else tuned[1]
    if num_segments is None:
        if _ENV_SET["RADIANCE_KV4H_SEGMENTS"] and SEGMENTS > 0 and not prefill:   # never for prefill
            assert SEGMENTS & (SEGMENTS - 1) == 0, f"RADIANCE_KV4H_SEGMENTS={SEGMENTS}: a power of two"
            num_segments = SEGMENTS
        elif tuned and not prefill:
            num_segments = tuned[2]
    p_ref = P_REF if p_ref is None else p_ref
    ext = _ext(row_tiles, q_planes, head // k_group, p_ref)
    assert query.dtype == torch.bfloat16 and kv_cache.dtype == torch.uint8
    assert tile in ext.TILES, (tile, ext.TILES)
    q8, q_descale = quantize_query(query, 127.0 if q_planes == 2 else 7.0)   # one Triton launch (a
    #                        kernel-prologue version cost as much in occupancy as it saved in launches)
    num_blocks, num_kv_heads, block_size, content = kv_cache.shape
    assert content == 2 * L.TPT, (content, L.TPT)
    # the kernel addresses one (block, head) region as block_size * 2 * TPT contiguous bytes
    assert kv_cache.stride(3) == 1 and kv_cache.stride(2) == content, kv_cache.stride()
    assert block_size > 0 and block_size % L.TG == 0, block_size
    assert out.is_contiguous() and out.shape == query.shape
    nqpkv = num_q_heads // num_kv_heads
    block_q = ext.BLOCK_M // nqpkv
    assert block_q > 0, (num_q_heads, num_kv_heads, ext.BLOCK_M)
    if num_seqs == 0 or num_tokens == 0:
        return output if output is not None else out
    num_q_blocks = num_tokens // block_q + num_seqs
    scale_log2 = (softmax_scale / head) * 1.4426950408889634
    # the split count only has to cover what the kernel reads: the window (plus the queries)
    eff_k = min(int(max_seqlen_k), window + int(max_seqlen_q)) if window > 0 else int(max_seqlen_k)

    auto_segments = num_segments is None
    if auto_segments:
        # AITER's split count for this shape (with the fork's tune installed): q / KV as fp8
        all_decode = int(max_seqlen_q) == 1
        cu_count = UA.get_num_sms()
        params = UA._UAParams(
            q=query, k=kv_cache, v=kv_cache, out=out, cu_seqlens_q=cu_seqlens_q, seqused_k=seqused_k,
            block_table=block_table, softmax_scale=softmax_scale, softcap=0.0, causal=bool(causal),
            sliding_window=0, max_seqlen_q=max_seqlen_q, max_seqlen_k=eff_k,
            num_tokens=num_tokens, num_query_heads=num_q_heads, num_kv_heads=num_kv_heads,
            num_queries_per_kv=nqpkv, head_size=head, head_size_v=head, num_seqs=num_seqs,
            total_num_q_blocks=num_q_blocks, num_2d_prgms=num_q_blocks * num_kv_heads,
            num_blocks=num_blocks, block_size=block_size, k_width=16, scale_k_width=4,
            block_scales_size=16, q_dtype=e4m3_dtype, kv_cache_dtype=e4m3_dtype,
            all_decode=all_decode, shuffled_kv_cache=False, use_alibi_slopes=False,
            use_qq_bias=False, num_sms=cu_count, target_num_prgms=cu_count * 4,
        )
        if UA.use_2d_kernel(params, "triton"):
            num_segments = 1
        else:
            num_segments = int(UU.get_unified_attention_config("kv_split", params, backend="triton")["NUM_SEGMENTS"])
    nseg = max(1, int(num_segments))
    if window > 0 and auto_segments:
        # a window of ~2k keys is ~65 tiles: WINDOW_TPS tiles per segment, not one (the reduce
        # costs too; graph-timed drafter call 78 us at 64 segments, 46 us at 8); AITER's reduce
        # wants a power of two
        cap = max(1, -(-eff_k // (tile * (window_tps or WINDOW_TPS))))
        nseg = min(nseg, 1 << (cap.bit_length() - 1))
    if nseg > 1:
        segm_out = torch.empty(num_tokens, num_q_heads, nseg, head, dtype=torch.float32, device=query.device)
        segm_max = torch.empty(num_tokens, num_q_heads, nseg, dtype=torch.float32, device=query.device)
        segm_sum = torch.empty_like(segm_max)
        so, sm, ss, o = segm_out.data_ptr(), segm_max.data_ptr(), segm_sum.data_ptr(), 0
    else:
        segm_out = segm_max = segm_sum = None
        so = sm = ss = 0
        o = out.data_ptr()
    bt = block_table if block_table.dtype == torch.int32 else block_table.to(torch.int32)
    assert bt.stride(1) == 1
    ext.launch(head, tile, q8.data_ptr(), q_descale.data_ptr(), kv_cache.data_ptr(), bt.data_ptr(),
               seqused_k.data_ptr(), cu_seqlens_q.data_ptr(), so, sm, ss, o, float(scale_log2),
               num_q_blocks, num_kv_heads, num_q_heads, nqpkv, block_q,
               kv_cache.stride(0), kv_cache.stride(1), block_size, bt.stride(0),
               nseg, num_seqs, int(bool(causal)), window, torch.cuda.current_stream().cuda_stream)
    LAST_LAUNCH["cfg"] = dict(nseg=nseg, block_q=block_q, q_blocks=num_q_blocks, head=head, window=window, tile=tile,
                              row_tiles=row_tiles, q_planes=q_planes, k_group=k_group, reduce_warps=reduce_warps,
                              p_ref=p_ref)
    if nseg == 1:
        return unrotate_output(out, output) if output is not None else out
    if output is not None:
        if reduce_warps is None and tuned and not prefill:
            reduce_warps = tuned[3]
        return reduce_unrotate(segm_out, segm_max, segm_sum, output, reduce_warps)
    # without `output`: the rotated result in `out`, merged by the same kernel without the
    # un-rotation (AITER's reduce_segments keeps segments by seq_len alone and would drop
    # segments the kernel filled from a causal prefix or a window)
    if reduce_warps is None and tuned and not prefill:
        reduce_warps = tuned[3]
    reduce_unrotate(segm_out, segm_max, segm_sum, out, reduce_warps, unrotate=False)
    return out


__all__ = ["kv4h_attention", "reduce_unrotate", "quantize_query", "unrotate_output", "quantize_query_torch",
           "unrotate_output_torch", "LAST_LAUNCH", "triton"]
