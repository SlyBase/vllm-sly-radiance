"""HIP W4A16 GEMV for the symmetric int4 projections (DFlash2 drafter, int4 lm_head) at M 1..16.

RADIANCE_W4A16_HIP_GEMV=1 routes `rdna_hybrid_w4a16_apply` (sly/patch_w4a16_hip_gemv.py) to
radiance_w4a16_gemv.hip's gemv_u4 for M 1..16 with symmetric weights (no zero point, no fused activation); everything
else keeps the stock path. Default 0.

The stock layout is ExLlama-shuffled nibbles (n 0..15, q = n - 8), int8 [N, K/2] / int32 [N, K/8] or the 4-D tiles
of patch_w4a16_tiles, plus bf16 scales [N, K/G]. The kernel reads its own copy, built lazily at the first eager call:
  codes  uint8  [ceil(N/16), K/64, 512]  16 rows x 64 K per 512 B, lane order and nibble order of the .hip header
  scales uint16 [ceil(N/16), K/64, 32]   bf16 bits, one per 32 K block and row: a G = 128 scale appears 4 times
The stock weight stays (prefill, M > 16, still reads it), so the copy costs ~1.125x the int4 bytes in extra memory,
bounded by RADIANCE_W4A16_MAX_CACHE_MB. The copy is never built while a CUDA graph is being captured: vLLM runs every
captured shape eagerly first, which fills the cache; a weight not cached by then simply keeps the stock kernel.
"""
import os
import sys

import torch

ENABLED = os.environ.get("RADIANCE_W4A16_HIP_GEMV", "0") == "1"
MAX_CACHE_MB = int(os.environ.get("RADIANCE_W4A16_MAX_CACHE_MB", "2048"))
MAX_M = 16  # rows per gemv_u4 launch (the kernel's range)
TILE = 16  # weight rows per tile (one wmma operand)
GROUP = 64  # K per tile group = two 32-K scale blocks
GROUPS = (32, 64, 128)  # scale group sizes; each divides into whole 32-K blocks
TILE_BYTES = 512 + 64  # code bytes + scale bytes of one (tile, group)
CHUNK = 16384  # rows per packing step: the lm_head (248320 rows) never builds [N, K] temporaries
# nibble position p of a code dword holds wmma slot _SLOT_OF_POS[p] (u4_decode emits slots from positions 0 2 4 6 1 3 5 7)
_SLOT_OF_POS = [0, 4, 1, 5, 2, 6, 3, 7]

_ext = None
if ENABLED:
    try:
        import radiance_w4a16_gemv as _ext
    except Exception as e:  # noqa: BLE001  # .so missing: the stock kernels serve
        sys.stderr.write(f"[radiance.w4a16] radiance_w4a16_gemv unavailable, stock kernels: {e!r}\n")
        _ext = None

_U4_CACHE: dict = {}  # (w_q ptr, w_s ptr, N, K, G) -> (codes, scales); False = refused by the budget
_U4_BYTES = 0
_SEEN: set = set()  # shapes that printed their proof line


def _is_gpu(t: torch.Tensor) -> bool:
    return t.is_cuda


def _stream(t: torch.Tensor) -> int:
    return torch.cuda.current_stream().cuda_stream


def _int32(w: torch.Tensor) -> torch.Tensor:
    """Stock weight in any of its forms -> ExLlama int32 [N, K/8]. The 4-D form is patch_w4a16_tiles' LAYOUT 1
    [N/16, K/128, 16, 64] int8, whose int32 view [N/16, K/128, 16, 16] is [row block, K block, row, word]."""
    w32 = w.view(torch.int32) if w.dtype == torch.int8 else w
    if w32.dim() == 4:
        w32 = w32.permute(0, 2, 1, 3).reshape(w32.shape[0] * 16, w32.shape[1] * 16)
    return w32


def unpack_exllama_nibbles(w_packed: torch.Tensor) -> torch.Tensor:
    """ExLlama-shuffled weights -> uint8 [N, K] nibbles 0..15 in K order. An int32 packs K values
    v0 | v2 << 4 | v4 << 8 | v6 << 12 | v1 << 16 | v3 << 20 | v5 << 24 | v7 << 28."""
    w32 = _int32(w_packed)
    n, k8 = w32.shape
    out = torch.empty((n, k8, 8), dtype=torch.uint8, device=w32.device)
    for i in range(8):  # K offset i sits at bit 16 (i % 2) + 4 (i // 2); one int32 temporary at a time
        out[:, :, i] = ((w32 >> (16 * (i & 1) + 4 * (i >> 1))) & 0xF).to(torch.uint8)
    return out.reshape(n, k8 * 8)


@torch.no_grad()
def pack_u4(w_packed: torch.Tensor, w_scale: torch.Tensor, group_size: int):
    """(stock weight, bf16 scales [N, K/G]) -> (codes u8 [NT, K/64, 512], scales u16 [NT, K/64, 32]).

    Per (tile of 16 rows, group of 64 K) the codes are [half h][row c][block b][step j][4 B]: lane 16 h + c of a wave
    reads its 16 B at lane * 16; its dword 2 b + j holds the codes of K 64 g + 32 b + 16 h + 8 j + slot, nibble
    position p <- slot _SLOT_OF_POS[p]. Scales are [row c][block b] bf16 bits. Rows past N: code 0, scale 0."""
    w32 = _int32(w_packed)
    n, k = w32.shape[0], w32.shape[1] * 8
    assert group_size in GROUPS and k % GROUP == 0 and k % group_size == 0, (group_size, k)
    assert w_scale.dtype == torch.bfloat16 and tuple(w_scale.shape) == (n, k // group_size), (w_scale.shape, n, k)
    assert CHUNK % TILE == 0
    nt, kg = -(-n // TILE), k // GROUP
    qt = torch.zeros((nt, kg, 512), dtype=torch.uint8, device=w32.device)
    st = torch.zeros((nt, kg, 32), dtype=torch.bfloat16, device=w32.device)
    for r0 in range(0, n, CHUNK):
        r1 = min(r0 + CHUNK, n)
        nc = -(-(r1 - r0) // TILE)  # tiles of this chunk
        pad = nc * TILE - (r1 - r0)
        u = unpack_exllama_nibbles(w32[r0:r1])
        sc = w_scale[r0:r1].repeat_interleave(group_size // 32, dim=1)  # one scale per 32-K block
        if pad:
            u = torch.cat([u, u.new_zeros(pad, k)])
            sc = torch.cat([sc, sc.new_zeros(pad, k // 32)])
        codes = u.view(nc, TILE, kg, 2, 2, 2, 8)[..., _SLOT_OF_POS]  # [t, c, g, b, h, j, position]
        byts = codes[..., 0::2] | (codes[..., 1::2] << 4)  # [t, c, g, b, h, j, 4]
        t0 = r0 // TILE
        qt[t0:t0 + nc] = byts.permute(0, 2, 4, 1, 3, 5, 6).reshape(nc, kg, 512)  # [t, g, h, c, b, j, 4]
        st[t0:t0 + nc] = sc.reshape(nc, TILE, kg, 2).permute(0, 2, 1, 3).reshape(nc, kg, 32)
    return qt, st.view(torch.uint16)


def _nk(w_q: torch.Tensor):
    """(N, K) of a stock int4 weight, None if it is not one of the layouts above."""
    if w_q.dim() == 4 and w_q.dtype == torch.int8 and tuple(w_q.shape[2:]) == (16, 64):
        return w_q.shape[0] * 16, w_q.shape[1] * 128
    if w_q.dim() == 2 and w_q.dtype == torch.int8:
        return w_q.shape[0], w_q.shape[1] * 2
    if w_q.dim() == 2 and w_q.dtype == torch.int32:
        return w_q.shape[0], w_q.shape[1] * 8
    return None


def _tiles(w_q, w_s, n, k, g):
    """The kernel's copy of a weight: built once, never during graph capture, bounded by the memory budget."""
    global _U4_BYTES
    key = (w_q.data_ptr(), w_s.data_ptr(), n, k, g)
    ent = _U4_CACHE.get(key)
    if ent is not None:
        return ent or None
    if torch.cuda.is_current_stream_capturing():  # an allocation here would be baked into the graph
        return None
    need = -(-n // TILE) * (k // GROUP) * TILE_BYTES
    if _U4_BYTES + need > MAX_CACHE_MB << 20:
        _U4_CACHE[key] = False
        sys.stderr.write(f"[radiance.w4a16] gemv_u4 {n}x{k} g{g}: {need >> 20} MiB would pass the "
                         f"RADIANCE_W4A16_MAX_CACHE_MB={MAX_CACHE_MB} budget ({_U4_BYTES >> 20} MiB held), stock kernel\n")
        return None
    ent = _U4_CACHE[key] = pack_u4(w_q, w_s, g)
    _U4_BYTES += need
    if (n, k, g) not in _SEEN:
        _SEEN.add((n, k, g))
        sys.stderr.write(f"[radiance.w4a16] gemv_u4 tiled {n}x{k} g{g}: {need / 2 ** 20:.1f} MiB "
                         f"(cache {_U4_BYTES >> 20} MiB)\n")
    return ent


def prepack_weights(w_q, w_s, w_zp, group_size: int, prefix: str = "") -> bool:
    """Build the kernel's tile copy at load time (before profile_run and KV-cache sizing) so the KV budget
    sees the allocation. Skips merged MLP gate_up_proj when RADIANCE_W4A16_SILU (default 1) bypasses apply."""
    if not ENABLED or _ext is None or w_zp is not None or group_size not in GROUPS:
        return False
    if os.environ.get("RADIANCE_W4A16_SILU", "1") != "0" and ".gate_up_proj" in prefix:
        return False
    if w_s is None or w_s.dtype != torch.bfloat16 or not _is_gpu(w_q):
        return False
    nk = _nk(w_q)
    if nk is None:
        return False
    n, k = nk
    if k % GROUP or k % group_size or tuple(w_s.shape) != (n, k // group_size):
        return False
    return _tiles(w_q, w_s, n, k, group_size) is not None


def try_w4a16_hip_gemv(x, w_q, w_s, w_zp, group_size, bias=None, act_fn=None):
    """y = x @ W^T (+ bias) on the HIP GEMV; None = not ours (the caller runs the stock kernel)."""
    if not ENABLED or _ext is None or act_fn is not None or w_zp is not None:
        return None
    if x.dtype != torch.bfloat16 or w_s.dtype != torch.bfloat16 or group_size not in GROUPS or not _is_gpu(x):
        return None
    nk = _nk(w_q)
    if nk is None:
        return None
    n, k = nk
    if x.shape[-1] != k or k % GROUP or k % group_size or tuple(w_s.shape) != (n, k // group_size):
        return None
    m = x.numel() // k
    if not 0 < m <= MAX_M:
        return None
    if bias is not None and (bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,) or bias.stride(0) != 1):
        return None
    ent = _tiles(w_q, w_s, n, k, group_size)
    if ent is None:
        return None
    qt, st = ent
    x2 = x.reshape(m, k)
    if x2.stride(-1) != 1 or x2.stride(0) % 8 or x2.data_ptr() % 16:  # the kernel's 16 B vector loads
        x2 = x2.clone(memory_format=torch.contiguous_format)
    y = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _ext.gemv_u4(x2.data_ptr(), qt.data_ptr(), st.data_ptr(), y.data_ptr(), bias.data_ptr() if bias is not None else 0,
                 m, n, k, x2.stride(0), y.stride(0), 0, qt.shape[0], _stream(x2), -1)
    return y.reshape(x.shape[:-1] + (n,))
