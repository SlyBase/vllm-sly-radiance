"""KV scatter for the Inkling 2 DSpark draft on ROCm.

vLLM's FlashAttentionImpl.do_kv_cache_update calls `reshape_and_cache_flash`, a CUDA op that is not
importable on ROCm. Same write as one Triton kernel: no data-dependent shapes, so it can sit inside a
captured CUDA/HIP graph. Cache layout (blocks, heads, page, 2 * head_dim) = K | V; slot < 0 = padding.
"""
import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _scatter(Cache, K, V, Slots, sb, sh, sp, H: tl.constexpr, D: tl.constexpr,
             PAGE: tl.constexpr, BD: tl.constexpr, ks, vs):
    t = tl.program_id(0)
    slot = tl.load(Slots + t)
    if slot < 0:
        return
    h = tl.program_id(1)
    d = tl.arange(0, BD)
    m = d < D
    base = (slot // PAGE).to(tl.int64) * sb + h * sh + (slot % PAGE).to(tl.int64) * sp
    tl.store(Cache + base + d, tl.load(K + t * ks + h * D + d, m, other=0), m)
    tl.store(Cache + base + D + d, tl.load(V + t * vs + h * D + d, m, other=0), m)


def kv_scatter(cache, k, v, slots):
    """cache[slot // page, :, slot % page] = k | v for every slot >= 0 (k, v: [T, H, D])."""
    T, H, D = k.shape
    assert k.stride(-1) == 1 and v.stride(-1) == 1 and k.stride(1) == D and v.stride(1) == D
    _scatter[(T, H)](cache, k, v, slots, cache.stride(0), cache.stride(1), cache.stride(2),
                     H=H, D=D, PAGE=cache.shape[2], BD=triton.next_power_of_2(D),
                     ks=k.stride(0), vs=v.stride(0))
