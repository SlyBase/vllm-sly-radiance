"""Triton KV scatter for the Inkling 2 DSpark draft (replaces the CUDA-only reshape_and_cache_flash)."""
import pytest
import torch


def reference(cache, k, v, slots):
    out = cache.clone()
    page, d = cache.shape[2], k.shape[-1]
    for t, s in enumerate(slots.tolist()):
        if s >= 0:
            out[s // page, :, s % page, :d] = k[t]
            out[s // page, :, s % page, d:] = v[t]
    return out


@pytest.mark.parametrize("slots", [[3, 9, -1, 17, 31], [-1, -1], [0, 1, 2, 3, 4, 5, 6, 7]])
def test_scatter_matches_reference(slots):
    from radiance_inkling2_kv import kv_scatter

    B, H, N, D = 4, 2, 8, 16
    g = torch.Generator().manual_seed(0)
    cache = torch.randn(B, H, N, 2 * D, generator=g)
    k, v = torch.randn(len(slots), H, D, generator=g), torch.randn(len(slots), H, D, generator=g)
    s = torch.tensor(slots, dtype=torch.int64)
    want = reference(cache, k, v, s)
    kv_scatter(cache, k, v, s)
    torch.testing.assert_close(cache, want, rtol=0, atol=0)


def test_non_contiguous_key_and_value_views():
    """k, v are slices of the fused qkv tensor: row stride != H * D."""
    from radiance_inkling2_kv import kv_scatter

    H, D, N = 2, 16, 8
    qkv = torch.randn(3, 5 * H * D)
    k, v = qkv[:, H * D:2 * H * D].view(3, H, D), qkv[:, 2 * H * D:3 * H * D].view(3, H, D)
    cache = torch.zeros(2, H, N, 2 * D)
    s = torch.tensor([1, 8, -1])
    want = reference(cache, k, v, s)
    kv_scatter(cache, k, v, s)
    torch.testing.assert_close(cache, want, rtol=0, atol=0)
