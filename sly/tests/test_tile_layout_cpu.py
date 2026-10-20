"""CPU check of the fragment-tiled layout index math vs row-major (skips without torch)."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "mxfp4"))
import tile_layout as T  # noqa: E402

SHAPES = [(1, 128), (16, 128), (17, 256), (513, 6144), (1536, 6144), (33, 17408)]


@pytest.mark.parametrize("M,K", SHAPES)
def test_tile_untile_roundtrip(M, K):
    a = torch.randint(0, 256, (M, K), dtype=torch.uint8)
    assert torch.equal(T.untile(T.tile(a), M, K), a)


@pytest.mark.parametrize("M,K", SHAPES)
def test_kernel_offset_matches_tile(M, K):
    a = torch.randint(0, 256, (M, K), dtype=torch.uint8)
    t = T.tile(a)
    g = torch.Generator().manual_seed(0)
    rows = torch.randint(0, M, (200,), generator=g).tolist() + [0, M - 1]
    cols = torch.randint(0, K, (200,), generator=g).tolist() + [0, K - 1]
    for r, c in zip(rows, cols):
        assert t[T.kernel_offset(r, c, K)] == a[r, c]


def test_fragment_is_256_contiguous_bytes_lane_order():
    K = 128
    a = torch.arange(32 * K, dtype=torch.int64).remainder(251).to(torch.uint8).view(32, K)
    t = T.tile(a)
    mt, ks = 1, 3                                   # fragment (1, 3)
    frag = t[(mt * (K // 16) + ks) * 256:][:256].view(32, 8)
    for lane in range(32):
        r, c0 = mt * 16 + lane % 16, ks * 16 + (lane // 16) * 8
        assert torch.equal(frag[lane], a[r, c0:c0 + 8])
