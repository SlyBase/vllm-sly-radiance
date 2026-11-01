"""CPU check of sly/kv4/kv4i_offline.py on a synthetic dump: the truth must agree with plain SDPA on the
decoded cache, a dump whose output IS the truth scores 0, and the small-norm query that the folded
scale / HEAD pushes into e4m3 subnormals must hurt the emulated kernel more than the K/16 split."""
import math
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kv4"))
import radiance_kv4_format as F  # noqa: E402
import radiance_kv4h_format as H  # noqa: E402
import kv4i_offline as O  # noqa: E402


def _dump(ctx=1800, qlen=64, nh=6, nkv=1, qsig=0.6):
    g = torch.Generator().manual_seed(3)
    k = torch.randn(nkv, ctx, 256, generator=g) * 2
    v = torch.randn(nkv, ctx, 256, generator=g)
    nblk = -(-ctx // O.BS)
    kv = torch.zeros(nblk, nkv, O.BS, 272, dtype=torch.uint8)
    for b in range(nblk):
        for h in range(nkv):
            sl = slice(b * O.BS, min(ctx, (b + 1) * O.BS))
            kv[b, h] = H.encode_block(k[h, sl], v[h, sl], O.BS).reshape(O.BS, 272)
    query = torch.randn(qlen, nh, 256, generator=g) * qsig
    scale = 1 / 16
    kr, vr = O.decode(kv, ctx)
    qpos = torch.arange(ctx - qlen, ctx)
    truth = torch.stack([F.unrotate(torch.softmax(
        (F.rotate(query[:, h]) @ kr[0].T * scale / 256).masked_fill(
            torch.arange(ctx)[None] > qpos[:, None], -math.inf), -1) @ vr[0]) for h in range(nh)], 1)
    q8 = (F.rotate(query) / 16).to(torch.float8_e4m3fn)
    return dict(query=query, q8=q8, q_scale=16.0, scale=scale, kv=kv, seq_lens=torch.tensor([ctx]),
                query_start_loc=torch.tensor([0, qlen]), num_heads=nh, num_kv_heads=nkv,
                out=truth.clone(), ref=truth.clone())


def test_truth_matches_sdpa_and_a_perfect_dump_scores_zero():
    rel, _, info = O.analyse(_dump(), rows=64)
    assert info["rows"] == 64
    assert rel["route (dump)"].max() < 1e-5 and rel["tetra (dump)"].max() < 1e-5


def test_folded_query_underflow_is_visible_and_the_k16_split_removes_it():
    rel, fold, _ = O.analyse(_dump(), rows=64)
    assert fold.mean() > 0.5, "test query does not reach the e4m3 subnormal range"
    assert rel["Q only"].mean() > 2 * rel["fix K/16"].mean()
