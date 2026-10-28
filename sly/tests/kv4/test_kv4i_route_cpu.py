"""CPU test for the int4 libr4d route decision (sly/kv4/kv4i_route.py).

The route is a guard, and a guard that is wrong in the permissive direction corrupts output instead of
falling back, so every decline reason and the one accept case are pinned here. Also pins the block
table formula: vLLM block ids -> libr4d's 16-token group ids (block * 104 + group).

Pure Python, no torch, no vllm, no GPU: runs on the bare CI runner.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kv4"))

import kv4i_route as R  # noqa: E402

OK = dict(enabled=True, has_entry=True, head=256, k_group=256, cache_block=R.KV4_BLOCK,
          cache_shape_ok=True, cache_strides_ok=True, window=0, causal=True, fused_out=False,
          out_dtype="bfloat16", capturing=False, max_query_len=4096, max_seq_len=49152, num_tokens=4096,
          num_seqs=1, min_q=512, min_ctx=R.MIN_CTX_DEFAULT)


def _decline(**over):
    kw = dict(OK)
    kw.update(over)
    return R.decline(**kw)


def test_accepts_a_long_causal_prefill():
    assert _decline() is None


def test_refuses_every_shape_it_cannot_read():
    cases = {
        "R4D_KV4I unset": dict(enabled=False),
        "no int4 prefill entry": dict(has_entry=False),
        "head 128": dict(head=128),
        "K scale group 128": dict(k_group=128),
        "no 4-D KV4 cache": dict(cache_shape_ok=False, cache_block=-1),
        "KV4 block 64": dict(cache_block=64, ),
        "not contiguous per block/head": dict(cache_strides_ok=False),
        "sliding window": dict(window=2048),
        "non-causal": dict(causal=False),
        "fused output": dict(fused_out=True),
        "output dtype": dict(out_dtype="float16"),
        "graph capture": dict(capturing=True),
        "max query length 8": dict(max_query_len=8, num_tokens=8),
        "ragged or short": dict(num_tokens=4096 + 8, num_seqs=2),          # 4096 + 8 in 2 runs
        "ragged or short query runs (12288": dict(num_tokens=12288, num_seqs=2, max_query_len=8192),
        "no query runs": dict(num_seqs=0, num_tokens=0),
        # an explicit RADIANCE_KV4I_MIN_CTX (default 0 = no gate) still declines below it
        "context 8192": dict(max_seq_len=8192, min_ctx=16384),
    }
    for needle, over in cases.items():
        got = _decline(**over)
        assert got is not None and needle in got, f"{over} accepted as {got!r}"


def test_a_long_run_at_shallow_context_takes_the_route_by_default():
    # 4096 new tokens, 4096 keys: with the Triton glue the route wins here too (+1.7 % at 6000), so no gate
    assert R.MIN_CTX_DEFAULT == 0
    assert _decline(max_query_len=4096, max_seq_len=4096) is None
    # an explicit RADIANCE_KV4I_MIN_CTX still keeps a shallow run on Tetra, and a deep one routed
    assert _decline(max_query_len=4096, max_seq_len=4096, min_ctx=16384) is not None
    assert _decline(max_query_len=4096, max_seq_len=49152, min_ctx=16384) is None
    # two equal runs batch fine, two unequal ones do not
    assert _decline(num_tokens=8192, num_seqs=2) is None
    assert _decline(num_tokens=8000, num_seqs=2) is not None


def test_none_causal_is_not_true():
    # the drafter's metadata carries causal=False, but a missing flag must not be read as causal
    assert _decline(causal=None) is not None


def test_group_row_follows_the_kernel_layout():
    # two KV4 blocks, three groups each: ids 7 and 9 -> 7*3+(0,1,2) then 9*3+(0,1,2)
    assert R.group_row([7, 9], gpb=3) == [21, 22, 23, 27, 28, 29]
    # the production granularity: one 1664-token block is 104 groups
    row = R.group_row([5])
    assert row[0] == 5 * R.KV4_GPB and row[-1] == 5 * R.KV4_GPB + R.KV4_GPB - 1 and len(row) == R.KV4_GPB


def test_group_table_pads_for_the_prefetch():
    t = R.group_table([[1], [2, 3]], gpb=2)
    assert len(t) == 2
    # width of the widest row (2 blocks x 2 groups = 4), then the 3-entry prefetch pad
    assert t[0] == [2, 3, 0, 0, 0, 0, 0]
    assert t[1] == [4, 5, 6, 7, 0, 0, 0]
    assert all(len(r) == len(t[0]) for r in t)
