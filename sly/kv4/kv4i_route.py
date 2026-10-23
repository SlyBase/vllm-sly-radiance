"""The decision "may this launch go to the int4 libr4d prefill kernel?", as plain values.

No torch, no vllm: the back end reads its objects, this module decides. That keeps the guard and the
block-table math checkable on a CPU-only runner (sly/tests/kv4/test_kv4i_route_cpu.py) while the
launch itself stays in radiance_kv4h_attn.py.

The kernel (sly/r4d/r4d_kv4i.patch) reads KV4 blocks of 1664 tokens as 104 groups of 16 tokens, takes
ONE query length for the whole launch, is causal-only, head 256, K scale group 256, and reads
`seqused_k` as one int32 per sequence.
"""
KV4_GPB = 104                       # 16-token groups per 1664-token KV4 block (kernel's KV4_GPB)
KV4_BLOCK = KV4_GPB * 16            # 1664 tokens per block
GROUP = 16                          # tokens per table entry (the kernel's BS)
TABLE_PAD = 3                       # the double-buffered prefetch reads up to 3 groups past the row
MIN_Q_DEFAULT = 512                 # smallest query length that pays for the libr4d launch


def decline(*, enabled, has_entry, head, k_group, cache_block, cache_shape_ok, cache_strides_ok,
            window, causal, fused_out, out_dtype, capturing, max_query_len, runs, min_q) -> str | None:
    """Why the launch must stay on TetraAttention, or None when libr4d takes it. First hit wins."""
    if not enabled:
        return "R4D_KV4I unset"
    if not has_entry:
        return "this r4d has no int4 prefill entry"
    if head != 256:
        return f"head {head} (libr4d int4 reads head 256)"
    if k_group != 256:
        return f"K scale group {k_group} (libr4d int4 reads 256)"
    if not cache_shape_ok:
        return "no 4-D KV4 cache"
    if cache_block != KV4_BLOCK:
        return f"KV4 block {cache_block} (kernel is built for {KV4_BLOCK})"
    if not cache_strides_ok:
        return "KV4 cache is not contiguous per block/head"
    if window:
        return f"sliding window {window}"
    if causal is not True:
        return "non-causal batch (the drafter's)"
    if fused_out:
        return "fused output quantisation"
    if out_dtype != "bfloat16":
        return f"output dtype {out_dtype}"
    if capturing:
        return "graph capture"
    if max_query_len < min_q:
        return f"max query length {max_query_len} < {min_q}"
    if not runs:
        return "no query runs"
    lo, hi = min(runs), max(runs)
    if lo < min_q:
        return f"a sequence carries {lo} < {min_q} new tokens"
    if lo != hi:
        return f"ragged query lengths {lo}..{hi} (the kernel takes one q_len per launch)"
    return None


def group_row(block_ids, gpb: int = KV4_GPB) -> list[int]:
    """One sequence's vLLM block ids -> libr4d's 16-token group ids, in order.

    Group G of a KV4 block b is addressed as b * gpb + j (j = 0..gpb-1), because the cache layout
    inside a (block, head) region is gpb consecutive 2176-byte groups of 16 tokens.
    """
    out = []
    for b in block_ids:
        base = int(b) * gpb
        out.extend(range(base, base + gpb))
    return out


def group_table(rows, gpb: int = KV4_GPB, pad: int = TABLE_PAD) -> list[list[int]]:
    """The kernel's block table: one row per sequence, padded so the prefetch never reads past the end."""
    table = [group_row(r, gpb) for r in rows]
    if table:
        width = max(len(r) for r in table)
        table = [r + [0] * (width - len(r)) for r in table]          # ragged rows: same width, zero groups
    return [r + [0] * pad for r in table]
