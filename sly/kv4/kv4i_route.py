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
# The routed kernel is 1.01x of the fp8 one. With the first route's torch glue (fp32 256x256 matmuls for the
# q rotation and output un-rotation, ~3 ms per layer launch at 4096 tokens) the route measured -5.9/-5.1/-3.8/
# -1.1 % prefill at 1.5k/5.9k/11.8k/23.5k and +4.2 % at 47k (210 W), hence the old gate of 16384. The Triton
# glue (RADIANCE_KV4I_GLUE=triton, ~0.5 ms) removed that cost: engine sweep, R9700 300 W, CHUNK 4096, route vs
# TetraAttention +0.6/+1.7/+3.4/+6.7/+12.3 % at 1514/6000/12000/24000/48000, and KL against an fp8-KV reference
# 0.01709 vs Tetra's 0.02026 over four 48k-token documents, lower in every context bucket from 0-4K up (torch
# glue: 0.01705). So no context gate by default; RADIANCE_KV4I_MIN_CTX=N still sets one.
MIN_CTX_DEFAULT = 0


def decline(*, enabled, has_entry, head, k_group, cache_block, cache_shape_ok, cache_strides_ok,
            window, causal, fused_out, out_dtype, capturing, max_query_len, max_seq_len, num_tokens,
            num_seqs, min_q, min_ctx) -> str | None:
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
    if max_seq_len < min_ctx:
        return f"context {max_seq_len} < {min_ctx} keys (the route's rotation costs more than it wins)"
    if not num_seqs:
        return "no query runs"
    # every run <= max_query_len, so the total equals seqs * max_query_len only if ALL runs are equal
    # to it (and max_query_len >= min_q was checked above): no device read needed
    if num_tokens != num_seqs * max_query_len:
        return (f"ragged or short query runs ({num_tokens} tokens in {num_seqs} sequences, longest "
                f"{max_query_len}; the kernel takes one q_len per launch)")
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
