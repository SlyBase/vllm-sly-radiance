"""NVFP4 expert GEMM without a native FP4 MMA (gfx1201): dequantize tiles, bf16 dot.

Reference = dequantize both operands in torch, fp32 matmul, then the per-expert and per-row outer
scales -- exactly what `tl.dot_scaled(e2m1) * ExpertScale * RowScale` computes on Blackwell.
"""
import types

import pytest
import torch

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def dequant(packed, scales):
    """[R, K/2] uint8 (low nibble = even column) + [R, K/16] e4m3 -> [R, K] fp32."""
    lo, hi = (packed & 0xF).long(), (packed >> 4).long()
    codes = torch.stack([lo, hi], -1).reshape(packed.shape[0], -1)
    table = torch.tensor(E2M1)
    vals = table[codes & 7] * torch.where(codes >> 3 == 1, -1.0, 1.0)
    return vals * scales.float().repeat_interleave(16, 1)


def rand_operand(rows, k, gen):
    packed = torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, generator=gen)
    # e4m3 scales in a sane band (no NaN/zero): 2^-2 .. 2^2
    scales = (torch.rand(rows, k // 16, generator=gen) * 3.75 + 0.25).to(torch.float8_e4m3fn)
    return packed, scales


def r8c4(scales):
    """[R, C] row-major -> R8c4 flat: (r, c) at [r//8, c//4, r%8, c%4] (block_scales._scale_offsets)."""
    r, c = scales.shape
    t = scales.reshape(r // 8, 8, c // 4, 4).permute(0, 2, 1, 3)
    return t.reshape(-1).contiguous()


def make_case(experts=2, n=32, k=64, tile=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    w, ws = zip(*[rand_operand(n, k, gen) for _ in range(experts)])
    weight, wscale = torch.stack(w), torch.stack(ws)
    max_ctas = experts
    rows = tile * max_ctas
    a, a_s = rand_operand(rows, k, gen)
    # tiles: [expert id per CTA | row_end per CTA | active count]; every tile full except the last
    row_end = torch.tensor([tile, tile * 2 - 3][:max_ctas], dtype=torch.int32)
    tiles = torch.cat([torch.arange(experts, dtype=torch.int32), row_end,
                       torch.tensor([max_ctas], dtype=torch.int32)])
    expert_scale = torch.tensor([0.5, 2.0][:experts])
    row_scale = torch.rand(rows, generator=gen) + 0.5
    plan = types.SimpleNamespace(tile=tile, max_ctas=max_ctas, rows=rows)
    return dict(act=a, act_scale=r8c4(a_s), a_scale_rm=a_s, row_scale=row_scale, weight=weight,
                weight_scale=wscale, expert_scale=expert_scale, tiles=tiles, plan=plan,
                row_end=row_end, n=n, k=k)


def reference(c):
    plan, out = c["plan"], torch.zeros(c["plan"].rows, c["n"])
    a = dequant(c["act"], c["a_scale_rm"])
    for t in range(plan.max_ctas):
        e = int(c["tiles"][t])
        lo, hi = t * plan.tile, min(int(c["row_end"][t]), (t + 1) * plan.tile)
        w = dequant(c["weight"][e], c["weight_scale"][e])
        out[lo:hi] = (a[lo:hi] @ w.T) * c["expert_scale"][e] * c["row_scale"][lo:hi, None]
    return out


def run_and_compare(c):
    import triton.language as tl
    from radiance_inkling2_moe import nvfp4_expert_gemm

    out = nvfp4_expert_gemm(c["act"], c["act_scale"], c["row_scale"], c["weight"],
                            c["weight_scale"], c["expert_scale"], c["tiles"], c["plan"],
                            dot_dtype=tl.float32)  # bf16 is exact too; the CPU interpreter is not
    ref = reference(c)
    valid = torch.zeros(c["plan"].rows, dtype=torch.bool)
    for t in range(c["plan"].max_ctas):
        valid[t * c["plan"].tile: min(int(c["row_end"][t]), (t + 1) * c["plan"].tile)] = True
    assert out.shape == ref.shape
    assert ref[valid].abs().max() > 0  # the comparison must not be vacuous
    torch.testing.assert_close(out[valid].float(), ref[valid], rtol=2e-2, atol=2e-2)  # bf16 output


def test_dequant_gemm_matches_reference():
    run_and_compare(make_case())


def test_one_hot_activation_selects_one_weight_column():
    """Sparse input: column 2 = code 4 (2.0) only. (The nibble order itself is not testable here:
    both operands use the same order, and swapping a pair inside one 16-column scale group leaves
    the dot product unchanged -- an equivalent mutant, not a gap.)"""
    c = make_case()
    c["act"].zero_()
    c["act"][:, 1] = 0x04
    run_and_compare(c)


def test_ragged_k_and_n_not_multiples_of_the_block():
    run_and_compare(make_case(n=48, k=192, seed=3))  # N=48 with BN=64, K=192 with BK=128
