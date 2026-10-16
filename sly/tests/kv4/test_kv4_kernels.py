#!/usr/bin/env python3
"""GPU tests for the KV4 kernels: the store kernel against the torch reference bit for bit, the
attention kernel against an fp32 reference. Runs inside the serving image (triton, torch, vllm).

Run:  python3 sly/tests/kv4/test_kv4_kernels.py            (exit 0 = pass)
      KV4_TESTS=store python3 ...                           (subset: store | attn)
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kv4"))

import radiance_kv4_format as F  # noqa: E402
from radiance_kv4_store import kv4_store  # noqa: E402
from radiance_kv4_kernel import kv4_attention, quantize_query, unrotate_output  # noqa: E402

dev = "cuda"
torch.manual_seed(0)
NKV = 4


def _kv(n, kind="outlier"):
    k = torch.randn(n, NKV, F.HEAD, device=dev)
    v = torch.randn(n, NKV, F.HEAD, device=dev)
    if kind == "outlier":
        k[..., :6] *= 20.0
        k[..., 100] *= 12.0
        v *= torch.empty(n, NKV, 1, device=dev).uniform_(0.2, 3.0)
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


def _check_store(name, n=1000, bs=880, num_blocks=4, with_pad=True):
    k, v = _kv(n)
    cache = torch.zeros(num_blocks, NKV, bs, F.CONTENT, dtype=torch.uint8, device=dev)
    slots = torch.randperm(num_blocks * bs, device=dev)[:n].to(torch.int64)
    if with_pad:
        slots[::7] = -1  # vLLM pads the slot mapping with -1
    kv4_store(k, v, cache, slots, **({} if name is None else {"codebooks": (name, name)}))
    kf, vf = k.float().reshape(-1, F.HEAD), v.float().reshape(-1, F.HEAD)
    ref = F.encode_content(kf, vf) if name is None else torch.cat([F.encode_side(kf, name, F.NGROUP_K), F.encode_side(vf, name, F.NGROUP_V)], -1)
    ref = ref.reshape(n, NKV, F.CONTENT)
    live = slots >= 0
    got = cache[slots[live] // bs, :, slots[live] % bs, :]  # (n_live, NKV, CONTENT)
    exp = ref[live]
    diff = (got != exp)
    if diff.any():
        bad = diff.sum().item()
        # show where: scales or nibbles, K or V
        where = diff.nonzero()[:8].tolist()
        raise AssertionError(f"{name}: {bad} of {diff.numel()} bytes differ, first at {where}")
    # padded slots untouched: every cell not written stays zero
    written = torch.zeros(num_blocks * bs, dtype=torch.bool, device=dev)
    written[slots[live]] = True
    untouched = cache.permute(0, 2, 1, 3).reshape(num_blocks * bs, -1)[~written]
    assert int(untouched.abs().sum()) == 0
    print(f"ok   store {name}: {int(live.sum())} tokens x {NKV} heads bit-identical to the reference")


def test_store_uniform():
    _check_store("uniform")


def test_store_lloyd():
    _check_store("lloyd")


def test_store_geo():
    _check_store("geo")


def test_store_kernel_format():
    _check_store(None)


def test_store_small_block():
    _check_store(None, n=200, bs=16, num_blocks=20)


def test_store_roundtrip_values():
    """What the attention kernel will see equals the reference's view of the same bytes."""
    k, v = _kv(64)
    cache = torch.zeros(1, NKV, 880, F.CONTENT, dtype=torch.uint8, device=dev)
    slots = torch.arange(64, device=dev, dtype=torch.int64)
    kv4_store(k, v, cache, slots)
    cells = cache[0, :, :64, :].permute(1, 0, 2).reshape(64 * NKV, F.CONTENT)
    kk, vv = F.decode_content(cells)
    k_rot = F.rotate(k.float().reshape(-1, F.HEAD))
    rel = float((kk - k_rot).norm() / k_rot.norm())
    assert rel < 0.12, rel
    print(f"ok   store roundtrip: rotated-K relative error {rel:.4f}")


# ---------------------------------------------------------------------------------------------
# attention kernel
# ---------------------------------------------------------------------------------------------

NQ = 24
SCALE = 1.0 / (F.HEAD ** 0.5)


def _ref_attn(q, k_all, v_all, seq_lens, q_lens):
    """fp32 causal GQA reference in the original basis. q (sum q, NQ, HEAD); k_all/v_all lists of
    (seq_len, NKV, HEAD)."""
    outs, qi = [], 0
    g = NQ // NKV
    for s, (sl, ql) in enumerate(zip(seq_lens, q_lens)):
        qs = q[qi:qi + ql].float().view(ql, NKV, g, F.HEAD)
        ks, vs = k_all[s].float(), v_all[s].float()
        sc = torch.einsum("qhgd,khd->hgqk", qs, ks) * SCALE
        ctx = sl - ql
        qpos = torch.arange(ql, device=dev)[:, None] + ctx
        kpos = torch.arange(sl, device=dev)[None, :]
        sc.masked_fill_(kpos > qpos, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        outs.append(torch.einsum("hgqk,khd->qhgd", p, vs).reshape(ql, NQ, F.HEAD))
        del sc, p
        qi += ql
    return torch.cat(outs)


def _rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def _check_attn(label, seq_lens, q_lens, bs, name="uniform", tol=0.04):
    nseq = len(seq_lens)
    blocks_per = [-(-s // bs) for s in seq_lens]
    num_blocks = sum(blocks_per) + 1
    block_table = torch.zeros(nseq, max(blocks_per), dtype=torch.int32, device=dev)
    cache = torch.zeros(num_blocks, NKV, bs, F.CONTENT, dtype=torch.uint8, device=dev)
    k_all, v_all, nb = [], [], 1
    for s in range(nseq):
        ids = torch.arange(nb, nb + blocks_per[s], device=dev)
        block_table[s, :blocks_per[s]] = ids.to(torch.int32)
        nb += blocks_per[s]
        k, v = _kv(seq_lens[s])
        k_all.append(k)
        v_all.append(v)
        pos = torch.arange(seq_lens[s], device=dev)
        slots = (ids[pos // bs] * bs + pos % bs).to(torch.int64)
        kv4_store(k, v, cache, slots)
    total_q = sum(q_lens)
    q = torch.randn(total_q, NQ, F.HEAD, device=dev).to(torch.bfloat16)
    cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    seqused = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    q8, desc = quantize_query(q)
    out = torch.empty(total_q, NQ, F.HEAD, dtype=torch.bfloat16, device=dev)
    kv4_attention(q8, desc, cache, out, cu_q, max(q_lens), seqused, max(seq_lens), SCALE, block_table)
    torch.cuda.synchronize()
    got = unrotate_output(out)
    assert torch.isfinite(got).all(), "non-finite output"
    # A: what the kernel is supposed to compute -- attention over the reconstructed K/V (exact
    #    codebook values, no fp8 rounding: the kernel's dots see the integers and apply the scales
    #    on the score tiles) with the fp8-quantized q; differs from the kernel only by the fp8
    #    rounding of P and the summation order
    q_deq = unrotate_output((q8.float() * desc[..., None]))
    k_rec = [F.reconstruct(k.float().reshape(-1, F.HEAD), F.CODEBOOK_K, False, F.NGROUP_K).reshape(k.shape) for k in k_all]
    v_rec = [F.reconstruct(v.float().reshape(-1, F.HEAD), F.CODEBOOK_V, False, F.NGROUP_V).reshape(v.shape) for v in v_all]
    ref_a = _ref_attn(q_deq, k_rec, v_rec, seq_lens, q_lens)
    rel_a = _rel(got, ref_a)
    # B: the exact attention -- the format error, for the record
    ref_b = _ref_attn(q, k_all, v_all, seq_lens, q_lens)
    rel_b = _rel(got, ref_b)
    assert rel_a < tol, f"{label} {name}: kernel vs reconstructed-KV reference relL2 {rel_a:.4f}"
    print(f"ok   attn {label:28s} {name:7s} vs reconstructed {rel_a:.4f}   vs exact {rel_b:.4f}")


KERNEL_CODEBOOKS = ("kernel",)   # uniform K / geo V; the other formats live in the reference only


def test_attn_prefill_2d():
    for name in KERNEL_CODEBOOKS:
        _check_attn("prefill 512", [512], [512], 880, name)
        _check_attn("prefill 2x300 bs16", [300, 300], [300, 300], 16, name)


def test_attn_decode_3d():
    for name in KERNEL_CODEBOOKS:
        _check_attn("decode q=1 4k", [4096], [1], 880, name)
        _check_attn("decode q=1 16k", [16384], [1], 880, name)
        _check_attn("decode x4 bs16", [1000, 2000, 3000, 4000], [1] * 4, 16, name)


def test_attn_verify8():
    for name in KERNEL_CODEBOOKS:
        _check_attn("verify8 x4 4k", [4096] * 4, [8] * 4, 880, name)
        _check_attn("verify8 x4 8k", [8192] * 4, [8] * 4, 880, name)
        _check_attn("verify8 x8 ragged", [3000, 4500, 1200, 8000, 880, 1760, 6000, 2200], [8] * 8, 880, name)


def test_attn_mixed_prefill_decode():
    _check_attn("mixed", [2048, 5000, 7000], [2048, 1, 8], 880, "kernel")


if __name__ == "__main__":
    which = os.environ.get("KV4_TESTS", "all")
    fails = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        if which != "all" and not name.startswith(f"test_{which}"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
