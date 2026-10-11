#!/usr/bin/env python3
"""KV4H (the HIP-kernel layout): format reference on the CPU, store kernel and attention kernel
on the GPU. Run inside the serving image:  python3 sly/tests/kv4/test_kv4h.py   (exit 0 = pass)
KV4H_TESTS=format|store|attn selects a subset.
"""
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kv4"))

import radiance_kv4_format as F  # noqa: E402
import radiance_kv4h_format as H  # noqa: E402

torch.manual_seed(0)
NKV = 4


def _kv(n, device="cpu", head=H.HEAD):
    k = torch.randn(n, NKV, head, device=device)
    v = torch.randn(n, NKV, head, device=device)
    k[..., :6] *= 20.0
    k[..., 100] *= 12.0
    v *= torch.empty(n, NKV, 1, device=device).uniform_(0.2, 3.0)
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


# ---------------------------------------------------------------------------------------------
# format (CPU)
# ---------------------------------------------------------------------------------------------

def test_format_offsets_disjoint():
    B = 64
    seen = torch.zeros(B * 2 * H.TPT, dtype=torch.int32)
    t = torch.arange(B)
    for g in range(H.NG):
        seen[H.k_scale_off(t, g)] += 1
    for p in range(H.PACKED):
        seen[H.k_nib_off(t, p)] += 1
        seen[H.v_nib_off(t, p, B)] += 1
    seen[H.v_scale_off(t, B)] += 1
    assert int(seen.max()) == 1, "two fields share a byte"
    used = int((seen > 0).sum())
    assert used == B * (H.NG + H.PACKED + 1 + H.PACKED), used                   # 2 + 128 + 1 + 128 per token
    assert int(seen[:H.k_region_bytes(B)].sum()) == B * (H.NG + 128)              # K region holds only K


def test_format_roundtrip_and_bounds():
    B = 48
    k, v = _kv(B)
    k, v = k[:, 0].float(), v[:, 0].float()
    # the defaults (QMIN -8, clip 0.8): bytes round-trip, codes in [QMIN, 7]
    region = H.encode_block(k, v, B)
    assert region.shape == (B * 2 * H.TPT,) and region.dtype == torch.uint8
    kr, vr = H.decode_block(region, B)
    kc, ks = H.quantize_k(k)
    vc, vs = H.quantize_v(v)
    assert torch.equal(kr, H._dequant(kc, ks)) and torch.equal(vr, H._dequant(vc, vs))
    assert int(kc.min()) >= H.QMIN and int(kc.max()) <= 7 and int(vc.min()) >= H.QMIN and int(vc.max()) <= 7
    # no clipping (clip 1, symmetric codes): every element within half a step after the rotation
    region = H.encode_block(k, v, B, qmin=-7, clip=1.0)
    kr, _ = H.decode_block(region, B)
    kc, ks = H.quantize_k(k, qmin=-7, clip=1.0)
    assert int(kc.min()) >= -7 and int(kc.max()) <= 7
    k_rot = F.rotate(k)
    step = ks[:, :, None].expand(-1, -1, H.GROUP).reshape(-1, H.HEAD)
    assert torch.all((kr - k_rot).abs() <= step * 0.5 + 1e-4)
    rel = float((F.unrotate(kr) - k).norm() / k.norm())
    print(f"ok   format roundtrip: K relative error {rel:.4f} (group-{H.GROUP} scales, no clip)")
    assert rel < 0.15                      # synthetic outlier rows (x20 on 6 dims); real K sits at ~0.11 rel MSE^0.5


def test_format_head128_roundtrip():
    B = 32
    k, v = _kv(B, head=128)
    k, v = k[:, 0].float(), v[:, 0].float()
    L = H.layout(128)
    region = H.encode_block(k, v, B)
    assert region.shape == (B * 2 * L.TPT,) and 2 * L.TPT == 136
    kr, vr = H.decode_block(region, B, head=128)
    kc, ks = H.quantize_k(k)
    vc, vs = H.quantize_v(v)
    assert ks.shape[-1] == 128 // L.GROUP and torch.equal(kr, H._dequant(kc, ks)) and torch.equal(vr, H._dequant(vc, vs))
    seen = torch.zeros(B * 2 * L.TPT, dtype=torch.int32)
    t = torch.arange(B)
    seen[L.k_scale_off(t, 0)] += 1
    for p in range(L.PACKED):
        seen[L.k_nib_off(t, p)] += 1
        seen[L.v_nib_off(t, p, B)] += 1
    seen[L.v_scale_off(t, B)] += 1
    assert int(seen.max()) == 1 and int((seen > 0).sum()) == B * (1 + 64 + 1 + 64)


def test_format_partial_block_leaves_rest_zero():
    B = 32
    k, v = _kv(10)
    region = H.encode_block(k[:, 1].float(), v[:, 1].float(), B)
    kr, vr = H.decode_block(region, B)
    assert torch.all(kr[10:] == 0) and torch.all(vr[10:] == 0)
    # scale bytes of unused tokens are zero, so their decoded scale is 0, not garbage
    t = torch.arange(10, B)
    assert int(region[H.v_scale_off(t, B)].sum()) == 0


# ---------------------------------------------------------------------------------------------
# store kernel (GPU)
# ---------------------------------------------------------------------------------------------

def _check_store(n, bs, num_blocks, head=H.HEAD, steps=1, qmin=-7, clip=1.0):
    from radiance_kv4h_store import kv4h_store
    dev = "cuda"
    L = H.layout(head)
    k, v = _kv(n, dev, head)
    cache = torch.zeros(num_blocks, NKV, bs, 2 * L.TPT, dtype=torch.uint8, device=dev)
    slots = torch.randperm(num_blocks * bs, device=dev)[:n].to(torch.int64)
    slots[::7] = -1
    kv4h_store(k, v, cache, slots, scale_steps=steps, qmin=qmin, clip=clip)
    live = slots >= 0
    # reference: per (block, head), encode the tokens that landed there in slot order
    ref = torch.zeros_like(cache)
    for b in range(num_blocks):
        for h in range(NKV):
            sel = live & (slots // bs == b)
            if not sel.any():
                continue
            order = torch.argsort(slots[sel] % bs)
            idx = sel.nonzero().squeeze(1)[order]
            tpos = (slots[idx] % bs)
            # encode_block wants a dense prefix; scatter token by token instead
            for i, tp in zip(idx.tolist(), tpos.tolist()):
                one = H.encode_block(k[i, h][None].float(), v[i, h][None].float(), bs, steps=steps, qmin=qmin, clip=clip)
                # move token 0's bytes to position tp
                src_t = torch.tensor([0])
                dst_t = torch.tensor([tp])
                reg = ref[b, h].reshape(-1)
                for g in range(L.NG):
                    reg[L.k_scale_off(dst_t, g)] = one[L.k_scale_off(src_t, g)]
                p = torch.arange(L.PACKED)
                reg[L.k_nib_off(dst_t[:, None], p[None, :])] = one[L.k_nib_off(src_t[:, None], p[None, :])].to(dev)
                reg[L.v_scale_off(dst_t, bs)] = one[L.v_scale_off(src_t, bs)]
                reg[L.v_nib_off(dst_t[:, None], p[None, :], bs)] = one[L.v_nib_off(src_t[:, None], p[None, :], bs)].to(dev)
    diff = cache != ref
    frac = float(diff.float().mean())
    if steps > 1 or clip != 1.0:
        # the scale search picks the smallest error: a near tie between two scales may go the
        # other way with the kernel's rotation sums, then the whole row differs. Check that the
        # kernel's choice is as good as the oracle's and that such rows stay rare.
        errs = []
        for b in range(num_blocks):
            for h in range(NKV):
                sel = live & (slots // bs == b)
                if not sel.any():
                    continue
                idx = sel.nonzero().squeeze(1)
                tp = slots[idx] % bs
                kr = F.rotate(k[idx, h].float()); vr = F.rotate(v[idx, h].float())
                for got in (cache, ref):
                    kd, vd = H.decode_block(got[b, h].reshape(-1), bs, head=head)
                    errs.append((((kd[tp] - kr) ** 2).sum() + ((vd[tp] - vr) ** 2).sum()).item())
        e_kernel, e_ref = sum(errs[0::2]), sum(errs[1::2])
        assert e_kernel <= e_ref * 1.002, (e_kernel, e_ref)
        assert frac < 2e-3, f"{int(diff.sum())} bytes differ ({frac:.2e})"
        print(f"ok   store steps {steps} qmin {qmin} clip {clip}: error kernel/oracle {e_kernel / e_ref:.5f}, "
              f"{int(diff.sum())} bytes differ (bs {bs}, head {head})")
        return
    # the kernel rotates with a bf16 dot (fp32 accumulation) whose summation order differs from the
    # torch matmul of the reference: a code may round the other way at an exact .5, nothing else
    if diff.any():
        lo_d = ((cache & 0xF).int() - (ref & 0xF).int()).abs()
        hi_d = ((cache >> 4).int() - (ref >> 4).int()).abs()
        lo_d = torch.minimum(lo_d, 16 - lo_d)
        hi_d = torch.minimum(hi_d, 16 - hi_d)
        assert int(lo_d.max()) <= 1 and int(hi_d.max()) <= 1, "a code differs by more than 1"
        assert frac < 1e-4, f"{int(diff.sum())} bytes differ ({frac:.2e})"
    print(f"ok   store: {int(live.sum())} tokens x {NKV} heads, {int(diff.sum())} bytes off by one code (bs {bs}, head {head})")


def test_store_small():
    _check_store(120, 32, 6)


def test_store_880():
    _check_store(700, 880, 2)


def test_store_head128():
    _check_store(300, 880, 2, head=128)


def test_store_scale_search():
    for head in (256, 128):
        _check_store(300, 32, 12, head=head, qmin=-8)
        _check_store(300, 32, 12, head=head, steps=9)
        _check_store(300, 32, 12, head=head, steps=9, qmin=-8)
        _check_store(300, 32, 12, head=head, qmin=-8, clip=0.8)


def test_format_scale_search_error():
    """the scale search and the 16th level only lower the error (CPU, oracle)"""
    k, v = _kv(256)
    kr, vr = F.rotate(k[:, 0]), F.rotate(v[:, 0])
    e = {}
    for steps, qmin, clip in ((1, -7, 1.0), (9, -7, 1.0), (9, -8, 1.0), (1, -8, 0.8)):
        ek = ((F.rotate(H.reconstruct_k(k[:, 0], steps=steps, qmin=qmin, clip=clip)) - kr) ** 2).sum()
        ev = ((F.rotate(H.reconstruct_v(v[:, 0], steps=steps, qmin=qmin, clip=clip)) - vr) ** 2).sum()
        e[(steps, qmin, clip)] = float(ek + ev)
    assert e[(9, -7, 1.0)] < e[(1, -7, 1.0)] and e[(9, -8, 1.0)] <= e[(9, -7, 1.0)] and e[(1, -8, 0.8)] < e[(1, -7, 1.0)], e
    print("ok   scale search: rel. error " + ", ".join(f"{k}: {v / e[(1, -7, 1.0)]:.3f}" for k, v in e.items()))


# ---------------------------------------------------------------------------------------------
# attention kernel (GPU, HIP int4 WMMA)
# ---------------------------------------------------------------------------------------------

NQ = 24
SCALE = 1.0 / (H.HEAD ** 0.5)


def _ref_attn(q, k_all, v_all, seq_lens, q_lens, causal=True, window=0, nq=NQ):
    dev = q.device
    head = q.shape[-1]
    outs, qi = [], 0
    g = nq // NKV
    for s_, (sl, ql) in enumerate(zip(seq_lens, q_lens)):
        qs = q[qi:qi + ql].float().view(ql, NKV, g, head)
        ks, vs = k_all[s_].float(), v_all[s_].float()
        sc = torch.einsum("qhgd,khd->hgqk", qs, ks) / math.sqrt(head)
        ctx = sl - ql
        qpos = torch.arange(ql, device=dev)[:, None] + ctx
        kpos = torch.arange(sl, device=dev)[None, :]
        if causal:
            sc.masked_fill_(kpos > qpos, float("-inf"))
        if window > 0:        # vLLM: qabs - t < window, and t - qabs < window when non-causal
            bad = (qpos - kpos) >= window
            if not causal:
                bad |= (kpos - qpos) >= window
            sc.masked_fill_(bad, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        outs.append(torch.einsum("hgqk,khd->qhgd", p, vs).reshape(ql, nq, head))
        del sc, p
        qi += ql
    return torch.cat(outs)


def _check_attn(label, seq_lens, q_lens, bs, nseg=None, tol=0.05, causal=True, window=0, head=H.HEAD, nq=NQ, tile=None,
                row_tiles=None, q_planes=None, k_group=None, stale=False, sink=0.0, p_ref=None, quiet=False):
    """stale: the cache starts as random bytes, so the slots past seq_len hold garbage (vLLM
    reuses blocks without clearing them). sink > 0: every query shares a direction per kv head,
    key 0 gets a logit of ~sink along it and the other keys are small -- a peaked head whose
    diffuse tail sits far below the running max (the P-resolution case). Returns rel_a."""
    from radiance_kv4h_kernel import kv4h_attention, quantize_query_torch, unrotate_output, LAST_LAUNCH
    from radiance_kv4h_store import kv4h_store
    dev = "cuda"
    L = H.layout(head, k_group)
    scale = 1.0 / math.sqrt(head)
    nseq = len(seq_lens)
    blocks_per = [-(-s_ // bs) for s_ in seq_lens]
    num_blocks = sum(blocks_per) + 1
    block_table = torch.zeros(nseq, max(blocks_per), dtype=torch.int32, device=dev)
    cache = torch.zeros(num_blocks, NKV, bs, 2 * L.TPT, dtype=torch.uint8, device=dev)
    if stale:
        cache.random_(0, 256)
    k_all, v_all, nb = [], [], 1
    for s_ in range(nseq):
        ids = torch.arange(nb, nb + blocks_per[s_], device=dev)
        block_table[s_, :blocks_per[s_]] = ids.to(torch.int32)
        nb += blocks_per[s_]
        k, v = _kv(seq_lens[s_], dev, head)
        if sink:
            if s_ == 0:
                g0 = torch.Generator(device="cpu").manual_seed(7)
                qbase = torch.randn(NKV, head, generator=g0).to(dev)
            k = 0.3 * torch.randn(seq_lens[s_], NKV, head, device=dev)
            k[0] = sink * math.sqrt(head) * qbase / (qbase * qbase).sum(-1, keepdim=True)
            # real V has a large shared component (the K/V dumps: removing the mean is +4 dB on
            # V); with zero-mean V a dropped tail would average out and hide the effect
            if s_ == 0:
                vbase = 2.0 * torch.randn(NKV, head, generator=g0).to(dev)
            v = vbase[None] + 0.5 * v
        k_all.append(k)
        v_all.append(v)
        pos = torch.arange(seq_lens[s_], device=dev)
        kv4h_store(k, v, cache, (ids[pos // bs] * bs + pos % bs).to(torch.int64), k_group=k_group)
    total_q = sum(q_lens)
    q = torch.randn(total_q, nq, head, device=dev)
    if sink:
        q = qbase.repeat_interleave(nq // NKV, 0)[None] + 0.3 * q
    q = q.to(torch.bfloat16)
    cu_q = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
    seqused = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    out = torch.empty(total_q, nq, head, dtype=torch.bfloat16, device=dev)
    # the fused path (un-rotation in the merge, strided destination) and the rotated result
    dst = torch.zeros(total_q, nq, head + 32, dtype=torch.bfloat16, device=dev)[:, :, :head]
    knobs = dict(tile=tile, row_tiles=row_tiles, q_planes=q_planes, k_group=k_group, p_ref=p_ref)
    kv4h_attention(q, cache, out, cu_q, max(q_lens), seqused, max(seq_lens), scale, block_table, nseg,
                   causal=causal, window=window, output=dst, **knobs)
    out2 = torch.empty_like(out)
    kv4h_attention(q, cache, out2, cu_q, max(q_lens), seqused, max(seq_lens), scale, block_table, nseg,
                   causal=causal, window=window, **knobs)
    torch.cuda.synchronize()
    got = dst.float()
    got2 = unrotate_output(out2).float() if (LAST_LAUNCH["cfg"]["nseg"] > 1) else unrotate_output(out2).float()
    assert torch.isfinite(got).all(), "non-finite output"
    rel_ab = float((got - got2).norm() / got2.norm())
    assert rel_ab < 2e-2, f"{label}: fused merge vs rotated merge path differ: relL2 {rel_ab:.4f}"
    cfg = LAST_LAUNCH.get("cfg", {})                  # the knobs the launch resolved (args, env, table)
    q8, desc = quantize_query_torch(q, 7.0 if cfg.get("q_planes") == 1 else 127.0)   # same rounding (<= 1 LSB)
    q_deq = unrotate_output(q8.float() * desc[..., None])
    k_rec = [H.reconstruct_k(k.float().reshape(-1, head), cfg.get("k_group", H.GROUP)).reshape(k.shape) for k in k_all]
    v_rec = [H.reconstruct_v(v.float().reshape(-1, head)).reshape(v.shape) for v in v_all]
    ref_a = _ref_attn(q_deq, k_rec, v_rec, seq_lens, q_lens, causal, window, nq)
    rel_a = float((got - ref_a).norm() / ref_a.norm())
    ref_b = _ref_attn(q, k_all, v_all, seq_lens, q_lens, causal, window, nq)
    rel_b = float((got - ref_b).norm() / ref_b.norm())
    if quiet:
        return rel_a
    assert rel_a < tol, f"{label}: kernel vs reconstructed-KV reference relL2 {rel_a:.4f} ({cfg})"
    print(f"ok   attn {label:30s} tile={cfg.get('tile')} r{cfg.get('row_tiles')}q{cfg.get('q_planes')}g{cfg.get('k_group')} "
          f"nseg={cfg.get('nseg')} vs reconstructed {rel_a:.4f}   vs exact {rel_b:.4f}   fused-vs-rotated {rel_ab:.4f}")
    return rel_a


def test_attn_sink_p_ref():
    """a dominant key 0 and a long diffuse tail: with P quantized against the running max the
    tail's tokens fall under 0.5 / 255 and drop out of P.V (the sum keeps them); P against the
    tile's own max (KV4H_PREF=1) keeps the tail"""
    # seeded: the error of this construction varies with the draw for both references (40 seeds:
    # 1.0-5.5 %), the tile-max reference was never worse (median 0.66x of the running max's)
    for sl, sink, seed in ((8192, 9.0, 0), (8192, 9.0, 1), (32768, 11.0, 0)):
        r = {}
        for pref in (0, 1):
            torch.manual_seed(seed)
            r[pref] = _check_attn(f"sink {sink} {sl // 1024}k pref {pref}", [sl], [8], 1664, sink=sink, p_ref=pref, quiet=True)
        print(f"ok   attn sink {sink} at {sl // 1024}k seed {seed}: vs reconstructed, P vs running max {r[0]:.4f}, "
              f"P vs tile max {r[1]:.4f}")
        assert r[1] <= r[0] * 1.02 and r[1] < 0.08, r
    _check_attn("pref 1 verify8 x4 4k", [4096] * 4, [8] * 4, 880, p_ref=1)
    _check_attn("pref 1 decode 16k", [16384], [1], 1664, p_ref=1)
    _check_attn("pref 1 prefill 2x300", [300, 300], [300, 300], 32, p_ref=1)
    _check_attn("pref 1 drafter h128 w2048 nc", [20000], [8], 1664, causal=False, window=2048, head=128, nq=16, p_ref=1)


def test_attn_review_cases():
    """shapes from the 2026-10-11 review: a causal multi-block launch whose kernel split fills
    segments past what AITER's seq_len-only reduce kept (the rotated path used to merge with it),
    and garbage bytes past seq_len (stale tokens set the tile's V lift)"""
    _check_attn("review: causal q-blocks nseg=4", [131], [8], 32, nseg=4, tile=32, row_tiles=2)
    _check_attn("review: causal q-blocks nseg=16", [1000], [8], 32, nseg=16, tile=32, row_tiles=2)
    _check_attn("review: window + q-blocks nseg=8", [3000], [16], 32, nseg=8, window=512, row_tiles=2)
    _check_attn("review: stale bytes decode", [1000, 77], [1, 1], 32, stale=True)
    _check_attn("review: stale bytes verify8", [1000, 300], [8, 8], 880, stale=True)


def test_attn_decode():
    _check_attn("decode q=1 4k", [4096], [1], 880)
    _check_attn("decode q=1 4k nseg=1", [4096], [1], 880, nseg=1)
    _check_attn("decode q=1 16k", [16384], [1], 880)
    _check_attn("decode x4 bs32", [1000, 2000, 3000, 4000], [1] * 4, 32)


def test_attn_verify8():
    _check_attn("verify8 x4 4k", [4096] * 4, [8] * 4, 880)
    _check_attn("verify8 x4 8k nseg=16", [8192] * 4, [8] * 4, 880, nseg=16)
    _check_attn("verify8 x8 ragged", [3000, 4500, 1200, 8000, 880, 1760, 6000, 2200], [8] * 8, 880)


def test_attn_noncausal():
    """The DFlash drafter's block attention: 8 queries at the end of the context, every query sees
    every token (including the block's own 8)."""
    _check_attn("noncausal verify8 x4 4k", [4096] * 4, [8] * 4, 880, causal=False)
    _check_attn("noncausal x2 ragged nseg=8", [3000, 1000], [8, 8], 880, nseg=8, causal=False)
    _check_attn("noncausal prefill 300 bs32", [300], [300], 32, nseg=1, causal=False)


def test_attn_window():
    """Sliding window, causal (window 512 at 4k context) and the drafter's shape: head 128,
    32 / 8 heads... here 16 / 4 (NKV = 4), window 2048, non-causal, 8 queries, contexts 1k-20k."""
    _check_attn("window 512 causal x2", [4096, 2000], [8, 1], 880, window=512)
    _check_attn("window 512 causal nseg=1", [4096], [8], 880, nseg=1, window=512)
    _check_attn("drafter h128 w2048 nc 20k", [20008, 1000, 2056], [8] * 3, 880, causal=False, window=2048, head=128, nq=16)
    _check_attn("drafter h128 w2048 nc nseg=4", [8200], [8], 880, nseg=4, causal=False, window=2048, head=128, nq=16)
    _check_attn("h128 causal 4k", [4096, 3000], [8, 1], 880, head=128, nq=16)
    _check_attn("h128 prefill 300 w256 bs32", [300], [300], 32, nseg=1, window=256, head=128, nq=16)


def test_attn_tiles():
    """Every tile size on decode, verify, window, head-128 and prefill shapes."""
    for tile in (16, 64):
        _check_attn(f"tile {tile} decode x4 bs32", [1000, 2000, 3000, 4000], [1] * 4, 32, tile=tile)
        _check_attn(f"tile {tile} verify8 x8 ragged", [3000, 4500, 1200, 8000, 880, 1760, 6000, 2200], [8] * 8, 880, tile=tile)
        _check_attn(f"tile {tile} window 512 x2", [4096, 2000], [8, 1], 880, window=512, tile=tile)
        _check_attn(f"tile {tile} drafter h128 nc", [8200, 1000], [8, 8], 880, causal=False, window=2048, head=128, nq=16, tile=tile)
        _check_attn(f"tile {tile} prefill 2x300 bs32", [300, 300], [300, 300], 32, nseg=1, tile=tile)


def test_attn_row_overflow():
    """Rows past block_q * nqpkv (block_q = BLOCK_M // nqpkv, nqpkv = 6: rows 60-63, or 30-31 at
    2 row tiles) belong to the NEXT query block. Two programs computing the same (token, head) with
    different causal limits split the segments differently and overwrite each other's partials:
    those rows must be dead. Context chosen so the two limits fall into different tiles."""
    _check_attn("row overflow r2 verify8 nseg=64", [4096], [8], 880, nseg=64, row_tiles=2)
    _check_attn("row overflow r2 verify8 x3", [4096, 2056, 1032], [8, 8, 8], 880, nseg=32, row_tiles=2)
    _check_attn("row overflow r4 verify16", [4096], [16], 880, nseg=64, row_tiles=4)


def test_attn_knobs():
    """The build knobs: 2 row tiles, an int4 query, K scale groups of 256 and 64 dims (head 128: 64)."""
    _check_attn("rowtiles 2 verify8 x4 4k", [4096] * 4, [8] * 4, 880, row_tiles=2)
    _check_attn("rowtiles 2 prefill 300 bs32", [300], [300], 32, nseg=1, row_tiles=2)
    _check_attn("rowtiles 2 drafter h128", [8200, 1000], [8, 8], 880, causal=False, window=2048, head=128, nq=16, row_tiles=2)
    _check_attn("qplanes 1 verify8 x4 4k", [4096] * 4, [8] * 4, 880, q_planes=1)
    _check_attn("qplanes 1 decode 16k", [16384], [1], 880, q_planes=1)
    _check_attn("kgroup 256 verify8 x4 4k", [4096] * 4, [8] * 4, 880, k_group=256)
    _check_attn("kgroup 64 verify8 x2 4k", [4096, 3000], [8, 8], 880, k_group=64)
    _check_attn("kgroup 64 h128 window nc", [8200], [8], 880, causal=False, window=2048, head=128, nq=16, k_group=64)
    _check_attn("all knobs tile64 r2 q1 g256", [4096, 2000], [8, 1], 880, tile=64, row_tiles=2, q_planes=1, k_group=256)


def test_attn_query_kernels():
    """Triton quantize_query / unrotate_output against the torch versions."""
    from radiance_kv4h_kernel import quantize_query, unrotate_output, quantize_query_torch, unrotate_output_torch
    for n, head in ((1, 256), (8, 256), (77, 256), (8, 128)):
        q = torch.randn(n, NQ, head, device="cuda").to(torch.bfloat16)
        q8, d = quantize_query(q)
        q8r, dr = quantize_query_torch(q)
        assert torch.allclose(d, dr, rtol=1e-5, atol=0), "descale"
        diff = (q8.int() - q8r.int()).abs()
        assert int(diff.max()) <= 1 and float((diff > 0).float().mean()) < 1e-3, f"q8: max {int(diff.max())}, frac {float((diff > 0).float().mean())}"
        o = torch.randn(n, NQ, head, device="cuda").to(torch.bfloat16)
        # strided destination (a wider buffer, as vLLM's output is a view)
        dst = torch.zeros(n, NQ, head + 64, device="cuda", dtype=torch.bfloat16)[:, :, :head]
        unrotate_output(o, dst)
        ref = unrotate_output_torch(o)
        rel = float((dst.float() - ref).norm() / ref.norm())
        assert rel < 5e-3, rel
        assert torch.allclose(unrotate_output(o).float(), ref, rtol=1e-2, atol=1e-3)
    print("ok   query kernels: int8 rotation (<=1 LSB on <0.1 %) and un-rotation match torch")


def test_attn_prefill():
    _check_attn("prefill 512 nseg=1", [512], [512], 880, nseg=1)
    _check_attn("prefill 2x300 bs32", [300, 300], [300, 300], 32, nseg=1)
    _check_attn("mixed", [2048, 5000, 7000], [2048, 1, 8], 880)


if __name__ == "__main__":
    which = os.environ.get("KV4H_TESTS", "all")
    fails = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        if which != "all" and not name.startswith(f"test_{which}"):
            continue
        try:
            fn()
            print("ok  ", name)
        except Exception as e:  # noqa: BLE001
            fails += 1
            print("FAIL", name, repr(e))
    sys.exit(1 if fails else 0)
