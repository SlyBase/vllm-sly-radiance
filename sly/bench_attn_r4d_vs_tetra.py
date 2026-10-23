#!/usr/bin/env python3
"""Prefill attention per chunk: libr4d (fp8 KV, the R4D_HYBRID path) vs TetraAttention (int4 KV).

Question: is the TetraAttention prefill kernel >~1.3x slower than libr4d's at depth? (KV4 vs FP8 whole
prefill: -1.4 % at 1.5k ... -11.4 % at 47k; if attention is the cause it must be ~25-40 % slower deep.)

One causal chunk of CHUNK new tokens attending to PAST + CHUNK keys, production geometry (24 q / 4 kv
heads, head 256, GQA 6), one sequence, cuda events, median of --reps after warmup (bench_kv4_attn.timeit).

  A  r4d   radiance_r4d_hybrid_attn as installed in the image ($SP/radiance_r4d_hybrid_attn.py): its own
           _geometry, _block_table16 (880-token AITER block read as 55 x 16), _q_kernel (e4m3 query +
           set_q_fp8(q_scale) under RADIANCE_ATTN_Q_FP8=1) and _PREFILL[0] (fp8 KV), launched with the
           argument list of R4DHybridAttentionImpl.forward. Image ENV defaults are set if unset:
           R4D_ATTN_FP8=3 R4D_ATTN_DB=1 R4D_ATTN_TILE=16 RADIANCE_ATTN_Q_FP8=1.
  B  tetra radiance_kv4h_kernel.kv4h_attention as TetraAttentionImpl.forward calls it (rotate + quantize q,
           HIP kernel, un-rotate into the output), block 1664, cache written by kv4h_store; and the HIP
           launch alone (same q8 / descale / knobs, re-launched).

  # inside a container of the runtime image, repo mounted at /w (sly/ and sly/kv4/ found via sys.path)
  python3 /w/sly/bench_attn_r4d_vs_tetra.py
  python3 /w/sly/bench_attn_r4d_vs_tetra.py --tile 16
  python3 /w/sly/bench_attn_r4d_vs_tetra.py --check --pasts 0,4096,16384
"""
import argparse
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kv4"))

NQ, NKV, HEAD = 24, 4, 256
BS_FP8, BS_INT4 = 880, 1664          # production attention blocks: fp8 KV (fp16 ssm state), int4 cells
IMAGE_ENV = {"R4D_ATTN_FP8": "3", "R4D_ATTN_DB": "1", "R4D_ATTN_TILE": "16", "RADIANCE_ATTN_Q_FP8": "1"}
THRESHOLD = 1.3
dev = "cuda"


# -- pure helpers (CPU-tested in sly/tests/test_bench_attn_r4d_vs_tetra_cpu.py) ---------------------
def shape(past: int, chunk: int, bs: int) -> dict:
    """seq (keys seen by the chunk), blocks per sequence for an attention block of bs tokens."""
    seq = past + chunk
    return dict(seq=seq, blocks=-(-seq // bs))


def verdict(ratios: dict, threshold: float = THRESHOLD) -> str:
    """ratios: past -> tetra/r4d time. Judged at the deepest past (where the whole-prefill gap is)."""
    deep = max(ratios)
    r = ratios[deep]
    if r > threshold:
        return f"VERDICT: tetra {r:.2f}x r4d at past {deep} > {threshold}x -> attention explains the gap, port worth it"
    return f"VERDICT: tetra {r:.2f}x r4d at past {deep} <= {threshold}x -> attention alone does not explain the gap"


def ints(s: str) -> list:
    return [int(x) for x in s.split(",") if x != ""]


# -- GPU part ---------------------------------------------------------------------------------------
def make_kv(seq, seed=0):
    import torch
    g = torch.Generator(device=dev).manual_seed(seed)
    k = torch.randn(seq, NKV, HEAD, device=dev, generator=g).to(torch.bfloat16)
    v = torch.randn(seq, NKV, HEAD, device=dev, generator=g).to(torch.bfloat16)
    return k, v


def r4d_arm(q, k, v, seq, chunk, scale, q_scale):
    """-> (fn, out). Mirrors R4DHybridAttentionImpl.forward (radiance_r4d_hybrid_attn.py:416-490)."""
    import torch
    import radiance_r4d_hybrid_attn as HY
    nb = -(-seq // BS_FP8)
    # AITER layout (block, kv head, slot, 2*head), K then V per slot; block 0 unused like the engine's null block
    kv8 = torch.zeros(nb + 1, NKV, BS_FP8, 2 * HEAD, dtype=torch.float8_e4m3fn, device=dev)
    tok = torch.zeros(nb * BS_FP8, NKV, 2 * HEAD, dtype=torch.bfloat16, device=dev)
    tok[:seq, :, :HEAD], tok[:seq, :, HEAD:] = k, v
    kv8[1:] = tok.view(nb, BS_FP8, NKV, 2 * HEAD).transpose(1, 2).to(torch.float8_e4m3fn)
    del tok
    impl = SimpleNamespace(_geom=None, head_size=HEAD)
    geom = HY.R4DHybridAttentionImpl._geometry(impl, kv8)
    assert not isinstance(geom, str), geom
    variant, mult, head_stride, ratio = geom
    assert variant == 0, "fp8 cache must select the fp8-KV prefill entry"
    bt = (torch.arange(nb, device=dev) + 1).view(1, nb).to(torch.int32)
    m = SimpleNamespace(hyb_cache=None, block_table=bt)
    runs = (("r4d", 0, 1, chunk, 0),)
    table, lo = HY.R4DHybridAttentionImpl._block_table16(m, runs, mult, ratio)
    launch = HY._PREFILL[variant]
    seq_lens = torch.tensor([seq], dtype=torch.int32, device=dev)
    query = (q.float() / q_scale).to(torch.float8_e4m3fn)     # the attention layer's static-scale fp8 query
    out = torch.empty(chunk, NQ, HEAD, dtype=torch.bfloat16, device=dev)
    scratch = torch.empty(256, dtype=torch.uint8, device=dev)
    max_blocks, block_stride = table.shape[1], HY.BLOCK_SIZE * kv8.shape[3]
    print(f"  r4d: {launch.__name__} from {HY.__file__}, block {BS_FP8} = {ratio} x {HY.BLOCK_SIZE}, mult {mult}")

    def fn():
        qk = HY._q_kernel(query, q_scale)
        launch(qk.data_ptr(), kv8.data_ptr(), table.data_ptr(), seq_lens.data_ptr(), out.data_ptr(), 0, 0,
               scratch.data_ptr(), 1, chunk, NQ, NKV, HEAD, HY.BLOCK_SIZE, max_blocks, block_stride, head_stride,
               scale, 0, seq, torch.cuda.current_stream().cuda_stream)
    return fn, out


def tetra_arm(q, k, v, seq, chunk, scale, knobs):
    """-> (backend fn, kernel-only fn, out). Mirrors TetraAttentionImpl.forward (sly/kv4/radiance_kv4h_attn.py:165-175)."""
    import torch
    import radiance_kv4h_format as H
    import radiance_kv4h_kernel as KH
    from radiance_kv4h_store import kv4h_store
    nb = -(-seq // BS_INT4)
    L = H.layout(HEAD, min(knobs.get("k_group") or KH.K_GROUP, HEAD))
    cache = torch.zeros(nb + 1, NKV, BS_INT4, 2 * L.TPT, dtype=torch.uint8, device=dev)
    kv4h_store(k, v, cache, torch.arange(BS_INT4, BS_INT4 + seq, device=dev, dtype=torch.int64),
               k_group=knobs.get("k_group"))
    bt = (torch.arange(nb, device=dev) + 1).view(1, nb).to(torch.int32)
    cu = torch.tensor([0, chunk], dtype=torch.int32, device=dev)
    seqused = torch.tensor([seq], dtype=torch.int32, device=dev)
    output = torch.empty(chunk, NQ, HEAD, dtype=torch.bfloat16, device=dev)

    def fn():
        out_rot = torch.empty(chunk, NQ, HEAD, dtype=torch.bfloat16, device=dev)
        KH.kv4h_attention(q, cache, out_rot, cu, chunk, seqused, seq, scale, bt, causal=True, window=0,
                          output=output, **knobs)

    fn()
    c = dict(KH.LAST_LAUNCH["cfg"])
    ext = KH._ext(c["row_tiles"], c["q_planes"], HEAD // c["k_group"], c["p_ref"])
    q8, qd = KH.quantize_query(q, 127.0 if c["q_planes"] == 2 else 7.0)
    nseg = c["nseg"]
    out_k = torch.empty(chunk, NQ, HEAD, dtype=torch.bfloat16, device=dev)
    seg = [torch.empty(chunk, NQ, nseg, *s, dtype=torch.float32, device=dev) for s in ((HEAD,), (), ())] if nseg > 1 else None
    so, sm, ss = (t.data_ptr() for t in seg) if seg else (0, 0, 0)
    o = 0 if seg else out_k.data_ptr()
    scale_log2 = (scale / HEAD) * 1.4426950408889634

    def kernel():   # the ext.launch of kv4h_attention (radiance_kv4h_kernel.py:439-443), nothing else
        ext.launch(HEAD, c["tile"], q8.data_ptr(), qd.data_ptr(), cache.data_ptr(), bt.data_ptr(), seqused.data_ptr(),
                   cu.data_ptr(), so, sm, ss, o, float(scale_log2), c["q_blocks"], NKV, NQ, NQ // NKV, c["block_q"],
                   cache.stride(0), cache.stride(1), BS_INT4, bt.stride(0), nseg, 1, 1, 0,
                   torch.cuda.current_stream().cuda_stream)
    return fn, kernel, output, c


def reference(q, k, v, past, scale):
    """fp32 SDPA of the chunk (query rows at positions past..past+chunk-1), one q head at a time."""
    import torch
    import torch.nn.functional as Fn
    chunk, seq = q.shape[0], k.shape[0]
    mask = torch.arange(seq, device=dev)[None, :] <= (past + torch.arange(chunk, device=dev))[:, None]
    out = torch.empty(chunk, NQ, HEAD, dtype=torch.float32, device=dev)
    for h in range(NQ):
        kh, vh = k[:, h // (NQ // NKV)].float(), v[:, h // (NQ // NKV)].float()
        out[:, h] = Fn.scaled_dot_product_attention(q[:, h].float()[None], kh[None], vh[None], attn_mask=mask[None],
                                                    scale=scale)[0]
    return out


def err(out, ref):
    d = out.float() - ref
    return f"max abs {d.abs().max().item():.3e}  rel L2 {(d.norm() / ref.norm()).item():.3e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=4096, help="new tokens per prefill chunk (serve CHUNK)")
    ap.add_argument("--pasts", default="0,4096,16384,32768,49152", help="context already in the cache")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--q-scale", type=float, default=1.0, help="the layer's static q scale (_q_scale_float)")
    ap.add_argument("--tile", type=int, default=0, help="TetraAttention tokens per tile (0 = env / tune table)")
    ap.add_argument("--no-tune", action="store_true", help="skip installing sly/radiance_attn_decode (production has it)")
    ap.add_argument("--check", action="store_true", help="also compare both outputs to an fp32 SDPA reference")
    a = ap.parse_args()
    for kk, vv in IMAGE_ENV.items():          # before libr4d's first launch: R4D_ATTN_FP8 is read once
        os.environ.setdefault(kk, vv)
    import torch
    from bench_kv4_attn import timeit
    if not a.no_tune:
        import radiance_attn_decode as T
        print("decode tune installed:", T.install())
    print("device:", torch.cuda.get_device_name(), "env:", {kk: os.environ[kk] for kk in IMAGE_ENV},
          "tile:", a.tile or "default")
    scale = 1.0 / math.sqrt(HEAD)
    ratios = {}
    for past in ints(a.pasts):
        seq = shape(past, a.chunk, BS_FP8)["seq"]
        k, v = make_kv(seq, seed=past)
        q = torch.randn(a.chunk, NQ, HEAD, device=dev, generator=torch.Generator(device=dev).manual_seed(1)).to(torch.bfloat16)
        fa, outa = r4d_arm(q, k, v, seq, a.chunk, scale, a.q_scale)
        ta = timeit(fa, a.reps)
        print(f"past {past:>6} (keys {seq:>6}): r4d {ta:9.1f} us")
        ref = reference(q, k, v, past, scale) if a.check else None
        if a.check:
            fa()
            torch.cuda.synchronize()
            print(f"    check r4d  : {err(outa, ref)}")
        knobs = {"tile": a.tile} if a.tile else {}
        fb, fk, outb, cfg = tetra_arm(q, k, v, seq, a.chunk, scale, knobs)
        tb, tk = timeit(fb, a.reps), timeit(fk, a.reps)
        ratios[past] = tb / ta
        print(f"    tetra tile={cfg['tile']} nseg={cfg['nseg']}: backend {tb:9.1f} us "
              f"({tb / ta:4.2f}x)  kernel {tk:9.1f} us ({tk / ta:4.2f}x)")
        if a.check:
            fb()
            torch.cuda.synchronize()
            print(f"    check tetra: {err(outb, ref)}")
        sys.stdout.flush()
        del k, v, ref
        torch.cuda.empty_cache()
    print("  ".join(f"{past}:{x:.2f}x" for past, x in sorted(ratios.items())) + "   " + verdict(ratios))


if __name__ == "__main__":
    main()
