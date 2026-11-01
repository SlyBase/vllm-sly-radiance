#!/usr/bin/env python3
"""KV4 attention vs AITER fp8 unified attention on the verify shapes: us per call, eager launches.

Production geometry: head 256, 24 q / 4 kv heads, block 880, fp8 q. Both paths run with the fork's
decode tune (sly/radiance_attn_decode.py) installed, so the fp8 side is what the server runs and
the KV4 side gets the same cells. The KV4 number is the whole backend path (rotate + quantize q,
kernel, un-rotate the output) and the kernel alone.

  # inside a container from the runtime image (PYTHONPATH must find sly/ and sly/kv4/)
  python3 sly/bench_kv4_attn.py --nseqs 1,4,8 --widths 1,8 --depths 4096,32768,98304
  python3 sly/bench_kv4_attn.py --nseqs 1 --prefill 2048,8192        # 2D path: width = depth
  python3 sly/bench_kv4_attn.py --tile 64 --rowtiles 2 --qplanes 1 --kgroup 256 --segments 16   # KV4H knob sweep
"""
import argparse
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "kv4"))

import radiance_kv4_format as F  # noqa: E402
from radiance_kv4_kernel import kv4_attention, quantize_query  # noqa: E402
from radiance_kv4_store import kv4_store  # noqa: E402
import radiance_kv4h_format as H  # noqa: E402
from radiance_kv4h_kernel import kv4h_attention  # noqa: E402
from radiance_kv4h_store import kv4h_store  # noqa: E402

dev = "cuda"
NQ, NKV, HEAD, BS = 24, 4, 256, 880
KNOBS: dict = {}          # kv4h launch knobs from the command line (tile, row_tiles, q_planes, k_group, num_segments)
SCALE = 1.0 / math.sqrt(HEAD)
FP8 = torch.float8_e4m3fn


def timeit(fn, reps):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


def bench(nseq, width, depth, reps, codebook):
    from aiter.ops.triton.attention.unified_attention import unified_attention as aiter_ua
    blocks_per = -(-depth // BS)
    num_blocks = nseq * blocks_per + 1
    block_table = (torch.arange(nseq * blocks_per, device=dev).view(nseq, blocks_per) + 1).to(torch.int32)
    seq_lens = [depth] * nseq
    seqused = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
    cu_q = torch.arange(0, nseq * width + 1, width, dtype=torch.int32, device=dev)
    q = torch.randn(nseq * width, NQ, HEAD, device=dev).to(torch.bfloat16)
    # fp8 cache in AITER's layout (blocks, nkv, bs, 2*head) with K/V split views
    kv8 = torch.randn(num_blocks, NKV, BS, 2 * HEAD, device=dev).to(FP8)
    k8, v8 = kv8.transpose(1, 2).split(HEAD, dim=-1)
    q8_static = (q.float() / 1.0).to(FP8)
    one = torch.ones(1, dtype=torch.float32, device=dev)
    out8 = torch.empty_like(q)
    # kv4 cache from the same shapes (contents random; timing only)
    kv4 = torch.zeros(num_blocks, NKV, BS, F.CONTENT, dtype=torch.uint8, device=dev)
    kk = torch.randn(min(depth, 8192), NKV, HEAD, device=dev).to(torch.bfloat16)
    kv4_store(kk, kk, kv4, torch.arange(BS, BS + kk.shape[0], device=dev, dtype=torch.int64))
    out4 = torch.empty_like(q)

    def f_fp8():
        aiter_ua(q8_static, k8, v8, out8, cu_q, width, seqused, depth, SCALE, True, (-1, -1),
                 block_table, 0.0, one, one, one)

    q8, desc = quantize_query(q)

    def f_kv4_kernel():
        kv4_attention(q8, desc, kv4, out4, cu_q, width, seqused, depth, SCALE, block_table)

    # kv4h: the HIP int4-WMMA kernel on its own layout
    kvh = torch.zeros(num_blocks, NKV, BS, 2 * H.TPT, dtype=torch.uint8, device=dev)
    kv4h_store(kk, kk, kvh, torch.arange(BS, BS + kk.shape[0], device=dev, dtype=torch.int64), k_group=KNOBS.get("k_group"))
    outh = torch.empty_like(q)
    outh_final = torch.empty_like(q)

    def f_kv4h():          # the backend path: q quantization, HIP kernel, merge with un-rotation
        kv4h_attention(q, kvh, outh, cu_q, width, seqused, depth, SCALE, block_table, output=outh_final, **KNOBS)

    t8 = timeit(f_fp8, reps)
    t4k = timeit(f_kv4_kernel, reps)
    th = timeit(f_kv4h, reps)
    import radiance_kv4h_kernel as KH
    print(f"  seqs={nseq} width={width} depth={depth:>6}:  fp8 {t8:8.1f} us   kv4 triton {t4k:8.1f} us "
          f"({t4k / t8:4.2f}x)   kv4h HIP {th:8.1f} us ({th / t8:4.2f}x)  {KH.LAST_LAUNCH.get('cfg', '')}")
    sys.stdout.flush()
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nseqs", default="1,4")
    ap.add_argument("--widths", default="8,1")
    ap.add_argument("--depths", default="4096,32768")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--prefill", default="", help="comma list of prompt lengths timed as width = depth (2D path)")
    ap.add_argument("--tile", type=int, default=0, help="KV4H tokens per tile: 16, 32 or 64 (0 = the launcher's default)")
    ap.add_argument("--rowtiles", type=int, default=0, help="KV4H 16-row tiles per program: 2 or 4")
    ap.add_argument("--qplanes", type=int, default=0, help="KV4H query planes: 2 (int8 query) or 1 (int4 query)")
    ap.add_argument("--kgroup", type=int, default=0, help="KV4H K dims per scale: 64, 128 or 256")
    ap.add_argument("--segments", type=int, default=0, help="KV4H split count (0 = AITER's table / the tune)")
    ap.add_argument("--reduce-warps", type=int, default=0, help="KV4H merge kernel warps (0 = auto)")
    ap.add_argument("--pref", type=int, default=-1, help="KV4H P reference: 0 running max, 1 tile max (-1 = env default)")
    ap.add_argument("--block-size", type=int, default=0, help="attention block (default 880; the engine uses 1664 with int4 cells)")
    ap.add_argument("--no-tune", action="store_true", help="aiter stock tables (A/B control)")
    ap.add_argument("--force", default=None,
                    help="cell override, e.g. 'block_m=64,tile=16,warps=4,stages=1,waves=2,segments=32'")
    a = ap.parse_args()
    if not a.no_tune:
        import radiance_attn_decode as T
        print("tune installed:", T.install())
        if a.force:
            f = dict(kv.split("=") for kv in a.force.split(","))
            f = {k: (v if v == "auto" or v.startswith("pcu") else int(v)) for k, v in f.items()}
            for k, v in (("r_warps", 4), ("r_stages", 1), ("r_waves", 2)):
                f.setdefault(k, v)
            T.FORCE = f
            print("FORCE:", f)
    for k, v in (("tile", a.tile), ("row_tiles", a.rowtiles), ("q_planes", a.qplanes), ("k_group", a.kgroup),
                 ("num_segments", a.segments), ("reduce_warps", a.reduce_warps)):
        if v:
            KNOBS[k] = v
    if a.pref >= 0:
        KNOBS["p_ref"] = a.pref
    if a.block_size:
        global BS
        BS = a.block_size
    print("device:", torch.cuda.get_device_name(), "kv4h knobs:", KNOBS or "defaults")
    shapes = [(nseq, width, depth) for width in map(int, a.widths.split(","))
              for nseq in map(int, a.nseqs.split(",")) for depth in map(int, a.depths.split(","))]
    if a.prefill:
        shapes += [(nseq, n, n) for nseq in map(int, a.nseqs.split(",")) for n in map(int, a.prefill.split(","))]
    for nseq, width, depth in shapes:
        try:
            bench(nseq, width, depth, a.reps, None)
        except torch.OutOfMemoryError:
            print(f"  seqs={nseq} width={width} depth={depth}: OOM (GPU shared with the server?)")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
