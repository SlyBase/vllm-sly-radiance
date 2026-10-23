#!/usr/bin/env python3
"""Grid sweep of the KV4H launch and format knobs, graph-timed per shape, against the fp8 kernel.

Launch knobs (free per call): tile 16 / 32 / 64, row tiles 2 / 4, split count (auto, 8 .. 128).
Format knobs (the store depends on them, so they are chosen once per serve): query planes
2 (int8) / 1 (int4), K dims per scale 64 / 128 / 256. Every (format) combination is its own
.so (radiance_kv4h_kernel._ext builds it); every (format, shape, launch) point is one CUDA graph
replayed REPS times, median, in microseconds -- the same thing the engine does (max_seqlen_k =
262144 as at capture time, so the auto split count is the captured one). The fp8 arm is AITER's
unified attention with the fork's decode tune, graph-timed the same way.

  python3 sly/kv4/tune_kv4h.py --csv /out/kv4h_tune.csv                  # ~25 min, GPU exclusive
  python3 sly/kv4/tune_kv4h.py --quick --csv /out/q.csv                   # 1 format, fewer shapes
  python3 sly/kv4/tune_kv4h.py --formats 2:128 --tiles 32,64 --rowtiles 4 --segments auto,32 --reduce-warps 2,4,8
  python3 sly/kv4/tune_kv4h.py --report /out/kv4h_tune.csv                # best per shape / format

Output: a CSV row per point (format, shape, knobs, us, fp8 us) and, at the end or with --report,
the best launch knobs per (format, shape) and the per-format sum over the shapes.
"""
import argparse
import csv
import itertools
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

dev = "cuda"
TARGET = dict(nq=24, nkv=4, head=256, window=0, causal=True)
DRAFTER = dict(nq=32, nkv=8, head=128, window=2048, causal=False)
BS = 1664               # the engine's attention block with int4 cells (fp8: 880); --block-size
MAX_K = 262144          # what the engine captures its decode graphs with


def shapes(quick, only="", prefill_qlen=2048):
    """(name, model, nseq, qlen, ctx); prefill_qlen: the chunk of the prefill shapes (--prefill-qlen,
    the serve's CHUNK; the shipped table was swept at 2048)"""
    if only == "prefill":
        out = []
        for ctx in (0, 16384, 32768, 65536, 98304):
            out.append((f"prefill{prefill_qlen // 1024}k c{ctx // 1024}k", TARGET, 1, prefill_qlen, max(ctx, prefill_qlen)))
        for qlen, ctx in ((512, 32768), (1024, 98304), (1664, 98304)):
            out.append((f"follow{qlen} c{ctx // 1024}k", TARGET, 1, qlen, ctx))
        return out
    out = []
    ctxs = (4096, 32768) if quick else (4096, 32768, 131072)
    seqs = (1, 4) if quick else (1, 4, 8)
    for ns in seqs:
        for ctx in ctxs:
            out.append((f"decode1 s{ns} c{ctx // 1024}k", TARGET, ns, 1, ctx))
            out.append((f"verify8 s{ns} c{ctx // 1024}k", TARGET, ns, 8, ctx))
    # prefill: the engine's chunk (2048 batched tokens by default here) on a fresh / long prefix, and the
    # follow-up turns the 1664-token prefix-cache granularity produces (a few hundred to ~1.7k tokens
    # recomputed on a long cache)
    for ctx in ((2048, 16384) if quick else (2048, 16384, 65536)):
        out.append((f"prefill2k c{ctx // 1024}k", TARGET, 1, 2048, ctx))
    if not quick:
        for qlen, ctx in ((512, 32768), (1664, 32768), (512, 98304), (1664, 98304)):
            out.append((f"follow{qlen} c{ctx // 1024}k", TARGET, 1, qlen, ctx))
    for ns in seqs:
        out.append((f"drafter s{ns} c32k", DRAFTER, ns, 8, 32768))
    return out


def gtime(fn, reps):
    import torch
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1000.0)
    ts.sort()
    return ts[len(ts) // 2]


class Shape:
    def __init__(self, name, model, nseq, qlen, ctx, k_group):
        import torch
        import radiance_kv4h_format as H
        from radiance_kv4h_store import kv4h_store
        self.name, self.model, self.nseq, self.qlen, self.ctx = name, model, nseq, qlen, ctx
        m = model
        self.L = H.layout(m["head"], k_group)
        seq = ctx if qlen >= ctx else ctx + qlen
        self.seq = seq
        nb = -(-seq // BS)
        num_blocks = nseq * nb + 1
        self.bt = (torch.arange(nseq * nb, device=dev).view(nseq, nb) + 1).to(torch.int32)
        self.seqused = torch.full((nseq,), seq, dtype=torch.int32, device=dev)
        self.cu = torch.arange(0, nseq * qlen + 1, qlen, dtype=torch.int32, device=dev)
        self.q = torch.randn(nseq * qlen, m["nq"], m["head"], device=dev).to(torch.bfloat16)
        self.cache = torch.zeros(num_blocks, m["nkv"], BS, 2 * self.L.TPT, dtype=torch.uint8, device=dev)
        kk = torch.randn(min(seq, 4096), m["nkv"], m["head"], device=dev).to(torch.bfloat16)
        kv4h_store(kk, kk, self.cache, torch.arange(BS, BS + kk.shape[0], device=dev, dtype=torch.int64), k_group=k_group)
        self.out = torch.empty_like(self.q)
        self.fin = torch.empty_like(self.q)
        self.scale = 1.0 / math.sqrt(m["head"])
        self.max_q = qlen
        self.max_k = MAX_K if qlen <= 16 else seq      # prefill chunks are eager in the engine

    def run(self, **knobs):
        from radiance_kv4h_kernel import kv4h_attention
        m = self.model
        kv4h_attention(self.q, self.cache, self.out, self.cu, self.max_q, self.seqused, self.max_k, self.scale,
                       self.bt, causal=m["causal"], window=m["window"], output=self.fin, **knobs)

    def r4d(self):
        """libr4d's paged prefill (what the fp8 production path runs for chunks >= 512 tokens) on an
        fp8 cache with the fp8 serve's 880-token blocks, launched like radiance_r4d_hybrid_attn;
        None for decode / verify shapes and the drafter."""
        if self.model is DRAFTER or self.qlen < 512:
            return None
        import torch
        import radiance_r4d_attn as RA
        if RA.r4d is None or RA._PREFILL[0] is None:
            return None
        m = self.model
        bs8, sub = 880, RA.BLOCK_SIZE
        nb = -(-self.seq // bs8)
        kv8 = torch.randn(self.nseq * nb + 1, m["nkv"], bs8, 2 * m["head"], device=dev).to(torch.float8_e4m3fn)
        C = kv8.shape[3]
        mult, ratio = kv8.stride(0) // (sub * C), bs8 // sub
        ids = (torch.arange(self.nseq * nb, device=dev).view(self.nseq, nb) + 1).to(torch.int32) * mult
        table = (ids.unsqueeze(-1) + torch.arange(ratio, dtype=torch.int32, device=dev)).reshape(self.nseq, nb * ratio).contiguous()
        q = self.q.contiguous()
        out = torch.empty_like(q)
        scratch = torch.empty(256, dtype=torch.uint8, device=dev)
        launch = RA._PREFILL[0]

        def f():
            launch(q.data_ptr(), kv8.data_ptr(), table.data_ptr(), self.seqused.data_ptr(), out.data_ptr(), 0, 0,
                   scratch.data_ptr(), self.nseq, self.qlen, m["nq"], m["nkv"], m["head"], sub, table.shape[1],
                   sub * C, kv8.stride(1), self.scale, 0, self.seq, torch.cuda.current_stream().cuda_stream)
        return f

    def fp8(self):
        """AITER fp8 unified attention (the fork's tune installed) on the same shape; None for the
        drafter (its fp8 path is vLLM's TRITON_ATTN with the fork's window launch, not AITER)."""
        if self.model is DRAFTER:
            return None
        import torch
        from aiter.ops.triton.attention.unified_attention import unified_attention as aiter_ua
        FP8 = torch.float8_e4m3fn
        m = self.model
        nb = self.bt.shape[1]
        kv8 = torch.randn(self.nseq * nb + 1, m["nkv"], BS, 2 * m["head"], device=dev).to(FP8)
        k8, v8 = kv8.transpose(1, 2).split(m["head"], dim=-1)
        q8 = self.q.float().to(FP8)
        one = torch.ones(1, dtype=torch.float32, device=dev)
        out8 = torch.empty_like(self.q)

        def f():
            aiter_ua(q8, k8, v8, out8, self.cu, self.max_q, self.seqused, self.max_k, self.scale, True, (-1, -1),
                     self.bt, 0.0, one, one, one)
        return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--report", default="", help="only summarize an existing CSV")
    ap.add_argument("--emit-table", default="", help="write the knob table (JSON) the launcher reads from an existing CSV")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", default="", help="'prefill': only the prefill / follow-up shapes (libr4d reference arm)")
    ap.add_argument("--prefill-qlen", type=int, default=2048, help="chunk of the --only prefill shapes (serve CHUNK, e.g. 4096)")
    ap.add_argument("--pipe", type=int, default=0, help="KV4H_PIPE schedule mask for every timed launch (0 = stock)")
    ap.add_argument("--formats", default="2:128,1:128,2:256,2:64,1:256", help="qplanes:kgroup list")
    ap.add_argument("--tiles", default="16,32,64")
    ap.add_argument("--rowtiles", default="2,4")
    ap.add_argument("--segments", default="auto,8,16,32,64")
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--block-size", type=int, default=1664, help="attention block (engine: 1664 with int4 cells)")
    ap.add_argument("--reduce-warps", default="", help="comma list of RADIANCE_KV4H_REDUCE_WARPS values to sweep (default: the launcher's)")
    ap.add_argument("--no-tune", action="store_true", help="AITER stock tables for the fp8 arm")
    a = ap.parse_args()
    if a.emit_table:
        return emit_table(a.report or a.csv, a.emit_table)
    if a.report:
        return report(a.report)
    import torch
    from radiance_kv4h_kernel import LAST_LAUNCH
    if not a.no_tune:
        import radiance_attn_decode as T
        print("decode tune installed:", T.install())
    formats = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in a.formats.split(",")]
    if a.quick:
        formats = formats[:1]
    global BS
    BS = a.block_size
    tiles = [int(x) for x in a.tiles.split(",")]
    rowtiles = [int(x) for x in a.rowtiles.split(",")]
    segments = [None if x == "auto" else int(x) for x in a.segments.split(",")]
    rwarps = [int(x) for x in a.reduce_warps.split(",")] if a.reduce_warps else [None]
    rows = []
    csvf = open(a.csv, "w", newline="") if a.csv else None
    wr = csv.writer(csvf) if csvf else None
    if wr:
        wr.writerow(["qplanes", "kgroup", "shape", "nseq", "qlen", "ctx", "tile", "rowtiles", "segments", "nseg", "rwarps", "us",
                     "fp8_us", "r4d_us"])
    fp8_cache = {}
    r4d_cache = {}
    print("device:", torch.cuda.get_device_name())
    for qp, kg in formats:
        print(f"=== format: query planes {qp}, K group {kg}")
        sys.stdout.flush()
        for name, model, nseq, qlen, ctx in shapes(a.quick, a.only, a.prefill_qlen):
            if model["head"] % kg:
                continue
            sh = Shape(name, model, nseq, qlen, ctx, kg)
            if name not in fp8_cache:
                f = sh.fp8()
                fp8_cache[name] = gtime(f, a.reps) if f else float("nan")
                rf = sh.r4d()
                r4d_cache[name] = gtime(rf, a.reps) if rf else float("nan")
            points = [(t, r, s, w) for t in tiles for r in rowtiles for s in (segments if qlen <= 16 else [None])
                      for w in (rwarps if qlen <= 16 else [None])]
            random.shuffle(points)
            best = None
            for t, r, s, w in points:
                try:
                    us = gtime(lambda: sh.run(tile=t, row_tiles=r, q_planes=qp, k_group=kg, num_segments=s,
                                              reduce_warps=w, pipe=a.pipe), a.reps)
                    nseg = LAST_LAUNCH["cfg"]["nseg"]
                except Exception as e:  # noqa: BLE001
                    us, nseg = float("nan"), -1
                    print(f"  {name} tile={t} r={r} seg={s} rw={w}: {type(e).__name__}: {str(e)[:60]}")
                row = (qp, kg, name, nseq, qlen, ctx, t, r, s or 0, nseg, w or 0, us, fp8_cache[name], r4d_cache[name])
                rows.append(row)
                if wr:
                    wr.writerow(row)
                    csvf.flush()
                if us == us and (best is None or us < best[0]):
                    best = (us, t, r, s, nseg, w)
            fp = fp8_cache[name]
            ratio = f"{best[0] / fp:.2f}x fp8" if fp == fp else "(no fp8 arm)"
            rr = r4d_cache[name]
            if rr == rr:
                ratio += f"  {best[0] / rr:.2f}x libr4d ({rr:.0f} us)"
            print(f"  {name:22s} best {best[0]:8.1f} us  tile={best[1]} rowtiles={best[2]} segments={best[3] or 'auto'}"
                  f"({best[4]}) rwarps={best[5] or 'auto'}  {ratio}")
            sys.stdout.flush()
            torch.cuda.empty_cache()
    if csvf:
        csvf.close()
        report(a.csv)


def _mode(shape):
    for m in ("drafter", "prefill", "decode1"):
        if shape.startswith(m):
            return m
    return "verify"


def emit_table(csv_path, out_path):
    """The launcher's table: per (format, mode, sequences | prefill context) the launch knobs
    with the smallest summed time over the other sweep dimension (contexts for decode / verify:
    a CUDA graph is captured per batch size, not per context, so the choice must hold across
    contexts; for prefill chunks the context is known at launch)."""
    import json
    rows = [r for r in csv.DictReader(open(csv_path)) if float(r["us"]) == float(r["us"])]
    agg = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(lambda: defaultdict(int))
    for r in rows:
        m = _mode(r["shape"])
        key = (int(r["qplanes"]), int(r["kgroup"]), m, int(r["ctx"]) if m == "prefill" else int(r["nseq"]))
        knob = (int(r["tile"]), int(r["rowtiles"]), int(r["segments"]), int(r.get("rwarps") or 0))
        agg[key][knob] += float(r["us"])
        cnt[key][knob] += 1
    table = {}
    for key, d in agg.items():
        full = max(cnt[key].values())
        cands = {k: v for k, v in d.items() if cnt[key][k] == full}
        best = min(cands, key=cands.get)
        qp, kg, m, b = key
        table.setdefault(f"{qp}:{kg}", {}).setdefault(m, {})[str(b)] = {
            "tile": best[0], "row_tiles": best[1], "num_segments": best[2] or None,
            "reduce_warps": best[3] or None, "us": round(cands[best], 1)}
    meta = {"source": str(csv_path),
            "key": "format 'qplanes:kgroup' -> mode (decode1 | verify | prefill | drafter) -> sequences (prefill: context) -> knobs"}
    json.dump({"meta": meta, "table": table}, open(out_path, "w"), indent=1, sort_keys=True)
    print(f"wrote {out_path}: {sum(len(v) for f in table.values() for v in f.values())} entries")


def report(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["us"] = float(r["us"])
        r["fp8_us"] = float(r["fp8_us"])
    print("\n== best launch knobs per (format, shape); 'default' = tile 32, rowtiles 4, auto segments")
    fmt_sum = defaultdict(float)
    fmt_sum_default = defaultdict(float)
    for (qp, kg), fr in itertools.groupby(sorted(rows, key=lambda r: (r["qplanes"], r["kgroup"], r["shape"])),
                                          key=lambda r: (r["qplanes"], r["kgroup"])):
        print(f"-- query planes {qp}, K group {kg}")
        for shape, sr in itertools.groupby(fr, key=lambda r: r["shape"]):
            sr = [r for r in sr if r["us"] == r["us"]]
            if not sr:
                continue
            b = min(sr, key=lambda r: r["us"])
            d = [r for r in sr if r["tile"] == "32" and r["rowtiles"] == "4" and r["segments"] == "0"]
            d_us = d[0]["us"] if d else float("nan")
            fmt_sum[(qp, kg)] += b["us"]
            fmt_sum_default[(qp, kg)] += d_us
            fp = b["fp8_us"]
            print(f"   {shape:22s} best {b['us']:8.1f} us (tile {b['tile']}, rowtiles {b['rowtiles']}, segments "
                  f"{b['segments'] if b['segments'] != '0' else 'auto'}/{b['nseg']})   default {d_us:8.1f}   "
                  f"fp8 {fp:8.1f}  -> best {b['us'] / fp:.2f}x" if fp == fp else
                  f"   {shape:22s} best {b['us']:8.1f} us (tile {b['tile']}, rowtiles {b['rowtiles']}, segments "
                  f"{b['segments'] if b['segments'] != '0' else 'auto'}/{b['nseg']})   default {d_us:8.1f}")
    print("\n== sum over the shapes (us): best knobs vs default knobs, per format")
    for k in sorted(fmt_sum):
        print(f"   query planes {k[0]}, K group {k[1]:>3s}: best {fmt_sum[k]:9.0f}   default {fmt_sum_default[k]:9.0f}")


if __name__ == "__main__":
    main()
