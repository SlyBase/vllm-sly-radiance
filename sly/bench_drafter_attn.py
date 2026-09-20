#!/usr/bin/env python3
"""vLLM's Triton unified attention on the DFlash2 drafter's call: the stock launch vs tuned launches.

The drafter (syvai/Qwen3.8-27B-DFlash2-W4A16, 5 layers, every layer sliding-window 2048) runs one forward per
engine step through vLLM's own `unified_attention` (--speculative-config.attention_backend TRITON_ATTN):
32 q / 8 kv heads (4 per kv head), head 128, bf16 queries, fp8 per-tensor KV, window (2047, 0),
8 queries per sequence (bonus + 7 mask tokens), seq_len = context + 8, attention page 896, NON-causal (the
drafter's attention metadata has causal=False; keys allowed: seq_len - 2055 .. seq_len - 1). Context K/V are
written into the cache before the forward, so what runs is a q_len-8 attention over the last 2056 keys.

Measured in rocprofv3 on 0.2.7 (one stream): 31.6 us per call at 0.1k context, 200 us from 32k on, 5 calls per
step, i.e. ~20-40 GB/s for ~4 MB of window. The launch explains it: `unified_attention` picks BLOCK_M 16 =
BLOCK_Q 4 for GQA 4, and the 3D (split-KV) path is closed for max_seqlen_q > 1, so the grid is
(8 // 4 + 1) q-blocks x 8 kv heads = 24 workgroups on 32 CUs, each walking its window sequentially.

Two things are tried here, everything else is the same kernel:
  * the 2D launch with another BLOCK_M / TILE / warps / stages;
  * the 3D launch (IS_3D, NUM_SEGMENTS_PER_SEQ) -- but its segments partition [0, seq_len), not the window, so at
    99k context the window lives in one of S segments. `shift` therefore slices the block table and seq_len down
    to the window first (drop whole leading blocks: positions are only used relative to the sequence's end, RoPE
    is already in the cached K) and splits that.

Time per call = CUDA-graph replay of R calls over R disjoint KV pools / R, median of --reps; it includes the
slicing ops and the reduce. Correctness: --check compares every cell against the stock call and against an
fp32 torch reference.

  PYTHONPATH=<repo>/sly python3 bench_drafter_attn.py --check      # also drives radiance_attn_drafter itself ("module")
  python3 bench_drafter_attn.py --table --csv /out/drafter_table.csv
  python3 bench_drafter_attn.py --sweep 3d --nseqs 1,8 --depths 4096,32768 --warm 6 --csv /out/d3.csv
  python3 bench_drafter_attn.py --rank /out/d3.csv --top 12
"""
import argparse
import csv
import itertools
import os
import statistics
import subprocess
import sys
import time

NQ, NKV, HS = 32, 8, 128
NQPKV = NQ // NKV
WINDOW = 2048
QLEN = 8
BLOCK = 896
NB_WIN = (WINDOW + QLEN - 1 + BLOCK - 2) // BLOCK + 1        # table entries a window (+ its queries) can touch
MAXLEN = 262144
CAUSAL = False                                                # the drafter runs non-causal (dflash use_non_causal)
DEPTHS = [1024, 2048, 4096, 8192, 32768, 98304]
NSEQS = [1, 2, 4, 8]
BYTES_PER_TOKEN = NKV * 2 * HS                                # K + V, fp8, all kv heads
NBP = NB_WIN                                                  # physical blocks per (slot, sequence) in a pool
POOL_BUDGET = 4 << 30

# cell fields: mode "2d" | "3d", bm, tile, warps, stages (None = Triton default), segs (3d), shift (bool)
CELLS = {
    "replica": dict(mode="2d", bm=16, tile=32, warps=None, stages=None, segs=1, shift=False),   # what vLLM launches
    "shift3d": dict(mode="3d", bm=16, tile=32, warps=2, stages=1, segs=8, shift=True),
    "noshift3d": dict(mode="3d", bm=16, tile=32, warps=2, stages=1, segs=8, shift=False),
}


def set_window(w):
    """--window: the drafter checkpoint has 2048; larger values price a wider window (dflash_config.swa_window_size)."""
    global WINDOW, NB_WIN, NBP
    WINDOW = w
    NB_WIN = (WINDOW + QLEN - 1 + BLOCK - 2) // BLOCK + 1
    NBP = NB_WIN


def parse_cell(spec):
    if "=" not in spec:
        return spec, dict(CELLS[spec])
    name, _, body = spec.partition("=")
    cell = dict(CELLS["shift3d"])
    for kv in body.split(","):
        k, _, v = kv.partition(":")
        cell[k] = v if k in ("mode", "skip") else (v == "1" if k == "shift" else (None if v == "None" else int(v)))
    return name, cell


def tag(c):
    return (f"{c['mode']}_bm{c['bm']}_t{c['tile']}_w{c['warps']}_s{c['stages']}_g{c['segs']}"
            f"{'_shift' if c['shift'] else ''}")


class Bench:
    def __init__(self, args):
        import torch
        import vllm.v1.attention.ops.triton_unified_attention as TU
        from vllm.platforms import current_platform
        from vllm.v1.kv_cache_interface import KVQuantMode
        self.torch, self.TU, self.KVQuantMode = torch, TU, KVQuantMode
        try:                                     # the shipped module: its gate, segment rule and fused slicing
            import radiance_attn_drafter as RAD
            RAD.install()
            self.RAD = RAD if RAD._state.get("wrapper") is not None else None
        except Exception as e:
            print(f"radiance_attn_drafter not usable: {e!r}", file=sys.stderr)
            self.RAD = None
        self.dev = torch.device("cuda")
        self.fp8 = current_platform.fp8_dtype()
        self.args = args
        self.pool_budget = (256 << 20) if args.worker else POOL_BUDGET
        self.pools = {}
        self.bad = {}
        self.arange_nbw = torch.arange(NB_WIN, device=self.dev, dtype=torch.int32)
        g = torch.Generator(device=self.dev).manual_seed(0)
        self.tile_data = torch.randn(NBP, NKV, BLOCK, 2 * HS, device=self.dev, generator=g).to(self.fp8)

    def pool(self, nseq):
        torch = self.torch
        if nseq in self.pools:
            return self.pools[nseq]
        self.pools.clear()
        torch.cuda.empty_cache()
        per_slot = nseq * NBP * NKV * BLOCK * 2 * HS
        R = max(1, min(32, self.pool_budget // per_slot))
        buf = torch.empty(R * nseq * NBP, NKV, BLOCK, 2 * HS, dtype=self.fp8, device=self.dev)
        for i in range(0, buf.shape[0], NBP):
            buf[i:i + NBP].copy_(self.tile_data)
        k, v = buf.transpose(1, 2).split(HS, dim=-1)          # [B, N, H, hs] views, as the backend does
        nbt = -(-(max(DEPTHS) + QLEN) // BLOCK)
        cols = torch.arange(nbt, device=self.dev, dtype=torch.int32) % NBP
        tables = [((r * nseq + torch.arange(nseq, device=self.dev, dtype=torch.int32))[:, None] * NBP
                   + cols[None, :]).contiguous() for r in range(R)]
        self.pools[nseq] = dict(buf=buf, k=k, v=v, tables=tables, R=R)
        return self.pools[nseq]

    def case(self, nseq):
        torch = self.torch
        p = self.pool(nseq)
        ntok = nseq * QLEN
        return dict(
            p, nseq=nseq,
            q=torch.randn(ntok, NQ, HS, device=self.dev).to(torch.bfloat16),
            out=torch.empty(ntok, NQ, HS, dtype=torch.bfloat16, device=self.dev),
            cu=torch.arange(nseq + 1, device=self.dev, dtype=torch.int32) * QLEN,
            seq=torch.full((nseq,), QLEN, dtype=torch.int32, device=self.dev),
            scale=torch.ones(1, device=self.dev, dtype=torch.float32).expand(nseq, NKV))

    # ---- the call vLLM makes (triton_attn.py forward) ----
    def stock_kwargs(self, c, r):
        return dict(
            q=c["q"], k=c["k"], v=c["v"], out=c["out"], cu_seqlens_q=c["cu"], max_seqlen_q=QLEN, seqused_k=c["seq"],
            max_seqlen_k=MAXLEN + QLEN, softmax_scale=HS ** -0.5, causal=CAUSAL, alibi_slopes=None, use_alibi_sqrt=False,
            window_size=(WINDOW - 1, 0), block_table=c["tables"][r], softcap=0, q_descale=None, k_descale=c["scale"],
            v_descale=c["scale"], seq_threshold_3D=None, num_par_softmax_segments=None, softmax_segm_output=None,
            softmax_segm_max=None, softmax_segm_expsum=None, sinks=None, output_scale=None, mm_prefix_range=None,
            rswa_prefix_lens=None, rswa_window=None, kv_quant_mode=self.KVQuantMode.FP8_PER_TENSOR,
            k_scale_cache=None, v_scale_cache=None, chunk_lookback=-1, use_td=False)

    def stock(self, c, r):
        return self.TU.unified_attention(**self.stock_kwargs(c, r))

    def module(self, c, r):
        """vLLM's call, routed through the installed radiance_attn_drafter wrapper."""
        assert self.RAD is not None, "radiance_attn_drafter is not installed"
        self.RAD.ENABLED = True
        return self.RAD._state["wrapper"](**self.stock_kwargs(c, r))

    # ---- the tuned launch: same kernels, other constexprs, optionally on the window only ----
    def custom(self, c, r, cfg, out=None):
        torch, TU = self.torch, self.TU
        q, cu, nseq = c["q"], c["cu"], c["nseq"]
        out = c["out"] if out is None else out
        bt, seq = c["tables"][r], c["seq"]
        if cfg["shift"]:
            first = (seq - (WINDOW + QLEN - 1)).clamp_min(0) // BLOCK
            idx = (first[:, None] + self.arange_nbw[None, :]).clamp_max(bt.shape[1] - 1).long()
            bt, seq = bt.gather(1, idx), seq - first * BLOCK
        if cfg.get("skip") == "kernel":                          # timing only: the slicing ops
            return
        bm, tile, three = cfg["bm"], cfg["tile"], cfg["mode"] == "3d"
        bq = bm // NQPKV
        segs = cfg["segs"] if three else 1
        total_q_blocks = q.shape[0] // bq + nseq
        grid = (total_q_blocks, NKV, segs) if three else (total_q_blocks, NKV)
        if three:
            so = torch.empty(q.shape[0], NQ, segs, HS, dtype=torch.float32, device=self.dev)
            sm = torch.empty(q.shape[0], NQ, segs, dtype=torch.float32, device=self.dev)
            se = torch.empty(q.shape[0], NQ, segs, dtype=torch.float32, device=self.dev)
        else:
            so = sm = se = None
        launch = {}
        if cfg["warps"] is not None:
            launch["num_warps"] = cfg["warps"]
        if cfg["stages"] is not None:
            launch["num_stages"] = cfg["stages"]
        k, v = c["k"], c["v"]
        TU.kernel_unified_attention[grid](
            output_ptr=out, segm_output_ptr=so, segm_max_ptr=sm, segm_expsum_ptr=se, query_ptr=q, key_cache_ptr=k,
            value_cache_ptr=v, sink_ptr=None, block_tables_ptr=bt, seq_lens_ptr=seq, alibi_slopes_ptr=None,
            qq_bias_ptr=None, k_scale_cache_ptr=None, v_scale_cache_ptr=None, scale=HS ** -0.5, q_scale=None,
            k_scale=c["scale"], v_scale=c["scale"], out_scale=1.0, softcap=0, num_query_heads=NQ,
            num_queries_per_kv=NQPKV, block_table_stride=bt.stride(0), query_stride_0=q.stride(0),
            query_stride_1=q.stride(1), output_stride_0=out.stride(0), output_stride_1=out.stride(1),
            qq_bias_stride_0=0, BLOCK_SIZE=BLOCK, TILE_SIZE=tile, HEAD_SIZE=HS, HEAD_SIZE_PADDED=HS,
            USE_ALIBI_SLOPES=False, USE_ALIBI_SQRT=False, USE_QQ_BIAS=False, USE_SOFTCAP=False, USE_SINKS=False,
            SLIDING_WINDOW=WINDOW, USE_CAUSAL=CAUSAL, USE_PER_SEQ_CAUSAL=False, per_seq_causal_ptr=None,
            USE_MM_PREFIX=False, MAX_MM_RANGES=0, mm_prefix_range_ptr=None, rswa_prefix_lens_ptr=seq,
            R_SWA_WINDOW=0, USE_R_SWA=False, stride_k_cache_0=k.stride(0), stride_k_cache_1=k.stride(1),
            stride_k_cache_2=k.stride(2), stride_k_cache_3=k.stride(3), stride_v_cache_0=v.stride(0),
            stride_v_cache_1=v.stride(1), stride_v_cache_2=v.stride(2), stride_v_cache_3=v.stride(3),
            stride_ks_blk=None, stride_ks_slot=None, stride_ks_head=None, stride_vs_blk=None,
            stride_vs_slot=None, stride_vs_head=None, query_start_len_ptr=cu, BLOCK_Q=bq, num_seqs=nseq,
            BLOCK_M=bm, NUM_SEGMENTS_PER_SEQ=segs, USE_FP8=False, IS_3D=three,
            KV_QUANT_MODE=self.KVQuantMode.FP8_PER_TENSOR, Q_IS_FP8=False, CHUNK_LOOKBACK=-1, CHUNK_SIZE=-1,
            USE_TD=False, USE_TD_QO=False, MM_PREFIX_CLAMP_SW=False, **launch)
        if three and cfg.get("skip") != "reduce":
            TU.reduce_segments[(q.shape[0], NQ)](
                output_ptr=out, segm_output_ptr=so, segm_max_ptr=sm, segm_expsum_ptr=se, seq_lens_ptr=seq,
                num_seqs=nseq, num_query_heads=NQ, out_scale_inv=1.0, output_stride_0=out.stride(0),
                output_stride_1=out.stride(1), block_table_stride=bt.stride(0), TILE_SIZE=tile, HEAD_SIZE=HS,
                HEAD_SIZE_PADDED=HS, query_start_len_ptr=cu, BLOCK_Q=bq, NUM_SEGMENTS_PER_SEQ=segs, USE_FP8=False)

    def call(self, c, r, cfg):
        if cfg == "module":
            return self.module(c, r)
        return self.stock(c, r) if cfg is None else self.custom(c, r, cfg)

    def graph(self, c, cfg):
        torch = self.torch
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        before = self.RAD._state["calls"] if cfg == "module" else 0
        with torch.cuda.stream(s):                               # compiles every variant this launch needs
            for r in range(c["R"]):
                self.call(c, r, cfg)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        if cfg == "module":                                        # a declined call would silently be vLLM's own
            assert self.RAD._state["calls"] - before >= c["R"], "the module did not take the call"
            assert not self.RAD._state["failed"], "the module fell back"
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for r in range(c["R"]):
                self.call(c, r, cfg)
        return g

    def time_graph(self, g, c, depth):
        torch = self.torch
        c["seq"].fill_(depth + QLEN)
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        one = e0.elapsed_time(e1)
        iters = max(2, min(400, int(30.0 / max(one, 0.005))))
        reps = []
        for _ in range(self.args.reps):
            e0.record()
            for _ in range(iters):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            reps.append(e0.elapsed_time(e1) * 1000.0 / iters / c["R"])
        return statistics.median(reps)

    def run(self, name, cfg, nseq, depths):
        if name in self.bad:
            return None
        c = self.case(nseq)
        try:
            g = self.graph(c, cfg)
            rows = [(d, self.time_graph(g, c, d)) for d in depths]
            del g
        except Exception as e:
            self.bad[name] = f"{type(e).__name__}: {str(e)[:160]}"
            print(f"  FAIL {name}: {self.bad[name]}", file=sys.stderr, flush=True)
            return None
        return rows


def load_done(path):
    done = set()
    if path and os.path.exists(path):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["config"], int(r["nseq"]), int(r["depth"])))
    return done


def sweep_cells(kind):
    out = {}
    if kind == "2d":
        for bm, tile, w, s in itertools.product((16, 32), (16, 32, 64), (1, 2, 4, 8), (1, 2)):
            cell = dict(mode="2d", bm=bm, tile=tile, warps=w, stages=s, segs=1, shift=False)
            out[tag(cell)] = cell
    elif kind == "3d":
        for bm, tile, w, s, g in itertools.product((16, 32), (16, 32, 64), (1, 2, 4), (1, 2), (2, 4, 8, 16)):
            cell = dict(mode="3d", bm=bm, tile=tile, warps=w, stages=s, segs=g, shift=True)
            out[tag(cell)] = cell
    else:
        raise SystemExit(f"unknown sweep {kind}")
    return out


def print_table(path, base="stock", top=0):
    data = {}
    for r in csv.DictReader(open(path, newline="")):
        data.setdefault(int(r["nseq"]), {}).setdefault(r["config"], {})[int(r["depth"])] = float(r["us"])
    for nseq, cfgs in sorted(data.items()):
        depths = sorted({d for c in cfgs.values() for d in c})
        print(f"\nnseq={nseq}   us/call (speedup vs {base})")
        print(f"{'config':44s} " + " ".join(f"{d // 1024 if d >= 1024 else d}k".rjust(13) for d in depths))
        ref = cfgs.get(base, {})
        rows = sorted(cfgs.items(), key=lambda kv: (kv[0] != base, sum(kv[1].values()) / len(kv[1])))
        keep = rows[:1] + rows[1:][:top or None] if rows and rows[0][0] == base else rows[:top or None]
        for name, c in keep:
            cells = []
            for d in depths:
                if d not in c:
                    cells.append(" " * 13)
                elif d in ref and name != base:
                    cells.append(f"{c[d]:7.1f} {ref[d] / c[d]:4.1f}x")
                else:
                    cells.append(f"{c[d]:7.1f}{'':6s}")
            print(f"{name[:44]:44s} " + " ".join(cells))


def rank(path, top=15):
    import math
    per = {}
    for r in csv.DictReader(open(path, newline="")):
        per.setdefault((int(r["nseq"]), int(r["depth"])), {})[r["config"]] = float(r["us"])
    shapes = sorted(per)
    common = set.intersection(*(set(per[k]) for k in shapes)) if shapes else set()
    rows = []
    for name in common:
        ratios = [per[k][name] / min(per[k].values()) for k in shapes]
        rows.append((math.exp(sum(map(math.log, ratios)) / len(ratios)), max(ratios), name))
    rows.sort()
    print(f"{len(shapes)} shapes, {len(common)} configs on all of them (score = geomean time / best-per-shape)")
    print(f"{'score':>7s} {'worst':>7s}  config")
    for sc, worst, name in rows[:top]:
        print(f"{sc:7.3f} {worst:7.3f}  {name}")
    if "stock" in common:
        print(f"stock: {[x for x in rows if x[2] == 'stock'][0][0]:.3f}")


def strip_warm(argv):
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
        elif x == "--warm":
            skip = True
        elif not x.startswith("--warm="):
            out.append(x)
    return out


def cmd_grid(a):
    if a.sweep:
        configs = list(sweep_cells(a.sweep).items())
        if a.with_stock:
            configs.insert(0, ("stock", None))
    else:
        sep = ";" if ";" in a.configs else ","
        configs = [(s, None) if s == "stock" else ((s, "module") if s == "module" else parse_cell(s))
                   for s in a.configs.split(sep)]
    nseqs = [int(x) for x in a.nseqs.split(",")]
    depths = [int(x) for x in a.depths.split(",")]
    if a.warm and not a.worker:
        t0 = time.time()
        procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", f"{i}/{a.warm}"]
                                  + strip_warm(sys.argv[1:]), stdout=subprocess.DEVNULL) for i in range(a.warm)]
        for p in procs:
            p.wait()
        print(f"warm-up: {a.warm} workers done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    b = Bench(a)
    if a.worker:
        i, n = (int(x) for x in a.worker.split("/"))
        configs, nseqs, depths = configs[i::n], sorted({nseqs[0], nseqs[-1]}), depths[:1]
        a.reps, a.csv = 1, None
    done = load_done(a.csv)
    fh = writer = None
    if a.csv:
        new = not os.path.exists(a.csv)
        fh = open(a.csv, "a", newline="")
        writer = csv.writer(fh)
        if new:
            writer.writerow(["config", "nseq", "depth", "us", "GBps_window"])
    for nseq in nseqs:
        for name, cfg in configs:
            todo = [d for d in depths if (name, nseq, d) not in done]
            if not todo:
                continue
            t0 = time.time()
            rows = b.run(name, cfg, nseq, todo)
            if rows is None:
                continue
            if writer:
                for d, us in rows:
                    win = min(d + QLEN, WINDOW + QLEN - 1)
                    writer.writerow([name, nseq, d, f"{us:.2f}", f"{nseq * win * BYTES_PER_TOKEN / (us * 1e3):.1f}"])
                fh.flush()
            print(f"nseq={nseq} {name[:48]:48s} " + " ".join(f"{d // 1024 if d >= 1024 else d}k:{us:.1f}" for d, us in rows)
                  + f"   ({time.time() - t0:.0f}s)", flush=True)
    if a.csv:
        fh.close()
        print_table(a.csv, top=a.top)


def reference(b, c, depth, nseq):
    """fp32 sliding-window causal attention of the 8 queries of every sequence, from the cached fp8 data."""
    torch = b.torch
    outs = []
    L = depth + QLEN
    for s in range(nseq):
        bt = c["tables"][0][s]
        lo = max(0, L - (WINDOW + QLEN - 1))
        pos = torch.arange(lo, L, device=b.dev)
        blk = bt[pos // BLOCK].long()
        off = pos % BLOCK
        K = c["buf"][blk, :, off, :HS].float()                   # [n, NKV, HS]
        V = c["buf"][blk, :, off, HS:].float()
        q = c["q"][s * QLEN:(s + 1) * QLEN].float()              # [8, NQ, HS]
        o = torch.empty_like(q)
        qabs = depth + torch.arange(QLEN, device=b.dev)
        if CAUSAL:
            allowed = (pos[None, :] <= qabs[:, None]) & (pos[None, :] > qabs[:, None] - WINDOW)
        else:                                                    # key < seq_len and |key - query| < window
            allowed = ((pos[None, :] < L) & (pos[None, :] > qabs[:, None] - WINDOW)
                       & (pos[None, :] < qabs[:, None] + WINDOW))
        for h in range(NQ):
            kvh = h // NQPKV
            sc = (q[:, h] @ K[:, kvh].T) * HS ** -0.5
            sc = sc.masked_fill(~allowed, float("-inf"))
            o[:, h] = torch.softmax(sc, -1) @ V[:, kvh]
        outs.append(o)
    return torch.cat(outs)


def cmd_check(a):
    """Every cell against the stock call and an fp32 reference; the tune must be no less accurate than stock."""
    b = Bench(a)
    torch = b.torch
    cells = dict(CELLS)
    cells["shift3d_g16_t16"] = dict(mode="3d", bm=16, tile=16, warps=1, stages=1, segs=16, shift=True)
    cells["bm32_2d"] = dict(mode="2d", bm=32, tile=32, warps=4, stages=1, segs=1, shift=False)
    worst = 0.0
    for nseq, depth in ((1, 3000), (3, 9000), (8, 40000), (2, 1500), (1, 2100), (4, 12000)):
        c = b.case(nseq)
        g = torch.Generator(device=b.dev).manual_seed(2)
        nbt = c["tables"][0].shape[1]
        for i in range(nseq * NBP):                              # slot 0 only: fresh random K/V per physical block
            c["buf"][i].copy_(torch.randn(NKV, BLOCK, 2 * HS, device=b.dev, generator=g).to(b.fp8))
        c["seq"].fill_(depth + QLEN)
        ref = reference(b, c, depth, nseq)
        b.stock(c, 0)
        torch.cuda.synchronize()
        base = c["out"].clone()
        e_stock = ((base.float() - ref).norm() / ref.norm()).item()
        line = [f"stock {e_stock:.4f}"]
        if b.RAD is not None:
            out = torch.empty_like(base)
            kw = b.stock_kwargs(c, 0)
            kw["out"] = out
            b.RAD.ENABLED = True
            n0 = b.RAD._state["calls"]
            b.RAD._state["wrapper"](**kw)
            torch.cuda.synchronize()
            assert b.RAD._state["calls"] == n0 + 1 and not b.RAD._state["failed"], "the module did not take the call"
            e = ((out.float() - ref).norm() / ref.norm()).item()
            d = ((out.float() - base.float()).abs().max()).item()
            worst = max(worst, e)
            line.append(f"MODULE {e:.4f} (max|d| {d:.3f}, segs {b.RAD.segments_for(nseq)})")
            if e > max(1.5 * e_stock, 0.02):
                print(f"nseq={nseq} d={depth}: " + "  ".join(line) + "  WORSE", flush=True)
                sys.exit(2)
        for name, cfg in cells.items():
            out = torch.empty_like(base)
            b.custom(c, 0, cfg, out=out)
            torch.cuda.synchronize()
            e = ((out.float() - ref).norm() / ref.norm()).item()
            d = ((out.float() - base.float()).abs().max()).item()
            worst = max(worst, e)
            ok = e <= max(1.5 * e_stock, 0.02)
            line.append(f"{name} {e:.4f} (max|d| {d:.3f}){'' if ok else ' WORSE'}")
            if not ok:
                print(f"nseq={nseq} d={depth}: " + "  ".join(line), flush=True)
                sys.exit(2)
        print(f"nseq={nseq} d={depth} (nbt={nbt}): rel err vs fp32  " + "  ".join(line), flush=True)
    print(f"check ok (worst custom rel err {worst:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="stock,replica,shift3d")
    ap.add_argument("--sweep", choices=("2d", "3d"))
    ap.add_argument("--with-stock", action="store_true")
    ap.add_argument("--table", action="store_true", help="stock, replica (must equal stock), shift3d, noshift3d")
    ap.add_argument("--nseqs", default=",".join(map(str, NSEQS)))
    ap.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--csv")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--causal", type=int, default=0, help="1 = causal mask (default 0: what the drafter runs)")
    ap.add_argument("--window", type=int, default=0, help="price another sliding window (default: the checkpoint's 2048)")
    ap.add_argument("--print", dest="print_csv")
    ap.add_argument("--rank")
    a = ap.parse_args()
    if a.print_csv:
        return print_table(a.print_csv, top=a.top)
    if a.rank:
        return rank(a.rank, top=a.top or 15)
    global CAUSAL
    CAUSAL = bool(a.causal)
    if a.window:
        set_window(a.window)
    if a.table:
        a.configs = "stock,replica,shift3d,module"
    if a.check:
        return cmd_check(a)
    return cmd_grid(a)


if __name__ == "__main__":
    main()
