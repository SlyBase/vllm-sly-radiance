#!/usr/bin/env python3
"""aiter unified_attention on the DFlash verify shapes: stock tables vs the radiance_attn_decode tune.

Production shape: head 256, 24 q / 4 kv heads (6 per kv head), fp8 q + fp8 KV, block_size 896, the KV
cache in the ROCM_AITER_UNIFIED_ATTN layout ([blocks, kv heads, block, 2 * head] with K and V packed in
the content dim, split into two transposed views), verify width = num_speculative_tokens + 1 = 8.

What is measured is what the engine runs: one CUDA graph per (config, sequences, width), captured with
max_seqlen_k = max_model_len (262144, the prod unit's --max-model-len -- vLLM captures FULL graphs
with that value, so NUM_SEGMENTS and the 2D/3D choice are fixed there) and replayed with seq_lens =
every depth of the sweep (--capture-at-depth captures one graph per depth instead, to show what the
capture-time value costs). Each graph holds R calls over R disjoint KV pools so the bytes come from
DRAM, not from the 64 MB infinity cache (a single 4k-token sequence is 8 MB). Time per call = replay
time / R, median of --reps; "us" covers the attention kernel and the reduce, "attn_us" (with
--split-reduce) the attention kernel alone.

  # inside a throw-away container from the runtime image, GPU exclusive, no model:
  python3 bench_decode_attn.py --table --csv /out/table.csv
  python3 bench_decode_attn.py --sweep kernel --nseqs 1,4,8 --depths 32768,98304 --warm 6 --csv /out/kernel.csv
  python3 bench_decode_attn.py --sweep splits --base wide32 --csv /out/splits.csv
  python3 bench_decode_attn.py --check           # tuned vs stock vs an fp32 torch reference
  python3 bench_decode_attn.py --prefill         # 2D prefill config candidates
  python3 bench_decode_attn.py --rank /out/kernel.csv --top 15   # best cell over all swept shapes

`radiance_attn_decode` must be importable (PYTHONPATH=<repo>/sly, or the site-packages copy) and must not
be switched off in the environment. Configs (--configs): `stock` = aiter's tables untouched, `tuned` =
the shipped rule in radiance_attn_decode, anything else a forced cell from CELLS or a literal
`name=field:value,...` (unnamed fields inherit from wide32). --warm N compiles a sweep's variants in N
parallel processes first (Triton's LLVM step is CPU-bound and the sweep is ~100 constexpr variants),
then times them in this one. Results append to --csv and finished cells are skipped on a re-run, so an
interrupted window resumes where it stopped.
"""
import argparse
import csv
import itertools
import os
import statistics
import subprocess
import sys
import time

NQ, NKV, HS, BLOCK = 24, 4, 256, 896     # Qwen3.8-27B full-attention layers, prod attention page
MAXLEN = 262144                          # --max-model-len of the prod unit = capture-time max_seqlen_k
DEPTHS = [4096, 8192, 16384, 32768, 65536, 98304, 131072]
NSEQS = [1, 2, 3, 4, 5, 6, 7, 8]
BYTES_PER_TOKEN = NKV * 2 * HS           # K + V, fp8, all kv heads: one pass over one sequence
POOL_BUDGET = 9 << 30                    # KV pool bytes per sequence count

# Forced cells. Fields: block_m tile warps stages waves segments (int or "pcu<N>", the machine-fill rule at
# N workgroups per CU) r_warps r_stages r_waves. stock3d is aiter's own gfx1201 3D table (what stock runs
# up to 6 sequences; from 7 on stock takes the 2D kernel), narrow16 / wide32 the numbers the retired
# select_3d_config hook shipped for a BLOCK_M 16 / BLOCK_M 64 launch.
CELLS = {
    "stock3d": dict(block_m=16, tile=64, warps=2, stages=2, waves=2, segments="pcu16", r_warps=2, r_stages=1, r_waves=2),
    "narrow16": dict(block_m=16, tile=16, warps=2, stages=1, waves=6, segments="pcu16", r_warps=8, r_stages=1, r_waves=2),
    "wide32": dict(block_m=64, tile=32, warps=4, stages=1, waves=6, segments="pcu16", r_warps=8, r_stages=1, r_waves=2),
}


def parse_cell(spec):
    if "=" not in spec:
        return spec, dict(CELLS[spec])
    name, _, body = spec.partition("=")
    cell = dict(CELLS["wide32"])
    for kv in body.split(","):
        k, _, v = kv.partition(":")
        cell[k] = v if k == "segments" and v.startswith("pcu") else int(v)
    return name, cell


def tag(cell):
    return (f"bm{cell['block_m']}_t{cell['tile']}_w{cell['warps']}_s{cell['stages']}_e{cell['waves']}"
            f"_g{cell['segments']}_r{cell['r_warps']}{cell['r_stages']}{cell['r_waves']}")


class Bench:
    def __init__(self, args):
        import torch
        import aiter.ops.triton.attention.unified_attention as UA
        import aiter.ops.triton.utils.unified_attention_utils as UU
        import radiance_attn_decode as RAD
        self.torch, self.UA, self.RAD = torch, UA, RAD
        assert UA.DEVICE_ARCH == "gfx1201", UA.DEVICE_ARCH
        assert RAD.install() and getattr(UU.get_unified_attention_config, "_radiance_decode_tune", False), \
            "radiance_attn_decode is not installed (RADIANCE_ATTN_DECODE_TUNE=0?): every column would be stock"
        self.dev = torch.device("cuda")
        self.fp8 = torch.float8_e4m3fn
        self.args = args
        # warm-up workers only compile: a small pool (one rotation slot) keeps N of them inside the card
        self.pool_budget = (512 << 20) if args.worker else POOL_BUDGET
        self.pools = {}
        self.bad = {}
        # one random 16-block tile is enough content for a timing pool (--check uses its own data)
        g = torch.Generator(device=self.dev).manual_seed(0)
        self.tile_data = torch.randn(16, NKV, BLOCK, 2 * HS, device=self.dev, generator=g).to(self.fp8)

    def pool(self, nseq):
        """R disjoint block pools for `nseq` sequences; only the current sequence count stays allocated."""
        torch = self.torch
        if nseq in self.pools:
            return self.pools[nseq]
        self.pools.clear()
        torch.cuda.empty_cache()
        nb_max = -(-max(DEPTHS) // BLOCK)
        per = nseq * nb_max * NKV * BLOCK * 2 * HS
        R = max(1, min(16, self.pool_budget // per))
        buf = torch.empty(R * nseq * nb_max, NKV, BLOCK, 2 * HS, dtype=self.fp8, device=self.dev)
        for i in range(0, buf.shape[0], 16):
            n = min(16, buf.shape[0] - i)
            buf[i:i + n].copy_(self.tile_data[:n])
        k, v = buf.transpose(1, 2).split(HS, dim=-1)          # [B, N, H, hs] views, as _split_kv_cache
        tables = [(r * nseq * nb_max + torch.arange(nseq * nb_max, device=self.dev, dtype=torch.int32))
                  .view(nseq, nb_max).contiguous() for r in range(R)]
        self.pools[nseq] = dict(buf=buf, k=k, v=v, tables=tables, R=R)
        return self.pools[nseq]

    def case(self, nseq, width):
        torch = self.torch
        p = self.pool(nseq)
        ntok = nseq * width
        return dict(
            p, nseq=nseq, width=width,
            q=torch.randn(ntok, NQ, HS, device=self.dev).to(self.fp8),
            out=torch.empty(ntok, NQ, HS, dtype=torch.bfloat16, device=self.dev),
            cu=torch.arange(nseq + 1, device=self.dev, dtype=torch.int32) * width,
            seq=torch.full((nseq,), 1, dtype=torch.int32, device=self.dev),
            scale=torch.tensor([1.0], device=self.dev))

    def call(self, c, r, skip_reduce=False, max_k=MAXLEN):
        return self.UA.unified_attention(
            q=c["q"], k=c["k"], v=c["v"], out=c["out"], cu_seqlens_q=c["cu"], max_seqlen_q=c["width"],
            seqused_k=c["seq"], max_seqlen_k=max_k, softmax_scale=HS ** -0.5, causal=True,
            window_size=(-1, -1), block_table=c["tables"][r], softcap=0,
            q_descale=c["scale"], k_descale=c["scale"], v_descale=c["scale"], skip_reduce=skip_reduce)

    def select(self, name, cell):
        """stock / tuned / forced cell -> radiance_attn_decode state."""
        self.RAD.ENABLED = name != "stock"
        self.RAD.FORCE = cell
        self.RAD.FORCE_2D = None

    def graph(self, c, skip_reduce=False, max_k=MAXLEN):
        torch = self.torch
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):                               # compiles every variant this launch needs
            for r in range(c["R"]):
                self.call(c, r, skip_reduce, max_k)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for r in range(c["R"]):
                self.call(c, r, skip_reduce, max_k)
        return g

    def time_graph(self, g, c, depth):
        torch = self.torch
        c["seq"].fill_(depth)
        g.replay()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        one = e0.elapsed_time(e1)                                # ms per replay
        iters = max(2, min(200, int(40.0 / max(one, 0.01))))
        reps = []
        for _ in range(self.args.reps):
            e0.record()
            for _ in range(iters):
                g.replay()
            e1.record()
            torch.cuda.synchronize()
            reps.append(e0.elapsed_time(e1) * 1000.0 / iters / c["R"])
        return statistics.median(reps)

    def run(self, name, cell, nseq, width, depths):
        """[(depth, us, attn_us)], or None when the cell does not compile / launch."""
        if name in self.bad:
            return None
        c = self.case(nseq, width)
        self.select(name, cell)
        a = self.args
        rows = []
        try:
            shared = None if a.capture_at_depth else self.graph(c)
            shared_a = self.graph(c, True) if (a.split_reduce and not a.capture_at_depth) else None
            for d in depths:
                g = shared if shared is not None else self.graph(c, max_k=d)
                us = self.time_graph(g, c, d)
                ga = shared_a
                if ga is None and a.split_reduce:
                    ga = self.graph(c, True, max_k=d)
                aus = self.time_graph(ga, c, d) if ga is not None else float("nan")
                rows.append((d, us, aus))
                if shared is None:
                    del g, ga                          # per-depth graphs hold their split buffers: free them
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
                done.add((r["config"], int(r["nseq"]), int(r["width"]), int(r["depth"])))
    return done


def sweep_cells(kind, base):
    if kind == "kernel":                      # BLOCK_M x TILE x warps x stages x waves at a machine-fill split
        out = {}
        for bm, tile, w, s, e in itertools.product((16, 32, 64), (16, 32, 64), (2, 4, 8), (1, 2), (2, 6)):
            if bm == 16 and w == 8:
                continue
            cell = dict(CELLS["wide32"], block_m=bm, tile=tile, warps=w, stages=s, waves=e, segments="pcu16")
            out[f"k_{tag(cell)}"] = cell
        return out
    name, b = parse_cell(base)
    if kind == "splits":
        return {f"{name}_g{g}": dict(b, segments=g) for g in (4, 8, 16, 32, 64, 128, 256)}
    if kind == "reduce":
        return {f"{name}_r{w}{s}{e}": dict(b, r_warps=w, r_stages=s, r_waves=e)
                for w, s, e in itertools.product((1, 2, 4, 8), (1, 2), (2,))}
    raise SystemExit(f"unknown sweep {kind}")


def print_table(path, base="stock", top=0):
    """Per (nseq, width) one block: rows = configs, columns = depth, cells = us (speedup vs base)."""
    data = {}
    for r in csv.DictReader(open(path, newline="")):
        data.setdefault((int(r["nseq"]), int(r["width"])), {}).setdefault(r["config"], {})[int(r["depth"])] = float(r["us"])
    for (nseq, width), cfgs in sorted(data.items()):
        depths = sorted({d for c in cfgs.values() for d in c})
        print(f"\nnseq={nseq} width={width}   us/call (speedup vs {base})")
        print(f"{'config':40s} " + " ".join(f"{d // 1024}k".rjust(13) for d in depths))
        ref = cfgs.get(base, {})
        rows = sorted(cfgs.items(), key=lambda kv: (kv[0] not in (base, "tuned"), sum(kv[1].values()) / len(kv[1])))
        keep = [kv for kv in rows if kv[0] in (base, "tuned")] + [kv for kv in rows if kv[0] not in (base, "tuned")][:top or None]
        for name, c in keep:
            cells = []
            for d in depths:
                if d not in c:
                    cells.append(f"{'-':>13s}")
                elif d in ref and name != base:
                    cells.append(f"{c[d]:7.0f} {ref[d] / c[d]:4.2f}x")
                else:
                    cells.append(f"{c[d]:7.0f}{'':6s}")
            print(f"{name[:40]:40s} " + " ".join(cells))


def rank(path, top=15, weights=None):
    """Sweep ranking: per shape (nseq, width, depth) every config's time relative to the best config there;
    a config's score is the geometric mean of those ratios over the shapes it covers (1.00 = best
    everywhere), `worst` its single worst ratio. Only configs present on every shape are ranked."""
    import math
    per = {}
    for r in csv.DictReader(open(path, newline="")):
        per.setdefault((int(r["nseq"]), int(r["width"]), int(r["depth"])), {})[r["config"]] = float(r["us"])
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
        configs = list(sweep_cells(a.sweep, a.base).items())
        if a.with_stock:
            configs.insert(0, ("stock", None))
    else:
        sep = ";" if ";" in a.configs else ","          # literal cells carry commas: separate those with ;
        configs = [(s, None) if s in ("stock", "tuned") else parse_cell(s) for s in a.configs.split(sep)]
    nseqs = [int(x) for x in a.nseqs.split(",")]
    widths = [int(x) for x in a.widths.split(",")]
    depths = [int(x) for x in a.depths.split(",")]

    if a.warm and not a.worker:
        t0 = time.time()
        procs = [subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", f"{i}/{a.warm}"]
                                  + strip_warm(sys.argv[1:]), stdout=subprocess.DEVNULL) for i in range(a.warm)]
        for p in procs:
            p.wait()
        print(f"warm-up: {a.warm} workers done in {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    b = Bench(a)
    if a.worker:                              # compile only: first + last sequence count, one depth, no output
        i, n = (int(x) for x in a.worker.split("/"))
        configs, nseqs, widths, depths = configs[i::n], sorted({nseqs[0], nseqs[-1]}), widths[:1], depths[:1]
        a.reps, a.csv = 1, None
    done = load_done(a.csv)
    fh = writer = None
    if a.csv:
        new = not os.path.exists(a.csv)
        fh = open(a.csv, "a", newline="")
        writer = csv.writer(fh)
        if new:
            writer.writerow(["config", "nseq", "width", "depth", "us", "attn_us", "GBps_1pass"])
    for nseq in nseqs:
        for width in widths:
            for name, cell in configs:
                todo = [d for d in depths if (name, nseq, width, d) not in done]
                if not todo:
                    continue
                t0 = time.time()
                rows = b.run(name, cell, nseq, width, todo)
                if rows is None:
                    continue
                if writer:
                    for d, us, aus in rows:
                        writer.writerow([name, nseq, width, d, f"{us:.2f}", f"{aus:.2f}",
                                         f"{nseq * d * BYTES_PER_TOKEN / (us * 1e3):.1f}"])
                    fh.flush()
                if name == "tuned":
                    pl = b.RAD._plan(nseq * width, nseq, width, MAXLEN, NQ // NKV, NKV, 32)
                    print(f"  shipped plan nseq={nseq} w={width}: " + " ".join(f"{k}={v}" for k, v in pl.items()))
                print(f"nseq={nseq} w={width} {name[:44]:44s} " +
                      " ".join(f"{d // 1024}k:{us:.0f}" for d, us, _ in rows) + f"   ({time.time() - t0:.0f}s)",
                      flush=True)
    if a.csv:
        fh.close()
        print_table(a.csv, top=a.top)


def reference(b, c, depth, nseq, width):
    """fp32 attention for the verify batch from the same (dequantised) fp8 data, causal, per sequence."""
    torch = b.torch
    outs = []
    nb = -(-depth // BLOCK)
    for s in range(nseq):
        blocks = c["tables"][0][s, :nb].long()
        K = c["k"][blocks].reshape(nb * BLOCK, NKV, HS)[:depth].float()
        V = c["v"][blocks].reshape(nb * BLOCK, NKV, HS)[:depth].float()
        q = c["q"][s * width:(s + 1) * width].float()             # [w, 24, hs]
        o = torch.empty_like(q)
        pos = torch.arange(depth, device=q.device)[None, :]
        lim = (depth - width + torch.arange(width, device=q.device))[:, None]
        for h in range(NQ):
            kvh = h // (NQ // NKV)
            sc = (q[:, h] @ K[:, kvh].T) * HS ** -0.5              # [w, depth]
            sc = sc.masked_fill(pos > lim, float("-inf"))
            o[:, h] = torch.softmax(sc, -1) @ V[:, kvh]
        outs.append(o)
    return torch.cat(outs)


def cmd_check(a):
    """Tuned and stock against an fp32 reference on random per-block data. The kernel rounds P to fp8
    before the P.V dot, so ~1e-2 relative error is its own floor; the tune must not be worse than stock."""
    b = Bench(a)
    torch = b.torch
    worst = 0.0
    for nseq, width, depth in ((1, 8, 4096), (3, 8, 9000), (8, 8, 20000), (2, 6, 5000), (1, 9, 33000), (7, 8, 12000),
                           (2, 5, 7000), (4, 3, 6000)):
        c = b.case(nseq, width)
        g = torch.Generator(device=b.dev).manual_seed(1)
        nb = -(-depth // BLOCK)
        for i in c["tables"][0][:, :nb].flatten().tolist():
            c["buf"][i].copy_(torch.randn(NKV, BLOCK, 2 * HS, device=b.dev, generator=g).to(b.fp8))
        c["seq"].fill_(depth)
        ref = reference(b, c, depth, nseq, width)
        res = {}
        for name in ("stock", "tuned"):
            b.select(name, None)
            b.call(c, 0)
            torch.cuda.synchronize()
            res[name] = ((c["out"].float() - ref).norm() / ref.norm()).item()
        ok = res["tuned"] <= max(1.5 * res["stock"], 0.05)
        worst = max(worst, res["tuned"])
        print(f"nseq={nseq} w={width} d={depth}: rel err vs fp32 ref  stock {res['stock']:.4f}  "
              f"tuned {res['tuned']:.4f}  {'ok' if ok else 'WORSE'}", flush=True)
        if not ok:
            sys.exit(2)
    print(f"check ok (worst tuned rel err {worst:.4f})")


def cmd_prefill(a):
    """2D prefill/extend at 2048 query tokens: aiter's Q_GEQ_256 entry vs the fp8 large-prefill tune the
    retired hook carried (TILE 16 waves 1) and neighbours. Eager, as the piecewise steps run it."""
    b = Bench(a)
    torch = b.torch
    cands = {"stock": None,
             "t16_e1": dict(TILE_SIZE=16, waves_per_eu=1),
             "t32_e1": dict(TILE_SIZE=32, waves_per_eu=1),
             "t16_e6": dict(TILE_SIZE=16),
             "t16_w2_e1": dict(TILE_SIZE=16, waves_per_eu=1, num_warps=2)}
    qlen = a.qlen
    q = torch.randn(qlen, NQ, HS, device=b.dev).to(b.fp8)
    out = torch.empty(qlen, NQ, HS, dtype=torch.bfloat16, device=b.dev)
    one = torch.tensor([1.0], device=b.dev)
    for past in (0, 32768, 98304):
        depth = past + qlen
        nb = -(-depth // BLOCK)
        buf = torch.empty(nb, NKV, BLOCK, 2 * HS, dtype=b.fp8, device=b.dev)
        for i in range(0, nb, 16):
            n = min(16, nb - i)
            buf[i:i + n].copy_(b.tile_data[:n])
        k, v = buf.transpose(1, 2).split(HS, dim=-1)
        bt = torch.arange(nb, device=b.dev, dtype=torch.int32).view(1, nb)
        cu = torch.tensor([0, qlen], device=b.dev, dtype=torch.int32)
        seq = torch.tensor([depth], device=b.dev, dtype=torch.int32)
        base = None
        for name, over in cands.items():
            b.RAD.ENABLED, b.RAD.FORCE, b.RAD.FORCE_2D = True, None, over
            times = []
            try:
                for _ in range(6):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    b.UA.unified_attention(q=q, k=k, v=v, out=out, cu_seqlens_q=cu, max_seqlen_q=qlen, seqused_k=seq,
                                           max_seqlen_k=depth, softmax_scale=HS ** -0.5, causal=True,
                                           window_size=(-1, -1), block_table=bt, softcap=0, q_descale=one,
                                           k_descale=one, v_descale=one)
                    e1.record()
                    torch.cuda.synchronize()
                    times.append(e0.elapsed_time(e1))
            except Exception as e:
                print(f"past={past} {name}: FAIL {type(e).__name__}: {str(e)[:100]}")
                continue
            ms = statistics.median(times[2:])
            base = base or ms
            print(f"past={past:6d} qlen={qlen} {name:10s} {ms:8.3f} ms  ({base / ms:4.2f}x vs stock)", flush=True)
        del buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="stock,tuned")
    ap.add_argument("--sweep", choices=("kernel", "splits", "reduce"))
    ap.add_argument("--base", default="wide32", help="cell the splits/reduce sweep varies")
    ap.add_argument("--with-stock", action="store_true", help="add the stock row to a sweep")
    ap.add_argument("--table", action="store_true", help="before/after grid: stock, tuned, narrow16, stock3d")
    ap.add_argument("--nseqs", default=",".join(map(str, NSEQS)))
    ap.add_argument("--widths", default="8")
    ap.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--split-reduce", action="store_true", help="also time the attention kernel alone (skip_reduce)")
    ap.add_argument("--capture-at-depth", action="store_true", help="capture each depth's graph with max_seqlen_k = depth")
    ap.add_argument("--csv")
    ap.add_argument("--top", type=int, default=0, help="print only the best N sweep rows per shape")
    ap.add_argument("--warm", type=int, default=0, help="compile variants in N parallel processes first")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--prefill", action="store_true")
    ap.add_argument("--qlen", type=int, default=2048)
    ap.add_argument("--print", dest="print_csv", help="print the table of an existing csv and exit")
    ap.add_argument("--rank", help="rank the configs of an existing sweep csv and exit")
    a = ap.parse_args()
    if a.print_csv:
        return print_table(a.print_csv, top=a.top)
    if a.rank:
        return rank(a.rank, top=a.top or 15)
    if a.table:
        a.configs = "stock,tuned,narrow16,stock3d"
    if a.check:
        return cmd_check(a)
    if a.prefill:
        return cmd_prefill(a)
    return cmd_grid(a)


if __name__ == "__main__":
    main()
