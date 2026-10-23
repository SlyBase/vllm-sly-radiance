#!/usr/bin/env python3
"""GPU gate for RADIANCE_KV4H_PIPE (the prefill schedule mask of radiance_kv4h.hip). Not run in CI.

Three steps, inside a container of the serving image (PYTHONPATH must find sly/ and sly/kv4/):

  1. asm   hipcc -S of every (pipe, tile) instance: VGPR, spill, LDS, FLAT loads, barriers per loop.
           The image's compiler decides the spill, not the one the knob was written against.
  2. ident every pipe x tile against pipe 0 at the same tile, torch.equal on the bf16 output, on the
           production geometry (24 q / 4 kv heads, head 256, block 1664) at chunk 4096 and contexts
           0 / 16k / 32k / 64k, plus mixed / window / non-causal / head-128 / decode shapes.
  3. abba  kernel time per prefill chunk (CHUNK 4096) at 2k / 16k / 32k / 64k context, arms ordered
           A B B A per round, median per arm; A = the shipped launch (pipe 0, the tune table's tile).

  python3 sly/kv4/gate_kv4h_pipe.py                       # all three, default candidates
  python3 sly/kv4/gate_kv4h_pipe.py --steps abba --cands 3:32,5:16,1:32,0:32 --rounds 12
  exit 0 = every instance bit-identical (step 2); step 3 only reports.
"""
import argparse
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

NQ, NKV, HEAD, BS = 24, 4, 256, 1664
dev = "cuda"


def asm_stats(pipes, tiles):
    """hipcc -S of the .hip per pipe: per head-256 tile instance (vgpr, spill, lds, flat loads, barriers)."""
    src = HERE / "radiance_kv4h.hip"
    inc = subprocess.check_output([sys.executable, "-m", "pybind11", "--includes"], text=True).split()
    for pipe in pipes:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "k.s"
            subprocess.run(["hipcc", "-O3", "-fPIC", "-std=c++20", "--offload-arch=gfx1201", "--offload-device-only", "-S",
                            *inc, "-DKV4H_NRT=4", "-DKV4H_QP=1", "-DKV4H_NS=1", "-DKV4H_PREF=1", f"-DKV4H_PIPE={pipe}",
                            str(src), "-o", str(out)], check=True)
            t = out.read_text()
        for tile in tiles:
            name = f"_Z14kv4h_attentionILi256ELi{tile // 16}E"
            i = t.index("\n" + name) + 1
            body = t[i:t.index("s_endpgm", i)]
            meta = t[t.index(".name:           " + name):]
            g = lambda k: int(re.search(rf"\.{k}:\s+(\d+)", meta).group(1))  # noqa: E731
            print(f"  asm pipe={pipe} tile={tile}: vgpr {g('vgpr_count')} spill {g('vgpr_spill_count')} "
                  f"lds {g('group_segment_fixed_size')} flat_load {body.count('flat_load')} "
                  f"global_load {body.count('global_load')} barriers {body.count('s_barrier_wait')}")


class Case:
    """One attention launch on a random 4-bit cache: seq_lens / q_lens per sequence."""

    def __init__(self, seq_lens, q_lens, bs=BS, head=HEAD, nq=NQ, causal=True, window=0):
        import torch
        import radiance_kv4h_format as H
        from radiance_kv4h_store import kv4h_store
        L = H.layout(head)
        nb = [-(-s // bs) for s in seq_lens]
        self.cache = torch.zeros(sum(nb) + 1, NKV, bs, 2 * L.TPT, dtype=torch.uint8, device=dev)
        self.bt = torch.zeros(len(seq_lens), max(nb), dtype=torch.int32, device=dev)
        b0 = 1
        for i, s in enumerate(seq_lens):
            ids = torch.arange(b0, b0 + nb[i], device=dev)
            self.bt[i, :nb[i]] = ids.to(torch.int32)
            b0 += nb[i]
            # random K/V for the whole sequence (bytes only matter for identity / timing)
            for c0 in range(0, s, 16384):
                n = min(16384, s - c0)
                k = torch.randn(n, NKV, head, device=dev).to(torch.bfloat16)
                pos = torch.arange(c0, c0 + n, device=dev)
                kv4h_store(k, k * 0.5, self.cache, (ids[pos // bs] * bs + pos % bs).to(torch.int64))
        self.q = torch.randn(sum(q_lens), nq, head, device=dev).to(torch.bfloat16)
        self.cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0).tolist()), dtype=torch.int32, device=dev)
        self.seqused = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
        self.max_q, self.max_k = max(q_lens), max(seq_lens)
        self.scale = 1.0 / math.sqrt(head)
        self.causal, self.window = causal, window
        self.out = torch.empty_like(self.q)
        self.fin = torch.empty_like(self.q)

    def run(self, **kn):
        from radiance_kv4h_kernel import kv4h_attention
        kv4h_attention(self.q, self.cache, self.out, self.cu, self.max_q, self.seqused, self.max_k, self.scale, self.bt,
                       causal=self.causal, window=self.window, output=self.fin, **kn)
        return self.fin


def identity(pipes, tiles):
    import torch
    cases = [("chunk4096 c0", dict(seq_lens=[4096], q_lens=[4096]))]
    for ctx in (16384, 32768, 65536):
        cases.append((f"chunk4096 c{ctx // 1024}k", dict(seq_lens=[ctx + 4096], q_lens=[4096])))
    cases += [
        ("mixed prefill+verify+decode", dict(seq_lens=[2048, 5000, 7000], q_lens=[2048, 8, 1], bs=880)),
        ("prefill 2x300 bs32", dict(seq_lens=[300, 300], q_lens=[300, 300], bs=32)),
        ("window 512 prefill", dict(seq_lens=[4096], q_lens=[1000], bs=880, window=512)),
        ("noncausal prefill 300", dict(seq_lens=[300], q_lens=[300], bs=32, causal=False)),
        ("h128 drafter nc w2048", dict(seq_lens=[8200, 1000], q_lens=[8, 8], bs=880, head=128, nq=16, causal=False,
                                       window=2048)),
        ("decode x4 16k", dict(seq_lens=[16384] * 4, q_lens=[1] * 4)),
    ]
    bad = 0
    for label, kw in cases:
        c = Case(**kw)
        for tile in tiles:
            ref = c.run(pipe=0, tile=tile, num_segments=None).clone()
            for pipe in pipes:
                if pipe == 0:
                    continue
                got = c.run(pipe=pipe, tile=tile).clone()
                ok = torch.equal(got, ref)
                bad += not ok
                print(f"  ident {label:28s} tile={tile} pipe={pipe}: {'bit-identical' if ok else 'DIFFERS max|d| %.3g' % float((got.float() - ref.float()).abs().max())}")
        del c
        torch.cuda.empty_cache()
    return bad


def abba(cands, rounds, chunk):
    import torch
    for ctx in (2048, 16384, 32768, 65536):
        seq = ctx if ctx == chunk else ctx + chunk
        c = Case([max(seq, chunk)], [chunk])
        arms = [("A tuned", dict(pipe=0))] + [(f"pipe={p} tile={t}", dict(pipe=p, tile=t)) for p, t in cands]
        times = {a: [] for a, _ in arms}

        def t1(kn):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            c.run(**kn)
            e.record()
            torch.cuda.synchronize()
            return s.elapsed_time(e) * 1000.0

        for _, kn in arms:            # build + warm every arm first
            for _ in range(3):
                t1(kn)
        for _ in range(rounds):
            for (a, kn) in arms[1:]:
                times["A tuned"].append(t1(arms[0][1]))
                times[a].append(t1(kn))
                times[a].append(t1(kn))
                times["A tuned"].append(t1(arms[0][1]))
        med = {a: sorted(v)[len(v) // 2] for a, v in times.items()}
        base = med["A tuned"]
        print(f"  abba chunk {chunk} ctx {ctx // 1024}k: " + "   ".join(
            f"{a} {us:8.1f} us ({100 * (us / base - 1):+.1f} %)" for a, us in med.items()))
        del c
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="asm,ident,abba")
    ap.add_argument("--pipes", default="0,1,2,3,4,5")
    ap.add_argument("--tiles", default="16,32,64")
    ap.add_argument("--cands", default="1:64,1:32,3:32,3:16,5:16,0:32,0:16", help="pipe:tile arms of the ABBA step")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()
    steps = a.steps.split(",")
    pipes = [int(x) for x in a.pipes.split(",")]
    tiles = [int(x) for x in a.tiles.split(",")]
    os.environ.setdefault("RADIANCE_KV4H_BUILD_DIR", "/tmp/radiance_kv4h_gate")
    if "asm" in steps:
        asm_stats(pipes, tiles)
    bad = 0
    if "ident" in steps:
        bad = identity(pipes, tiles)
        print(f"identity: {'PASS' if not bad else f'FAIL ({bad} instances differ)'}")
    if "abba" in steps:
        abba([tuple(int(y) for y in x.split(":")) for x in a.cands.split(",")], a.rounds, a.chunk)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
