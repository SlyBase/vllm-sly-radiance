"""hipEvent probe of the ~123 us idle that follows every kernel longer than ~1.1 ms (no radiance code).

Sequence per iteration: one long elementwise kernel (torch.add over BYTES) + one tiny kernel. Variants:
  eager        plain stream launches
  graph        the same N iterations captured once in a CUDA/HIP graph and replayed
  eager2s      long kernel on stream A, tiny kernels on stream B (event-joined), i.e. the next dependent
               launch does not sit right behind the long kernel in the same queue
  split<k>     the long add split in k pieces (each < 1.1 ms when k >= 2)
Prints us per iteration (median of REPS timed blocks of N iterations) and the excess over the summed
kernel time measured from the 'long only' variant.
"""
import os, statistics

import torch

N = int(os.environ.get("N", "20"))
REPS = int(os.environ.get("REPS", "7"))
dev = "cuda"


def timeit(fn):
    fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e) * 1e3 / N)
    return statistics.median(out)


for mb in (128, 256, 512):  # bf16 tensors: ~0.66 / 1.3 / 2.6 ms adds at 210 W
    n = mb * (1 << 20) // 2
    a = torch.randn(n, device=dev, dtype=torch.bfloat16)
    b = torch.randn(n, device=dev, dtype=torch.bfloat16)
    c = torch.empty_like(a)
    t = torch.zeros(16, device=dev)

    def eager():
        for _ in range(N):
            torch.add(a, b, out=c)
            t.add_(1)

    def long_only_split(k):
        ch = n // k

        def f():
            for _ in range(N):
                for j in range(k):
                    torch.add(a[j * ch:(j + 1) * ch], b[j * ch:(j + 1) * ch], out=c[j * ch:(j + 1) * ch])
                t.add_(1)
        return f

    sa, sb = torch.cuda.Stream(), torch.cuda.Stream()

    def eager2s():
        cur = torch.cuda.current_stream()
        sa.wait_stream(cur)
        for _ in range(N):
            with torch.cuda.stream(sa):
                torch.add(a, b, out=c)
                ev = torch.cuda.Event()
                ev.record(sa)
            sb.wait_event(ev)
            with torch.cuda.stream(sb):
                t.add_(1)
            sa.wait_stream(sb)
        cur.wait_stream(sa)

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        eager()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        eager()

    res = {"eager": timeit(eager), "graph": timeit(g.replay), "eager2s": timeit(eager2s)}
    for k in (2, 4):
        res[f"split{k}"] = timeit(long_only_split(k))
    # kernel time without gaps: profile one eager block
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        eager()
        torch.cuda.synchronize()
    ks = [e for e in p.events() if e.device_type.name == "CUDA"]
    busy = sum(e.device_time for e in ks) / N if hasattr(ks[0], "device_time") else float("nan")
    print(f"add {mb} MB: kernel busy/iter {busy:.1f} us | " +
          " | ".join(f"{k} {v:.1f} us (+{v - busy:.1f})" for k, v in res.items()), flush=True)
