"""In-process A/B of the norm/act-quant producers on the PRODUCTION engine (one process, one compile).

Same harness as ci/prefill_torch_profile.py (production argv, VLLM_ENABLE_V1_MULTIPROCESSING=0). The arms
are switched between requests with the extension's set_nq_reg() hook (RADIANCE_FUSED_NQ_REG), so both
arms share clocks, compile cache and memory layout. Needs a radiance_mxfp4_fp8.so that has set_nq_reg.

Env: GEN=1 (greedy 128-token decode smoke, GENSHA line), ARMS (comma list, one `nq_reg:msplit_rows` per arm digit, default 0:0,1:0), OUT, LENS (default 1514,2048,15800), ORDER (arm per measured request, default 01100110 = ABBA ABBA,
used for the plain and for the profiled requests), NPROF (profiled requests per length, default 4, taken
from the head of ORDER), CHECK=1 (before timing, over CHECK_LENS (default 1514,2048,3520 = chunks 1514 / 880+1168 / 2x1760): every eager producer call with M > 64 is relaunched on
its real inputs into private buffers with REG=0 and REG=1 and the q / scale / residual bytes compared ->
`NQCHECK` lines with sha256 of the stock bytes; the inputs of the first call per (kernel, M) in DUMP_M
are saved to OUT/dump_<kernel>_<M>.pt for the standalone bench).
"""
import gzip, hashlib, json, os, random, shutil, sys, time, uuid

import torch
from torch.profiler import ProfilerActivity, profile

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.inputs import TokensPrompt
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.llm_engine import LLMEngine

OUT = os.environ.get("OUT", "/w/out")
LENS = [int(x) for x in os.environ.get("LENS", "1514,2048,15800").split(",")]
ORDER = [int(c) for c in os.environ.get("ORDER", "01100110")]
# arm digit -> (RADIANCE_FUSED_NQ_REG, RADIANCE_MXFP4_AT_MSPLIT rows); default arm 0 = stock, 1 = NQ_REG
ARMS = [tuple(int(v) for v in x.split(":")) for x in os.environ.get("ARMS", "0:0,1:0").split(",")]
NPROF = int(os.environ.get("NPROF", "4"))
CHECK = os.environ.get("CHECK", "0") == "1"
CHECK_LENS = [int(x) for x in os.environ.get("CHECK_LENS", "1514,2048,3520").split(",")]
DUMP_M = {int(x) for x in os.environ.get("DUMP_M", "880,1514,1760").split(",") if x}
os.makedirs(OUT, exist_ok=True)
args = make_arg_parser(FlexibleArgumentParser()).parse_args(sys.argv[1:])
engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(args))

import radiance_fused_norm as fn  # noqa: E402  (the module object the engine uses)

ext = fn._ext
assert ext is not None and hasattr(ext, "set_nq_reg"), "mount the radiance_mxfp4_fp8.so with set_nq_reg"


def set_arm(arm):
    ext.set_nq_reg(ARMS[arm][0])
    if hasattr(ext, "set_at_msplit"):  # experimental M-split build only
        ext.set_at_msplit(ARMS[arm][1])


set_arm(0)
stats, dumped = {}, set()
MS_CHECK = max(r for _, r in ARMS)


class _Dev:
    """Zero-copy tensor view of a raw device pointer (__cuda_array_interface__)."""

    def __init__(self, ptr, shape, typestr):
        self.__cuda_array_interface__ = {"data": (ptr, False), "shape": shape, "typestr": typestr,
                                         "version": 2}


def view(ptr, n, typestr="|u1"):
    return torch.as_tensor(_Dev(ptr, (n,), typestr), device="cuda")


def compare(name, M, launch, sizes, inputs):
    """launch(out_ptrs) writes the outputs; sizes = byte counts of the outputs."""
    if torch.cuda.is_current_stream_capturing():
        return
    outs = []
    for reg in (0, 1):
        bufs = [torch.zeros(n, dtype=torch.uint8, device="cuda") for n in sizes]  # tiled pad rows stay 0
        ext.set_nq_reg(reg)
        launch([b.data_ptr() for b in bufs])
        outs.append(bufs)
    ext.set_nq_reg(0)
    torch.cuda.synchronize()
    st = stats.setdefault((name, M), {"calls": 0, "diff": 0, "sha": hashlib.sha256()})
    st["calls"] += 1
    st["diff"] += int(any(not torch.equal(a, b) for a, b in zip(*outs)))
    for a in outs[0]:
        st["sha"].update(a.cpu().numpy().tobytes())
    if M in DUMP_M and (name, M) not in dumped:
        dumped.add((name, M))
        torch.save({k: view(p, n).clone().cpu() for k, (p, n) in inputs.items()}, f"{OUT}/dump_{name}_{M}.pt")


class Proxy:
    def __getattr__(self, k):
        return getattr(ext, k)

    @staticmethod
    def launch_silu_mul_quant(gu, q, scale, M, N, stream, tiled=0):
        ext.launch_silu_mul_quant(gu=gu, q=q, scale=scale, M=M, N=N, stream=stream, tiled=tiled)
        if M > 64:
            qn = ((M + 15) // 16 * 16 if tiled else M) * N
            compare("silu_mul_quant", M,
                    lambda o: ext.launch_silu_mul_quant(gu=gu, q=o[0], scale=o[1], M=M, N=N, stream=stream,
                                                        tiled=tiled),
                    [qn, 4 * M], {"gu": (gu, M * 2 * N * 2)})

    @staticmethod
    def launch_add_rms_quant(y, res, w, q, scale, res_out, M, K, eps, stream, tiled=0):
        ext.launch_add_rms_quant(y, res, w, q, scale, res_out, M, K, eps, stream, tiled)
        if M > 64:
            qn = ((M + 15) // 16 * 16 if tiled else M) * K
            compare("add_rms_quant", M,
                    lambda o: ext.launch_add_rms_quant(y, res, w, o[0], o[1], o[2], M, K, eps, stream, tiled),
                    [qn, 4 * M, 2 * M * K], {"y": (y, 2 * M * K), "res": (res, 2 * M * K), "w": (w, 2 * K)})

    @staticmethod
    def launch_gdn_norm_quant(x, z, zs, w, q, scale, M, N, eps, stream, tiled=0):
        ext.launch_gdn_norm_quant(x, z, zs, w, q, scale, M, N, eps, stream, tiled)
        if M > 64:
            qn = ((M + 15) // 16 * 16 if tiled else M) * N
            compare("gdn_norm_quant", M,
                    lambda o: ext.launch_gdn_norm_quant(x, z, zs, w, o[0], o[1], M, N, eps, stream, tiled),
                    [qn, 4 * M], {"x": (x, 2 * M * N), "z": (z, 2 * ((M - 1) * zs + N)), "w": (w, 256)})


import radiance_mxfp4 as rmx  # noqa: E402


class GemmProxy:
    """Relaunch every A-tiled single-partition GEMM with M > 64 unsplit and split (MS_CHECK rows), compare C."""

    def __getattr__(self, k):
        return getattr(ext, k)

    @staticmethod
    def launch_at(a, w, ws, wref, as_, c, M, N, K, stream):
        ext.launch_at(a, w, ws, wref, as_, c, M, N, K, stream)
        if M > 64 and not torch.cuda.is_current_stream_capturing():
            outs = []
            for rows in (0, MS_CHECK):
                buf = torch.zeros(M * N * 2, dtype=torch.uint8, device="cuda")
                ext.set_at_msplit(rows)
                ext.launch_at(a, w, ws, wref, as_, buf.data_ptr(), M, N, K, stream)
                outs.append(buf)
            ext.set_at_msplit(0)
            torch.cuda.synchronize()
            st = stats.setdefault((f"gemm_at N={N} K={K}", M), {"calls": 0, "diff": 0, "sha": hashlib.sha256()})
            st["calls"] += 1
            st["diff"] += int(not torch.equal(outs[0], outs[1]))
            st["sha"].update(hashlib.sha256(outs[0].cpu().numpy().tobytes()).digest())


rng = random.Random(1234)
sp = SamplingParams(max_tokens=1, temperature=0)


def one(n):
    ids = [rng.randrange(1000, 120000) for _ in range(n)]
    t = time.perf_counter()
    engine.add_request(uuid.uuid4().hex, TokensPrompt(prompt_token_ids=ids), sp)
    steps = 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t, steps


if CHECK:
    fn._ext = Proxy()
    if MS_CHECK:
        rmx._ext = GemmProxy()
    for n in CHECK_LENS:
        one(n)
    fn._ext = ext
    rmx._ext = ext
    for (name, M), st in sorted(stats.items()):
        print(f"NQCHECK {name} M={M} calls={st['calls']} diff={st['diff']} sha={st['sha'].hexdigest()[:16]}",
              flush=True)

if os.environ.get("GEN", "0") == "1":
    # Decode smoke: greedy generation on a fixed prompt with the arms as captured at startup (CUDA graphs are
    # captured with the env value of RADIANCE_FUSED_NQ_REG; set_nq_reg does not change them). Compare GENSHA
    # between two processes (env on / off).
    g = random.Random(7)
    ids = [g.randrange(1000, 120000) for _ in range(600)]
    engine.add_request("gen", TokensPrompt(prompt_token_ids=ids), SamplingParams(max_tokens=128, temperature=0))
    toks = []
    while engine.has_unfinished_requests():
        for o in engine.step():
            if o.finished:
                toks = list(o.outputs[0].token_ids)
    print(f"GENSHA {hashlib.sha256(str(toks).encode()).hexdigest()[:16]} n={len(toks)} head={toks[:8]}", flush=True)

res = []
for n in LENS:
    one(n)  # warmup
    for i, arm in enumerate(ORDER):
        set_arm(arm)
        dt, st = one(n)
        res.append(["plain", n, i, arm, dt, st])
        print(f"SPLIT plain {n} arm={arm} {dt * 1e3:.1f} ms steps={st}", flush=True)
    for i, arm in enumerate(ORDER[:NPROF]):
        set_arm(arm)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False,
                     with_stack=False) as p:
            dt, st = one(n)
            time.sleep(1.0)  # let the activity buffers drain before the profiler stops (traces lost kernels)
        res.append(["prof", n, i, arm, dt, st])
        print(f"SPLIT prof  {n} arm={arm} {dt * 1e3:.1f} ms steps={st}", flush=True)
        tmp = f"/tmp/t_{n}_{i}.json"
        p.export_chrome_trace(tmp)
        with open(tmp, "rb") as f, gzip.open(f"{OUT}/trace_{n}_{i}_arm{arm}.json.gz", "wb") as g:
            shutil.copyfileobj(f, g)
        os.remove(tmp)
set_arm(0)
json.dump(res, open(f"{OUT}/requests.json", "w"))
print("SPLIT done", flush=True)
os._exit(0)  # skip engine teardown (in-process engine shutdown can hang on ROCm)
