"""Profile single prefill requests in-process with torch.profiler, on the PRODUCTION engine config.

Why not the server: rocprofv3 around `vllm serve` hangs in queue_interposition, and vLLM's /start_profile
records no GPU kernels on ROCm because the EngineCore is a spawned child (kineto/roctracer in a spawned
process after the parent touched the GPU: 0 kernel events, measured). VLLM_ENABLE_V1_MULTIPROCESSING=0
keeps the engine in this process, where kineto sees every kernel. Same argv, env and code path as the
server otherwise (cudagraph_mode FULL_DECODE_ONLY: prefill never runs as a CUDA graph in production either).

argv = the `vllm serve` arguments of the production unit. Env: OUT (dir), LENS (default 1514,2048,15800).
Per length: 1 warmup, 3 plain requests (wall), 3 profiled (one trace each). Random token ids, so no
prefix-cache hits. Analyse with ci/prefill_torch_split.py OUT.

Run on the GPU host: take the unit's `docker run` line, swap `--rm --name vllm7-mxfp4` for your own name,
drop `-p 8000:8000`, add `-v <dir>:/w -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_CUSTOM_SCOPES_FOR_PROFILING=1
-e OUT=/w/out --entrypoint python3` and put `/w/prefill_torch_profile.py` right after the image name.
~6 min at 210 W (load + compile cache warm). Kineto overhead measured: wall within 1-2 % of unprofiled.
"""
import gzip, json, os, random, shutil, sys, time, uuid

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
os.makedirs(OUT, exist_ok=True)
args = make_arg_parser(FlexibleArgumentParser()).parse_args(sys.argv[1:])
engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(args))
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


res = []
for n in LENS:
    one(n)
    for i in range(3):
        dt, st = one(n)
        res.append(["plain", n, i, dt, st])
        print(f"SPLIT plain {n} {dt * 1e3:.1f} ms steps={st}", flush=True)
    for i in range(3):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False,
                     with_stack=False) as p:
            dt, st = one(n)
        res.append(["prof", n, i, dt, st])
        print(f"SPLIT prof  {n} {dt * 1e3:.1f} ms steps={st}", flush=True)
        tmp = f"/tmp/t_{n}_{i}.json"
        p.export_chrome_trace(tmp)
        with open(tmp, "rb") as f, gzip.open(f"{OUT}/trace_{n}_{i}.json.gz", "wb") as g:
            shutil.copyfileobj(f, g)
        os.remove(tmp)
json.dump(res, open(f"{OUT}/requests.json", "w"))
print("SPLIT done", flush=True)
os._exit(0)  # skip engine teardown (in-process engine shutdown can hang on ROCm)
