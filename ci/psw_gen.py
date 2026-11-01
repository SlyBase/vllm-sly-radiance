"""Greedy decode probe on the PRODUCTION engine (in-process, production argv), for PSW on/off A/B.

Run once per arm (RADIANCE_MXFP4_AT_PSW=0 / =2 in the env, CUDA graphs are captured at startup).
Three prompts x REPS repeats, temperature 0, MAXTOK tokens each:
  p0  600 random token ids             (short prefill, decode band only)
  p1  2000 random token ids            (prefill chunk 2048 -> A-tiled PSW path, then decode)
  p2  a natural-language story prompt  (chat template, like the server decode guard)
Prints `GEN <prompt> rep=<r> sha=<16> n=<tokens> head=<8 ids>` per generation and a GENSUM line.
Equal sha across arms = same tokens = decode (and prefill) bit-identical in practice; equal sha
across repeats inside one arm = the arm itself is deterministic.
"""
import hashlib, os, random, sys

from vllm import SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.inputs import TokensPrompt
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.v1.engine.llm_engine import LLMEngine

REPS = int(os.environ.get("REPS", "2"))
MAXTOK = int(os.environ.get("MAXTOK", "128"))
args = make_arg_parser(FlexibleArgumentParser()).parse_args(sys.argv[1:])
engine = LLMEngine.from_engine_args(EngineArgs.from_cli_args(args))
tok = engine.get_tokenizer()

g = random.Random(7)
prompts = {
    "p0": [g.randrange(1000, 120000) for _ in range(600)],
    "p1": [g.randrange(1000, 120000) for _ in range(2000)],
    "p2": tok.encode(tok.apply_chat_template([{"role": "user", "content": "Write a long story about a lighthouse keeper."}],
                                             tokenize=False, add_generation_prompt=True, enable_thinking=False),
                     add_special_tokens=False),
}
shas = []
n_req = 0
for r in range(REPS):
    for name, ids in prompts.items():
        n_req += 1
        rid = f"{name}-{r}-{n_req}"
        engine.add_request(rid, TokensPrompt(prompt_token_ids=list(ids)),
                           SamplingParams(max_tokens=MAXTOK, temperature=0, ignore_eos=True, seed=1234))
        toks = []
        while engine.has_unfinished_requests():
            for o in engine.step():
                if o.finished:
                    toks = list(o.outputs[0].token_ids)
        sha = hashlib.sha256(str(toks).encode()).hexdigest()[:16]
        shas.append(f"{name}:{sha}")
        print(f"GEN {name} rep={r} sha={sha} n={len(toks)} head={toks[:8]}", flush=True)
print("GENSUM psw=" + os.environ.get("RADIANCE_MXFP4_AT_PSW", "0") + " " + " ".join(shas), flush=True)
