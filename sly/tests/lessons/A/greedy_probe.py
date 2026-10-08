#!/usr/bin/env python3
"""greedy_probe.py URL OUT.json   -- fixed token-id prompts of 72..904 tokens, temperature 0, 32 tokens,
                                     with logprobs; prompt lengths hit the M bands of the GEMM (wide: 129..256).
   greedy_probe.py cmp A.json B.json -- identical-output count, first-token and all-token logprob deltas.
Run it FIRST on a freshly started server (the prompts repeat across arms, not inside one arm)."""
import json, random, sys, urllib.request

LENS = [72, 72, 110, 110, 160, 160, 200, 200, 256, 256, 400, 904]
MODEL = "slybase/Swift-1.5-Qwen3.8-27B-heretic-MXFP4-GPTQ"


def run(url, out):
    rng = random.Random(4242)
    res = []
    for i, n in enumerate(LENS):
        ids = [rng.randrange(2000, 60000) for _ in range(n)]
        body = {"model": MODEL, "prompt": ids, "max_tokens": 32, "temperature": 0, "ignore_eos": True,
                "logprobs": 1}
        r = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(),
                                   {"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(r, timeout=300))
        c = d["choices"][0]
        res.append({"n": n, "text": c["text"], "lp": c["logprobs"]["token_logprobs"]})
    json.dump(res, open(out, "w"))
    print(f"greedy probe: {len(res)} requests -> {out}")


def cmp(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    same = sum(x["text"] == y["text"] for x, y in zip(A, B))
    d1 = max(abs(x["lp"][0] - y["lp"][0]) for x, y in zip(A, B))
    dall = max(abs(p - q) for x, y in zip(A, B) for p, q in zip(x["lp"], y["lp"]))
    bad = [x["n"] for x, y in zip(A, B) if x["text"] != y["text"]]
    print(f"greedy identical {same}/{len(A)}  max|dlogprob| first token {d1:.4f}, any token {dall:.4f}"
          + (f"  differing prompt lengths {bad}" if bad else ""))
    return same, len(A), d1


if __name__ == "__main__":
    if sys.argv[1] == "cmp":
        cmp(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1], sys.argv[2])
