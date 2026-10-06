#!/usr/bin/env python3
"""greedy_dump.py URL OUT.json -- greedy (T=0) completions of a fixed prompt set, one at a time and
eight at a time (the M=8..64 verify shapes), saved for byte comparison between server arms."""
import concurrent.futures as cf
import json
import sys
import urllib.request

U, OUT = sys.argv[1].rstrip("/"), sys.argv[2]
PROMPTS = ["Write a Python class implementing an LRU cache with type hints.",
           "Explain how the TLS 1.3 handshake works, step by step.",
           "Schreibe eine kurze Geschichte ueber eine Bergwanderung im Nebel.",
           "Solve step by step: a train travels 120 km in 1.5 hours, then 80 km in 1 hour. Average speed?",
           "Nenne 12 Ideen fuer ein Wochenende mit Kindern bei Regen.",
           "Implement binary search in Rust with generics.",
           "What is the derivative of x^x? Show the derivation.",
           "Erklaere den Unterschied zwischen Prozessen und Threads unter Linux."]
model = json.load(urllib.request.urlopen(U + "/v1/models"))["data"][0]["id"]


def gen(p, n=200):
    b = {"model": model, "messages": [{"role": "user", "content": p}], "max_tokens": n, "temperature": 0,
         "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(U + "/v1/chat/completions", json.dumps(b).encode(),
                               {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))["choices"][0]["message"]["content"]


out = {"single": [gen(p) for p in PROMPTS]}
with cf.ThreadPoolExecutor(8) as ex:
    out["conc8"] = list(ex.map(gen, PROMPTS))
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT, sum(len(x) for x in out["single"]), "chars single")
