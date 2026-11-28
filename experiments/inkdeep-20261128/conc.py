#!/usr/bin/env python3
"""Concurrency probe: does c1 decode at 64k context lose vs c2/c4?

One shared 64k-token haystack (prefix-cached after a warmup request), then C
simultaneous decode requests with distinct questions, temp 0, max_tokens 200.
Reports per-C wall time, total completion tokens and aggregate tok/s (2 reps).
Usage: conc.py URL out.json
"""
import json
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, OUT = sys.argv[1], sys.argv[2]
MODEL = "inkling2-mini"
rnd = random.Random(1234 + 64000)
words = ["".join(rnd.choices("abcdefghijklmnopqrstuvwxyz", k=rnd.randint(3, 9))) for _ in range(4000)]
hay = " ".join(rnd.choices(words, k=62000))  # ~64k tokens after template


def ask(i, max_tokens=200):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": hay + f"\n\nRepeat the number {1000 + i} exactly, then write one short sentence about the sea."}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        URL + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    tok = d["usage"]["completion_tokens"]
    return {"i": i, "tok": tok, "s": round(dt, 2)}


res = []
# warmup: prime the prefix cache with the shared haystack
ask(0, max_tokens=8)
for c in (1, 2, 4):
    for rep in (1, 2):
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=c) as ex:
            items = list(ex.map(lambda i: ask(i), range(c)))
        wall = time.time() - t0
        tot = sum(x["tok"] for x in items)
        res.append({"c": c, "rep": rep, "wall_s": round(wall, 2), "tokens": tot,
                    "agg_tps": round(tot / wall, 1), "items": items})
        print(f"c={c} rep={rep} wall={wall:.1f}s tokens={tot} agg={tot / wall:.1f} tok/s", flush=True)
json.dump({"res": res}, open(OUT, "w"), indent=1)