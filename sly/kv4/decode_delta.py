#!/usr/bin/env python3
"""Pure decode rate against a running server, by the difference method: every stream decodes
SHORT and then LONG tokens on the same (prefix-cached) prompt, the difference cancels prefill,
queueing and the first-token cost. Reports aggregate tok/s, tokens per step (1 + accepted drafts
per draft, from /metrics) and the step time -- the robust number, since the accepted length moves
with the numerics of every arm.

  decode_delta.py TAG [--streams 4] [--prefix 18000] [--short 150] [--long 450] [--reps 1]

env: DD_URL (http://127.0.0.1:9090), DD_KEY_FILE (~/.config/qwen38/api-key), DD_MODEL (qwen3.8-27b)
"""
import argparse
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("DD_URL", "http://127.0.0.1:9090")
KEY = open(os.path.expanduser(os.environ.get("DD_KEY_FILE", "~/.config/qwen38/api-key"))).read().strip()
MODEL = os.environ.get("DD_MODEL", "qwen3.8-27b")
ANIMALS = ["pelican", "otter", "heron", "badger", "lynx", "marmot", "puffin", "ibex"]


def post(body, timeout=3600):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def spec_counts():
    with urllib.request.urlopen(URL + "/metrics", timeout=30) as r:
        txt = r.read().decode()

    def g(name):
        return sum(float(x) for x in re.findall(rf"^{name}(?:\{{[^}}]*\}})? ([0-9.e+]+)$", txt, re.M))
    return g("vllm:spec_decode_num_accepted_tokens_total"), g("vllm:spec_decode_num_drafts_total")


def prompt(i, prefix_tokens):
    unit = f"The quick brown {ANIMALS[i % len(ANIMALS)]} jumps over the lazy dog number {i}. " * 12 + "\n"
    reps = max(1, prefix_tokens // 150)          # one unit is ~150 tokens
    return "".join(unit for _ in range(reps)) + \
        f"\n\nWrite a long story about a {ANIMALS[i % len(ANIMALS)]} named Barnaby who learns to fly across the ocean."


def chat(text, n):
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}], "max_tokens": n,
            "temperature": 0.0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    return post(body)["usage"]["completion_tokens"]


def phase(texts, n):
    a0, d0 = spec_counts()
    t0 = time.time()
    with ThreadPoolExecutor(len(texts)) as ex:
        toks = list(ex.map(lambda t: chat(t, n), texts))
    dt = time.time() - t0
    a1, d1 = spec_counts()
    return dt, sum(toks), a1 - a0, d1 - d0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--streams", type=int, default=4)
    ap.add_argument("--prefix", type=int, default=18000, help="approximate prompt tokens per stream")
    ap.add_argument("--short", type=int, default=150)
    ap.add_argument("--long", type=int, default=450)
    ap.add_argument("--reps", type=int, default=1)
    a = ap.parse_args()
    texts = [prompt(i, a.prefix) for i in range(a.streams)]
    phase(texts, 1)                              # prefill every prefix once (prefix cache)
    for rep in range(a.reps):
        ts, ns, _, _ = phase(texts, a.short)
        tl, nl, acc, drafts = phase(texts, a.long)
        rate = (nl - ns) / (tl - ts)
        tps = 1.0 + (acc / drafts if drafts else 0.0)   # tokens per step and stream (long phase)
        steps = (a.long - a.short) / tps
        print(f"[{a.tag}] {a.streams} streams x ~{a.prefix // 1000}k: aggregate decode {rate:6.1f} tok/s "
              f"({rate / a.streams:5.1f} per stream)  step {(tl - ts) / steps * 1000:5.1f} ms  "
              f"tokens/step/stream {tps:4.2f}  ({a.short}: {ts:.2f}s, {a.long}: {tl:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
