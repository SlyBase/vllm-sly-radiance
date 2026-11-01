#!/usr/bin/env python3
"""probe_h.py URL OUT.json -- greedy text of short prompts (below RADIANCE_R4D_PREFILL_MIN_Q) and long
prompts (above it), plus the prompt log-probabilities of one long text.

SHORT prompts (< 512 prompt tokens) never reach libr4d in R4D_HYBRID, so their greedy text must equal
the control's exactly. LONG prompts do: text may drift late (different prefill kernel), reported as the
first diverging token position. The prompt log-probs of a ~3000-token text quantify the prefill kernel
difference directly (mean NLL and per-token |delta| against the control arm, see test.sh).

Run it FIRST after a start: prefix caching is on, so every prompt is sent exactly once on a cold cache
and chunks identically in every arm. Stdlib only."""
import json
import sys
import urllib.request

URL, OUT = sys.argv[1], sys.argv[2]
MODEL = None


def post(path, body, timeout=900):
    r = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


MODEL = json.load(urllib.request.urlopen(URL + "/v1/models"))["data"][0]["id"]

# natural text of controlled length: the probes' own docstrings and a fixed paragraph, repeated
PARA = (
    "The scheduler hands the engine a batch of requests every step. Each request carries the tokens it "
    "still has to prefill, or the handful of draft tokens it wants verified. Prefill chunks are large and "
    "compute bound; verification batches are tiny and bandwidth bound, which is why one attention kernel "
    "rarely suits both. A paged KV cache stores keys and values in blocks, and the block table maps a "
    "sequence position to the block that holds it. Quantising the cache to eight bits halves the traffic "
    "but moves the rounding into the attention scores. "
)


def text_of(words):
    w = PARA.split()
    return " ".join((w * (words // len(w) + 1))[:words])


def complete(prompt, n=96):
    d = post("/v1/completions", {"model": MODEL, "prompt": prompt, "max_tokens": n, "temperature": 0})
    return d["choices"][0]["text"], d["usage"]["prompt_tokens"]


res = {"short": [], "long": [], "lp": None}
for words in (6, 40, 120, 220, 300):
    prompt = "Continue the following text.\n\n" + text_of(words)
    t, pt = complete(prompt)
    res["short"].append({"words": words, "prompt_tokens": pt, "text": t})
    print(f"short words={words:4d} prompt_tokens={pt:4d} -> {t[:60]!r}", flush=True)
for words in (700, 2200):
    prompt = "Continue the following text.\n\n" + text_of(words)
    t, pt = complete(prompt)
    res["long"].append({"words": words, "prompt_tokens": pt, "text": t})
    print(f"long  words={words:4d} prompt_tokens={pt:5d} -> {t[:60]!r}", flush=True)

# prompt log-probs of a ~3000-token natural text (token ids so the entry of the real token is findable)
src = ""
for f in ("/root/lessons/accprobe3.py", "/root/lessons/bbgap2.py"):
    try:
        src += open(f).read() + "\n"
    except OSError:
        pass
src = (src or text_of(3000))[:14000]
ids = post("/tokenize", {"model": MODEL, "prompt": src})["tokens"][:3000]
d = post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})
pl = d["choices"][0].get("prompt_logprobs")
if pl:
    nll = []
    for tid, ent in zip(ids, pl):
        if ent is None:
            nll.append(None)
            continue
        e = ent.get(str(tid))
        nll.append(-e["logprob"] if e else None)
    res["lp"] = nll
    ok = [x for x in nll if x is not None]
    print(f"prompt logprobs: {len(ok)} tokens, mean NLL {sum(ok) / max(1, len(ok)):.5f}", flush=True)
else:
    print("prompt_logprobs not returned by this server", flush=True)
json.dump(res, open(OUT, "w"))
