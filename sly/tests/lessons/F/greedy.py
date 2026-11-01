#!/usr/bin/env python3
"""greedy.py URL OUT.json -- 8 fixed prompts, temperature 0, 160 tokens; saves the texts.
greedy.py --cmp A.json B.json -- per prompt: identical, or the char index of the first divergence."""
import json
import sys
import urllib.request

PROMPTS = ["Write a Python class implementing an LRU cache with type hints and unit tests.",
           "Explain how TLS 1.3 handshake works step by step.",
           "Erklaere den Unterschied zwischen Prozessen und Threads unter Linux ausfuehrlich.",
           "Write a short story about a robot learning to paint.",
           "Solve step by step: a train travels 120 km in 1.5 hours, then 80 km in 1 hour. What is the average speed?",
           "Nenne 15 Ideen fuer ein Wochenende mit Kindern bei Regen, jeweils mit einem Satz Begruendung.",
           "Implement binary search in Rust with generics and document edge cases.",
           "Wie funktioniert ein Kalman-Filter? Erklaere mit einem Beispiel."]


def post(url, path, body):
    r = urllib.request.Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=600))


if sys.argv[1] == "--cmp":
    a, b = (json.load(open(f)) for f in sys.argv[2:4])
    same = 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            same += 1
            print(f"  prompt {i}: identical ({len(x)} chars)")
        else:
            k = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            print(f"  prompt {i}: first divergence at char {k} of {len(x)}/{len(y)}")
    print(f"GREEDY identical {same}/{len(a)}")
    sys.exit(0)

url, out = sys.argv[1], sys.argv[2]
model = json.load(urllib.request.urlopen(url + "/v1/models"))["data"][0]["id"]
NOTHINK = {"enable_thinking": False}
post(url, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Hallo"}],
                                   "max_tokens": 8, "temperature": 0, "chat_template_kwargs": NOTHINK})
texts = []
for p in PROMPTS:
    d = post(url, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": p}],
                                           "max_tokens": 160, "temperature": 0, "ignore_eos": True,
                                           "chat_template_kwargs": NOTHINK})
    texts.append(d["choices"][0]["message"]["content"])
json.dump(texts, open(out, "w"), ensure_ascii=False, indent=1)
print(f"greedy: {len(texts)} prompts -> {out}")
