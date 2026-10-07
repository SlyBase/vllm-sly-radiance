#!/usr/bin/env python3
"""Greedy output capture / comparison for the adaptive-width A/B (package D).

  greedy_eq.py run URL TAG OUT.json   single-stream (sequential) and 8-way concurrent greedy requests
  greedy_eq.py cmp REF.json OTHER.json   per-mode: identical streams and mean common prefix (chars)

single: nothing is shrunk (fewer than 32 unshrunk rows), the output must be IDENTICAL to the control.
conc  : 8 requests at once; the decode GEMM split depends on the batch shape, so bytes may drift by an
        ulp-level tie flip (the same drift the engine has across concurrency levels). Read the common
        prefix against the control-vs-control comparison (off vs off2), not against 100 %.
"""
import json
import sys
import threading
import urllib.request

P = [
    "Write a Python class implementing an LRU cache with type hints and unit tests.",
    "Explain how TLS 1.3 handshake works step by step.",
    "Write a short story about a robot learning to paint.",
    "Implement binary search in Rust with generics and document edge cases.",
    "Erklaere den Unterschied zwischen Prozessen und Threads unter Linux ausfuehrlich.",
    "Wie funktioniert ein Kalman-Filter? Erklaere mit einem Beispiel.",
    "Nenne 15 Ideen fuer ein Wochenende mit Kindern bei Regen, jeweils mit einem Satz Begruendung.",
    "Schreibe eine SQL-Abfrage mit CTE, die pro Kunde den Umsatz der letzten 3 Monate berechnet, und erklaere sie.",
]


def req(url, model, prompt, n):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
         "temperature": 0, "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(b).encode(),
                               {"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=900) as f:
        return json.loads(f.read())["choices"][0]["message"]["content"]


def run(url, tag, out):
    model = json.loads(urllib.request.urlopen(url + "/v1/models").read())["data"][0]["id"]
    req(url, model, "Hallo", 8)
    res = {"tag": tag, "single": [], "conc": [None] * len(P)}
    for p in P[:4]:
        res["single"].append(req(url, model, p, 192))

    def w(i):
        res["conc"][i] = req(url, model, P[i], 192)

    ts = [threading.Thread(target=w, args=(i,)) for i in range(len(P))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    json.dump(res, open(out, "w"))
    print(f"greedy_eq: {tag}: {len(res['single'])} single + {len(res['conc'])} conc streams -> {out}")


def prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def cmp(ref, oth):
    A, B = json.load(open(ref)), json.load(open(oth))
    print(f"greedy_eq: {A['tag']} vs {B['tag']}")
    ok = True
    for mode in ("single", "conc"):
        a, b = A[mode], B[mode]
        same = sum(1 for x, y in zip(a, b) if x == y)
        pre = [prefix(x, y) / max(len(x), 1) for x, y in zip(a, b)]
        print(f"  {mode:6s}: identical {same}/{len(a)}  mean common prefix {100 * sum(pre) / len(pre):.1f} %")
        if mode == "single" and same != len(a):
            ok = False
    print("  single-stream identical:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2].rstrip("/"), sys.argv[3], sys.argv[4])
    else:
        cmp(sys.argv[2], sys.argv[3])
