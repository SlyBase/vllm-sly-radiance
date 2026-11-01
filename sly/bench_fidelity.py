#!/usr/bin/env python3
"""Prompt-logprob fidelity probe against a running server: per-token top-K logprobs on fixed texts -> JSON.

The quality gate for quantization / kernel numerics changes. One arm is one server start; compare arms
with sly/check_fidelity.py against a reference arm (typically the same checkpoint served exactly, e.g.
RADIANCE_NVFP4_DIAG=native). Deterministic: a repeat on the same server is bit-identical, so paired
differences resolve ~0.01 NLL on ~14k tokens -- far below GSM8K's +-2.7 pp. Needs no sampling, no
spec decode and no graph capture (prompt_logprobs is prefill only), so run arms with --enforce-eager.

Default texts are present on every Debian/Ubuntu host: 4 stdlib json modules + textwrap (code), a GPL-3
excerpt (legal prose). Add more with --text NAME=PATH (first --max-chars characters are used).

usage: bench_fidelity.py OUT.json [--base http://127.0.0.1:8000] [--text de=/path/de.md ...] [--top 20]
"""
import argparse
import glob
import json
import urllib.request


def post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=1800))


def default_texts(max_chars):
    srcs = sorted(glob.glob("/usr/lib/python3*/json/*.py"))[:4] + sorted(glob.glob("/usr/lib/python3*/textwrap.py"))[:1]
    texts = {f.rsplit("/", 1)[-1]: open(f).read()[:6000] for f in srcs}
    gpl = sorted(glob.glob("/usr/share/common-licenses/GPL-3"))
    if gpl:
        texts["prose_license"] = open(gpl[0]).read()[2000:9000]
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--text", action="append", default=[], help="NAME=PATH, repeatable")
    ap.add_argument("--no-default-texts", action="store_true")
    ap.add_argument("--max-chars", type=int, default=9000)
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()

    model = json.load(urllib.request.urlopen(a.base + "/v1/models"))["data"][0]["id"]
    texts = {} if a.no_default_texts else default_texts(a.max_chars)
    for spec in a.text:
        name, _, path = spec.partition("=")
        texts[name] = open(path).read()[:a.max_chars]
    out = {"model": model, "top": a.top, "texts": {}}
    for name, t in texts.items():
        r = post(a.base, "/v1/completions", {"model": model, "prompt": t, "max_tokens": 1, "temperature": 0,
                                             "prompt_logprobs": a.top})
        pl = r["choices"][0].get("prompt_logprobs") or r.get("prompt_logprobs")
        ids = post(a.base, "/tokenize", {"model": model, "prompt": t})["tokens"]
        assert len(ids) == len(pl), (len(ids), len(pl))
        rows = [[tid, pos[str(tid)]["logprob"], {k: v["logprob"] for k, v in pos.items()}]
                for tid, pos in zip(ids[1:], pl[1:]) if pos]
        out["texts"][name] = rows
        print(f"{name:14} n={len(rows)} nll={-sum(r[1] for r in rows) / len(rows):.4f}", flush=True)
    json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
