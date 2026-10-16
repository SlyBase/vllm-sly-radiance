"""Teacher-forced scoring of long texts against the running server (prompt logprobs, top-K).

usage: KVACC_DIR=<dir with raw/*.txt> kl_score.py <arm>   -> <dir>/out/<arm>/<doc>.npz
       (lp_true, top_ids[K], top_lp[K]); then kl_compare.py [arms] (reference arm: bf16).
Start the engine per arm (bf16 reference: QWEN38V_KV_DTYPE=auto; the same --max-model-len for all).
env: KVACC_TOKENS (120000), KVACC_K (20), KVACC_DOCS (comma list, default all in raw/), KVACC_URL,
KVACC_KEY / KVACC_KEY_FILE, KVACC_MODEL
Token ids are cut once (first run, via the server's /tokenize) and cached in ids/, so every arm
scores exactly the same sequence.
"""
import json, os, sys, time, urllib.request
import numpy as np

D = os.path.abspath(os.environ.get("KVACC_DIR", "kvacc"))   # raw/ texts in, ids/ + out/<arm>/ written
URL = os.environ.get("KVACC_URL", "http://127.0.0.1:9090")
KEY = os.environ.get("KVACC_KEY") or open(os.path.expanduser(os.environ.get("KVACC_KEY_FILE", "~/.config/qwen38/api-key"))).read().strip()
MODEL = os.environ.get("KVACC_MODEL", "qwen3.8-27b")
N = int(os.environ.get("KVACC_TOKENS", "120000"))
K = int(os.environ.get("KVACC_K", "20"))


def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {KEY}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def strip_gutenberg(t):
    a = t.find("*** START OF")
    b = t.find("*** END OF")
    if a >= 0:
        t = t[t.find("\n", a) + 1:b if b > a else None]
    return t


def ids_for(doc):
    os.makedirs(f"{D}/ids", exist_ok=True)
    p = f"{D}/ids/{doc}.json"
    if os.path.exists(p):
        return json.load(open(p))
    text = strip_gutenberg(open(f"{D}/raw/{doc}.txt", errors="replace").read())
    ids = []
    for i in range(0, len(text), 200_000):  # /tokenize in pieces, cut at piece borders is fine
        ids += post("/tokenize", {"model": MODEL, "prompt": text[i:i + 200_000],
                                  "add_special_tokens": False})["tokens"]
        if len(ids) >= N:
            break
    ids = ids[:N]
    json.dump(ids, open(p, "w"))
    return ids


def main():
    arm = sys.argv[1]
    docs = os.environ.get("KVACC_DOCS") or ",".join(
        sorted(f[:-4] for f in os.listdir(f"{D}/raw") if f.endswith(".txt")))
    if arm == "bf16r" and not os.environ.get("KVACC_DOCS"):
        docs = docs.split(",")[0]  # noise floor: one repeat is enough
    os.makedirs(f"{D}/out/{arm}", exist_ok=True)
    for doc in docs.split(","):
        ids = ids_for(doc)
        t0 = time.time()
        r = post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 1,
                                     "echo": True, "logprobs": K, "temperature": 0,
                                     "return_tokens_as_token_ids": True})
        lp = r["choices"][0]["logprobs"]
        n = len(ids)
        lp_true = np.full(n, np.nan, np.float32)
        top_ids = np.full((n, K), -1, np.int64)
        top_lp = np.full((n, K), -np.inf, np.float32)
        for i in range(1, n):  # position i = prediction of ids[i] from ids[:i]
            lp_true[i] = lp["token_logprobs"][i]
            items = sorted(lp["top_logprobs"][i].items(), key=lambda kv: -kv[1])
            for j, (tok, v) in enumerate(items[:K]):  # dict may hold K+1 (the true token)
                top_ids[i, j] = int(tok.split(":")[1])
                top_lp[i, j] = v
        np.savez_compressed(f"{D}/out/{arm}/{doc}.npz", ids=np.array(ids), lp_true=lp_true,
                            top_ids=top_ids, top_lp=top_lp)
        print(f"[{arm}] {doc}: {n} tok in {time.time() - t0:.0f} s, "
              f"ppl {np.exp(-np.nanmean(lp_true[1:])):.3f}", flush=True)


main()
