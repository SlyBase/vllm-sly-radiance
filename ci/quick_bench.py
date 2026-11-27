#!/usr/bin/env python3
"""Regression gates and the ~5 minute GPU "quick" tier (stdlib only).

Pure checkers (unit tests: ci/tests/test_quick_bench.py) that the accept gate and the CLI share:
  hook_problems         required / forbidden server-log lines per profile (qwen | inkling)
  greedy_problems       4 greedy prompts vs the stored output of the last release, + allow-list
  concurrency_problems  aggregate tok/s rises 1 -> 2 -> 4 -> 8, and 2 streams >= 1.2x one stream
  floor_problems        drafter tok/update (code) >= floor
  multi_env_problems    harness lint: one `-e KEY=VAL` per -e
  verdict               quick-tier measurements vs a reference JSON

  unrecorded_problems   release tier: fail closed while a reference is not recorded
  release_gate_problems both profiles must have a green `release-tier/<profile>` commit status
  gsm8k_correct         scoring of the GSM8K-50 quality check (floor: reference - 2 items)

Live part (`run`): against a server that is already up --
  decode (Mia method, prose+code, 400 tok, ignore_eos, thinking on, T0.7 top_k 20, 5 reps, median)
  + tok/update from /metrics deltas, prefill 8k x2, concurrency 1 vs 4 streams, greedy, hook proof.

  quick_bench.py run --profile inkling --base http://HOST:8000 --model NAME --log server.log
  quick_bench.py run ... --record      # write the measured values as the new reference
See ci/README.md ("GPU tiers").
"""
import argparse
import json
import random
import re
import statistics as st
import sys
import threading
import time
import urllib.request
from pathlib import Path

REFS = Path(__file__).resolve().parent / "accept" / "references"
# accept profiles (ci/accept/profiles/<name>.json) that a release has to pass, one GPU window each
RELEASE_PROFILES = ("vllm7-mxfp4", "inkling")
QUICK_TWO_STREAM_MIN = 1.05
GSM8K_CONC = 1  # streams of the GSM8K check (greedy at c1 is reproducible across starts)
GSM8K_SLACK = 2  # items below the stored reference that still pass (50 items: 4 %)
RECORD_HINT = "references not recorded, run ci/quick_backtest.sh RECORD=1"

# ---------------------------------------------------------------------------------------------
# hook proof
# ---------------------------------------------------------------------------------------------
# Strings verified against real host-51 logs (1.13.4 good Qwen, eager Qwen, Inkling ci-262).
HOOKS = {
    "qwen": {
        "required": [
            ("[radiance.fused_norm] add_rms_quant active", r"\[radiance\.fused_norm\] add_rms_quant active"),
            ("CompilationMode != NONE", r"compilation_config=\{'mode': <CompilationMode\.(?!NONE)"),
        ],
        "forbidden": [("CompilationMode.NONE", r"compilation_config=\{'mode': <CompilationMode\.NONE")],
    },
    "inkling": {
        "required": [
            ("RADIANCE_CODA_FP4_PROOF n=1 mxfp8->nvfp4", r"RADIANCE_CODA_FP4_PROOF n=1 mxfp8->nvfp4"),
            ("Capturing model for DFlash2 speculator", r"Capturing model for DFlash2 speculator"),
            ("drafter conv projections int4 0", r"drafter conv projections int4 0\b"),
        ],
        "forbidden": [("drafter conv projections int4 >0", r"drafter conv projections int4 [1-9]")],
    },
}
INKLING_GRAPH_SIZES = {16, 24, 32, 40, 48, 56, 64}


def hook_problems(log: str, profile: str) -> list[str]:
    spec = HOOKS[profile]
    out = [f"missing: {n}" for n, rx in spec["required"] if not re.search(rx, log)]
    out += [f"forbidden: {n}" for n, rx in spec["forbidden"] if re.search(rx, log)]
    if profile == "inkling":
        sizes = [set(map(int, m.split(","))) for m in
                 re.findall(r"'cudagraph_capture_sizes': \[(\d[\d, ]*)\]", log)]
        if not any(INKLING_GRAPH_SIZES <= s for s in sizes):
            out.append("cudagraph_capture_sizes do not cover 16..64 (steps of 8)")
    return out


# ---------------------------------------------------------------------------------------------
# greedy / concurrency / drafter floor / lint
# ---------------------------------------------------------------------------------------------
def greedy_problems(cur: list, ref: list, allow: dict) -> list[str]:
    """allow: {"<prompt index>": "reason"} -- an entry without a reason does not count."""
    if len(cur) != len(ref):
        return [f"{len(cur)} greedy answers, reference has {len(ref)}"]
    out = []
    for i, (a, b) in enumerate(zip(cur, ref)):
        key = lambda x: (x.get("finish"), x.get("content"), x.get("reasoning"))  # noqa: E731
        if key(a) == key(b) or str(allow.get(str(i), "")).strip():
            continue
        ta, tb = (a.get("reasoning") or "") + (a.get("content") or ""), (b.get("reasoning") or "") + (b.get("content") or "")
        at = next((n for n, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        out.append(f"prompt {i} differs from the last release (first difference at char {at}); "
                   f"fix it or document the reason in the allow-list")
    return out


def concurrency_problems(agg: dict, two_stream_min: float = 1.2) -> list[str]:
    levels = sorted(agg)
    out = [f"aggregate tok/s not rising: {a}->{b} streams {agg[a]:.1f}->{agg[b]:.1f}"
           for a, b in zip(levels, levels[1:]) if agg[b] <= agg[a]]
    if 1 in agg and 2 in agg and agg[2] < two_stream_min * agg[1]:
        out.append(f"2 streams {agg[2] / agg[1]:.2f}x of 1 stream, need >= {two_stream_min}x")
    return out


def category_tok_per_update(bb: dict) -> dict:
    """median tok/update per category of a BetterBench single-stream JSON (same rule as accept.py)"""
    out = {}
    for cat, entries in (bb.get("single_stream") or {}).items():
        v = [e["tokens_per_update"] for e in entries if e.get("ok") and e.get("tokens_per_update") is not None]
        if v:
            out[cat] = round(st.median(v), 1)
    return out


def floor_problems(tpu: dict, floor: float, cat: str = "code") -> list[str]:
    if cat not in tpu:
        return [f"tok/update ({cat}) not measured"]
    return [] if tpu[cat] >= floor else [f"tok/update ({cat}) {tpu[cat]} < floor {floor}"]


MULTI_ENV = re.compile(r"""-e\s+(?:"|')?[A-Za-z_]\w*=[^\s"']*\s+[A-Za-z_]\w*=""")


def multi_env_problems(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if MULTI_ENV.search(ln) and not ln.lstrip().startswith("#")]


# ---------------------------------------------------------------------------------------------
# quick-tier verdict
# ---------------------------------------------------------------------------------------------
def verdict(m: dict, ref: dict, thr: dict) -> list[tuple]:
    """(name, ok, detail) per referenced metric; a reference entry that is null is not calibrated yet: skipped."""
    rows = []

    def rel(name, val, base, pct):
        if base is None:
            return
        if val is None:
            rows.append((name, False, "not measured"))
            return
        rows.append((name, val >= base * (1 - pct / 100), f"{val} vs {base} (-{pct}% allowed)"))

    for cat, base in (ref.get("decode_tps") or {}).items():
        rel(f"decode {cat}", (m.get("decode_tps") or {}).get(cat), base, thr["decode_pct"])
    rel("prefill", m.get("prefill_pps"), ref.get("prefill_pps"), thr["prefill_pct"])
    for cat, floor in (ref.get("tok_per_update") or {}).items():
        v = (m.get("tok_per_update") or {}).get(cat)
        rows.append((f"tok/update {cat}", v is not None and v >= floor, f"{v} vs floor {floor}"))
    if ref.get("gsm8k_correct") is not None:
        v = m.get("gsm8k_correct")
        floor = ref["gsm8k_correct"] - GSM8K_SLACK
        rows.append(("gsm8k", v is not None and v >= floor, f"{v}/50 vs floor {floor} (reference {ref['gsm8k_correct']})"))
    for k in ("c2", "c4", "c8"):
        floor = ref.get(f"{k}_over_c1_min")
        if floor is not None:
            v = m.get(f"{k}_over_c1")
            rows.append((f"{k}/c1", v is not None and v >= floor, f"{v} vs floor {floor}"))
    return rows


def unrecorded_problems(ref: dict, greedy_present: bool) -> list[str]:
    """Release mode never skips an uncalibrated (null) reference entry: it fails closed instead."""
    missing = []
    d = ref.get("decode_tps") or {}
    if not d or any(v is None for v in d.values()):
        missing.append("decode_tps")
    if ref.get("prefill_pps") is None:
        missing.append("prefill_pps")
    t = ref.get("tok_per_update") or {}
    if not t or any(v is None for v in t.values()):
        missing.append("tok_per_update")
    missing += [k for k in ("c2_over_c1_min", "c4_over_c1_min", "c8_over_c1_min", "gsm8k_correct")
                if ref.get(k) is None]
    if not greedy_present:
        missing.append("greedy reference")
    return [f"{RECORD_HINT} (missing: {', '.join(missing)})"] if missing else []


def release_gate_problems(statuses: dict) -> list[str]:
    """statuses: {context: state} of the commit being tagged."""
    return [f"release tier for profile {p} is not green (release-tier/{p}: {statuses.get(f'release-tier/{p}', 'missing')})"
            for p in RELEASE_PROFILES if statuses.get(f"release-tier/{p}") != "success"]


def gsm8k_correct(content: str, gold: str) -> bool:
    m = re.findall(r"####\s*\$?\s*(-?[\d,]*\.?\d+)", content)
    if not m:  # fallback: last number of the visible answer
        m = re.findall(r"-?\d[\d,]*\.?\d*", content[-200:])
    if not m:
        return False
    try:
        return abs(float(m[-1].replace(",", "").rstrip(".")) - float(gold)) < 1e-6
    except ValueError:
        return False


def load_gsm8k(n: int = 50) -> list:
    return [json.loads(line) for line in (REFS / "gsm8k50.jsonl").read_text().splitlines() if line.strip()][:n]


# ---------------------------------------------------------------------------------------------
# /metrics
# ---------------------------------------------------------------------------------------------
_PROM = re.compile(r"^(vllm:spec_decode_num_(?:drafts|accepted_tokens)_total)(?:\{[^}]*\})?\s+([-\d.e+]+)\s*$")


def parse_prom(text: str) -> dict:
    out: dict = {}
    for line in text.splitlines():
        m = _PROM.match(line)
        if m:
            out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out


def tok_per_update(a: dict, b: dict):
    d = b.get("vllm:spec_decode_num_drafts_total", 0.0) - a.get("vllm:spec_decode_num_drafts_total", 0.0)
    acc = b.get("vllm:spec_decode_num_accepted_tokens_total", 0.0) - a.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    return round(1 + acc / d, 3) if d > 0 else None


# ---------------------------------------------------------------------------------------------
# live probes (need a server; not unit-tested beyond stream())
# ---------------------------------------------------------------------------------------------
PROSE = "Write a detailed step-by-step explanation of how a hash map works, including collision handling, resizing, and time complexity. Be thorough."
TAIL = "Output only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers this needs. Keep writing code."
PROMPTS = {"prose": PROSE,
           "code": "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.\n" + TAIL}
WARMUP = {"prose": PROSE, "code": "warmup_noop\ndef warmup_noop(x): return x unchanged.\n" + TAIL}
GREEDY_PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number using memoization, with a docstring and three assert tests.",
    "Implement binary search in Python and state its time complexity in two sentences.",
    "Write a bash one-liner that counts the lines of all .py files below the current directory, then explain it briefly.",
    "Explain what a mutex is and give a short C++ example using std::mutex.",
]
THINK = {"enable_thinking": True, "thinking": True, "thinking_mode": "enabled"}


def _est(s):
    return max(1, round(len(s) / 4)) if s else 0


def stream(url: str, body: dict, timeout: int = 360) -> dict:
    """Mia's timing: first/last chunk that carries text; decode_tps = (completion_tokens-1) / that span."""
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    tf = tl = None
    chunks, usage = 0, {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() in ("", "[DONE]"):
                continue
            j = json.loads(line[5:])
            if isinstance(j.get("usage"), dict):
                usage = j["usage"]
            ch = (j.get("choices") or [{}])[0]
            d = ch.get("delta") or {}
            n = _est(ch.get("text") or d.get("content") or "") + _est(d.get("reasoning") or d.get("reasoning_content") or "")
            if n:
                now = time.perf_counter()
                tf, tl, chunks = tf or now, now, chunks + n
    ct = usage.get("completion_tokens") or chunks
    ms = (tl - tf) * 1000 if tf and tl and tl > tf else 0
    return {"ttft_ms": round((tf - t0) * 1000, 2) if tf else 0, "completion_tokens": ct,
            "decode_tps": round((ct - 1) / ms * 1000, 2) if ms and ct > 1 else 0}


def post(url: str, body: dict, timeout: int = 600) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def prom(base: str) -> dict:
    return parse_prom(urllib.request.urlopen(base + "/metrics", timeout=30).read().decode())


def chat_body(model, prompt, max_tokens, force=True, stream_=True):
    b = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
         "temperature": 0.7, "top_p": 1, "top_k": 20, "stream": stream_, "chat_template_kwargs": THINK}
    if stream_:
        b["stream_options"] = {"include_usage": True}
    if force:
        b.update({"min_tokens": max_tokens, "ignore_eos": True, "stop": []})
    return b


def decode_probe(base, model, reps=5, tokens=400):
    tps, tpu = {}, {}
    for kind in ("prose", "code"):
        stream(base + "/v1/chat/completions", chat_body(model, WARMUP[kind], 32, force=False))
        before = prom(base)
        runs = [stream(base + "/v1/chat/completions", chat_body(model, PROMPTS[kind], tokens)) for _ in range(reps)]
        tpu[kind] = tok_per_update(before, prom(base))
        tps[kind] = round(st.median(r["decode_tps"] for r in runs), 2)
    return tps, tpu


def prefill_probe(base, model, n=8192, reps=2):
    """parser-free token-id prompts, fresh ids each time (no prefix hit); pp/s = n / TTFT."""
    rnd = random.Random(7)
    ttfts = []
    for i in range(reps + 1):  # first one warms the prefill kernels
        ids = [rnd.randrange(1000, 20000) for _ in range(n)]
        r = stream(base + "/v1/completions", {"model": model, "prompt": ids, "max_tokens": 2, "temperature": 0,
                                              "stream": True, "ignore_eos": True})
        if i:
            ttfts.append(r["ttft_ms"])
    return round(n / (st.median(ttfts) / 1000), 1)


def concurrency_probe(base, model, levels=(1, 2, 4), requests=8, tokens=128):
    """aggregate tok/s per stream count. Every level runs twice and only the second round counts: the first
    round meets a batch size the engine has never run (Inkling: cold kernels, read as a 2-stream collapse)."""
    def one_round(level):
        left, done, lock = list(range(requests)), [], threading.Lock()

        def worker():
            while True:
                with lock:
                    if not left:
                        return
                    left.pop()
                r = post(base + "/v1/chat/completions", chat_body(model, PROMPTS["code"], tokens, stream_=False))
                with lock:
                    done.append(r["usage"]["completion_tokens"])

        t0 = time.perf_counter()
        ts = [threading.Thread(target=worker) for _ in range(level)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        return round(sum(done) / (time.perf_counter() - t0), 2)

    agg = {}
    for level in levels:
        one_round(level)  # warm-up, discarded
        agg[level] = one_round(level)
    return agg


def gsm8k_items(base, model, n=50, conc=GSM8K_CONC, max_tokens=4096):
    """GSM8K-n at T=0 (production chat template, thinking on): per-item correct flags. A truncated
    or failed request counts as wrong -- the reference was recorded the same way. One request at a
    time: batching changes spec-decode numerics (same image: 44..47/50 at 8 streams)."""
    from concurrent.futures import ThreadPoolExecutor
    suffix = "\n\nThink step by step. End your reply with the final numeric answer on its own line as: #### <number>"

    def one(it):
        try:
            body = {"model": model, "messages": [{"role": "user", "content": it["question"] + suffix}],
                    "temperature": 0, "top_p": 1, "max_tokens": max_tokens, "seed": 1000 + it["id"],
                    "chat_template_kwargs": THINK}
            r = post(base + "/v1/chat/completions", body, timeout=1500)
            return gsm8k_correct(r["choices"][0]["message"].get("content") or "", it["gold"])
        except Exception:  # noqa: BLE001
            return False

    with ThreadPoolExecutor(conc) as ex:
        return list(ex.map(one, load_gsm8k(n)))


def gsm8k_probe(base, model, n=50):
    return sum(gsm8k_items(base, model, n))


def greedy_run(base, model):
    out = []
    for p in GREEDY_PROMPTS:
        r = post(base + "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": p}],
                                                  "temperature": 0, "top_p": 1, "max_tokens": 192, "seed": 1})
        c = r["choices"][0]
        m = c["message"]
        out.append({"finish": c.get("finish_reason"), "content": m.get("content") or "",
                    "reasoning": m.get("reasoning_content") or m.get("reasoning") or ""})
    return out


def measure(base, model, release=False, gsm8k_n=50):
    m = {}
    m["decode_tps"], m["tok_per_update"] = decode_probe(base, model)
    m["prefill_pps"] = prefill_probe(base, model)
    m["concurrency"] = concurrency_probe(base, model, levels=(1, 2, 4, 8) if release else (1, 2, 4))
    for k in m["concurrency"]:
        if k > 1:
            m[f"c{k}_over_c1"] = round(m["concurrency"][k] / m["concurrency"][1], 3)
    m["greedy"] = greedy_run(base, model)
    if release:
        m["gsm8k_correct"] = gsm8k_probe(base, model, gsm8k_n)
    return m


def run_quick(profile, base, model, log_text, ref_dir=REFS, release=False, gsm8k_n=50, recording=False):
    """-> (report dict, list of failure strings). Hook proof first: a dead hook makes the numbers moot.
    release=True: concurrency up to 8 streams + GSM8K-50, and an unrecorded reference FAILS (no skip);
    recording=True (the backtest's RECORD=1 on a known-good image) is the only way to get past that."""
    ref_p = ref_dir / f"quick-{profile}.json"
    ref = json.loads(ref_p.read_text()) if ref_p.exists() else {}
    thr = ref.get("thresholds") or {"decode_pct": 2.0, "prefill_pct": 3.0}
    fails = [f"hook proof: {p}" for p in hook_problems(log_text, profile)]
    g_ref = ref_dir / f"quick-{profile}-greedy.json"
    unrec = unrecorded_problems(ref, g_ref.exists()) if release and not recording else []
    if unrec:  # nothing to compare against: do not spend GPU time on probes
        return {"profile": profile, "measured": {}, "rows": [], "unrecorded": unrec, "failures": fails + unrec}, fails + unrec
    m = measure(base, model, release, gsm8k_n)
    rows = verdict(m, ref, thr)
    fails += [f"{n}: {d}" for n, ok, d in rows if not ok]
    if release:  # gate 3: aggregate tok/s rises with the streams (the c2/c4/c8 floors above are the ratios)
        # floor 1.05, not the BetterBench 1.2: this probe (8 x 128 tokens) measured 1.14 / 1.20 on the good Inkling image, 0.68 on the collapsed one
        fails += [f"concurrency: {p}" for p in concurrency_problems(m["concurrency"], two_stream_min=QUICK_TWO_STREAM_MIN)]
    allow_p = ref_dir / "greedy_allow.json"
    allow = (json.loads(allow_p.read_text()) if allow_p.exists() else {}).get(profile, {})
    if g_ref.exists():
        fails += [f"greedy: {p}" for p in greedy_problems(m["greedy"], json.loads(g_ref.read_text()), allow)]
    else:
        print(f"note: no greedy reference {g_ref.name} yet -- run with --record", file=sys.stderr)
    return {"profile": profile, "measured": m, "rows": rows, "unrecorded": [], "failures": fails}, fails


def record(profile, report, ref_dir=REFS):
    m = report["measured"]
    old_p = ref_dir / f"quick-{profile}.json"
    old = json.loads(old_p.read_text()) if old_p.exists() else {}
    ref = {"decode_tps": m["decode_tps"], "prefill_pps": m["prefill_pps"],
           "tok_per_update": {"code": round(m["tok_per_update"]["code"] * 0.95, 2)},
           "thresholds": {"decode_pct": 2.0, "prefill_pct": 3.0}}
    for k in ("c2", "c4", "c8"):  # ratio floors at 90 % of the measurement; unmeasured ones keep the seed
        key = f"{k}_over_c1"
        ref[f"{key}_min"] = round(m[key] * 0.9, 2) if key in m else old.get(f"{key}_min")
    ref["gsm8k_correct"] = m.get("gsm8k_correct", old.get("gsm8k_correct"))
    (ref_dir / f"quick-{profile}.json").write_text(json.dumps(ref, indent=2) + "\n")
    (ref_dir / f"quick-{profile}-greedy.json").write_text(json.dumps(m["greedy"], indent=1) + "\n")


# ---------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="quick tier against a live server")
    r.add_argument("--profile", choices=sorted(HOOKS), required=True)
    r.add_argument("--base", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--log", required=True, help="full server log of this start")
    r.add_argument("--out")
    r.add_argument("--record", action="store_true", help="write the measurements as the new reference")
    r.add_argument("--release", action="store_true", help="release tier: + 8 streams, GSM8K-50, fails closed on null references")
    r.add_argument("--gsm8k-n", type=int, default=50)
    s_ = sub.add_parser("release-check", help="both release-tier statuses of a commit are green (stdin: JSON {context: state})")
    g = sub.add_parser("gsm8k", help="GSM8K-n check only (c1, greedy): prints {correct, items} as JSON, for recording/reproducibility")
    g.add_argument("--base", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--n", type=int, default=50)
    h = sub.add_parser("hooks", help="hook proof on a log file")
    h.add_argument("log")
    h.add_argument("--profile", choices=sorted(HOOKS), required=True)
    c = sub.add_parser("conc", help="concurrency gate over BetterBench concurrency JSON file(s)")
    c.add_argument("json", nargs="+")
    f = sub.add_parser("floor", help="drafter floor over a BetterBench decode JSON")
    f.add_argument("json")
    f.add_argument("--floor", type=float, required=True)
    f.add_argument("--category", default="code")
    a = ap.parse_args(argv)

    if a.cmd == "gsm8k":
        items = gsm8k_items(a.base.rstrip("/"), a.model, a.n)
        print(json.dumps({"correct": sum(items), "items": items}))
        return 0
    if a.cmd == "release-check":
        probs = release_gate_problems(json.load(sys.stdin))
    elif a.cmd == "hooks":
        probs = hook_problems(Path(a.log).read_text(errors="replace"), a.profile)
    elif a.cmd == "conc":
        agg = {int(e["level"]): e["aggregate_tps"] for p in a.json for e in json.loads(Path(p).read_text()).get("concurrency", [])}
        probs = concurrency_problems(agg)
    elif a.cmd == "floor":
        probs = floor_problems(category_tok_per_update(json.loads(Path(a.json).read_text())), a.floor, a.category)
    else:
        report, probs = run_quick(a.profile, a.base.rstrip("/"), a.model, Path(a.log).read_text(errors="replace"),
                                  release=a.release, gsm8k_n=a.gsm8k_n, recording=a.record)
        for n, ok, d in report["rows"]:
            print(f"{'ok  ' if ok else 'FAIL'} {n}: {d}")
        if a.out:
            Path(a.out).write_text(json.dumps(report, indent=1))
        if a.record:
            record(a.profile, report)
            print("reference written to", REFS)
            probs = []
    for p in probs:
        print(f"::error::{p}")
    print("PASS" if not probs else f"FAIL ({len(probs)})")
    return 1 if probs else 0


if __name__ == "__main__":
    sys.exit(main())
