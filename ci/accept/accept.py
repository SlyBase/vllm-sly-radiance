#!/usr/bin/env python3
"""GPU acceptance gate for a candidate vllm-sly-radiance image.

Runs on the CPU-only CI LXC and drives the single R9700 in LXC 2408 through the
restricted `gpu-window` helper over SSH (Betriebsregel Nr. 1: never two GPU
services at once, always restore production afterwards).

    python3 ci/accept/accept.py \
        --profile ci/accept/profiles/vllm7-mxfp4.json \
        --image vllm-sly-radiance:0.2.5-rocm10.0 \
        --mode fast \
        --out out/accept-0.2.5

Phases
  1. acquire  -- take the GPU lock; refuses when production is serving traffic
  2. start    -- stop vllm7, start the candidate (double start), read KV pool
  3. logs     -- required / forbidden `[radiance]` markers
  4. smoke    -- four chat completions (arithmetic, reasoning, tool call, needle)
  5. bench    -- BetterBench concurrency sweep + speculative-decoding counters
  6. gsm8k    -- accuracy, full mode only
  7. release  -- always: restore docker-vllm7.service and verify /health == 200

Every phase result is compared against `--baseline`; the gate exits non-zero on
the first hard failure but still runs the release phase.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent

# The needle prompt is generated, not stored: ~100k tokens of filler with one
# marker sentence, so the long-prefill path (and the mamba-align fix) is exercised.
NEEDLE_FILLER = (
    "The maintenance log for rack {i} records nominal temperatures, stable fan "
    "curves and no error counters worth reporting. "
)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# gpu-window helper (SSH)
# --------------------------------------------------------------------------- #
class GpuWindow:
    """Thin client for /usr/local/sbin/gpu-window in LXC 2408."""

    def __init__(self, target: str, key: str | None, owner: str, ttl: int, dry_run: bool = False):
        self.target = target
        self.key = key
        self.owner = owner
        self.ttl = ttl
        self.dry_run = dry_run
        self.held = False

    def _ssh(self, *args: str, timeout: int = 900) -> str:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
        if self.key:
            cmd += ["-i", self.key]
        cmd += [self.target, " ".join(shlex.quote(a) for a in args)]
        if self.dry_run:
            log(f"DRY-RUN ssh: {' '.join(args)}")
            return "{}"
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(
                f"gpu-window {args[0]} failed (rc={proc.returncode}): "
                f"{proc.stderr.strip() or proc.stdout.strip()}"
            )
        return proc.stdout

    def _json(self, *args: str, timeout: int = 900) -> dict:
        out = self._ssh(*args, timeout=timeout)
        try:
            return json.loads(out.strip().splitlines()[-1]) if out.strip() else {}
        except json.JSONDecodeError as exc:  # helper must stay machine readable
            raise RuntimeError(f"gpu-window {args[0]} returned non-JSON: {out[:400]}") from exc

    def status(self) -> dict:
        return self._json("status", timeout=120)

    def acquire(self, force: bool = False) -> dict:
        args = ["acquire", self.owner, str(self.ttl)] + (["force"] if force else [])
        res = self._json(*args, timeout=180)
        self.held = True
        return res

    def start(self, image: str, service: str) -> dict:
        return self._json("start", image, service, timeout=2400)

    def logs(self, lines: int = 4000) -> str:
        return self._ssh("logs", str(lines), timeout=300)

    def release(self) -> dict:
        try:
            return self._json("release", timeout=1800)
        finally:
            self.held = False


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #
def http_get(url: str, timeout: int = 30) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001 - connection refused etc.
        return 0, str(exc)


def chat(base_url: str, model: str, payload: dict, timeout: int = 900) -> dict:
    body = json.dumps({"model": model, **payload}).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


PROM_RE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-\d.eE+]+)$")


def scrape_metrics(base_url: str) -> dict[str, float]:
    """Sum every sample of the counters/histograms the gate cares about."""
    status, text = http_get(f"{base_url}/metrics", timeout=60)
    if status != 200:
        raise RuntimeError(f"/metrics returned {status}")
    wanted = (
        "vllm:spec_decode_num_accepted_tokens_total",
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_drafts_total",
        "vllm:iteration_tokens_total_sum",
        "vllm:iteration_tokens_total_count",
        "vllm:num_preemptions_total",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
    )
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = PROM_RE.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        if name in wanted:
            out[name] = out.get(name, 0.0) + float(m.group("value"))
    return out


def spec_delta(before: dict, after: dict) -> dict:
    def d(key: str) -> float:
        return after.get(key, 0.0) - before.get(key, 0.0)

    steps = d("vllm:iteration_tokens_total_count")
    tokens = d("vllm:iteration_tokens_total_sum")
    drafts = d("vllm:spec_decode_num_draft_tokens_total")
    accepted = d("vllm:spec_decode_num_accepted_tokens_total")
    return {
        "steps": steps,
        "iteration_tokens": tokens,
        "mean_tokens_per_step": round(tokens / steps, 4) if steps else 0.0,
        "acceptance_rate": round(accepted / drafts, 4) if drafts else 0.0,
        "preemptions": d("vllm:num_preemptions_total"),
    }


# --------------------------------------------------------------------------- #
# Phase 3: logs
# --------------------------------------------------------------------------- #
def check_logs(text: str, profile: dict) -> dict:
    missing, hits = [], {}
    for name, pattern in profile.get("required_log_patterns", []):
        m = re.search(pattern, text, re.MULTILINE)
        if m:
            hits[name] = m.group(0)[:200]
        else:
            missing.append(name)

    lines = text.splitlines()
    starts = []  # offset of each line, to map a match back to its line number
    off = 0
    for ln in lines:
        starts.append(off)
        off += len(ln) + 1
    known = profile.get("known_warnings", [])

    def line_index(pos: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    forbidden = []
    for name, pattern in profile.get("forbidden_log_patterns", []):
        for m in re.finditer(pattern, text, re.MULTILINE):
            idx = line_index(m.start())
            # A traceback's own line rarely names the cause -- judge it by its
            # surroundings, so the known benign ones stay suppressed.
            context = "\n".join(lines[max(0, idx - 3): idx + 4])
            if any(w in context for w in known):
                continue
            forbidden.append({"name": name, "line": lines[idx].strip()[:300]})
            break

    kv = None
    for m in re.finditer(r"GPU KV cache size:\s*([\d,._]+)\s*tokens", text):
        kv = int(re.sub(r"[,._]", "", m.group(1)))
    return {"ok": not missing and not forbidden, "missing": missing,
            "forbidden": forbidden, "hits": hits, "kv_tokens_from_log": kv}


# --------------------------------------------------------------------------- #
# Phase 4: smoke
# --------------------------------------------------------------------------- #
WEATHER_TOOL = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    },
}]


def build_needle(marker: str, filler_sentences: int) -> str:
    parts = [NEEDLE_FILLER.format(i=i) for i in range(filler_sentences)]
    parts.insert(len(parts) // 2, f"The service code for the north wing is {marker}. ")
    parts.append(
        "\n\nQuestion: what is the service code for the north wing? "
        "Answer with the code only."
    )
    return "".join(parts)


def run_smoke(base_url: str, model: str, profile: dict, dry_run: bool = False) -> dict:
    if dry_run:
        log("DRY-RUN: skipping smoke requests")
        return {"ok": True, "cases": [], "skipped": "dry-run"}
    results = []
    for case in profile.get("smoke", []):
        name = case["name"]
        payload: dict = {"temperature": 0, "max_tokens": case.get("max_tokens", 512)}
        if case.get("prompt_needle"):
            cfg = case["prompt_needle"]
            prompt = build_needle(cfg["marker"], cfg["filler_sentences"])
        else:
            prompt = case["prompt"]
        payload["messages"] = [{"role": "user", "content": prompt}]
        if case.get("expect_tool_call"):
            payload["tools"] = WEATHER_TOOL
            payload["tool_choice"] = "auto"
        t0 = time.time()
        try:
            resp = chat(base_url, model, payload, timeout=case.get("timeout", 900))
        except Exception as exc:  # noqa: BLE001
            results.append({"name": name, "ok": False, "error": str(exc)[:300]})
            continue
        msg = resp["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        calls = [c["function"]["name"] for c in (msg.get("tool_calls") or [])]
        ok = True
        # The reasoning parser must swallow the think block in every answer -- production
        # emits an empty <think></think> for easy prompts, so "reasoning_content is
        # non-empty" would be flaky, but a leaked tag is always a broken parser.
        leaked = "<think>" in content or "</think>" in content
        ok = ok and not leaked
        detail = {"reasoning_chars": len(reasoning), "think_tag_leaked": leaked}
        if case.get("expect"):
            ok = ok and case["expect"].lower() in content.lower()
            detail["expect"] = case["expect"]
        if case.get("expect_tool_call"):
            ok = ok and case["expect_tool_call"] in calls
            detail["tool_calls"] = calls
        results.append({
            "name": name, "ok": ok, "elapsed_s": round(time.time() - t0, 1),
            "answer": content[:200], "finish_reason": resp["choices"][0].get("finish_reason"),
            **detail,
        })
    return {"ok": all(r["ok"] for r in results), "cases": results}


# --------------------------------------------------------------------------- #
# Phase 5/6: BetterBench + GSM8K
# --------------------------------------------------------------------------- #
BB_SNIPPET = (
    "import sys;"
    "from betterbench.cli import main;"
    "sys.argv = {argv!r};"
    "sys.exit(main())"
)


def run_betterbench(python: str, cwd: Path, base_url: str, model: str,
                    config: Path, out_file: Path, notes: dict) -> dict:
    argv = ["bb", "run",
            "--endpoint", f"{base_url}/v1",
            "--model", model,
            "--config", str(config),
            "--concurrency",
            "--no-html",
            "--out", str(out_file)]
    for key, value in notes.items():
        argv += ["--note", f"{key}={value}"]
    cmd = [python, "-c", BB_SNIPPET.format(argv=argv)]
    log(f"betterbench: {' '.join(argv[1:])}")
    proc = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, timeout=7200)
    if proc.returncode != 0:
        raise RuntimeError(f"betterbench failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    data = json.loads(out_file.read_text())
    levels = {int(e["level"]): e for e in data.get("concurrency", [])}
    return {
        "out": str(out_file),
        "aggregate_tps": {str(k): round(v["aggregate_tps"], 2) for k, v in sorted(levels.items())},
        "ok_requests": {str(k): [v["ok"], v["requests"]] for k, v in sorted(levels.items())},
    }


def run_gsm8k(python: str, base_url: str, model: str, limit: int, out_dir: Path) -> dict:
    model_args = (f"model={model},base_url={base_url}/v1/chat/completions,"
                  f"num_concurrent=4,max_retries=3,timeout=600")
    cmd = [python, "-m", "lm_eval",
           "--model", "local-chat-completions",
           "--model_args", model_args,
           "--tasks", "gsm8k_cot_zeroshot",
           "--limit", str(limit),
           "--apply_chat_template",
           "--gen_kwargs", "temperature=0,max_gen_toks=4096",
           "--output_path", str(out_dir)]
    log(f"gsm8k: limit={limit}")
    proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10800)
    if proc.returncode != 0:
        raise RuntimeError(f"lm_eval failed (rc={proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    files = sorted(out_dir.rglob("results_*.json"))
    if not files:
        raise RuntimeError("lm_eval produced no results_*.json")
    data = json.loads(files[-1].read_text())
    task = data["results"]["gsm8k_cot_zeroshot"]
    acc = task.get("exact_match,flexible-extract", task.get("exact_match,strict-match"))
    return {"out": str(files[-1]), "exact_match": round(float(acc), 4),
            "strict": task.get("exact_match,strict-match"),
            "flexible": task.get("exact_match,flexible-extract")}


# --------------------------------------------------------------------------- #
# Verdict
# --------------------------------------------------------------------------- #
def compare(results: dict, baseline: dict, thresholds: dict, mode: str) -> list[dict]:
    checks: list[dict] = []

    def add(name, ok, actual, expected, hard=True):
        checks.append({"name": name, "ok": bool(ok), "actual": actual,
                       "expected": expected, "hard": hard})

    startup = results.get("start", {})
    logs = results.get("logs", {})
    add("startup", startup.get("kv_tokens") is not None,
        startup.get("kv_tokens"), "server up, KV pool reported")
    add("log markers", logs.get("ok"),
        {"missing": logs.get("missing"), "forbidden": [f["name"] for f in logs.get("forbidden", [])]},
        "all required markers, no forbidden patterns")

    kv_base = baseline.get("kv_tokens")
    kv = startup.get("kv_tokens")
    if kv_base and kv:
        ratio = kv / kv_base
        add("kv pool", ratio >= thresholds["kv_tokens_min_ratio"],
            f"{kv} ({ratio:.3f}x)", f">= {thresholds['kv_tokens_min_ratio']:.2f}x of {kv_base}")

    s_base = baseline.get("startup_s")
    s_now = startup.get("startup_s")
    if s_base and s_now:
        add("startup time", s_now <= s_base * thresholds["startup_s_max_ratio"],
            f"{s_now:.0f}s", f"<= {thresholds['startup_s_max_ratio']:.1f}x of {s_base:.0f}s",
            hard=False)

    smoke = results.get("smoke", {})
    if smoke.get("skipped"):
        # A dry run never sends a request; reporting "all pass" here would claim
        # a result nobody measured.
        add("smoke", None, f"skipped ({smoke['skipped']})", "4/4 pass", hard=False)
    else:
        add("smoke", smoke.get("ok"),
            [c["name"] for c in smoke.get("cases", []) if not c["ok"]] or "all pass",
            "4/4 pass")

    bb = results.get("betterbench", {}).get("aggregate_tps", {})
    bb_base = baseline.get("aggregate_tps", {})
    for level, value in bb.items():
        base = bb_base.get(level)
        if not base:
            continue
        ratio = value / base
        add(f"throughput conc {level}", ratio >= thresholds["aggregate_tps_min_ratio"],
            f"{value} t/s ({ratio:.3f}x)",
            f">= {thresholds['aggregate_tps_min_ratio']:.2f}x of {base} t/s")

    spec = results.get("spec", {})

    # Draft quality is acceptance_rate, not tokens/step. The first real gate run
    # (0.2.5, 2026-09-17) failed on tokens/step by 0.0008 while acceptance_rate
    # went UP: 24666 iteration tokens over 3016 steps against the baseline's
    # 24705 over 2984. The same work, spread over 1.1 % more engine steps --
    # tokens/step measures how full the batch was per step, which the arrival
    # pattern of 24 BetterBench requests moves around on its own. A metric that
    # drifts with scheduling is a bad hard gate, so it is now a soft sanity
    # check with a tolerance that reflects the observed spread.
    acc_base = baseline.get("acceptance_rate")
    if acc_base and spec.get("acceptance_rate"):
        delta = spec["acceptance_rate"] - acc_base
        add("acceptance rate", delta >= thresholds["acceptance_rate_min_delta"],
            f"{spec['acceptance_rate']} ({delta:+.4f})",
            f">= {acc_base} {thresholds['acceptance_rate_min_delta']:+.3f}")
    tps_base = baseline.get("mean_tokens_per_step")
    if tps_base and spec.get("mean_tokens_per_step"):
        delta = spec["mean_tokens_per_step"] - tps_base
        add("tokens/step", delta >= thresholds["mean_tokens_per_step_min_delta"],
            f"{spec['mean_tokens_per_step']} ({delta:+.3f})",
            f">= {tps_base} {thresholds['mean_tokens_per_step_min_delta']:+.2f}",
            hard=False)
    if spec:
        add("preemptions", spec.get("preemptions", 0) <= thresholds["preemptions_max"],
            spec.get("preemptions"), f"<= {thresholds['preemptions_max']}")

    if mode == "full":
        g = results.get("gsm8k", {})
        g_base = baseline.get("gsm8k")
        if g and g_base:
            delta = g["exact_match"] - g_base
            add("gsm8k", delta >= thresholds["gsm8k_min_delta"],
                f"{g['exact_match']} ({delta:+.3f})",
                f">= {g_base} {thresholds['gsm8k_min_delta']:+.2f}")

    return checks


def render_markdown(meta: dict, checks: list[dict], results: dict,
                    hard_fail: list[str]) -> str:
    # The verdict is decided once, in main(), and handed in here. Deriving it a
    # second time from `checks` alone was wrong for exactly the case that
    # matters: a run that aborts before BetterBench never produces those checks,
    # so "no failed check" read as PASS.
    verdict = "FAIL" if hard_fail else "PASS"
    soft = [c for c in checks if not c["ok"] and not c["hard"]]
    lines = [
        f"## Acceptance gate: **{verdict}**",
        "",
        f"- image: `{meta['image']}` &middot; profile: `{meta['profile']}` &middot; mode: `{meta['mode']}`",
        f"- baseline: `{meta['baseline']}`",
        f"- duration: {meta['duration_s'] / 60:.0f} min &middot; run: {meta['started']}",
        "",
        "| check | result | actual | expected |",
        "| --- | --- | --- | --- |",
    ]
    for c in checks:
        icon = "✅" if c["ok"] else ("⚠️" if not c["hard"] else "❌")
        lines.append(f"| {c['name']} | {icon} | {c['actual']} | {c['expected']} |")
    if soft:
        lines += ["", f"_{len(soft)} soft check(s) missed -- not blocking._"]
    if results.get("error"):
        lines += ["", f"**The run aborted: `{results['error']}`** -- every check below "
                      "the abort never ran, so this is not a clean bill of health."]
    bb = results.get("betterbench", {}).get("aggregate_tps")
    if bb:
        lines += ["", "BetterBench aggregate t/s: " +
                  ", ".join(f"conc {k} = {v}" for k, v in bb.items())]
    rel = results.get("release", {})
    lines += ["", f"Production restored: `{rel.get('service', 'docker-vllm7.service')}` "
                  f"health={rel.get('health')} kv={rel.get('kv_tokens')}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--image", required=True, help="candidate image tag, e.g. vllm-sly-radiance:0.2.5-rocm10.0")
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--baseline", help="defaults to ci/accept/baselines/<profile name>.json")
    ap.add_argument("--out", default="out/accept", help="result directory")
    ap.add_argument("--host", default=os.environ.get("ACCEPT_HOST", "192.168.178.51"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("ACCEPT_PORT", "8000")))
    ap.add_argument("--ssh-target", default=os.environ.get("ACCEPT_SSH", "gpuwin@192.168.178.51"))
    ap.add_argument("--ssh-key", default=os.environ.get("ACCEPT_SSH_KEY"))
    ap.add_argument("--owner", default=os.environ.get("GITHUB_RUN_ID", "manual"))
    ap.add_argument("--ttl", type=int, default=int(os.environ.get("ACCEPT_TTL", "10800")))
    ap.add_argument("--force", action="store_true", help="take the GPU even if production saw traffic")
    ap.add_argument("--bb-python", default=os.environ.get("ACCEPT_BB_PYTHON", "/opt/accept/venv/bin/python"))
    ap.add_argument("--bb-dir", default=os.environ.get("ACCEPT_BB_DIR", "/opt/accept/betterbench"))
    ap.add_argument("--eval-python", default=os.environ.get("ACCEPT_EVAL_PYTHON", "/opt/accept/venv/bin/python"))
    ap.add_argument("--gsm8k-limit", type=int, default=200)
    ap.add_argument("--record-baseline", action="store_true",
                    help="write the measured values back into the baseline file "
                         "(use once against the current production image to calibrate)")
    ap.add_argument("--dry-run", action="store_true", help="print the gpu-window calls, run nothing")
    args = ap.parse_args()

    profile_path = Path(args.profile)
    profile = json.loads(profile_path.read_text())
    name = profile_path.stem
    baseline_path = Path(args.baseline) if args.baseline else ROOT / "baselines" / f"{name}.json"
    baseline_file = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    # Mode-specific numbers (throughput depends on concurrency_requests) override
    # the mode-independent ones (KV pool, GSM8K, tokens/step).
    baseline = {k: v for k, v in baseline_file.items() if k != "modes"}
    baseline.update(baseline_file.get("modes", {}).get(args.mode, {}))
    thresholds = {**profile["thresholds"], **baseline.get("thresholds", {})}
    config = ROOT / "configs" / f"{args.mode}.json"

    # Absolute, because run_betterbench() hands --out straight to a subprocess
    # that runs with cwd=--bb-dir (/opt/accept/betterbench). A relative --out
    # made BetterBench write its results under the benchmark checkout while
    # accept.py looked for them under its own cwd -- the benchmark ran fine and
    # the run died afterwards on a missing file.
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://{args.host}:{args.port}"
    started = datetime.now(timezone.utc)
    t0 = time.time()

    results: dict = {}
    failures: list[str] = []
    win = GpuWindow(args.ssh_target, args.ssh_key, args.owner, args.ttl, args.dry_run)

    try:
        log(f"status: {json.dumps(win.status())}")
        log(f"acquiring GPU window (owner={args.owner}, ttl={args.ttl}s)")
        results["acquire"] = win.acquire(force=args.force)

        log(f"starting candidate {args.image} ({profile['service']})")
        results["start"] = win.start(args.image, profile["service"])
        log(f"candidate up: {json.dumps(results['start'])}")

        log("checking logs")
        results["logs"] = check_logs(win.logs(), profile)
        if not results["logs"]["ok"]:
            failures.append("logs")
            log(f"log check failed: missing={results['logs']['missing']} "
                f"forbidden={[f['name'] for f in results['logs']['forbidden']]}")

        model = profile["model"]
        log("smoke tests")
        results["smoke"] = run_smoke(base_url, model, profile, args.dry_run)
        if not results["smoke"]["ok"]:
            failures.append("smoke")

        if args.dry_run:
            log("DRY-RUN: skipping betterbench and gsm8k")
        else:
            before = scrape_metrics(base_url)
            log(f"betterbench ({args.mode})")
            results["betterbench"] = run_betterbench(
                args.bb_python, Path(args.bb_dir), base_url, model, config,
                out_dir / f"betterbench-{args.mode}.json",
                {"image": args.image, "profile": name, "mode": args.mode, "run": args.owner},
            )
            after = scrape_metrics(base_url)
            results["spec"] = spec_delta(before, after)
            log(f"spec: {json.dumps(results['spec'])}")

            if args.mode == "full":
                results["gsm8k"] = run_gsm8k(args.eval_python, base_url, model,
                                             args.gsm8k_limit, out_dir / "lmeval")
                log(f"gsm8k: {results['gsm8k']['exact_match']}")

    except Exception as exc:  # noqa: BLE001 - every failure still releases the GPU
        log(f"ERROR: {exc}")
        results["error"] = str(exc)[:2000]
        failures.append("error")
    finally:
        if win.held or args.dry_run:
            log("releasing GPU window, restoring production")
            try:
                results["release"] = win.release()
                log(f"release: {json.dumps(results['release'])}")
            except Exception as exc:  # noqa: BLE001
                results["release"] = {"error": str(exc)[:500]}
                failures.append("release")
                log(f"RELEASE FAILED: {exc} -- production may be down, check LXC 2408")

    checks = compare(results, baseline, thresholds, args.mode)
    hard_fail = [c["name"] for c in checks if not c["ok"] and c["hard"]]
    # `failures` carries the aborts that happen outside the check table -- the
    # run raised before it finished measuring, or the release failed. It was
    # built and then dropped: hard_fail was rebuilt from `checks` alone, and
    # the checks a crashed run never reached simply do not exist, so the gate
    # reported PASS with a green table. A run that did not complete is not a
    # pass.
    hard_fail += [f for f in failures if f not in hard_fail]
    release_ok = results.get("release", {}).get("health") == 200
    if not release_ok:
        hard_fail.append("production restore")

    meta = {
        "image": args.image, "profile": name, "mode": args.mode,
        "baseline": str(baseline_path), "started": started.strftime("%Y-%m-%d %H:%M:%SZ"),
        "duration_s": round(time.time() - t0, 1), "owner": args.owner,
    }
    report = {"meta": meta, "verdict": "FAIL" if hard_fail else "PASS",
              "failed_checks": hard_fail, "checks": checks, "results": results}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    # A dry run measures nothing, so it must not touch the baseline. Without
    # this guard it still stamped `recorded` into the file -- a calibration
    # that never happened -- and the `aggregate_tps` assignment below would
    # have replaced real numbers with `{}`. Same class of bug as "dry run must
    # not report smoke as passed".
    if args.record_baseline and args.dry_run:
        log("DRY-RUN: not writing the baseline")
    elif args.record_baseline and "error" not in results:
        new = dict(baseline_file)
        new["recorded"] = {"image": args.image, "at": meta["started"], "mode": args.mode}
        if results.get("start", {}).get("kv_tokens"):
            new["kv_tokens"] = results["start"]["kv_tokens"]
            new["startup_s"] = results["start"].get("startup_s")
        if results.get("spec", {}).get("mean_tokens_per_step"):
            new["mean_tokens_per_step"] = results["spec"]["mean_tokens_per_step"]
            new["acceptance_rate"] = results["spec"].get("acceptance_rate")
        if results.get("gsm8k"):
            new["gsm8k"] = results["gsm8k"]["exact_match"]
        # Only overwrite this mode's throughput when the benchmark actually
        # produced numbers: `fast` and `full` use different concurrency levels,
        # and an empty result here would silently blank the other mode's
        # reference values.
        measured_tps = results.get("betterbench", {}).get("aggregate_tps") or {}
        if measured_tps:
            modes = dict(new.get("modes", {}))
            modes[args.mode] = {**modes.get(args.mode, {}), "aggregate_tps": measured_tps}
            new["modes"] = modes
        else:
            log(f"no betterbench numbers -- keeping the existing '{args.mode}' aggregate_tps")
        baseline_path.write_text(json.dumps(new, indent=2) + "\n")
        log(f"baseline updated: {baseline_path}")

    md = render_markdown(meta, checks, results, hard_fail)
    (out_dir / "report.md").write_text(md + "\n")
    print("\n" + md + "\n")

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(md + "\n")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())
