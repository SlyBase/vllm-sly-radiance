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
  5. chat     -- multi-turn decode at growing context (the gate used to measure
                 decode only on 60-120 token prompts; a spec-depth or acceptance
                 change can lose half its throughput on a real chat and still
                 win the weighted median, see the 1.9.0 k=15 case)
  6. bench    -- BetterBench sweep; per CATEGORY decode medians gate on their own
                 (`category_decode_tps_min_ratio`), the weighted median does not
                 hide a losing category any more
  7. gsm8k    -- accuracy, full mode only
  8. release  -- always: restore docker-vllm7.service and verify /health == 200

Every phase result is compared against `--baseline`; the gate exits non-zero on
the first hard failure but still runs the release phase.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
sys.path.insert(0, str(REPO / "ci"))
import quick_bench  # noqa: E402  (regression gates shared with the quick tier)

# The needle prompt is generated, not stored: ~100k tokens of filler with one
# marker sentence, so the long-prefill path (and the mamba-align fix) is exercised.
NEEDLE_FILLER = (
    "The maintenance log for rack {i} records nominal temperatures, stable fan "
    "curves and no error counters worth reporting. "
)

# gpu-window refuses while production is busy; instead of failing the gate on
# the first refusal, wait and retry. `force` stays an explicit opt-in (CLI flag
# / dispatch input) -- the retry never escalates to it.
BUSY_HINT = "retry later or use force"


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# gpu-window helper (SSH)
# --------------------------------------------------------------------------- #
# Pinned host key of the helper host (LXC 2408); the runner user has no known_hosts of its own.
KNOWN_HOSTS = Path(__file__).resolve().parent / "known_hosts"


class GpuWindow:
    """Thin client for /usr/local/sbin/gpu-window in LXC 2408."""

    def __init__(self, target: str, key: str | None, owner: str, ttl: int,
                 dry_run: bool = False, load_format: str = "auto"):
        self.target = target
        self.key = key
        self.owner = owner
        self.ttl = ttl
        self.dry_run = dry_run
        self.load_format = load_format
        self.held = False

    def _ssh(self, *args: str, timeout: int = 900) -> str:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
               "-o", f"UserKnownHostsFile={KNOWN_HOSTS}"]
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
        return self._json("start", image, service, self.load_format, timeout=2400)

    def logs(self, lines: int = 999_999) -> str:
        # Full startup log, not a tail window: start_once() recreates the candidate,
        # so everything it has logged IS the startup. vLLM 0.34 emits ~8000 lines
        # before the first request (HIPCachingAllocator capture warnings); the old
        # --tail 4000 cut the install banners (lines ~100-200) and failed every gate
        # since the 0.34 bump while all functional checks were green (1.13.0-1.13.4).
        # 999999 = the max the gpu-window-ssh trust boundary accepts (is_num is
        # ^[0-9]{1,6}$); ~100x the startup volume, effectively the full log.
        return self._ssh("logs", str(lines), timeout=300)

    def release(self) -> dict:
        try:
            return self._json("release", timeout=1800)
        finally:
            self.held = False


def inkling_docker_argv(image: str, name: str, port: int, model_dir: str, drafter_dir: str,
                        cache_dir: str, extra_env: list[str] | None = None) -> list[str]:
    """`docker run` of the Inkling release arm with NO model env (the image defaults are under test); the hand-run twin is
    start_inkling() in ci/quick_backtest.sh. One `-e` per variable."""
    env = ["HIP_VISIBLE_DEVICES=0", "GPU_MAX_HW_QUEUES=1", "HF_HUB_OFFLINE=1", *(extra_env or [])]
    argv = ["docker", "run", "-d", "--name", name, "--device=/dev/kfd", "--device=/dev/dri",
            "--group-add", "video", "--security-opt", "label=disable", "--ipc=host"]
    for kv in env:
        argv += ["-e", kv]
    argv += ["-p", f"{port}:8000", "-v", f"{model_dir}:/model:ro", "-v", f"{drafter_dir}:/models/dflash2:ro",
             "-v", f"{cache_dir}/triton:/root/.triton", image, "/model",
             "--served-model-name", "inkling2-mini", "--gpu-memory-utilization", "0.92",
             "--limit-mm-per-prompt", '{"image":0}', "--reasoning-parser", "inkling",
             "--enable-auto-tool-choice", "--tool-call-parser", "inkling", "--max-model-len", "32768",
             "--max-num-seqs", "8", "--no-enable-prefix-caching", "--no-async-scheduling",
             "--speculative-config",
             '{"method":"dflash","model":"/models/dflash2","num_speculative_tokens":7,'
             '"draft_sample_method":"probabilistic","attention_backend":"TRITON_ATTN","kv_cache_dtype":"auto"}']
    return argv


def qwen_docker_argv(image: str, name: str, port: int, model_dir: str,
                     drafter_dir: str | None, cache_dir: str,
                     extra_env: list[str] | None = None) -> list[str]:
    """`docker run` of the Qwen release arm: pinned argv verbatim from the
    homelab swift-mxfp4 profile (lxc_vllm7.yml + docker-vllm7.service.j2) and
    the 1.13.4 reference arm on host 51 (/root/qfix/out/QP2/docker_cmd.txt).
    One `-e` per variable; `drafter_dir` is unused -- Qwen serves one model.
    ci/quick_backtest.sh start_qwen() is its hand-run twin and is covered by
    ci/tests/test_qwen_argv.py's drift test.
    """
    env = ["HIP_VISIBLE_DEVICES=0", "HF_HUB_OFFLINE=1",
           "GPU_MAX_HW_QUEUES=1", "RADIANCE_MXFP4=1",
           "RADIANCE_MXFP4_W4A8=1", "RADIANCE_MXFP4_W4A8_MIN_M=0",
           "RADIANCE_MXFP4_DECODE_MAX_M=128", "RADIANCE_MXFP4_SANITIZE=0",
           "RADIANCE_LMHEAD_FP8=1", "RADIANCE_LMHEAD_INT4=1",
           "RADIANCE_FUSED_NORM_QUANT=1", "RADIANCE_KV_GROUP_SIZE=8",
           "RADIANCE_EMBED_INT8=1", "RADIANCE_EMBED_BITS=4",
           "RADIANCE_MXFP4_A_TILED_MIN_M=513", "RADIANCE_MXFP4_TN4_MIN_M=513",
           "RADIANCE_FUSED_NQ_REG=1", "RADIANCE_MXFP4_AT_SKIP_PAD=1",
           "RADIANCE_MXFP4_TILED_OPROJ=1", "R4D_ATTN_FP8=3",
           "RADIANCE_MXFP4_WPERM=1", "RADIANCE_MXFP4_WIDE_MAX_M=192",
           "RADIANCE_ADAPTIVE_WIDTH=perseq", "RADIANCE_CACTUS_DELTA=0.5",
           "RADIANCE_MXFP4_BCQ=/models/swift-mxfp4/bcq_tables.pt",
           "RADIANCE_LMHEAD_INT4_CODES=/models/swift-mxfp4/lm_head_int4.pt",
           *(extra_env or [])]
    argv = ["docker", "run", "-d", "--name", name, "--device=/dev/kfd",
            "--device=/dev/dri", "--group-add", "44", "--group-add", "993",
            "--security-opt", "seccomp=unconfined", "--ipc=host"]
    for kv in env:
        argv += ["-e", kv]
    argv += ["-p", f"{port}:8000",
             "-v", "/root/hf-cache:/root/.cache/huggingface",
             "-v", f"{model_dir}:/models/swift-mxfp4:ro",
             "-v", f"{cache_dir}/triton:/root/.triton",
             "-v", f"{cache_dir}/torch_compile:/root/.cache/vllm/torch_compile_cache",
             "-v", f"{cache_dir}/aiter:/root/.aiter",
             image,
             "--model", "/models/swift-mxfp4",
             "--served-model-name", "slybase/Swift-1.5-Qwen3.8-27B-MXFP4-GPTQ",
             "amd/Qwen3.8-27B-Quark-AWQ-MXFP4",
             "slybase/ThinkingCap-Qwen3.8-27B-MXFP4",
             "bottlecapai/ThinkingCap-Qwen3.8-27B-NVFP4",
             "--max-model-len", "262144",
             "--gpu-memory-utilization", "0.98",
             "--kv-cache-dtype", "fp8",
             "--port", "8000",
             "--speculative-config.method", "dflash",
             "--speculative-config.model", "syvai/Qwen3.8-27B-DFlash2-W4A16",
             "--speculative-config.num_speculative_tokens", "7",
             "--speculative-config.draft_sample_method", "probabilistic",
             "--speculative-config.attention_backend", "TRITON_ATTN",
             "--override-generation-config",
             '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}',
             "--max-num-seqs", "8",
             "--max-num-batched-tokens", "4096",
             "--mamba-ssm-cache-dtype", "bfloat16",
             "--compilation-config.cudagraph_mode", "FULL_DECODE_ONLY",
             "--compilation-config.cudagraph_capture_sizes",
             "[8,16,24,32,40,48,56,64]",
             "--skip-mm-profiling", "--enable-mm-embeds",
             "--limit-mm-per-prompt.image", "0",
             "--limit-mm-per-prompt.video", "0",
             "--attention-backend", "R4D_HYBRID",
             "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",
             "--reasoning-parser", "qwen3",
             "--chat-template", "/opt/qwen-fixed.jinja",
             "--default-chat-template-kwargs",
             '{"reasoning_effort": "medium"}',
             "--enable-prefix-caching", "--no-enable-extensible-kv-cache",
             "--kv-cache-memory-bytes", "15569256448"]
    return argv


class LocalDockerWindow(GpuWindow):
    """A gpu-window whose candidate is a container we start ourselves (no service unit in the helper).

    Runs on the GPU host (the runner needs docker and may stop/start the prod unit). Contract:
        acquire (lock, via the helper) -> systemctl stop <prod> -> docker run -> health ...
        docker rm -f -> systemctl start <prod> -> release (helper: verifies /health)
    The runner is not root: stop/start go through `sudo -n` (sudoers rule of the homelab playbook
    lxc-gpu-window.yml).
    `cleanup()` is idempotent and runs from release(), from the atexit hook and from SIGTERM/SIGINT (a
    cancelled GitHub job), so the container is removed and prod is restored on every exit path we control.
    A SIGKILL is covered by gpu-window-guard.timer (restores prod when the TTL runs out).
    """

    def __init__(self, target, key, owner, ttl, dry_run=False, load_format="auto", *, run=None, ssh=None,
                 health=None, sleep=time.sleep, prod_service="docker-vllm7.service", port=8001,
                 container="vllm-accept-inkling", model_dir=None, drafter_dir=None, cache_dir=None,
                 docker_argv=None, health_wait_s=1200, install_signals=True, extra_env=None):
        super().__init__(target, key, owner, ttl, dry_run, load_format)
        self._run = run or self._subprocess_run
        if ssh:
            self._ssh = lambda *a, timeout=900: ssh(*a)
        self._health = health or (lambda url: http_get(url, timeout=10)[0])
        self._sleep, self.prod, self.port, self.container = sleep, prod_service, port, container
        self.model_dir = model_dir or os.environ.get("ACCEPT_INKLING_MODEL", "/root/inkling2-mini")
        self.drafter_dir = drafter_dir or os.environ.get("ACCEPT_INKLING_DRAFTER", "/root/dflash2-ink/model")
        self.cache_dir = cache_dir or os.environ.get("ACCEPT_INKLING_CACHE", "/root/accept-inkling-cache")
        self._docker_argv = docker_argv or inkling_docker_argv
        self.health_wait_s, self.extra_env = health_wait_s, extra_env
        self._prod_stopped = self._started = self._cleaned = False
        if install_signals:
            self.install_signal_handlers()
            atexit.register(self.cleanup)

    @staticmethod
    def install_signal_handlers() -> None:
        def bye(signum, frame):  # noqa: ARG001 - raise so every `finally` / atexit runs
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, bye)
        signal.signal(signal.SIGINT, bye)

    @staticmethod
    def _subprocess_run(argv, check=True, timeout=300, merge_stderr=False, **kw):  # noqa: ARG004
        proc = subprocess.run(argv, stdout=subprocess.PIPE, text=True, timeout=timeout,
                              stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE)
        if check and proc.returncode != 0:
            raise RuntimeError(f"{' '.join(argv[:3])} failed (rc={proc.returncode}): "
                               f"{(proc.stderr or proc.stdout).strip()[:400]}")
        return proc.stdout

    def start(self, image: str, service: str) -> dict:  # noqa: ARG002 - `service` is the profile's, unused
        t0 = time.time()
        self._cleaned = False
        # flag first: a stop that fails half way must still be answered by a start in cleanup()
        self._prod_stopped = True
        self._run(["sudo", "-n", "systemctl", "stop", self.prod], timeout=600)
        self._run(["docker", "rm", "-f", self.container], check=False)
        self._started = True
        self._run(self._docker_argv(image, self.container, self.port, self.model_dir,
                                    self.drafter_dir, self.cache_dir, self.extra_env))
        url = f"http://127.0.0.1:{self.port}/health"
        waited = 0
        while self._health(url) != 200:
            if self._run(["docker", "inspect", "-f", "{{.State.Running}}", self.container],
                         check=False).strip() == "false":
                raise RuntimeError("candidate container died during startup")
            if waited >= self.health_wait_s:
                raise RuntimeError(f"candidate not healthy after {self.health_wait_s}s")
            self._sleep(2)
            waited += 2
        text = self.logs()
        m = re.findall(r"GPU KV cache size:\s*([\d,._]+)\s*tokens", text)
        return {"container": self.container, "image": image, "startup_s": round(time.time() - t0, 1),
                "kv_tokens": int(re.sub(r"[,._]", "", m[-1])) if m else None}

    def logs(self, lines: int = 999_999) -> str:  # noqa: ARG002
        # merge_stderr: the radiance hook lines are stderr writes and docker logs keeps the streams apart
        return self._run(["docker", "logs", self.container], check=False, merge_stderr=True)

    def cleanup(self) -> None:
        if self._cleaned:
            return
        self._cleaned = True
        errors = []
        if self._started or self._prod_stopped:
            try:
                self._run(["docker", "rm", "-f", self.container], check=False)
            except Exception as exc:  # noqa: BLE001 - restoring prod must still happen
                errors.append(str(exc))
        if self._prod_stopped:
            try:
                self._run(["sudo", "-n", "systemctl", "start", self.prod], timeout=600)
                self._prod_stopped = False
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
        self._started = False
        if errors:
            raise RuntimeError("; ".join(errors))

    def release(self) -> dict:
        err = None
        try:
            self.cleanup()
        except Exception as exc:  # noqa: BLE001 - the helper's release (guard, health check) still runs
            err = exc
        res = super().release()
        if err:
            raise err
        return res


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


def served_models(base_url: str, timeout: int = 30) -> list[str]:
    """IDs the container lists at /v1/models. The gate fails fast when the
    profile's model is not among them -- that is the 'Qwen leg measured Inkling'
    class: the served model is the only thing that makes every later number
    mean what the profile says it means."""
    status, text = http_get(f"{base_url}/v1/models", timeout=timeout)
    if status != 200:
        raise RuntimeError(f"/v1/models returned {status}: {text[:200]}")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:  # noqa: BLE001
        raise RuntimeError(f"/v1/models returned non-JSON: {text[:200]}") from exc
    return [m.get("id", "") for m in data.get("data", [])
            if isinstance(m, dict)]


def ensure_served_model(base_url: str, model: str) -> None:
    ids = served_models(base_url)
    if model not in ids:
        raise RuntimeError(f"served model is not the profile model: "
                           f"expected {model!r}, served {ids}")


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
# Phase 5: chat replay (multi-turn decode at growing context)
# --------------------------------------------------------------------------- #
# The BetterBench single-stream prompts are 60-120 tokens. A lever that changes
# tokens/update (speculative depth, acceptance rule, KV dtype) can cost half the
# throughput on a real chat and still win the weighted median: k=15 on 1.9.0 won
# +10.3 % weighted while losing every prose category and dropping 1k-context chat
# from 72 to 34 t/s. This phase measures exactly that: growing context, cold
# prefix cache, decode-only tok/s.
CHAT_FILLER = (
    "The maintenance log for rack {i} records nominal temperatures, stable fan "
    "curves and no error counters worth reporting. "
)
CHAT_ANSWER = (
    "Noted: rack {i} ran nominal, the fans were stable and no error counter "
    "moved. I keep the interval at the documented value and re-check the seal "
    "at the next turn. "
)


def stream_decode(base_url: str, model: str, payload: dict, timeout: int = 900) -> dict:
    """One streaming completion; returns TTFT and DECODE-only tok/s.

    vLLM emits one content delta per emitted token, so the delta count is the
    completion length. Decode t/s excludes the time to first token -- a
    total-wall number reports prefill speed for short answers and would make a
    context-length comparison meaningless.
    """
    body = json.dumps({"model": model, "stream": True, **payload}).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            chunk = json.loads(data)
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0].get("delta") or {}
            if delta.get("content"):
                if ttft is None:
                    ttft = time.time() - t0
                tokens += 1
    wall = time.time() - t0
    decode_s = wall - (ttft or 0.0)
    return {
        "ttft_ms": round((ttft or 0.0) * 1000, 1),
        "completion_tokens": tokens,
        "wall_s": round(wall, 2),
        "decode_tps": round(tokens / decode_s, 1) if decode_s > 0.5 and tokens else 0.0,
    }


def build_chat(depth_tokens: int, turns: int) -> list[dict]:
    """A multi-turn conversation of roughly `depth_tokens` prompt tokens."""
    per_turn = max(1, depth_tokens // max(1, turns))
    sentences = max(1, (per_turn * 4) // len(CHAT_FILLER))   # ~4 characters per token
    messages: list[dict] = []
    for turn in range(turns):
        messages.append({"role": "user",
                         "content": "".join(CHAT_FILLER.format(i=turn) for _ in range(sentences))})
        messages.append({"role": "assistant",
                         "content": "".join(CHAT_ANSWER.format(i=turn) for _ in range(3))})
    messages.append({"role": "user",
                     "content": "Summarise the maintenance log above in eight bullet points."})
    return messages


def run_chat_replay(base_url: str, model: str, profile: dict, mode: str) -> dict:
    cfg = profile.get("chat_replay") or {}
    depths = list(cfg.get("depths") or [4000, 8000, 16000])
    if mode == "fast":
        depths = depths[: int(cfg.get("fast_levels", 1))]
    turns = int(cfg.get("turns", 3))
    max_tokens = int(cfg.get("max_tokens", 256))
    levels: dict[str, dict] = {}
    for depth in depths:
        payload = {
            "messages": build_chat(int(depth), turns),
            "max_tokens": max_tokens,
            "temperature": 0.7,
            # Unique salt => cold prefix cache, so the number describes the
            # context length and not a cached replay of an identical prompt.
            "cache_salt": f"accept-chat-{depth}-{time.time_ns()}",
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            levels[str(depth)] = stream_decode(base_url, model, payload)
        except Exception as exc:  # noqa: BLE001
            levels[str(depth)] = {"error": str(exc)[:200]}
        log(f"chat {depth}: {levels[str(depth)]}")
    return {"ok": all("error" not in v for v in levels.values()), "levels": levels}


# --------------------------------------------------------------------------- #
# Phase 6: BetterBench + GSM8K
# --------------------------------------------------------------------------- #
def _median(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return round(vals[mid], 1)
    return round((vals[mid - 1] + vals[mid]) / 2, 1)


def category_medians(data: dict) -> tuple[dict, dict]:
    """Per-category MEDIAN decode t/s and tokens/update of the single-stream phase.

    This is the number the weighted median hides: on 1.9.0 k=15 the weighted
    value rose +10.3 % while prose lost 7.2 % and chat 8.6 %, because the mix
    carries 55 % weight on code/json/file_edit.
    """
    tps: dict[str, float] = {}
    tpu: dict[str, float] = {}
    for cat, entries in (data.get("single_stream") or {}).items():
        ok = [e for e in entries if e.get("ok")]
        med_tps = _median([e.get("decode_tps") for e in ok])
        if med_tps is not None:
            tps[cat] = med_tps
            med_tpu = _median([e.get("tokens_per_update") for e in ok])
            if med_tpu is not None:
                tpu[cat] = med_tpu
    return tps, tpu


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
    cat_tps, cat_tpu = category_medians(data)
    return {
        "out": str(out_file),
        "aggregate_tps": {str(k): round(v["aggregate_tps"], 2) for k, v in sorted(levels.items())},
        "ok_requests": {str(k): [v["ok"], v["requests"]] for k, v in sorted(levels.items())},
        "categories_decode_tps": cat_tps,
        "categories_tokens_per_update": cat_tpu,
    }


def run_gsm8k(python: str, base_url: str, model: str, limit: int, out_dir: Path) -> dict:
    # 16 concurrent completions: with temperature=0 the answers are deterministic, so
    # the accuracy is independent of the serving concurrency, and 200 items decode in a
    # fraction of the time. (Was 4: the eval, not the server, was the bottleneck, and
    # every extra minute is production down.)
    model_args = (f"model={model},base_url={base_url}/v1/chat/completions,"
                  f"num_concurrent=16,max_retries=3,timeout=600")
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
def gate_checks(results: dict, profile: dict, add) -> None:
    """Regression gates of ci/quick_bench.py as hard checks: hook proof, greedy vs the last release,
    concurrency shape and drafter floor (the last two from the BetterBench result of fast/full)."""
    qp = profile.get("quick_profile")
    if "hooks" in results:
        add("hook proof", not results["hooks"], results["hooks"] or "all present", "every required hook line")
    q = results.get("quick")
    if q:
        add("references recorded", not q.get("unrecorded"), q.get("unrecorded") or "yes",
            "every quick-tier reference recorded (release mode never skips a null)")
        for name, ok, detail in q["rows"]:
            add(f"quick {name}", ok, detail, "reference JSON")
        probs = [f for f in q["failures"] if f.startswith("greedy")]
        add("quick greedy", not probs, probs or "ok", "matches the last release")
    if results.get("greedy") is not None and qp:
        ref_dir = ROOT / "references"
        ref = json.loads((ref_dir / f"quick-{qp}-greedy.json").read_text())
        allow = json.loads((ref_dir / "greedy_allow.json").read_text()).get(qp, {})
        probs = quick_bench.greedy_problems(results["greedy"], ref, allow)
        add("greedy vs last release", not probs, probs or "4/4 identical",
            "identical, or allow-listed with a reason")
    bb = results.get("betterbench") or {}
    agg = {int(k): v for k, v in (bb.get("aggregate_tps") or {}).items()}
    if len(agg) > 1:
        probs = quick_bench.concurrency_problems(agg)
        add("concurrency shape", not probs, probs or agg, "rising 1>2>4>8, 2 streams >= 1.2x")
    floor = profile.get("tok_per_update_floor")
    if floor and bb:
        probs = quick_bench.floor_problems(bb.get("categories_tokens_per_update") or {}, floor)
        add("drafter floor", not probs, probs or "ok", f"tok/update (code) >= {floor}")


def compare(results: dict, baseline: dict, thresholds: dict, mode: str, profile: dict | None = None) -> list[dict]:
    checks: list[dict] = []

    def add(name, ok, actual, expected, hard=True):
        # `ok is None` means "not measured" (skipped phase, no baseline data).
        # Collapsing it to False made an unmeasured check look like a failure
        # and a measured-and-passed one indistinguishable from an absent one.
        checks.append({"name": name, "ok": None if ok is None else bool(ok),
                       "actual": actual, "expected": expected, "hard": hard})

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

    gate_checks(results, profile or {}, add)

    smoke = results.get("smoke", {})
    if not smoke:
        pass  # quick tier: no smoke phase
    elif smoke.get("skipped"):
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
        # Conc 1 is an arrival metric: 24 sequential requests with idle gaps
        # between them, so the aggregate mixes cold TTFT with host state and
        # spreads ~20% run to run (0.4.5 warm runs: 91.8-108% of a freshly
        # calibrated baseline), while conc 8 overlaps and is stable at ~1.0x.
        # Same class of problem that demoted tokens/step to soft, but conc 1
        # still gates real regressions -- wider band, not soft.
        min_ratio = thresholds.get(f"aggregate_tps_min_ratio_conc_{level}",
                                   thresholds["aggregate_tps_min_ratio"])
        add(f"throughput conc {level}", ratio >= min_ratio,
            f"{value} t/s ({ratio:.3f}x)",
            f">= {min_ratio:.2f}x of {base} t/s")

    spec = results.get("spec", {})

    # Per-category decode. The weighted median is a MIXTURE, so a lever that
    # wins code/json/file_edit and loses prose looks like a win -- exactly how
    # 1.9.0's k=15 shipped (weighted +10.3 %, prose -7.2 %, chat -8.6 %, and
    # 30 t/s in the user's chat). Every measured category is its own check; the
    # threshold is per profile, because a category that legitimately changes
    # has to be re-calibrated in the same PR that changes it.
    bb_cats = results.get("betterbench", {}).get("categories_decode_tps") or {}
    cat_base = baseline.get("categories_decode_tps") or {}
    cat_min = thresholds.get("category_decode_tps_min_ratio", 0.95)
    for cat in sorted(cat_base):
        base = cat_base[cat]
        value = bb_cats.get(cat)
        if not base or value is None:
            continue
        ratio = value / base
        add(f"throughput {cat}", ratio >= cat_min,
            f"{value} t/s ({ratio:.3f}x)",
            f">= {cat_min:.2f}x of {base} t/s")
    if bb_cats and not cat_base:
        # A baseline recorded before this check existed carries no category
        # medians. Say so instead of reporting a silent pass.
        add("throughput per category", None,
            f"baseline has no category medians ({len(bb_cats)} measured)",
            "re-record the baseline", hard=False)

    # Chat replay: decode t/s at growing context, cold prefix cache. This is the
    # profile the weights cannot describe at all -- BetterBench prompts are
    # 60-120 tokens, a real chat is not. Soft in fast mode (one level, short
    # run), hard in full mode.
    chat_res = results.get("chat") or {}
    chat_base = baseline.get("chat_decode_tps") or {}
    chat_min = thresholds.get("chat_tps_min_ratio", 0.95)
    for depth in sorted(chat_base, key=lambda d: int(d)):
        base = chat_base[depth]
        level = (chat_res.get("levels") or {}).get(depth) or {}
        value = level.get("decode_tps")
        if not base or not value:
            continue
        ratio = value / base
        add(f"chat {int(depth) // 1000}k decode", ratio >= chat_min,
            f"{value} t/s ({ratio:.3f}x)",
            f">= {chat_min:.2f}x of {base} t/s", hard=(mode == "full"))
    if chat_res and not chat_res.get("ok"):
        add("chat replay", False, "a level failed", "every level answered")

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
    cats = results.get("betterbench", {}).get("categories_decode_tps")
    if cats:
        lines += ["", "BetterBench decode per category (median t/s): " +
                  ", ".join(f"{c} = {v}" for c, v in sorted(cats.items()))]
    chat = (results.get("chat") or {}).get("levels")
    if chat:
        lines += ["", "Chat replay, decode t/s (cold prefix cache): " +
                  ", ".join(f"{int(d) // 1000}k ctx = {v.get('decode_tps')}"
                            for d, v in sorted(chat.items(), key=lambda kv: int(kv[0])))]
    rel = results.get("release", {})
    lines += ["", f"Production restored: `{rel.get('service', 'docker-vllm7.service')}` "
                  f"health={rel.get('health')} kv={rel.get('kv_tokens')}"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
def _acquire_with_retry(win: "GpuWindow", args: argparse.Namespace) -> dict:
    """acquire, but retry while production is busy (or the lock is held).

    gpu-window's `force` skips the busy check; that stays an explicit opt-in.
    This only makes the gate tolerant of a busy production instead of failing
    on the first refusal: back off and re-check. Any *other* refusal (e.g. a
    foreign lock, bad ttl) is surfaced immediately -- we never paper over it.
    """
    if args.dry_run:
        return win.acquire(force=args.force)

    max_wait = int(os.environ.get("ACCEPT_ACQUIRE_WAIT", "3600"))   # total patience
    delay = int(os.environ.get("ACCEPT_ACQUIRE_DELAY", "90"))      # backoff (s)
    t0 = time.time()
    attempt = 0
    while True:
        try:
            return win.acquire(force=args.force)
        except RuntimeError as exc:
            msg = str(exc)
            retriable = BUSY_HINT in msg or "already held" in msg or "locked" in msg
            if not retriable or time.time() - t0 >= max_wait:
                raise
            attempt += 1
            log(f"GPU window busy (attempt {attempt}): {msg.strip()[:160]} -- "
                f"waiting {delay}s (patience {int(max_wait - (time.time() - t0))}s left)")
            time.sleep(delay)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", required=True)
    ap.add_argument("--image", required=True, help="candidate image tag, e.g. vllm-sly-radiance:0.2.5-rocm10.0")
    ap.add_argument("--mode", choices=["quick", "release", "fast", "full"], default="fast")
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
    ap.add_argument("--record-baseline-on-pass", action="store_true",
                    help="like --record-baseline, but only when the verdict is PASS (accept-night: the baseline "
                         "moves forward with a release that passed, never with a regression)")
    ap.add_argument("--deadline", type=float, default=None, metavar="EPOCH",
                    help="unix time the measurement must be done by (accept-night: 05:45 Europe/Berlin). A phase "
                         "that would start after it aborts the run -- the release phase still runs, so the GPU "
                         "goes back in time")
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
    load_format = profile.get("load_format", "auto")
    if profile.get("window") == "local-docker":
        # no service unit in the helper: the candidate is our own container on the GPU host (class docstring)
        base_url = f"http://127.0.0.1:{profile.get('port', 8001)}"
        if profile.get("docker_argv") == "qwen":
            # Qwen: own pinned argv (homelab swift-mxfp4 + 1.13.4 reference), own container
            docker_argv = qwen_docker_argv
            container = "vllm-accept-qwen"
            model_dir = os.environ.get("ACCEPT_QWEN_MODEL", "/root/mcb-ckpt-v2")
            cache_dir = os.environ.get("ACCEPT_QWEN_CACHE", "/root/vllm7-cache")
            drafter_dir = None
        else:
            docker_argv = inkling_docker_argv
            container = "vllm-accept-inkling"
            model_dir = os.environ.get("ACCEPT_INKLING_MODEL", "/root/inkling2-mini")
            cache_dir = os.environ.get("ACCEPT_INKLING_CACHE", "/root/accept-inkling-cache")
            drafter_dir = os.environ.get("ACCEPT_INKLING_DRAFTER", "/root/dflash2-ink/model")
        win = LocalDockerWindow(
            args.ssh_target, args.ssh_key, args.owner, args.ttl, args.dry_run, load_format,
            docker_argv=docker_argv, container=container,
            prod_service=profile.get("prod_service", "docker-vllm7.service"), port=profile.get("port", 8001),
            run=(lambda argv, **kw: log(f"DRY-RUN: {' '.join(argv[:4])}") or "") if args.dry_run else None,
            health=(lambda url: 200) if args.dry_run else None,
            model_dir=model_dir, drafter_dir=drafter_dir, cache_dir=cache_dir,
            install_signals=not args.dry_run)
    else:
        win = GpuWindow(args.ssh_target, args.ssh_key, args.owner, args.ttl, args.dry_run,
                        load_format=load_format)

    def before(phase: str) -> None:
        if args.deadline and time.time() > args.deadline:
            raise RuntimeError(f"deadline passed before phase '{phase}' -- the window is out of time")

    try:
        log(f"status: {json.dumps(win.status())}")
        log(f"acquiring GPU window (owner={args.owner}, ttl={args.ttl}s)")
        results["acquire"] = _acquire_with_retry(win, args)

        before("start")
        log(f"starting candidate {args.image} ({profile.get('service') or 'own container'})")
        results["start"] = win.start(args.image, profile.get("service", ""))
        log(f"candidate up: {json.dumps(results['start'])}")
        # Fail fast: the window must serve the profile's model, not production.
        # That is the 'Qwen leg measured Inkling' class -- the served model is
        # the only thing that makes every later number mean what the profile
        # says it means (guard message is kept stable: consumers match on it).
        ensure_served_model(base_url, profile["model"])

        log("checking logs")
        log_text = win.logs()
        results["logs"] = check_logs(log_text, profile)
        if profile.get("quick_profile"):
            results["hooks"] = quick_bench.hook_problems(log_text, profile["quick_profile"])
        if not results["logs"]["ok"]:
            failures.append("logs")
            log(f"log check failed: missing={results['logs']['missing']} "
                f"forbidden={[f['name'] for f in results['logs']['forbidden']]}")

        model = profile["model"]
        before("smoke")
        log("smoke tests")
        if args.mode in ("quick", "release"):
            # quick (~5 min, PR builds): hooks + greedy + decode/prefill/concurrency 1/2/4 vs the reference
            # JSON. release (~10 min, tagging): + 8 streams + GSM8K-50, an unrecorded reference fails closed.
            # No smoke, chat replay, BetterBench or lm-eval GSM8K here (ci/README.md, "GPU tiers")
            if args.dry_run:
                log("DRY-RUN: skipping the quick probes")
            else:
                before("quick")
                results["quick"], _ = quick_bench.run_quick(
                    profile["quick_profile"], base_url, model, log_text, release=args.mode == "release")
        else:
            results["smoke"] = run_smoke(base_url, model, profile, args.dry_run)
            if not results["smoke"]["ok"]:
                failures.append("smoke")

        if args.dry_run or args.mode in ("quick", "release"):
            log("skipping greedy, betterbench and gsm8k")
        else:
            before("greedy")
            results["greedy"] = quick_bench.greedy_run(base_url, model)
            before("chat")
            log(f"chat replay (multi-turn, growing context, {args.mode})")
            results["chat"] = run_chat_replay(base_url, model, profile, args.mode)
            if not results["chat"]["ok"]:
                failures.append("chat")

            before("betterbench")
            metrics_before = scrape_metrics(base_url)
            log(f"betterbench ({args.mode})")
            results["betterbench"] = run_betterbench(
                args.bb_python, Path(args.bb_dir), base_url, model, config,
                out_dir / f"betterbench-{args.mode}.json",
                {"image": args.image, "profile": name, "mode": args.mode, "run": args.owner},
            )
            after = scrape_metrics(base_url)
            results["spec"] = spec_delta(metrics_before, after)
            log(f"spec: {json.dumps(results['spec'])}")

            if args.mode == "full":
                before("gsm8k")
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

    checks = compare(results, baseline, thresholds, args.mode, profile)
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
    elif (args.record_baseline or (args.record_baseline_on_pass and not hard_fail)) and "error" not in results:
        new = dict(baseline_file)
        new["recorded"] = {"image": args.image, "at": meta["started"], "mode": args.mode}
        new["image"] = args.image
        new["measured"] = meta["started"].split(" ")[0]
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
        # Per-category medians are mode-independent (single stream), so they are
        # not nested under modes[]. Without them the per-category checks stay
        # silently skipped -- the exact hole the 1.9.0 k=15 release fell through.
        cat_tps = results.get("betterbench", {}).get("categories_decode_tps") or {}
        if cat_tps:
            new["categories_decode_tps"] = cat_tps
            new["categories_tokens_per_update"] = (
                results["betterbench"].get("categories_tokens_per_update") or {})
        else:
            log("no per-category medians -- keeping the existing categories_decode_tps")
        # Chat replay levels are additive: a fast run measures fewer depths than a
        # full run, and the missing ones must keep their reference value.
        chat_levels = {d: v["decode_tps"]
                       for d, v in (results.get("chat", {}).get("levels") or {}).items()
                       if v.get("decode_tps")}
        if chat_levels:
            new["chat_decode_tps"] = {**(new.get("chat_decode_tps") or {}), **chat_levels}
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
