#!/usr/bin/env python3
"""CPU tests for the Inkling-2 prefill-shape warmup (sly/inkling2/radiance_inkling2_warmup.py + sly/patch_inkling2_warmup.py).

Why: vLLM's kernel warmup only runs 9-token prefills. The first request of every new prompt length then JIT-compiles
Triton kernels (_attn_prologue/_rel_attention/_sconv/_route_tokens/_expert_gemm/_situ_quantize/_mxfp8_gemm/_seq_metadata/
_k_group/_scatter): 2.8-9.5 s per first request in the 2026-11-25 T0 probe (n=8/32/128/256). The warmup runs one real
single-request prefill + one spec-decode step per ladder size before the API is up.
No torch, no vllm, no GPU.

Run:  python3 sly/tests/test_inkling2_warmup_cpu.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sly" / "inkling2"))
sys.path.insert(0, str(ROOT / "sly"))
sys.path.insert(0, str(ROOT / "ci"))
sys.path.insert(0, str(ROOT))
import radiance_inkling2_warmup as w  # noqa: E402
from patch_list import patch_names  # noqa: E402


def test_ladder_covers_every_size_class():
    sizes = w.ladder(max_tokens=4096, max_len=32768)
    assert sizes == sorted(set(sizes)) and sizes[0] >= 2 and sizes[-1] == 4096
    # Triton specialises integer args on ==1 and %16==0: both classes must exist in every tile bucket
    for lo, hi in ((2, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 512), (513, 4096)):
        inb = [n for n in sizes if lo <= n <= hi]
        assert any(n % 16 == 0 for n in inb), (lo, hi)
        assert any(n % 16 for n in inb), (lo, hi)
    # the GEMM switch (M >= 513) is bracketed
    assert 512 in sizes and 513 in sizes


def test_ladder_respects_the_caps():
    assert max(w.ladder(max_tokens=2048, max_len=32768)) == 2048
    assert max(w.ladder(max_tokens=4096, max_len=1000)) <= 999       # prompt + 1 sampled token must fit
    assert w.ladder(max_tokens=1, max_len=100) == []


def test_wanted_only_inkling_and_default_on():
    assert w.wanted("inkling2_loop", {}) is True
    assert w.wanted("qwen3_5", {}) is False
    assert w.wanted("inkling2_loop", {"RADIANCE_INKLING2_PREFILL_WARMUP": "0"}) is False
    assert w.wanted("qwen3_5", {"RADIANCE_INKLING2_PREFILL_WARMUP": "1"}) is False   # never on a foreign model
    assert w.wanted(None, {}) is False


def test_loop_survives_one_failing_size_and_always_cleans_up():
    ran, cleaned = [], []

    def one(n):
        ran.append(n)
        if n == 33:
            raise RuntimeError("boom")

    ok, failed = w.run_ladder([16, 33, 64], one, cleanup=lambda n: cleaned.append(n))
    assert ran == [16, 33, 64] and ok == 2 and failed == [33]
    assert cleaned == [16, 33, 64]


FAKE_WORKER = '''class Worker:
    def _compile_or_warm_up_model(self):
        if (
            self.vllm_config.kernel_config.enable_jit_warmup
            or self.cache_config.enable_extensible_kv_cache
        ):
            # A workspace resize after capture frees what the graphs point at.
            warmup_kernels(self.model_runner, self.execute_model, self.sample_tokens)
        return 1
'''


def _patch(root):
    import subprocess
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(root), PYTHONPATH=str(ROOT))
    return subprocess.run([sys.executable, str(ROOT / "sly" / "patch_inkling2_warmup.py")], env=env,
                          capture_output=True, text=True)


def test_patch_hooks_after_warmup_kernels_and_is_idempotent():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "vllm/v1/worker/gpu_worker.py"
        f.parent.mkdir(parents=True)
        (Path(d) / "vllm/models/inkling2").mkdir(parents=True)
        f.write_text(FAKE_WORKER)
        r = _patch(d)
        assert r.returncode == 0, r.stdout + r.stderr
        s = f.read_text()
        assert s.index("warmup_kernels(self.model_runner") < s.index("warmup_prefill_ladder(self)")
        compile(s, "gpu_worker.py", "exec")
        again = _patch(d)
        assert again.returncode == 0 and "NOOP" in again.stdout and f.read_text() == s


def test_patch_is_a_noop_without_inkling():
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "vllm/v1/worker/gpu_worker.py"
        f.parent.mkdir(parents=True)
        f.write_text(FAKE_WORKER)
        r = _patch(d)
        assert r.returncode == 0 and "NOOP" in r.stdout and f.read_text() == FAKE_WORKER


def test_patch_and_module_ship_in_the_image():
    assert "sly/patch_inkling2_warmup" in patch_names()
    assert "radiance_inkling2_warmup.py" in (ROOT / "Dockerfile").read_text()


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
    print("OK")
