#!/usr/bin/env python3
"""CPU tests for the drafter sliding-window prefill skip (sly/patch_dflash_prefill_skip.py).

During a chunked prefill the DFlash2 drafter's draft tokens are discarded by the scheduler and
its context K/V is only ever read inside its sliding window. A chunk that ends at least
`window + margin` tokens before the end of the prompt therefore needs neither `fc`, the context-KV
fill nor the draft step. The tests pin the decision rule, the safety gates and the wiring.
No torch, no triton, no GPU.

Run:  python3 sly/tests/test_dflash_prefill_skip_cpu.py
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sly"))
sys.path.insert(0, str(ROOT / "ci"))
from patch_list import patch_names  # noqa: E402

import radiance_dflash_skip as skip  # noqa: E402

KNOB = "RADIANCE_DFLASH_PREFILL_WINDOW_SKIP"
PATCH = ROOT / "sly" / "patch_dflash_prefill_skip.py"


def _spec(layer_types=("sliding_attention",) * 5, window=2048, prefix_caching=False, dp=1, block=16):
    hf = NS(layer_types=list(layer_types), sliding_window=window, dflash_config={})
    return NS(
        draft_model_config=NS(hf_config=hf),
        vllm_config=NS(cache_config=NS(enable_prefix_caching=prefix_caching)),
        kv_cache_config=NS(kv_cache_groups=[NS(layer_names=["d0"], kv_cache_spec=NS(block_size=block))]),
        draft_attn_layer_names={"d0"},
        dp_size=dp,
    )


def _batch(computed, scheduled, prefill_len):
    a = lambda x: np.asarray(x, dtype=np.int32)  # noqa: E731
    return NS(num_reqs=len(computed), num_computed_prefill_tokens_np=a(computed),
              num_scheduled_tokens=a(scheduled), prefill_len_np=a(prefill_len))


def _with_knob(value, fn):
    old = os.environ.pop(KNOB, None)
    if value is not None:
        os.environ[KNOB] = value
    try:
        return fn()
    finally:
        os.environ.pop(KNOB, None)
        if old is not None:
            os.environ[KNOB] = old


def test_default_is_off():
    s = _spec()
    assert _with_knob(None, lambda: skip.skip_context(s, _batch([0], [4096], [47000]))) is False
    assert _with_knob("0", lambda: skip.skip_context(s, _batch([0], [4096], [47000]))) is False


def test_old_chunks_skip_and_the_tail_does_not():
    L = 47000
    s = _spec()
    # window 2048 + margin 512: a chunk ending at or before L - 2560 is skipped
    last_skipped_end = L - 2048 - 512
    assert _with_knob("1", lambda: skip.skip_context(s, _batch([0], [4096], [L]))) is True
    assert skip.skip_context(s, _batch([last_skipped_end - 4096], [4096], [L])) is True
    assert skip.skip_context(s, _batch([last_skipped_end - 4095], [4096], [L])) is False
    assert skip.skip_context(s, _batch([L - 4096], [4096], [L])) is False      # final chunk


def test_short_prefill_chunks_do_not_skip():
    s = _spec()
    L = 47000
    # the chunk after a skipped one must itself be at least window + margin tokens, otherwise its
    # draft read reaches back into the skipped range (--max-num-batched-tokens 2048 -> 2032)
    assert _with_knob("1", lambda: skip.skip_context(s, _batch([0], [2032], [L]))) is False
    # --max-num-batched-tokens 4096 -> 4080 tokens: covered by the next chunk
    assert _with_knob("1", lambda: skip.skip_context(s, _batch([0], [4080], [L]))) is True
    assert skip.skip_context(s, _batch([0], [2559], [L])) is False
    assert skip.skip_context(s, _batch([0], [2560], [L])) is True


def test_decode_and_mixed_batches_are_untouched():
    s = _spec()
    L = 47000
    # decode row: all prefill tokens computed (computed >= prefill_len)
    assert _with_knob("1", lambda: skip.skip_context(s, _batch([L], [8], [L]))) is False
    # one old prefill chunk next to a decode row: the decode row needs the full path
    assert skip.skip_context(s, _batch([0, L], [4096, 8], [L, L])) is False
    # two long prefills, both old: skip
    assert skip.skip_context(s, _batch([0, 4096], [4096, 4096], [L, L])) is True
    # short prompt (window larger than the prompt) never skips
    assert skip.skip_context(s, _batch([0], [1500], [1500])) is False


def test_dummy_and_profile_runs_never_skip():
    s = _spec()
    b = _batch([0], [4096], [47000])
    assert _with_knob("1", lambda: skip.skip_context(s, b, dummy_run=True)) is False
    assert _with_knob("1", lambda: skip.skip_context(s, b, is_profile=True)) is False
    # the profile run precedes set_attn: the speculator has no kv_cache_config yet, so the
    # config must not be read (and not memoised) on a dummy/profile call
    bare = _spec()
    del bare.kv_cache_config
    assert _with_knob("1", lambda: skip.skip_context(bare, b, is_profile=True)) is False
    assert getattr(bare, "_radiance_pf_skip", None) is None


def test_margin_covers_two_kv_blocks():
    s = _spec(block=1056)           # hybrid page-size aligned block
    # margin = max(512, 2 * 1056) = 2112: a chunk ending at L - 2048 - 512 must NOT skip
    assert _with_knob("1", lambda: skip.skip_context(s, _batch([0], [47000 - 2560], [47000]))) is False
    assert skip.skip_context(s, _batch([0], [47000 - 2048 - 2112], [47000])) is True


def test_margin_ignores_the_targets_hybrid_group():
    def spec(block, names=("d0",), sw=None):
        s = _spec(block=block)
        s.draft_attn_layer_names = set(names)
        s.kv_cache_config.kv_cache_groups.insert(
            0, NS(layer_names=["t0"], kv_cache_spec=NS(block_size=65536, sliding_window=None)))
        s.kv_cache_config.kv_cache_groups[1].kv_cache_spec.sliding_window = sw
        return s
    for block, margin in ((16, 512), (1024, 2048)):
        cfg = _with_knob("1", lambda: skip._config(spec(block)))
        assert cfg == (2048, margin), cfg
    # drafter group not identifiable: smallest sliding-window group, never the 65536 hybrid one
    assert _with_knob("1", lambda: skip._config(spec(16, names=("x",), sw=2048))) == (2048, 512)
    assert _with_knob("1", lambda: skip._config(spec(16, names=("x",)))) == (2048, 512)


def test_unsafe_configs_disable_the_knob():
    b = _batch([0], [4096], [47000])
    # prefix caching: cached drafter blocks of skipped positions would hold garbage
    assert _with_knob("1", lambda: skip.skip_context(_spec(prefix_caching=True), b)) is False
    # any full-attention drafter layer reads the whole prompt
    mixed = ("sliding_attention",) * 4 + ("full_attention",)
    assert _with_knob("1", lambda: skip.skip_context(_spec(layer_types=mixed), b)) is False
    # no window configured
    assert _with_knob("1", lambda: skip.skip_context(_spec(window=None), b)) is False
    # data parallel: the draft step is a collective
    assert _with_knob("1", lambda: skip.skip_context(_spec(dp=2), b)) is False


PROPOSE_HEAD = '''class DFlashSpeculator:
    def propose(self, input_batch, dummy_run=False, is_profile=False):
        num_reqs = input_batch.num_reqs
        num_target_tokens = input_batch.num_tokens
        num_query_tokens = num_reqs * self.num_query_per_req
        return self.draft_tokens[:num_reqs]
'''


def _run_patch(site):
    env = dict(os.environ, PYTHONPATH=str(ROOT), RADIANCE_PATCH_TARGET_DIR=str(site))
    return subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True,
                          cwd=str(ROOT / "sly"))


def test_patch_hooks_propose_once_and_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
        f.parent.mkdir(parents=True)
        f.write_text(PROPOSE_HEAD)
        r = _run_patch(td)
        assert r.returncode == 0, r.stdout + r.stderr
        out = f.read_text()
        compile(out, "speculator.py", "exec")
        assert out.count("skip_context(") == 1
        # the hook sits before any work of propose(): first statement after num_reqs
        lines = out.splitlines()
        i = next(k for k, l in enumerate(lines) if "num_reqs = input_batch.num_reqs" in l)
        assert "radiance_dflash_skip" in "\n".join(lines[i + 1:i + 6])
        r2 = _run_patch(td)
        assert r2.returncode == 0 and "NOOP" in r2.stdout, r2.stdout
        assert f.read_text() == out


def test_wiring_loop_copy_and_docs():
    assert "sly/patch_dflash_prefill_skip" in patch_names(), "patch is not in the Dockerfile loop"
    assert "sly/radiance_dflash_skip.py" in (ROOT / "Dockerfile").read_text(), \
        "helper module is not COPYed into site-packages"
    assert "patch_dflash_prefill_skip.py" in (ROOT / "sly" / "README.md").read_text()
    assert KNOB in (ROOT / "sly" / "README.md").read_text()
    assert f'"{KNOB}", "0"' in (ROOT / "sly" / "radiance_dflash_skip.py").read_text(), "default must be 0"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("dflash prefill skip tests OK")
