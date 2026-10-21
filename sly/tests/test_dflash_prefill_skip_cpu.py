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
    # prefix caching: cached drafter blocks of skipped positions would hold garbage, unless the
    # scheduler side keeps them out of the cache (APC sub-knob, tests below)
    assert _with_knob("1", lambda: skip.skip_context(_spec(prefix_caching=True), b)) is False
    # any full-attention drafter layer reads the whole prompt
    mixed = ("sliding_attention",) * 4 + ("full_attention",)
    assert _with_knob("1", lambda: skip.skip_context(_spec(layer_types=mixed), b)) is False
    # no window configured
    assert _with_knob("1", lambda: skip.skip_context(_spec(window=None), b)) is False
    # data parallel: the draft step is a collective
    assert _with_knob("1", lambda: skip.skip_context(_spec(dp=2), b)) is False


# ---- prefix caching: the skipped drafter blocks never become cache hits -------------------------

APC = "RADIANCE_DFLASH_PREFILL_WINDOW_SKIP_APC"


def _env(**kv):
    """Run fn with the env vars set (None unsets); restores them."""
    def run(fn):
        old = {k: os.environ.get(k) for k in kv}
        for k, v in kv.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v
        try:
            return fn()
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v
    return run


def _kvm(block=16, window=2048):
    """Scheduler-side KVCacheManager stand-in: a target hybrid group and the drafter's SWA group."""
    groups = [NS(layer_names=["t0"], kv_cache_spec=NS(block_size=65536, sliding_window=None)),
              NS(layer_names=["d0"], kv_cache_spec=NS(block_size=block, sliding_window=window))]
    return NS(kv_cache_config=NS(kv_cache_groups=groups))


def _swa_manager(block=16, window=2048):
    return NS(kv_cache_spec=NS(block_size=block, sliding_window=window))


def test_apc_sub_knob_enables_the_worker_side():
    s = _spec(prefix_caching=True)
    b = _batch([0], [4096], [47000])
    on = {KNOB: "1", APC: "1"}
    assert _env(**on)(lambda: skip.skip_context(s, b)) is True
    s2 = _spec(prefix_caching=True)
    assert _env(**{KNOB: "1", APC: "0"})(lambda: skip.skip_context(s2, b)) is False
    s3 = _spec(prefix_caching=True)
    assert _env(**{KNOB: "0", APC: "1"})(lambda: skip.skip_context(s3, b)) is False   # needs the main knob
    s4 = _spec(prefix_caching=False)   # APC off: the sub-knob changes nothing
    assert _env(**{KNOB: "1", APC: None})(lambda: skip.skip_context(s4, b)) is True


def test_note_chunk_is_inert_without_both_knobs():
    for env in ({KNOB: "1", APC: None}, {KNOB: None, APC: "1"}, {KNOB: "0", APC: "1"}):
        r = NS(num_tokens=47000)
        k = _kvm()
        _env(**env)(lambda k=k, r=r: skip.note_chunk(k, r, 4096))
        assert not hasattr(r, "radiance_pf_skip_end"), env


def test_note_chunk_marks_exactly_the_chunks_the_worker_may_skip():
    L, w, m = 47000, 2048, 512
    k = _kvm()
    r = NS(num_tokens=L)
    _env(**{KNOB: "1", APC: "1"})(lambda: skip.note_chunk(k, r, 4096))
    assert r.radiance_pf_skip_end == 4096
    skip.note_chunk(k, r, L - w - m)                 # last chunk the worker may skip
    assert r.radiance_pf_skip_end == L - w - m
    skip.note_chunk(k, r, L - w - m + 1)             # a tail chunk: the mark does not move
    assert r.radiance_pf_skip_end == L - w - m
    # the worker skips a small chunk whenever a batch-mate is big, so there is no size gate here
    r2 = NS(num_tokens=L)
    _env(**{KNOB: "1", APC: "1"})(lambda: skip.note_chunk(k, r2, 100))
    assert r2.radiance_pf_skip_end == 100


def test_note_chunk_margin_follows_the_smallest_swa_block():
    k = _kvm(block=1056)                              # margin = max(512, 2 * 1056) = 2112
    r = NS(num_tokens=47000)
    _env(**{KNOB: "1", APC: "1"})(lambda: skip.note_chunk(k, r, 47000 - 2048 - 2112))
    assert r.radiance_pf_skip_end == 47000 - 2048 - 2112
    r2 = NS(num_tokens=47000)
    skip.note_chunk(k, r2, 47000 - 2048 - 2112 + 1)
    assert not hasattr(r2, "radiance_pf_skip_end")
    # no SWA group at all: nothing to protect
    r3 = NS(num_tokens=47000)
    k2 = NS(kv_cache_config=NS(kv_cache_groups=[_kvm().kv_cache_config.kv_cache_groups[0]]))
    _env(**{KNOB: "1", APC: "1"})(lambda: skip.note_chunk(k2, r3, 4096))
    assert not hasattr(r3, "radiance_pf_skip_end")


def test_mask_unwritten_drops_the_blocks_below_the_mark():
    r = NS(radiance_pf_skip_end=100)                  # block 16: blocks 0..6 hold a skipped position
    mgr = _swa_manager()
    assert skip.mask_unwritten(mgr, r, 0, 10, None) == [False] * 7 + [True] * 3
    assert skip.mask_unwritten(mgr, r, 5, 10, None) == [False, False, True, True, True]
    assert skip.mask_unwritten(mgr, r, 7, 10, None) is None          # nothing below the mark: untouched
    base = [True, False, True, True, True]                           # an existing mask stays ANDed
    assert skip.mask_unwritten(mgr, r, 5, 10, base) == [False, False, True, True, True]
    assert skip.mask_unwritten(mgr, r, 7, 12, [True, False, True, True, True]) == [True, False, True, True, True]
    # exact multiple: block 5 (80..95) is below 96, block 6 (96..111) holds position 96..99 only if mark>96
    assert skip.mask_unwritten(mgr, NS(radiance_pf_skip_end=96), 0, 8, None) == [False] * 6 + [True] * 2


def test_mask_unwritten_ignores_other_groups_and_unmarked_requests():
    r = NS(radiance_pf_skip_end=100)
    full = NS(kv_cache_spec=NS(block_size=16, sliding_window=None))   # full attention / mamba group
    assert skip.mask_unwritten(full, r, 0, 10, None) is None
    assert skip.mask_unwritten(_swa_manager(), NS(), 0, 10, None) is None
    base = [True] * 3
    assert skip.mask_unwritten(_swa_manager(), NS(), 0, 3, base) is base


def test_a_registered_drafter_block_is_always_written():
    _env(**{KNOB: "1", APC: "1"})(_registered_blocks_are_written)


def _registered_blocks_are_written():
    """Simulate scheduler (note_chunk + mask_unwritten) and worker (skip_context) over many shapes:
    a block the scheduler lets into the prefix cache must hold only drafter-KV the worker wrote."""
    rng = np.random.default_rng(7)
    skipped_runs = 0
    for _ in range(300):
        L = int(rng.integers(500, 90000))
        chunk = int(rng.choice([1024, 2048, 4096, 8192]))
        block = int(rng.choice([8, 16, 1056]))
        s = _spec(block=block, prefix_caching=True)
        k = _kvm(block=block)
        mgr = _swa_manager(block=block)
        r = NS(num_tokens=L)
        written = np.zeros(L, dtype=bool)
        registered, cached, pos = [], 0, 0
        while pos < L:
            n = min(chunk, L - pos)
            end = pos + n
            skip.note_chunk(k, r, end)
            if not skip.skip_context(s, _batch([pos], [n], [L])):
                written[pos:end] = True
            nf = end // block
            mask = skip.mask_unwritten(mgr, r, cached, nf, None)
            registered += [i for j, i in enumerate(range(cached, nf)) if mask is None or mask[j]]
            cached, pos = max(cached, nf), end
        for i in registered:
            assert written[i * block:(i + 1) * block].all(), (L, chunk, block, i)
        if not written.all():       # not vacuous: the skip fired, and the tail stays cacheable
            skipped_runs += 1
            assert any(i * block >= L - 4096 for i in registered), (L, chunk, block)
    assert skipped_runs > 30, skipped_runs


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
        for name, src in (("kv_cache_manager.py", KVM_TAIL), ("single_type_kv_cache_manager.py", MGR_TAIL)):
            (Path(td) / "vllm/v1/core" / name).parent.mkdir(parents=True, exist_ok=True)
            (Path(td) / "vllm/v1/core" / name).write_text(src)
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


KVM_TAIL = """class K:
    def allocate_slots(self):
        num_tokens_to_cache = min(total_computed_tokens + num_new_tokens, request.num_tokens)
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)
"""
MGR_TAIL = """class M:
    def cache_blocks(self):
        block_mask = self.reachable_block_mask(
            dcp_world_size=self.dcp_world_size,
        )
        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            block_mask=block_mask,
        )
"""
CORE = "vllm/v1/core/"


def _real(name):
    f = ROOT / ".dryrun" / "src" / "vllm" / "vllm" / "v1" / "core" / name
    return f.read_text() if f.exists() else None


def _patch_sources(kvm_src, mgr_src):
    with tempfile.TemporaryDirectory() as td:
        for name, src in (("kv_cache_manager.py", kvm_src), ("single_type_kv_cache_manager.py", mgr_src),
                          ("spec_decode/dflash/speculator.py", PROPOSE_HEAD)):
            f = Path(td) / ("vllm/v1/core/" + name if "/" not in name else "vllm/v1/worker/gpu/" + name)
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(src)
        r = _run_patch(td)
        r2 = _run_patch(td)
        return r, r2, (Path(td) / "vllm/v1/core/kv_cache_manager.py").read_text(), \
            (Path(td) / "vllm/v1/core/single_type_kv_cache_manager.py").read_text()


def _check_cache_hooks(kvm_src, mgr_src):
    r, r2, kvm, mgr = _patch_sources(kvm_src, mgr_src)
    assert r.returncode == 0, r.stdout + r.stderr
    compile(kvm, "kv_cache_manager.py", "exec")
    compile(mgr, "single_type_kv_cache_manager.py", "exec")
    # the mark is set before the chunk is cached, the mask before the blocks are hashed
    assert kvm.index("note_chunk(") < kvm.index("coordinator.cache_blocks(request, num_tokens_to_cache)")
    assert mgr.index("mask_unwritten(") < mgr.index("self.block_pool.cache_full_blocks(")
    assert "block_mask=block_mask" in mgr
    assert "NOOP" in r2.stdout and r2.returncode == 0, r2.stdout
    assert (kvm, mgr) == _patch_sources(kvm_src, mgr_src)[2:]


def test_patch_hooks_the_cache_path_once_and_is_idempotent():
    _check_cache_hooks(KVM_TAIL, MGR_TAIL)


def test_patch_hooks_apply_to_the_pinned_vllm_tree():
    kvm, mgr = _real("kv_cache_manager.py"), _real("single_type_kv_cache_manager.py")
    if kvm is None or mgr is None:
        import pytest
        pytest.skip("pinned vLLM tree absent (run ci/patch_dryrun.sh)")
    _check_cache_hooks(kvm, mgr)


def test_wiring_loop_copy_and_docs():
    assert "sly/patch_dflash_prefill_skip" in patch_names(), "patch is not in the Dockerfile loop"
    assert "sly/radiance_dflash_skip.py" in (ROOT / "Dockerfile").read_text(), \
        "helper module is not COPYed into site-packages"
    assert "patch_dflash_prefill_skip.py" in (ROOT / "sly" / "README.md").read_text()
    assert KNOB in (ROOT / "sly" / "README.md").read_text()
    assert f'"{KNOB}", "0"' in (ROOT / "sly" / "radiance_dflash_skip.py").read_text(), "default must be 0"


def test_launcher_ships_the_skip_on_and_overridable():
    """The module default stays 0 (stock path without the launcher); serve-mxfp4.sh turns it on, as
    `${VAR:-1}` so `RADIANCE_DFLASH_PREFILL_WINDOW_SKIP=0` still reverts it."""
    txt = (ROOT / "serve-mxfp4.sh").read_text()
    assert f'-e {KNOB}="${{{KNOB}:-1}}"' in txt, "serve-mxfp4.sh must ship the drafter skip default on"
    for off in ("R4D_ATTN_DB", "RADIANCE_GDN_SCAN_VARIANT"):
        assert off not in txt.replace("R4D_ATTN_DB\n", ""), f"{off} must stay out of the launcher defaults"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("dflash prefill skip tests OK")

