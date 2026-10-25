#!/usr/bin/env python3
"""CPU tests for the drafter partial prefill skip (sly/patch_dflash_prefill_partial.py).

The final prefill chunk's draft reads the drafter's context K/V only from prompt_len - window - margin
on, and the draft tokens of every earlier chunk are dropped. `radiance_dflash_skip.context_rows` cuts
`fc` and the context-K/V fill of a one-request prefill chunk to those rows, and drops the draft step
of a chunk that is not the last. The tests pin the decision rule, and run the patched vLLM 0.32.0
`propose` (fixtures/dflash_propose_v0.32.0.py.txt) on whole chunked prompts against numpy fakes:

  * every position from prompt_len - window - margin on holds K/V computed from its own hidden state,
    and no row below r0 stores anything (no stale hidden state, no stale slot);
  * those positions went through the same GEMM row count regime (M >= 513 or the exact M) for `fc`
    and the K/V projection as on the stock path, i.e. they are bit-identical to it;
  * exactly one draft step runs, on the prompt's last chunk;
  * the knob off leaves the stock behaviour, eager and FULL-graph.

DFLASH_SPECULATOR_SRC=<speculator.py with both patches applied> runs the same harness against that
file instead of the fixture (e.g. the ci/patch_dryrun.sh tree, or the one in the image).
No torch, no triton, no GPU.

Run:  python3 sly/tests/test_dflash_prefill_partial_cpu.py
"""
import ast
import os
import subprocess
import sys
import tempfile
import textwrap
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sly"))
sys.path.insert(0, str(ROOT / "ci"))
from patch_list import patch_names  # noqa: E402

import radiance_dflash_skip as skip  # noqa: E402

KNOB = "RADIANCE_DFLASH_PREFILL_PARTIAL_SKIP"
WHOLE = "RADIANCE_DFLASH_PREFILL_WINDOW_SKIP"
ENVS = (KNOB, WHOLE, "RADIANCE_DFLASH_PREFILL_WINDOW_SKIP_MARGIN", "RADIANCE_DFLASH_FC_TILED_MIN_M")
PATCH = ROOT / "sly" / "patch_dflash_prefill_partial.py"
SKIP_PATCH = ROOT / "sly" / "patch_dflash_prefill_skip.py"
FIXTURE = ROOT / "sly" / "tests" / "fixtures" / "dflash_propose_v0.32.0.py.txt"
REL = Path("vllm/v1/worker/gpu/spec_decode/dflash/speculator.py")
W = 2048


@contextmanager
def _env(**kv):
    old = {k: os.environ.pop(k, None) for k in ENVS}
    os.environ.update({k: v for k, v in kv.items() if v is not None})
    try:
        yield
    finally:
        for k in ENVS:
            os.environ.pop(k, None)
            if old[k] is not None:
                os.environ[k] = old[k]


def _cfg(block=16, prefix_caching=False, dp=1, layer_types=("sliding_attention",) * 5, window=W):
    hf = NS(layer_types=list(layer_types), sliding_window=window, dflash_config={})
    return dict(
        draft_model_config=NS(hf_config=hf),
        vllm_config=NS(cache_config=NS(enable_prefix_caching=prefix_caching)),
        kv_cache_config=NS(kv_cache_groups=[NS(layer_names=["d0"], kv_cache_spec=NS(block_size=block))]),
        draft_attn_layer_names={"d0"},
        dp_size=dp,
    )


def _batch(computed, scheduled, prefill_len, num_tokens=None):
    a = lambda x: np.asarray(x, dtype=np.int32)  # noqa: E731
    return NS(num_reqs=len(computed), num_computed_prefill_tokens_np=a(computed),
              num_scheduled_tokens=a(scheduled), prefill_len_np=a(prefill_len),
              num_tokens=int(sum(scheduled)) if num_tokens is None else num_tokens)


def _rows(batch, **spec_kw):
    return skip.context_rows(NS(**_cfg(**spec_kw)), batch)


# --- the decision rule ---------------------------------------------------------------------------

def test_default_is_off():
    b = _batch([0], [4096], [6000])
    with _env():
        assert _rows(b) is None
    with _env(**{KNOB: "0"}):
        assert _rows(b) is None
    with _env(**{WHOLE: "1"}):  # the whole skip's knob does not turn the partial one on
        assert _rows(b) is None


def test_6000_at_chunk_4096():
    # window 2048 + margin 512: positions >= 3440 are read; the whole skip never acts on this prompt
    with _env(**{KNOB: "1"}):
        assert _rows(_batch([0], [4096], [6000])) == (3440, False)   # 656 rows, no draft
        assert _rows(_batch([4096], [1904], [6000])) is None         # last chunk, all rows: stock


def test_12000_at_chunk_4096():
    with _env(**{KNOB: "1"}):
        assert _rows(_batch([0], [4096], [12000])) == (4096, False)     # whole old chunk
        assert _rows(_batch([4096], [4096], [12000])) == (4096, False)  # ends at 8192 <= 9440
        assert _rows(_batch([8192], [3808], [12000])) == (1248, True)   # 2560 of 3808 rows


def test_kept_rows_never_drop_below_the_large_m_tile():
    L = 3896 + 2560                    # needed from 3896: 200 rows of chunk 1
    with _env(**{KNOB: "1"}):
        assert _rows(_batch([0], [4096], [L])) == (4096 - 513, False)
    with _env(**{KNOB: "1", "RADIANCE_DFLASH_FC_TILED_MIN_M": "1024"}):
        assert _rows(_batch([0], [4096], [L])) == (4096 - 1024, False)
    with _env(**{KNOB: "1", "RADIANCE_DFLASH_FC_TILED_MIN_M": "256"}):  # below 513: 513 still holds
        assert _rows(_batch([0], [4096], [L])) == (4096 - 513, False)
    with _env(**{KNOB: "1"}):
        # a chunk shorter than the floor keeps all its rows
        assert _rows(_batch([0], [300], [5000])) == (300, False)          # old, whole skip
        assert _rows(_batch([2400], [300], [5000])) == (0, False)         # needed, all rows, no draft
        assert _rows(_batch([4700], [300], [5000])) is None               # last chunk: stock


def test_margin_follows_the_whole_skip():
    with _env(**{KNOB: "1"}):  # 2 blocks of 1056 = 2112 > 512
        assert _rows(_batch([0], [4096], [6000]), block=1056) == (6000 - 2048 - 2112, False)
    with _env(**{KNOB: "1", "RADIANCE_DFLASH_PREFILL_WINDOW_SKIP_MARGIN": "1000"}):
        assert _rows(_batch([0], [4096], [6000])) == (6000 - 3048, False)


def test_short_prompts_and_single_chunks():
    with _env(**{KNOB: "1"}):
        assert _rows(_batch([0], [1514], [1514])) is None                 # one chunk: stock
        assert _rows(_batch([0], [2048], [2500])) == (0, False)           # all needed, no draft
        assert _rows(_batch([2048], [452], [2500])) is None


def test_decode_mixed_and_multi_request_steps_are_untouched():
    L = 47000
    with _env(**{KNOB: "1"}):
        assert _rows(_batch([L], [8], [L])) is None                       # decode / spec verify
        assert _rows(_batch([0, L], [4096, 8], [L, L])) is None           # mixed
        assert _rows(_batch([0, 0], [4096, 4096], [L, L])) is None        # two prefills
        assert _rows(_batch([0], [4096], [L], num_tokens=4100)) is None   # rows != scheduled tokens


def test_dummy_profile_and_unsafe_configs():
    b = _batch([0], [4096], [6000])
    with _env(**{KNOB: "1"}):
        s = NS(**_cfg())
        assert skip.context_rows(s, b, dummy_run=True) is None
        assert skip.context_rows(s, b, is_profile=True) is None
        assert getattr(s, "_radiance_pf_partial", None) is None   # not memoised on a profile call
        assert _rows(b, prefix_caching=True) is None
        assert _rows(b, dp=2) is None
        assert _rows(b, window=None) is None
        assert _rows(b, layer_types=("sliding_attention",) * 4 + ("full_attention",)) is None


# --- the patched propose, end to end ---------------------------------------------------------------

PAD = -1
STALE = -7          # hidden-state rows nobody wrote this step
GARBAGE = 1 << 40   # slot ids past the batch, as the real kernel leaves them
MAXTOK = 8192 + 64


class T:
    """The few tensor ops propose uses, on numpy."""

    def __init__(self, a):
        self.a = a

    def __getitem__(self, k):
        return T(self.a[k])

    def copy_(self, o):
        self.a[...] = o.a
        return self

    def fill_(self, v):
        self.a[...] = v
        return self

    @property
    def shape(self):
        return self.a.shape

    def max(self):
        return T(np.asarray(self.a.max()))

    def item(self):
        return self.a.item()


class _Any:
    def __getattr__(self, name):
        return _Any()

    def __getitem__(self, k):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()


class _Torch:
    Tensor = object

    @staticmethod
    def cat(ts, dim=-1):
        return T(np.concatenate([t.a for t in ts], axis=dim))


class _Mode:
    NONE, PIECEWISE, FULL = "NONE", "PIECEWISE", "FULL"


def _prepare_dflash_inputs(*args):
    slots, ib = args[3], args[9]   # positional, as propose calls it
    n, c0 = ib.num_tokens, int(ib.num_computed_prefill_tokens_np[0])
    slots.a[:n] = c0 + np.arange(n)   # slot id = position
    slots.a[n:] = GARBAGE


class _Graph:
    def __init__(self, spec):
        self.spec = spec

    def run_fullgraph(self, batch_desc):  # _prepare_graph_forward, then the draft model
        s = self.spec
        s._precompute_context_kv(0, s._num_graph_context_tokens(batch_desc.num_reqs))
        s._generate_draft(batch_desc.num_reqs)


class _Model:
    def __init__(self, spec):
        self.spec = spec

    def combine_hidden_states(self, x):
        pos = x.a[:, 0]
        self.spec.fc_m[pos[pos >= 0]] = x.shape[0]
        self.spec.work["fc"] += x.shape[0]
        return T(x.a[:, :1].copy())


class _Spec:
    num_query_per_req = 8
    num_speculative_steps = 7
    max_model_len = 1 << 20
    max_num_tokens = MAXTOK
    pcp_manager = None
    draft_kv_cache_group_id = 0
    draft_kv_cache_group_ids = [0]
    supports_decode_metadata_update = False
    _group_causal = False
    dp_rank = 0
    dcp_size = 1

    def __init__(self, prompt, cg_mode, **cfg):
        self.__dict__.update(_cfg(**cfg))
        self.cg_mode = cg_mode
        self.hidden_states = T(np.full((MAXTOK, 1), STALE, dtype=np.int64))
        self._context_slot_mappings = T(np.zeros((1, MAXTOK), dtype=np.int64))
        self.draft_tokens = T(np.zeros((4, 7), dtype=np.int64))
        self.model = _Model(self)
        self.query_cudagraph_manager = _Graph(self)
        self.input_buffers = self.block_tables = self.context_positions = _Any()
        self.sample_indices = self.sample_pos = self.sample_idx_mapping = _Any()
        self.temperature = self.seeds = _Any()
        self.parallel_drafting_token_id = self.max_num_reqs = self.sample_from_anchor = 0
        self.kv_h = np.full(prompt, -1, dtype=np.int64)   # hidden state each position's K/V came from
        self.kv_m = np.zeros(prompt, dtype=np.int64)      # M of the K/V projection that stored it
        self.kv_fc = np.zeros(prompt, dtype=np.int64)     # M of the fc call its hidden state came from
        self.fc_m = np.zeros(prompt, dtype=np.int64)
        self.work = {"fc": 0, "kv": 0}                     # rows through fc / the K/V projection
        self.bad = []
        self.drafts = []
        self.step = None

    def _num_graph_context_tokens(self, num_reqs):
        return min(num_reqs * (self.num_speculative_steps + 1), self.max_num_tokens)

    def _precompute_context_kv(self, start, end, dummy_run=False):
        if dummy_run:
            return
        self.work["kv"] += end - start
        h = self.hidden_states.a[start:end, 0]
        sl = self._context_slot_mappings.a[0, start:end]
        st = sl != PAD
        if (sl[st] >= len(self.kv_h)).any() or (h[st] < 0).any():
            self.bad.append((self.step, start, end))
            st &= (sl < len(self.kv_h)) & (h >= 0)
        self.kv_h[sl[st]] = h[st]
        self.kv_m[sl[st]] = end - start
        self.kv_fc[sl[st]] = self.fc_m[h[st]]

    def prepare_context_anchor(self, input_batch, num_rejected):
        pass

    def _build_uniform_attn_metadata(self, **kw):
        return None

    def _prepare_eplb_forward(self, n):
        pass

    def _generate_draft(self, num_reqs, *a, **k):
        self.drafts.append(self.step)


def _dispatch(manager, num_reqs, num_tokens, **kw):
    return NS(num_tokens=num_tokens, cg_mode=manager.spec.cg_mode, num_reqs=num_reqs), None


def _patched_source():
    """propose of the fixture after the skip and partial patches (or of DFLASH_SPECULATOR_SRC)."""
    if os.environ.get("DFLASH_SPECULATOR_SRC"):
        return Path(os.environ["DFLASH_SPECULATOR_SRC"]).read_text()
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / REL
        f.parent.mkdir(parents=True)
        f.write_text(FIXTURE.read_text())
        for p in (SKIP_PATCH, PATCH):
            r = _run_patch(p, td)
            assert r.returncode == 0, r.stdout + r.stderr
        return f.read_text()


def _propose_fn(src):
    mod = ast.parse(src)
    cls = next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == "DFlashSpeculator")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "propose")
    fn.decorator_list = []
    code = "from __future__ import annotations\n" + textwrap.dedent(ast.unparse(fn))
    ns = dict(torch=_Torch, CUDAGraphMode=_Mode, PAD_SLOT_ID=PAD, prepare_dflash_inputs=_prepare_dflash_inputs,
              dispatch_cg_and_sync_dp=_dispatch, build_slot_mappings_by_layer=lambda *a: None)
    exec(compile(code, "propose", "exec"), ns)
    return ns["propose"]


_PROPOSE = {}


def _propose():
    if "fn" not in _PROPOSE:
        _PROPOSE["fn"] = _propose_fn(_patched_source())
    return _PROPOSE["fn"]


def _prefill(prompt, chunk, cg_mode, **cfg):
    """Run one prompt through propose chunk by chunk; the spec records what the drafter did."""
    s = _Spec(prompt, cg_mode, **cfg)
    for c0 in range(0, prompt, chunk):
        n = min(chunk, prompt - c0)
        s.step = c0
        ib = _batch([c0], [n], [prompt])
        ib.seq_lens_cpu_upper_bound = T(np.asarray([c0 + n]))
        ib.idx_mapping = None
        aux = [T((c0 + np.arange(n))[:, None].astype(np.int64)) for _ in range(5)]
        _propose()(s, ib, None, None, aux[0], aux, None, None, None, None, None, None)
    return s


def _regime(m, floor):
    return np.where(m >= floor, -1, m)   # all M >= floor share one tile; below it the exact M counts


PROMPTS = (1, 9, 513, 1514, 2559, 2561, 4096, 4097, 5000, 6000, 6456, 8193, 12000, 20000)
CHUNKS = (8, 520, 2048, 2560, 4096, 8192)


def _check(prompt, chunk, mode, whole, block, tiled):
    env = {WHOLE: whole, "RADIANCE_DFLASH_FC_TILED_MIN_M": tiled}
    with _env(**env):
        stock = _prefill(prompt, chunk, mode, block=block)
    with _env(**env, **{KNOB: "1"}):
        part = _prefill(prompt, chunk, mode, block=block)
    floor = max(513, int(tiled or 0))   # the drafter GEMMs' large-M regime, not read from the module
    margin = max(512, 2 * block)
    need = slice(max(0, prompt - W - margin), prompt)
    pos = np.arange(prompt)[need]
    tag = (prompt, chunk, mode, whole, block, tiled)
    assert not part.bad and not stock.bad, (tag, part.bad, stock.bad)
    assert (part.kv_h[need] == pos).all(), tag                     # every read position is fresh
    assert (stock.kv_h[need] == pos).all(), tag
    for got, ref in ((part.kv_m, stock.kv_m), (part.kv_fc, stock.kv_fc)):
        assert (_regime(got[need], floor) == _regime(ref[need], floor)).all(), tag   # bit-identical
    last = (prompt - 1) // chunk * chunk
    assert part.drafts == [last], (tag, part.drafts)               # one draft, on the last chunk
    assert stock.drafts[-1] == last, tag
    return part, stock


def test_patched_propose_eager_and_full_graph():
    for mode in (_Mode.NONE, _Mode.FULL):
        for whole in (None, "1"):
            for block, tiled in ((16, "513"), (1056, "513"), (16, "0"), (16, "1024")):
                for prompt in PROMPTS:
                    for chunk in CHUNKS:
                        if chunk > 8 or prompt <= 4097:  # 8: chunks at the FULL graph's own rows
                            _check(prompt, chunk, mode, whole, block, tiled)


def test_patched_propose_row_counts():
    # drafter context rows actually stored, partial vs stock (chunk 4096, the shipped knobs)
    def rows(s):
        return int((s.kv_h >= 0).sum()), s.work["fc"], s.work["kv"]

    got = {L: tuple(map(rows, _check(L, 4096, _Mode.FULL, "1", 16, "513"))) for L in (6000, 12000, 20000)}
    # (stored, fc rows, K/V rows) partial vs stock. 6000: the whole skip never acts (chunk 1 ends at
    # 4096 > 3440), the partial keeps 656 + 1904 rows. 12000 / 20000: the whole skip drops the chunks
    # ending <= L - 2560, the partial cuts the last one to 2560 (+ the FULL graph's 8 padded rows).
    assert got == {6000: ((2560, 2560, 2560), (6000, 6000, 6000)),
                   12000: ((2560, 2560, 2568), (3808, 3808, 3808)),
                   20000: ((2560, 2560, 2568), (3616, 3616, 3616))}, got


def test_knob_off_is_the_stock_function():
    with _env(**{WHOLE: "1"}):
        for mode in (_Mode.NONE, _Mode.FULL):
            for prompt, chunk in ((6000, 4096), (12000, 4096), (3000, 520)):
                s = _prefill(prompt, chunk, mode)
                assert len(s.drafts) == -(-prompt // chunk) - sum(
                    1 for c0 in range(0, prompt, chunk)
                    if min(chunk, prompt - c0) >= W + 512 and c0 + chunk <= prompt - W - 512)


# --- patch mechanics and wiring --------------------------------------------------------------------

def _run_patch(patch, site):
    env = dict(os.environ, PYTHONPATH=str(ROOT), RADIANCE_PATCH_TARGET_DIR=str(site))
    return subprocess.run([sys.executable, str(patch)], env=env, capture_output=True, text=True,
                          cwd=str(ROOT / "sly"))


def test_patch_applies_after_the_skip_patch_and_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / REL
        f.parent.mkdir(parents=True)
        f.write_text(FIXTURE.read_text())
        r = _run_patch(PATCH, td)   # the partial patch anchors on the skip hook
        assert r.returncode != 0 and "FAIL" in r.stdout + r.stderr, r.stdout
        assert _run_patch(SKIP_PATCH, td).returncode == 0
        r = _run_patch(PATCH, td)
        assert r.returncode == 0 and r.stdout.count("  OK  ") == 4, r.stdout + r.stderr
        out = f.read_text()
        compile(out, "speculator.py", "exec")
        assert out.count("context_rows(") == 1
        assert out.index("skip_context(") < out.index("context_rows(")
        r2 = _run_patch(PATCH, td)
        assert r2.returncode == 0 and r2.stdout.count("NOOP") == 4, r2.stdout
        assert f.read_text() == out


def test_wiring_loop_and_docs():
    names = patch_names()
    assert "sly/patch_dflash_prefill_partial" in names, "patch is not in the Dockerfile loop"
    assert names.index("sly/patch_dflash_prefill_skip") < names.index("sly/patch_dflash_prefill_partial")
    assert "patch_dflash_prefill_partial.py" in (ROOT / "sly" / "README.md").read_text()
    assert KNOB in (ROOT / "README.md").read_text()
    assert f'"{KNOB}", "0"' in (ROOT / "sly" / "radiance_dflash_skip.py").read_text(), "default must be 0"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
    print("dflash prefill partial skip tests OK")
