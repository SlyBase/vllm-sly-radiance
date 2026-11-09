#!/usr/bin/env python3
"""CPU test for the disk-tier fanout + invalidation logic (kv-cache/patch_offload_fs_fanout.py,
kv-cache/patch_lookup_invalidate.py). The logic is extracted from the patch source -- constants,
helpers and methods unparse'd from the real injected code -- and run on CPU:

  * _RADIANCE_FANOUT_TARGET_BYTES / _RADIANCE_FANOUT_MAX are the real module knobs,
  * _radiance_batches splits n_items into disjoint, order-preserving slices -- every item lands in
    exactly one slice and the slices cover the whole range,
  * _radiance_fanout_degree returns 1 for a 0/1-block job, caps by the thread count, and divides the
    budget by the in-flight job count (fairness across concurrent jobs),
  * _radiance_split fans a store/load job into one partial per batch WITHOUT reading the KV tensor:
    a tensor that raises on any device->host read is captured into each partial but never read,
  * invalidate() drops only a cached-absent (False) verdict, leaving a present (True) and an in-flight
    (None) verdict untouched.

Runnable on CPU in a plain `docker run` (no device flags) after the disk patches are applied:
  python3 ci/check_disk_tier_cpu.py
"""
import ast
import functools
import os
import re
import sys
import tempfile
import threading
import textwrap
import types

HERE = os.path.dirname(os.path.abspath(__file__))
KV = os.path.join(os.path.dirname(HERE), "kv-cache")


def _kw(node, name):
    for k in node.keywords:
        if k.arg == name:
            return k.value
    return None


def _sentinel_value(node):
    """The sentinel string of an apply()/apply_any() call, matched as a keyword or positionally."""
    kw = _kw(node, "sentinel")
    if isinstance(kw, ast.Constant):
        return kw.value
    idx = 3 if getattr(node.func, "id", "") == "apply" else 2
    if len(node.args) > idx and isinstance(node.args[idx], ast.Constant):
        return node.args[idx].value
    return None


def _patch_ns(patch_name):
    """Module-level string constants in the patch (e.g. _SPLIT_BODY), so a `new` node that
    concatenates them evaluates to the real injected text."""
    tree = ast.parse(open(os.path.join(KV, patch_name)).read())
    ns = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    ns[t.id] = node.value.value
    return ns


def _new_content(node, patch_ns, variant_index=0):
    """The `new` source an apply()/apply_any() injects (the `variant_index`-th variant for apply_any),
    evaluated when it references patch constants."""
    if getattr(node.func, "id", "") == "apply":
        cand = node.args[2] if len(node.args) > 2 else _kw(node, "new")
    else:
        variants = node.args[1] if len(node.args) > 1 else _kw(node, "variants")
        if not (isinstance(variants, ast.List) and variants.elts):
            return None
        if variant_index >= len(variants.elts):
            return None
        first = variants.elts[variant_index]
        if not (isinstance(first, ast.Tuple) and len(first.elts) > 1):
            return None
        cand = first.elts[1]
    if isinstance(cand, ast.Constant):
        return cand.value
    if isinstance(cand, ast.BinOp) and isinstance(cand.op, ast.Add):
        return eval(compile(ast.Expression(cand), "<new>", "eval"), patch_ns)
    return None


def _load_const(patch_name, var_name):
    """Return the string value of the module-level `var_name = '...'` assignment in the patch -- the
    complete method body, independent of the (possibly partial) apply `new` wrapper."""
    tree = ast.parse(open(os.path.join(KV, patch_name)).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == var_name \
                and isinstance(node.value, ast.Constant):
            return node.value.value
    raise SystemExit(f"no module-level {var_name!r} in {patch_name}")


def _load_new_string(patch_name, sentinel_name, variant_index=0):
    """Return the `new` source string of the apply()/apply_any() in the patch whose sentinel matches
    `sentinel_name` (matched positionally or as a keyword); for apply_any, the `variant_index`-th
    variant's `new` (0 = first)."""
    tree = ast.parse(open(os.path.join(KV, patch_name)).read())
    patch_ns = _patch_ns(patch_name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", "") not in ("apply", "apply_any"):
            continue
        if _sentinel_value(node) != sentinel_name:
            continue
        content = _new_content(node, patch_ns, variant_index)
        if content is not None:
            return content
    raise SystemExit(f"no apply with sentinel {sentinel_name!r} in {patch_name}")


def _extract_top(new_src, funcs=None, collect_assigns=True):
    """Unparse the top-level Assign nodes and (when `funcs` is given) the FunctionDef nodes named in
    `funcs` out of the injected source, dedenting so the test runs the real code. `funcs` filters out
    sibling members (decorators, other methods) that would not exec on their own; `collect_assigns`
    drops the module-level Assigns (e.g. a closure factory whose siblings reference call-site locals)."""
    tree = ast.parse(textwrap.dedent(new_src))
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and collect_assigns:
            out.append(ast.unparse(node))
        elif isinstance(node, ast.FunctionDef) and (funcs is None or node.name in funcs):
            out.append(ast.unparse(node))
    return "\n\n".join(out)


fails = [0]


def report(ok, what, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {what} {detail}", flush=True)
    if not ok:
        fails[0] += 1
    return ok


class RaisingTensor:
    """A stand-in KV tensor that raises on any device->host read. If the disk tier ever reads the
    tensor on the CPU (`.cpu()`, `.item()`, indexing, iteration) this fires and the test fails."""

    def __init__(self, name="kv"):
        self._name = name

    def cpu(self):
        raise RuntimeError(f"device->host read of {self._name}")

    def item(self):
        raise RuntimeError(f"device->host read of {self._name}")

    def tolist(self):
        raise RuntimeError(f"device->host read of {self._name}")

    def __getitem__(self, i):
        raise RuntimeError(f"device->host read of {self._name}")

    def __iter__(self):
        raise RuntimeError(f"device->host read of {self._name}")

    def __repr__(self):
        return f"RaisingTensor({self._name})"


os.environ["RADIANCE_FS_FANOUT_TARGET_MB"] = "32"
os.environ["RADIANCE_FS_FANOUT_MAX"] = "0"

_fanout_src = _extract_top(
    _load_new_string("patch_offload_fs_fanout.py", "_RADIANCE_FANOUT_TARGET_BYTES"),
    {"_radiance_fanout_degree", "_radiance_batches"},
)
_split_src = _extract_top(
    _load_const("patch_offload_fs_fanout.py", "_SPLIT_BODY"),
    {"_radiance_batch_ranges", "_radiance_split"},
)
_inv_src = _extract_top(
    _load_const("patch_lookup_invalidate.py", "_INVALIDATE_METHOD"),
    {"invalidate"},
)
# 0.32 load path: the per-batch closure factory. Variant 1 = the 0.32 closure (variant 0 is the
# 0.30 plain partial). Only the factory is exercised; the dispatch assigns are dropped.
_load_src = _extract_top(
    _load_new_string("patch_offload_fs_fanout.py", "# radiance R3.14 load-fanout", variant_index=1),
    {"_make_load_task"},
    collect_assigns=False,
)

ns = {"functools": functools, "os": os}
exec(_fanout_src, ns)
exec(_split_src, ns)
exec(_inv_src, ns)
exec(_load_src, ns)
_radiance_fanout_degree = ns["_radiance_fanout_degree"]
_radiance_batches = ns["_radiance_batches"]
_radiance_split = ns["_radiance_split"]
invalidate = ns["invalidate"]

# --- _radiance_batches: disjoint, order-preserving, full coverage -----------------------------
def coverage(n_items, n_batches):
    slices = list(_radiance_batches(n_items, n_batches))
    items = [i for a, b in slices for i in range(a, b)]
    return items, slices

for n_items in (0, 1, 2, 5, 7, 64, 200):
    for n_batches in (1, 2, 3, 8, 12):
        items, slices = coverage(n_items, n_batches)
        ordered = all(slices[i][1] <= slices[i + 1][0] for i in range(len(slices) - 1))
        report(items == list(range(n_items)) and ordered,
               f"coverage {n_items}x{n_batches}", f"-> {len(items)} items")

# a 0- or 1-block job takes the single full-range partial (never an empty task list)
single = list(_radiance_batches(1, 4))
report(single == [(0, 1)], "0/1-block single full-range partial")

# --- _radiance_fanout_degree: boundaries, caps, in-flight fairness -----------------------------
report(_radiance_fanout_degree(0, 8, 12, 0, 27000832) == 1, "0 blocks -> degree 1")
report(_radiance_fanout_degree(1, 8, 12, 0, 27000832) == 1, "1 block -> degree 1")
report(_radiance_fanout_degree(100, 1, 12, 0, 27000832) == 1, "1 thread -> degree 1")
report(_radiance_fanout_degree(100, 8, 12, 0, 27000832) <= 8, "degree capped by thread count")
d1 = _radiance_fanout_degree(100, 8, 12, 1, 27000832)
d2 = _radiance_fanout_degree(100, 8, 12, 2, 27000832)
report(d1 >= d2 and d2 >= 1, f"more in-flight jobs -> fewer batches ({d1}->{d2})")
report(_radiance_fanout_degree(100, 8, 12, 0, 0) <= 12, "block_size 0 falls back to total_threads cap")

# --- _radiance_split: fans out, and never reads the KV tensor ---------------------------------
class FakeManager:
    _primary_kv_view = RaisingTensor("kv")
    _block_size = 27000832
    _use_o_direct = False
    _radiance_total_threads = 12

    class _Pool:
        _inflight_jobs = 0
    _pool = _Pool()


# _radiance_split delegates to self._radiance_batch_ranges; bind it to the class (a plain
# function as a class attribute is not auto-bound) so the unbound call (self passed as the
# FakeManager class) resolves it.
FakeManager._radiance_batch_ranges = types.MethodType(ns["_radiance_batch_ranges"], FakeManager)

paths = [f"/kvcache/blocks/k_{i}.bin" for i in range(40)]
offsets = [i * 27000832 for i in range(40)]
tasks = _radiance_split(FakeManager, lambda io_fn, *a: None, paths, offsets, 8)
report(len(tasks) > 1, f"40-block job fanned into {len(tasks)} partials")
paths_seen = []
for t in tasks:
    # each partial closes over a slice of paths; functools.partial stores positional args in .args
    paths_seen.extend(t.args[0])
report(sorted(paths_seen) == sorted(paths), "every path appears in exactly one partial")
try:
    for t in tasks:
        t()  # calling the io_fn must not read the KV tensor
    report(True, "_radiance_split never reads the KV tensor (raising tensor untouched)")
except RuntimeError as exc:
    report(False, "_radiance_split never reads the KV tensor", str(exc))

# a 0/1-block job returns a single full-range partial (no empty task list)
one = _radiance_split(FakeManager, lambda io_fn, *a: None, [paths[0]], [offsets[0]], 8)
report(len(one) == 1 and len(one[0].args[0]) == 1, "0/1-block single partial")

# --- invalidate(): drops only a cached-absent (False) verdict ---------------------------------
class _State:
    def __init__(self, result):
        self.result = result


class FakeLookup:
    def __init__(self):
        self._lookup_state = {
            "absent": _State(False),
            "present": _State(True),
            "inflight": _State(None),
        }


fake = FakeLookup()
invalidate(fake, ["absent", "present", "inflight"])
report("absent" not in fake._lookup_state, "invalidate drops the cached-absent verdict")
report("present" in fake._lookup_state, "invalidate leaves the present verdict")
report("inflight" in fake._lookup_state, "invalidate leaves the in-flight verdict")

# --- 0.32 load closure: per-batch OSError handling records a GLOBAL prefix, keeps the min -----
class _LoadMgr:
    def __init__(self):
        self._load_progress = {}
        self._radiance_load_lock = threading.Lock()
        self._primary_kv_view = RaisingTensor("kv")
        self._block_size = 1
        self._use_o_direct = False


class _QuietLogger:
    def debug(self, *a, **k):
        pass


def _fail_exc(n):
    exc = OSError("simulated mid-batch failure")
    exc.num_succeeded = n
    return exc


_behave = {"fn": lambda *a: None}


def _batch_load_block(*a, **k):
    return _behave["fn"](*a, **k)


_load_mgr = _LoadMgr()
LOAD_JOB_ID = "J1"
ns["self"] = _load_mgr
ns["job_id"] = LOAD_JOB_ID
ns["logger"] = _QuietLogger()
ns["batch_load_block"] = _batch_load_block
_make_load_task = ns["_make_load_task"]

# success: a batch that loads cleanly leaves _load_progress untouched (no spurious miss)
_behave["fn"] = lambda *a: None
_make_load_task(["p0"], [0], 0)()
report(_load_mgr._load_progress.get(LOAD_JOB_ID) is None,
       "load success leaves _load_progress untouched")

# single batch (global start 0), 2 blocks loaded before the failure -> prefix 0+2 = 2
_behave["fn"] = lambda *a: (_ for _ in ()).throw(_fail_exc(2))
try:
    _make_load_task(["p0", "p1", "p2"], [0, 1, 2], 0)()
except OSError:
    pass
report(_load_mgr._load_progress.get(LOAD_JOB_ID) == 2,
       "single-batch failure records global prefix (start+num_succeeded)")

# two batches: batch A (start 0) fails with 2 succeeded -> prefix 2; batch B (start 5) fails with
# 4 succeeded -> prefix 9. The job's contiguous prefix is the MIN (2), regardless of thread order.
for order in (("A", "B"), ("B", "A")):
    _load_mgr._load_progress = {}
    if order == ("A", "B"):
        seq = ((0, 2), (5, 4))
    else:
        seq = ((5, 4), (0, 2))
    for start, succeeded in seq:
        _behave["fn"] = lambda *a: (_ for _ in ()).throw(_fail_exc(succeeded))
        try:
            _make_load_task([f"p{start}"], [start], start)()
        except OSError:
            pass
    report(_load_mgr._load_progress.get(LOAD_JOB_ID) == 2,
           f"multi-batch failure keeps the MIN prefix (order {order})")

# --- instrumentation: the widened LOOKUP_ASYNC_DELAY ladders must stay strictly ascending ---
# prometheus_client raises ValueError: Buckets not in sorted order at metric construction, so an
# overlapping replacement (the 0.32 variant re-listed 5, 10 before _WIDE) aborts the whole start.
def _ladder_of(txt):
    """The numeric bucket ladder of the `buckets=(...)` in an injected hunk (the text is a
    mid-expression fragment, so it is matched, not parsed)."""
    m = re.search(r"buckets=\(([^)]*)\)", txt, re.S)
    if not m:
        return []
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", m.group(1))]


_instr_tree = ast.parse(open(os.path.join(KV, "patch_offload_instrumentation.py")).read())
_instr_ns = _patch_ns("patch_offload_instrumentation.py")
for _node in ast.walk(_instr_tree):
    if not isinstance(_node, ast.Call) or getattr(_node.func, "id", "") not in ("apply", "apply_any"):
        continue
    if _sentinel_value(_node) != "radiance: extended past 10 s":
        continue
    _is_any = _node.func.id == "apply_any"
    _variants = _node.args[1] if _is_any else None
    _vi_list = range(len(_variants.elts)) if isinstance(_variants, ast.List) else [0]
    for _vi in _vi_list:
        _txt = _new_content(_node, _instr_ns, _vi)
        _ladder = _ladder_of(_txt) if _txt else []
        report(bool(_ladder) and _ladder == sorted(_ladder),
               f"widened stall ladder ascending (hunk {'any' if _is_any else 'one'}, variant {_vi})",
               f"n={len(_ladder)} lo={_ladder[0]} hi={_ladder[-1]}" if _ladder else "no literal buckets=")

# --- every self.<attr> a disk patch reads must exist in the vLLM tree ---------------------------
# The CPU-tier gauge hunk read self._num_blocks / _num_evictable_cache_blocks / _get_num_free_blocks
# while both supported pins name them for chunks, so get_stats() raised AttributeError on the first
# engine step and the GPU run died at the first request. The scan runs over the (patched) tree the
# checker is given, so names the patches introduce are visible too.
_VLLM = os.environ.get("VLLM_TREE")
if not _VLLM:
    try:
        import vllm

        _VLLM = os.path.dirname(vllm.__file__)
    except Exception:
        _VLLM = None

_unused_path = os.path.join(HERE, "unused_patches.txt")
_unused = {os.path.basename(ln.split()[0]) for ln in open(_unused_path) if ln.split() and not ln.startswith("#")}

_all_py = {}
for _d, _sub, _files in os.walk(_VLLM):
    for _f in _files:
        if _f.endswith(".py"):
            with open(os.path.join(_d, _f), errors="ignore") as _fh:
                _all_py[os.path.join(_d, _f)] = _fh.read()


def _attr_names(path):
    return set(re.findall(r"\bself\.(\w+)", _all_py.get(path, "")))


def consts_of(tree):
    """Module-level constants of a patch that are assigned a quoted vLLM file path."""
    consts = {}
    for _n in tree.body:
        if isinstance(_n, ast.Assign) and len(_n.targets) == 1 and isinstance(_n.targets[0], ast.Name):
            _paths = re.findall(r"[\"']([^\"']*\.py)[\"']", ast.unparse(_n.value))
            if _paths:
                consts[_n.targets[0].id] = _paths[0]
    return consts


_patch_txt, _patch_hunks = {}, {}
for _f in sorted(x for x in os.listdir(KV) if x.startswith("patch_") and x.endswith(".py")):
    _tree = ast.parse(open(os.path.join(KV, _f)).read())
    _patch_txt[_f] = "\n".join(
        n.value
        for n in ast.walk(_tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    )
    _ns = _patch_ns(_f)
    _consts = consts_of(_tree)
    _hunks = []
    for _n in ast.walk(_tree):
        if not isinstance(_n, ast.Call) or getattr(_n.func, "id", "") not in ("apply", "apply_any"):
            continue
        if not (_n.args and isinstance(_n.args[0], ast.Name)):
            continue
        _rel = _consts.get(_n.args[0].id, "")
        if not _rel:
            continue
        _files = [c for c in _all_py if c.endswith("/" + _rel)]
        if not _files:
            continue
        _is_any = _n.func.id == "apply_any"
        _variants = _n.args[1] if _is_any else None
        _vi_list = range(len(_variants.elts)) if isinstance(_variants, ast.List) else [0]
        for _vi in _vi_list:
            _txt_new = _new_content(_n, _ns, _vi)
            if _txt_new:
                _hunks.append((_files, _txt_new))
    _patch_hunks[_f] = _hunks

# a name one patch assigns, any other patch may read (e.g. _radiance_cold_grid_stop is
# defined in patch_sched_canonical_chunks.py and read in patch_canon_wait.py)
_defined = set()
for _txt in _patch_txt.values():
    _defined |= set(re.findall(r"\.(\w+)\s*(?::[^=\n]*)?=(?!=)", _txt))
    _defined |= set(re.findall(r"\bdef (\w+)\(", _txt))

for _f, _hunks in _patch_hunks.items():
    if _f in _unused or not _hunks:
        continue
    _missing = set()
    for _files, _txt_new in _hunks:
        _reads = set(re.findall(r"\bself\.(\w+)", _txt_new))
        _allowed = set().union(*(_attr_names(p) for p in _files)) | _defined
        _missing |= _reads - _allowed
    report(not _missing, f"{_f}: every self.<attr> its hunks read exist in the files they patch",
           f"missing: {' '.join(sorted(_missing))}" if _missing else "")

# labelnames= must be a tuple: a bare string's len is its character count, and the
# connector's _get_prometheus_metric asserts len(labelnames) == len(labelvalues), so
# `TIER = "tier"` registers 4 labels and every 1-label emission raises AssertionError.
_str_consts = set()
for _txt in _patch_txt.values():
    _str_consts |= set(m.group(1) for m in re.finditer(
        r'^\s*([A-Z_][A-Z_0-9]*) = "', _txt, re.M))
_bad = set()
for _f, _hunks in _patch_hunks.items():
    if _f in _unused:
        continue
    for _files, _txt_new in _hunks:
        for _m in re.finditer(r"labelnames=([A-Za-z_][\w.]*)[,)]", _txt_new):
            if _m.group(1).split(".")[-1] in _str_consts:
                _bad.add(_m.group(1))
report(not _bad, "labelnames= takes a tuple, not a bare-string constant",
       f"bare-string labelnames: {' '.join(sorted(_bad))}" if _bad else "")

# --- attrs read off the scheduler's group-config objects ---------------------------
# patch_offload_debug_instrument.py's v2 drop-site hunk reads group_config.is_mamba_group, but that
# field only exists where patch_mamba_stride.py adds it -- and that patch self-skips on vLLM >= 0.30
# (skip_if_upstream on requires_cow_source, which the pinned tree has), so the read raised
# AttributeError: 'GroupOffloadConfig' object has no attribute 'is_mamba_group' on the first request.
# A direct read is allowed only when the name is a member of the patched file or an ACTIVE patch
# (not unused, not skipped on this tree) injects it; getattr() is the documented escape hatch.
def _skip_needles(_tree):
    """(target rel path, needle) pairs from any skip_if_upstream(CONST, "needle", ...) call in a patch,
    however it is written (bare call, `if skip_if_upstream(...): sys.exit(0)`, `if not skip_if_upstream(...)`).
    A patch whose needle is already in the pinned tree never reaches its later hunks, so the fields it
    would have added are not available to the patches that stay active."""
    consts = consts_of(_tree)
    out = []
    for _n in ast.walk(_tree):
        if not (isinstance(_n, ast.Call) and getattr(_n.func, "id", "") == "skip_if_upstream"):
            continue
        if len(_n.args) >= 2 and isinstance(_n.args[0], ast.Name) and isinstance(_n.args[1], ast.Constant):
            out.append((consts.get(_n.args[0].id, ""), _n.args[1].value))
    return out


_skipped = set()
for _f in sorted(_patch_hunks):
    for _rel, _needle in _skip_needles(ast.parse(open(os.path.join(KV, _f)).read())):
        if _rel and any(_needle in _all_py[p] for p in _all_py if p.endswith("/" + _rel)):
            _skipped.add(_f)

_defined_active = set()
_patch_added = {}
for _f, _hunks in _patch_hunks.items():
    if _f in _unused or _f in _skipped:
        continue
    _defined_active |= set(re.findall(r"\.(\w+)\s*(?::[^=\n]*)?=(?!=)", _patch_txt[_f]))
    _defined_active |= set(re.findall(r"\bdef (\w+)\(", _patch_txt[_f]))
    for _files, _txt_new in _hunks:
        for _p in _files:
            _patch_added.setdefault(_p, set()).update(
                m.group(1) for m in re.finditer(r"^\s{4,}(\w+)\s*[:=](?!=)", _txt_new, re.M))


# the names the pinned tree uses as members at all (a field read must exist SOMEWHERE: the
# hunks are injected into different files than the class that carries it lives in)
_tree_members = set()
for _t in _all_py.values():
    _tree_members |= set(re.findall(r"\.(\w+)", _t))
_grp_reads = 0
_grp_missing = set()
for _f, _hunks in _patch_hunks.items():
    if _f in _unused or _f in _skipped or not _hunks:
        continue
    for _files, _txt_new in _hunks:
        _reads = set(re.findall(r"\b(?:group_config|group)\.(\w+)", _txt_new))
        if not _reads:
            continue
        _grp_reads += 1
        _allowed = _tree_members | _defined_active | set().union(
            *(_patch_added.get(p, set()) for p in _files))
        _grp_missing |= _reads - _allowed
report(not _grp_missing, f"group/group_config attrs the hunks read exist on this tree "
       f"({_grp_reads} hunks read one; skipped patches' fields do not count)",
       f"missing: {' '.join(sorted(_grp_missing))}" if _grp_missing else "")

print(f"{fails[0]} failures")
sys.exit(1 if fails[0] else 0)
