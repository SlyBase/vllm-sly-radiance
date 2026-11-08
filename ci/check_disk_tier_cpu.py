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
import sys
import textwrap

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


def _new_content(node):
    """The `new` source an apply()/apply_any() injects (the first variant's `new` for apply_any)."""
    if getattr(node.func, "id", "") == "apply":
        cand = node.args[2] if len(node.args) > 2 else _kw(node, "new")
        return cand.value if isinstance(cand, ast.Constant) else None
    variants = node.args[1] if len(node.args) > 1 else _kw(node, "variants")
    if isinstance(variants, ast.List) and variants.elts:
        first = variants.elts[0]
        if isinstance(first, ast.Tuple) and len(first.elts) > 1:
            new = first.elts[1]
            return new.value if isinstance(new, ast.Constant) else None
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


def _new_content(node, patch_ns):
    """The `new` source an apply()/apply_any() injects, evaluated when it references patch constants."""
    if getattr(node.func, "id", "") == "apply":
        cand = node.args[2] if len(node.args) > 2 else _kw(node, "new")
    else:
        variants = node.args[1] if len(node.args) > 1 else _kw(node, "variants")
        if not (isinstance(variants, ast.List) and variants.elts):
            return None
        first = variants.elts[0]
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


def _load_new_string(patch_name, sentinel_name):
    """Return the `new` source string of the apply()/apply_any() in the patch whose sentinel matches
    `sentinel_name` (matched positionally or as a keyword)."""
    tree = ast.parse(open(os.path.join(KV, patch_name)).read())
    patch_ns = _patch_ns(patch_name)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "id", "") not in ("apply", "apply_any"):
            continue
        if _sentinel_value(node) != sentinel_name:
            continue
        content = _new_content(node, patch_ns)
        if content is not None:
            return content
    raise SystemExit(f"no apply with sentinel {sentinel_name!r} in {patch_name}")


def _extract_top(new_src, funcs=None):
    """Unparse the top-level Assign nodes and (when `funcs` is given) the FunctionDef nodes named in
    `funcs` out of the injected source, dedenting so the test runs the real code. `funcs` filters out
    sibling members (decorators, other methods) that would not exec on their own."""
    tree = ast.parse(textwrap.dedent(new_src))
    out = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
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
    {"_radiance_split"},
)
_inv_src = _extract_top(
    _load_const("patch_lookup_invalidate.py", "_INVALIDATE_METHOD"),
    {"invalidate"},
)

ns = {"functools": functools, "os": os}
exec(_fanout_src, ns)
exec(_split_src, ns)
exec(_inv_src, ns)
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

print(f"{fails[0]} failures")
sys.exit(1 if fails[0] else 0)
