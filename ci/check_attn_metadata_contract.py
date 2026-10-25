#!/usr/bin/env python3
"""Attention-metadata contract gate: every `m.<attr>` our attention back ends read on a parameter
annotated with a vLLM *Metadata dataclass must exist on that class in the pinned vLLM tree (fields,
methods, properties, bases; AST only -- no vllm, no torch, no GPU). Caught in CI instead of on the
first GPU forward: vLLM 0.31 dropped RocmAttentionMetadata.use_cascade and the AttributeError only
surfaced on a real TETRA start.

Allowed: getattr(m, 'x', default) / hasattr(m, 'x') with a literal, and attrs the same file assigns
(`m.x = ...`, our builders add r4d_plan etc.). Bare reads and getattr without a default FAIL.
ponytail: only ANNOTATED parameters are tracked (no alias/dataflow); unannotated `m` helpers are not
seen -- annotate them to cover them.

Usage: python3 ci/check_attn_metadata_contract.py --vllm-src DIR [--repo ROOT]
       DIR = the vLLM checkout of the dry-run (.dryrun/src/vllm); absent -> skipped, exit 0.
"""
import argparse
import ast
import sys
from pathlib import Path


def index(sources):
    """{class name: set(attrs)} with base classes (by simple name) merged in."""
    raw = {}
    for s in sources:
        for c in (n for n in ast.walk(ast.parse(s)) if isinstance(n, ast.ClassDef)):
            a = set()
            for b in c.body:
                if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                    a.add(b.target.id)
                elif isinstance(b, ast.Assign):
                    a |= {t.id for t in b.targets if isinstance(t, ast.Name)}
                elif isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a.add(b.name)
            bases = [b.id if isinstance(b, ast.Name) else b.value.id for b in c.bases
                     if isinstance(b, ast.Name) or (isinstance(b, ast.Subscript) and isinstance(b.value, ast.Name))]
            raw[c.name] = (a, bases)

    def merged(n, seen=()):
        a, bases = raw[n]
        for b in bases:
            if b in raw and b not in seen:
                a = a | merged(b, seen + (n,))
        return a
    return {n: merged(n) for n in raw}


def _types(ann):
    """Simple names inside an annotation (X, 'X', X | None, mod.X)."""
    if isinstance(ann, ast.Constant) and isinstance(ann.value, str):
        ann = ast.parse(ann.value, mode="eval").body
    return {n.id for n in ast.walk(ann) if isinstance(n, ast.Name)} | \
           {n.attr for n in ast.walk(ann) if isinstance(n, ast.Attribute)}


def scan(source, path, classes):
    """One 'path:line: message' per finding."""
    tree = ast.parse(source)
    out = []
    # attrs the file itself sets / probes: m.x = ..., setattr(m, 'x', ..), hasattr(m, 'x')
    ok = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("hasattr", "setattr") \
                and len(n.args) >= 2 and isinstance(n.args[1], ast.Constant):
            ok.add(n.args[1].value)
    for fn in (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))):
        tracked = {}   # param name -> allowed attrs
        for a in fn.args.args + fn.args.kwonlyargs:
            if a.annotation is None:
                continue
            for t in _types(a.annotation):
                if t.endswith("Metadata") and "Common" not in t:
                    if t in classes:
                        tracked[a.arg] = classes[t]
                    else:
                        out.append(f"{path}:{a.lineno}: {a.arg}: annotated {t}, not found in the pinned vLLM tree")
        if not tracked:
            continue
        for n in ast.walk(fn):
            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load) and isinstance(n.value, ast.Name) \
                    and n.value.id in tracked and n.attr not in tracked[n.value.id] | ok \
                    and not (n.attr.startswith("__") and n.attr.endswith("__")):
                out.append(f"{path}:{n.lineno}: {n.value.id}.{n.attr} is not a field/method of the pinned metadata class")
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr" \
                    and len(n.args) == 2 and isinstance(n.args[0], ast.Name) and n.args[0].id in tracked \
                    and isinstance(n.args[1], ast.Constant) and n.args[1].value not in tracked[n.args[0].id] | ok:
                out.append(f"{path}:{n.lineno}: getattr({n.args[0].id}, {n.args[1].value!r}) without default, "
                           f"not on the pinned metadata class")
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-src", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parent.parent))
    a = ap.parse_args()
    backends = Path(a.vllm_src) / "vllm/v1/attention"
    if not backends.is_dir():
        print(f"skip: no pinned vLLM tree at {backends}")
        return 0
    classes = index([p.read_text() for p in backends.rglob("*.py")])
    repo = Path(a.repo)
    findings = []
    for p in sorted(repo.rglob("*.py")):
        rel = p.relative_to(repo).as_posix()
        if rel.startswith((".dryrun/", "sly/tests/", "ci/")) or "attn_metadata" not in (s := p.read_text()):
            continue
        findings += scan(s, rel, classes)
    for f in findings:
        print(f"::error::{f}")
    print(f"attn metadata contract: {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
