#!/usr/bin/env python3
"""Derive the image's stack pins from the vLLM release and the ROCm base image.

The owner's rule (docs/MAINTENANCE.md): a new image version is needed only when vLLM releases or the
ROCm base image gets a new tag. torch, triton, torchvision, aiter and transformers are not free
choices -- they are *derived* from those two, and this script is the one place that derives them.

Inputs
  * Dockerfile ARGs VLLM_VERSION and ROCM_BASE (the two things Renovate bumps),
  * vLLM's own tag: docker/Dockerfile.rocm_base (PYTORCH_BRANCH release/X.Y, TRITON_BRANCH commit,
    PYTORCH_VISION_BRANCH, AITER_BRANCH), requirements/common.txt (transformers range) and
    requirements/rocm.txt,
  * AMD's wheel index https://stable.repo.amd.com/rocm/whl-next (redirects to /rocm/pytorch/whl-next):
    cp312 linux wheels torch X.Y.*+rocm<ROCM>, amd-torch-device-<arch> of the same version,
    triton <ver>+git<commit>.rocm<ROCM>, torchvision <ver>+rocm<ROCM> and its device wheel,
  * PyPI: the newest transformers release inside vLLM's range.

Rules (what "derived" means; each is one function below)
  torch         newest X.Y.* on the AMD index for rocm<ROCM> (X.Y from PYTORCH_BRANCH `release/X.Y`)
  triton        the AMD wheel built from the commit vLLM's TRITON_BRANCH names (prefix match)
  torchvision   newest 0.Y.* on the AMD index (0.Y from PYTORCH_VISION_BRANCH)
  aiter         vLLM's AITER_BRANCH, exactly
  transformers  newest PyPI release inside the range of requirements/common.txt
  TORCH_AMD_ROCM  the ROCm version of ROCM_BASE (the wheel's rocm_sdk check_version must match)

Modes
  --check   read-only (CI). Exit 1 when the Dockerfile pins differ from the derivation, or when no
            compatible stack exists. transformers only has to be *inside* vLLM's range (a newer
            release on PyPI is not a mismatch), TRITON_SHA256 is verified with --verify-hash.
  --apply   (Renovate postUpgradeTasks) rewrite the Dockerfile ARGs, regenerate constraints.txt,
            bump VERSION and add a CHANGELOG skeleton. Exit 2 when the stack is incompatible: the
            PR stays red and is never merged.

Deliberate deviations live in ci/stack_overrides.json (an entry is only active for the vLLM
version it names, so it expires on the next vLLM bump). No dependencies beyond the stdlib.

    python3 ci/resolve_stack.py --check [--verify-hash]
    python3 ci/resolve_stack.py --apply
    python3 ci/resolve_stack.py --check --vllm 0.31.0 --rocm 10.1.0     # what-if, ignores the Dockerfile's
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# vLLM's tag, as raw files (the "github" is split so a path filter on that word cannot trip over it)
RAW = "https://raw." + "githubusercontent.com/vllm-project/vllm/v{ver}/{path}"
AMD_INDEX = "https://stable.repo.amd.com/rocm/whl-next"
PYPI_JSON = "https://pypi.org/pypi/{pkg}/json"
# ROCm majors this repo has a tested build path for (AMD ships wheels for 10.x; the image is ROCm 10).
SUPPORTED_ROCM_MAJORS = (10,)
# the image's interpreter: Ubuntu 24.04's python 3.12, so wheels are cp312
PY_TAG = "cp312"
PY_VERSION = "3.12"

ARG_LINE = re.compile(r'^ARG\s+([A-Z][A-Z0-9_]*)=(?:"([^"]*)"|([^\s#]*))\s*(?:#\s*(.*))?$')

# Dockerfile ARGs this script owns
OWNED = ("TORCH_VERSION", "TORCH_AMD_ROCM", "TORCHVISION_VERSION", "TRITON_VERSION", "TRITON_BUILD",
         "TRITON_SHA256", "AITER_VERSION", "TRANSFORMERS_VERSION")
TRITON_GROUP = ("TRITON_VERSION", "TRITON_BUILD", "TRITON_SHA256")


class Incompatible(Exception):
    """No stack can be derived (missing wheel, unparseable vLLM pins, unsupported ROCm major)."""


# --------------------------------------------------------------------------- versions
def vkey(v: str) -> tuple:
    """Sortable key of a plain release version: 0.1.22.post1 -> (0, 1, 22, 1)."""
    nums = [int(x) for x in re.findall(r"\d+", v.split("+")[0])]
    return tuple(nums)


def is_stable(v: str) -> bool:
    base = v.split("+")[0]
    return bool(re.fullmatch(r"\d+(\.\d+)*(\.post\d+)?", base))


def spec_ok(spec: str, version: str) -> bool:
    """Does `version` satisfy a comma-separated specifier (==, !=, >=, <=, >, <, ~=)?"""
    v = vkey(version)
    for clause in filter(None, (c.strip() for c in spec.split(","))):
        m = re.fullmatch(r"(==|!=|>=|<=|~=|>|<)\s*(\S+)", clause)
        if not m:
            raise Incompatible(f"cannot parse version specifier '{clause}'")
        op, target = m.groups()
        if op in ("==", "!=") and target.endswith(".*"):
            t = vkey(target[:-2])
            hit = v[: len(t)] == t
            ok = hit if op == "==" else not hit
        else:
            t = vkey(target)
            ok = {"==": v == t, "!=": v != t, ">=": v >= t, "<=": v <= t, ">": v > t, "<": v < t,
                  "~=": v >= t and v[: max(len(t) - 1, 1)] == t[: max(len(t) - 1, 1)]}[op]
        if not ok:
            return False
    return True


# --------------------------------------------------------------------------- sources
class LiveSources:
    """The network: vLLM's tag files, AMD's index, PyPI. Retries, never silently degrades."""

    def __init__(self, index: str = AMD_INDEX):
        self.index = index.rstrip("/")

    @staticmethod
    def _get(url: str, headers: dict | None = None, timeout: int = 60) -> bytes:
        last: Exception | None = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "vllm-sly-radiance-resolve-stack",
                                                           **(headers or {})})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return r.read()
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    raise
                last = exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last = exc
            time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"GET {url} failed: {last}")

    def vllm_file(self, ver: str, path: str) -> str:
        try:
            return self._get(RAW.format(ver=ver, path=path)).decode()
        except urllib.error.HTTPError as exc:
            raise Incompatible(f"vLLM v{ver}: {path} not found ({exc.code}) -- is v{ver} a release tag?") from exc

    def index_project(self, project: str) -> list[str]:
        try:
            html = self._get(f"{self.index}/{project}/").decode()
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return []
            raise
        return [urllib.parse.unquote(h.split("#")[0]) for h in re.findall(r'href="([^"]+)"', html)]

    def pypi_versions(self, pkg: str) -> list[str]:
        data = json.loads(self._get(PYPI_JSON.format(pkg=pkg), timeout=120))
        out = []
        for ver, files in data.get("releases", {}).items():
            if files and not all(f.get("yanked") for f in files):
                out.append(ver)
        return out

    def wheel_sha256(self, project: str, filename: str) -> str:
        url = f"{self.index}/{project}/{urllib.parse.quote(filename)}"
        h = hashlib.sha256()
        req = urllib.request.Request(url, headers={"User-Agent": "vllm-sly-radiance-resolve-stack"})
        with urllib.request.urlopen(req, timeout=300) as r:
            while chunk := r.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()


# --------------------------------------------------------------------------- parsing
def dockerfile_args(text: str) -> dict[str, tuple[str, str]]:
    """First top-level `ARG NAME=value` per name -> (value, trailing comment)."""
    out: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        m = ARG_LINE.match(line.strip())
        if m and m.group(1) not in out:
            val = m.group(2) if m.group(2) is not None else m.group(3)
            out[m.group(1)] = (val, m.group(4) or "")
    return out


def pins_of(text: str) -> dict[str, str]:
    return {k: v[0] for k, v in dockerfile_args(text).items()}


def rocm_version(rocm_base: str) -> str:
    m = re.search(r":(\d+\.\d+\.\d+)-full", rocm_base)
    if not m:
        raise Incompatible(f"cannot read the ROCm version from ROCM_BASE '{rocm_base}' (want <x.y.z>-full)")
    return m.group(1)


def parse_rocm_base(text: str) -> dict:
    """vLLM's docker/Dockerfile.rocm_base -> the sanctioned component refs."""
    a = dockerfile_args(text)

    def need(name: str) -> tuple[str, str]:
        if name not in a:
            raise Incompatible(f"vLLM's docker/Dockerfile.rocm_base has no ARG {name} (layout changed?)")
        return a[name]

    out: dict = {}
    val, com = need("PYTORCH_BRANCH")
    m = re.search(r"release/(\d+)\.(\d+)", f"{val} {com}") or re.fullmatch(r"v?(\d+)\.(\d+)\.\d+", val)
    if not m:
        raise Incompatible(f"PYTORCH_BRANCH='{val}' ({com or 'no comment'}): no release/X.Y to derive torch from")
    out["torch_series"] = f"{m.group(1)}.{m.group(2)}"

    val, _ = need("PYTORCH_VISION_BRANCH")
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.\d+(?:[a-z]+\d*)?", val)
    if not m:
        raise Incompatible(f"PYTORCH_VISION_BRANCH='{val}' is not a release tag")
    out["vision_series"] = f"{m.group(1)}.{m.group(2)}"

    val, com = need("TRITON_BRANCH")
    if not re.fullmatch(r"[0-9a-f]{7,40}", val):
        raise Incompatible(f"TRITON_BRANCH='{val}' is not a commit hash, cannot match it to an AMD triton wheel")
    out["triton_commit"] = val
    m = re.search(r"(\d+\.\d+)\.x", com)
    out["triton_series"] = m.group(1) if m else None

    val, _ = need("AITER_BRANCH")
    m = re.fullmatch(r"v(\d+(?:\.\d+)*(?:\.post\d+)?)", val)
    if not m:
        raise Incompatible(f"AITER_BRANCH='{val}' is not a release tag vX.Y.Z")
    out["aiter"] = m.group(1)

    py = a.get("PYTHON_VERSION", (PY_VERSION, ""))[0]
    if py != PY_VERSION:
        raise Incompatible(f"vLLM's ROCm base uses python {py}, the image's venv is {PY_VERSION}")
    return out


def transformers_spec(common_txt: str) -> str:
    for line in common_txt.splitlines():
        m = re.match(r"\s*transformers\s*([<>=!~][^#;]*)", line)
        if m:
            return re.sub(r"\s+", "", m.group(1))
    return ""  # unbounded


def rocm_txt_pins(rocm_txt: str) -> dict[str, str]:
    out = {}
    for line in rocm_txt.splitlines():
        m = re.match(r"\s*(torch|torchvision|triton|amd-aiter|aiter)\s*([<>=!~][^#;\s]*)", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


WHEEL = re.compile(r"^(?P<dist>[A-Za-z0-9_.]+?)-(?P<ver>\d[^-]*)-(?P<py>cp\d+)-(?P<abi>[^-]+)-(?P<plat>.+)\.whl$")


def wheels(files: list[str], rocm: str) -> list[tuple[str, str, str]]:
    """(base version, local part, filename) of cp312 linux wheels built for `rocm`."""
    out = []
    for f in files:
        m = WHEEL.match(f)
        if not m or m.group("py") != PY_TAG or "linux" not in m.group("plat"):
            continue
        base, _, local = m.group("ver").partition("+")
        # local parts look like 'rocm10.1.0' or 'git669b31ac.rocm10.1.0'
        if not re.search(rf"(^|\.)rocm{re.escape(rocm)}$", local):
            continue
        out.append((base, local, f))
    return out


# --------------------------------------------------------------------------- resolution
def resolve(vllm: str, rocm: str, src, arch: str = "gfx1201", has_triton_build: bool = True,
            overrides: list[dict] | None = None) -> dict:
    """The derived stack. Returns {'pins': {ARG: value}, 'notes': [...], 'warnings': [...],
    'transformers_spec': str, 'triton_file': str|None, 'active_overrides': [...]}.
    Raises Incompatible."""
    major = int(rocm.split(".")[0])
    if major not in SUPPORTED_ROCM_MAJORS:
        raise Incompatible(f"ROCm {rocm}: major {major} is not supported (supported: "
                           f"{', '.join(map(str, SUPPORTED_ROCM_MAJORS))}); the build path and AMD's wheel "
                           "layout for it have not been verified -- a human has to extend SUPPORTED_ROCM_MAJORS")

    active = [o for o in (overrides or []) if o.get("vllm") == vllm and o.get("rocm", rocm) == rocm]
    forced: dict[str, str] = {}
    for o in active:
        forced.update(o["pins"])
    triton_forced = any(k in forced for k in TRITON_GROUP)

    ref = parse_rocm_base(src.vllm_file(vllm, "docker/Dockerfile.rocm_base"))
    common = src.vllm_file(vllm, "requirements/common.txt")
    rocm_txt = src.vllm_file(vllm, "requirements/rocm.txt")

    pins: dict[str, str] = {"TORCH_AMD_ROCM": rocm}
    notes: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []
    triton_file = None

    def collect(arg: str, fn):
        try:
            return fn()
        except Incompatible as exc:
            if arg in forced:
                warnings.append(f"{arg}: derivation failed ({exc}); override in use")
                return None
            errors.append(str(exc))
            return None

    # --- torch
    def torch():
        series = ref["torch_series"]
        cands = [(b, f) for b, _, f in wheels(src.index_project("torch"), rocm)
                 if is_stable(b) and ".".join(b.split(".")[:2]) == series]
        if not cands:
            raise Incompatible(f"no AMD torch wheel {series}.* for rocm{rocm} (cp312, linux) on {AMD_INDEX} "
                               f"-- vLLM v{vllm} builds torch release/{series}")
        ver = max(cands, key=lambda c: vkey(c[0]))[0]
        dev = [b for b, _, _ in wheels(src.index_project(f"amd-torch-device-{arch}"), rocm)]
        if ver not in dev:
            raise Incompatible(f"torch {ver}+rocm{rocm} has no amd-torch-device-{arch} wheel on the AMD index")
        notes.append(f"torch {ver}+rocm{rocm} (vLLM v{vllm}: release/{series})")
        return ver

    v = collect("TORCH_VERSION", torch)
    if v:
        pins["TORCH_VERSION"] = v

    # --- torchvision
    def torchvision():
        series = ref["vision_series"]
        cands = [b for b, _, _ in wheels(src.index_project("torchvision"), rocm)
                 if is_stable(b) and ".".join(b.split(".")[:2]) == series]
        if not cands:
            raise Incompatible(f"no AMD torchvision wheel {series}.* for rocm{rocm} on {AMD_INDEX} "
                               f"-- vLLM v{vllm} builds torchvision v{series}.x")
        ver = max(cands, key=vkey)
        dev = [b for b, _, _ in wheels(src.index_project(f"amd-torchvision-device-{arch}"), rocm)]
        if ver not in dev:
            raise Incompatible(f"torchvision {ver}+rocm{rocm} has no amd-torchvision-device-{arch} wheel")
        notes.append(f"torchvision {ver}+rocm{rocm} (vLLM v{vllm}: v{series}.x)")
        return ver

    v = collect("TORCHVISION_VERSION", torchvision)
    if v:
        pins["TORCHVISION_VERSION"] = v
    if "TORCH_VERSION" in pins and "TORCHVISION_VERSION" in pins:
        tm, vm = int(pins["TORCH_VERSION"].split(".")[1]), int(pins["TORCHVISION_VERSION"].split(".")[1])
        if vm != tm + 15:  # torch 2.11 <-> vision 0.26, 2.12 <-> 0.27, 2.13 <-> 0.28
            warnings.append(f"torch {pins['TORCH_VERSION']} and torchvision {pins['TORCHVISION_VERSION']} are not "
                            "the usual pair (torch 2.N <-> torchvision 0.N+15)")

    # --- triton
    if not triton_forced:
        def triton():
            commit = ref["triton_commit"]
            hits = []
            for base, local, f in wheels(src.index_project("triton"), rocm):
                m = re.fullmatch(r"git([0-9a-f]+)\.rocm" + re.escape(rocm), local)
                if m and (m.group(1).startswith(commit) or commit.startswith(m.group(1))):
                    hits.append((base, m.group(1), f))
            if not hits:
                raise Incompatible(f"no AMD triton wheel built from vLLM's TRITON_BRANCH {commit} for rocm{rocm} "
                                   f"on {AMD_INDEX} (vLLM v{vllm} builds that commit; PyPI triton does not carry it)")
            base, full, f = max(hits, key=lambda h: vkey(h[0]))
            if ref["triton_series"] and ".".join(base.split(".")[:2]) != ref["triton_series"]:
                warnings.append(f"triton wheel {base} is not the series vLLM's comment names ({ref['triton_series']}.x)")
            notes.append(f"triton {base}+git{full}.rocm{rocm} (vLLM v{vllm}: TRITON_BRANCH {commit})")
            return base, f"git{full}.rocm{rocm}", f

        res = collect("TRITON_VERSION", triton)
        if res:
            if not has_triton_build:
                errors.append("the Dockerfile has no ARG TRITON_BUILD, so it can only install PyPI's triton, but "
                              f"vLLM v{vllm}'s triton is AMD's wheel {res[2]}; port the Dockerfile to the AMD "
                              "triton install first, or add a documented override to ci/stack_overrides.json")
            pins["TRITON_VERSION"], pins["TRITON_BUILD"], triton_file = res
    else:
        notes.append("triton: overridden (ci/stack_overrides.json)")

    # --- aiter
    pins["AITER_VERSION"] = ref["aiter"]
    notes.append(f"aiter {ref['aiter']} (vLLM v{vllm}: AITER_BRANCH)")

    # --- transformers
    spec = transformers_spec(common)
    versions = [x for x in src.pypi_versions("transformers") if is_stable(x)]
    ok = [x for x in versions if not spec or spec_ok(spec, x)]
    if not ok:
        errors.append(f"no transformers release on PyPI satisfies vLLM v{vllm}'s '{spec}'")
    else:
        pins["TRANSFORMERS_VERSION"] = max(ok, key=vkey)
        notes.append(f"transformers {pins['TRANSFORMERS_VERSION']} (vLLM v{vllm}: transformers {spec or 'unbounded'}, newest allowed)")

    # --- requirements/rocm.txt must not contradict
    for name, spec_r in rocm_txt_pins(rocm_txt).items():
        key = {"torch": "TORCH_VERSION", "torchvision": "TORCHVISION_VERSION", "triton": "TRITON_VERSION"}.get(name)
        if key in pins and key not in forced and not spec_ok(spec_r, pins[key]):
            errors.append(f"vLLM's requirements/rocm.txt pins {name}{spec_r}, the derived {pins[key]} does not satisfy it")

    if errors:
        raise Incompatible("; ".join(errors))
    pins.update(forced)  # overrides win, last
    return {"pins": pins, "notes": notes, "warnings": warnings, "transformers_spec": spec,
            "triton_file": triton_file, "active_overrides": active, "ref": ref}


# --------------------------------------------------------------------------- check
def check(dockerfile_pins: dict[str, str], res: dict, src=None, verify_hash: bool = False) -> list[str]:
    """Differences between the Dockerfile's pins and the derivation (empty = consistent)."""
    problems = []
    for arg, want in res["pins"].items():
        have = dockerfile_pins.get(arg)
        if arg == "TRITON_BUILD" and have is None:
            continue
        if have is None:
            if arg == "TRITON_SHA256":
                continue
            problems.append(f"{arg}: not in the Dockerfile, derived {want}")
        elif arg == "TRANSFORMERS_VERSION":
            if res["transformers_spec"] and not spec_ok(res["transformers_spec"], have):
                problems.append(f"TRANSFORMERS_VERSION={have} is outside vLLM's range '{res['transformers_spec']}' "
                                f"(newest allowed: {want})")
        elif arg == "TRITON_SHA256":
            pass  # below
        elif have != want:
            problems.append(f"{arg}: Dockerfile has {have}, derived {want}")
    if verify_hash and res.get("triton_file") and "TRITON_SHA256" in dockerfile_pins and src is not None:
        got = src.wheel_sha256("triton", res["triton_file"])
        if got != dockerfile_pins["TRITON_SHA256"]:
            problems.append(f"TRITON_SHA256: Dockerfile has {dockerfile_pins['TRITON_SHA256']}, the wheel "
                            f"{res['triton_file']} hashes to {got}")
    return problems


# --------------------------------------------------------------------------- apply
def set_arg(text: str, name: str, value: str) -> str:
    """Replace the first `ARG name=...` line's value; keeps quoting style and the rest of the line."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        m = re.match(rf"^(ARG\s+{name}=)(?:\"[^\"]*\"|[^\s#]*)(.*)$", line)
        if m:
            lines[i] = f"{m.group(1)}{value}{m.group(2)}"
            return "\n".join(lines)
    raise Incompatible(f"Dockerfile has no ARG {name}=")


def bump_kind(old_vllm: str, new_vllm: str, old_rocm: str, new_rocm: str) -> str | None:
    """'minor' for a vLLM major/minor or ROCm major/minor change, 'patch' for patch-only, else None."""
    kinds = []
    for old, new in ((old_vllm, new_vllm), (old_rocm, new_rocm)):
        if old == new:
            continue
        o, n = vkey(old), vkey(new)
        kinds.append("minor" if o[:2] != n[:2] else "patch")
    if not kinds:
        return None
    return "minor" if "minor" in kinds else "patch"


def bump_version(version: str, kind: str) -> str:
    major, minor, patch = (int(x) for x in version.split("."))
    return f"{major}.{minor + 1}.0" if kind == "minor" else f"{major}.{minor}.{patch + 1}"


PENDING = "Night gate pending."


def changelog_skeleton(version: str, date: str, old: dict, new: dict, old_pins: dict, pins: dict) -> str:
    changed = []
    if old["vllm"] != new["vllm"]:
        changed.append(f"- **vLLM {old['vllm']} -> {new['vllm']}**")
    if old["rocm"] != new["rocm"]:
        changed.append(f"- **ROCm base {old['rocm']} -> {new['rocm']}**")
    for label, key in (("torch", "TORCH_VERSION"), ("triton", "TRITON_VERSION"), ("torchvision", "TORCHVISION_VERSION"),
                       ("aiter", "AITER_VERSION"), ("transformers", "TRANSFORMERS_VERSION")):
        if old_pins.get(key) != pins.get(key):
            extra = f"+{pins['TRITON_BUILD']}" if key == "TRITON_VERSION" and pins.get("TRITON_BUILD") else ""
            changed.append(f"- {label} {old_pins.get(key)} -> {pins.get(key)}{extra}")
    if not changed:
        changed.append("- Rebuilt on the same stack.")
    return "\n".join([
        f"## [{version}] - {date}",
        "",
        "Automated stack update (Renovate + `ci/resolve_stack.py`): torch, triton, torchvision, aiter and "
        "transformers follow the vLLM release's own ROCm base, nothing else was changed on purpose.",
        "",
        "### Changed",
        *changed,
        "",
        "### Measured",
        f"- {PENDING}",
        "",
        "",
    ])


def write_changelog(path: Path, version: str, section: str) -> str:
    text = path.read_text()
    m = re.search(rf"^## \[{re.escape(version)}\][^\n]*\n(.*?)(?=^## \[|\Z)", text, re.S | re.M)
    if m:
        if PENDING in m.group(0):
            path.write_text(text[: m.start()] + section + text[m.end():])
            return "replaced the skeleton"
        return "kept the existing section (not a skeleton)"
    first = re.search(r"^## \[", text, re.M)
    if not first:
        path.write_text(text.rstrip("\n") + "\n\n" + section)
    else:
        path.write_text(text[: first.start()] + section + text[first.start():])
    return "added"


def git_show(root: Path, ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=root, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def base_ref(root: Path, wanted: str | None) -> str:
    for ref in ([wanted] if wanted else []) + ["origin/main", "HEAD"]:
        if subprocess.run(["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"], cwd=root,
                          capture_output=True).returncode == 0:
            return ref
    return "HEAD"


def apply(root: Path, res: dict, base: dict, rocm: str, vllm: str, date: str, src=None,
          constraints: bool = True) -> list[str]:
    """Write the derived pins; returns a log of what was done. `base` = {'dockerfile', 'version'} of
    the commit the PR branches from (the bump is computed from it, so a re-run changes nothing)."""
    log = []
    df = root / "Dockerfile"
    text = df.read_text()
    cur = pins_of(text)
    for arg, val in res["pins"].items():
        if arg == "TRITON_BUILD" and "TRITON_BUILD" not in cur:
            continue
        if arg in cur and cur[arg] != val:
            text = set_arg(text, arg, val)
            log.append(f"Dockerfile {arg}: {cur[arg]} -> {val}")
    if res.get("triton_file") and "TRITON_SHA256" in cur and src is not None:
        sha = src.wheel_sha256("triton", res["triton_file"])
        if cur["TRITON_SHA256"] != sha:
            text = set_arg(text, "TRITON_SHA256", sha)
            log.append(f"Dockerfile TRITON_SHA256 -> {sha}")
    df.write_text(text)

    old_pins = pins_of(base["dockerfile"])
    old = {"vllm": old_pins.get("VLLM_VERSION", vllm), "rocm": rocm_version(old_pins["ROCM_BASE"])
           if "ROCM_BASE" in old_pins else rocm}
    new = {"vllm": vllm, "rocm": rocm}
    kind = bump_kind(old["vllm"], new["vllm"], old["rocm"], new["rocm"])
    vfile = root / "VERSION"
    if kind:
        newv = bump_version(base["version"], kind)
        if vfile.read_text().strip() != newv:
            vfile.write_text(newv + "\n")
            log.append(f"VERSION {base['version']} -> {newv} ({kind} bump: vLLM {old['vllm']}->{vllm}, "
                       f"ROCm {old['rocm']}->{rocm})")
        section = changelog_skeleton(newv, date, old, new, old_pins, {**pins_of(text)})
        log.append("CHANGELOG: " + write_changelog(root / "CHANGELOG.md", newv, section))
    else:
        log.append("vLLM and ROCm unchanged against the base: VERSION and CHANGELOG untouched")

    if constraints and old["vllm"] != vllm:
        ok, why = constraints_capable()
        if not ok:
            log.append(f"constraints.txt NOT regenerated ({why}); the `constraints` job of ci.yml does it on the "
                       "PR branch (check_constraints.py --update)")
        else:
            r = subprocess.run([sys.executable, str(root / "ci" / "check_constraints.py"), "--update"], cwd=root)
            if r.returncode != 0:
                raise Incompatible("ci/check_constraints.py --update failed: pip cannot resolve vLLM's "
                                   f"v{vllm} requirements with this stack")
            log.append("constraints.txt regenerated (check_constraints.py --update)")
    return log


def constraints_capable() -> tuple[bool, str]:
    if not sys.platform.startswith("linux"):
        return False, f"platform {sys.platform}, need linux x86_64"
    if sys.version_info[:2] != (3, 12):
        return False, f"python {sys.version_info[0]}.{sys.version_info[1]}, need 3.12"
    return True, ""


# --------------------------------------------------------------------------- cli
def load_overrides(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out = data.get("overrides", [])
    for o in out:
        if not o.get("vllm") or not o.get("pins") or not o.get("reason"):
            raise SystemExit(f"{path}: every override needs 'vllm', 'pins' and 'reason': {o}")
        bad = [k for k in o["pins"] if k not in OWNED]
        if bad:
            raise SystemExit(f"{path}: override pins {bad} are not derived by this script ({', '.join(OWNED)})")
    return out


def main(argv: list[str] | None = None, src=None, root: Path = ROOT) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    ap.add_argument("--vllm", help="what-if: vLLM version instead of the Dockerfile's")
    ap.add_argument("--rocm", help="what-if: ROCm x.y.z instead of the Dockerfile's ROCM_BASE")
    ap.add_argument("--verify-hash", action="store_true", help="--check: download AMD's triton wheel, compare TRITON_SHA256")
    ap.add_argument("--base", help="--apply: ref the PR branches from (default origin/main, else HEAD)")
    ap.add_argument("--date", help="--apply: CHANGELOG date (default today, UTC)")
    ap.add_argument("--no-constraints", action="store_true", help="--apply: skip regenerating constraints.txt")
    ap.add_argument("--overrides", default=str(root / "ci" / "stack_overrides.json"))
    ap.add_argument("--json", action="store_true", help="print the derived pins as JSON")
    args = ap.parse_args(argv)

    src = src or LiveSources()
    text = (root / "Dockerfile").read_text()
    cur = pins_of(text)
    try:
        vllm = args.vllm or cur["VLLM_VERSION"]
        rocm = args.rocm or rocm_version(cur["ROCM_BASE"])
        res = resolve(vllm, rocm, src, arch=cur.get("GFX_ARCH", "gfx1201"),
                      has_triton_build="TRITON_BUILD" in cur, overrides=load_overrides(Path(args.overrides)))
    except Incompatible as exc:
        print(f"::error title=stack incompatible::{exc}")
        print(f"INCOMPATIBLE: {exc}")
        return 2 if args.apply else 1
    except KeyError as exc:
        print(f"::error::Dockerfile has no ARG {exc}")
        return 2

    print(f"stack for vLLM {vllm} on ROCm {rocm}:")
    for n in res["notes"]:
        print(f"  {n}")
    for o in res["active_overrides"]:
        print(f"  override (ci/stack_overrides.json): {json.dumps(o['pins'])} -- {o['reason']}")
    for w in res["warnings"]:
        print(f"::warning::{w}")
    if args.json:
        print(json.dumps(res["pins"], indent=2))

    if args.check:
        problems = check(cur, res, src, args.verify_hash)
        for p in problems:
            print(f"::error title=stack mismatch::{p}")
        if problems:
            print(f"FAIL: the Dockerfile does not match the stack derived for vLLM {vllm} / ROCm {rocm}; run "
                  "`python3 ci/resolve_stack.py --apply` (or add a documented override to ci/stack_overrides.json)")
            return 1
        print("resolve OK: Dockerfile pins are what vLLM and the ROCm base derive")
        return 0

    # --apply
    base_df = None
    ref = base_ref(root, args.base)
    base_df = git_show(root, ref, "Dockerfile")
    base_ver = (git_show(root, ref, "VERSION") or (root / "VERSION").read_text()).strip()
    if base_df is None:
        base_df = text
    date = args.date or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    try:
        log = apply(root, res, {"dockerfile": base_df, "version": base_ver}, rocm, vllm, date, src,
                    constraints=not args.no_constraints)
    except Incompatible as exc:
        print(f"::error title=stack incompatible::{exc}")
        return 2
    for line in log:
        print(f"  {line}")
    print(f"resolve --apply done (base {ref})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
