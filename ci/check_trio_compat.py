#!/usr/bin/env python3
"""Trio compatibility gate: torch / triton / torchvision in the Dockerfile must fit together.

The official trio is declared by the torch wheel itself: its PyPI metadata
(`requires_dist`) pins `triton==<exact>` and `torchvision==<exact>` -- that is
the sanctioned pair for the pinned torch release. The Dockerfile is the single
source of truth for the pins (`ci/read_pins.py`), so this check reads the pins
from there and compares them against the metadata of the pinned releases --
no GPU, no build, seconds.

Two classes of findings:

  hard fail (exit 1)
    * the pinned triton is not the `triton==X` the pinned torch wheel requires
      (torch 2.11.0 requires `triton==3.6.0` exactly; the torch build links
      against that exact triton, a different one is an ABI gamble);
    * the pinned torchvision declares a torch floor the pinned torch cannot meet
      (torchvision 0.29.1 is ABI-stable only w.r.t. torch 2.14 and declares
      `torch>=2.14`; torch 2.11 fails that range). The from-source build
      bypasses pip's requirement check, so this breaks at compile time or at
      first import -- invisible to every other PR check (the PR #69 shape: a
      renovate group PR that bumps vision and triton from a newer torch line
      while the torch `allowedVersions` hold keeps 2.11).

  warn (does not fail the run)
    * the pinned torchvision is not the exact version its torch wheel pairs
      with it (sanctioned-trio drift: built from source, may still work --
      the GPU gate is the real proof);
    * a PyPI outage: the check degrades to a warning instead of a red run that
      is not a real incompatibility (the manual `release.yml` path remains).

The image ships no torchaudio, so there is no pin to compare it against.

    python3 ci/check_trio_compat.py            # human-readable
    python3 ci/check_trio_compat.py --json     # machine-readable (CI)
    python3 ci/check_trio_compat.py --offline  # skip the PyPI lookup (pins only)
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ci"))
from read_pins import read_pins  # noqa: E402

PYPI_URL = "https://pypi.org/pypi/{pkg}/{version}/json"
CACHE_DIR = Path.home() / ".cache" / "trio-compat"
CACHE_TTL_S = 12 * 3600  # a release's metadata does not change

# (PyPI package name, Dockerfile pin name)
COMPONENTS = [
    ("torch", "TORCH_VERSION"),
    ("triton", "TRITON_VERSION"),
    ("torchvision", "TORCHVISION_VERSION"),
]


class Finding:
    def __init__(self, level, component, detail):
        self.level = level  # "fail" | "warn"
        self.component = component
        self.detail = detail

    def to_dict(self):
        return {"level": self.level, "component": self.component, "detail": self.detail}


def _version_key(version):
    """Numeric sort/tuple key for a PEP 440-ish version string."""
    m = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", version.strip())
    if not m:
        return (0,)
    return tuple(int(g) if g is not None else 0 for g in m.groups())


def _parse_requires(requirement):
    """`triton==3.6.0; platform_system == "Linux"` -> ("triton", "==3.6.0")."""
    req = requirement.split(";")[0].strip()
    m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(\S+)$", req)
    if not m:
        return None
    return m.group(1).lower(), m.group(2)


def _spec_satisfied(spec, version):
    """Does `version` satisfy `spec` (`==X`, `!=X`, `>=X`, `>X`, `<=X`, `<X`, `~=X.Y`)?"""
    for clause in spec.split(","):
        m = re.match(r"^(==|!=|>=|<=|~=|>|<)\s*(.+)$", clause.strip())
        if not m:
            return False
        op, target = m.group(1), m.group(2)
        v, t = _version_key(version), _version_key(target)
        if not {
            "==": v == t, "!=": v != t, ">=": v >= t, "<=": v <= t,
            ">": v > t, "<": v < t, "~=": v >= t and v[:2] == t[:2],
        }[op]:
            return False
    return True


def _pypi_requires_dist(pkg, version):
    """PyPI requires_dist for one exact release; None on outage (3 retries)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"{pkg}-{version}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < CACHE_TTL_S:
        return json.loads(cache.read_text())["requires_dist"]
    import urllib.request

    for attempt in range(3):
        try:
            req = urllib.request.Request(
                PYPI_URL.format(pkg=pkg, version=version),
                headers={"User-Agent": "vllm-sly-radiance-trio-compat"},
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
            requires = data["info"]["requires_dist"] or []
            cache.write_text(json.dumps({"requires_dist": requires}))
            return requires
        except Exception:
            if attempt == 2:
                return None
            time.sleep(2 ** attempt)
    return None


def _deps_named(requires, names):
    """The (name, spec) pairs in `requires` for the given package names."""
    out = []
    for r in requires or []:
        parsed = _parse_requires(r)
        if parsed and parsed[0] in names:
            out.append(parsed)
    return out


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--offline", action="store_true", help="skip the PyPI lookup (pins only)")
    args = ap.parse_args(argv)

    findings = []
    pins = read_pins()
    pin_by_name = {docker_name: pins.get(docker_name, "") for _, docker_name in COMPONENTS}

    # ---- always-on part: the pins must exist ---------------------------------------
    for _, docker_name in COMPONENTS:
        if not pins.get(docker_name):
            findings.append(Finding("fail", docker_name, "pin missing from the Dockerfile"))
    if any(f.level == "fail" for f in findings):
        _emit(args, findings, pin_by_name)
        return

    if args.offline:
        _emit(args, findings, pin_by_name)
        return

    # ---- online part: the pinned torch declares the official trio ------------------
    torch_ver = pin_by_name["TORCH_VERSION"]
    requires = _pypi_requires_dist("torch", torch_ver)
    if requires is None:
        findings.append(
            Finding(
                "warn",
                "TORCH_VERSION",
                f"PyPI lookup for torch=={torch_ver} failed; the trio compatibility "
                "cannot be verified (degraded to a warning, not a red run).",
            )
        )
        _emit(args, findings, pin_by_name)
        return

    # triton: the torch wheel pins it EXACTLY -> a mismatch is a hard fail
    triton_pin = pin_by_name["TRITON_VERSION"]
    for name, spec in _deps_named(requires, ("triton",)):
        if not _spec_satisfied(spec, triton_pin):
            findings.append(
                Finding(
                    "fail",
                    "TRITON_VERSION",
                    f"torch=={torch_ver} requires triton {spec}, the Dockerfile pins "
                    f"triton {triton_pin} -- the torch build links against its own triton, "
                    "a different one is an ABI gamble.",
                )
            )

    # torchvision: the torch wheel pairs it EXACTLY -> a drift is a warn (from-source
    # build, may still work), but the pinned torchvision's own torch floor is a hard
    # fail: the from-source build bypasses pip's check, so it breaks at compile time
    # or at first import (the PR #69 shape).
    tv_pin = pin_by_name["TORCHVISION_VERSION"]
    for name, spec in _deps_named(requires, ("torchvision",)):
        if not _spec_satisfied(spec, tv_pin):
            m = re.match(r"^==\s*(.+)$", spec)
            if m:
                findings.append(
                    Finding(
                        "warn",
                        "TORCHVISION_VERSION",
                        f"torch=={torch_ver} officially pairs torchvision {m.group(1)} "
                        f"(the Dockerfile pins {tv_pin}, built from source against "
                        f"{torch_ver} -- sanctioned-trio drift, the GPU gate is the proof).",
                    )
                )
            else:
                findings.append(
                    Finding(
                        "warn",
                        "TORCHVISION_VERSION",
                        f"torch=={torch_ver} declares torchvision {spec}, the Dockerfile "
                        f"pins torchvision {tv_pin}.",
                    )
                )

    tv_requires = _pypi_requires_dist("torchvision", tv_pin)
    if tv_requires is None:
        findings.append(
            Finding(
                "warn",
                "TORCHVISION_VERSION",
                f"PyPI lookup for torchvision=={tv_pin} failed; its torch floor "
                "cannot be verified (degraded to a warning, not a red run).",
            )
        )
    else:
        for name, spec in _deps_named(tv_requires, ("torch",)):
            if _spec_satisfied(spec, torch_ver):
                continue
            if spec.startswith((">=", "~=", ">")):
                findings.append(
                    Finding(
                        "fail",
                        "TORCHVISION_VERSION",
                        f"torchvision {tv_pin} requires torch {spec} (its ABI line), the "
                        f"Dockerfile pins torch {torch_ver} -- the from-source build "
                        "bypasses pip's check, so this breaks at compile time or at "
                        "first import.",
                    )
                )
            else:
                findings.append(
                    Finding(
                        "warn",
                        "TORCHVISION_VERSION",
                        f"torchvision {tv_pin} declares torch {spec}, the Dockerfile pins "
                        f"torch {torch_ver} -- not the officially paired version.",
                    )
                )

    _emit(args, findings, pin_by_name)


def _emit(args, findings, pin_by_name):
    worst = "pass"
    if any(f.level == "fail" for f in findings):
        worst = "fail"
    elif any(f.level == "warn" for f in findings):
        worst = "warn"

    if args.json:
        print(
            json.dumps(
                {"status": worst, "pins": pin_by_name, "findings": [f.to_dict() for f in findings]},
                indent=2,
            )
        )
    else:
        print("pins: " + "  ".join(f"{n}={v}" for n, v in pin_by_name.items()))
        if not findings:
            print("trio OK: pins fit together (checked against the pinned releases' metadata)")
        for f in findings:
            prefix = "::error::" if f.level == "fail" else "::warning::"
            print(f"{prefix}{f.component}: {f.detail}")

    sys.exit(1 if worst == "fail" else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
