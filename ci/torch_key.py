#!/usr/bin/env python3
"""Print the ghcr.io tag of the torch wheel image the Dockerfile's torch-wheel target produces.

build.yml uses the tag to take the ~2 h torch compile out of the runner's local BuildKit cache:
if ghcr.io/slybase/vllm-sly-radiance-torch:<tag> exists it is passed as TORCH_FROM, otherwise the
torch-wheel target is built from source and pushed under that tag for the next build.

The tag is `<torch>-rocm<major.minor>-<gfx>-<hash>`, the hash over everything that decides what the
wheel contains: the buildbase + torch-build + torch-wheel stage definitions (comments and blank
lines stripped, so a reworded comment does not cost a rebuild) and the values of ROCM_BASE,
GFX_ARCH and TORCH_VERSION. A torch bump, a new base digest or an edited torch build flag each
give a new tag; an aiter, vLLM or transformers bump does not.

    python3 ci/torch_key.py                      # tag for the Dockerfile defaults
    python3 ci/torch_key.py --rocm-base IMAGE    # tag for a ROCM_BASE override (build.yml input)
"""
import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_pins import ROOT, read_pins  # noqa: E402

START = re.compile(r"^FROM\s+\S+\s+AS\s+buildbase\s*$")
END = re.compile(r"^FROM\s+\$\{TORCH_FROM\}\s+AS\s+torch\s*$")


def torch_stages(dockerfile):
    lines, inside = [], False
    for line in dockerfile.read_text().splitlines():
        if START.match(line):
            inside = True
        elif END.match(line):
            break
        if inside:
            s = line.strip()
            if s and not s.startswith("#"):
                lines.append(" ".join(s.split()))
    if not lines:
        sys.exit("torch_key: buildbase .. 'FROM ${TORCH_FROM} AS torch' not found in the Dockerfile")
    return lines


def torch_key(dockerfile=ROOT / "Dockerfile", rocm_base=None):
    pins = read_pins(dockerfile)
    rocm_base = rocm_base or pins["ROCM_BASE"]
    h = hashlib.sha256()
    for part in [*torch_stages(dockerfile), rocm_base, pins["GFX_ARCH"], pins["TORCH_VERSION"]]:
        h.update(part.encode() + b"\n")
    m = re.match(r"^[^:]+:(\d+\.\d+)\.", rocm_base)
    rocm_mm = m.group(1) if m else "x"
    return f"{pins['TORCH_VERSION']}-rocm{rocm_mm}-{pins['GFX_ARCH']}-{h.hexdigest()[:12]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rocm-base", default="", help="ROCM_BASE override (empty = Dockerfile default)")
    print(torch_key(rocm_base=ap.parse_args().rocm_base or None))


if __name__ == "__main__":
    main()
