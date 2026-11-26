#!/usr/bin/env python3
"""Print the component pins from the Dockerfile's top-level `ARG NAME=value` defaults.

The Dockerfile is the single source of truth for every pin (torch, triton, vLLM, aiter, ...);
CI reads them from here instead of keeping a second copy that would drift. Only the first
`ARG NAME=value` per name counts -- the later bare `ARG NAME` re-declarations inside stages
carry no value.

    python3 ci/read_pins.py            # KEY=value lines
    python3 ci/read_pins.py --shell    # export KEY='value' lines (for `eval`)
    python3 ci/read_pins.py --json
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARG_RE = re.compile(r"^ARG\s+([A-Z][A-Z0-9_]*)=(.*?)\s*$")


def read_pins(dockerfile=ROOT / "Dockerfile"):
    pins = {}
    for line in dockerfile.read_text().splitlines():
        m = ARG_RE.match(line)
        if m and m.group(1) not in pins:
            pins[m.group(1)] = m.group(2)
    return pins


def vllm_source_ref(pins=None):
    """The vLLM git ref every CI consumer reads: the pinned commit (ARG VLLM_REF) when set, else
    the release tag v<VLLM_VERSION>. One rule, so the patch dry run, the constraints replay and
    the stack resolver all see the tree the image actually builds."""
    pins = read_pins() if pins is None else pins
    return pins.get("VLLM_REF") or f"v{pins['VLLM_VERSION']}"


def main(argv):
    pins = read_pins()
    if "--vllm-ref" in argv:
        print(vllm_source_ref(pins))
    elif "--json" in argv:
        print(json.dumps(pins, indent=2))
    elif "--shell" in argv:
        for k, v in pins.items():
            print(f"export {k}='{v}'")
    else:
        for k, v in pins.items():
            print(f"{k}={v}")


if __name__ == "__main__":
    main(sys.argv[1:])
