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


def main(argv):
    pins = read_pins()
    if "--json" in argv:
        print(json.dumps(pins, indent=2))
    elif "--shell" in argv:
        for k, v in pins.items():
            print(f"export {k}='{v}'")
    else:
        for k, v in pins.items():
            print(f"{k}={v}")


if __name__ == "__main__":
    main(sys.argv[1:])
