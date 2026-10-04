#!/usr/bin/env python3
"""Rewrite the dependency metadata of AMD's torch wheels to match this image's stack.

AMD's whl-next torch (and its amd-torch-device-<arch> wheel) declare `rocm[libraries]==<ver>`,
`rocm-bootstrap`, `rocm-sdk-device-<arch>` and an exact `triton==3.8.0+git...`. This image takes
ROCm from /opt/rocm (rocm_sdk/ preloads from there) and pins triton itself, so none of those are
installed -- and pip then treats the installed torch as unsatisfied and replaces it with the newest
PyPI torch (a CUDA build) the first time anything that depends on torch is resolved. Only METADATA
and RECORD change; every binary in the wheel stays byte-identical.

    fix_amd_torch_metadata.py <triton-version> <in.whl> <out-dir>
"""
import base64
import hashlib
import re
import sys
import zipfile
from pathlib import Path

DROP = re.compile(r"^Requires-Dist:\s*(rocm(\[|\s|=|$)|rocm-bootstrap|rocm-sdk-|triton(\s|=|$))", re.I)


def record_line(name, data):
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    return f"{name},sha256={digest},{len(data)}"


def main(triton, src, out_dir):
    src = Path(src)
    dst = Path(out_dir) / src.name
    with zipfile.ZipFile(src) as zin:
        names = zin.namelist()
        meta = next(n for n in names if n.endswith(".dist-info/METADATA"))
        record = next(n for n in names if n.endswith(".dist-info/RECORD"))
        lines = zin.read(meta).decode().split("\n")
        dropped = [ln for ln in lines if DROP.match(ln)]
        kept = [ln for ln in lines if not DROP.match(ln)]
        if any(ln.strip() == "Name: torch" for ln in kept):
            at = next(i for i, ln in enumerate(kept) if ln.startswith("Requires-Dist:"))
            kept.insert(at, f"Requires-Dist: triton=={triton}")
        new_meta = "\n".join(kept).encode()
        rec = []
        with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                if info.filename == record:
                    continue
                data = new_meta if info.filename == meta else zin.read(info)
                zout.writestr(info, data)
                if not info.filename.endswith("/"):
                    rec.append(record_line(info.filename, data))
            rec.append(f"{record},,")
            zout.writestr(record, "\n".join(rec) + "\n")
    print(f"{src.name}: dropped {len(dropped)} requirement(s): {[d.split(':', 1)[1].strip() for d in dropped]}")


if __name__ == "__main__":
    main(*sys.argv[1:])
