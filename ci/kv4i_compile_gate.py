#!/usr/bin/env python3
"""Compile-only gate for the O_KV4I int4-KV prefill variant (sly/r4d/r4d_kv4i.patch). No GPU.

Takes a libr4d checkout that already has r4d_extras_rx10.patch and r4d_gdn_scan_variants.patch applied
(the Dockerfile kernels-stage state), copies it twice, applies r4d_kv4i.patch to one copy, emits the
device assembly of r4d_attn_paged_h256_gqa6.hip for both (the same unit ci/attn_sched_lint.py reads),
and checks:

  1. IDENTITY  every kernel of the stock assembly exists in the kv4i assembly with an identical
               instruction stream and identical VGPR / SGPR / scratch numbers (the shipped instances
               must not move because an opt-in bit was added);
  2. NEW       the new r4d_attn_prefill_kernel instances (OPT carries O_KV4I) are present for tiles
               16, 32 and 48, and none of them spills (scratch == 0, VGPR <= 240).

    python3 ci/kv4i_compile_gate.py --src /root/kv4i-work/libr4d \\
        --patch sly/r4d/r4d_kv4i.patch [--image ghcr.io/slybase/vllm-sly-radiance:1.8.0-rocm10.1]

hipcc runs in the image via `docker run` with NO device flags (compile only); --hipcc "hipcc" runs it
directly when already inside the image. Exit 0 = all green, 1 = a violation.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attn_sched_lint import KERNEL, parse  # noqa: E402

O_KV4I = 33554432
FLAGS = ("-O3 -std=c++17 -fPIC --offload-arch=gfx1201 -Wno-unused-result -ffp-contract=off "
         "--offload-device-only -S r4d_attn_paged_h256_gqa6.hip -o ")


LBL = re.compile(r"\.L[A-Za-z_]+\d+_\d+")


def norm(k):
    """Instruction stream with the function-numbered local labels (.LBB21_3) renamed by first use:
    adding a kernel earlier in the unit renumbers every later one without changing a single opcode."""
    seen = {}
    return [(m, LBL.sub(lambda x: seen.setdefault(x.group(0), f".L{len(seen)}"), o)) for m, o in k.ins]


def emit(tree: Path, out: str, image: str | None) -> Path:
    cmd = f"hipcc {FLAGS}{out}"
    if image:
        cmd = ["docker", "run", "--rm", "-v", f"{tree}:/w", "-w", "/w", "--entrypoint", "sh", image, "-c", cmd]
        subprocess.run(cmd, check=True)
    else:
        subprocess.run(cmd, shell=True, check=True, cwd=tree)
    return tree / out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="libr4d tree with the two existing patches applied")
    ap.add_argument("--patch", required=True, help="sly/r4d/r4d_kv4i.patch")
    ap.add_argument("--image", default="ghcr.io/slybase/vllm-sly-radiance:1.8.0-rocm10.1")
    ap.add_argument("--hipcc", action="store_true", help="run hipcc directly (inside the image)")
    args = ap.parse_args(argv)
    image = None if args.hipcc else args.image

    work = Path(tempfile.mkdtemp(prefix="kv4i-gate-", dir=args.src + "/.."))
    try:
        stock, new = work / "stock", work / "new"
        for t in (stock, new):
            shutil.copytree(args.src, t, ignore=shutil.ignore_patterns(".git", "*.s", "*.o"))
        subprocess.run(["git", "apply", str(Path(args.patch).resolve())], check=True, cwd=new)
        a = parse(str(emit(stock, "attn.s", image)))
        b = parse(str(emit(new, "attn.s", image)))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    bad = []
    for name, k in a.items():
        n = b.get(name)
        if n is None:
            bad.append(f"IDENTITY missing in kv4i build: {name}")
        elif (norm(k), k.vgpr, k.sgpr, k.spill) != (norm(n), n.vgpr, n.sgpr, n.spill):
            bad.append(f"IDENTITY changed: {name} ({len(k.ins)} -> {len(n.ins)} instructions, "
                       f"vgpr {k.vgpr}->{n.vgpr}, spill {k.spill}->{n.spill})")
    print(f"identity: {len(a)} stock kernels compared, "
          f"{sum(1 for x in bad if x.startswith('IDENTITY'))} differ")

    tiles = set()
    for name, k in sorted(b.items()):
        m = KERNEL.search(name)
        if not m or name in a or not int(m.group(7)) & O_KV4I:
            continue
        tile = int(m.group(2))
        tiles.add(tile)
        qf8 = "QF8" if int(m.group(7)) & 16777216 else "bf16q"
        print(f"new   tile {tile:2d} {qf8:5s}: {len(k.ins):5d} instructions, vgpr {k.vgpr}, "
              f"sgpr {k.sgpr}, scratch {k.spill}")
        if k.spill or (k.vgpr or 0) > 240:
            bad.append(f"NEW spills or over budget: {name}")
    if tiles != {16, 32, 48}:
        bad.append(f"NEW instances for tiles {sorted(tiles)}, expected [16, 32, 48]")

    for x in bad:
        print("FAIL", x)
    print("GATE", "FAIL" if bad else "OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
