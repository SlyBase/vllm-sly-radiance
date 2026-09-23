#!/usr/bin/env python3
"""Split the /opt/vllm venv into a cold and a hot image layer, so an update pulls only the hot one.

The venv is ~2.4 GB, and almost all of it (torch, triton, aiter's metadata, pyarrow, ...) changes
only when the stack itself is bumped. The part a radiance release changes -- the patched vLLM and
aiter sources, the radiance_* modules, r4d.so and the HIP kernels -- is ~250 MB. Shipped as one
COPY, every release is a new ~900 MB (compressed) layer that every host downloads in full.

    split_venv.py snapshot <root> <manifest>      after the wheel install: record every entry
    split_venv.py split <root> <manifest> <out>   after the patches: copy each entry to
                                                  <out>/cold or <out>/hot (root is only read)

An entry goes to cold when it is byte-identical to its snapshot and is not under HOT_PACKAGES;
everything else goes to hot: files a patch rewrote (wherever they are, e.g. torch/_dynamo/utils.py),
files the build added, and the whole of the packages the patch chain works on. Those packages are hot
as a whole so that a patch that starts touching one more vLLM file does not reshuffle the cold layer.

Every mtime in <out> is then set to SOURCE_DATE_EPOCH (default 1980-01-01). A rebuild's wheel
install or patch run writes new mtimes; without the reset the cold layer would differ from the
previous release's by timestamps alone and get a new digest, which is exactly the download this
split exists to avoid. With it, the cold layer's digest depends on file contents, modes and paths
only, so an unchanged stack reproduces the previous release's cold layer and a pull skips it.

`split` copies rather than moves and never writes to <root>: a rename inside the build container's
overlayfs copies each lower-layer file up with an fsync, which for the ~82k files of the venv did
not finish in ten minutes on the ZFS-backed build host.
"""
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

# Relative to site-packages. The patch loop edits these in place, and the build copies configs
# into both, so they would land in hot piecemeal anyway.
HOT_PACKAGES = ("vllm", "aiter")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _entries(root):
    """Yield (relpath, lstat) for everything under root, parents before children."""
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(dirnames) + sorted(filenames):
            p = os.path.join(dirpath, name)
            yield os.path.relpath(p, root), os.lstat(p)


def _describe(root, rel, st):
    p = os.path.join(root, rel)
    mode = stat.S_IMODE(st.st_mode)
    if stat.S_ISLNK(st.st_mode):
        return ["l", os.readlink(p), 0]
    if stat.S_ISDIR(st.st_mode):
        return ["d", "", mode]
    return ["f", _sha256(p), mode]


def snapshot(root, manifest):
    snap = {rel: _describe(root, rel, st) for rel, st in _entries(root)}
    Path(manifest).write_text(json.dumps(snap, sort_keys=True))
    print(f"[split_venv] snapshot: {len(snap)} entries of {root} -> {manifest}")


def _hot_prefixes(root):
    sps = sorted(Path(root).glob("lib/python3*/site-packages"))
    if len(sps) != 1:
        raise SystemExit(f"[split_venv] expected one site-packages under {root}, found {sps}")
    sp = sps[0].relative_to(root)
    return tuple(str(sp / pkg) for pkg in HOT_PACKAGES)


def split(root, manifest, out):
    snap = json.loads(Path(manifest).read_text())
    hot_prefixes = _hot_prefixes(root)
    for pkg in hot_prefixes:
        if not os.path.isdir(os.path.join(root, pkg)):
            raise SystemExit(f"[split_venv] HOT_PACKAGES entry {pkg} does not exist")
    cold_dir, hot_dir = os.path.join(out, "cold"), os.path.join(out, "hot")
    for d in (cold_dir, hot_dir):
        os.makedirs(d)
        os.chmod(d, stat.S_IMODE(os.lstat(root).st_mode))

    def is_hot_path(rel):
        return any(rel == p or rel.startswith(p + os.sep) for p in hot_prefixes)

    size = {"cold": 0, "hot": 0}
    count = {"cold": 0, "hot": 0}
    rewritten = []  # hot only because a patch changed them: worth seeing in the build log
    for rel, st in _entries(root):
        desc = _describe(root, rel, st)
        same = snap.get(rel) == desc
        if desc[0] == "d":
            # Every directory is recreated on its side up front, so empty ones survive too; a cold
            # directory's hot files later recreate its path in hot. Mode now, mtime at the end.
            d = os.path.join(cold_dir if same and not is_hot_path(rel) else hot_dir, rel)
            os.makedirs(d, exist_ok=True)
            os.chmod(d, desc[2])
            continue
        side = "cold" if same and not is_hot_path(rel) else "hot"
        if side == "hot" and rel in snap and not is_hot_path(rel):
            rewritten.append(rel)
        dest = os.path.join(cold_dir if side == "cold" else hot_dir, rel)
        parent = os.path.dirname(dest)
        if not os.path.isdir(parent):
            os.makedirs(parent)
            # mirror the source directories' modes on the parents created in hot
            src_parent, dst_parent = os.path.dirname(os.path.join(root, rel)), parent
            while dst_parent not in (cold_dir, hot_dir):
                os.chmod(dst_parent, stat.S_IMODE(os.lstat(src_parent).st_mode))
                src_parent, dst_parent = os.path.dirname(src_parent), os.path.dirname(dst_parent)
        src = os.path.join(root, rel)
        if desc[0] == "l":
            os.symlink(desc[1], dest)
        else:
            shutil.copy2(src, dest)
        size[side] += st.st_size if desc[0] == "f" else 0
        count[side] += 1

    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", "315532800"))
    for d in (cold_dir, hot_dir):
        for dirpath, dirnames, filenames in os.walk(d, topdown=False):
            for name in dirnames + filenames:
                os.utime(os.path.join(dirpath, name), (epoch, epoch), follow_symlinks=False)
        os.utime(d, (epoch, epoch))

    for side in ("cold", "hot"):
        print(f"[split_venv] {side}: {count[side]} entries, {size[side] / 2**20:.0f} MB")
    print(f"[split_venv] hot packages: {', '.join(HOT_PACKAGES)}; "
          f"rewritten outside them ({len(rewritten)}):")
    for rel in rewritten:
        print(f"[split_venv]   {rel}")


def main(argv):
    if len(argv) == 3 and argv[0] == "snapshot":
        snapshot(argv[1], argv[2])
    elif len(argv) == 4 and argv[0] == "split":
        split(argv[1], argv[2], argv[3])
    else:
        raise SystemExit(__doc__.split("\n\n")[1])


if __name__ == "__main__":
    main(sys.argv[1:])
