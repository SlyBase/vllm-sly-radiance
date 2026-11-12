#!/usr/bin/env python3
"""CPU test: DRY_RUN must not remove a running container in `serve-mxfp4.sh`.

Why (upstream ggz14 08fc42d, mirror PR #231): DRY_RUN=1 is documented as "print the container
command instead of running it", but the leftover-container cleanup `"${RUNTIME}" rm -f "$NAME"`
ran before the flag was consulted -- a dry run tore down a live server of the same NAME.

The test EXECUTES the launcher's actual cleanup line in bash with recording stub runtimes, so it
checks behaviour, not the spelling of the guard. No docker, no GPU.

Run:  python3 sly/tests/test_launcher_dry_run_cpu.py
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MXFP4_SERVE = ROOT / "serve-mxfp4.sh"


def _cleanup_line():
    """The one leftover-container removal line (`if [ ... ]; then "$RUNTIME" rm -f "$NAME" ...`)."""
    lines = [l for l in MXFP4_SERVE.read_text().splitlines()
             if 'rm -f "$NAME"' in l and l.strip().startswith("if [")]
    assert len(lines) == 1, f"expected exactly one cleanup line, found {len(lines)}"
    return lines[0]


def _run(runtime, dry_run, tmp_path):
    calls = tmp_path / "calls.txt"
    calls.touch()
    script = (
        "set -euo pipefail\n"
        'NAME=radiance-dryrun-test\n'
        'docker() { printf \'%s\\n\' "$*" >> "$CALLS"; }\n'
        'podman() { printf \'%s\\n\' "$*" >> "$CALLS"; }\n'
        f'RUNTIME={runtime}\n'
        + _cleanup_line() + "\n"
        'cat "$CALLS"\n'
    )
    env = {k: v for k, v in os.environ.items() if k != "DRY_RUN"}
    env["CALLS"] = str(calls)
    if dry_run is not None:
        env["DRY_RUN"] = dry_run
    r = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_docker_without_dry_run_removes_leftover(tmp_path):
    assert "rm -f radiance-dryrun-test" in _run("docker", None, tmp_path)


def test_docker_dry_run_keeps_running_container(tmp_path):
    assert _run("docker", "1", tmp_path) == ""


def test_docker_dry_run_zero_is_still_dry(tmp_path):
    # Script convention: any non-empty DRY_RUN means dry run (`${DRY_RUN:+echo}` at the exec).
    assert _run("docker", "0", tmp_path) == ""


def test_podman_never_removes(tmp_path):
    assert _run("podman", None, tmp_path) == ""
    assert _run("podman", "1", tmp_path) == ""


def test_sibling_launchers_have_no_unguarded_name_rm():
    # serve.sh / serve-tp*.sh must not grow an unguarded `rm -f "$NAME"` (only serve-mxfp4.sh
    # carries the cleanup, and it is DRY_RUN-guarded).
    for script in sorted(ROOT.glob("serve*.sh")):
        for line in script.read_text().splitlines():
            if 'rm -f "$NAME"' in line:
                assert "DRY_RUN" in line, f"{script.name}: unguarded rm -f \"$NAME\": {line}"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        tp = Path(td)
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn(tp)
                print("ok", name)
    print("launcher DRY_RUN tests OK")
