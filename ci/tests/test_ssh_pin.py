"""The gpu-window ssh calls pin the helper host key; the runner has no known_hosts."""
import subprocess
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accept"))
import accept  # noqa: E402


def test_ssh_argv_pins_host_key():
    with mock.patch.object(accept.subprocess, "run") as run:
        run.return_value = subprocess.CompletedProcess([], 0, stdout="{}", stderr="")
        accept.GpuWindow("gpuwin@192.168.178.51", None, "o", 60)._ssh("status")
    cmd = run.call_args[0][0]
    assert "StrictHostKeyChecking=yes" in cmd
    assert f"UserKnownHostsFile={accept.KNOWN_HOSTS}" in cmd
    assert not any("accept-new" in c for c in cmd)


def test_known_hosts_file_has_ed25519_key():
    line = accept.KNOWN_HOSTS.read_text().split()
    assert line[0] == "192.168.178.51" and line[1] == "ssh-ed25519"
