"""CPU test: radiance_draft.install() on a vLLM without the MTP proposer module (>= 0.34 V2 runner / DFlash).

The controller hooks `vllm.v1.spec_decode.llm_base_proposer`, which the V2 runner does not have. That is
intended (DFlash drafts all positions in one pass), so the log must say so once instead of reporting a
failed install. Any other install error must still be reported as a failure. Stubbed modules, no vLLM, no GPU.
"""
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import radiance_draft as d  # noqa: E402


@pytest.fixture()
def run(monkeypatch):
    logs = []
    monkeypatch.setattr(d, "_log", logs.append)
    monkeypatch.setattr(d, "DYNAMIC", True)
    monkeypatch.setitem(sys.modules, "radiance_draft_gpu", object())
    monkeypatch.setitem(sys.modules, "torch", type(sys)("torch"))

    def go(hooks):
        monkeypatch.setattr(d, "_install_drafter_hooks", hooks)
        monkeypatch.setattr(d, "_install_runner_wrap", lambda: None)
        d.install()
        return logs

    return go


def _missing(name):
    def hooks():
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    return hooks


def test_missing_proposer_module_logs_intentional_inactivity_once(run):
    logs = run(_missing("vllm.v1.spec_decode.llm_base_proposer"))
    assert len(logs) == 1
    assert "intentionally inactive" in logs[0] and "vLLM >= 0.34" in logs[0] and "DFlash" in logs[0]
    assert "No module named" not in logs[0] and "disabled" not in logs[0]


def test_other_install_errors_are_still_reported(run):
    def hooks():
        raise RuntimeError("boom")
    logs = run(hooks)
    assert len(logs) == 1 and "install failed" in logs[0] and "boom" in logs[0]


def test_other_missing_modules_are_still_reported(run):
    logs = run(_missing("something.else"))
    assert "install failed" in logs[0]
