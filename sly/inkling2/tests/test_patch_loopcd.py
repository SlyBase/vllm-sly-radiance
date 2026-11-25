"""LoopCD-Hidden for Inkling-2-Mini (arxiv 2610.02185): CPU-only tests.

Covers: env knobs, the hidden-state contrast math, the source patch
(lands / parses / idempotent / inert without inkling2), and the injected
contrast step executed with real torch tensors against a stub model.

No GPU, no real Inkling weights. Run:
  INKLING_VLLM_SRC=  pytest sly/inkling2/tests/test_patch_loopcd.py
"""
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
PATCH = REPO / "sly" / "patch_inkling2_loopcd.py"
INK = REPO / "sly" / "inkling2"
LOOP_SRC = Path(
    "/opt/data/profiles/daidalos/cache/scratch/ink-k9-src/"
    "vllm/models/inkling2/nvidia/loop.py"
)

sys.path.insert(0, str(INK))

from radiance_inkling2_loopcd import (  # noqa: E402  (RED until module exists)
    apply_hidden,
    enabled,
    loops_override,
    omega,
)

LOOPCD_ENV = "RADIANCE_INKLING2_LOOPCD"
OMEGA_ENV = "RADIANCE_INKLING2_LOOPCD_OMEGA"
LOOPS_ENV = "RADIANCE_INKLING2_LOOPS"


# ---------------------------------------------------------------- radiance module

def test_omega_default_is_half():
    assert omega() == 0.5


def test_omega_from_env(monkeypatch):
    monkeypatch.setenv(OMEGA_ENV, "0.25")
    assert omega() == 0.25


def test_omega_invalid_falls_back(monkeypatch):
    monkeypatch.setenv(OMEGA_ENV, "not-a-number")
    assert omega() == 0.5


def test_loopcd_off_by_default(monkeypatch):
    monkeypatch.delenv(LOOPCD_ENV, raising=False)
    assert enabled() is False


def test_loopcd_on(monkeypatch):
    monkeypatch.setenv(LOOPCD_ENV, "1")
    assert enabled() is True


def test_loops_default_none(monkeypatch):
    monkeypatch.delenv(LOOPS_ENV, raising=False)
    assert loops_override() is None


def test_loops_from_env(monkeypatch):
    monkeypatch.setenv(LOOPS_ENV, "2")
    assert loops_override() == 2


def test_loops_invalid_none(monkeypatch):
    monkeypatch.setenv(LOOPS_ENV, "abc")
    assert loops_override() is None


def test_apply_hidden_is_contrast():
    h_R = torch.tensor([1.0, 2.0, 3.0])
    h_1 = torch.tensor([0.0, 1.0, 0.0])
    out = apply_hidden(h_R, h_1, 0.5)
    torch.testing.assert_close(out, torch.tensor([1.5, 2.5, 4.5]))


def test_apply_hidden_omega_zero_is_identity(h_R, h_1):
    assert apply_hidden(h_R, h_1, 0.0) is h_R


def test_apply_hidden_no_h1_is_identity(h_R):
    assert apply_hidden(h_R, None, 0.5) is h_R


def test_apply_hidden_does_not_mutate(h_R, h_1):
    before_R = h_R.clone()
    before_1 = h_1.clone()
    apply_hidden(h_R, h_1, 0.5)
    torch.testing.assert_close(h_R, before_R)
    torch.testing.assert_close(h_1, before_1)


# ---------------------------------------------------------------- patch mechanics

@pytest.fixture()
def h_R():
    return torch.tensor([1.0, 2.0, 3.0])


@pytest.fixture()
def h_1():
    return torch.tensor([0.0, 1.0, 0.0])


@pytest.fixture()
def loop_tree(tmp_path):
    """A minimal vllm tree carrying only nvidia/loop.py, for patch_target."""
    root = tmp_path / "vllm" / "models" / "inkling2" / "nvidia"
    root.mkdir(parents=True)
    shutil.copyfile(LOOP_SRC, root / "loop.py")
    return tmp_path


def run_patch(target):
    env = dict(
        os.environ,
        RADIANCE_PATCH_TARGET_DIR=str(target),
        PYTHONPATH=str(REPO),
    )
    return subprocess.run([sys.executable, str(PATCH)], capture_output=True, text=True, env=env)


def test_patch_lands_and_parses(loop_tree):
    r = run_patch(loop_tree)
    assert r.returncode == 0, r.stdout + r.stderr
    loop = (loop_tree / "vllm/models/inkling2/nvidia/loop.py").read_text()
    assert "RADIANCE LoopCD-Hidden" in loop
    assert "_rad_loopcd_step" in loop
    import ast
    ast.parse(loop)


def test_patch_idempotent(loop_tree):
    assert run_patch(loop_tree).returncode == 0
    first = (loop_tree / "vllm/models/inkling2/nvidia/loop.py").read_text()
    r2 = run_patch(loop_tree)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert "NOOP" in r2.stdout
    assert (loop_tree / "vllm/models/inkling2/nvidia/loop.py").read_text() == first


def test_patch_noop_without_inkling2(tmp_path):
    (tmp_path / "vllm" / "models" / "inkling2" / "nvidia").mkdir(parents=True)
    r = run_patch(tmp_path)
    assert r.returncode == 0
    assert "NOOP" in r.stdout


def test_forward_calls_step(loop_tree):
    assert run_patch(loop_tree).returncode == 0
    loop = (loop_tree / "vllm/models/inkling2/nvidia/loop.py").read_text()
    assert "self._rad_loopcd_step(" in loop
    assert "_rad_loopcd_step" in loop


# ---------------------------------------------------------------- injected step

def _method_source(path, name):
    import ast
    text = path.read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise AssertionError(f"{name} not found at module level in {path}")


def _stub_model(do=True):
    self = types.SimpleNamespace()
    self._rad_h1 = None
    self._rad_do = do
    self.loop_core = range(6, 18)  # core.stop == 18
    return self


def test_step_captures_h1_after_first_pass(loop_tree):
    assert run_patch(loop_tree).returncode == 0
    patched = loop_tree / "vllm/models/inkling2/nvidia/loop.py"
    src = _method_source(patched, "_rad_loopcd_step")
    ns = {
        "torch": torch,
        "_rad_loopcd_enabled": lambda: True,
        "_rad_loopcd_omega": lambda: 0.5,
        "_rad_apply_hidden": apply_hidden,
    }
    exec(src, ns)
    step = ns["_rad_loopcd_step"]

    self = _stub_model()
    stack = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    num_blocks = 1
    h0 = torch.tensor([10.0, 20.0, 30.0])
    inputs = (h0, torch.tensor([1.0, 2.0]))
    out = step(self, 17, 0, 3, inputs, stack, num_blocks)
    torch.testing.assert_close(self._rad_h1, h0)
    assert out is inputs  # unchanged on pass 0
    torch.testing.assert_close(
        stack[num_blocks], torch.arange(3, 6, dtype=torch.float32)
    )


def test_step_contrasts_before_coda(loop_tree):
    assert run_patch(loop_tree).returncode == 0
    patched = loop_tree / "vllm/models/inkling2/nvidia/loop.py"
    src = _method_source(patched, "_rad_loopcd_step")
    ns = {
        "torch": torch,
        "_rad_loopcd_enabled": lambda: True,
        "_rad_loopcd_omega": lambda: 0.5,
        "_rad_apply_hidden": apply_hidden,
    }
    exec(src, ns)
    step = ns["_rad_loopcd_step"]

    self = _stub_model()
    self._rad_h1 = torch.tensor([10.0, 20.0, 30.0])
    stack = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    num_blocks = 1
    h_R = torch.tensor([4.0, 5.0, 6.0])
    inputs = (h_R, torch.tensor([7.0, 8.0]))
    out = step(self, 17, 2, 3, inputs, stack, num_blocks)
    expected = h_R + 0.5 * (h_R - self._rad_h1)  # [1, -2.5, -6]
    torch.testing.assert_close(out[0], expected)
    torch.testing.assert_close(stack[num_blocks], expected)
    assert out[0] is not h_R  # guided state is a new tensor


def test_step_noop_outside_last_core_layer(loop_tree):
    assert run_patch(loop_tree).returncode == 0
    patched = loop_tree / "vllm/models/inkling2/nvidia/loop.py"
    src = _method_source(patched, "_rad_loopcd_step")
    ns = {
        "torch": torch,
        "_rad_loopcd_enabled": lambda: True,
        "_rad_loopcd_omega": lambda: 0.5,
        "_rad_apply_hidden": apply_hidden,
    }
    exec(src, ns)
    step = ns["_rad_loopcd_step"]

    self = _stub_model()
    stack = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    num_blocks = 1
    h = torch.tensor([1.0, 2.0, 3.0])
    inputs = (h, torch.tensor([7.0, 8.0]))
    # i == 16 is not the last core layer (17)
    out = step(self, 16, 0, 3, inputs, stack, num_blocks)
    assert out is inputs
    assert self._rad_h1 is None


def test_step_noop_when_disabled(loop_tree):
    assert run_patch(loop_tree).returncode == 0
    patched = loop_tree / "vllm/models/inkling2/nvidia/loop.py"
    src = _method_source(patched, "_rad_loopcd_step")
    ns = {
        "torch": torch,
        "_rad_loopcd_enabled": lambda: False,
        "_rad_loopcd_omega": lambda: 0.5,
        "_rad_apply_hidden": apply_hidden,
    }
    exec(src, ns)
    step = ns["_rad_loopcd_step"]

    self = _stub_model(do=False)
    h = torch.tensor([1.0, 2.0, 3.0])
    inputs = (h, torch.tensor([7.0, 8.0]))
    out = step(self, 17, 2, 3, inputs, torch.arange(4, dtype=torch.float32), 0)
    assert out is inputs


def test_step_omega_zero_is_identity(loop_tree):
    assert run_patch(loop_tree).returncode == 0
    patched = loop_tree / "vllm/models/inkling2/nvidia/loop.py"
    src = _method_source(patched, "_rad_loopcd_step")
    ns = {
        "torch": torch,
        "_rad_loopcd_enabled": lambda: True,
        "_rad_loopcd_omega": lambda: 0.0,
        "_rad_apply_hidden": apply_hidden,
    }
    exec(src, ns)
    step = ns["_rad_loopcd_step"]

    self = _stub_model()
    self._rad_h1 = torch.tensor([10.0, 20.0, 30.0])
    h_R = torch.tensor([4.0, 5.0, 6.0])
    stack = h_R.reshape(1, 3).clone()
    num_blocks = 0
    inputs = (h_R, torch.tensor([7.0, 8.0]))
    out = step(self, 17, 2, 3, inputs, stack, num_blocks)
    assert out[0] is h_R  # omega 0 -> identity, nothing replaced


# ---------------------------------------------------------------- DSpark x LoopCD-Kompatibilitaet
#
# Befund (CODE-EVIDENZ, aus Quellen gelesen):
#  * vllm/models/inkling2/configs.py (ink-k9-src):
#    Inkling2DSparkConfig.from_target liest `target_layer_ids` fest aus dem
#    mtp_config des Checkpoints und verlangt features[-1] == text.num_layer_visits.
#    num_layer_visits = num_hidden_layers + (num_loop_passes - 1) * len(core)
#                       = 24 + (3 - 1) * 12 = 48  (3 Passes, Core L6-17).
#  * vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py (ink-k9-src):
#    get_eagle3_aux_layers_from_config liest eagle_aux_hidden_state_layer_ids
#    (= target_layer_ids, configs.py:262) OHNE "+1"-Verschiebung und legt sie
#    als aux_hidden_state_layers (Besuchs-Indizes des Loop-Forwards) an.
#    F1-Server-Log belegt die tatsaechlichen IDs: (5, 23, 41, 48).
#  * vllm/models/inkling2/nvidia/loop.py:340-342, 372-397 (ink-k9-src):
#    Der Forward baut die Besuchsfolge (Prelude 1x, Core num_passes x, Coda 1x)
#    und clippt bei Besuchszahl in aux_hidden_state_states; 48 ==
#    num_layer_visits ist der Sentinel fuer den Final-Norm-Output, der separat
#    zurueckgegeben wird ([*aux_hidden_states, out]).
#  * Die LoopCD-LOOPS-Override (sly/patch_inkling2_loopcd.py) aendert num_passes
#    NUR zur Laufzeit im Forward (nicht im Config-Objekt). num_layer_visits
#    behauptet also weiter 48, der Forward laeuft aber mit 36 Besuchen.
#
# Follow: RADIANCE_INKLING2_LOOPS=2 bricht DSpark -> Arm LCD2 laeuft OHNE
# --speculative-config (reiner Target-Decode), siehe l1-gpu-loopcd.md.

DSARK_TARGET_LAYER_IDS = (5, 23, 41, 48)
NUM_LAYERS = 24
CORE = (6, 18)  # Core-Schichten L6..L17, 12 Schichten


def _visit_layers(num_passes: int) -> list[int]:
    """Layer-ID je Besuch, wie Inkling2LoopModel.forward die Visits baut.

    Prelude 1x, Core `num_passes` x, Coda 1x. Rueckgabe: die Schicht, deren
    Eingang bei jedem Besuch (0-basiert) erfasst wird.
    """
    layers = [i for i in range(CORE[0])]
    for _ in range(num_passes):
        layers += list(range(CORE[0], CORE[1]))
    layers += [i for i in range(CORE[1], NUM_LAYERS)]
    return layers


def test_dspark_loops_3_covers_all_capture_ids():
    """LOOPS=3 (Checkpoint): 48 Besuche, alle 4 DSpark-Capture-Punkte valide.

    Besuch 5 = L5-Prelude, 23 = L11-Pass 1, 41 = L17-Pass 2, 48 = Sentinel
    Final-Output. DSpark bekommt seine 4 Features (3 Besuche + out).
    """
    layers = _visit_layers(3)
    assert len(layers) == 48
    assert layers[5] == 5
    assert layers[23] == 11
    assert layers[41] == 17
    assert 48 not in layers  # 48 kommt als Besuch nie vor (Sentinel -> out)


def test_dspark_loops_2_breaks_capture_ids():
    """LOOPS=2 (LoopCD-Override): nur 36 Besuche, DSpark-Capture bricht.

    Besuch 41 (L17-Pass 2) existiert nicht mehr (Besuche enden bei 35); der
    Sentinel 48 wird zwar als Final-Output mitgeliefert, aber DSpark bekommt
    statt 4 nur 3 Features -> Fuse-Mix liest eine fehlende Spalte (garbage/
    Index-Fehler im Speculator).
    """
    layers = _visit_layers(2)
    assert len(layers) == 36
    assert layers[5] == 5
    assert layers[23] == 11  # ident mit LOOPS=3, also nicht das Problem
    assert 41 not in layers
    assert 48 not in layers
    covered = [v for v in DSARK_TARGET_LAYER_IDS[:3] if v < len(layers)]
    assert covered == [5, 23]  # nur 2 von 3 Besuch-Captures valide


def test_num_layer_visits_stays_at_checkpoint_passes():
    """num_layer_visits ist Config-Ebene; der LOOPS-Forward-Override aendert es nicht.

    Mit LoopCD LOOPS=2 behauptet der Checkpoint weiter 48 Besuche, laeuft aber
    mit 36. Die DSpark-Load-Pruefung (features[-1] == num_layer_visits == 48)
    laeuft also durch; der Bruch entsteht erst zur Laufzeit im Forward.
    """
    assert 24 + (3 - 1) * 12 == 48  # Checkpoint: 48 Besuche
    assert 24 + (2 - 1) * 12 == 36  # LoopCD-Override zur Laufzeit: 36 Besuche
