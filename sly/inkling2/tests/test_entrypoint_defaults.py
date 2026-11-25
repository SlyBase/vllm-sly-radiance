"""radiance_entrypoint.sh: Inkling-2-Mini serve defaults (SLAK + adaptive k), CPU only, no vLLM.

Runs the real entrypoint with a fake `vllm` on PATH that dumps its environment. Only the lossless
knobs are defaults: NSD changed the greedy output in W3 (gate 2/4, degraded text), so it is opt-in.
"""
import json
import os
import subprocess
from pathlib import Path

import pytest

ENTRY = Path(__file__).resolve().parents[3] / "radiance_entrypoint.sh"
KNOBS = {
    "RADIANCE_INKLING2_DRAFT_SHORTLIST": "4096",
    "RADIANCE_INKLING2_ADAPTIVE_K": "3,5,7",
}


@pytest.fixture()
def serve(tmp_path):
    fake = tmp_path / "bin" / "vllm"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\npython3 -c 'import json,os;print(json.dumps(dict(os.environ)))'\n")
    fake.chmod(0o755)

    def run(args, **env):
        base = {k: v for k, v in os.environ.items() if not k.startswith("RADIANCE_")}
        base["PATH"] = f"{fake.parent}:{base['PATH']}"
        base.update(env)
        out = subprocess.run(["bash", str(ENTRY), *args], capture_output=True, text=True, env=base, check=True)
        return json.loads(out.stdout.strip().splitlines()[-1])

    return run


@pytest.mark.parametrize("args", [
    ["--model", "thinkingmachines/Inkling-2-Mini", "--no-async-scheduling"],
    ["thinkingmachines/Inkling-2-Mini"],
    ["--model=/models/inkling-2-mini"],
])
def test_inkling_gets_the_two_lossless_defaults(serve, args):
    env = serve(args)
    assert {k: env.get(k) for k in KNOBS} == KNOBS


def test_lossy_knobs_stay_opt_in(serve):
    env = serve(["--model", "thinkingmachines/Inkling-2-Mini"])
    assert not [k for k in env if k.startswith(("RADIANCE_NSD", "RADIANCE_CACTUS"))]


def test_explicit_env_wins_even_when_empty(serve):
    env = serve(["--model", "thinkingmachines/Inkling-2-Mini"],
                RADIANCE_INKLING2_DRAFT_SHORTLIST="0", RADIANCE_INKLING2_ADAPTIVE_K="")
    assert env["RADIANCE_INKLING2_DRAFT_SHORTLIST"] == "0"
    assert env["RADIANCE_INKLING2_ADAPTIVE_K"] == ""  # empty = explicitly off


def test_other_models_are_unchanged(serve):
    env = serve(["--model", "amd/Qwen3.8-27B-Quark-AWQ-MXFP4"])
    assert not [k for k in KNOBS if k in env]


MODEL = ["--model", "thinkingmachines/Inkling-2-Mini"]


def _spec(method):
    return {"method": method, "model": "/m/drafter", "num_speculative_tokens": 7}


@pytest.mark.parametrize("cfg", [
    json.dumps(_spec("dflash")),                      # spaces after : and ,
    json.dumps(_spec("dflash"), separators=(",", ":")),  # compact
    json.dumps(_spec("DFlash")),                      # case-insensitive
])
@pytest.mark.parametrize("form", ["split", "equals"])
def test_dflash2_drafter_gets_no_slak_or_adaptive_k(serve, cfg, form):
    spec = ["--speculative-config", cfg] if form == "split" else [f"--speculative-config={cfg}"]
    env = serve(MODEL + spec)
    assert not [k for k in KNOBS if k in env]  # MEASURED: -1 % tok/s with the DFlash2 drafter


@pytest.mark.parametrize("cfg", [
    json.dumps(_spec("dspark")),
    json.dumps(_spec("dspark"), separators=(",", ":")),
    json.dumps({"method": "mtp"}),
])
def test_other_drafter_methods_keep_the_defaults(serve, cfg):
    env = serve(MODEL + ["--speculative-config", cfg])
    assert {k: env.get(k) for k in KNOBS} == KNOBS


def test_no_speculative_config_keeps_the_defaults(serve):
    # the DSpark drafter is the model's built-in default; only an explicit DFlash method opts out
    env = serve(MODEL)
    assert {k: env.get(k) for k in KNOBS} == KNOBS


def test_explicit_env_wins_with_the_dflash2_drafter(serve):
    env = serve(MODEL + ["--speculative-config", json.dumps(_spec("dflash"))],
                RADIANCE_INKLING2_DRAFT_SHORTLIST="2048")
    assert env["RADIANCE_INKLING2_DRAFT_SHORTLIST"] == "2048"
    assert "RADIANCE_INKLING2_ADAPTIVE_K" not in env


CODA = "RADIANCE_INKLING2_CODA_FP4"


@pytest.mark.parametrize("extra", [
    [],                                                                  # DSpark default
    ["--speculative-config", json.dumps(_spec("dspark"))],
    ["--speculative-config", json.dumps(_spec("dflash"))],               # DFlash2: coda is drafter-independent
])
def test_inkling_gets_coda_nvfp4_by_default(serve, extra):
    # MEASURED KLD 0.0033 nats (gate 0.010), dPPL +0.26 %, +3.9 % tok/s: user-approved lossy default
    assert serve(MODEL + extra)[CODA] == "1"


@pytest.mark.parametrize("val", ["0", ""])
def test_explicit_env_turns_coda_off(serve, val):
    assert serve(MODEL, **{CODA: val})[CODA] == val


def test_other_models_get_no_coda_knob(serve):
    assert CODA not in serve(["--model", "amd/Qwen3.8-27B-Quark-AWQ-MXFP4"])


QWEN = "amd/Qwen3.8-27B-Quark-AWQ-MXFP4"
DRAFT = {"method": "dflash", "model": "/models/inkling-2-mini-dflash2", "num_speculative_tokens": 7}


@pytest.mark.parametrize("args", [
    [QWEN, "--speculative-config", json.dumps(DRAFT)],
    ["--model", QWEN, "--speculative-config", json.dumps(DRAFT | {"method": "dspark"})],
    [QWEN, "--served-model-name", "inkling-2-mini"],  # a served name is a label, not the target (decided: ignored)
    ["--served-model-name", "inkling", "--model", QWEN],
    ["--model", QWEN, "--tokenizer", "thinkingmachines/Inkling-2-Mini"],
])
def test_inkling_text_outside_the_target_model_sets_no_defaults(serve, args):
    env = serve(args)
    assert not [k for k in (*KNOBS, CODA) if k in env]


@pytest.mark.parametrize("args", [
    ["--model=thinkingmachines/Inkling-2-Mini", "--served-model-name", "qwen"],
    ["--host", "0.0.0.0", "--model", "/m/inkling-2-mini"],
    ["thinkingmachines/Inkling-2-Mini", "--host", "0.0.0.0"],
])
def test_inkling_target_is_found_in_any_position(serve, args):
    assert serve(args)[CODA] == "1"


def test_no_arguments_do_not_crash_the_target_detection(serve):
    assert CODA not in serve([])
