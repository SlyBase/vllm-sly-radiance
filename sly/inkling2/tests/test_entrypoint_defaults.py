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
