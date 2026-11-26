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
    fake.write_text("#!/bin/sh\npython3 -c 'import json,os,sys;print(json.dumps(dict(os.environ, _ARGS=sys.argv[1:])))' \"$@\"\n")
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


def _ckpt(tmp_path_factory, cfg):
    d = tmp_path_factory.mktemp("ckpt")  # neutral name, like the documented `/model` mount (tmp_path embeds the test name)
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


@pytest.mark.parametrize("cfg", [
    {"model_type": "inkling2_loop", "architectures": ["Inkling2LoopForConditionalGeneration"]},  # real host-51 value
    {"model_type": "x", "architectures": ["Inkling2ForCausalLM"]},
])
def test_local_dir_with_an_inkling_config_gets_the_defaults(serve, tmp_path_factory, cfg):
    env = serve([_ckpt(tmp_path_factory, cfg), "--speculative-config", json.dumps(DRAFT | {"method": "dspark"})])
    assert {k: env.get(k) for k in KNOBS} == KNOBS and env[CODA] == "1"


def test_local_dir_with_a_qwen_config_gets_none_even_with_an_inkling_drafter(serve, tmp_path_factory):
    cfg = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForCausalLM"]}
    env = serve([_ckpt(tmp_path_factory, cfg), "--speculative-config", json.dumps(DRAFT)])
    assert not [k for k in (*KNOBS, CODA) if k in env]


@pytest.mark.parametrize("extra", [["--reasoning-parser", "inkling"], ["--reasoning-parser=inkling"]])
def test_reasoning_parser_inkling_is_a_fallback_signal(serve, extra):
    assert serve(["/nonexistent/model", *extra])[CODA] == "1"  # target unreadable


def test_other_reasoning_parser_is_no_signal(serve):
    assert CODA not in serve(["/nonexistent/model", "--reasoning-parser", "qwen3"])


CHUNK = "--max-num-batched-tokens"
CTX = ["--max-model-len", "32768"]


def test_inkling_gets_the_4096_prefill_chunk_up_to_131072_context(serve):
    # MEASURED (PTM4 vs PT, 210 W): +16/+14/+15 % pp/s at 8k/16k/32k. Cost: 0.8 GiB less KV (4640 -> 4216 blocks) and a
    # larger KV need per request: at max-model-len 202752 and 0.92 utilisation the engine refuses to start (10.13 GiB
    # needed, 8.23 available; chunk 2048 starts). Estimated limit with chunk 4096: 140416 tokens.
    for ctx in ("32768", "131072"):
        args = serve(MODEL + ["--max-model-len", ctx])["_ARGS"]
        assert args[args.index(CHUNK) + 1] == "4096" and args.count(CHUNK) == 1


def test_chunk_default_skips_long_or_unknown_context(serve):
    assert CHUNK not in serve(MODEL + ["--max-model-len", "202752"])["_ARGS"]
    assert CHUNK not in serve(MODEL + ["--max-model-len=262144"])["_ARGS"]
    assert CHUNK not in serve(MODEL)["_ARGS"]  # no --max-model-len: the model's own (long) default


def test_chunk_default_reads_the_equals_form(serve):
    args = serve(MODEL + ["--max-model-len=32768"])["_ARGS"]
    assert args[args.index(CHUNK) + 1] == "4096"


@pytest.mark.parametrize("given", [[CHUNK, "2048"], [f"{CHUNK}=8192"]])
def test_explicit_chunk_flag_wins(serve, given):
    args = serve(MODEL + CTX + given)["_ARGS"]
    assert sum(a.startswith(CHUNK) for a in args) == 1
    assert "4096" not in args


@pytest.mark.parametrize("val", ["0", ""])
def test_chunk_env_off_leaves_the_vllm_default(serve, val):
    assert CHUNK not in serve(MODEL + CTX, RADIANCE_INKLING2_CHUNK=val)["_ARGS"]


def test_chunk_env_sets_the_size_whatever_the_context(serve):
    for extra in (CTX, ["--max-model-len", "202752"], []):
        args = serve(MODEL + extra, RADIANCE_INKLING2_CHUNK="8192")["_ARGS"]
        assert args[args.index(CHUNK) + 1] == "8192"


def test_other_models_get_no_chunk_flag(serve):
    assert CHUNK not in serve(["--model", "amd/Qwen3.8-27B-Quark-AWQ-MXFP4"] + CTX)["_ARGS"]


CG = "--compilation-config"
CG_DEFAULT = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 40, 48, 56, 64]


def _cg(args):
    return json.loads(args[args.index(CG) + 1])["cudagraph_capture_sizes"]


def test_inkling_gets_decode_graphs_up_to_8_streams(serve):
    # MEASURED: vLLM derives the decode graph request counts from the token sizes (cdiv(size, 1 + k)), so the old
    # recipe [1..8] captured ONE decode graph (1 request); 2+ streams ran eager (2 streams: 61 -> 130 tok/s/stream).
    for spec in ([], ["--speculative-config", json.dumps(_spec("dflash"))], ["--speculative-config", json.dumps(_spec("dspark"))]):
        args = serve(MODEL + spec)["_ARGS"]
        assert args.count(CG) == 1 and _cg(args) == CG_DEFAULT


@pytest.mark.parametrize("given", [[CG, "{}"], [f"{CG}={{}}"], [f"{CG}.mode=0"], ["-cc", "{}"], ["-O3"]])
def test_explicit_compilation_config_wins(serve, given):
    args = serve(MODEL + given)["_ARGS"]
    assert CG not in args[:args.index(given[0])] and sum(a.startswith((CG, "-cc", "-O")) for a in args) == 1


def test_cudagraph_env_sets_or_switches_off(serve):
    assert _cg(serve(MODEL, RADIANCE_INKLING2_CUDAGRAPH_SIZES="1,2,8")["_ARGS"]) == [1, 2, 8]
    for off in ("0", ""):
        assert CG not in serve(MODEL, RADIANCE_INKLING2_CUDAGRAPH_SIZES=off)["_ARGS"]


def test_other_models_get_no_graph_sizes(serve):
    assert CG not in serve(["--model", "amd/Qwen3.8-27B-Quark-AWQ-MXFP4"])["_ARGS"]
