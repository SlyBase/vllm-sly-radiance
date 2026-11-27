"""The Qwen release arm starts its own container (local-docker window) with the
pinned argv: verbatim from the homelab swift-mxfp4 profile
(infrastructure/proxmox/ansible/group_vars/lxc_vllm7.yml +
playbooks/templates/docker-vllm7.service.j2) and the 1.13.4 reference arm on
host 51 (/root/qfix/out/QP2/docker_cmd.txt).

ci/quick_backtest.sh start_qwen() is the hand-run twin of the same arm on host
51; its argv must not drift from qwen_docker_argv(). This file is the single
source of truth for the gate argv and for the drift test against quick_backtest.
"""
import re
import shlex
import sys
import unittest
from pathlib import Path

CI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI / "accept"))
import accept  # noqa: E402

IMAGE = "vllm-sly-radiance:1.14.1-rocm10.1"
NAME = "vllm-accept-qwen"
PORT = 8001
MODEL_DIR = "/root/mcb-ckpt-v2"
CACHE_DIR = "/root/vllm7-cache"

ENV = (
    "HIP_VISIBLE_DEVICES=0", "HF_HUB_OFFLINE=1", "GPU_MAX_HW_QUEUES=1",
    "RADIANCE_MXFP4=1", "RADIANCE_MXFP4_W4A8=1",
    "RADIANCE_MXFP4_W4A8_MIN_M=0", "RADIANCE_MXFP4_DECODE_MAX_M=128",
    "RADIANCE_MXFP4_SANITIZE=0", "RADIANCE_LMHEAD_FP8=1",
    "RADIANCE_LMHEAD_INT4=1", "RADIANCE_FUSED_NORM_QUANT=1",
    "RADIANCE_KV_GROUP_SIZE=8", "RADIANCE_EMBED_INT8=1",
    "RADIANCE_EMBED_BITS=4", "RADIANCE_MXFP4_A_TILED_MIN_M=513",
    "RADIANCE_MXFP4_TN4_MIN_M=513", "RADIANCE_FUSED_NQ_REG=1",
    "RADIANCE_MXFP4_AT_SKIP_PAD=1", "RADIANCE_MXFP4_TILED_OPROJ=1",
    "R4D_ATTN_FP8=3", "RADIANCE_MXFP4_WPERM=1",
    "RADIANCE_MXFP4_WIDE_MAX_M=192", "RADIANCE_ADAPTIVE_WIDTH=perseq",
    "RADIANCE_CACTUS_DELTA=0.5",
    "RADIANCE_MXFP4_BCQ=/models/swift-mxfp4/bcq_tables.pt",
    "RADIANCE_LMHEAD_INT4_CODES=/models/swift-mxfp4/lm_head_int4.pt",
)


def qp2() -> list[str]:
    """Pinned Qwen argv: verbatim from /root/qfix/out/QP2/docker_cmd.txt on
    host 51, with image/name/port/model-dir/cache-dir as parameters."""
    argv: list[str] = ["docker", "run", "-d", "--name", NAME,
                       "--device=/dev/kfd", "--device=/dev/dri",
                       "--group-add", "44", "--group-add", "993",
                       "--security-opt", "seccomp=unconfined", "--ipc=host"]
    for kv in ENV:
        argv += ["-e", kv]
    argv += [
        "-p", f"{PORT}:8000",
        "-v", "/root/hf-cache:/root/.cache/huggingface",
        "-v", f"{MODEL_DIR}:/models/swift-mxfp4:ro",
        "-v", f"{CACHE_DIR}/triton:/root/.triton",
        "-v", f"{CACHE_DIR}/torch_compile:/root/.cache/vllm/torch_compile_cache",
        "-v", f"{CACHE_DIR}/aiter:/root/.aiter",
        IMAGE,
        "--model", "/models/swift-mxfp4",
        "--served-model-name", "slybase/Swift-1.5-Qwen3.8-27B-MXFP4-GPTQ",
        "amd/Qwen3.8-27B-Quark-AWQ-MXFP4",
        "slybase/ThinkingCap-Qwen3.8-27B-MXFP4",
        "bottlecapai/ThinkingCap-Qwen3.8-27B-NVFP4",
        "--max-model-len", "262144", "--gpu-memory-utilization", "0.98",
        "--kv-cache-dtype", "fp8", "--port", "8000",
        "--speculative-config.method", "dflash",
        "--speculative-config.model", "syvai/Qwen3.8-27B-DFlash2-W4A16",
        "--speculative-config.num_speculative_tokens", "7",
        "--speculative-config.draft_sample_method", "probabilistic",
        "--speculative-config.attention_backend", "TRITON_ATTN",
        "--override-generation-config",
        '{"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}',
        "--max-num-seqs", "8", "--max-num-batched-tokens", "4096",
        "--mamba-ssm-cache-dtype", "bfloat16",
        "--compilation-config.cudagraph_mode", "FULL_DECODE_ONLY",
        "--compilation-config.cudagraph_capture_sizes",
        "[8,16,24,32,40,48,56,64]",
        "--skip-mm-profiling", "--enable-mm-embeds",
        "--limit-mm-per-prompt.image", "0",
        "--limit-mm-per-prompt.video", "0",
        "--attention-backend", "R4D_HYBRID",
        "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_xml",
        "--reasoning-parser", "qwen3",
        "--chat-template", "/opt/qwen-fixed.jinja",
        "--default-chat-template-kwargs",
        '{"reasoning_effort": "medium"}',
        "--enable-prefix-caching", "--no-enable-extensible-kv-cache",
        "--kv-cache-memory-bytes", "15569256448",
    ]
    return argv


class QwenArgv(unittest.TestCase):
    def test_qwen_argv_is_pinned_verbatim_from_qp2(self):
        got = accept.qwen_docker_argv(IMAGE, NAME, PORT, MODEL_DIR, None,
                                      CACHE_DIR)
        self.assertEqual(got, qp2())

    def test_quick_backtest_qgood_matches_the_pinned_argv(self):
        """Drift test against ci/quick_backtest.sh start_qwen() (the Q-GOOD arm
        on host 51): the docker part and every serving flag from
        --max-model-len onward must be identical. Documented differences:
        quick_backtest.sh serves --served-model-name qwen-quick (one alias)
        while the pinned arm serves the homelab swift-mxfp4 alias list (the
        guard checks the profile model is served), and the pinned arm adds the
        RADIANCE_MXFP4_BCQ / RADIANCE_LMHEAD_INT4_CODES env from the
        swift-mxfp4 profile (host 51: /root/qfix/out/QP2/docker_cmd.txt)."""
        sh = (CI / "quick_backtest.sh").read_text()
        m = re.search(r"start_qwen\(\)\s*\{(.*?)\n\}", sh, re.DOTALL)
        self.assertIsNotNone(m, "start_qwen() not found in ci/quick_backtest.sh")
        body = m.group(1)
        line = re.sub(r"\\\s*\n", " ", body)
        run = next(ln.strip() for ln in line.splitlines()
                   if ln.lstrip().startswith("docker run"))
        run = (run.replace("$CACHE/qwen", f"{CACHE_DIR}/qwen")
                 .replace("$CACHE", CACHE_DIR)
                 .replace('"$1"', IMAGE)
                 .replace('"$C"', NAME)
                 .replace("$C", NAME)  # bare, just in case
                 .replace('"$PORT:8000"', f"{PORT}:8000"))
        argv = shlex.split(run)
        if argv and argv[-1] == ">/dev/null":
            argv.pop()  # shell redirect, not a docker argument
        self.assertEqual(argv[0], "docker")
        self.assertEqual(argv[1], "run")
        img = argv.index(IMAGE)
        mine = accept.qwen_docker_argv(IMAGE, NAME, PORT, MODEL_DIR, None,
                                       CACHE_DIR)
        mi = mine.index(IMAGE)
        # docker part (flags + mounts) identical; cache scratch roots differ
        # (quick_backtest.sh: $CACHE/qwen/{triton,...}; the gate: homelab's
        # vllm_compile_cache_dir/{triton,...}) so mask their values
        def mask_cache(av: list[str]) -> list[str]:
            out, i = [], 0
            while i < len(av):
                if av[i] == "-v" and i + 1 < len(av) and any(k in av[i + 1]
                        for k in ("triton:/root", "torch_compile", "/.aiter")):
                    out += ["-v", "<cache>"]
                    i += 2
                else:
                    out.append(av[i])
                    i += 1
            return out
        self.assertEqual(mask_cache(argv)[1:img], mask_cache(mine)[1:mi])
        # serving flags from --max-model-len onward identical
        tail_mine = mi + mine[mi:].index("--max-model-len")
        self.assertEqual(argv[argv.index("--max-model-len"):],
                         mine[tail_mine:])
        # quick_backtest.sh serves a single alias; the pinned arm serves the
        # full swift-mxfp4 alias list
        self.assertEqual(argv[argv.index("--served-model-name") + 1],
                         "qwen-quick")
        # every -e of quick_backtest.sh is contained in the pinned arm
        def envs(av: list[str]) -> set[str]:
            return {av[i + 1] for i, a in enumerate(av) if a == "-e"}
        self.assertTrue(envs(argv) <= envs(mine),
                        f"quick_backtest.sh env not subset: "
                        f"{envs(argv) - envs(mine)}")


class ServedModelGuard(unittest.TestCase):
    MODEL = "slybase/Swift-1.5-Qwen3.8-27B-MXFP4-GPTQ"

    def setUp(self) -> None:
        self._get = accept.http_get

        def fake_get(url: str, timeout: int = 30) -> tuple[int, str]:
            if url.endswith("/v1/models"):
                return 200, ('{"data": [{"id": "' + self.MODEL + '"}, '
                             '{"id": "inkling2-mini"}]}')
            return 404, "not found"

        accept.http_get = fake_get  # test seam; restored in tearDown

    def tearDown(self) -> None:
        accept.http_get = self._get

    def test_guard_passes_when_profile_model_is_served(self):
        accept.ensure_served_model("http://127.0.0.1:8001", self.MODEL)

    def test_guard_fails_fast_with_clear_message_when_model_is_foreign(self):
        with self.assertRaisesRegex(RuntimeError,
                                    "served model is not the profile model"):
            accept.ensure_served_model("http://127.0.0.1:8001", "qwen-quick")
