"""Regression: the gate must scan the FULL candidate startup log, not a tail window.

vLLM 0.34 + ROCm 10.1 emit ~8000 lines before the first request (mostly
HIPCachingAllocator OOM warnings during graph capture). The radiance install
banners sit at lines ~100-200; `gpu-window logs 4000` (docker logs --tail 4000)
cut them, so every gate since the 0.34 bump failed "log markers" while all
functional checks were green -- v1.13.0-v1.13.4 stayed unreleased.
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "accept"))
from accept import GpuWindow, check_logs  # noqa: E402

PROFILE = json.loads(
    (Path(__file__).resolve().parents[1] / "accept" / "profiles" / "vllm7-mxfp4.json").read_text())

# Real startup lines of vllm-sly-radiance:1.13.4-rocm10.1 (LXC 2408, 2026-11-20):
# the install-time banners first, the weight-load / first-forward lines later.
EARLY = [
    "[radiance.gemm] skinny GEMM kernel ENABLED for M in [6,64]: 256x2048",
    "[radiance] decode attn tune installed (verify-batch config, 1 module binding, wide=auto min_fill=0.5)",
    "[radiance] drafter attn tune installed (window-sliced split-KV for the DFlash drafter, segments=auto)",
    "[radiance] lookup draft: installed (V2 runner, DFlash2 speculator; enter 8 stay 3 hot 3)",
    "[radiance.gdn] gdn_chunk_scan ENABLED (head_k 128, head_v 128, chunk 64; replaces wy + delta_h + chunk_o)",
    "[radiance.mxfp4] W4A8 fp8-WMMA GEMM ENABLED for M>0 (NOTE: W4A8, not the checkpoint's W4A4)",
    "[radiance.fused_norm] armed: add_rms_quant=1 silu_mul_quant=1 gdn_norm_quant=1 a_tiled_min_m=513",
    "(APIServer pid=1) INFO 11-20 03:26:18 [model.py:772] Resolved architecture: DFlash2DraftModel",
    "RADIANCE_DYNAMIC_DRAFT=ON  controller=policy  tau=0.5  schedule=off",
    "[radiance.mxfp4] decode kernel ON (M<=128), 72 MiB split-K scratch",
    "[radiance.mxfp4] BCQ block tables loaded from /models/swift-mxfp4/bcq_tables.pt",
]
LATE = [
    "(EngineCore pid=424) INFO 11-20 03:27:32 [radiance_lmhead_int4.py:292] [radiance] lm_head quantised to int4 g128: [248320, 5120], 1.76 GiB freed",
    "[radiance] lm_head int4 codes from /models/swift-mxfp4/lm_head_int4.pt (248320 of 248320 rows)",
    "embed_tokens quantised to int4",
    "[radiance.mxfp4] linear layers: 304/304 on our kernel, 0 FORCED ONTO AITER (kernel cannot run them); aiter below M=0 (DISABLED)",
    "(EngineCore pid=424) INFO 11-20 03:27:43 [kv_cache_utils.py:1168] GPU KV cache size: 425,532 tokens, Maximum concurrency for 262,144 tokens per request: 1.",
    "[radiance] drafter attn plan: window=2048 q_len<=8 split-KV BLOCK_M=32 TILE=32 warps=4 splits=16 (seqs=8 tokens=64)",
]
FILLER = ("[rank0]:[W1120 03:27:13.973592007 HIPCachingAllocator.cpp:3934] "
          "memory allocation failed with OOM on device 0 while trying to allocate 2097152 bytes")


def startup_log(filler_lines: int) -> str:
    return "\n".join(EARLY + [FILLER] * filler_lines + LATE)


class FullStartupLog(unittest.TestCase):
    def test_check_logs_passes_on_full_log_and_fails_on_tail_4000(self):
        full = startup_log(9000)
        ok = check_logs(full, PROFILE)
        assert ok["ok"], ok["missing"]
        assert ok["kv_tokens_from_log"] == 425532
        tail = "\n".join(full.splitlines()[-4000:])
        cut = check_logs(tail, PROFILE)
        assert not cut["ok"]
        assert "quark mxfp4 W4A8 gemm" in cut["missing"]
        assert "dflash drafter" in cut["missing"]
        # the late lines survive the tail window -- exactly the observed split
        assert "all linear layers on our kernel" in cut["hits"]

    def test_logs_requests_the_full_log(self):
        seen = {}

        class RecordingWindow(GpuWindow):
            def _ssh(self, *args, timeout=900):
                seen["args"] = args
                return ""

        RecordingWindow("gpuwin@host", None, "test", 60).logs()
        assert seen["args"][0] == "logs"
        # a tail window is only safe if it cannot be smaller than the startup
        # log; 1e9 lines (~100 GB) is effectively the full log of a fresh container
        assert int(seen["args"][1]) >= 1_000_000, seen["args"]


if __name__ == "__main__":
    unittest.main()
