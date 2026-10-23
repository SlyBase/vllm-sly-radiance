"""CPU checks of sly/bench_attn_r4d_vs_tetra.py's pure helpers (the GPU arms run in the image only)."""
import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "bench_attn_r4d_vs_tetra.py"
spec = importlib.util.spec_from_file_location("bench_attn_r4d_vs_tetra", SRC)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)       # module level imports no torch / GPU code


def test_shape():
    assert B.shape(0, 4096, 880) == dict(seq=4096, blocks=5)
    assert B.shape(49152, 4096, 880) == dict(seq=53248, blocks=61)
    assert B.shape(49152, 4096, 1664) == dict(seq=53248, blocks=32)
    assert B.BS_FP8 % 16 == 0 and B.BS_INT4 % 16 == 0     # libr4d reads the fp8 block as 16-slot sub-blocks


def test_verdict_uses_deepest_past():
    assert "port worth it" in B.verdict({0: 1.0, 49152: 1.35})
    assert "does not explain" in B.verdict({0: 1.6, 49152: 1.2})
    assert "does not explain" in B.verdict({49152: 1.3})      # strictly above the threshold


def test_image_env_defaults():
    assert B.IMAGE_ENV == {"R4D_ATTN_FP8": "3", "R4D_ATTN_DB": "1", "R4D_ATTN_TILE": "16", "RADIANCE_ATTN_Q_FP8": "1"}
    assert B.ints("0,1,3,5") == [0, 1, 3, 5] and B.ints("") == []


def test_verdict_kv4i_goes_at_1_15_at_the_deepest_past():
    assert "port goes ahead" in B.verdict_kv4i({0: 2.0, 49152: 1.15})
    assert "stop" in B.verdict_kv4i({0: 1.0, 49152: 1.16})
