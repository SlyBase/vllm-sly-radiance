#!/usr/bin/env python3
"""CPU tests for the 4-bit KV reference library (sly/kvq/reference.py), KV-cache roadmap PR-1.

numpy only: no torch, no vllm, no GPU. The reference is the oracle for the S1 error study and for the
PR-3 write kernel, so every test pins a concrete value.

Run:  python3 -m pytest sly/tests/test_kvq_reference.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kvq import reference as R  # noqa: E402


def block(*vals):
    x = np.zeros(32, np.float32)
    x[: len(vals)] = vals
    return x


def test_ocp_scale_byte_for_amax_6():
    assert R.e8m0_ocp(block(6.0)) == 127            # floor(log2 6) - 2 = 0 -> bias 127


def test_ocp_saturates_7_6_to_6():
    assert R.dequant_mxfp4(R.quant_mxfp4(block(7.6), "ocp"))[0] == 6.0


def test_e2m1_ties_round_to_even_code():
    # scale 2^0 (amax 6 in slot 31); ties 0.25 0.75 1.25 1.75 2.5 3.5 5 -> codes 0 2 2 4 4 6 6
    x = block(0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -5.0)
    x[31] = 6.0
    out = R.dequant_mxfp4(R.quant_mxfp4(x, "ocp"))
    np.testing.assert_array_equal(out[:8], [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, -4.0])


def test_pack_low_nibble_is_even_index():          # same order as radiance_mxfp4_fp8.hip:20
    q = R.quant_mxfp4(block(0.5, 6.0), "ocp")       # codes 1 and 7, scale 2^0
    assert q.payload[0] == 0x71
    assert q.scales[0] == 127


def test_negative_code_sets_bit_3():
    assert R.quant_mxfp4(block(-6.0), "ocp").payload[0] == 0x0F


def test_one_ulp_above_a_tie_rounds_up():
    # bit-exactness vs the kernel: 0.25 + 1 ulp is nearer to 0.5 than to 0 -> code 1, not the even code 0
    x = block(np.nextafter(np.float32(0.25), np.float32(1)))
    x[31] = 6.0
    assert R.quant_mxfp4(x, "ocp").payload[0] & 0x0F == 1


def test_oas_takes_next_scale_for_paper_example():
    # arXiv 2603.08713 s4.2: OCP maps amax to [4, 8), OAS to (3.5, 7]; 7.6 -> 3.8 -> e2m1 4, times 2 = 8
    x = block(7.6)
    assert R.e8m0_oas(x) == R.e8m0_ocp(x) + 1
    assert R.dequant_mxfp4(R.quant_mxfp4(x, "oas"))[0] == 8.0


def test_oas_keeps_ocp_scale_at_and_below_7():
    for amax in (4.0, 6.0, 7.0):
        assert R.e8m0_oas(block(amax)) == R.e8m0_ocp(block(amax)), amax
    above = np.nextafter(np.float32(7.0), np.float32(8))   # threshold + 1 ulp
    assert R.e8m0_oas(block(above)) == R.e8m0_ocp(block(above)) + 1


def test_e8m0_never_emits_nan_byte():
    x = np.full(32, np.finfo(np.float32).max, np.float32)
    assert R.e8m0_ocp(x) <= 254
    assert R.e8m0_oas(x) <= 254


def test_block16_has_two_scales_per_32_values():
    x = np.zeros(32, np.float32); x[0] = 6.0; x[16] = 0.75
    q = R.quant_mxfp4(x, "oas", block=16)
    assert list(q.scales) == [127, 124]                   # 0.75 -> floor(log2) -1 - 2 = -3
    np.testing.assert_array_equal(R.dequant_mxfp4(q)[[0, 16]], [6.0, 0.75])


def test_hadamard32_is_self_inverse():
    x = np.random.default_rng(11).standard_normal((4, 32)).astype(np.float32)
    np.testing.assert_allclose(R.hadamard32(R.hadamard32(x)), x, atol=1e-5)


def test_hadamard32_of_unit_vector():
    e0 = np.zeros(32, np.float32); e0[0] = 1.0
    np.testing.assert_allclose(R.hadamard32(e0), np.full(32, 32 ** -0.5), atol=1e-7)


def test_paired_rotation_keeps_scores():
    q, k = np.random.default_rng(3).standard_normal((2, 256)).astype(np.float32)
    np.testing.assert_allclose(R.hadamard32_blocks(q) @ R.hadamard32_blocks(k), q @ k, rtol=1e-5)


def test_v_rotation_needs_output_inverse():
    rng = np.random.default_rng(5)
    p = rng.random((1, 8)).astype(np.float32)
    v = rng.standard_normal((8, 256)).astype(np.float32)
    o = R.hadamard32_blocks(p @ R.hadamard32_blocks(v))
    np.testing.assert_allclose(o, p @ v, atol=1e-5)


# No unit test claims "rotation lowers error": that depends on the distribution and is an S1 measurement.


def test_int4_tokenwise_round_trip_on_grid_is_exact():
    x = np.tile(np.array([*range(-7, 8), 0], np.float32), 16)[None, :]   # 256 values, amax 7
    np.testing.assert_array_equal(R.dequant_int4(R.quant_int4(x, group=256)), x)


def test_int4_scale_is_fp16_amax_over_7():
    x = np.zeros((1, 64), np.float32); x[0, 3] = -3.5
    q = R.quant_int4(x, group=64)
    assert q.scales.dtype == np.float16 and float(q.scales[0, 0]) == 0.5
    assert int(q.codes[0, 3]) == -7


def test_bytes_per_vector():
    assert R.bytes_per_vector("mxfp4-ocp", 256) == 136       # 128 payload + 8 E8M0
    assert R.bytes_per_vector("mxfp4-16-oas", 256) == 144
    assert R.bytes_per_vector("int4-token-256", 256) == 130  # + one fp16 scale
    assert R.bytes_per_vector("int4-token-64", 256) == 136
    assert R.bytes_per_vector("fp8", 256) == 256


def test_unknown_format_is_rejected():
    with pytest.raises(ValueError, match="kv format"):
        R.roundtrip(np.zeros((1, 256), np.float32), "mxfp3")


def test_attention_error_is_zero_for_lossless_and_matches_hand_value():
    rng = np.random.default_rng(7)
    q = rng.standard_normal((2, 256)).astype(np.float32)
    k = rng.standard_normal((5, 256)).astype(np.float32)
    v = rng.standard_normal((5, 256)).astype(np.float32)
    assert R.attention_error(q, k, v, k, v) == 0.0
    # V scaled by 1.1 with K unchanged -> O scales by 1.1 exactly -> relative error 0.1
    assert abs(R.attention_error(q, k, v, k, 1.1 * v) - 0.1) < 1e-6


@pytest.mark.parametrize("rule", ["ocp", "oas"])
def test_scale_rules_are_vectorized_over_blocks(rule):
    # the S1 study quantizes ~10^7 blocks; per-block Python calls are too slow
    rng = np.random.default_rng(13)
    g = (rng.standard_normal((50, 32)) * np.exp2(rng.integers(-12, 12, (50, 1)))).astype(np.float32)
    g[3] = 0.0
    g[4, 0] = 7.5 * 2.0 ** -5
    f = R.SCALE_RULES[rule]
    np.testing.assert_array_equal(f(g), [f(row) for row in g])
