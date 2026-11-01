"""Self-check of prefill_torch_split.py on a synthetic kineto trace. `pytest ci/` or run directly."""
import gzip
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill_torch_split as P  # noqa: E402

GEMM2 = "void radiance_mxfp4_fp8_gemm_atiled<2, true, 128, false>(unsigned char const*)"
GEMM4 = "void radiance_mxfp4_fp8_gemm_atiled<4, true, 64, false>(unsigned char const*)"
SILU = "void radiance_silu_mul_quant<256, true>(bf16 const*)"
GDN = "r4d_gdn_chunk_scan_kernel(unsigned short const*)"
FUSED = "triton_poi_fused_0.kd"


def trace(path, m=1000):
    ev, t, cid = [], 0.0, 0

    def op(name, kname, dur, grid, gap):
        nonlocal t, cid
        cid += 1
        ev.append(dict(ph="X", cat="cpu_op", name=name, ts=t, dur=5.0))
        ev.append(dict(ph="X", cat="cuda_runtime", name="hipLaunchKernel", ts=t + 1, dur=1.0,
                       args=dict(correlation=cid)))
        ev.append(dict(ph="X", cat="kernel", name=kname, ts=t + 10, dur=dur,
                       args=dict(correlation=cid, grid=[grid, 6, 1])))
        t += 10 + dur + gap

    ev.append(dict(ph="X", cat="user_annotation", name=f"hybrid_triton_w4a16 {m}x5120x25600", ts=0, dur=1))
    op("radiance::mxfp4_linear_pq", GEMM2, 1000.0, 544, 100)          # gate_up, N=544*64, 100 us gap
    op("radiance::silu_mul_quant", SILU, 200.0, m, 5)
    op("radiance::mxfp4_linear_pq", GEMM2, 500.0, 80, 5)              # down: after silu -> K=17408
    op("vllm::qwen_gdn_attention_core", GDN, 300.0, 1, 2000)          # misfiled as attention by name rules
    op("aten::add", FUSED, 50.0, 0, 5)                                # no custom op -> name rule
    op("radiance::mxfp4_linear_pq", GEMM4, 250.0, 40, 0)              # TN=4: N=40*128 -> o/out_proj
    with gzip.open(path, "wt") as f:
        json.dump(dict(traceEvents=ev), f)


def test_split():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "trace_1000_0.json.gz")
        trace(p)
        r = P.main([d, "--json", os.path.join(d, "o.json")])["1000"]["median"]
        c = r["cls_ms"]
        assert abs(c["GEMM"] - 1.75) < 1e-9 and abs(c["norm-quant"] - 0.2) < 1e-9
        assert abs(c["GDN"] - 0.3) < 1e-9 and "attention" not in c
        assert abs(c["inductor-fused"] - 0.05) < 1e-9
        assert abs(c["idle<=1ms"] - 0.155) < 1e-9   # gaps 110 + 3 * 15 us (launch offset 10 us each)
        assert abs(c["idle>1ms"] - 2.01) < 1e-9
        assert abs(sum(c.values()) - r["span_ms"]) < 1e-9                   # classes + idle == span
        g = r["gemm"]
        assert set(g) == {"gate_up:atiled", "down:atiled", "o/out_proj:atiled"}
        assert abs(g["gate_up:atiled"]["tfs"] - 2 * 1000 * 34816 * 5120 / 1e-3 / 1e12) < 1e-6
        assert abs(g["down:atiled"]["tfs"] - 2 * 1000 * 5120 * 17408 / 5e-4 / 1e12) < 1e-6
        assert abs(g["o/out_proj:atiled"]["tfs"] - 2 * 1000 * 5120 * 6144 / 2.5e-4 / 1e12) < 1e-6


if __name__ == "__main__":
    test_split()
    print("ok")
