"""Parser self-check on a synthetic rocprofv3-style trace.  `pytest ci/` or `python3 ci/test_prefill_trace_split.py`."""
import csv
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prefill_trace_split as P  # noqa: E402

HDR = ["Kind", "Agent_Id", "Queue_Id", "Stream_Id", "Thread_Id", "Dispatch_Id", "Kernel_Id", "Kernel_Name",
       "Correlation_Id", "Start_Timestamp", "End_Timestamp", "LDS_Block_Size", "Scratch_Size", "VGPR_Count",
       "Accum_VGPR_Count", "SGPR_Count", "Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z",
       "Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z"]
GEMM = "void radiance::gemm_nt_w4a8_mxfp4<256>(...)"
ATTN = "kernel_unified_attention_2d"
GDN = "chunk_gated_delta_rule_fwd_kernel_h"
NORM = "rms_norm_quant_fp8_kernel"
OTHER = "elementwise_copy"
MS = 1_000_000


def write_trace(path):
    """Burst A (startup, junk), burst B ('2k'), burst C ('64k'), 10 s apart. Per layer: GEMM 100us (down:
    grid 5120, other: 17408), attn 50, gdn 30, norm 10, other 5; 20 us gap between kernels, queue 1."""
    rows, t, did = [], 0, 0

    def emit(name, dur, grid):
        nonlocal t, did
        rows.append(["KERNEL_DISPATCH", 1, 1, 1, 1, did, did, name, did, t, t + dur, 0, 0, 64, 0, 32,
                     256, 1, 1, grid, 1, 1])
        did += 1
        t += dur + 20_000

    def burst(layers, n_down):
        for i in range(layers):
            emit(GEMM, 100_000, 5120 if i < n_down else 17408)
            emit(ATTN, 50_000, 4096); emit(GDN, 30_000, 64); emit(NORM, 10_000, 8); emit(OTHER, 5_000, 8)

    burst(3, 1); t += 10_000 * MS      # start-up burst
    burst(64, 64); t += 10_000 * MS    # "2k": 64 layers
    burst(128, 128)                    # "64k"
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HDR); w.writerows(rows)


def test_split():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "host"))
        p = os.path.join(d, "host", "123_kernel_trace.csv")
        write_trace(p)
        assert P.classify(GEMM) == "GEMM" and P.classify(ATTN) == "attention" and P.classify(GDN) == "GDN"
        assert P.classify(NORM) == "norm-quant" and P.classify(OTHER) == "other"
        assert P.find_trace(d) == p
        res = P.main([d, "--forwards", "2k=1,64k=2", "--json", os.path.join(d, "o.json")])
        a, b = res["2k"], res["64k"]
        assert a["n"] == 64 * 5 and b["n"] == 128 * 5
        assert a["cls"]["GEMM"] == [64 * 100_000, 64]
        assert b["cls"]["attention"] == [128 * 50_000, 128]
        assert a["cls"]["GDN"][0] == 64 * 30_000 and a["cls"]["norm-quant"][0] == 64 * 10_000
        assert a["cls"]["other"][0] == 64 * 5_000
        # 5 kernels per layer, 4 gaps inside + 1 between layers = 5 gaps/layer minus the very last
        assert a["gap_n"] == 64 * 5 - 1 and a["gap_p50"] == 20_000 and a["gap_ns"] == (64 * 5 - 1) * 20_000
        assert a["busy_ns"] == 64 * 195_000 and a["span_ns"] == a["busy_ns"] + a["gap_ns"]
        (nm, grid, wg), (ns, n) = a["gemm"][0]
        assert grid == (5120, 1, 1) and wg == (256, 1, 1) and n == 64
        assert [g for (_, g, _), _ in b["gemm"]] == [(5120, 1, 1)] and b["gemm"][0][1][1] == 128
        assert a["kern"][0][0] == GEMM
        assert os.path.getsize(os.path.join(d, "o.json")) > 100
    print("ok")


if __name__ == "__main__":
    test_split()
