#!/usr/bin/env python3
"""CPU test for trace_phases.py (rocprofv3 kernel_trace.csv -> per-request GPU busy / idle).

Run:  python3 sly/inkling2/ttft/test_trace_phases_cpu.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_phases import phases  # noqa: E402

MS = 1_000_000  # ns


def row(s_ms, e_ms, name="k"):
    return {"Start_Timestamp": str(int(s_ms * MS)), "End_Timestamp": str(int(e_ms * MS)), "Kernel_Name": name}


def test_two_phases_split_by_idle_gap():
    rows = [row(0, 1), row(2, 3), row(4, 5), row(5000, 5001), row(5002, 5004)]
    p = phases(rows, gap_ms=500)
    assert len(p) == 2
    assert p[0]["kernels"] == 3 and p[1]["kernels"] == 2
    assert abs(p[0]["span_ms"] - 5.0) < 1e-6
    assert abs(p[0]["busy_ms"] - 3.0) < 1e-6          # 1 + 1 + 1
    assert abs(p[0]["idle_ms"] - 2.0) < 1e-6
    assert abs(p[1]["busy_ms"] - 3.0) < 1e-6          # 1 + 2


def test_overlap_is_not_double_counted():
    rows = [row(0, 10), row(5, 12)]                   # two streams overlapping
    p = phases(rows, gap_ms=500)
    assert len(p) == 1 and abs(p[0]["busy_ms"] - 12.0) < 1e-6 and p[0]["idle_ms"] == 0


def test_top_kernels_by_time():
    rows = [row(0, 4, "big"), row(5, 6, "small"), row(7, 8, "small")]
    p = phases(rows, gap_ms=500, top=1)
    assert p[0]["top"][0][0] == "big"


def test_empty():
    assert phases([], gap_ms=500) == []


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
    print("OK")
