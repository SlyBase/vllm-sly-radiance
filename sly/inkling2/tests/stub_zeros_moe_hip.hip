// RED stand-in for radiance_inkling2_moe_hip: same API, writes zeros.  tests/test_moe_hip.py must
// FAIL against it (build: hipcc ... stub_zeros_moe_hip.hip -o radiance_inkling2_moe_hip.so).
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
namespace py = pybind11;

static void zeros(uintptr_t out, int max_ctas, int N, uintptr_t stream) {
    (void)hipMemsetAsync(reinterpret_cast<void *>(out), 0, (size_t)max_ctas * 8 * N * 2,
                         reinterpret_cast<hipStream_t>(stream));
}

PYBIND11_MODULE(radiance_inkling2_moe_hip, m) {
    m.def("nvfp4_gemm", [](uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t,
                           uintptr_t out, uintptr_t, int max_ctas, int, int N, int, int, int,
                           uintptr_t stream) { zeros(out, max_ctas, N, stream); });
    m.def("mxfp8_gemm", [](uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t, uintptr_t out,
                           uintptr_t, int max_ctas, int, int N, int, int, int, uintptr_t stream) {
        zeros(out, max_ctas, N, stream);
    });
    m.def("geometry", [] {
        py::dict d;
        d["bn"] = 32; d["lpr"] = 8; d["quantum"] = 32; d["nj_max"] = 9; d["max_rows"] = 8;
        d["k_max"] = 2304;
        return d;
    });
}
