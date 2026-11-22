// Stub build of radiance_inkling2_gemv for the RED half of the GPU window: gemv_q8 and gemv_bf16 zero y.
// The correctness tests must fail against it. Same module name as the real kernel (the test
// loader imports a build by path under that name).
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
namespace py = pybind11;

PYBIND11_MODULE(radiance_inkling2_gemv, m) {
  m.attr("NCFG") = 8;
  m.attr("CFGS") = py::list();
  m.attr("BF16_NCFG") = 3;
  py::list cfgs;
  for (int i = 0; i < 3; i++) cfgs.append(py::make_tuple(1, 4));
  m.attr("BF16_CFGS") = cfgs;
  m.def("gemv_bf16", [](uintptr_t x, uintptr_t w, uintptr_t bias, uintptr_t y, int64_t M, int64_t N,
                        int64_t K, int64_t SX, int64_t SW, int64_t SY, int64_t out32, uintptr_t stream,
                        int64_t cfg) {
    if (M >= 1 && M <= 16)  // never writes more than M * N elements, whatever the arguments
      (void)hipMemsetAsync((void*)y, 0, (size_t)M * (size_t)N * (out32 ? 4 : 2), (hipStream_t)stream);
  }, "stub: zero output", py::arg("x"), py::arg("w"), py::arg("bias"), py::arg("y"), py::arg("M"),
     py::arg("N"), py::arg("K"), py::arg("SX"), py::arg("SW"), py::arg("SY"), py::arg("out32"),
     py::arg("stream"), py::arg("cfg") = -1);
  m.def("gemv_q8", [](uintptr_t x, uintptr_t w, uintptr_t scale, uintptr_t bias, uintptr_t y,
                      int64_t M, int64_t N, int64_t K, int64_t SX, int64_t SW, int64_t SY,
                      int64_t out32, uintptr_t scratch, uintptr_t scratch_bytes,
                      uintptr_t stream, int64_t cfg) {
    (void)hipMemsetAsync((void*)y, 0, (size_t)M * (size_t)N * (out32 ? 4 : 2), (hipStream_t)stream);
  }, "stub: zero output", py::arg("x"), py::arg("w"), py::arg("scale"), py::arg("bias"),
     py::arg("y"), py::arg("M"), py::arg("N"), py::arg("K"), py::arg("SX"), py::arg("SW"),
     py::arg("SY"), py::arg("out32"), py::arg("scratch"), py::arg("scratch_bytes"),
     py::arg("stream"), py::arg("cfg") = -1);
}
