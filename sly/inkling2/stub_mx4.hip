// Stub build of radiance_inkling2_mx4 for the RED half of the GPU window: gemv_mx4 and dequant_mx4 zero their
// output. The correctness tests must fail against it. Same module name as the real kernel (the test loader
// imports a build by path under that name).
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
namespace py = pybind11;

PYBIND11_MODULE(radiance_inkling2_mx4, m) {
  m.attr("NCFG") = 12;
  py::list cfgs;
  for (int i = 0; i < 12; i++) cfgs.append(py::make_tuple(1, 4, 4));
  m.attr("CFGS") = cfgs;
  m.def("auto_cfg_mx4", [](int64_t, int64_t, int64_t) { return 0; }, py::arg("M"), py::arg("N"), py::arg("K"));
  m.def("gemv_mx4", [](uintptr_t x, uintptr_t q, uintptr_t s, uintptr_t y, int64_t M, int64_t N, int64_t K,
                       int64_t SX, int64_t SY, int64_t out32, int64_t ntiles, uintptr_t stream, int64_t cfg) {
    if (M >= 1 && M <= 16)  // never writes more than M * N elements, whatever the arguments
      (void)hipMemsetAsync((void*)y, 0, (size_t)M * (size_t)N * (out32 ? 4 : 2), (hipStream_t)stream);
  }, "stub: zero output", py::arg("x"), py::arg("q"), py::arg("s"), py::arg("y"), py::arg("M"), py::arg("N"),
     py::arg("K"), py::arg("SX"), py::arg("SY"), py::arg("out32"), py::arg("ntiles"), py::arg("stream"),
     py::arg("cfg") = -1);
  m.def("dequant_mx4", [](uintptr_t q, uintptr_t s, uintptr_t y, int64_t N, int64_t K, int64_t ntiles,
                          uintptr_t stream) {
    if (N >= 1 && K >= 1)
      (void)hipMemsetAsync((void*)y, 0, (size_t)N * (size_t)K * 2, (hipStream_t)stream);
  }, "stub: zero output", py::arg("q"), py::arg("s"), py::arg("y"), py::arg("N"), py::arg("K"),
     py::arg("ntiles"), py::arg("stream"));
}
