// RADIANCE gfx1201 MXFP4 (e2m1 + E8M0 per 32 K) W4A16 low-M GEMV, M 1..16, plus the prefill dequant:
//   gemv_mx4     y[M,N] = x[M,K] . deq(W)[N,K]^T, bf16 activations, fp32 accumulation, bf16 or fp32 output
//   dequant_mx4  W tiles -> bf16 [N,K] row-major (the prefill route: M > 16 runs a bf16 GEMM on it)
//
// Why: the MXFP4 W4A8 route (radiance_mxfp4_fp8: fp8 activation quant launch + the Qwen-tuned fp8-WMMA decode
// kernel) streams the 4-bit bytes at 300-330 GB/s and pays ~12 us per call even for N = 384. This kernel is the
// int8 GEMV's design (radiance_inkling2_gemv.hip, gemv_q8) with 4-bit codes: one launch, no activation
// quantization, no split-K scratch, graph-capture safe, 500+ GB/s of 4-bit bytes.
//   * e2m1 x 2^(E8M0 - 127) is exact: the codes are widened EXACTLY to bf16 (magnitudes 0 .5 1 1.5 2 3 4 6 and
//     their negatives), the block scale is applied to the exact K = 32 fp32 partial sum. Result = dequantize-
//     then-dot up to fp32 summation order. Scale byte 0 means factor 0 (the quantizer never emits it with a
//     non-zero code; padding rows are all-zero bytes).
//   * widen by byte LUTs: v_perm_b32 picks the 4 bf16 HIGH bytes (3F 3F 3F 40 ..) and the 4 LOW bytes of the 4
//     magnitudes in one instruction each (selector = the magnitude bits of 4 nibbles, w & 0x07070707), the sign
//     bit is OR-ed into the high byte, a second perm interleaves low/high bytes into bf16 pairs. ~2 VALU
//     ops per weight. A dword of 8 nibbles is ONE wmma operand (8 bf16 slots); the nibble order is
//     slot 0..7 <- nibble 0 2 4 6 1 3 5 7 (the order the perms produce them), which the pre-tiled copy
//     bakes in (radiance_inkling2_mxfp4dense.pack_mx4).
//   * the weights are read from a TILED copy. A tile is 16 weight rows, a group is 64 K = 2 scale blocks. Per
//     (tile t, group g) the code bytes are [half h][row c][block b][step j][4 B] at (t * KG + g) * 512 +
//     lane * 16, lane = 16 h + c: one wave load = 512 contiguous bytes (4 whole cache lines), no LDS staging.
//     Dword 2 b + j of the lane holds the 8 codes of K 64 g + 32 b + 16 h + 8 j + slot. The scale BYTES are
//     [row c][block b] at (t * KG + g) * 32 + c * 2 + b (lane reads its row's two scales: one u16).
//   * v_wmma_f32_16x16x16_bf16, roles as in gemv_q8: A = x (lane 16 h + i = x row i, rows >= M are zero), B =
//     weights (lane 16 h + c = weight row c); the accumulator C[x row 8 h + e][weight row c] sits in lane
//     16 h + c, so ONE scale per lane applies to all 8 accumulator registers; per block 2 wmma steps (8 + 8 K per
//     lane half) on a fresh zero accumulator, then acc += tmp * 2^(e - 127).
//   * a wave owns T tiles and a share of the K groups; the W waves of a block meet in LDS at the end. D weight
//     groups (512 B per tile) are in flight per wave; x (L2 resident) is prefetched one group ahead.
//   Launch config cfg = (T, W, D) from MX_CFGS; cfg -1 picks by shape.
// Guards (thrown, never a GPU fault): M 1..16, K % 64, SX % 8, SY >= N, 16 B aligned x / q / s (/ y), tiles >=
// ceil(N / 16), cfg.
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

#if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32
#error "gemv_mx4 is written for wave32 (gfx12)"
#endif

#define MAX_M 16
#define MX_TILE 16
#define MX_GROUP 64
#define MX_GROUP_BYTES 512
#define MX_GROUP_SCALES 32
#define MX_NCFG 12
static const int MX_T[MX_NCFG] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
static const int MX_W[MX_NCFG] = {2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8};
static const int MX_D[MX_NCFG] = {4, 4, 4, 8, 8, 8, 4, 4, 4, 8, 8, 8};  // even: x double buffer parity

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ uint16_t f2bf(float v) {  // round to nearest even (finite values)
  const uint32_t u = __float_as_uint(v);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// One dword of 8 e2m1 codes -> 4 dwords of bf16 pairs (one wmma operand), exact. Slot s <- nibble 0 2 4 6 1 3 5 7.
__device__ __forceinline__ u32x4 mx4_decode(uint32_t w) {
  const uint32_t se = w & 0x07070707u;         // magnitudes of nibbles 0 2 4 6
  const uint32_t so = (w >> 4) & 0x07070707u;  // magnitudes of nibbles 1 3 5 7
  // bf16 high / low byte of 0 .5 1 1.5 2 3 4 6: entries 0..3 in the second (low) source, 4..7 in the first
  const uint32_t he = __builtin_amdgcn_perm(0x40404040u, 0x3F3F3F00u, se) | ((w << 4) & 0x80808080u);
  const uint32_t ho = __builtin_amdgcn_perm(0x40404040u, 0x3F3F3F00u, so) | (w & 0x80808080u);
  const uint32_t le = __builtin_amdgcn_perm(0xC0804000u, 0xC0800000u, se);
  const uint32_t lo = __builtin_amdgcn_perm(0xC0804000u, 0xC0800000u, so);
  return (u32x4){__builtin_amdgcn_perm(he, le, 0x05010400u), __builtin_amdgcn_perm(he, le, 0x07030602u),
                 __builtin_amdgcn_perm(ho, lo, 0x05010400u), __builtin_amdgcn_perm(ho, lo, 0x07030602u)};
}

__device__ __forceinline__ float e8m0_factor(uint32_t e) { return __uint_as_float((e & 0xFFu) << 23); }

template <int T, int W, int D>
__global__ __launch_bounds__(W * 32) void gemv_mx4_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ q, const uint8_t* __restrict__ s,
    void* __restrict__ y, int M, int N, int SX, int SY, int out32, int ngroups, int ntiles) {
  extern __shared__ f32x8 redq[];  // [W][T][32 lanes] accumulators for the cross-wave sum
  const int tid = threadIdx.x;
  const int lane = tid & 31, wv = tid >> 5;
  const int c = lane & 15, h = lane >> 4;
  const int tile0 = blockIdx.x * T;
  const uint8_t* qp[T];
  const uint8_t* sp[T];
#pragma unroll
  for (int t = 0; t < T; t++) {  // tiles past the last one read the last tile and are never stored
    const int64_t tile = min(tile0 + t, ntiles - 1);
    qp[t] = q + tile * ngroups * MX_GROUP_BYTES + lane * 16;
    sp[t] = s + tile * ngroups * MX_GROUP_SCALES + c * 2;
  }
  const bool xlive = c < M;  // rows of the A operand past M are zero
  const uint16_t* xr = x + (int64_t)(xlive ? c : 0) * SX + h * 16;
  const int g0 = (int)((int64_t)ngroups * wv / W);  // this wave's groups [g0, g1)
  const int g1 = (int)((int64_t)ngroups * (wv + 1) / W);
  f32x8 acc[T];
#pragma unroll
  for (int t = 0; t < T; t++) acc[t] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  struct WBuf {
    u32x4 w[T];
    uint32_t sc[T];
  };
  struct XBuf {
    u32x4 x[2][2];
  };
  auto loadw = [&](WBuf& b, int g) {
#pragma unroll
    for (int t = 0; t < T; t++) {
      b.w[t] = __builtin_nontemporal_load((const u32x4*)(qp[t] + (int64_t)g * MX_GROUP_BYTES));
      b.sc[t] = *(const uint16_t*)(sp[t] + (int64_t)g * MX_GROUP_SCALES);
    }
  };
  auto loadx = [&](XBuf& b, int g) {
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
      for (int j = 0; j < 2; j++)
        b.x[k][j] = xlive ? *(const u32x4*)(xr + g * MX_GROUP + k * 32 + j * 8) : (u32x4){0, 0, 0, 0};
  };
  auto consume = [&](const WBuf& b, const XBuf& xb) {
#pragma unroll
    for (int t = 0; t < T; t++)
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const u32x4 w0 = mx4_decode(k ? b.w[t].z : b.w[t].x);
        const u32x4 w1 = mx4_decode(k ? b.w[t].w : b.w[t].y);
        f32x8 tmp = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
        tmp = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(bf16x8, xb.x[k][0]),
                                                                  __builtin_bit_cast(bf16x8, w0), tmp);
        tmp = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(bf16x8, xb.x[k][1]),
                                                                  __builtin_bit_cast(bf16x8, w1), tmp);
        const float sc = e8m0_factor(b.sc[t] >> (8 * k));
#pragma unroll
        for (int e = 0; e < 8; e++) acc[t][e] = __builtin_fmaf(tmp[e], sc, acc[t][e]);
      }
  };
  WBuf wb[D];
  XBuf xb[2];
#pragma unroll
  for (int d = 0; d < D - 1; d++)
    if (g0 + d < g1) loadw(wb[d], g0 + d);
  if (g0 < g1) loadx(xb[0], g0);
  for (int g = g0; g < g1; g += D) {
#pragma unroll
    for (int d = 0; d < D; d++) {
      if (g + d + D - 1 < g1) loadw(wb[(d + D - 1) % D], g + d + D - 1);
      if (g + d + 1 < g1) loadx(xb[(d + 1) & 1], g + d + 1);
      if (g + d < g1) consume(wb[d], xb[d & 1]);
    }
  }
#pragma unroll
  for (int t = 0; t < T; t++) redq[(wv * T + t) * 32 + lane] = acc[t];
  __syncthreads();
  for (int id = tid; id < T * 256; id += W * 32) {  // id = (tile t, x row e of the lane half, lane l)
    const int t = id >> 8, e = (id >> 5) & 7, l = id & 31;
    float v = 0.f;
#pragma unroll
    for (int g = 0; g < W; g++) v += redq[(g * T + t) * 32 + l][e];
    const int m = 8 * (l >> 4) + e, n = (tile0 + t) * 16 + (l & 15);
    if (m >= M || n >= N) continue;
    if (out32) ((float*)y)[(int64_t)m * SY + n] = v;
    else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(v);
  }
}

struct Args {
  uintptr_t x, q, s, y, stream;
  int M, N, K, SX, SY, out32;
};

template <int T, int W, int D>
static void run_mx4(const Args& a) {
  const int ntiles = (a.N + MX_TILE - 1) / MX_TILE;
  hipLaunchKernelGGL((gemv_mx4_kernel<T, W, D>), dim3((ntiles + T - 1) / T), dim3(W * 32), (size_t)W * T * 1024,
                     (hipStream_t)a.stream, (const uint16_t*)a.x, (const uint8_t*)a.q, (const uint8_t*)a.s,
                     (void*)a.y, a.M, a.N, a.SX, a.SY, a.out32, a.K / MX_GROUP, ntiles);
}

static void run_mx4_cfg(int cfg, const Args& a) {
  switch (cfg) {
    case 0: run_mx4<1, 2, 4>(a); break;
    case 1: run_mx4<1, 4, 4>(a); break;
    case 2: run_mx4<1, 8, 4>(a); break;
    case 3: run_mx4<1, 2, 8>(a); break;
    case 4: run_mx4<1, 4, 8>(a); break;
    case 5: run_mx4<1, 8, 8>(a); break;
    case 6: run_mx4<2, 2, 4>(a); break;
    case 7: run_mx4<2, 4, 4>(a); break;
    case 8: run_mx4<2, 8, 4>(a); break;
    case 9: run_mx4<2, 2, 8>(a); break;
    case 10: run_mx4<2, 4, 8>(a); break;
    case 11: run_mx4<2, 8, 8>(a); break;
  }
}

// cfg -1, from the rotated-pool CUDA-graph sweep at M = 1, 8, 16 (bench_mx4.py --cfgs, us per call at M = 8):
//   N 384 / 1024 (k/v, kv_ctx):  cfg 2  (T=1 W=8 D=4)  4.4 / 6.0 us vs 5.3 / 6.0 for cfg 1
//   N 6552 (global q side):      cfg 7  (T=2 W=4 D=4)  18.9 vs 20.0 for cfg 1
//   N >= 131072 (unembed):       cfg 11 (T=2 W=8 D=8)  394 vs 442 for cfg 1 (M = 1: 378, M = 16: 409)
//   everything else (3072, 2304 x 3072, 8600): cfg 1 (T=1 W=4 D=4), within ~5 % of the best cfg
// Mid-size shapes are launch bound: ~4 us fixed + bytes at ~550 GB/s.
static int pick_cfg_mx4(int M, int N, int K) {
  if (N >= (1 << 17)) return 11;
  if (N < 2048) return 2;
  if (N >= 6000 && N < 7000) return 7;
  return 1;
}

// ===================================================================== prefill dequant
// One thread per (tile, group, lane): its 16 B of codes -> 4 x 8 bf16 (scaled, exact), stored as 4 x 16 B.
__global__ void dequant_mx4_kernel(const uint8_t* __restrict__ q, const uint8_t* __restrict__ s,
                                   uint16_t* __restrict__ y, int N, int K, int ngroups, int64_t total) {
  const int64_t id = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= total) return;
  const int lane = (int)(id & 31);
  const int64_t tg = id >> 5;  // tile * ngroups + group
  const int g = (int)(tg % ngroups), tile = (int)(tg / ngroups);
  const int c = lane & 15, h = lane >> 4;
  const int row = tile * MX_TILE + c;
  if (row >= N) return;
  const u32x4 v = *(const u32x4*)(q + tg * MX_GROUP_BYTES + lane * 16);
  const uint32_t sc2 = *(const uint16_t*)(s + tg * MX_GROUP_SCALES + c * 2);
#pragma unroll
  for (int b = 0; b < 2; b++) {
    const float sc = e8m0_factor(sc2 >> (8 * b));
#pragma unroll
    for (int j = 0; j < 2; j++) {
      const u32x4 d = mx4_decode(b ? (j ? v.w : v.z) : (j ? v.y : v.x));
      u32x4 o;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const float lo = __uint_as_float(d[i] << 16) * sc, hi = __uint_as_float(d[i] & 0xFFFF0000u) * sc;
        o[i] = (uint32_t)f2bf(lo) | ((uint32_t)f2bf(hi) << 16);
      }
      *(u32x4*)(y + (int64_t)row * K + g * MX_GROUP + b * 32 + h * 16 + j * 8) = o;
    }
  }
}

PYBIND11_MODULE(radiance_inkling2_mx4, m) {
  m.attr("NCFG") = MX_NCFG;
  py::list cfgs;
  for (int i = 0; i < MX_NCFG; i++) cfgs.append(py::make_tuple(MX_T[i], MX_W[i], MX_D[i]));
  m.attr("CFGS") = cfgs;
  m.def("auto_cfg_mx4", [](int64_t M, int64_t N, int64_t K) { return pick_cfg_mx4((int)M, (int)N, (int)K); },
        "launch config gemv_mx4 picks for cfg -1", py::arg("M"), py::arg("N"), py::arg("K"));
  m.def("gemv_mx4", [](uintptr_t x, uintptr_t q, uintptr_t s, uintptr_t y, int64_t M, int64_t N, int64_t K,
                       int64_t SX, int64_t SY, int64_t out32, int64_t ntiles, uintptr_t stream, int64_t cfg) {
    if (M < 1 || M > MAX_M) throw std::invalid_argument("M must be 1..16");
    if (N < 1 || K < MX_GROUP || K % MX_GROUP || SX % 8 || SX < K || SY < N)
      throw std::invalid_argument("need N >= 1, K % 64 == 0, SX % 8 == 0, SX >= K, SY >= N");
    if (x % 16 || q % 16 || s % 16) throw std::invalid_argument("x, q, s must be 16 B aligned");
    if (ntiles < (N + MX_TILE - 1) / MX_TILE) throw std::invalid_argument("fewer tiles than ceil(N / 16)");
    if (cfg < -1 || cfg >= MX_NCFG) throw std::invalid_argument("cfg out of range");
    if (cfg < 0) cfg = pick_cfg_mx4((int)M, (int)N, (int)K);
    run_mx4_cfg((int)cfg, Args{x, q, s, y, stream, (int)M, (int)N, (int)K, (int)SX, (int)SY, (int)(out32 != 0)});
  }, "MXFP4 tiled-weight low-M GEMV: y[M,N] = x[M,K] bf16 @ deq(W)[N,K]^T, fp32 accumulation",
     py::arg("x"), py::arg("q"), py::arg("s"), py::arg("y"), py::arg("M"), py::arg("N"), py::arg("K"),
     py::arg("SX"), py::arg("SY"), py::arg("out32"), py::arg("ntiles"), py::arg("stream"), py::arg("cfg") = -1);
  m.def("dequant_mx4", [](uintptr_t q, uintptr_t s, uintptr_t y, int64_t N, int64_t K, int64_t ntiles,
                          uintptr_t stream) {
    if (N < 1 || K < MX_GROUP || K % MX_GROUP) throw std::invalid_argument("need N >= 1, K % 64 == 0");
    if (q % 16 || s % 16 || y % 16) throw std::invalid_argument("q, s, y must be 16 B aligned");
    if (ntiles < (N + MX_TILE - 1) / MX_TILE) throw std::invalid_argument("fewer tiles than ceil(N / 16)");
    const int64_t ngroups = K / MX_GROUP, total = ((N + MX_TILE - 1) / MX_TILE) * ngroups * 32;
    hipLaunchKernelGGL(dequant_mx4_kernel, dim3((total + 255) / 256), dim3(256), 0, (hipStream_t)stream,
                       (const uint8_t*)q, (const uint8_t*)s, (uint16_t*)y, (int)N, (int)K, (int)ngroups, total);
  }, "tiled MXFP4 weights -> bf16 [N, K] row-major (exact)", py::arg("q"), py::arg("s"), py::arg("y"),
     py::arg("N"), py::arg("K"), py::arg("ntiles"), py::arg("stream"));
}
