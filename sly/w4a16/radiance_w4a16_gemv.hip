// RADIANCE gfx1201 symmetric W4A16 (unsigned nibble n, q = n - 8, bf16 group scales) low-M GEMV, M 1..16:
//   gemv_u4   y[M,N] = x[M,K] . deq(W)[N,K]^T (+ bias[N]), bf16 activations, fp32 accumulation, bf16 or fp32 out
//
// Why: M = 8 (a DFlash2 drafter step, the int4 lm_head) is past the skinny HIP kernel's 5 rows and falls to the
// Triton W4A16 kernel at 230-420 GB/s. This is the pre-tiled wave32 WMMA GEMV design of radiance_inkling2_mx4.hip
// (one launch, no split-K scratch, graph-capture safe) with the 4-bit decode and the scale layout of a symmetric
// integer format:
//   * q = n - 8 is exact in bf16 for every nibble (BF16 -8..7: C100 C0E0 C0C0 C0A0 C080 C040 C000 BF80 0000 3F80
//     4000 4040 4080 40A0 40C0 40E0). Decode by the fp32 magic number: 2^23 | n 16^p is the float 2^23 + n 16^p
//     (p <= 4 keeps it in the mantissa), ONE fma by 16^-p and the constant -(2^23 16^-p + 8) leaves exactly n - 8
//     (+0 for q = 0). The high half of that fp32 IS the bf16 (an integer of at most 4 significant bits), v_perm_b32
//     packs two of them. No branches, no tables, 2 VALU ops per weight plus one perm per pair.
//   * a dword of 8 nibbles is ONE wmma operand (8 bf16 slots). Slot 0..7 <- nibble position 0 2 4 6 1 3 5 7 of the
//     dword (u4_decode); the tiles bake that order in (radiance_w4a16_gemv_hip.pack_u4).
//   * the weights are read from a TILED copy. A tile is 16 weight rows, a group is 64 K = 2 scale blocks of 32.
//     Per (tile t, group g) the code bytes are [half h][row c][block b][step j][4 B] at (t * KG + g) * 512 +
//     lane * 16, lane = 16 h + c: one wave load = 512 contiguous bytes. Dword 2 b + j of the lane holds the 8
//     codes of K 64 g + 32 b + 16 h + 8 j + slot.
//   * scales: bf16 bit patterns [tile][group][row c][block b] = 32 u16 = 64 B per (t, g); the lane of row c reads
//     its two block scales as ONE dword at (t * KG + g) * 64 + c * 4 (low half block 0). A group size G of 32, 64
//     or 128 is a multiple of 32: the packer repeats each scale G / 32 times, ONE kernel serves all of them.
//     Rows past N (last tile): code 0, scale 0 (factor 0).
//   * v_wmma_f32_16x16x16_bf16 as in gemv_mx4: A = x (lane 16 h + i = x row i, rows >= M are zero), B = weights
//     (lane 16 h + c = weight row c); the accumulator C[x row 8 h + e][weight row c] sits in lane 16 h + c, so ONE
//     scale per lane applies to all 8 accumulator registers; per block 2 wmma steps on a fresh zero accumulator,
//     then acc += tmp * scale (the scale applies to the exact K = 32 partial sum).
//   * a wave owns T tiles and a share of the K groups; the W waves of a block meet in LDS at the end; D weight
//     groups are in flight per wave, x (L2 resident) is prefetched one group ahead. bias (bf16 [N], optional) is
//     added to the fp32 sum before the bf16 rounding.
//   Launch config cfg = (T, W, D) from U4_CFGS; cfg -1 picks by shape.
// Guards (thrown by gemv_u4, never a GPU fault): M 1..16, K % 64, SX % 8, SX >= K, SY >= N, 16 B aligned x / q / s /
// y, bias 2 B aligned, tiles >= ceil(N / 16), cfg.
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

#if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32
#error "gemv_u4 is written for wave32 (gfx12)"
#endif

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

// BEGIN host-testable device math (compiled and run on the host by tests/test_w4a16_gemv_host.py)
#define MAX_M 16
#define U4_TILE 16
#define U4_GROUP 64
#define U4_GROUP_BYTES 512
#define U4_GROUP_SCALE_BYTES 64
#define U4_NCFG 12

__device__ __forceinline__ uint16_t f2bf(float v) {  // round to nearest even (finite values)
  const uint32_t u = __float_as_uint(v);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// nibble P of the low 16 bits of w -> the float n - 8, exactly
template <int P>
__device__ __forceinline__ float u4_val(uint32_t w) {
  constexpr float inv = 1.0f / (float)(1u << (4 * P));
  return __builtin_fmaf(__uint_as_float((w & (0xFu << (4 * P))) | 0x4B000000u), inv, -(8388608.0f * inv + 8.0f));
}

// two floats that are small integers -> one dword of two bf16 (their high halves)
__device__ __forceinline__ uint32_t bf16x2(float lo, float hi) {
  return __builtin_amdgcn_perm(__float_as_uint(hi), __float_as_uint(lo), 0x07060302u);
}

// One dword of 8 nibbles -> 4 dwords of bf16 pairs (one wmma operand), exact. Slot s <- nibble 0 2 4 6 1 3 5 7.
__device__ __forceinline__ u32x4 u4_decode(uint32_t w) {
  const uint32_t h = w >> 16;
  return u32x4{bf16x2(u4_val<0>(w), u4_val<2>(w)), bf16x2(u4_val<0>(h), u4_val<2>(h)),
               bf16x2(u4_val<1>(w), u4_val<3>(w)), bf16x2(u4_val<1>(h), u4_val<3>(h))};
}

// the two bf16 block scales of a lane's row, packed in one dword (low half = block 0)
__device__ __forceinline__ float sc_lo(uint32_t p) { return __uint_as_float(p << 16); }
__device__ __forceinline__ float sc_hi(uint32_t p) { return __uint_as_float(p & 0xFFFF0000u); }

// nullptr = the launch is safe, else why not
static const char* u4_check(int64_t M, int64_t N, int64_t K, int64_t SX, int64_t SY, uintptr_t x, uintptr_t q,
                            uintptr_t s, uintptr_t y, uintptr_t bias, int64_t ntiles, int64_t cfg) {
  if (M < 1 || M > MAX_M) return "M must be 1..16";
  if (N < 1 || K < U4_GROUP || K % U4_GROUP || SX % 8 || SX < K || SY < N)
    return "need N >= 1, K % 64 == 0, SX % 8 == 0, SX >= K, SY >= N";
  if (x % 16 || q % 16 || s % 16 || y % 16) return "x, q, s, y must be 16 B aligned";
  if (bias % 2) return "bias must be 2 B aligned";
  if (ntiles < (N + U4_TILE - 1) / U4_TILE) return "fewer tiles than ceil(N / 16)";
  if (cfg < -1 || cfg >= U4_NCFG) return "cfg out of range";
  return nullptr;
}
// END host-testable device math

static const int U4_T[U4_NCFG] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
static const int U4_W[U4_NCFG] = {2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8};
static const int U4_D[U4_NCFG] = {4, 4, 4, 8, 8, 8, 4, 4, 4, 8, 8, 8};  // even: x double buffer parity

template <int T, int W, int D>
__global__ __launch_bounds__(W * 32) void gemv_u4_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ q, const uint8_t* __restrict__ s,
    const uint16_t* __restrict__ bias, void* __restrict__ y, int M, int N, int SX, int SY, int out32, int ngroups,
    int ntiles) {
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
    qp[t] = q + tile * ngroups * U4_GROUP_BYTES + lane * 16;
    sp[t] = s + tile * ngroups * U4_GROUP_SCALE_BYTES + c * 4;
  }
  const bool xlive = c < M;  // rows of the A operand past M are zero
  const uint16_t* xr = x + (int64_t)(xlive ? c : 0) * SX + h * 16;
  const int g0 = (int)((int64_t)ngroups * wv / W);  // this wave's groups [g0, g1)
  const int g1 = (int)((int64_t)ngroups * (wv + 1) / W);
  f32x8 acc[T];
#pragma unroll
  for (int t = 0; t < T; t++) acc[t] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};

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
      b.w[t] = __builtin_nontemporal_load((const u32x4*)(qp[t] + (int64_t)g * U4_GROUP_BYTES));
      b.sc[t] = *(const uint32_t*)(sp[t] + (int64_t)g * U4_GROUP_SCALE_BYTES);
    }
  };
  auto loadx = [&](XBuf& b, int g) {
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
      for (int j = 0; j < 2; j++)
        b.x[k][j] = xlive ? *(const u32x4*)(xr + g * U4_GROUP + k * 32 + j * 8) : u32x4{0, 0, 0, 0};
  };
  auto consume = [&](const WBuf& b, const XBuf& xb) {
#pragma unroll
    for (int t = 0; t < T; t++)
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const u32x4 w0 = u4_decode(k ? b.w[t][2] : b.w[t][0]);
        const u32x4 w1 = u4_decode(k ? b.w[t][3] : b.w[t][1]);
        f32x8 tmp = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
        tmp = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(bf16x8, xb.x[k][0]),
                                                                  __builtin_bit_cast(bf16x8, w0), tmp);
        tmp = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(bf16x8, xb.x[k][1]),
                                                                  __builtin_bit_cast(bf16x8, w1), tmp);
        const float sc = k ? sc_hi(b.sc[t]) : sc_lo(b.sc[t]);
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
    if (bias) v += __uint_as_float((uint32_t)bias[n] << 16);
    if (out32) ((float*)y)[(int64_t)m * SY + n] = v;
    else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(v);
  }
}

struct Args {
  uintptr_t x, q, s, y, bias, stream;
  int M, N, K, SX, SY, out32;
};

template <int T, int W, int D>
static void run_u4(const Args& a) {
  const int ntiles = (a.N + U4_TILE - 1) / U4_TILE;
  hipLaunchKernelGGL((gemv_u4_kernel<T, W, D>), dim3((ntiles + T - 1) / T), dim3(W * 32), (size_t)W * T * 1024,
                     (hipStream_t)a.stream, (const uint16_t*)a.x, (const uint8_t*)a.q, (const uint8_t*)a.s,
                     (const uint16_t*)a.bias, (void*)a.y, a.M, a.N, a.SX, a.SY, a.out32, a.K / U4_GROUP, ntiles);
}

static void run_u4_cfg(int cfg, const Args& a) {
  switch (cfg) {
    case 0: run_u4<1, 2, 4>(a); break;
    case 1: run_u4<1, 4, 4>(a); break;
    case 2: run_u4<1, 8, 4>(a); break;
    case 3: run_u4<1, 2, 8>(a); break;
    case 4: run_u4<1, 4, 8>(a); break;
    case 5: run_u4<1, 8, 8>(a); break;
    case 6: run_u4<2, 2, 4>(a); break;
    case 7: run_u4<2, 4, 4>(a); break;
    case 8: run_u4<2, 8, 4>(a); break;
    case 9: run_u4<2, 2, 8>(a); break;
    case 10: run_u4<2, 4, 8>(a); break;
    case 11: run_u4<2, 8, 8>(a); break;
  }
}

// cfg -1: the picks of gemv_mx4 (same wave / tile structure, 4-bit codes) until sly/w4a16/bench_w4a16_gemv.py --cfgs
// has swept the Qwen shapes on the GPU; the bench prints the winner per shape.
static int pick_cfg_u4(int M, int N, int K) {
  if (N >= (1 << 17)) return 11;
  if (N < 2048) return 2;
  return 1;
}

PYBIND11_MODULE(radiance_w4a16_gemv, m) {
  m.attr("NCFG") = U4_NCFG;
  py::list cfgs;
  for (int i = 0; i < U4_NCFG; i++) cfgs.append(py::make_tuple(U4_T[i], U4_W[i], U4_D[i]));
  m.attr("CFGS") = cfgs;
  m.def("auto_cfg_u4", [](int64_t M, int64_t N, int64_t K) { return pick_cfg_u4((int)M, (int)N, (int)K); },
        "launch config gemv_u4 picks for cfg -1", py::arg("M"), py::arg("N"), py::arg("K"));
  m.def("gemv_u4", [](uintptr_t x, uintptr_t q, uintptr_t s, uintptr_t y, uintptr_t bias, int64_t M, int64_t N,
                      int64_t K, int64_t SX, int64_t SY, int64_t out32, int64_t ntiles, uintptr_t stream,
                      int64_t cfg) {
    if (const char* err = u4_check(M, N, K, SX, SY, x, q, s, y, bias, ntiles, cfg)) throw std::invalid_argument(err);
    if (cfg < 0) cfg = pick_cfg_u4((int)M, (int)N, (int)K);
    run_u4_cfg((int)cfg, Args{x, q, s, y, bias, stream, (int)M, (int)N, (int)K, (int)SX, (int)SY, (int)(out32 != 0)});
  }, "symmetric W4A16 tiled-weight low-M GEMV: y[M,N] = x[M,K] bf16 @ ((n - 8) * scale)[N,K]^T (+ bias), fp32 accumulation",
     py::arg("x"), py::arg("q"), py::arg("s"), py::arg("y"), py::arg("bias"), py::arg("M"), py::arg("N"), py::arg("K"),
     py::arg("SX"), py::arg("SY"), py::arg("out32"), py::arg("ntiles"), py::arg("stream"), py::arg("cfg") = -1);
}
