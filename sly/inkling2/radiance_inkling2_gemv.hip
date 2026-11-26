// RADIANCE gfx1201 low-M GEMV, M 1..16, two entry points in one module:
//   gemv_bf16  y[M,N] = x[M,K] . W[N,K]^T [+ bias], bf16 weights read as they are, fp32 accumulation
//   gemv_q8    int8 (Q8_0) weights on the same matrix unit, below
//
// ===== gemv_bf16 (default route of the dispatcher) =====
// vLLM's Triton bf16 GEMV pads M to 16 rows; its tl.dot tiles stream M=1..16 at 540-600 GB/s on the big
// shapes but only 410-480 on the small ones. This kernel is a weight STREAM that lets the matrix unit do
// all M rows at once (the speed does not depend on M; an ALU dot2 version lost 15-40 % at M 12-16):
//   * y^T tile = W tile x X^T with v_wmma_f32_16x16x16_bf16: A = 16 weight rows x 16 K, B = 16 K x 16
//     columns, column m of B is activation row m (columns >= M are zero).
//   * wmma lane layout (gfx12 wave32): lane = 16*h + r holds row r of A and column r of B, K elements
//     8h..8h+7 of the step (one 16 B load each); the accumulator is C[row 8h+e][col r], e = 0..7: no
//     cross-lane reduction anywhere. K order inside a step is free as long as A and B agree.
//   * loading A straight from global memory costs 16 rows x 32 B per wave instruction and ran at 100-580
//     GB/s depending on the launch shape (partial cache lines). So the weights go through LDS: a wave loads
//     4 rows x 128 B per instruction (whole cache lines, `__builtin_nontemporal_load`: read once), writes the
//     16 x 64 tile to its private LDS slice with chunk' = chunk ^ (row & 7) (conflict free), and reads it
//     back in the wmma A layout. Double buffered in registers: the next 64 K group is in flight while the
//     current one goes through LDS and the matrix unit.
//   * x (B operand) is read straight from L2 (36 KB at M=8, K=2304, shared by every wave).
//   * a wave owns T row tiles (16 rows each) and a share of the K groups; the W waves of a block split K and
//     meet in LDS at the end: no barrier in the stream, small N still has many waves.
//   * rows past N read row N-1 and are never stored; K may be any multiple of 8; no split-K scratch, no
//     allocation: graph-capture safe.
// Launch config cfg = (T, W) from BF16_CFGS; cfg -1 picks by shape.
// Guards (thrown, never a GPU fault): M 1..16, K % 8, SW % 8, SX % 8, 16 B aligned x and w, cfg.
//
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

#define MAX_M 16

__device__ __forceinline__ uint16_t f2bf(float v) {  // round to nearest even (finite values)
  const uint32_t u = __float_as_uint(v);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// ===================================================================== gemv_bf16
#if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32
#error "gemv_bf16 is written for wave32 (gfx12)"
#endif

typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));

#define BF_GROUP 64  // K elements per group = 4 wmma steps = 8 chunks of 16 B per row
#define BF_NCFG 3
static const int BF_T[BF_NCFG] = {1, 1, 1};  // row tiles of 16 per wave
static const int BF_W[BF_NCFG] = {2, 4, 8};  // K-splitting waves per block

// `nt` selects non-temporal (1) or plain weight loads; both measured equal, the host always passes 1. It stays
// a runtime argument because the compiler's schedule of this loop is fragile (see load()): this is the exact
// source form measured at 566-596 GB/s, a branch-free rewrite of the same loop ran at 320-390.
template <int T, int W>
__global__ __launch_bounds__(W * 32) void gemv_bf16_kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const uint16_t* __restrict__ bias,
    void* __restrict__ y, int M, int N, int K, int SX, int SW, int SY, int out32, int ngroups, int nt,
    int64_t BX, int64_t BW, int64_t BY) {
  extern __shared__ u32x4 tileb[];  // [W][T][16 rows][8 chunks]; reused for the cross-wave reduction
  // grid.y = batch (loop controllers: one weight / x / y per stream, element strides BX / BW / BY); 0 / 0 / 0 and
  // grid.y = 1 for the plain call
  const int64_t bz = blockIdx.y;
  x += bz * BX;
  w += bz * BW;
  y = out32 ? (void*)((float*)y + bz * BY) : (void*)((uint16_t*)y + bz * BY);
  const int tid = threadIdx.x;
  const int lane = tid & 31, wv = tid >> 5;
  const int r = lane & 15, h = lane >> 4;
  const int nbase = blockIdx.x * (16 * T);
  const int lr = lane >> 3, lc = lane & 7;  // loader: row 4q + lr of the tile, chunk lc
  const uint16_t* wl[T][4];  // row starts
#pragma unroll
  for (int t = 0; t < T; t++)
#pragma unroll
    for (int q = 0; q < 4; q++)  // rows past N read row N-1, never stored
      wl[t][q] = w + (int64_t)min(nbase + 16 * t + 4 * q + lr, N - 1) * SW;
  const bool xlive = r < M;  // columns of B past M are zero
  const uint16_t* xr0 = x + (int64_t)(xlive ? r : 0) * SX;
  u32x4* tile = tileb + wv * T * 128;
  const int g0 = (int)((int64_t)ngroups * wv / W), g1 = (int)((int64_t)ngroups * (wv + 1) / W);
  f32x8 acc[T];
#pragma unroll
  for (int t = 0; t < T; t++) acc[t] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  // Predicated loads: a lane past K loads nothing (never reads outside the row). Do not "fix" this into
  // unconditional clamped loads plus selects: the compiler schedules the next group's loads behind the
  // LDS stores of the current one and the stream loses 30-40 % (measured, also with sched_barrier).
  auto load = [&](u32x4 (&wr)[T][4], u32x4 (&xr)[4], int g) {
#pragma unroll
    for (int t = 0; t < T; t++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const bool ok = g * BF_GROUP + lc * 8 < K;
        const uint16_t* p = wl[t][q] + g * BF_GROUP + lc * 8;
        wr[t][q] = !ok ? (u32x4){0, 0, 0, 0}
                       : nt ? __builtin_nontemporal_load((const u32x4*)p) : *(const u32x4*)p;
      }
#pragma unroll
    for (int j = 0; j < 4; j++)
      xr[j] = (xlive && g * BF_GROUP + j * 16 + h * 8 < K)
                  ? *(const u32x4*)(xr0 + g * BF_GROUP + j * 16 + h * 8) : (u32x4){0, 0, 0, 0};
  };
  auto consume = [&](u32x4 (&wr)[T][4], u32x4 (&xr)[4], int g) {
    const bool live = g < g1;
    const int k0 = g * BF_GROUP;
#pragma unroll
    for (int t = 0; t < T; t++)
#pragma unroll
      for (int q = 0; q < 4; q++) {
        const int row = 4 * q + lr;
        tile[(t * 16 + row) * 8 + (lc ^ (row & 7))] = live && k0 + lc * 8 < K ? wr[t][q] : (u32x4){0, 0, 0, 0};
      }
    __builtin_amdgcn_wave_barrier();
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const u32x4 xv = live && xlive && k0 + j * 16 + h * 8 < K ? xr[j] : (u32x4){0, 0, 0, 0};
#pragma unroll
      for (int t = 0; t < T; t++) {
        const u32x4 a = tile[(t * 16 + r) * 8 + ((2 * j + h) ^ (r & 7))];
        acc[t] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(bf16x8, a),
                                                                  __builtin_bit_cast(bf16x8, xv), acc[t]);
      }
    }
    __builtin_amdgcn_wave_barrier();
  };
  u32x4 wa[T][4], xa[4], wb[T][4], xb[4];
  if (g0 < g1) load(wa, xa, g0);
  for (int g = g0; g < g1; g += 2) {
    if (g + 1 < g1) load(wb, xb, g + 1);
    consume(wa, xa, g);
    if (g + 1 < g1) {
      if (g + 2 < g1) load(wa, xa, g + 2);
      consume(wb, xb, g + 1);
    }
  }
  __syncthreads();  // every wave is done with its tile before the reduction reuses the LDS
  float* redb = (float*)tileb;
#pragma unroll
  for (int t = 0; t < T; t++) *(f32x8*)(redb + ((wv * T + t) * 32 + lane) * 8) = acc[t];
  __syncthreads();
  for (int id = tid; id < T * 256; id += W * 32) {  // id = (tile t, lane l, e): C[row 8*(l/16)+e][col l%16]
    float v = 0.f;
#pragma unroll
    for (int g = 0; g < W; g++) v += redb[g * T * 256 + id];
    const int t = id >> 8, l = (id >> 3) & 31, e = id & 7;
    const int m = l & 15, n = nbase + 16 * t + (l >> 4) * 8 + e;
    if (m >= M || n >= N) continue;
    v += bias ? __uint_as_float((uint32_t)bias[n] << 16) : 0.f;
    if (out32) ((float*)y)[(int64_t)m * SY + n] = v;
    else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(v);
  }
}

struct ArgsB {
  uintptr_t x, w, bias, y, stream;
  int M, N, K, SX, SW, SY, out32;
  int nb = 1;
  int64_t BX = 0, BW = 0, BY = 0;
};

template <int T, int W>
static void run_bf16(const ArgsB& a) {
  const int ngroups = (a.K + BF_GROUP - 1) / BF_GROUP;
  const int blocks = (a.N + 16 * T - 1) / (16 * T);
  hipLaunchKernelGGL((gemv_bf16_kernel<T, W>), dim3(blocks, a.nb), dim3(W * 32), (size_t)W * T * 2048,
                     (hipStream_t)a.stream, (const uint16_t*)a.x, (const uint16_t*)a.w,
                     (const uint16_t*)a.bias, (void*)a.y, a.M, a.N, a.K, a.SX, a.SW, a.SY, a.out32, ngroups, 1,
                     a.BX, a.BW, a.BY);
}

static void run_bf16_cfg(int cfg, const ArgsB& a) {
  switch (cfg) {
    case 0: run_bf16<1, 2>(a); break;
    case 1: run_bf16<1, 4>(a); break;
    case 2: run_bf16<1, 8>(a); break;
  }
}

// cfg -1, from the rotated-pool sweep at M=8 (bench_gemv.py --cfgs, GB/s, W waves per 16-row tile):
//   N 128..4096 (k/v, kv_ctx, rnn_proj, wo, down) and N >= 16384 (gate_up, unembed): W=8
//   N 5120..8600 (qkv, qkvr): W=2 (8600x2304: 581 vs 577 / 576; 5120x2304: 558 vs 555 / 546)
static int pick_cfg_bf16(int N, int K) { return (N < 4096 || N >= 16384) ? 2 : 0; }

// ===== gemv_q8 =====
//   y[M,N] = sum over 32-value blocks b of  scale[n,b] * (x[m, 32b..32b+31] . q[n, 32b..32b+31])  [+ bias]
//   q int8 [N,K], scale fp32 [N,K/32] (symmetric, amax/127, round to nearest) = ggml Q8_0 with an fp32 scale,
//   1.125 B/weight; x stays bf16 (no activation quantization), fp32 accumulation, bf16 or fp32 output, x row
//   stride supported. Graph-capture safe: no allocation, no scratch, no sync, caller's stream.
//
// Why: the bf16 stream is at DRAM speed (~580 GB/s), so the lever is fewer bytes. An ALU kernel (int8 -> f32,
// one FMA per row of x and weight) loses beyond M = 4: the FMA count grows with M. This one does the K sum on the
// matrix unit like gemv_bf16, so its speed does not depend on M (M 1..16).
//   * the weights are read from a TILED copy (made once by radiance_inkling2_gemv_hip.pack_q8). A tile is 16 weight
//     rows; a group is 64 K = 2 scale blocks. Per (tile t, group g) the int8 bytes are [block b][half h][row c][16 B]
//     at ((t * KG + g) * 1024 + b * 512 + (16 h + c) * 16): lane 16 h + c of a wave reads exactly its 16 B, so one
//     wave load = 512 contiguous bytes (4 whole cache lines), no LDS staging, no partial lines. The fp32 scales are
//     [row c][block b] at (t * KG + g) * 32 floats + c * 2 (lane reads its row's two scales: 8 B).
//   * v_wmma_f32_16x16x16_bf16 with the roles swapped w.r.t. gemv_bf16: A = x (lane 16 h + i = x row i, columns >= M
//     are zero), B = weights (lane 16 h + c = weight row c). The accumulator C[x row 8 h + e][weight row c] is
//     then in lane 16 h + c: ONE scale per lane applies to all 8 accumulator registers.
//   * the K order inside a wmma step is free if A and B agree: lane (h, .) holds for block b the 16 consecutive K
//     32 b + 16 h ..+15; step 0 takes the first 8 of them, step 1 the next 8, for x and for the weights alike.
//   * int8 -> bf16 is exact: with u = q ^ 0x80 (unsigned q + 128) the float 0x4B000000 | u = 2^23 + u, minus
//     (2^23 + 128) = q; an integer of magnitude <= 128 has a zero low half, so the high half of the fp32 is its bf16.
//   * per block: 2 wmma (K = 32, fresh zero accumulator) then acc += tmp * scale: the scale is applied to an exact
//     K = 32 partial sum, like the reference's dequantize-then-dot up to fp32 summation order.
//   * a wave owns T tiles and a share of the 64 K groups; the W waves of a block split K and meet in LDS at the
//     end. D groups of loads are in flight per wave (software pipeline over D register buffers).
// Launch config cfg = (T, W, D) from Q8_CFGS; cfg -1 picks by shape.
// Guards (thrown, never a GPU fault): M 1..16, K % 64, SX % 8, 16 B aligned x / q / s, tiles >= ceil(N/16), cfg.

#define Q_TILE 16
#define Q_GROUP 64
#define Q_GROUP_BYTES 1024
#define Q_GROUP_SCALES 32
#define Q_NCFG 12
static const int Q_T[Q_NCFG] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
static const int Q_W[Q_NCFG] = {2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8};
static const int Q_D[Q_NCFG] = {2, 2, 2, 4, 4, 4, 2, 2, 2, 4, 4, 4};

typedef float f32x2 __attribute__((ext_vector_type(2)));

// 4 int8 (one dword) -> 4 bf16 (two dwords, element 0 in the low half of `lo`), exact.
__device__ __forceinline__ void i8x4_to_bf16x4(uint32_t w, uint32_t& lo, uint32_t& hi) {
  const uint32_t u = w ^ 0x80808080u;  // q + 128 per byte
  const float f0 = __uint_as_float(0x4B000000u | (u & 0xFFu)) - 8388736.f;  // 2^23 + 128
  const float f1 = __uint_as_float(0x4B000000u | ((u >> 8) & 0xFFu)) - 8388736.f;
  const float f2 = __uint_as_float(0x4B000000u | ((u >> 16) & 0xFFu)) - 8388736.f;
  const float f3 = __uint_as_float(0x4B000000u | (u >> 24)) - 8388736.f;
  lo = (__float_as_uint(f0) >> 16) | (__float_as_uint(f1) & 0xFFFF0000u);
  hi = (__float_as_uint(f2) >> 16) | (__float_as_uint(f3) & 0xFFFF0000u);
}

template <int T, int W, int D>
__global__ __launch_bounds__(W * 32) void gemv_q8_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ q, const float* __restrict__ s,
    const uint16_t* __restrict__ bias, void* __restrict__ y, int M, int N, int SX, int SY, int out32,
    int ngroups, int ntiles) {
  extern __shared__ f32x8 redq[];  // [W][T][32 lanes] accumulators for the cross-wave sum
  const int tid = threadIdx.x;
  const int lane = tid & 31, wv = tid >> 5;
  const int c = lane & 15, h = lane >> 4;
  const int tile0 = blockIdx.x * T;
  const uint8_t* qp[T];
  const float* sp[T];
#pragma unroll
  for (int t = 0; t < T; t++) {  // tiles past the last one read the last tile and are never stored
    const int64_t tile = min(tile0 + t, ntiles - 1);
    qp[t] = q + tile * ngroups * Q_GROUP_BYTES + lane * 16;
    sp[t] = s + tile * ngroups * Q_GROUP_SCALES + c * 2;
  }
  const bool xlive = c < M;  // rows of the A operand past M are zero
  const uint16_t* xr = x + (int64_t)(xlive ? c : 0) * SX + h * 16;
  const int g0 = (int)((int64_t)ngroups * wv / W);  // this wave's groups [g0, g1)
  const int g1 = (int)((int64_t)ngroups * (wv + 1) / W);
  f32x8 acc[T];
#pragma unroll
  for (int t = 0; t < T; t++) acc[t] = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};

  struct Buf {
    u32x4 w[T][2];
    f32x2 sc[T];
    u32x4 x[2][2];
  };
  auto load = [&](Buf& b, int g) {
#pragma unroll
    for (int t = 0; t < T; t++) {
#pragma unroll
      for (int k = 0; k < 2; k++)
        b.w[t][k] = __builtin_nontemporal_load((const u32x4*)(qp[t] + (int64_t)g * Q_GROUP_BYTES + k * 512));
      b.sc[t] = __builtin_nontemporal_load((const f32x2*)(sp[t] + (int64_t)g * Q_GROUP_SCALES));
    }
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
      for (int j = 0; j < 2; j++)
        b.x[k][j] = xlive ? *(const u32x4*)(xr + g * Q_GROUP + k * 32 + j * 8) : (u32x4){0, 0, 0, 0};
  };
  auto consume = [&](const Buf& b) {
#pragma unroll
    for (int t = 0; t < T; t++)
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const u32x4 v = b.w[t][k];
        uint32_t p0, p1, p2, p3, p4, p5, p6, p7;
        i8x4_to_bf16x4(v.x, p0, p1);
        i8x4_to_bf16x4(v.y, p2, p3);
        i8x4_to_bf16x4(v.z, p4, p5);
        i8x4_to_bf16x4(v.w, p6, p7);
        const u32x4 w0 = {p0, p1, p2, p3}, w1 = {p4, p5, p6, p7};
        f32x8 tmp = (f32x8){0, 0, 0, 0, 0, 0, 0, 0};
        tmp = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(bf16x8, b.x[k][0]),
                                                                  __builtin_bit_cast(bf16x8, w0), tmp);
        tmp = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(__builtin_bit_cast(bf16x8, b.x[k][1]),
                                                                  __builtin_bit_cast(bf16x8, w1), tmp);
        const float sc = k ? b.sc[t].y : b.sc[t].x;
#pragma unroll
        for (int e = 0; e < 8; e++) acc[t][e] = __builtin_fmaf(tmp[e], sc, acc[t][e]);
      }
  };
  Buf buf[D];
#pragma unroll
  for (int d = 0; d < D - 1; d++)
    if (g0 + d < g1) load(buf[d], g0 + d);
  for (int g = g0; g < g1; g += D) {
#pragma unroll
    for (int d = 0; d < D; d++) {
      if (g + d + D - 1 < g1) load(buf[(d + D - 1) % D], g + d + D - 1);
      if (g + d < g1) consume(buf[d]);
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

struct ArgsQ {
  uintptr_t x, q, s, bias, y, stream;
  int M, N, K, SX, SY, out32;
};

template <int T, int W, int D>
static void run_q8(const ArgsQ& a) {
  const int ntiles = (a.N + Q_TILE - 1) / Q_TILE;
  hipLaunchKernelGGL((gemv_q8_kernel<T, W, D>), dim3((ntiles + T - 1) / T), dim3(W * 32), (size_t)W * T * 1024,
                     (hipStream_t)a.stream, (const uint16_t*)a.x, (const uint8_t*)a.q, (const float*)a.s,
                     (const uint16_t*)a.bias, (void*)a.y, a.M, a.N, a.SX, a.SY, a.out32, a.K / Q_GROUP, ntiles);
}

static void run_q8_cfg(int cfg, const ArgsQ& a) {
  switch (cfg) {
    case 0: run_q8<1, 2, 2>(a); break;
    case 1: run_q8<1, 4, 2>(a); break;
    case 2: run_q8<1, 8, 2>(a); break;
    case 3: run_q8<1, 2, 4>(a); break;
    case 4: run_q8<1, 4, 4>(a); break;
    case 5: run_q8<1, 8, 4>(a); break;
    case 6: run_q8<2, 2, 2>(a); break;
    case 7: run_q8<2, 4, 2>(a); break;
    case 8: run_q8<2, 8, 2>(a); break;
    case 9: run_q8<2, 2, 4>(a); break;
    case 10: run_q8<2, 4, 4>(a); break;
    case 11: run_q8<2, 8, 4>(a); break;
  }
}

// cfg -1, from the rotated-pool sweep at M = 1, 8, 16 (bench_gemv.py --q8 --cfgs; GB/s of int8 + scale bytes at M=8):
//   N 6552, 8600 (qkv, qkvr): cfg 0 (W=2) 540 / 559 vs 503 / 560 for cfg 1; the same order at M=1 and 16
//   N >= 131072 (unembed): cfg 6 (T=2 W=2) 570 vs 540 for cfg 1 (M=1: 617 vs 584, M=16: 552 vs 521)
//   everything else (k/v, kv_ctx, wo, down, DSpark qkv, gate_up): cfg 1 (T=1 W=4 D=2), within ~3 % of the best cfg
// For N >= 5120 every cfg is within ~7 % of the best: the stream, not the launch shape, sets the speed there.
static int pick_cfg_q8(int M, int N, int K) {
  if (N >= (1 << 17)) return 6;
  if (N >= 6000 && N < 9000) return 0;
  return 1;
}

PYBIND11_MODULE(radiance_inkling2_gemv, m) {
  m.attr("BF16_NCFG") = BF_NCFG;
  py::list bcfgs;
  for (int i = 0; i < BF_NCFG; i++) bcfgs.append(py::make_tuple(BF_T[i], BF_W[i]));
  m.attr("BF16_CFGS") = bcfgs;
  m.def("gemv_bf16", [](uintptr_t x, uintptr_t w, uintptr_t bias, uintptr_t y, int64_t M, int64_t N,
                        int64_t K, int64_t SX, int64_t SW, int64_t SY, int64_t out32, uintptr_t stream,
                        int64_t cfg) {
    if (M < 1 || M > MAX_M) throw std::invalid_argument("M must be 1..16");
    if (N < 1 || K < 8 || K % 8 || SW % 8 || SW < K || SX % 8 || SX < K)
      throw std::invalid_argument("need N >= 1, K % 8 == 0, SW % 8 == 0, SW >= K, SX % 8 == 0, SX >= K");
    if (x % 16 || w % 16) throw std::invalid_argument("x and w must be 16 B aligned");
    if (cfg < -1 || cfg >= BF_NCFG) throw std::invalid_argument("cfg out of range");
    if (cfg < 0) cfg = pick_cfg_bf16((int)N, (int)K);
    run_bf16_cfg((int)cfg, ArgsB{x, w, bias, y, stream, (int)M, (int)N, (int)K, (int)SX, (int)SW, (int)SY,
                                 (int)(out32 != 0)});
  }, "bf16 weight low-M GEMV: y[M,N] = x[M,K] bf16 @ W[N,K]^T bf16 [+ bias], fp32 accumulation",
     py::arg("x"), py::arg("w"), py::arg("bias"), py::arg("y"), py::arg("M"), py::arg("N"),
     py::arg("K"), py::arg("SX"), py::arg("SW"), py::arg("SY"), py::arg("out32"), py::arg("stream"),
     py::arg("cfg") = -1);
  m.def("gemv_bf16_batched", [](uintptr_t x, uintptr_t w, uintptr_t y, int64_t nb, int64_t M, int64_t N,
                                int64_t K, int64_t SX, int64_t SW, int64_t SY, int64_t BX, int64_t BW,
                                int64_t BY, int64_t out32, uintptr_t stream, int64_t cfg) {
    if (nb < 1 || nb > 64) throw std::invalid_argument("nb must be 1..64");
    if (M < 1 || M > MAX_M) throw std::invalid_argument("M must be 1..16");
    if (N < 1 || K < 8 || K % 8 || SW % 8 || SW < K || SX % 8 || SX < K || SY < N)
      throw std::invalid_argument("need N >= 1, K % 8 == 0, SW % 8 == 0, SW >= K, SX % 8 == 0, SX >= K, SY >= N");
    if (BX < 0 || BW < 0 || BY < 0 || BX % 8 || BW % 8)
      throw std::invalid_argument("batch strides must be >= 0, BX % 8 == 0, BW % 8 == 0");
    if (x % 16 || w % 16) throw std::invalid_argument("x and w must be 16 B aligned");
    if (cfg < -1 || cfg >= BF_NCFG) throw std::invalid_argument("cfg out of range");
    if (cfg < 0) cfg = pick_cfg_bf16((int)N, (int)K);
    run_bf16_cfg((int)cfg, ArgsB{x, w, 0, y, stream, (int)M, (int)N, (int)K, (int)SX, (int)SW, (int)SY,
                                 (int)(out32 != 0), (int)nb, BX, BW, BY});
  }, "gemv_bf16 over nb independent (x, W, y) triples at element strides BX / BW / BY (grid.y), no bias",
     py::arg("x"), py::arg("w"), py::arg("y"), py::arg("nb"), py::arg("M"), py::arg("N"), py::arg("K"),
     py::arg("SX"), py::arg("SW"), py::arg("SY"), py::arg("BX"), py::arg("BW"), py::arg("BY"),
     py::arg("out32"), py::arg("stream"), py::arg("cfg") = -1);
  m.def("auto_cfg_q8", [](int64_t M, int64_t N, int64_t K) { return pick_cfg_q8((int)M, (int)N, (int)K); },
        "launch config gemv_q8 picks for cfg -1", py::arg("M"), py::arg("N"), py::arg("K"));
  m.attr("NCFG") = Q_NCFG;
  py::list cfgs;
  for (int i = 0; i < Q_NCFG; i++) cfgs.append(py::make_tuple(Q_T[i], Q_W[i], Q_D[i]));
  m.attr("CFGS") = cfgs;
  m.def("gemv_q8", [](uintptr_t x, uintptr_t q, uintptr_t s, uintptr_t bias, uintptr_t y, int64_t M,
                      int64_t N, int64_t K, int64_t SX, int64_t SY, int64_t out32, int64_t ntiles,
                      uintptr_t stream, int64_t cfg) {
    if (M < 1 || M > MAX_M) throw std::invalid_argument("M must be 1..16");
    if (N < 1 || K < Q_GROUP || K % Q_GROUP || SX % 8 || SX < K)
      throw std::invalid_argument("need N >= 1, K % 64 == 0, SX % 8 == 0, SX >= K");
    if (x % 16 || q % 16 || s % 16) throw std::invalid_argument("x, q, s must be 16 B aligned");
    if (ntiles < (N + Q_TILE - 1) / Q_TILE) throw std::invalid_argument("fewer tiles than ceil(N / 16)");
    if (cfg < -1 || cfg >= Q_NCFG) throw std::invalid_argument("cfg out of range");
    if (cfg < 0) cfg = pick_cfg_q8((int)M, (int)N, (int)K);
    run_q8_cfg((int)cfg, ArgsQ{x, q, s, bias, y, stream, (int)M, (int)N, (int)K, (int)SX, (int)SY,
                               (int)(out32 != 0)});
  }, "int8 (Q8_0, fp32 scale per 32 K) tiled-weight low-M GEMV: y[M,N] = x[M,K] bf16 @ deq(Wq)[N,K]^T [+ bias]",
     py::arg("x"), py::arg("q"), py::arg("s"), py::arg("bias"), py::arg("y"), py::arg("M"),
     py::arg("N"), py::arg("K"), py::arg("SX"), py::arg("SY"), py::arg("out32"), py::arg("ntiles"),
     py::arg("stream"), py::arg("cfg") = -1);
}
