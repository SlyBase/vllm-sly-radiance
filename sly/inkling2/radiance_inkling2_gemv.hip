// RADIANCE gfx1201 low-M GEMV, M 1..16, two entry points in one module:
//   gemv_bf16  y[M,N] = x[M,K] . W[N,K]^T [+ bias], bf16 weights read as they are, fp32 accumulation
//   gemv_q8    int8 (Q8_0) weights, below
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
// ===== gemv_q8 =====
//   y[M,N] = sum over 32-value blocks b of  scale[n,b] * (x[m, 32b..32b+31] . Wq[n, 32b..32b+31])  [+ bias]
//   Wq int8 [N,K], scale fp32 [N,K/32] (symmetric, amax/127, round to nearest) = ggml Q8_0 with an
//   fp32 scale, 1.125 B/weight; x stays bf16 -> fp32 (no activation quantization), fp32
//   accumulation, bf16 or fp32 output, x row stride supported. Graph-capture safe: no allocation,
//   no sync, launched on the caller's stream; split-K partials live in caller-owned scratch.
//
// Why quantized weights: vLLM's Triton bf16 GEMV already streams at DRAM bandwidth (~590 GB/s of
// 640), so the only lever is fewer bytes, and the kernel must stay bandwidth bound with the
// dequant on the critical path:
//   * an 8-lane group owns a weight row: each lane loads 32 B (2 x 16 B = one scale block of 32
//     int8) + its fp32 scale, the 8 lanes read 256 contiguous bytes (SLICE) per iteration; all K
//     of a row in K/256 iterations (2304, 3072, 9216 are exact multiples; the rest is masked).
//   * R rows per group share every x read; U iterations of loads are issued before their math.
//   * per block: 32 int8 -> f32 FMAs against x, then one FMA with the block scale.
//   * x (bf16) is staged once per block into LDS (zero padded, <= 40 KB, K chunked when M*K*2 does
//     not fit; chunks swizzled so a lane's 4 x 16 B reads are bank-conflict free), converted
//     bf16 -> f32 with a shift/and per use.
//   * reduction across the 8 lanes with DPP, fused into v_add_f32.
//   * split-K only where N is too small to give every CU two blocks: fp32 partials in scratch,
//     folded (bias, cast) by gemv_reduce.
// Launch config cfg = (R, U) from CFGS; cfg -1 picks by shape. M is a template parameter so the
// m-loops unroll.
//
// Guards (thrown, never a GPU fault): K % 32, SW % 16, SX % 8, aligned x / w / scale, M 1..16.
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;

#define BLOCK 128
#define SLICE 256          // K elements per iteration of one 8-lane group
#define LDS_BUDGET 40960   // bytes of x staged per block
#define MAX_M 16
#define NCFG 8

static const int CFG_R[NCFG] = {1, 1, 2, 2, 2, 4, 4, 4};
static const int CFG_U[NCFG] = {3, 9, 1, 3, 2, 1, 2, 3};

// Workgroup barrier that orders LDS only: __syncthreads() would also wait for the weight loads
// that are deliberately still in flight.
#define LDS_BARRIER()                                          \
  do {                                                         \
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local"); \
    __builtin_amdgcn_s_barrier();                              \
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local"); \
  } while (0)

template <int CTRL>
__device__ __forceinline__ float dpp(float v) {
  return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), CTRL, 0xF, 0xF, false));
}

// Sum over the 8 lanes of an aligned lane group; every lane ends with the total.
__device__ __forceinline__ float sum8(float v) {
  v += dpp<0xB1>(v);   // quad_perm [1,0,3,2]
  v += dpp<0x4E>(v);   // quad_perm [2,3,0,1]
  v += dpp<0x164>(v);  // row_xmask:4
  return v;
}

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xFFFF0000u); }
__device__ __forceinline__ float i8(uint32_t w, int byte) { return (float)(int)(int8_t)(w >> (8 * byte)); }
__device__ __forceinline__ uint16_t f2bf(float v) {  // round to nearest even (finite values)
  const uint32_t u = __float_as_uint(v);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

template <int M, int R, int U>
__global__ __launch_bounds__(BLOCK) void gemv_q8_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ w, const float* __restrict__ scale,
    const uint16_t* __restrict__ bias, void* __restrict__ y, float* __restrict__ partials,
    int N, int K, int SX, int SW, int SY, int out32, int nit, int itc, int nsplit) {
  extern __shared__ uint4 xs[];  // [M][itc * 32] groups of 8 bf16, swizzled per 256-slice
  const int tid = threadIdx.x;
  const int l8 = tid & 7;
  const int split = blockIdx.y;
  const int i0 = (int)(((int64_t)nit * split) / nsplit);
  const int i1 = (int)(((int64_t)nit * (split + 1)) / nsplit);
  const int n0 = blockIdx.x * (16 * R) + (tid >> 3) * R;
  const uint8_t* wr[R];
  const float* sr[R];
#pragma unroll
  for (int j = 0; j < R; j++) {
    const int n = min(n0 + j, N - 1);  // rows past N read row N-1 and are never stored
    wr[j] = w + (int64_t)n * SW + l8 * 32;
    sr[j] = scale + (int64_t)n * (K / 32) + l8;
  }
  float acc[R][M];
#pragma unroll
  for (int j = 0; j < R; j++)
#pragma unroll
    for (int m = 0; m < M; m++) acc[j][m] = 0.f;

  const int kc8 = itc * 32;
  uint4 wa[R][U], wb[R][U];
  float sc[R][U];
  // issue the weight loads of the batch starting at iteration `it`
  auto load = [&](int it, int ce) {
#pragma unroll
    for (int u = 0; u < U; u++) {
      const int itu = it + u;
      const bool ok = itu < ce && itu * SLICE + l8 * 32 < K;
#pragma unroll
      for (int j = 0; j < R; j++) {
        const uint4* p = (const uint4*)(wr[j] + (int64_t)itu * SLICE);
        wa[j][u] = ok ? p[0] : make_uint4(0, 0, 0, 0);
        wb[j][u] = ok ? p[1] : make_uint4(0, 0, 0, 0);
        sc[j][u] = ok ? sr[j][itu * 8] : 0.f;
      }
    }
  };
  for (int c = i0; c < i1; c += itc) {
    const int ce = min(c + itc, i1);
    const int g8 = (ce - c) * 32;
    LDS_BARRIER();  // the previous chunk's LDS reads are done
    load(c, ce);    // first weights fly while x is staged (loads complete in order; x comes from L2)
    for (int idx = tid; idx < M * g8; idx += BLOCK) {
      const int m = idx / g8, g = idx - m * g8;
      const int64_t k = (int64_t)c * SLICE + g * 8;
      uint4 v = make_uint4(0, 0, 0, 0);
      if (k < K) v = *(const uint4*)(x + (int64_t)m * SX + k);
      const int cc = g & 31;  // chunk in the slice -> (lane = cc / 4, j = cc % 4) -> j * 8 + lane
      xs[m * kc8 + ((g & ~31) | ((cc & 3) << 3) | (cc >> 2))] = v;
    }
    LDS_BARRIER();
    for (int it = c; it < ce; it += U) {
      if (it > c) load(it, ce);
#pragma unroll
      for (int u = 0; u < U; u++) {
        if (it + u >= ce) break;
        const int xb = (it + u - c) * 32 + l8;  // LDS chunk of this lane, j = 0; +8 per j
        float s[R][M];
#pragma unroll
        for (int j = 0; j < R; j++)
#pragma unroll
          for (int m = 0; m < M; m++) s[j][m] = 0.f;
#pragma unroll
        for (int h = 0; h < 2; h++) {  // two halves of the 32-value block
          float wf[R][16];
#pragma unroll
          for (int j = 0; j < R; j++) {
            const uint4 v = h ? wb[j][u] : wa[j][u];
            const uint32_t q[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
              for (int b = 0; b < 4; b++) wf[j][4 * i + b] = i8(q[i], b);
          }
#pragma unroll
          for (int m = 0; m < M; m++) {
            const uint4 xa = xs[m * kc8 + xb + (2 * h) * 8], xc = xs[m * kc8 + xb + (2 * h + 1) * 8];
            const uint32_t q[8] = {xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w};
            float xf[16];
#pragma unroll
            for (int i = 0; i < 8; i++) {
              xf[2 * i] = bf_lo(q[i]);
              xf[2 * i + 1] = bf_hi(q[i]);
            }
#pragma unroll
            for (int j = 0; j < R; j++)
#pragma unroll
              for (int e = 0; e < 16; e++) s[j][m] += wf[j][e] * xf[e];
          }
        }
#pragma unroll
        for (int j = 0; j < R; j++)
#pragma unroll
          for (int m = 0; m < M; m++) acc[j][m] += s[j][m] * sc[j][u];
      }
    }
  }
  float tot[R][M];
#pragma unroll
  for (int j = 0; j < R; j++)
#pragma unroll
    for (int m = 0; m < M; m++) tot[j][m] = sum8(acc[j][m]);
  if (l8 != 0) return;
#pragma unroll
  for (int j = 0; j < R; j++) {
    const int n = n0 + j;
    if (n >= N) continue;
#pragma unroll
    for (int m = 0; m < M; m++) {
      if (partials) {
        partials[((int64_t)split * M + m) * N + n] = tot[j][m];
      } else {
        const float v = tot[j][m] + (bias ? __uint_as_float((uint32_t)bias[n] << 16) : 0.f);
        if (out32) ((float*)y)[(int64_t)m * SY + n] = v;
        else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(v);
      }
    }
  }
}

// Fold split-K partials (fp32 [nsplit, M, N]) into y: bias + cast.
__global__ void gemv_reduce(const float* __restrict__ partials, const uint16_t* __restrict__ bias,
                            void* __restrict__ y, int M, int N, int SY, int nsplit, int out32) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t total = (int64_t)M * N;
  if (idx >= total) return;
  const int n = (int)(idx % N), m = (int)(idx / N);
  float s = 0.f;
  for (int g = 0; g < nsplit; g++) s += partials[g * total + idx];
  s += bias ? __uint_as_float((uint32_t)bias[n] << 16) : 0.f;
  if (out32) ((float*)y)[(int64_t)m * SY + n] = s;
  else ((uint16_t*)y)[(int64_t)m * SY + n] = f2bf(s);
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
    void* __restrict__ y, int M, int N, int K, int SX, int SW, int SY, int out32, int ngroups, int nt) {
  extern __shared__ u32x4 tileb[];  // [W][T][16 rows][8 chunks]; reused for the cross-wave reduction
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
};

template <int T, int W>
static void run_bf16(const ArgsB& a) {
  const int ngroups = (a.K + BF_GROUP - 1) / BF_GROUP;
  const int blocks = (a.N + 16 * T - 1) / (16 * T);
  hipLaunchKernelGGL((gemv_bf16_kernel<T, W>), dim3(blocks), dim3(W * 32), (size_t)W * T * 2048,
                     (hipStream_t)a.stream, (const uint16_t*)a.x, (const uint16_t*)a.w,
                     (const uint16_t*)a.bias, (void*)a.y, a.M, a.N, a.K, a.SX, a.SW, a.SY, a.out32, ngroups, 1);
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

// ===================================================================== gemv_q8
struct Args {
  uintptr_t x, w, scale, bias, y, scratch, scratch_bytes, stream;
  int M, N, K, SX, SW, SY, out32;
};

template <int M, int R, int U>
static void run(const Args& a) {
  const int nit = (a.K + SLICE - 1) / SLICE;
  const int blocks = (a.N + 16 * R - 1) / (16 * R);
  int nsplit = 1;
  if (blocks < 128) {  // too few blocks to give each of the 64 CUs two: split K
    nsplit = (128 + blocks - 1) / blocks;
    if (nsplit > nit) nsplit = nit;
    if (nsplit > 16) nsplit = 16;
    if (a.scratch == 0 || (int64_t)nsplit * M * a.N * 4 > (int64_t)a.scratch_bytes) nsplit = 1;
  }
  int itc = (nit + nsplit - 1) / nsplit;  // iterations of the longest split
  if (itc > (LDS_BUDGET / 512) / M) itc = (LDS_BUDGET / 512) / M;
  const size_t lds = (size_t)M * itc * 512;
  float* partials = nsplit > 1 ? (float*)a.scratch : nullptr;
  hipLaunchKernelGGL((gemv_q8_kernel<M, R, U>), dim3(blocks, nsplit), dim3(BLOCK), lds,
                     (hipStream_t)a.stream, (const uint16_t*)a.x, (const uint8_t*)a.w,
                     (const float*)a.scale, (const uint16_t*)a.bias, (void*)a.y, partials, a.N,
                     a.K, a.SX, a.SW, a.SY, a.out32, nit, itc, nsplit);
  if (nsplit > 1) {
    const int64_t total = (int64_t)M * a.N;
    hipLaunchKernelGGL(gemv_reduce, dim3((unsigned)((total + BLOCK - 1) / BLOCK)), dim3(BLOCK), 0,
                       (hipStream_t)a.stream, (const float*)a.scratch, (const uint16_t*)a.bias,
                       (void*)a.y, a.M, a.N, a.SY, nsplit, a.out32);
  }
}

template <int M>
static void run_m(int cfg, const Args& a) {
  switch (cfg) {
    case 0: run<M, 1, 3>(a); break;
    case 1: run<M, 1, 9>(a); break;
    case 2: run<M, 2, 1>(a); break;
    case 3: run<M, 2, 3>(a); break;
    case 4: run<M, 2, 2>(a); break;
    case 5: run<M, 4, 1>(a); break;
    case 6: run<M, 4, 2>(a); break;
    case 7: run<M, 4, 3>(a); break;
  }
}

// cfg -1: by M (and N for the smallest shapes), from the rotated-weight sweep in bench_gemv.py --cfgs:
// R=1 for M 1-2 (latency bound at small N), R=2 for M 3-4, R=4 above.
static int pick_cfg(int M, int N) {
  if (M == 1 || (M == 2 && N < 4096)) return 0;  // R=1 U=3
  if (M <= 4) return 2;                          // R=2 U=1
  return 5;                                      // R=4 U=1
}

static void launch(const Args& a, int cfg) {
  if (cfg < 0) cfg = pick_cfg(a.M, a.N);
  switch (a.M) {
#define CASE(m) case m: run_m<m>(cfg, a); break;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
    CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15) CASE(16)
#undef CASE
  }
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
  m.attr("NCFG") = NCFG;
  py::list cfgs;
  for (int i = 0; i < NCFG; i++) cfgs.append(py::make_tuple(CFG_R[i], CFG_U[i]));
  m.attr("CFGS") = cfgs;
  m.def("gemv_q8", [](uintptr_t x, uintptr_t w, uintptr_t scale, uintptr_t bias, uintptr_t y,
                      int64_t M, int64_t N, int64_t K, int64_t SX, int64_t SW, int64_t SY,
                      int64_t out32, uintptr_t scratch, uintptr_t scratch_bytes,
                      uintptr_t stream, int64_t cfg) {
    if (M < 1 || M > MAX_M) throw std::invalid_argument("M must be 1..16");
    if (N < 1 || K < 32 || K % 32 || SW % 16 || SW < K || SX % 8)
      throw std::invalid_argument("need N >= 1, K % 32 == 0, SW % 16 == 0, SW >= K, SX % 8 == 0");
    if (x % 16 || w % 16 || scale % 4 || scratch % 16)
      throw std::invalid_argument("x, w, scratch must be 16 B aligned, scale 4 B aligned");
    if (cfg < -1 || cfg >= NCFG) throw std::invalid_argument("cfg out of range");
    launch(Args{x, w, scale, bias, y, scratch, scratch_bytes, stream, (int)M, (int)N, (int)K,
                (int)SX, (int)SW, (int)SY, (int)(out32 != 0)}, (int)cfg);
  }, "int8 (Q8_0, fp32 scale per 32 K) weight low-M GEMV: y[M,N] = x[M,K] bf16 @ deq(Wq)[N,K]^T [+ bias]",
     py::arg("x"), py::arg("w"), py::arg("scale"), py::arg("bias"), py::arg("y"), py::arg("M"),
     py::arg("N"), py::arg("K"), py::arg("SX"), py::arg("SW"), py::arg("SY"), py::arg("out32"),
     py::arg("scratch"), py::arg("scratch_bytes"), py::arg("stream"), py::arg("cfg") = -1);
}
