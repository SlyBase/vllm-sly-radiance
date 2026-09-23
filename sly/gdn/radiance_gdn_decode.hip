// SPDX-License-Identifier: Apache-2.0
// HIP port of vLLM's fused GDN MTP decode kernel (csrc/libtorch_stable/gdn/fused_gdn_decode_kernel.cu,
// vLLM v0.29.0, Apache-2.0, copyright the vLLM project) for gfx1201 (RDNA4, wave32).
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Modified for HIP/gfx1201 by SlyBase (vllm-sly-radiance), 2026: see the port notes below.
//
// What it fuses, per (request, value head): the post-conv part of the gated-delta-net decode for a
// speculative verify batch (up to 8 tokens) -- q/k l2norm, the gating (softplus decay, sigmoid beta),
// the delta-rule state update with the state written back per accepted position, and the gated
// RMSNorm of the output. On this image the same work ran as the FLA Triton update kernel plus the
// glue around it (b/a .contiguous(), the separate norm/gate), ~6 extra launches per GDN layer and
// 48 GDN layers per target forward (rocprofv3, 2026-09-23).
//
// Port notes (the math is unchanged; line for line against the CUDA source):
//   * cp.async (PTX) -> the next state chunk is loaded into registers before this chunk's math and
//     stored to the other LDS stage after it (same double-buffer structure).
//   * __shfl_xor_sync / __shfl_sync -> __shfl_xor / __shfl. gfx1201 runs wave32 by default, so the
//     32-lane warp the kernel is written for maps one-to-one (static_assert below).
//   * __nv_bfloat16 -> __hip_bfloat16; launch bounds without the CUDA min-blocks hint.
//   * Registration is a single-file pybind11 module (like radiance_mxfp4_fp8.hip); argument checks
//     and the torch.ops._C op live in sly/gdn/radiance_gdn_decode.py.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;
typedef __hip_bfloat16 bf16;
typedef __hip_bfloat162 bf162;

namespace {

template <typename StateT, int ChunkV, int DimK, int Stages>
__device__ __forceinline__ void copy_state_chunk(StateT* shared_state, const StateT* state, int chunk,
                                                 int thread, int threads) {
  constexpr int kElementsPerCopy = 16 / sizeof(StateT);
  constexpr int kCopiesPerChunk = ChunkV * DimK / kElementsPerCopy;
  const int stage = chunk % Stages;
  for (int copy = thread; copy < kCopiesPerChunk; copy += threads) {
    const int element = copy * kElementsPerCopy;
    *reinterpret_cast<uint4*>(shared_state + stage * ChunkV * DimK + element) =
        *reinterpret_cast<const uint4*>(state + chunk * ChunkV * DimK + element);
  }
}

template <typename StateT>
__device__ __forceinline__ float4 load_state4(const StateT* state);

template <>
__device__ __forceinline__ float4 load_state4<float>(const float* state) {
  return *reinterpret_cast<const float4*>(state);
}

template <>
__device__ __forceinline__ float4 load_state4<bf16>(const bf16* state) {
  const bf162 lo = *reinterpret_cast<const bf162*>(state);
  const bf162 hi = *reinterpret_cast<const bf162*>(state + 2);
  return make_float4(__bfloat162float(lo.x), __bfloat162float(lo.y), __bfloat162float(hi.x),
                     __bfloat162float(hi.y));
}

template <typename StateT>
__device__ __forceinline__ void store_state4(StateT* state, float4 value);

template <>
__device__ __forceinline__ void store_state4<float>(float* state, float4 value) {
  *reinterpret_cast<float4*>(state) = value;
}

// HIP has no __floats2bfloat162_rn; two __float2bfloat16 (round-to-nearest-even) are the same bits.
__device__ __forceinline__ bf162 pack_bf162(float x, float y) {
  bf162 v;
  v.x = __float2bfloat16(x);
  v.y = __float2bfloat16(y);
  return v;
}

template <>
__device__ __forceinline__ void store_state4<bf16>(bf16* state, float4 value) {
  *reinterpret_cast<bf162*>(state) = pack_bf162(value.x, value.y);
  *reinterpret_cast<bf162*>(state + 2) = pack_bf162(value.z, value.w);
}

constexpr int kDimK = 128;
constexpr int kDimV = 128;
constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr int kWarps = kThreads / kWarp;
constexpr int kChunkV = 32;
constexpr int kNumChunks = kDimV / kChunkV;
constexpr int kRowsPerWarp = kChunkV / kWarps;
constexpr int kMaxMtpTokens = 8;
constexpr int kDtBiasFloat32 = 0;
constexpr int kDtBiasBFloat16 = 1;
constexpr int kDtBiasFloat16 = 2;
static_assert(kRowsPerWarp == 4, "the reduction pairs below assume four rows per warp");

struct GdnDecodeStrides {
  int64_t mixed_row;
  int64_t a_row;
  int64_t b_row;
  int64_t gate_row;
  int64_t state_slot;
};

__device__ __forceinline__ float sigmoid_fast(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_fast(float x) { return x * sigmoid_fast(x); }
__device__ __forceinline__ float softplus_fast(float x) { return x > 20.0f ? x : log1pf(__expf(x)); }

__device__ __forceinline__ float load_dt_bias(const void* dt_bias, int head, int dt_bias_type) {
  if (dt_bias_type == kDtBiasBFloat16) return __bfloat162float(static_cast<const bf16*>(dt_bias)[head]);
  if (dt_bias_type == kDtBiasFloat16) return __half2float(static_cast<const __half*>(dt_bias)[head]);
  return static_cast<const float*>(dt_bias)[head];
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) value += __shfl_xor(value, offset, kWarp);
  return value;
}

struct Sum2 {
  float x;
  float y;
};

__device__ __forceinline__ Sum2 warp_reduce_sum_pair(float x, float y) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_xor(x, offset, kWarp);
    y += __shfl_xor(y, offset, kWarp);
  }
  return {x, y};
}

template <typename StateT, int ValueHeadsPerKeyHead, bool SigmoidGate>
__global__ __launch_bounds__(kThreads) void gdn_decode_post_conv_mtp_kernel(
    const bf16* __restrict__ mixed_qkv, const bf16* __restrict__ a, const bf16* __restrict__ b,
    const float* __restrict__ a_log, const void* __restrict__ dt_bias,
    const int* __restrict__ state_indices, const int* __restrict__ cu_seqlens,
    const int* __restrict__ num_accepted_tokens, StateT* __restrict__ state,
    const bf16* __restrict__ output_gate, const void* __restrict__ norm_weight, bf16* __restrict__ out,
    int H, int HV, int state_indices_width, int dt_bias_type, bool norm_weight_is_bf16, float scale,
    float norm_eps, GdnDecodeStrides strides) {
  const int request = blockIdx.x;
  const int value_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & (kWarp - 1);
  const int warp = tid / kWarp;
  const int bos = cu_seqlens[request];
  const int eos = cu_seqlens[request + 1];
  const int num_tokens = eos - bos;
  if (num_tokens <= 0) return;

  const int accepted = num_accepted_tokens[request];
  const int source_slot = accepted > 0 && accepted <= state_indices_width
                              ? state_indices[request * state_indices_width + accepted - 1]
                              : 0;
  if (source_slot <= 0 || num_tokens > kMaxMtpTokens) {
    for (int linear = tid; linear < num_tokens * kDimV; linear += kThreads) {
      const int token = bos + linear / kDimV;
      const int value = linear % kDimV;
      const int64_t out_offset = (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(0.0f);
    }
    return;
  }

  const int key_head = value_head / ValueHeadsPerKeyHead;
  __shared__ StateT shared_state[2][kChunkV][kDimK];
  __shared__ float shared_q[kMaxMtpTokens][kDimK];
  __shared__ float shared_k[kMaxMtpTokens][kDimK];
  __shared__ bf16 shared_v[kMaxMtpTokens][kDimV];
  __shared__ bf16 shared_out[kMaxMtpTokens][kDimV];
  __shared__ float shared_decay[kMaxMtpTokens];
  __shared__ float shared_beta[kMaxMtpTokens];

  StateT* source_state =
      state + static_cast<int64_t>(source_slot) * strides.state_slot + value_head * kDimV * kDimK;
  copy_state_chunk<StateT, kChunkV, kDimK, 2>(&shared_state[0][0][0], source_state, 0, tid, kThreads);

  if (warp < num_tokens) {
    const int t = warp;
    const int token = bos + t;
    const int64_t mixed_base = static_cast<int64_t>(token) * strides.mixed_row;
    float q_values[4];
    float k_values[4];
    float q_square = 0.0f;
    float k_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      q_values[i] = __bfloat162float(mixed_qkv[mixed_base + key_head * kDimK + dim]);
      k_values[i] = __bfloat162float(mixed_qkv[mixed_base + H * kDimK + key_head * kDimK + dim]);
      shared_v[t][dim] = mixed_qkv[mixed_base + 2 * H * kDimK + value_head * kDimV + dim];
      q_square += q_values[i] * q_values[i];
      k_square += k_values[i] * k_values[i];
    }
    const Sum2 qk_sums = warp_reduce_sum_pair(q_square, k_square);
    const float q_scale = __shfl(lane == 0 ? rsqrtf(qk_sums.x + 1.0e-6f) * scale : 0.0f, 0, kWarp);
    const float k_scale = __shfl(lane == 0 ? rsqrtf(qk_sums.y + 1.0e-6f) : 0.0f, 0, kWarp);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int dim = lane + i * 32;
      shared_q[t][dim] = q_values[i] * q_scale;
      shared_k[t][dim] = k_values[i] * k_scale;
    }
    if (lane == 0) {
      const float a_value = __bfloat162float(a[static_cast<int64_t>(token) * strides.a_row + value_head]);
      const float b_value = __bfloat162float(b[static_cast<int64_t>(token) * strides.b_row + value_head]);
      const float g = -__expf(a_log[value_head]) *
                      softplus_fast(a_value + load_dt_bias(dt_bias, value_head, dt_bias_type));
      shared_decay[t] = __expf(g);
      shared_beta[t] = sigmoid_fast(b_value);
    }
  }
  __syncthreads();

  const int k_base = lane * 4;
  int rows[kRowsPerWarp];
#pragma unroll
  for (int row = 0; row < kRowsPerWarp; ++row) rows[row] = warp + row * kWarps;

  // cp.async replacement: the next chunk is loaded into registers at the top of the iteration and
  // written to the other LDS stage only after this chunk's math. Measured against a plain
  // synchronous copy (2026-09-23): the same within noise at 1-8 requests -- the kernel is bound by
  // the per-token state write-back, not by staging -- kept because it costs nothing (56 VGPRs,
  // 8 waves/SIMD).
  constexpr int kPref = (kChunkV * kDimK * (int)sizeof(StateT) / 16) / kThreads;
  static_assert(kPref * kThreads * 16 == kChunkV * kDimK * (int)sizeof(StateT), "chunk must tile the block");
#pragma unroll
  for (int chunk = 0; chunk < kNumChunks; ++chunk) {
    __syncthreads();
    uint4 pref[kPref];
    const bool has_next = chunk + 1 < kNumChunks;
    if (has_next) {
      const uint4* src = reinterpret_cast<const uint4*>(source_state + (chunk + 1) * kChunkV * kDimK);
#pragma unroll
      for (int p = 0; p < kPref; ++p) pref[p] = src[tid + p * kThreads];
    }

    float h[kRowsPerWarp][4];
#pragma unroll
    for (int row = 0; row < kRowsPerWarp; ++row) {
      const float4 state_value = load_state4(&shared_state[chunk & 1][rows[row]][k_base]);
      h[row][0] = state_value.x;
      h[row][1] = state_value.y;
      h[row][2] = state_value.z;
      h[row][3] = state_value.w;
    }

    for (int t = 0; t < num_tokens; ++t) {
      const float4 q4 = *reinterpret_cast<const float4*>(&shared_q[t][k_base]);
      const float4 k4 = *reinterpret_cast<const float4*>(&shared_k[t][k_base]);
      const float q_values[4] = {q4.x, q4.y, q4.z, q4.w};
      const float k_values[4] = {k4.x, k4.y, k4.z, k4.w};

      float dot_hk[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] *= shared_decay[t];
          dot_hk[row] += h[row][i] * k_values[i];
        }
      }
      const Sum2 dot_hk_01 = warp_reduce_sum_pair(dot_hk[0], dot_hk[1]);
      const Sum2 dot_hk_23 = warp_reduce_sum_pair(dot_hk[2], dot_hk[3]);
      const float reduced_hk[kRowsPerWarp] = {dot_hk_01.x, dot_hk_01.y, dot_hk_23.x, dot_hk_23.y};

      float dot_hq[kRowsPerWarp] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int row = 0; row < kRowsPerWarp; ++row) {
        const int value = chunk * kChunkV + rows[row];
        const float delta = (__bfloat162float(shared_v[t][value]) - reduced_hk[row]) * shared_beta[t];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          h[row][i] += k_values[i] * delta;
          dot_hq[row] += h[row][i] * q_values[i];
        }
      }
      const Sum2 dot_hq_01 = warp_reduce_sum_pair(dot_hq[0], dot_hq[1]);
      const Sum2 dot_hq_23 = warp_reduce_sum_pair(dot_hq[2], dot_hq[3]);
      if (lane == 0) {
        shared_out[t][chunk * kChunkV + rows[0]] = __float2bfloat16(dot_hq_01.x);
        shared_out[t][chunk * kChunkV + rows[1]] = __float2bfloat16(dot_hq_01.y);
        shared_out[t][chunk * kChunkV + rows[2]] = __float2bfloat16(dot_hq_23.x);
        shared_out[t][chunk * kChunkV + rows[3]] = __float2bfloat16(dot_hq_23.y);
      }

      const int destination_slot = state_indices[request * state_indices_width + t];
      if (destination_slot > 0) {
        StateT* destination_state = state + static_cast<int64_t>(destination_slot) * strides.state_slot +
                                    value_head * kDimV * kDimK;
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
          const int value = chunk * kChunkV + rows[row];
          store_state4(destination_state + value * kDimK + k_base,
                       make_float4(h[row][0], h[row][1], h[row][2], h[row][3]));
        }
      }
    }
    if (has_next) {
      uint4* dst = reinterpret_cast<uint4*>(&shared_state[(chunk + 1) & 1][0][0]);
#pragma unroll
      for (int p = 0; p < kPref; ++p) dst[tid + p * kThreads] = pref[p];
    }
  }
  __syncthreads();

  if (warp < num_tokens) {
    const int t = warp;
    float output_values[4];
    float sum_square = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      output_values[i] = __bfloat162float(shared_out[t][value]);
      sum_square += output_values[i] * output_values[i];
    }
    sum_square = warp_reduce_sum(sum_square);
    const float rstd = rsqrtf(sum_square / static_cast<float>(kDimV) + norm_eps);
    const int token = bos + t;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int value = lane + i * 32;
      const float gate_input = __bfloat162float(
          output_gate[static_cast<int64_t>(token) * strides.gate_row + value_head * kDimV + value]);
      const float gate = SigmoidGate ? sigmoid_fast(gate_input) : silu_fast(gate_input);
      const float weight = norm_weight_is_bf16 ? __bfloat162float(static_cast<const bf16*>(norm_weight)[value])
                                               : static_cast<const float*>(norm_weight)[value];
      const int64_t out_offset = (static_cast<int64_t>(token) * HV + value_head) * kDimV + value;
      out[out_offset] = __float2bfloat16(output_values[i] * rstd * weight * gate);
    }
  }
}

template <typename StateT, int R, bool SG>
void launch_typed(uintptr_t mixed_qkv, uintptr_t a, uintptr_t b, uintptr_t a_log, uintptr_t dt_bias,
                  uintptr_t state_indices, uintptr_t cu_seqlens, uintptr_t num_accepted, uintptr_t state,
                  uintptr_t output_gate, uintptr_t norm_weight, uintptr_t out, int num_requests, int H,
                  int HV, int width, int dt_bias_type, bool norm_bf16, float scale, float eps,
                  GdnDecodeStrides strides, uintptr_t stream) {
  hipLaunchKernelGGL((gdn_decode_post_conv_mtp_kernel<StateT, R, SG>), dim3(num_requests, HV), dim3(kThreads),
                     0, (hipStream_t)stream, (const bf16*)mixed_qkv, (const bf16*)a, (const bf16*)b,
                     (const float*)a_log, (const void*)dt_bias, (const int*)state_indices,
                     (const int*)cu_seqlens, (const int*)num_accepted, (StateT*)state, (const bf16*)output_gate,
                     (const void*)norm_weight, (bf16*)out, H, HV, width, dt_bias_type, norm_bf16, scale, eps,
                     strides);
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess)
    throw std::runtime_error(std::string("radiance_gdn_decode launch failed: ") + hipGetErrorString(err));
}

template <typename StateT, int R>
void launch_gate(bool sigmoid_gate, uintptr_t mixed_qkv, uintptr_t a, uintptr_t b, uintptr_t a_log,
                 uintptr_t dt_bias, uintptr_t state_indices, uintptr_t cu_seqlens, uintptr_t num_accepted,
                 uintptr_t state, uintptr_t output_gate, uintptr_t norm_weight, uintptr_t out, int n, int H,
                 int HV, int width, int dtb, bool nbf, float scale, float eps, GdnDecodeStrides s,
                 uintptr_t stream) {
  if (sigmoid_gate)
    launch_typed<StateT, R, true>(mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens, num_accepted,
                                  state, output_gate, norm_weight, out, n, H, HV, width, dtb, nbf, scale, eps, s,
                                  stream);
  else
    launch_typed<StateT, R, false>(mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens, num_accepted,
                                   state, output_gate, norm_weight, out, n, H, HV, width, dtb, nbf, scale, eps, s,
                                   stream);
}

}  // namespace

// Pointers and shapes are checked on the Python side (radiance_gdn_decode.py mirrors the CUDA op's
// STD_TORCH_CHECKs). state_is_fp32 selects the state type; ratio = HV / H in {1, 2, 3, 4, 8}.
static void launch(uintptr_t mixed_qkv, uintptr_t a, uintptr_t b, uintptr_t a_log, uintptr_t dt_bias,
                   uintptr_t state_indices, uintptr_t cu_seqlens, uintptr_t num_accepted, uintptr_t state,
                   uintptr_t output_gate, uintptr_t norm_weight, uintptr_t out, int num_requests, int H, int HV,
                   int width, int dt_bias_type, bool norm_bf16, double scale, double eps, int64_t mixed_row,
                   int64_t a_row, int64_t b_row, int64_t gate_row, int64_t state_slot, bool state_is_fp32,
                   bool sigmoid_gate, uintptr_t stream) {
  const GdnDecodeStrides s{mixed_row, a_row, b_row, gate_row, state_slot};
  const int ratio = HV / H;
#define RAD_GDN_R(R_)                                                                                         \
  do {                                                                                                        \
    if (state_is_fp32)                                                                                        \
      launch_gate<float, R_>(sigmoid_gate, mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens,        \
                             num_accepted, state, output_gate, norm_weight, out, num_requests, H, HV, width,  \
                             dt_bias_type, norm_bf16, (float)scale, (float)eps, s, stream);                   \
    else                                                                                                      \
      launch_gate<bf16, R_>(sigmoid_gate, mixed_qkv, a, b, a_log, dt_bias, state_indices, cu_seqlens,         \
                            num_accepted, state, output_gate, norm_weight, out, num_requests, H, HV, width,   \
                            dt_bias_type, norm_bf16, (float)scale, (float)eps, s, stream);                    \
  } while (0)
  switch (ratio) {
    case 1: RAD_GDN_R(1); break;
    case 2: RAD_GDN_R(2); break;
    case 3: RAD_GDN_R(3); break;
    case 4: RAD_GDN_R(4); break;
    case 8: RAD_GDN_R(8); break;
    default: throw std::runtime_error("radiance_gdn_decode: HV/H must be in {1, 2, 3, 4, 8}");
  }
#undef RAD_GDN_R
}

PYBIND11_MODULE(radiance_gdn_decode_ext, m) {
  m.def("launch", &launch, "fused GDN MTP decode (post-conv), HIP port of vLLM's CUDA kernel");
}
