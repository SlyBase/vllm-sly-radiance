// radiance_inkling2_moe.hip: MoE expert GEMV for Inkling-2-Mini decode / spec-verify on gfx1201.
//
// Replaces the Triton NVFP4 and MXFP8 expert kernels when the plan's token tile is 8 (a decode or
// spec-verify step of <= 8 tokens).  Such a step multiplies a few activation rows (<= 8 per CTA,
// ~1-3 on average) with a whole expert's weights, so it is a weight STREAM, not a tiled GEMM: every
// weight byte is read exactly once and the kernel only has to keep enough loads in flight.
//
// Reference semantics (vllm/models/inkling2/nvidia/ops/moe_sm120.py):
//   NVFP4: out[r][n] = (deq(act[r]) . deq(W[e][n])) * expert_scale[e] * row_scale[r]
//          e2m1 x e4m3 block scale per 16 k, act scales in the R8c4 layout
//   MXFP8: same without row_scale; e4m3 x e8m0 block scale per 32 k, byte 0 -> 2^-127
// `tiles` = [expert per CTA | row end per CTA | CTA count | padded rows]: CTA c holds rows
// 8c .. min(row_end, 8c + 8); CTAs >= the count do nothing; rows past row_end stay unwritten.
//
// Numerics: both operands stay in their storage format and meet in v_dot4_f32_fp8_fp8 (four e4m3
// products summed into fp32), one block scale at a time:
//   NVFP4: an e2m1 code (s e1 e0 m) is the e4m3 byte (s, e1 e0 m, 00) = e2m1 value * 2^-6 EXACTLY
//          (e = 0 lands in the e4m3 subnormals: m = 1 -> 2^-7 = 0.5 * 2^-6), so the products of
//          one 16-k group are exact multiples of 2^-12 and sum exactly in fp32; the 2^12 goes
//          into the epilogue.  Then group sum * w_scale * a_scale is one fp32 multiply-add.
//   MXFP8: e4m3 x e4m3 products are exact in fp32; a 32-k group sum is scaled by both e8m0s.
// The result equals the reference up to the fp32 summation order, then one bf16 rounding.
//
// Mapping: grid (N / BN blocks, max_ctas), BN = 4 weight rows per wave * INK_NW waves.  A weight
// row is streamed by LPR = 8 lanes, a lane takes one 32-k QUANTUM per step (NVFP4: 16 B + 2 scale
// bytes, MXFP8: 32 B + 1 scale byte), the 8 lanes of a row read 128 / 256 contiguous bytes.  All
// (<= NJ_MAX = 9) steps of a lane are loaded up front, so a wave has ~2.5 KB in flight.  The
// block first stages its CTA's valid act rows (<= lds_rows at once, more rows take extra passes)
// as fp8 + fp32 scales in LDS, issuing the act loads BEFORE the weight loads so the vmcnt wait
// for the act data does not drain the weight stream.  The 8 lanes of a row are summed with DPP.
//
// Graph safe: no host sync, no allocation, launch on the caller's stream.  A CTA whose expert id
// is outside [0, E) does nothing (a corrupt tile table must not fault the GPU).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace py = pybind11;

#ifndef INK_NW
#define INK_NW 8          // waves per block
#endif
constexpr int LPR = 8;                          // lanes per weight row
constexpr int ROWS_PER_WAVE = 32 / LPR;
constexpr int BN = INK_NW * ROWS_PER_WAVE;      // weight rows (output columns) per block
constexpr int QUANTUM = 32;                     // k values per lane and step
constexpr int NJ_MAX = 9;                       // steps per lane: K <= 9 * 8 * 32 = 2304
constexpr int K_MAX = NJ_MAX * LPR * QUANTUM;
constexpr int MAX_ROWS = 8;                     // rows per CTA = the token tile
constexpr int NT = INK_NW * 32;
constexpr int ITEMS = (MAX_ROWS * (K_MAX / QUANTUM) + NT - 1) / NT;  // act staging items per thread

typedef unsigned int u32x4 __attribute__((ext_vector_type(4)));
// Weights are streamed once: a non-temporal load keeps them out of the caches (measured on the
// R9700: NVFP4 w13 decode 540 -> 603 GB/s, 8-token step 573 -> 640 GB/s of ~640 GB/s peak).
__device__ __forceinline__ uint4 load_weight(const uint4 *p) {
    const u32x4 v = __builtin_nontemporal_load(reinterpret_cast<const u32x4 *>(p));
    return make_uint4(v.x, v.y, v.z, v.w);
}

__device__ __forceinline__ uint32_t e2m1_even(uint32_t x) {   // nibbles 0, 2, 4, 6 -> e4m3 bytes
    return ((x & 0x07070707u) << 2) | ((x & 0x08080808u) << 4);
}
__device__ __forceinline__ uint32_t e2m1_odd(uint32_t x) {    // nibbles 1, 3, 5, 7 -> e4m3 bytes
    return ((x & 0x70707070u) >> 2) | (x & 0x80808080u);
}
__device__ __forceinline__ float e8m0(uint32_t e) {           // upstream: exponent 0 -> 2^-127
    return __uint_as_float(e ? (e << 23) : 0x00400000u);
}
__device__ __forceinline__ float dot4(uint32_t a, uint32_t b, float c) {
    return __builtin_amdgcn_dot4_f32_fp8_fp8(a, b, c);
}
__device__ __forceinline__ float row_sum(float v) {           // the 8 lanes of one weight row
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0xB1, 0xF, 0xF, true));   // xor 1
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x4E, 0xF, 0xF, true));   // xor 2
    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x141, 0xF, 0xF, true));  // i <-> 7 - i
    return v;
}

template <bool NVFP4>
__global__ __launch_bounds__(NT) void moe_gemv(
    const uint8_t *__restrict__ act, const uint8_t *__restrict__ act_scale,
    const float *__restrict__ row_scale, const uint8_t *__restrict__ weight,
    const uint8_t *__restrict__ weight_scale, const float *__restrict__ expert_scale,
    __hip_bfloat16 *__restrict__ out, const int32_t *__restrict__ tiles,
    int max_ctas, int E, int N, int K, int lds_rows)
{
    constexpr int QB = NVFP4 ? 16 : 32;     // act and weight bytes per quantum
    constexpr int SB = NVFP4 ? 2 : 1;       // scale bytes (= groups) per quantum

    const int cta = blockIdx.y;
    if (cta >= tiles[2 * max_ctas]) return;
    const int expert = tiles[cta];
    if (expert < 0 || expert >= E) return;
    int nr = tiles[max_ctas + cta] - cta * MAX_ROWS;
    if (nr > MAX_ROWS) nr = MAX_ROWS;
    if (nr <= 0) return;

    const int nq = K / QUANTUM;
    const int nj = (nq + LPR - 1) / LPR;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5, li = lane & (LPR - 1);
    const int n = blockIdx.x * BN + wave * ROWS_PER_WAVE + (lane / LPR);
    const bool nok = n < N;
    const int cols = nq * SB;               // act scale columns per row

    extern __shared__ __attribute__((aligned(16))) uint8_t smem[];
    uint4 *s_act = reinterpret_cast<uint4 *>(smem);                              // [lds_rows][nq][2]
    float *s_sa = reinterpret_cast<float *>(smem + (size_t)lds_rows * nq * 32);  // [lds_rows][nq][SB]

    uint4 av0[ITEMS], av1[ITEMS];           // staged act rows: first / second 16 bytes of a quantum
    uint32_t asc[ITEMS];

    // act items: (row r, quantum q) = idx / nq, idx % nq; loads and LDS stores are split so the
    // weight loads can be issued in between (a vmcnt wait on the act data then only waits for it)
#define LOAD_ACT(R0, PROW)                                                                         \
    _Pragma("unroll") for (int i = 0; i < ITEMS; i++) {                                           \
        const int idx = tid + i * NT;                                                             \
        if (idx < (PROW) * nq) {                                                                  \
            const int r = idx / nq, q = idx - r * nq, row = cta * MAX_ROWS + (R0) + r;            \
            const uint4 *ap = reinterpret_cast<const uint4 *>(                                    \
                act + (size_t)row * (NVFP4 ? K / 2 : K) + (size_t)q * QB);                        \
            av0[i] = ap[0];                                                                       \
            if (!NVFP4) av1[i] = ap[1];                                                           \
            const int g = q * SB;                                                                 \
            const int off = ((row >> 3) * (cols >> 2) + (g >> 2)) * 32 + (row & 7) * 4 + (g & 3);  \
            asc[i] = NVFP4 ? *reinterpret_cast<const uint16_t *>(act_scale + off)                 \
                           : (uint32_t)act_scale[off];                                            \
        }                                                                                         \
    }
#define STORE_ACT(PROW)                                                                            \
    _Pragma("unroll") for (int i = 0; i < ITEMS; i++) {                                           \
        const int idx = tid + i * NT;                                                             \
        if (idx < (PROW) * nq) {                                                                  \
            const int r = idx / nq, q = idx - r * nq;                                             \
            uint4 *dst = s_act + ((size_t)r * nq + q) * 2;                                        \
            if (NVFP4) {                                                                          \
                const uint4 a = av0[i];                                                           \
                dst[0] = make_uint4(e2m1_even(a.x), e2m1_odd(a.x), e2m1_even(a.y), e2m1_odd(a.y)); \
                dst[1] = make_uint4(e2m1_even(a.z), e2m1_odd(a.z), e2m1_even(a.w), e2m1_odd(a.w)); \
                reinterpret_cast<float2 *>(s_sa)[(size_t)r * nq + q] =                            \
                    make_float2(__builtin_amdgcn_cvt_f32_fp8(asc[i], 0),                          \
                                __builtin_amdgcn_cvt_f32_fp8(asc[i], 1));                         \
            } else {                                                                              \
                const uint4 b0 = av0[i], b1 = av1[i];                                             \
                dst[0] = make_uint4(b0.x, b0.y, b0.z, b0.w);                                      \
                dst[1] = make_uint4(b1.x, b1.y, b1.z, b1.w);                                      \
                s_sa[(size_t)r * nq + q] = e8m0(asc[i]);                                          \
            }                                                                                     \
        }                                                                                         \
    }

    int prow = nr < lds_rows ? nr : lds_rows;
    LOAD_ACT(0, prow)

    // the weight stream: every step of this lane's row, issued before anything waits on a load
    uint4 wv0[NJ_MAX], wv1[NJ_MAX];
    uint32_t ws[NJ_MAX];
    const uint8_t *wrow = weight + ((size_t)expert * N + n) * (NVFP4 ? K / 2 : K);
    const uint8_t *srow = weight_scale + ((size_t)expert * N + n) * (NVFP4 ? K / 16 : K / 32);
#pragma unroll
    for (int j = 0; j < NJ_MAX; j++) {
        const int q = j * LPR + li;
        if (j < nj && nok && q < nq) {
            const uint4 *wp = reinterpret_cast<const uint4 *>(wrow + (size_t)q * QB);
            wv0[j] = load_weight(wp);
            if (!NVFP4) wv1[j] = load_weight(wp + 1);
            ws[j] = NVFP4 ? *reinterpret_cast<const uint16_t *>(srow + 2 * q) : (uint32_t)srow[q];
        } else {
            wv0[j] = make_uint4(0, 0, 0, 0);
            wv1[j] = make_uint4(0, 0, 0, 0);
            ws[j] = 0;
        }
    }

    const float es = expert_scale[expert] * (NVFP4 ? 4096.0f : 1.0f);   // 2^12: e2m1 as e4m3 = 2^-6 each
    for (int r0 = 0; r0 < nr; r0 += lds_rows) {
        prow = nr - r0 < lds_rows ? nr - r0 : lds_rows;
        if (r0 > 0) {
            __syncthreads();                // every wave is done with the previous rows' LDS
            LOAD_ACT(r0, prow)
        }
        STORE_ACT(prow)
        __syncthreads();

        float acc[MAX_ROWS];
#pragma unroll
        for (int r = 0; r < MAX_ROWS; r++) acc[r] = 0.0f;
#pragma unroll
        for (int j = 0; j < NJ_MAX; j++) {
            const int q = j * LPR + li;
            if (j < nj && q < nq) {
                if (NVFP4) {
                    const uint4 w4 = wv0[j];
                    const uint32_t e0 = e2m1_even(w4.x), o0 = e2m1_odd(w4.x);
                    const uint32_t e1 = e2m1_even(w4.y), o1 = e2m1_odd(w4.y);
                    const uint32_t e2 = e2m1_even(w4.z), o2 = e2m1_odd(w4.z);
                    const uint32_t e3 = e2m1_even(w4.w), o3 = e2m1_odd(w4.w);
                    const float sw0 = __builtin_amdgcn_cvt_f32_fp8(ws[j], 0);
                    const float sw1 = __builtin_amdgcn_cvt_f32_fp8(ws[j], 1);
#pragma unroll
                    for (int r = 0; r < MAX_ROWS; r++) {
                        if (r < prow) {
                            const uint4 *ap = s_act + ((size_t)r * nq + q) * 2;
                            const uint4 a0 = ap[0], a1 = ap[1];
                            const float2 sa = reinterpret_cast<const float2 *>(s_sa)[(size_t)r * nq + q];
                            float p0 = dot4(e0, a0.x, 0.0f);
                            p0 = dot4(o0, a0.y, p0);
                            p0 = dot4(e1, a0.z, p0);
                            p0 = dot4(o1, a0.w, p0);
                            float p1 = dot4(e2, a1.x, 0.0f);
                            p1 = dot4(o2, a1.y, p1);
                            p1 = dot4(e3, a1.z, p1);
                            p1 = dot4(o3, a1.w, p1);
                            acc[r] = fmaf(p0, sw0 * sa.x, acc[r]);
                            acc[r] = fmaf(p1, sw1 * sa.y, acc[r]);
                        }
                    }
                } else {
                    const uint4 w0 = wv0[j], w1 = wv1[j];
                    const float sw = e8m0(ws[j]);
#pragma unroll
                    for (int r = 0; r < MAX_ROWS; r++) {
                        if (r < prow) {
                            const uint4 *ap = s_act + ((size_t)r * nq + q) * 2;
                            const uint4 a0 = ap[0], a1 = ap[1];
                            float p = dot4(w0.x, a0.x, 0.0f);
                            p = dot4(w0.y, a0.y, p);
                            p = dot4(w0.z, a0.z, p);
                            p = dot4(w0.w, a0.w, p);
                            p = dot4(w1.x, a1.x, p);
                            p = dot4(w1.y, a1.y, p);
                            p = dot4(w1.z, a1.z, p);
                            p = dot4(w1.w, a1.w, p);
                            acc[r] = fmaf(p * sw, s_sa[(size_t)r * nq + q], acc[r]);
                        }
                    }
                }
            }
        }
        // the 8 lanes of a weight row hold partial sums; lane r stores row r (LPR == MAX_ROWS)
#pragma unroll
        for (int r = 0; r < MAX_ROWS; r++) {
            if (r < prow) {
                const float v = row_sum(acc[r]);
                if (nok && r == li) {
                    const int row = cta * MAX_ROWS + r0 + r;
                    out[(size_t)row * N + n] =
                        __float2bfloat16(v * es * (NVFP4 ? row_scale[row] : 1.0f));
                }
            }
        }
    }
}

// --------------------------------------------------------------------------- host entry points

static void check(bool ok, const char *what) {
    if (!ok) throw std::invalid_argument(what);
}

static void launch(bool nvfp4, uintptr_t act, uintptr_t act_scale, uintptr_t row_scale,
                   uintptr_t weight, uintptr_t weight_scale, uintptr_t expert_scale, uintptr_t out,
                   uintptr_t tiles, int max_ctas, int E, int N, int K, int tile, int lds_rows,
                   uintptr_t stream) {
    check(tile == MAX_ROWS, "tile must be 8 (decode / spec-verify tile)");
    check(max_ctas >= 1 && max_ctas <= 65535, "max_ctas out of range");
    check(E >= 1 && N >= 1, "experts and N must be positive");
    check(K >= QUANTUM && K <= K_MAX, "K outside the kernel's register budget (<= 2304)");
    check(K % (nvfp4 ? 64 : 128) == 0, "K must be a multiple of 64 (NVFP4) / 128 (MXFP8)");
    check(lds_rows >= 1 && lds_rows <= MAX_ROWS, "lds_rows must be in [1, 8]");
    check(act && act_scale && weight && weight_scale && expert_scale && out && tiles &&
              (!nvfp4 || row_scale), "null pointer");
    check(!(act & 15) && !(weight & 15) && !(out & 1) && !(act_scale & 1) && !(weight_scale & 1) &&
              !(tiles & 3) && !(expert_scale & 3) && !(row_scale & 3), "misaligned pointer");
    const int nq = K / QUANTUM;
    const size_t smem = (size_t)lds_rows * nq * (32 + 4 * (nvfp4 ? 2 : 1));
    dim3 grid((N + BN - 1) / BN, max_ctas), block(NT);
    auto *a = reinterpret_cast<const uint8_t *>(act);
    auto *as = reinterpret_cast<const uint8_t *>(act_scale);
    auto *rs = reinterpret_cast<const float *>(row_scale);
    auto *w = reinterpret_cast<const uint8_t *>(weight);
    auto *ws = reinterpret_cast<const uint8_t *>(weight_scale);
    auto *es = reinterpret_cast<const float *>(expert_scale);
    auto *o = reinterpret_cast<__hip_bfloat16 *>(out);
    auto *t = reinterpret_cast<const int32_t *>(tiles);
    auto s = reinterpret_cast<hipStream_t>(stream);
    if (nvfp4)
        hipLaunchKernelGGL(moe_gemv<true>, grid, block, smem, s, a, as, rs, w, ws, es, o, t,
                           max_ctas, E, N, K, lds_rows);
    else
        hipLaunchKernelGGL(moe_gemv<false>, grid, block, smem, s, a, as, rs, w, ws, es, o, t,
                           max_ctas, E, N, K, lds_rows);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
        throw std::runtime_error(std::string("moe_gemv launch: ") + hipGetErrorString(err));
}

static void nvfp4_gemm(uintptr_t act, uintptr_t act_scale, uintptr_t row_scale, uintptr_t weight,
                       uintptr_t weight_scale, uintptr_t expert_scale, uintptr_t out,
                       uintptr_t tiles, int max_ctas, int E, int N, int K, int tile, int lds_rows,
                       uintptr_t stream) {
    launch(true, act, act_scale, row_scale, weight, weight_scale, expert_scale, out, tiles,
           max_ctas, E, N, K, tile, lds_rows, stream);
}

static void mxfp8_gemm(uintptr_t act, uintptr_t act_scale, uintptr_t weight, uintptr_t weight_scale,
                       uintptr_t expert_scale, uintptr_t out, uintptr_t tiles, int max_ctas, int E,
                       int N, int K, int tile, int lds_rows, uintptr_t stream) {
    launch(false, act, act_scale, 0, weight, weight_scale, expert_scale, out, tiles, max_ctas, E,
           N, K, tile, lds_rows, stream);
}

static py::dict geometry() {
    py::dict d;
    d["bn"] = BN; d["lpr"] = LPR; d["quantum"] = QUANTUM; d["nj_max"] = NJ_MAX;
    d["max_rows"] = MAX_ROWS; d["k_max"] = K_MAX;
    return d;
}

PYBIND11_MODULE(radiance_inkling2_moe_hip, m) {
    m.doc() = "Inkling-2-Mini decode / spec-verify expert GEMV (NVFP4, MXFP8) for gfx1201";
    m.def("nvfp4_gemm", &nvfp4_gemm,
          "(act, act_scale, row_scale, weight, weight_scale, expert_scale, out, tiles, max_ctas, "
          "experts, N, K, tile, lds_rows, stream): raw device pointers");
    m.def("mxfp8_gemm", &mxfp8_gemm,
          "(act, act_scale, weight, weight_scale, expert_scale, out, tiles, max_ctas, experts, "
          "N, K, tile, lds_rows, stream): raw device pointers");
    m.def("geometry", &geometry, "launch geometry the Python address emulation mirrors");
}
