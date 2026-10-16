// gfx1201 WMMA fragment probe: pins down, on the hardware, which element each lane holds for the
// iu4 16x16x32, iu4 16x16x16 and fp8 16x16x16 instructions, and whether the int4 operand
// signedness flags behave. Every test multiplies asymmetric matrices and compares with a host
// reference; a wrong layout is a wrong result, not a near miss.
//
//   hipcc -O2 --offload-arch=gfx1201 probe_wmma.hip -o probe_wmma && ./probe_wmma
//
// Hypothesis under test (GPUOpen RDNA4 guide, rdna4-wmma-guide): for 16x16xK with E elements per
// lane (E = K/2 for 16-wide... no: E = 8 for K=16, 16 for K=32):
//   A (16 rows x K, K-major per lane):  lane l holds row l % 16, k = (l / 16) * E .. + E - 1
//   B (K x 16 cols):                     lane l holds col l % 16, same k range
//   C/D (16 x 16, fp32/i32 x 8):         lane l holds col l % 16, rows (l / 16) * 8 .. + 7
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

typedef int int2_t __attribute__((ext_vector_type(2)));
typedef int int8_t_ __attribute__((ext_vector_type(8)));
typedef float float8_t __attribute__((ext_vector_type(8)));

#define CHECK(x) do { hipError_t e = (x); if (e != hipSuccess) { printf("HIP error %s at %d\n", hipGetErrorString(e), __LINE__); exit(1); } } while (0)

// ---- iu4 16x16x32: A, B as packed nibbles (row-major A[16][32], B[32][16] on the host) ----
__global__ void k_iu4_32(const int8_t *A, const int8_t *B, int *D, int a_sign, int b_sign) {
  const int l = threadIdx.x;              // wave32
  const int E = 16;                       // nibbles per lane for K = 32
  int2_t a, b;
  // pack this lane's 16 nibbles under the hypothesis
  unsigned int aw[2] = {0, 0}, bw[2] = {0, 0};
  for (int i = 0; i < E; ++i) {
    int k = (l / 16) * E + i;
    unsigned int an = (unsigned int)(A[(l % 16) * 32 + k] & 0xF);
    unsigned int bn = (unsigned int)(B[k * 16 + (l % 16)] & 0xF);
    aw[i / 8] |= an << (4 * (i % 8));
    bw[i / 8] |= bn << (4 * (i % 8));
  }
  a[0] = (int)aw[0]; a[1] = (int)aw[1]; b[0] = (int)bw[0]; b[1] = (int)bw[1];
  int8_t_ c = {0, 0, 0, 0, 0, 0, 0, 0};
  if (a_sign && b_sign)      c = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c, false);
  else if (a_sign)           c = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, false, b, c, false);
  else if (b_sign)           c = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a, true, b, c, false);
  else                       c = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(false, a, false, b, c, false);
  for (int j = 0; j < 8; ++j) D[((l / 16) * 8 + j) * 16 + (l % 16)] = c[j];
}

// ---- iu4 16x16x16: 8 nibbles per lane (one int) ----
__global__ void k_iu4_16(const int8_t *A, const int8_t *B, int *D, int a_sign, int b_sign) {
  const int l = threadIdx.x;
  unsigned int aw = 0, bw = 0;
  for (int i = 0; i < 8; ++i) {
    int k = (l / 16) * 8 + i;
    aw |= ((unsigned int)(A[(l % 16) * 16 + k] & 0xF)) << (4 * i);
    bw |= ((unsigned int)(B[k * 16 + (l % 16)] & 0xF)) << (4 * i);
  }
  int a = (int)aw, b = (int)bw;
  int8_t_ c = {0, 0, 0, 0, 0, 0, 0, 0};
  if (a_sign && b_sign)      c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, true, b, c, false);
  else if (a_sign)           c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(true, a, false, b, c, false);
  else if (b_sign)           c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(false, a, true, b, c, false);
  else                       c = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12(false, a, false, b, c, false);
  for (int j = 0; j < 8; ++j) D[((l / 16) * 8 + j) * 16 + (l % 16)] = c[j];
}

// ---- fp8 16x16x16: 8 bytes per lane (int2) ----
__global__ void k_fp8_16(const unsigned char *A, const unsigned char *B, float *D) {
  const int l = threadIdx.x;
  unsigned char ab[8], bb[8];
  for (int i = 0; i < 8; ++i) {
    int k = (l / 16) * 8 + i;
    ab[i] = A[(l % 16) * 16 + k];
    bb[i] = B[k * 16 + (l % 16)];
  }
  int2_t a, b;
  a[0] = *(int *)&ab[0]; a[1] = *(int *)&ab[4]; b[0] = *(int *)&bb[0]; b[1] = *(int *)&bb[4];
  float8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
  c = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
  for (int j = 0; j < 8; ++j) D[((l / 16) * 8 + j) * 16 + (l % 16)] = c[j];
}

// ---- iu8 16x16x16: 8 bytes per lane (int2), a_sign / b_sign flags ----
__global__ void k_iu8_16(const int8_t *A, const int8_t *B, int *D, int a_sign, int b_sign) {
  const int l = threadIdx.x;
  unsigned char ab[8], bb[8];
  for (int i = 0; i < 8; ++i) {
    int k = (l / 16) * 8 + i;
    ab[i] = (unsigned char)A[(l % 16) * 16 + k];
    bb[i] = (unsigned char)B[k * 16 + (l % 16)];
  }
  int2_t a, b;
  a[0] = *(int *)&ab[0]; a[1] = *(int *)&ab[4]; b[0] = *(int *)&bb[0]; b[1] = *(int *)&bb[4];
  int8_t_ c = {0, 0, 0, 0, 0, 0, 0, 0};
  if (a_sign && b_sign)      c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
  else if (a_sign)           c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, false, b, c, false);
  else if (b_sign)           c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, true, b, c, false);
  else                       c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, a, false, b, c, false);
  for (int j = 0; j < 8; ++j) D[((l / 16) * 8 + j) * 16 + (l % 16)] = c[j];
}

// ---- v_cvt_pk_u8_f32 rounding / saturation ----
__global__ void k_cvt_pk(const float *x, unsigned int *out, int n) {
  const int i = threadIdx.x;
  if (i < n) out[i] = __builtin_amdgcn_cvt_pk_u8_f32(x[i], 0u, 0u);
}

static int byte_val(int n, bool sgn) { n &= 0xFF; return sgn ? (n >= 128 ? n - 256 : n) : n; }

static int nib_val(int n, bool sgn) { n &= 0xF; return sgn ? (n >= 8 ? n - 16 : n) : n; }

static float e4m3_to_f(unsigned char b) {
  int s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
  float v;
  if (e == 0) v = (m / 8.0f) * (1.0f / 64.0f);
  else v = (1.0f + m / 8.0f) * ldexpf(1.0f, e - 7);
  return s ? -v : v;
}

int main() {
  int fails = 0;
  // iu4, K = 32 and K = 16, all signedness combos
  for (int K : {32, 16}) {
    std::vector<int8_t> A(16 * K), B(K * 16);
    for (int i = 0; i < 16 * K; ++i) A[i] = (int8_t)((i * 7 + 3) & 0xF);
    for (int i = 0; i < K * 16; ++i) B[i] = (int8_t)((i * 11 + 5) & 0xF);
    int8_t *dA, *dB; int *dD;
    CHECK(hipMalloc(&dA, A.size())); CHECK(hipMalloc(&dB, B.size())); CHECK(hipMalloc(&dD, 256 * 4));
    CHECK(hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, B.data(), B.size(), hipMemcpyHostToDevice));
    for (int as = 0; as < 2; ++as) for (int bs = 0; bs < 2; ++bs) {
      if (K == 32) hipLaunchKernelGGL(k_iu4_32, 1, 32, 0, 0, dA, dB, dD, as, bs);
      else hipLaunchKernelGGL(k_iu4_16, 1, 32, 0, 0, dA, dB, dD, as, bs);
      CHECK(hipDeviceSynchronize());
      std::vector<int> D(256);
      CHECK(hipMemcpy(D.data(), dD, 256 * 4, hipMemcpyDeviceToHost));
      int bad = 0;
      for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) {
        int ref = 0;
        for (int k = 0; k < K; ++k) ref += nib_val(A[r * K + k], as) * nib_val(B[k * 16 + c], bs);
        if (ref != D[r * 16 + c]) ++bad;
      }
      printf("iu4 16x16x%d a_sign=%d b_sign=%d: %s (%d/256 wrong)\n", K, as, bs, bad ? "FAIL" : "ok", bad);
      fails += bad != 0;
    }
    hipFree(dA); hipFree(dB); hipFree(dD);
  }
  // iu8 16x16x16, all signedness combos
  {
    std::vector<int8_t> A(256), B(256);
    for (int i = 0; i < 256; ++i) { A[i] = (int8_t)((i * 37 + 11) & 0xFF); B[i] = (int8_t)((i * 53 + 7) & 0xFF); }
    int8_t *dA, *dB; int *dD;
    CHECK(hipMalloc(&dA, 256)); CHECK(hipMalloc(&dB, 256)); CHECK(hipMalloc(&dD, 256 * 4));
    CHECK(hipMemcpy(dA, A.data(), 256, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dB, B.data(), 256, hipMemcpyHostToDevice));
    for (int as = 0; as < 2; ++as) for (int bs = 0; bs < 2; ++bs) {
      hipLaunchKernelGGL(k_iu8_16, 1, 32, 0, 0, dA, dB, dD, as, bs);
      CHECK(hipDeviceSynchronize());
      std::vector<int> D(256);
      CHECK(hipMemcpy(D.data(), dD, 256 * 4, hipMemcpyDeviceToHost));
      int bad = 0;
      for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) {
        int ref = 0;
        for (int k = 0; k < 16; ++k) ref += byte_val(A[r * 16 + k], as) * byte_val(B[k * 16 + c], bs);
        if (ref != D[r * 16 + c]) ++bad;
      }
      printf("iu8 16x16x16 a_sign=%d b_sign=%d: %s (%d/256 wrong)\n", as, bs, bad ? "FAIL" : "ok", bad);
      fails += bad != 0;
    }
    hipFree(dA); hipFree(dB); hipFree(dD);
  }
  // v_cvt_pk_u8_f32: rounding mode and saturation
  {
    const float xs[12] = {0.4f, 0.5f, 0.6f, 1.5f, 2.5f, 3.5f, 254.4f, 254.5f, 255.4f, 300.f, -1.f, -0.4f};
    float *dx; unsigned int *dy;
    CHECK(hipMalloc(&dx, sizeof xs)); CHECK(hipMalloc(&dy, 12 * 4));
    CHECK(hipMemcpy(dx, xs, sizeof xs, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_cvt_pk, 1, 32, 0, 0, dx, dy, 12);
    CHECK(hipDeviceSynchronize());
    unsigned int ys[12];
    CHECK(hipMemcpy(ys, dy, 12 * 4, hipMemcpyDeviceToHost));
    printf("cvt_pk_u8_f32:");
    for (int i = 0; i < 12; ++i) printf(" %g->%u", xs[i], ys[i] & 0xFF);
    printf("\n");
  }
  // fp8 16x16x16
  {
    std::vector<unsigned char> A(256), B(256);
    for (int i = 0; i < 256; ++i) { A[i] = (unsigned char)(0x30 + ((i * 5) % 40)); B[i] = (unsigned char)(0x28 + ((i * 3) % 48)); }
    unsigned char *dA, *dB; float *dD;
    CHECK(hipMalloc(&dA, 256)); CHECK(hipMalloc(&dB, 256)); CHECK(hipMalloc(&dD, 256 * 4));
    CHECK(hipMemcpy(dA, A.data(), 256, hipMemcpyHostToDevice)); CHECK(hipMemcpy(dB, B.data(), 256, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(k_fp8_16, 1, 32, 0, 0, dA, dB, dD);
    CHECK(hipDeviceSynchronize());
    std::vector<float> D(256);
    CHECK(hipMemcpy(D.data(), dD, 256 * 4, hipMemcpyDeviceToHost));
    int bad = 0;
    for (int r = 0; r < 16; ++r) for (int c = 0; c < 16; ++c) {
      float ref = 0;
      for (int k = 0; k < 16; ++k) ref += e4m3_to_f(A[r * 16 + k]) * e4m3_to_f(B[k * 16 + c]);
      float got = D[r * 16 + c];
      if (fabsf(ref - got) > 1e-3f * (1 + fabsf(ref))) ++bad;
    }
    printf("fp8 16x16x16: %s (%d/256 wrong)\n", bad ? "FAIL" : "ok", bad);
    fails += bad != 0;
  }
  printf(fails ? "LAYOUT HYPOTHESIS WRONG for %d case(s)\n" : "all layouts confirmed\n", fails);
  return fails;
}
