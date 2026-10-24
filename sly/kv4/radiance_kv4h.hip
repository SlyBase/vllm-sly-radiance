// TetraAttention (working name KV4H), gfx1201: 4-bit K/V cache, int4 / int8 WMMA, split-KV attention.
//
// Reads the layout of radiance_kv4h_format.py (per (block, kv head): K groups of 16 tokens with
// two per-128-dim e4m3 scales per token and interleaved signed nibbles; V dim-major per 16-token
// group with one e4m3 scale per token). Computes, per program (query block of 64 rows, kv head,
// KV segment), the online-softmax partials AITER's Triton reduce_segments merges: segm_output
// [tok][head][seg][256] fp32, segm_max / segm_expsum [tok][head][seg]. With one segment it writes
// the normalized output itself (bf16, still in the rotated basis; the backend un-rotates).
//
// The kernel is instruction-issue bound (v8 ran ~1900 VALU ops per tile and wave; the matrix
// instructions are a minor part), so every per-element float op on the tile data is what costs:
//   * K nibbles go into v_wmma_i32_16x16x32_iu4 as stored (signed flag); the query is int4 (one
//     plane, default) or int8 fed as two int4 planes (16 * hi + lo). The 32-dim WMMAs of one K
//     scale group chain into one accumulator, so the group scale is applied once per group (one
//     group = the whole head by default, KV4H_NS = 1): 6 ops per score and group.
//   * P is uint8 (v_cvt_pk_u8_f32 packs it), V's nibbles are shifted into the high half of their
//     byte -- a signed int8 worth 16 * v -- and the product runs on v_wmma_i32_16x16x16_iu8, both
//     16-token halves chained: no nibble packing, one cvt + fma per output element per tile.
//   * Interior tiles (fully inside the causal prefix) skip the mask; the accumulator rescale runs
//     only when some row's max moved (wave-uniform branch); exp2 is the raw v_exp_f32.
//   * Tile scales are converted once in the staging step (K scales, V scales pre-lifted so the
//     tile's largest V scale maps P to 255).
//
// Layouts (verified by probe_wmma.hip on the card):
//   A (16 x K, K-major per lane): lane l holds row l % 16, k = (l / 16) * E .. + E - 1, E = K / 2
//   B (K x 16):                   lane l holds col l % 16, same k range
//   C / D (16 x 16, 8 per lane):  lane l holds col l % 16, rows (l / 16) * 8 .. + 7
// We compute S^T = K . Q^T (K rows = tokens as A, Q^T as B), so a lane's C fragment holds one
// QUERY (its column) and 8 TOKENS (its rows): the softmax over tokens needs one cross-lane max
// (l ^ 16), the per-token scales are per-j scalars, and P^T's layout is exactly the A operand
// of the P . V product (row = query, k = 8 tokens). V's B operand (8 tokens of one dim) is one
// 8-byte LDS read thanks to the dim-major V layout.
//
// Build (single file, no torch headers), see radiance_kv4h_kernel.py:
//   hipcc -O3 -fPIC -shared -std=c++20 --offload-arch=gfx1201 $(python -m pybind11 --includes) \
//         radiance_kv4h.hip -o radiance_kv4h.so
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <stdexcept>

namespace py = pybind11;
typedef int int2_t __attribute__((ext_vector_type(2)));
typedef int int8v __attribute__((ext_vector_type(8)));
typedef unsigned int uint4_t __attribute__((ext_vector_type(4)));

#define TG 16                     // tokens per layout group
#define V_SCALES 16               // bytes of V scales per 16-token group
// TILE (tokens per tile) = NH layout groups of 16: a template parameter (16 / 32 / 64), chosen
// per launch (RADIANCE_KV4H_TILE / _PREFILL_TILE in the launcher)
// Build-time knobs (one .so per combination, radiance_kv4h_kernel.py builds them on demand):
//   KV4H_NRT  row tiles of 16 query rows per program (2 | 4): BLOCK_M = 16 * NRT, waves = 2 * NRT
//   KV4H_QP   query planes (2: int8 query as two int4 planes, 1: int4 query, one WMMA per group)
//   KV4H_NS   K scale groups per head (1 | 2 | 4 -> scale per HEAD / NS dims)
//   KV4H_PREF P's uint8 reference (0: the running max over all tiles so far -- a tile far below
//             it, e.g. everything after a dominant early key, keeps only the few top levels and
//             tokens under 0.5 / 255 of the running max vanish from P.V while the sum counts
//             them; 1: the tile's own max, the full 255 levels per tile, the gap to the running
//             max carried as a per-query factor beta = exp2(m_tile - m_run) on the tile's P.V and
//             sum -- 8 multiplies and one LDS exchange per tile)
#ifndef KV4H_PREF
#define KV4H_PREF 0
#endif
#ifndef KV4H_NRT
#define KV4H_NRT 4
#endif
#ifndef KV4H_QP
#define KV4H_QP 2
#endif
#ifndef KV4H_NS
#define KV4H_NS 2
#endif

// per-head-size layout (radiance_kv4h_format.Layout): the cell is HEAD/2 nibble bytes plus 8 (head
// 256) or 4 (head 128: vLLM's int4 cell of the DFlash drafter, so its pages line up) scale bytes
template <int HEAD> struct L {
  static constexpr int PACKED = HEAD / 2;              // nibble bytes per token and side
  static constexpr int NG = HEAD / 32;                 // 32-dim WMMA groups
  static constexpr int TPT = PACKED + (HEAD == 256 ? 8 : 4);
  static constexpr int REG = TG * TPT;                 // bytes per 16-token group, one side
  static constexpr int K_AREA = TG * (TPT - PACKED);   // scale area before the K nibbles
  static constexpr int DH = HEAD / 2;                  // output dims per wave
  static constexpr int DT = DH / 16;                   // 16-dim tiles per wave
  static constexpr int KSTR = PACKED + 8;              // LDS row stride for K (no 16-way bank conflict)
  static constexpr int KCH = PACKED / 16;              // 16-byte chunks per K row
};

// ---- helpers -------------------------------------------------------------------------------

__device__ __forceinline__ float e4m3_to_f32(unsigned int b) {
  // exact: e4m3fn, bias 7, subnormals m * 2^-9, no inf (0x7F / 0xFF are NaN, treated as max)
  const unsigned int s = (b & 0x80u) << 24, e = (b >> 3) & 0xFu, m = b & 7u;
  float v;
  if (e == 0) v = (float)m * (1.0f / 512.0f);
  else v = __uint_as_float(((e + 120u) << 23) | (m << 20));
  return __uint_as_float(__float_as_uint(v) | s);
}

// 8 bytes' low nibbles -> 8 packed nibbles (one int); high nibbles likewise (query planes only)
__device__ __forceinline__ unsigned int pack_lo(unsigned int w0, unsigned int w1) {
  unsigned int x = w0 & 0x0F0F0F0Fu, y = w1 & 0x0F0F0F0Fu;
  x = (x | (x >> 4)) & 0x00FF00FFu; x = (x | (x >> 8)) & 0x0000FFFFu;
  y = (y | (y >> 4)) & 0x00FF00FFu; y = (y | (y >> 8)) & 0x0000FFFFu;
  return x | (y << 16);
}
__device__ __forceinline__ unsigned int pack_hi(unsigned int w0, unsigned int w1) {
  return pack_lo(w0 >> 4, w1 >> 4);
}

__device__ __forceinline__ float wave_max16(float v) {   // combine lanes l and l ^ 16
  return fmaxf(v, __shfl_xor(v, 16, 32));
}
__device__ __forceinline__ float wave_sum16(float v) {
  return v + __shfl_xor(v, 16, 32);
}
__device__ __forceinline__ float wave_max32(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  return v;
}
__device__ __forceinline__ float exp2_raw(float x) { return __builtin_amdgcn_exp2f(x); }

// ---- the kernel ------------------------------------------------------------------------------
//
// 8 waves: waves w and w + 4 share the 16 query rows of row tile w & 3, compute the same scores
// and softmax, and each accumulates one half of the 256 dims (64 fp32 accumulators per lane).

template <int HEAD, int NH, int NRT, int QP, int NS>
__global__ __launch_bounds__(NRT * 64) void kv4h_attention(
    const int8_t *__restrict__ q,            // [num_tokens][num_q_heads][HEAD] int8 (rotated, per-row scale)
    const float *__restrict__ q_descale,     // [num_tokens][num_q_heads]
    const unsigned char *__restrict__ cache, // [blocks][kv_heads][block_size * 2 * TPT]
    const int *__restrict__ block_table,     // [num_seqs][bt_stride]
    const int *__restrict__ seq_lens,        // [num_seqs]
    const int *__restrict__ cu_seqlens_q,    // [num_seqs + 1]
    float *__restrict__ segm_out,            // [num_tokens][num_q_heads][nseg][HEAD]
    float *__restrict__ segm_max,            // [num_tokens][num_q_heads][nseg]
    float *__restrict__ segm_sum,
    __bf16 *__restrict__ out,                // [num_tokens][num_q_heads][HEAD], nseg == 1 only
    float scale_log2,                        // softmax scale / HEAD * log2(e)
    int num_q_heads, int nqpkv, int block_q,
    long cache_bstride, long cache_hstride, int block_size, int bt_stride,
    int nseg, int num_seqs, int causal,      // causal 0: every query sees all seq_len tokens (DFlash block)
    int window) {                            // > 0: key t allowed iff qabs - t < window (and t - qabs < window if non-causal)
  constexpr int PACKED = L<HEAD>::PACKED, NG = L<HEAD>::NG, REG = L<HEAD>::REG;
  constexpr int BLOCK_M = NRT * 16, NWAVES = NRT * 2, NTHREADS = NWAVES * 32;
  static_assert(NS * TG <= L<HEAD>::K_AREA, "K scale groups must fit the scale area");
  static_assert(NG % NS == 0, "scale groups must align with 32-dim WMMA groups");
  constexpr int K_AREA = L<HEAD>::K_AREA, DH = L<HEAD>::DH, DT = L<HEAD>::DT, KSTR = L<HEAD>::KSTR;
  constexpr int KCH = L<HEAD>::KCH, TPT = L<HEAD>::TPT;
  constexpr int TILE = NH * TG;                 // tokens per tile
  constexpr int TPG = NTHREADS / NH;            // staging threads per 16-token group
  constexpr int KCPT = (TG * KCH + TPG - 1) / TPG;   // K 16-byte chunks per staging thread
  constexpr int VCPT = (PACKED + TPG - 1) / TPG;     // V chunks per staging thread
  constexpr int VSPL = (TILE + 31) / 32;             // V scales per lane of wave 1
  const int qb = blockIdx.x, kvh = blockIdx.y, seg = blockIdx.z;
  const int tid = threadIdx.x, lane = tid & 31;
  // wave-uniform values as SGPRs: the compiler then branches on them instead of masking lanes
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int rt = wave % NRT, dh = wave / NRT;

  // ---- which sequence / query rows (AITER's find_seq_idx, q-block mode) ----
  int lo = 0, hi = num_seqs;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    const int v = cu_seqlens_q[mid] / block_q + mid;
    if (v <= qb) lo = mid + 1; else hi = mid;
  }
  const int seq = lo - 1;
  const int q_start = cu_seqlens_q[seq];
  const int qlen = cu_seqlens_q[seq + 1] - q_start;
  const int qb_local = qb - (q_start / block_q + seq);
  if (qb_local * block_q >= qlen) return;
  const int seq_len = seq_lens[seq];
  const int context_len = seq_len - qlen;

  // this lane's query (its C column): row r = rt*16 + lane%16 of the 64-row block
  const int row = rt * 16 + (lane & 15);
  const int qpos = qb_local * block_q + row / nqpkv;
  const int qhead = kvh * nqpkv + row % nqpkv;
  // rows past block_q * nqpkv would be the NEXT block's first token: dead here (two programs with
  // different causal limits split the segments differently and would overwrite each other)
  const bool qvalid = row < block_q * nqpkv && qpos < qlen && qhead < num_q_heads;
  const int qtok = q_start + (qvalid ? qpos : 0);
  const float rs = qvalid ? scale_log2 * q_descale[(size_t)qtok * num_q_heads + qhead] : 0.f;
  const int qabs = context_len + qpos;                 // this query's position
  const int qlast = causal ? qabs : seq_len - 1;       // last token this query sees
  // the wave's first / last row positions: tiles inside both rows' limits need no mask
  const int qabs_lo = context_len + qb_local * block_q + (rt * 16) / nqpkv;
  const int qabs_hi = context_len + qb_local * block_q + (rt * 16 + 15) / nqpkv;
  const int qlast_lo = causal ? qabs_lo : seq_len - 1;

  // ---- Q^T as B operands: two int4 planes per 32-dim group, parked in LDS (the two dim-half
  //      waves of a row tile share them) ----
  __shared__ __align__(16) int2_t sQ[NRT][QP][NG][32];        // [row tile][plane][group][lane]
  if (dh == 0) {
    const int8_t *qrow = q + ((size_t)qtok * num_q_heads + qhead) * HEAD + (lane >> 4) * 16;
#pragma unroll
    for (int g = 0; g < NG; ++g) {
      const unsigned int *w = (const unsigned int *)(qrow + g * 32);
      unsigned int w0 = 0, w1 = 0, w2 = 0, w3 = 0;
      if (qvalid) { w0 = w[0]; w1 = w[1]; w2 = w[2]; w3 = w[3]; }
      int2_t lo_, hi_;
      lo_[0] = (int)pack_lo(w0, w1); lo_[1] = (int)pack_lo(w2, w3);
      hi_[0] = (int)pack_hi(w0, w1); hi_[1] = (int)pack_hi(w2, w3);
      sQ[rt][0][g][lane] = lo_;                    // QP == 1: the low nibble IS the int4 code
      if constexpr (QP == 2) sQ[rt][1][g][lane] = hi_;
    }
  }

  // ---- tile range of this segment: [first tile a window can reach, causal prefix of the block's
  //      last query); nseg segments share it. An empty segment writes a neutral partial so the
  //      reduce (which counts segments from seq_len alone) never reads garbage. ----
  const int max_prefix = causal ? min(seq_len, context_len + (qb_local + 1) * block_q) : seq_len;
  const int num_tiles = (max_prefix + TILE - 1) / TILE;
  const int tile_lo = window > 0 ? max(0, context_len + qb_local * block_q - window + 1) / TILE : 0;
  const int tiles_per_seg = (num_tiles - tile_lo + nseg - 1) / nseg;
  const int t_begin = tile_lo + seg * tiles_per_seg, t_end = min(t_begin + tiles_per_seg, num_tiles);

  // one LDS tile buffer; the NEXT tile is prefetched into registers during this tile's math and
  // parked after it (two barriers per tile)
  __shared__ __align__(16) unsigned char sK[NH][TG * KSTR];   // [group][token][PACKED + pad]
  __shared__ __align__(16) unsigned char sV[NH][PACKED * TG]; // [group][dim-pair][16 tokens]
  __shared__ __align__(16) float sKs[NH][NS * TG];            // [group][scale group][16 tokens], fp32
  __shared__ __align__(16) float sVl[TILE];                   // V scale * lift, per token of the tile
  __shared__ float sUnlift;                                   // max V scale / 255 / 16
  __shared__ __align__(16) float sAlpha[NWAVES][16];
  __shared__ __align__(16) float sBeta[NWAVES][16];        // KV4H_PREF 1: per wave, beta of its 16 queries                        // per wave: alpha of its 16 queries

  float m_run = -INFINITY, l_run = 0.f;
  float acc[DT][8];
#pragma unroll
  for (int d = 0; d < DT; ++d)
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[d][j] = 0.f;

  const unsigned char *hcache = cache + (size_t)kvh * cache_hstride;
  const long kreg = (long)block_size * TPT;
  const int half = lane >> 4;            // 0: tokens 0..7 of a 16-token group, 1: tokens 8..15
  const int jbase = half * 8;

  // ---- staging: TPG threads per 16-token group, each fetches KCPT x 16 B of K and VCPT x 16 B
  //      of V nibbles; the first NS*16 threads of a group its K scales (one byte each), wave 1
  //      all TILE V scales (lane l: tokens l, l + 32, ..). Fetch into registers first (in flight
  //      during the previous tile's math), then park. ----
  const int sh = tid / TPG, si = tid % TPG;
  constexpr int KSPT = (NS * TG + TPG - 1) / TPG;             // K scale bytes per staging thread
  uint4_t fk[KCPT], fv[VCPT];
  unsigned int fs[KSPT], fvs[VSPL];
  // (block index, token within block) of the next tile's NH 16-token groups, advanced by TILE
  // per fetch instead of dividing by the (runtime) block size every time
  int bi[NH], tin[NH];
  bi[0] = (t_begin * TILE) / block_size; tin[0] = (t_begin * TILE) % block_size;
#pragma unroll
  for (int h = 1; h < NH; ++h) {
    bi[h] = bi[h - 1]; tin[h] = tin[h - 1] + TG;
    if (tin[h] >= block_size) { tin[h] -= block_size; ++bi[h]; }
  }
  auto group_ptr = [&](int tj, int h) -> const unsigned char * {   // h: a runtime group index
    const int t = tj * TILE + h * TG;
    if (t >= seq_len) return nullptr;
    int b = bi[0], ti = tin[0];                                      // selects, not a runtime-indexed array
#pragma unroll
    for (int hh = 1; hh < NH; ++hh) if (h == hh) { b = bi[hh]; ti = tin[hh]; }
    const int blk = block_table[seq * bt_stride + b];
    return hcache + (size_t)blk * cache_bstride + (size_t)(ti / TG) * REG;
  };
  auto fetch = [&](int tj) {                                  // tiles are fetched in order
#pragma unroll
    for (int c = 0; c < KCPT; ++c) fk[c] = uint4_t{0, 0, 0, 0};
#pragma unroll
    for (int c = 0; c < VCPT; ++c) fv[c] = uint4_t{0, 0, 0, 0};
#pragma unroll
    for (int c = 0; c < KSPT; ++c) fs[c] = 0;
#pragma unroll
    for (int c = 0; c < VSPL; ++c) fvs[c] = 0;
#ifndef ABL_NOSTAGE
    const unsigned char *kg = group_ptr(tj, sh);
    if (kg) {
#pragma unroll
      for (int c = 0; c < KCPT; ++c) {
        const int ci = si + c * TPG;
        if (ci < TG * KCH) fk[c] = *(const uint4_t *)(kg + K_AREA + (ci / KCH) * PACKED + (ci % KCH) * 16);
      }
#pragma unroll
      for (int c = 0; c < VCPT; ++c) {
        const int ci = si + c * TPG;
        if (ci < PACKED) fv[c] = *(const uint4_t *)(kg + kreg + V_SCALES + ci * 16);
      }
#pragma unroll
      for (int c = 0; c < KSPT; ++c) {                               // K scale: group ci/16, token ci%16
        const int ci = si + c * TPG;
        if (ci < NS * TG) fs[c] = kg[ci];
      }
    }
    if (wave == 1) {
#pragma unroll
      for (int c = 0; c < VSPL; ++c) {
        const int t = lane + c * 32;                                  // token of the tile
        // only real tokens: bytes past seq_len (a reused block's stale tokens) would set the
        // tile's V lift and cost every real token P resolution
        if (t < TILE && tj * TILE + t < seq_len) {
          const unsigned char *vg = group_ptr(tj, t / TG);
          if (vg) fvs[c] = vg[kreg + (t % TG)];
        }
      }
    }
#endif
#pragma unroll
    for (int h = 0; h < NH; ++h) { tin[h] += TILE; while (tin[h] >= block_size) { tin[h] -= block_size; ++bi[h]; } }
  };
  auto park = [&]() {
#pragma unroll
    for (int c = 0; c < KCPT; ++c) {
      const int ci = si + c * TPG;
      if (ci < TG * KCH) *(uint4_t *)&sK[sh][(ci / KCH) * KSTR + (ci % KCH) * 16] = fk[c];
    }
#pragma unroll
    for (int c = 0; c < VCPT; ++c) {
      const int ci = si + c * TPG;
      if (ci < PACKED) *(uint4_t *)&sV[sh][ci * 16] = fv[c];
    }
#pragma unroll
    for (int c = 0; c < KSPT; ++c) {
      const int ci = si + c * TPG;
      if (ci < NS * TG) sKs[sh][ci] = e4m3_to_f32(fs[c]);
    }
    if (wave == 1) {
      float vs[VSPL], vmax = 0.f;
#pragma unroll
      for (int c = 0; c < VSPL; ++c) { vs[c] = e4m3_to_f32(fvs[c]); vmax = fmaxf(vmax, vs[c]); }
      vmax = wave_max32(vmax);
      const float lift = vmax > 0.f ? 255.f / vmax : 0.f;
#pragma unroll
      for (int c = 0; c < VSPL; ++c) if (lane + c * 32 < TILE) sVl[lane + c * 32] = vs[c] * lift;
      if (lane == 0) sUnlift = vmax * (1.f / 255.f / 16.f);
    }
  };

  if (t_begin < t_end) { fetch(t_begin); park(); }
  __syncthreads();                                            // also publishes sQ

  for (int tj = t_begin; tj < t_end; ++tj) {
    const int t0 = tj * TILE;
    if (tj + 1 < t_end) fetch(tj + 1);                        // in flight during this tile's math

    // ---- S^T for the tile's NH 16-token groups: per scale group, four 32-dim WMMAs chained ----
    float s[NH][8];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
      const unsigned char *krow = &sK[h][(lane & 15) * KSTR + half * 8];
      int8v c[NS][QP];
#pragma unroll
      for (int sg = 0; sg < NS; ++sg) {
#pragma unroll
        for (int pl = 0; pl < QP; ++pl) c[sg][pl] = int8v{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int gi = 0; gi < NG / NS; ++gi) {
          const int g = sg * (NG / NS) + gi;
          int2_t a;
          a[0] = *(const int *)(krow + g * 16);
          a[1] = *(const int *)(krow + g * 16 + 4);
          const int2_t ql_ = sQ[rt][0][g][lane];
#ifndef ABL_NOQK
          if constexpr (QP == 2) {
            const int2_t qh_ = sQ[rt][1][g][lane];
            c[sg][0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, qh_, c[sg][0], false);
            c[sg][1] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, false, ql_, c[sg][1], false);
          } else {                                   // int4 query: the plane is signed
            c[sg][0] = __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, ql_, c[sg][0], false);
          }
#else
          c[sg][0][0] += a[0] + ql_[0];
#endif
        }
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) s[h][j] = 0.f;
#pragma unroll
      for (int sg = 0; sg < NS; ++sg) {
        const float4 k0 = *(const float4 *)&sKs[h][sg * TG + jbase], k1 = *(const float4 *)&sKs[h][sg * TG + jbase + 4];
        const float ks[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          int dot;
          if constexpr (QP == 2) dot = (c[sg][0][j] << 4) + c[sg][1][j]; else dot = c[sg][0][j];
          s[h][j] = fmaf((float)dot, ks[j], s[h][j]);
        }
      }
    }
    // ---- mask (only tiles touching the causal / length boundary), online softmax over the
    //      tile's 32 tokens (16 per lane); scores are scaled inside the exp argument ----
    const bool interior = t0 + TILE <= seq_len && t0 + TILE - 1 <= qlast_lo &&
                          (window <= 0 || (qabs_hi - t0 < window && (causal || t0 + TILE - 1 - qabs_lo < window)));
    if (!interior) {
#pragma unroll
      for (int h = 0; h < NH; ++h)
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int t = t0 + h * TG + jbase + j;
          bool ok = t < seq_len && t <= qlast;
          if (window > 0) ok = ok && qabs - t < window && (causal || t - qabs < window);
          s[h][j] = ok ? s[h][j] : -INFINITY;
        }
    }
    float mx = -INFINITY;
#pragma unroll
    for (int h = 0; h < NH; ++h)
#pragma unroll
      for (int j = 0; j < 8; ++j) mx = fmaxf(mx, s[h][j]);
    mx = wave_max16(mx);
    const float m_new = fmaxf(m_run, mx * rs);
    const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
    const float alpha = exp2_raw(m_run - m_use);
#if KV4H_PREF
    const float m_tile = mx * rs;                              // this tile's max: P's reference
    const float m_ref = (m_tile == -INFINITY) ? 0.f : m_tile;
    const float beta = (m_tile == -INFINITY) ? 0.f : exp2_raw(m_tile - m_use);
#else
    const float m_ref = m_use;
#endif
    float lsum = 0.f;
#pragma unroll
    for (int h = 0; h < NH; ++h)
#pragma unroll
      for (int j = 0; j < 8; ++j) {
#ifndef ABL_NOEXP
        s[h][j] = exp2_raw(fmaf(s[h][j], rs, -m_ref));          // s is P from here on
#else
        s[h][j] = fmaf(s[h][j], rs, -m_use) * 0.001f + 0.5f;
#endif
        lsum += s[h][j];
      }
    lsum = wave_sum16(lsum);
#if KV4H_PREF
    l_run = l_run * alpha + lsum * beta;
#else
    l_run = l_run * alpha + lsum;
#endif
    m_run = m_new;
    // ---- P as uint8 (times the pre-lifted V token scale), 8 tokens per lane and half: the A
    //      operand of the P . V product ----
    int2_t pb[NH];
#pragma unroll
    for (int h = 0; h < NH; ++h) {
      const float4 v0 = *(const float4 *)&sVl[h * TG + jbase], v1 = *(const float4 *)&sVl[h * TG + jbase + 4];
      unsigned int w0 = 0, w1 = 0;
      w0 = __builtin_amdgcn_cvt_pk_u8_f32(s[h][0] * v0.x, 0u, w0);
      w0 = __builtin_amdgcn_cvt_pk_u8_f32(s[h][1] * v0.y, 1u, w0);
      w0 = __builtin_amdgcn_cvt_pk_u8_f32(s[h][2] * v0.z, 2u, w0);
      w0 = __builtin_amdgcn_cvt_pk_u8_f32(s[h][3] * v0.w, 3u, w0);
      w1 = __builtin_amdgcn_cvt_pk_u8_f32(s[h][4] * v1.x, 0u, w1);
      w1 = __builtin_amdgcn_cvt_pk_u8_f32(s[h][5] * v1.y, 1u, w1);
      w1 = __builtin_amdgcn_cvt_pk_u8_f32(s[h][6] * v1.z, 2u, w1);
      w1 = __builtin_amdgcn_cvt_pk_u8_f32(s[h][7] * v1.w, 3u, w1);
      pb[h][0] = (int)w0;
      pb[h][1] = (int)w1;
    }
    const float unlift = sUnlift;
#if KV4H_PREF
    // the P.V rows j are queries jbase + j, whose beta lives in lane jbase + j
    if (half == 0) sBeta[wave][lane] = beta;
    __builtin_amdgcn_wave_barrier();
    const float4 be0 = *(const float4 *)&sBeta[wave][jbase];
    const float4 be1 = *(const float4 *)&sBeta[wave][jbase + 4];
    const float bu[8] = {be0.x * unlift, be0.y * unlift, be0.z * unlift, be0.w * unlift,
                         be1.x * unlift, be1.y * unlift, be1.z * unlift, be1.w * unlift};
    __builtin_amdgcn_wave_barrier();
#endif
    // the P.V accumulator's 8 values per lane belong to 8 different QUERIES (rows jbase + j),
    // whose alpha lives in lane jbase + j: exchange through LDS, only when some row's max moved
    if (__any(alpha != 1.f)) {
      if (half == 0) sAlpha[wave][lane] = alpha;
      __builtin_amdgcn_wave_barrier();
      const float4 al0 = *(const float4 *)&sAlpha[wave][jbase];
      const float4 al1 = *(const float4 *)&sAlpha[wave][jbase + 4];
      const float alpha_j[8] = {al0.x, al0.y, al0.z, al0.w, al1.x, al1.y, al1.z, al1.w};
#pragma unroll
      for (int dt = 0; dt < DT; ++dt)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[dt][j] *= alpha_j[j];
      __builtin_amdgcn_wave_barrier();
    }
    const unsigned vsh = dh ? 0u : 4u;                        // wave-uniform: shift to the high half of the byte
    // ---- P . V for this wave's DH dims: DT dim tiles, the tile's NH 16-token groups chained
    //      (iu8: V's nibble sits in the high half of its byte = 16 * v as int8) ----
#pragma unroll
    for (int dt = 0; dt < DT; ++dt) {
      int8v d = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
      for (int h = 0; h < NH; ++h) {
        const unsigned char *vb = &sV[h][(dt * 16 + (lane & 15)) * TG + jbase];
        const unsigned int w0 = *(const unsigned int *)vb, w1 = *(const unsigned int *)(vb + 4);
        int2_t b;
        b[0] = (int)((w0 << vsh) & 0xF0F0F0F0u); b[1] = (int)((w1 << vsh) & 0xF0F0F0F0u);
#ifndef ABL_NOPV
        d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, pb[h], true, b, d, false);
#else
        d[0] += b[0] + pb[h][0];
#endif
      }
#pragma unroll
#if KV4H_PREF
      for (int j = 0; j < 8; ++j) acc[dt][j] = fmaf((float)d[j], bu[j], acc[dt][j]);
#else
      for (int j = 0; j < 8; ++j) acc[dt][j] = fmaf((float)d[j], unlift, acc[dt][j]);
#endif
    }
#ifndef ABL_NOSYNC
    __syncthreads();                                          // everyone done reading the tile
#endif
#ifndef ABL_NOPARK
    if (tj + 1 < t_end) park();
#endif
#ifndef ABL_NOSYNC
    __syncthreads();
#endif
  }

  // ---- epilogue: lane holds dim dh*DH + 16*dt + lane%16 of queries (rows) half*8 + j ----
  const int dcol = dh * DH + (lane & 15);
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int r = rt * 16 + half * 8 + j;                // query row of the block
    const int qp = qb_local * block_q + r / nqpkv;
    const int qh = kvh * nqpkv + r % nqpkv;
    if (r >= block_q * nqpkv || qp >= qlen || qh >= num_q_heads) continue;
    const size_t tokidx = (size_t)(q_start + qp) * num_q_heads + qh;
    // the running max / sum of query r live in the lanes whose column is r % 16
    const float mq = __shfl(m_run, (r & 15), 32);
    const float lq = __shfl(l_run, (r & 15), 32);
    if (nseg == 1) {
      const float inv = lq > 0.f ? 1.f / lq : 0.f;
      __bf16 *o = out + tokidx * HEAD;
#pragma unroll
      for (int dt = 0; dt < DT; ++dt) o[dt * 16 + dcol] = (__bf16)(acc[dt][j] * inv);
    } else {
      float *so = segm_out + (tokidx * nseg + seg) * HEAD;
#pragma unroll
      for (int dt = 0; dt < DT; ++dt) so[dt * 16 + dcol] = acc[dt][j];
      if (dcol == 0) {
        segm_max[tokidx * nseg + seg] = mq;
        segm_sum[tokidx * nseg + seg] = lq;
      }
    }
  }
}

// ---- host ---------------------------------------------------------------------------------------

static void launch(int head, int tile, uintptr_t q, uintptr_t q_descale, uintptr_t cache, uintptr_t block_table,
                   uintptr_t seq_lens, uintptr_t cu_seqlens_q, uintptr_t segm_out,
                   uintptr_t segm_max, uintptr_t segm_sum, uintptr_t out, float scale_log2,
                   int num_q_blocks, int num_kv_heads, int num_q_heads, int nqpkv, int block_q,
                   long cache_bstride, long cache_hstride, int block_size, int bt_stride,
                   int nseg, int num_seqs, int causal, int window, uintptr_t stream) {
  if (block_size <= 0 || block_size % TG) throw std::runtime_error("kv4h: block_size must be a positive multiple of 16");
  if (block_q <= 0 || nqpkv <= 0) throw std::runtime_error("kv4h: block_q and nqpkv must be positive");
  if (num_seqs <= 0 || num_q_blocks <= 0 || num_kv_heads <= 0 || nseg <= 0) return;   // nothing to do
  if (nseg > 1 && !(segm_out && segm_max && segm_sum)) throw std::runtime_error("kv4h: nseg > 1 needs the segment buffers");
  if (nqpkv * block_q > KV4H_NRT * 16) throw std::runtime_error("kv4h: block_q * nqpkv must fit the program's rows");
  if (nseg == 1 && !out) throw std::runtime_error("kv4h: nseg == 1 needs out");
  dim3 grid(num_q_blocks, num_kv_heads, nseg), block(KV4H_NRT * 64);
  auto args = [&](auto kern) {
    hipLaunchKernelGGL(kern, grid, block, 0, (hipStream_t)stream,
                       (const int8_t *)q, (const float *)q_descale, (const unsigned char *)cache,
                       (const int *)block_table, (const int *)seq_lens, (const int *)cu_seqlens_q,
                       (float *)segm_out, (float *)segm_max, (float *)segm_sum, (__bf16 *)out,
                       scale_log2, num_q_heads, nqpkv, block_q, cache_bstride, cache_hstride,
                       block_size, bt_stride, nseg, num_seqs, causal, window);
  };
#define KV4H_K(H, NHX) kv4h_attention<H, NHX, KV4H_NRT, KV4H_QP, KV4H_NS>
  if (head == 256 && tile == 16) args(KV4H_K(256, 1));
  else if (head == 256 && tile == 32) args(KV4H_K(256, 2));
  else if (head == 256 && tile == 64) args(KV4H_K(256, 4));
  else if (head == 128 && tile == 16) args(KV4H_K(128, 1));
  else if (head == 128 && tile == 32) args(KV4H_K(128, 2));
  else if (head == 128 && tile == 64) args(KV4H_K(128, 4));
  else throw std::runtime_error("kv4h: head size must be 128 or 256, tile 16, 32 or 64");
#undef KV4H_K
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("kv4h launch: ") + hipGetErrorString(e));
}

#ifndef KV4H_MODNAME
#define KV4H_MODNAME radiance_kv4h
#endif
PYBIND11_MODULE(KV4H_MODNAME, m) {        // one module name per knob build (several live in one process)
  m.def("launch", &launch, "KV4H split-KV attention (gfx1201, int4 / int8 WMMA), head 128 or 256, tile 16 / 32 / 64");
  m.attr("TILES") = py::make_tuple(16, 32, 64);
  m.attr("BLOCK_M") = KV4H_NRT * 16;
  m.attr("ROW_TILES") = KV4H_NRT;
  m.attr("Q_PLANES") = KV4H_QP;
  m.attr("K_SCALE_GROUPS") = KV4H_NS;
  m.attr("P_REF") = KV4H_PREF;
}
