// kv4i_unpack.h -- the int4 half of a libr4d prefill read of the TetraAttention (KV4) cache.
//
// Plain C++, no HIP types and no intrinsics, so a CPU test can compile it with any host compiler and
// check every formula against the torch oracle (sly/tests/kv4/test_kv4i_unpack_cpu.py). The HIP
// wiring (the LDS staging of the packed cells, the v_perm_b32 pair split, the WMMA operands) is NOT
// here: see docs/R4D-INT4-PREFILL.md for what is still to be written and why it needs the GPU gate.
//
// The cache cell (sly/kv4/radiance_kv4h_format.py, the oracle): nibbles are two's-complement int4
// (-8..7); byte p of a token holds dims 2p (low nibble) and 2p+1 (high nibble) on the K side, and
// dims d / d+PACKED on the V side; every (token, K group) and every token of V carries one e4m3 scale.
#pragma once
#include <cstdint>

namespace kv4i {

// Signed low / high nibble of one packed byte. `b & 0xF` alone is unsigned: -8 must stay -8.
static inline int low(uint8_t b)  { const int v = b & 0xF;        return v >= 8 ? v - 16 : v; }
static inline int high(uint8_t b) { const int v = (b >> 4) & 0xF; return v >= 8 ? v - 16 : v; }

// int4 code -> e4m3 bits, exact. |c| <= 8 and e4m3 has three mantissa bits, so every code is
// representable: no rounding happens here, which is what makes the LUT the cheapest form.
//   0 -> 0x00, 1 -> 0x38, 2 -> 0x40, 3 -> 0x44, 4 -> 0x48, 5 -> 0x4A, 6 -> 0x4C, 7 -> 0x4E
//   -1 -> 0xB8 ... -7 -> 0xCE, -8 -> 0xD0
static constexpr uint8_t CODE_E4M3[16] = {0x00, 0x38, 0x40, 0x44, 0x48, 0x4A, 0x4C, 0x4E,
                                          0xD0, 0xCE, 0xCC, 0xCA, 0xC8, 0xC4, 0xC0, 0xB8};
static inline uint8_t code_e4m3(int c) {
    // two's complement keeps -8..7 in 0..15: -8 -> 8, -1 -> 15, 0 -> 0, 7 -> 7 (the table's order)
    return CODE_E4M3[uint8_t(c) & 0xF];
}

// float -> e4m3 bits, round-to-nearest-even, saturating to +-448, subnormals in 2^-9 steps, NaN
// stays NaN. This is what torch's .to(torch.float8_e4m3fn) does, and it is the second rounding of
// variant A of the design note (the int4 code is scaled by its token scale before it is stored as a
// byte), so the CPU test compares it against torch code point by code point.
static inline uint8_t rne(float v) {
    const uint32_t u = [] (float f) { union { float f; uint32_t u; } x; x.f = f; return x.u; }(v);
    const uint8_t sign = (u >> 31) ? 0x80 : 0x0;
    if ((u & 0x7f800000u) == 0x7f800000u) {          // inf / NaN: e4m3 has one NaN (0x7F) and no inf
        return (u & 0x007fffffu) ? 0x7F : (sign | 0x7E);   // saturate inf to max, keep NaN
    }
    const int e = int((u >> 23) & 0xFF) - 127;        // unbiased exponent
    const uint32_t m = u & 0x007fffffu;
    if (e > 8) return sign | 0x7E;                    // >= 2^9 saturates to 448
    int q;                                            // quantised number of 2^-9 steps
    if (e >= -6) {                                    // normal range, 2^-6 .. 2^8
        const int shift = 20;                         // 23 - 3 mantissa bits
        const uint32_t mant3 = ((m | 0x00800000u) >> shift) & 7u;   // 3 mantissa bits, leading 1 dropped
        q = int(mant3 | uint32_t((e + 7) << 3));                   // bias 7, in 4 bits
        const uint32_t rest = m & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rest > half || (rest == half && (q & 1))) q += 1;             // ties to even, carries up
    } else {                                          // subnormal (2^-9 steps) or zero
        const int shift = 20 + (-6 - e);              // more right-shifting, fewer real bits
        if (shift > 31) return sign;                  // far below 2^-10: rounds to zero
        const uint32_t mant = m | 0x00800000u;
        q = int(mant >> shift);
        const uint32_t rest = mant & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rest > half || (rest == half && (q & 1))) q += 1;
    }
    if (q >= 0x7F) return sign | 0x7E;                // saturate (0x7F is NaN)
    return uint8_t(sign | q);
}

// Variant A: the byte the kernel stages is code * token scale, rounded once. `v > 0` only; a scale of
// 0 collapses the whole row to zero, which the oracle also does (amax 0 -> scale 0).
static inline uint8_t code_e4m3_scaled(int c, float s) { return rne(float(c) * s); }

// e4m3 bits -> the value the byte carries, as the multiply the kernel's score path expects. Only the
// 16 code points are decoded exactly here; used by the CPU test's inverse check.
static inline float code_value(int c) { const float v = float(c); return v; }

}  // namespace kv4i
