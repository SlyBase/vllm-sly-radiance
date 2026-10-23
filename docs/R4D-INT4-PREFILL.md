# Reading the int4 (KV4) cache in the libr4d prefill attention kernel — design

Status: kernel compiled and gated off (`O_KV4I`, `sly/r4d/r4d_kv4i.patch`), route written and gated off
(`R4D_KV4I=1`), CPU-verified end to end. **Kernel microbench on the R9700: the int4 libr4d kernel is 1.01x of
the fp8 libr4d kernel at past 49152** (TetraAttention is 1.81x), so the go/no-go below passes and the port
goes ahead. No engine-level number and no accuracy gate yet — both need the next GPU window.
Measured reason for the work, at 210 W and 300 W, is in
[BENCHMARKS.md](BENCHMARKS.md#tetraattention-kv4-prefill-gap-2026-10-23).

## The gap

Whole engine, prefill tok/s, KV4 (`TETRA`) against FP8 KV (`R4D_HYBRID`), same image 1.8.0:

| window | 1.5k | 5.9k | 11.8k | 23.5k | 47k |
|---|--:|--:|--:|--:|--:|
| 300 W | −4.5 % | −5.8 % | −8.3 % | −11.3 % | −15.9 % |
| 210 W | −1.4 % | −2.7 % | −4.5 % | −7.3 % | −11.4 % |

Kernel microbench at the serve shapes (4096-token chunk, causal, 24 q / 4 kv heads, head 256),
`Tetra back end / libr4d` time: 1.91x at past 0, 1.98x at 4k, 1.77x at 16k, 1.80x at 32k, 1.81x at 49k.
The backend's rotation and quantization cost 0.1–0.5 ms per chunk, so 1.7–1.85x of that is the
kernel itself. The 1.8.0 fp8 schedule ideas do not transfer (pipe 1 and 3 neutral, pipe 5 40–55 %
worse, tile 16 worse than tile 64) — the gap is the kernel, not its schedule.

## What the kernel has to read

`sly/kv4/radiance_kv4h_format.py` is the layout oracle. Per (block, kv head) region, groups of 16
tokens, K then V, 136 bytes per token and side at head 256:

- **K**: `[2 x 16 e4m3 scales][96 B spare][16 tokens x 128 B nibbles]`; byte `p` of a token holds dims
  `2p` (low nibble) and `2p+1` (high nibble).
- **V**: `[16 e4m3 scales][128 dim pairs x 16 tokens][112 B spare]`, dim-major, so a B operand of the
  P·V product is one contiguous 8- or 16-byte load.
- Nibbles are two's-complement int4 (−8..7, `RADIANCE_KV4H_QMIN=-7` optional), one e4m3 scale per
  token on each side (K may split the 256 dims into 1, 2 or 4 scale groups).
- K and V are Walsh-Hadamard rotated before quantization. The TetraAttention back end rotates the
  query and un-rotates the output, so **the kernel sees plain K/V** — general linear algebra, nothing
  rotation-specific has to be re-derived for libr4d.

## Two ways to get a usable operand

The int4 WMMA Tetra uses would be the natural fit, but libr4d's prefill kernel is built around the
8-bit legs (`O_QK8`, `O_PV8`) where K and V sit in LDS as **bytes**. So the port is
"nibble → e4m3 byte", and there are two exactness/cost pairs:

**Variant A — rescale at unpack (recommended).** The byte staged in LDS is
`rne_e4m3(code * s_token)`: one multiply and one round per element, done once per tile, and then the
score path is unchanged from the fp8 kernel (the launch constant carries no per-token factor).
Cost: a second rounding, e4m3 has three mantissa bits, so the relative error of the byte is up to
2^-4 on top of the int4 error. `sly/kv4/kv4i_unpack.h` implements this (`code_e4m3_scaled`) and the
CPU test proves it equals `torch.float8_e4m3fn` bit for bit over all 16 codes x 10 scales.

**Variant B — fold the scale into the score.** Stage the bare code (a 16-entry LUT, exact:
`code_e4m3`) and multiply `s_token` into `q·k` before the max/exp2 update. Exact for the int4 part,
but it is one extra multiply per score element inside the softmax loop, on a kernel that is already
VALU issue bound — that is the fp8 kernel's whole lesson (`O_CFOLD`/`O_MAX3` cut VALU and won
nothing, see NOT-ADOPTED). It also breaks the per-launch `mul = scale * kdesc * log2e` shape, so the
online max has to become per-key.

V is the same decision, applied before P is packed to e4m3 for the `iu8` P·V product; note the
denominator sums unquantized p, and e4m3's 448 (2^8.8) tightens `PGROW` to 8 octaves.

**What is deliberately not in the port:** the store path. KV blocks are written by
`kv4h_store`/Triton today and stay there; only the attention read changes.

## Routing

`radiance_kv4h_attn.py::TetraAttentionImpl.forward` is the only place that decides. Prefill launches
with more than 512 new tokens go to libr4d (the same threshold idea as `R4D_HYBRID`, which keeps the
libr4d prefill gain and AITER's decode), everything else stays: decode and verify stay on
TetraAttention (it is faster than the fp8 path there: 33.5 vs 35.0 ms per step at 18k, 1 stream), and
the DFlash drafter stays on its head-128 TetraAttention kernel, which this port does not cover. The
KV dtype is unchanged (`--kv-cache-dtype int4_per_token_head`), so no vLLM core code is touched: the
backend selects the kernel per launch, as the hybrid already does for fp8.

## What is written and what is TODO

Written and CPU-verified (`sly/tests/kv4/test_kv4i_unpack_cpu.py`, host compiler, no GPU):
signed nibble split, the exact 16-code LUT, e4m3 RNE encoding/saturation, and the scaled variant A
byte.

Written as of the `O_KV4I` patch (compile-gated, never run on a GPU): the LDS staging, the nibble → e4m3 unpack, the V
dim-major gather and the K group-256 case. `r4d.attn_prefill_h256_gqa6_int4kv` (needs `R4D_KV4I=1`) runs the
double-buffered byte legs; only `fetchKV`/`storeKV` differ (`sly/r4d/r4d_kv4i.patch`). What it does:

- the block table holds 16-token **group** ids (block × 104 + group), strides are bytes; one scalar base per wave plus a
  32-bit lane offset;
- K: a thread takes 16 dims of one key = 8 nibble bytes + that key's scale; unpack is `bfe` (signed nibble) +
  `v_cvt_f32_i32` + `v_mul_f32` by the scale + `v_cvt_pk_fp8_f32` = 3.5 VALU per element, one rounding (the exact-code
  LUT would still need byte → f32 → mul → e4m3 for variant A, so it is not cheaper);
- V: a thread takes byte row d of 8 tokens (low nibble → row d, high → row d + 128) and those 8 token scales, held in
  SGPRs (a VGPR copy spilled); tokens ≥ ctx get scale 0 (stale NaN scales must not reach P·V);
- QK, softmax, PV and the `O_QF8` query path are the shipped `O_QK8|O_PV8` code.

Compile gate (`ci/kv4i_compile_gate.py`): all 30 shipped kernels of the attention unit instruction-identical, the new
tile 16/32/48 instances spill-free at 211–216 VGPR. CPU proof of the staged bytes:
`sly/tests/kv4/test_kv4i_stage_cpu.py`. Not covered: hardware rounding of `v_cvt_pk_fp8_f32` against torch for the
scaled products, accuracy against the fp32 reference, and any speed.

Still TODO, needing the GPU gate: the `--kv4i` arm of `sly/bench_attn_r4d_vs_tetra.py` (go/no-go below), K scale groups
128 / 64, block sizes other than 1664, and the engine routing in `TetraAttentionImpl.forward`.

## Accuracy gate (decides the port, not the speed)

Measured against an fp32 SDPA reference on random K/V, relative L2: libr4d fp8 0.053–0.059,
TetraAttention 0.186–0.238. int4 K/V plus e4m3 Q and e4m3 P adds the Q/P error on top of the int4
error — that is a real quality step beyond today's KV4. The gate is
`sly/kv4/kl_score.py` (KL against a bf16 KV reference) **and** GSM8K-200 against
`ci/accept/baselines/`. Outside the band, the port is rejected regardless of its speed.

## Payoff ceiling and the go/no-go

Ceiling: +11 % prefill at 47k (210 W) if the int4 read ever reaches fp8-kernel parity; less in
reality, because unpacking is VALU work on a VALU-bound kernel and variant A adds a rounding. Short
prompts gain nothing (−1.4 % at 1.5k today).

Go/no-go, one measure: extend `sly/bench_attn_r4d_vs_tetra.py` so the int4 libr4d kernel is timed
against the fp8 libr4d kernel at the same shapes. **If the int4 kernel is within ~1.15x of the fp8
one at past 49152, the port goes ahead; above that, the unpack cost eats the gain and the work stops
there.** Everything above that line is a plan, not a result.

## Query fold: FOLDQ must stay OFF for the int4 route (measured 2026-10-23)

`O_FOLDQ` multiplies `scale / HEAD * log2e` into the query **before** the e4m3 pack. That fold is right
for the fp8 path (a ~0.1-0.5 query sits in e4m3's normal range) and wrong here: the query is stored as
`rotate(q) / q_scale`, the kernel multiplies `q_scale` back in f32, and the extra `scale / 256` then
pushes **98 % of the elements under 2^-6** — e4m3 subnormal, three bits or fewer, flushing to 0 at 2^-9.

Measured on three dumped launches (python text, ctx 20445, q 4089, `R4D_KV4I_DUMP`, `sly/kv4/kv4i_offline.py`,
fp32 truth rebuilt from the int4 bytes):

| rounding alone (layer 47) | rel-L2 | | arm | rel-L2 L47 | rel-L2 L3 |
|---|--:|---|---|--:|--:|
| Q folded -> e4m3 | 4.5 % | | fold on (as shipped) | 6.2 % | 3.8 % |
| K -> e4m3 (variant A) | 3.9 % | | fold off | **4.6 %** | **2.7 %** |
| P -> e4m3 | 1.2 % | | | | |
| V -> e4m3 | 0.9 % | | | | |

The route output also **shrinks** (`|out| / |truth|` 0.988-0.991 against Tetra's 0.997-0.999) — a
systematic loss of attention mass, which is what compounds over 16 full-attention layers instead of
averaging out. With the fold off the shrink halves (signed radial error -0.0126 -> -0.0076 in layer 47).

The launcher therefore builds `(P_OPT & ~O_FOLDQ) | O_QK8 | O_PV8 | O_DB | O_KV4I`; the fold's job
(8 f32 multiplies per m-tile, `sc[e] = s[e] * mul`) is cheap and exact where it now happens.
