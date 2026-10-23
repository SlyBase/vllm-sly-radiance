# Reading the int4 (KV4) cache in the libr4d prefill attention kernel — design

Status: design + CPU-verified helpers. No kernel is wired, nothing is dispatched, no speed is claimed.
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

TODO, all of it needing the GPU gate before it is worth writing:

- the LDS staging of the packed cells (the byte-tile strides `KSTR8`/`VSTR8` do not match the KV4
  region layout; a tile is `16 tokens x 128 B` inside a `block_size x 136 B` region),
- the `v_perm_b32` pair split that turns one packed byte into two e4m3 bytes in element order,
- the V dim-major → B-operand gather (16 consecutive tokens of one dim are contiguous today, which
  the iu8 product wants; the byte legs assume the token-major layout),
- the K scale-group chain (1, 2 or 4 groups per token) and the `O_KV4I` OPT bit that gates all of it
  off by default.

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
