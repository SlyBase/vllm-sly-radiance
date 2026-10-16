# TetraAttention: 4-bit KV cache for the full-attention layers

TetraAttention is a 4-bit KV cache plus the attention kernel that reads it, built for RDNA4 (gfx1201).
K and V are Hadamard-rotated, stored as int4 with one e4m3 scale per token, and fed to the card's int4
and int8 WMMA instructions exactly as they sit in the cache. It is an **opt-in alternative to the fp8
cache**: ~1.75x the KV pool for slightly less precision, at equal or better speed.

```bash
KVCACHE_DTYPE=kv4 ./serve-mxfp4.sh     # = --kv-cache-dtype int4_per_token_head --attention-backend TETRA
```

`install_kv4.py` runs in the Dockerfile patch loop (and still in the launcher's container prelude). It copies the modules into site-packages, registers
the backends and prebuilds the kernel with the image's hipcc (~20 s). The DFlash drafter's attention
runs on TetraAttention too. The working name was KV4H, and it survives in three places:
- the files (`radiance_kv4h_*`);
- the knobs (`RADIANCE_KV4H_*`);
- the backend alias `KV4H`.

## Format

Each (kv head, token) cell is 272 bytes at head 256: 136 for K, 136 for V. Head 128, used by the
drafter, is 136 bytes, which is vLLM's own int4 cell size, so the pages line up. Cells are laid out per
(block, head) in groups of 16 tokens, matching the order the matrix instructions read them.
`radiance_kv4h_format.py` is the torch oracle, and `radiance_kv4h_store.py` writes the same bytes.

| side | codes | scale | layout per 16-token group |
|---|---|---|---|
| K | signed int4, -8..7 | one e4m3 per token | `[16 scales][spare][16 tokens x 128 bytes]`; byte p = dims 2p (low nibble) and 2p+1 (high nibble) |
| V | signed int4, -8..7 | one e4m3 per token | `[16 scales][128 dim pairs x 16 tokens][spare]`, dim-major |

- **Rotation:** K and V are Walsh-Hadamard rotated with a fixed sign vector before quantization. The
  backend rotates the query and un-rotates the output, so the kernel works on plain K/V.
- **No zero point:** the iu4 WMMA reads the nibble directly with its signed flag.
- **Scale choice:** the first scale is `0.8 x absmax / 7`, rounded up to e4m3. This clips the rare
  extreme value of a near-Gaussian rotated row. On the model's own K/V it moves the SQNR from 17.6 to
  19.2 dB, which is the value a 9-step scale search reaches, at no extra store cost.

## Kernel (`radiance_kv4h.hip`)

One program covers 64 query rows, one kv head and one KV segment. Tiles of 16, 32 or 64 tokens are
prefetched into registers during the previous tile's math, then parked in LDS. Per tile:

- **S^T = K Q^T** runs on `v_wmma_i32_16x16x32_iu4`. The K nibbles are the A operand as stored, and the
  query is int4. An int8 query, fed as two int4 planes, is optional.
- **Online softmax** works in exp2 with the scale folded into the fma. The mask is only applied on
  tiles that touch a boundary.
- **P V** runs on `v_wmma_i32_16x16x16_iu8`.
  - P is uint8, quantized against the tile's own max. The gap to the running max is carried as a
    per-query factor.
  - V's nibble is shifted into the high half of its byte, so it becomes a signed int8 worth 16 v.
- **Segments:** split-KV segments are merged by a Triton kernel that un-rotates straight into vLLM's
  output.
- **Head sizes:** 256 (target) and 128 (drafter). Sliding windows and non-causal attention follow
  vLLM's conventions.

The kernel is bound by VALU issue rate (softmax, scales, addressing), not by the matrix units. That is
why every per-element operation on the tile was removed: one scale per token, no unpacking, and no
masks on interior tiles. `probe_wmma.hip` pins the fragment layouts on the card.

## Knobs

Every parameter is set in three ways:
- a `kv4h_attention()` / `kv4h_store()` argument;
- a `RADIANCE_KV4H_*` environment default;
- a flag of `../bench_kv4_attn.py`.

`tune_kv4h.py` sweeps them and writes `kv4h_tune.json`, which the launcher reads per shape. An explicit
argument wins over the environment, which wins over the table, which wins over the built-in default.

| knob | default | |
|---|---|---|
| `_TILE` / `_PREFILL_TILE` | table, else 32 | tokens per tile: 16, 32 or 64 |
| `_ROWTILES` | table, else 4 | 16-row tiles per program: 2 or 4 |
| `_QPLANES` | 1 | query width: 1 = int4, 2 = int8 as two planes |
| `_K_GROUP` | 256 | K dims per scale: 256 = one per token; 128 or 64 (format knob: store and kernel) |
| `_SEGMENTS` | table / AITER | split count for decode and verify |
| `_PREF` | 1 | P reference: 1 = tile max, 0 = running max |
| `_SCALE_CLIP`, `_QMIN`, `_SCALE_STEPS` | 0.8, -8, 1 | store: first scale candidate, lowest code, scale search steps |
| `_TUNE` | `kv4h_tune.json` | knob table; `off` disables it |

## Accuracy against a bf16 KV cache

- **Method:** teacher-forced top-20 logprobs over five texts of 120,000 tokens each (two novels,
  kernel docs, C, Python). The reference is the same engine with `--kv-cache-dtype auto`.
- **Script:** `kl_score.py` scores one arm against the running server; `kl_compare.py` compares arms.
- **Noise floor:** a bf16 rerun is bit-identical.

| | KL | KL at 64-128k | top-1 agreement | perplexity |
|---|---|---|---|---|
| fp8 | 0.0123 | 0.0111 | 95.58 % | +0.06 % |
| TetraAttention | 0.0126 | 0.0118 | 95.32 % | +0.54 % |

- **Over context:** the error is flat from 4k to 128k tokens.
- **Task benchmark:** GSM8K-200 (cot zero-shot, greedy) is 0.80 against fp8's 0.79.
- **Retrieval:** a needle at 97k is found at depths of 10, 50 and 90 %.

## Performance

**Kernel, one R9700.** `../bench_kv4_attn.py`, 24 query / 4 kv heads, block 1664. Each figure is the
whole backend path relative to AITER's fp8 kernel (1x = equal; lower = faster):

| shape | 18k | 98k |
|---|---|---|
| verify-8, 1 / 4 sequences | 0.75x / 0.76x | 0.71x / 0.61x |
| decode, 1 / 4 sequences | 0.40x / 0.24x | 0.23x / 0.21x |
| prefill chunk of 3328 tokens at 3k context | 1.05x | |

**Engine, one R9700.** Production arguments, vision on, one GPU window (fp8, TetraAttention,
TetraAttention, fp8). Decode is the pure rate: `decode_delta.py`, 150 vs 450 greedy tokens on a cached
prefix.

| | fp8 | TetraAttention |
|---|---|---|
| KV pool, 262k max len | 342,221 tokens | **593,126 tokens** |
| decode at 18k, ms per step, 1 stream | 35.0 | **33.5** |
| decode at 18k, ms per step, 4 streams | 46.5 | **44.1** |
| cold prefill 8k / 24k / 85k, `--max-num-batched-tokens 3456` | 2.34 / 7.72 / 38.0 s | **2.27 / 7.47 / 36.7 s** |

## Limits

- **Prefix-cache granularity.** With half-size cells, vLLM sets the attention block to 1664 tokens
  (fp8: 880) to keep the attention page at least as large as the Mamba page. A prefix hit therefore
  recomputes up to 1,663 tokens instead of 879.
- **Prefill budget.** Keep `--max-num-batched-tokens` (`CHUNK`) at 3456 or more. Below that, the
  draft-slot reservation leaves less than two blocks per step, and prefill falls behind fp8.
- **Offload tiers.** A tier holds bytes in the format that wrote them. Never point a kv4 serve at a tier
  written by an fp8 one, or the reverse.
- **Hardware.** Measured on the R9700 only. The layout and the instructions are gfx12-specific.

## Files

| file | what |
|---|---|
| `radiance_kv4h.hip` | the attention kernel and its pybind launch |
| `radiance_kv4h_kernel.py` | build per knob combination, query quantization, launch, segment merge, tune table |
| `radiance_kv4h_format.py` / `radiance_kv4h_store.py` | layout oracle / Triton store |
| `radiance_kv4h_attn.py` | the vLLM backend (`TETRA`, alias `KV4H`) |
| `install_kv4.py` | prelude installer |
| `tune_kv4h.py`, `kv4h_tune.json` | knob sweep and its table |
| `kl_score.py`, `kl_compare.py`, `decode_delta.py` | accuracy against bf16, pure decode rate |
| `radiance_kv4_*.py` | Triton reference path (`KV4`) and fake-quant backend (`KV4FAKE`) for format A/Bs |
| `patch_int4_pth_rocm.py` | vLLM's own int4 kernel for the drafter, an A/B control (`KV4_DRAFT_ATTN=TRITON_ATTN`) |
| `eval_format.py`, `probe_wmma.hip`, `probe_dot_scaled.py` | offline format evaluation and hardware probes |
| `../bench_kv4_attn.py`, `../tests/kv4/` | kernel bench and tests |
