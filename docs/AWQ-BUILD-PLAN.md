# AWQ build for Swift-1.5-Qwen3.8-27b-heretic — implementation plan

Status: draft. Owner: Daidalos. Date: 2026-10-22.
Supersedes `AWQ-IMPORT-PLAN.md` (that one assumed no W4A16 kernel on gfx1201 — wrong, see §3).

## 1. Goal

Publish an **AWQ (W4A16) build** of `akumaburn/Swift-1.5-Qwen3.8-27b-heretic` next to the
existing builds, so users can pick AWQ or GPTQ. The build must also be servable — and
quality-gated — on our own R9700.

## 2. Prior art: the upstream AWQ already exists

`ukisai/Swift-1.5-Qwen3.8-27b-W4A16-AWQ` (the non-abliterated parent, 16.7k downloads) is a
complete, working recipe. We copy it instead of inventing one.

| Property | Upstream value |
|---|---|
| Tool | LLM Compressor, compressed-tensors `0.18.0` |
| Format | `pack-quantized`, `quant_method: compressed-tensors` |
| Weights | int4, group 128, `strategy: group`, `symmetric: false`, `zp_dtype int8` |
| Activations | none (W4A16) |
| Ignored (303 entries) | all `model.visual.blocks.*` linears, `lm_head`, and per GDN layer `linear_attn.norm`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b` |
| Quantized | `mlp.{gate,up,down}_proj`, `self_attn.{q,k,v,o}_proj`, `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` |
| MTP head | untouched, shipped as a separate `model-mtp-bf16.safetensors` |
| Size | 18.7 GB (6 shards) |

The abliteration changed only `attn.o_proj`, `attn.out_proj` and `mlp.down_proj` values — the
module list is identical, so the ignore list transfers verbatim.

## 3. Correction: no AWQ kernel work is needed on the R9700

My earlier verdict ("AWQ only runs as a Triton dequant path on ROCm") was wrong for
compressed-tensors W4A16. Checked in the shipped vLLM:

| Layer | Finding |
|---|---|
| `compressed_tensors_wNa16.py` | `W4A16` routes through `choose_mp_linear_kernel` |
| `_POSSIBLE_KERNELS[ROCm]` | `RDNA3W4A16`, **`RDNAHybridW4A16`**, `TritonW4A16`, `Conch`, `Exllama` |
| `RDNAHybridW4A16LinearKernel.can_implement` | `on_gfx1x()` → gfx11 **and gfx12 (gfx1201 = R9700)**; quant types `uint4b8` **and `uint4` (asymmetric, zero points)**; group sizes 32/64/128 |
| handshake | comment in `process_weights_after_loading`: "AWQ weights arrive as (K, N) … compressed-tensors arrive as (N, K)" — both layouts are handled |
| our patches | `sly/patch_w4a16_tiles.py` adds `radiance_w4a16_tile` (1 KB 16×128 blocks) on gfx12x and the fused `radiance_w4a16_gemm` / `_silu` / `_split` ops |

So an AWQ / compressed-tensors W4A16 checkpoint loads **and decodes through the HIP skinny
kernel** on the R9700. On NVIDIA the same checkpoint takes `awq_marlin` / `Marlin`.

The one real gap: our **fused** W4A16 ops (`radiance_w4a16_gemm`, `_silu`, `_split`) take no
zero-point argument — they are symmetric-only (`zp_bias=8`). Asymmetric AWQ therefore runs on
the generic hybrid op: no fused silu, no split-K table, no per-shape tuning. That is a
*performance* option later, not a blocker.

## 4. Scope

| In scope | Out of scope |
|---|---|
| One AWQ build of the heretic model, upstream-matching recipe | Porting Marlin/AWQ kernels |
| Local serve + quality gate on the R9700 | Converters to `paroquant` |
| Publish card with numbers, sha256, ignore list | Re-quantizing ukisai's parent |
| Optional: asym fast path in the fused W4A16 ops | Calibration-time rotation (QuaRot) — that is a different build |

## 5. Sizes (computed from the checkpoint headers, not estimated)

| Item | Value |
|---|---|
| Quantizable parameters | 24.327 B (48 GDN layers × 0.383 B + 16 attention layers × 0.372 B) |
| `weight_packed` int4 | 12.16 GB |
| `weight_scale` fp16 (g128) | 0.380 GB |
| `weight_zero_point` int8/4-bit (g128) | 0.095 GB |
| `embed_tokens` + `lm_head` bf16 | 5.09 GB |
| MTP head (bf16, separate file) | 0.85 GB |
| Vision tower bf16 | ~0.70 GB |
| **Build size** | **~19.3 GB** (upstream measured 18.7 GB) |
| bf16 working set at quant time | **~55 GB** — does **not** fit one 32 GB card, see §5.1 |
| Disk | 52 GB source + 19 GB output + env |

### 5.1 One R9700 only — how the 55 GB run still happens

Hardware: **1 × R9700 (32 GB), host RAM 31 GB, no swap.** The bf16 model never fits in one
place. Three routes, in order of preference:

| Route | How | Cost | Risk |
|---|---|---|---|
| **P1 — llm-compressor with GPU + CPU + disk offload** (primary) | `from_pretrained(..., device_map="auto", max_memory={0: "29GiB", "cpu": "26GiB"}, offload_folder=<disk>)`, then `oneshot(pipeline="sequential")`: the calibration forward runs once with ~10 GB offloaded to disk, then every layer is quantized from cached inputs | one run, ~4-8 h | offload thrash / a pipeline detail that re-runs the forward per layer. **Verify in WP0** |
| **P2 — our own streaming quantizer** (fallback) | the pattern of `quantize_dflash_mxfp4.py`: read one layer from the shards, apply the AWQ per-channel scale, quantize int4/g128/asym, write, free. Peak RAM 2-3 GB. Needs per-module `absmean` from a separate calibration forward | +1-2 d code | the AWQ scale has to be folded into the previous op (`input_layernorm`, `linear_attn.norm`, `up_proj` rows) — Qwen3.5/GDN folding differs from the drafter's and must be re-derived |
| **P3 — rent one 80 GB CUDA GPU for the run** | identical recipe, no code | a few EUR, one afternoon | none technical; the output is backend-agnostic |

For P2 the calibration statistics can come from a forward pass on a 4-bit build that already
fits the card (our MXFP4 arm, or upstream's AWQ of the parent) — that is a **deviation** from
bf16 statistics and must be stated in the card and covered by the KL gate.

The *quantization arithmetic* itself needs no GPU at all: `int4 g128 asym` is elementwise, so
it can run on CPU from mmapped shards.

The **quality gate also fits one card**: the bf16 reference for first-token KL comes from the
layer-by-layer streaming scorer (`llm-quantization-quality`, reference tier B) — one layer
resident at a time, no full bf16 model in VRAM.


## 6. Work packages

### WP0 — environment and smoke test (1 d, first half needs no GPU)

1. Separate venv/container: `llmcompressor` + `compressed-tensors` + `transformers >= 5.15` +
   a ROCm torch wheel that knows gfx1201. Do **not** install it into the radiance image
   (torch/ROCm pins would fight).
2. **Offload smoke (decides P1 vs P2):** load the 27B bf16 with `device_map="auto"` and
   `max_memory={0:"29GiB","cpu":"26GiB"}` plus a disk `offload_folder`, run *one* calibration
   batch, and watch peak VRAM, peak RAM and whether the forward survives. If it survives, the
   whole plan is one `oneshot` call; if not, switch to P2 before writing any quant code.
3. Smoke the *runtime* side before spending the quantization run: serve ukisai's existing AWQ
   on gfx1201 with the radiance image and read the kernel line —
   `Using RDNAHybridW4A16LinearKernel for CompressedTensorsWNA16`. If it lands on
   `TritonW4A16` instead, the speed expectation changes and we stop to look.

Abort criterion: no kernel on ROCm accepts the checkpoint (`ValueError: no kernel`), or the
offload smoke dies in a way that P2 cannot cover.

### WP1 — quantization run (0.5 d machine time, one card)

Replicate the upstream recipe exactly, so the ignore list and the format match:

```
ignore = ["lm_head", "re:.*visual.*", "re:.*linear_attn\.(in_proj_a|in_proj_b|norm)$"]
recipe = AWQModifier(scheme="W4A16", targets="Linear", ignore=ignore)
         → QuantizationModifier/export  int4, group_size=128, symmetric=False, zp int8
```

Calibration: our fixed corpus (a ~5k-token text plus 8 chat-templated items, thinking off —
the same corpus as the KLD harness), 256-512 sequences, max length 2048. Record the corpus
sha256 in the card. Keep `model-mtp-bf16.safetensors` byte-identical from the source.

Write to a temp dir, rename at the end, sha256 every shard.

### WP2 — gate (1 GPU window, sequential arms)

| Arm | What it proves |
|---|---|
| A1: bf16 heretic | reference for KL and GSM8K |
| A2: AWQ build, gfx1201, hybrid kernel | our arm: quality + decode |
| A3: same build through stock TritonW4A16 (`VLLM_DISABLED_KERNELS=Triton...` off/on) | what the radiance patches are worth on this format |
| A4 (optional): symmetric re-run of WP1 | whether sym + fused path beats asym |

Metrics: first-token KL against bf16 on 100 prompts (Heretic's own method, response prefix
`\n \n\n` because this is a thinking model), GSM8K-200 with the fixed chat template, decode
tok/s at c=1 and c=8, prefill tok/s. Acceptance: KL ≤ the upstream AWQ's published value,
GSM8K within 1.5 points of bf16.

Note the asymmetry of our stack: our MXFP4/paroquant builds are *rotated*, AWQ is not — the
model card's own numbers put unrotated 4-bit at a much worse quant-only KL than rotated 8-bit
(0.1182 vs 0.0264). Say that in the card so nobody expects paroquant quality from AWQ.

### WP3 — publish (0.5 d)

Repo `Swift-1.5-Qwen3.8-27b-heretic-W4A16-AWQ`: weights, `config.json` with the quant config
verbatim, `model-mtp-bf16.safetensors`, and a README with: base revision, abliteration note,
format = AWQ W4A16 g128 asym, "runs with compressed-tensors or Marlin", the measured KL /
GSM8K / decode table, our R9700 kernel line, sha256, and the safety disclaimer inherited from
the base card.

### WP4 — optional: asymmetric fast path (1.5 d)

Extend `sly/patch_w4a16_tiles.py`: pass `zp` into the fused ops and fold the zero point as a
per-group epilogue correction (the mechanism `radiance_paroquant` already uses as its `SZ`
term). Gate it with the existing `sly/check_w4a16_fuse.py` harness and the WP2 arms. Do it
only if A2 vs A4 shows a decode gap worth closing.

### WP5 — optional: symmetric AWQ variant (0.5 d machine time)

`AWQModifier(symmetric=True)` re-run: same algorithm, sym grid, hits our fused path directly.
Publish only if A4 beats A2 by more than ~3 % decode.

## 7. Repo-side (vllm-sly) changes

| File | Change | Needed for |
|---|---|---|
| none | — | the build itself: no image file changes, no `sly/patch_*` |
| `docs/AWQ-BUILD-PLAN.md` | this plan | orientation |
| `README.md` | one line under the supported checkpoints: W4A16 (AWQ/compressed-tensors) runs on the hybrid kernel | users |
| `docs/TECHNICAL.md` | substep: which kernel a W4A16 checkpoint lands on, sym vs asym | the next person |
| `sly/patch_w4a16_tiles.py` | WP4 only | optional speed |
| `VERSION` + `CHANGELOG.md` | only if WP4/WP5 land in the image | release |

The quantization tooling lives **outside** the image (its own venv), so the image stays
untouched unless WP4 happens.

## 8. Risks

| Risk | Likelihood | Handling |
|---|---|---|
| `llmcompressor` needs `transformers >= 5.15` and a newer torch than the radiance image | high | separate venv, never inside the image; verify the qwen3_5 model class loads there first (WP0.2) |
| ROCm torch wheel lacks gfx1201 kernels for AWQ calibration | medium | CPU run with `device=cpu` (days) or rent one 80 GB CUDA GPU for the run; the output is portable |
| Runtime routes to `TritonW4A16` instead of the hybrid kernel | low | WP0.3 smoke before the quantization run |
| Asymmetric AWQ is slower on our card than our symmetric GPTQ (no fused silu / split-K) | medium | measure A2 vs A4; WP4 or WP5 |
| Users expect rotated-4-bit quality | high | state the KL class in the card, point to the paroquant/W8A8 rotated builds |
| llm-compressor offload (P1) does not work on one card | medium | WP0.2 decides early; fallback P2 costs 1-2 extra days, P3 costs a few EUR |
| P2 folding point for Qwen3.5/GDN derived wrongly | medium | per-layer output-error report (the `rel` column of `quantize_dflash_mxfp4.py`) plus the end-to-end KL gate |

## 9. Rollout

1. WP0 (env + offload smoke + runtime smoke) — 1 d.
2. WP1 run overnight — 0.5 d machine time.
3. WP2 gate in one GPU window, arms sequential, report per arm.
4. WP3 publish, then decide WP4/WP5 from the numbers.

Total to a published, gated AWQ build: **~3 days of work + 1 GPU window, on a single R9700.**


## 10. Open questions

1. Do the users want AWQ specifically, or just "another format that vLLM/ollama-class runtimes
   accept"? (GGUF already covers the low-friction side.)
2. Publish under `akumaburn` or under our org?
3. Is a symmetric variant acceptable for the card, given the fused-path speed gain?
