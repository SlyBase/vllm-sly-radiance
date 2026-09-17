# sly/ — SlyBase additions to vllm-radiance

This directory contains all of SlyBase's own patches and configs on top of
[StillDeadcode/vllm-radiance](https://codeberg.org/StillDeadcode/vllm-radiance).
Goal: RDNA4/gfx1201 MXFP4 support for `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` on
AMD Radeon AI PRO R9700.

## Contents

| File | Purpose |
|---|---|
| `patch_quark_mxfp4.py` | AITER Triton MXFP4 gate for gfx1201 (5 hunks) + registration of the HIP kernel plugin |
| `mxfp4/radiance_mxfp4.py` | `RadianceMxfp4W4A8LinearKernel` plugin (dispatches large M to the HIP kernel, otherwise AITER) |
| `mxfp4/radiance_mxfp4_fp8.hip` | Hand-written fp8 WMMA W4A8 GEMM kernel (prefill), from [ggz14/radiance-vllm-mxfp4](https://codeberg.org/ggz14/radiance-vllm-mxfp4). **2026-09-15**: `DEC_MAX_N` 32768 → 36864 — Qwen3.8-27B (TP=1) has a fused gate_up with N = 34816, which the old limit silently routed into the folded (prefill) kernel (64 calls/step at ~420 µs instead of ~190 µs = 30 of 72 ms per DFlash step); scratch + block counter in `radiance_mxfp4.py` enlarged accordingly |
| `mxfp4-configs/` | MXFP4 GEMM configs for AITER's config lookup (JSON, like `fp8-configs/`) — `gfx1201/.../gemm_afp4wfp4/DEFAULT.json` was originally taken 1:1 from AITER's own gfx950/gfx1250 default; AITER has **no** gfx1201 tuning for this GEMM family and hard-fails without a config file with an `AssertionError` (no built-in fallback). **2026-09-14 (Task #5, Build #8 first-request crash)**: this default crashed the EngineCore on the very first real request with `triton.runtime.errors.OutOfResources: out of resource: shared memory, Required: 67584-100352, Hardware limit: 65536` — gfx950/gfx1250 have significantly more LDS per CU (`_LDS_CAP_BYTES` in aiter: gfx1250=327680, gfx950=163840; **gfx1201 is missing from this map entirely**), gfx1201/RDNA4 only has 64KiB. Fixed via brute-force probing directly on the R9700 (real `gemm_afp4wfp4()` launch per M bucket, stepping `num_stages` down on `OutOfResources` until the launch succeeds): all 8 buckets only needed `num_stages` reduced 3→2 (or 2→1 for `any`), `BLOCK_SIZE_*` unchanged. This is a pure correctness fix (makes the kernel runnable at all) — **actual do_bench perf tuning of the block sizes themselves remains open Task #7 work** |
| `patch_short_prefill.py` | GDN fix: 1-token prefill is no longer misclassified as decode (Subagent E, originally a vllm5 bind-mount patch for 0.28.0, verified here verbatim against 0.29.0) |
| `patch_dflash_w4_packed.py` | DFlash drafter as a compressed-tensors W4A16 checkpoint (`syvai/Qwen3.8-27B-DFlash2-W4A16`): `qkv_proj` has `weight_packed`, no raw `.weight` → deferred like the fp8 case + dequantized via its own forward (ported from the vllm5 bind-mount patch 0.27.1, verified against 0.29.0 / vllm7 2026-09-15) |
| `mxfp4/radiance_lmhead_fp8.py` | `RadianceLMHeadFp8` (`RADIANCE_LMHEAD_FP8=1`): the Quark-excluded lm_head (bf16, 248320 × 5120 = 2.54 GB per call, with DFlash 2 calls/step = 8.6 of ~49 ms) is quantized to fp8 per output channel after loading and computed via row-wise `torch._scaled_mm` (hipBLASLt) with per-token fp8 activation; frees up 1.27 GiB for the KV cache. `RADIANCE_LMHEAD_FP8_MIN_M` (default 16) pads small M for hipBLASLt |
| `patch_lmhead_fp8.py` | Hooks `radiance_lmhead_fp8` into `QuarkConfig.get_quant_method` (before the exclude branch that would otherwise return `UnquantizedLinearMethod`) |
| `patch_gdn_nonspec_mask.py` | `patch_gdn_metadata`'s numpy path leaves `non_spec_sequence_masks_cpu` unassigned → `UnboundLocalError` on engine init as soon as `--speculative-config` is set; one-liner that reconstructs the tensor from the numpy mask |
| `patch_w4a16_tiles.py` | gfx1201 tile table for the DFlash2 W4A16 drafter in `rdna_hybrid_w4a16.py`: the HIP skinny kernel only kicks in at M ≤ 5, the drafter always has M = 8 × seqs → always Triton, and its gfx12x heuristic (tuned on Llama-3.1-8B) picks 16×16 tiles at M ≤ 32 (gate_up N=34816: 2176 workgroups, 267 GB/s). Table `(gs, K, N, M bucket 8/16/32/40/64)` for qkv/o/gate_up/down (0.1.5: + fc + the int4 lm_head, N=248320: stock 2623/2709/5327/2715/2824 → 1305/1333/1585/2099/2185 µs at M = 8/16/32/40/64), measured DRAM-cold with `bench_w4a16_tiles.py` (2026-09-15): gate_up 1.7–3.5×, down 1.5–2.6×, qkv 1.5–2.4× at M ≤ 32, 1.03–1.43× at M = 40/64. `RADIANCE_W4A16_TILES=0` = stock heuristic (A/B without a rebuild). As of 2026-09-15 (not yet built, will ship with the next build) additionally `fc` (`combine_hidden_states`, K=25600, runs before draft padding at M = 8 × seqs): 288/293/575/316/326 → 205/205/205/286/294 µs (bit-identical; ≤ 0.4 ms per step, below BetterBench resolution, hence no dedicated build) |
| `bench_w4a16_tiles.py` | Tile sweep for `_triton_w4a16_skinny_fmt_kernel` (BLOCK_M/N/K, num_warps, num_stages; weight rotation ≥ 160 MB against L2 hits; `--fc` = fc layer only, `--lmhead` = the int4 lm_head shape only) — source of the table in `patch_w4a16_tiles.py` |
| `mxfp4/radiance_lmhead_int4.py` | `RadianceLMHeadInt4` (`RADIANCE_LMHEAD_INT4=1`, 0.1.5): the lm_head as int4 W4A16 (symmetric, group-128 bf16 scales, `RADIANCE_LMHEAD_INT4_GS`) on the drafter's kernel path (`torch.ops.vllm.rdna_hybrid_w4a16_apply`: HIP `wvSplitK_int4_g` at M ≤ 5, else the Triton skinny kernel with the lm_head rows of the tile table). Quantised after loading in 8192-row chunks with a per-group MSE clip search (`RADIANCE_LMHEAD_INT4_CLIP=mse`, ratios 1.0…0.8, 1.6 s; `rtn` = plain amax/7), packed with vLLM's `pack_int4_exllama_shuffle`. 656 MB per call instead of fp8's 1.27 GB (already at 543 GB/s = 85 % of the read peak, so bytes were the only lever left): offline M = 8 2459 → 1305 µs (502 GB/s), 0.61 GiB more KV. Error ~3× fp8 (rms 10.8 % vs 3.7 % of the logit rms on the real weight; argmax flips on a flat random proxy 24 % vs 10 %) — ships only behind the GSM8K / accept-length gate |
| `mxfp4/radiance_embed_int8.py` | `RadianceEmbedInt8` (`RADIANCE_EMBED_INT8=1`, 0.2.2): embed_tokens (248320 × 5120 bf16 = 2.37 GiB, Quark-excluded, only ever gathered) as per-row symmetric int8 with fp32 scales, or group-128 RTN nibbles with `RADIANCE_EMBED_BITS=4`. Subclasses `UnquantizedEmbeddingMethod`, quantises after loading in 8192-row chunks and replaces `layer.weight` + `layer.weight_scale`; `embedding()` = index_select + rescale (plain torch, compile/CUDA-graph safe). TP=1 only. 1.18 GiB (int8) / 1.74 GiB (int4) more KV cache |
| `patch_lmhead_int4.py` | Hooks `radiance_lmhead_int4` in front of the fp8 block in `QuarkConfig.get_quant_method` (int4 wins when both envs are set); anchors on the fp8 block, so it runs after `patch_lmhead_fp8` in the Dockerfile loop |
| `check_lmhead_int4.py` | Offline numerics + timing of the lm_head variants on the real weight (fp32 reference, fp8, int4 g128 mse/rtn, g64): logit error, argmax flips, top-16 overlap, HIP-vs-Triton path consistency, per-call time per M. Throwaway container with the GPU exclusive and the HF cache mounted |
| `mxfp4/radiance_fused_norm.py` | `RADIANCE_FUSED_NORM_QUANT=1` (0.1.6, default off, production on since 2026-09-16): custom ops `radiance::add_rms_quant`, `radiance::silu_mul_quant`, `radiance::gdn_norm_quant` (+ fake impls) around the HIP kernels of the same name, and the gates (`input_ok`/`post_ok`/`mlp_ok`/`gdn_ok`: every consumer a folded radiance W4A8 layer, TP=1, bf16, width limits). Outputs `(q_fp8, scale)` for `mxfp4_linear_pq`. Per-fusion switches `RADIANCE_FUSED_NORM_QUANT_ADD_RMS/_SILU/_GDN` (default 1) |
| `patch_fused_norm_quant.py` | Hooks the above into `Qwen3NextDecoderLayer.forward` (both layernorms with a residual; layer 0's first norm and `model.norm` stay), `Qwen2MoeMLP.forward` (act_fn), the GDN `forward_hip`/`forward_cuda` (take the `(q, scale)` pair, carry the bf16 projection for `.dtype`/`.device`) and `_output_projection`; adds `RADIANCE_FUSED_NORM_QUANT*` to `envs.compile_factors()`. Anchors on already-patched files, runs last in the Dockerfile loop. The `.hip` `radiance_silu_mul_quant` MAXG 5 → 9 (N ≤ 18432, Qwen3.8-27B intermediate 17408; per-thread scratch sizing only) |
| `patch_kv_groups.py` | `RADIANCE_KV_GROUP_SIZE=<n>` overrides the KV-cache group size in `_get_kv_cache_groups_uniform_page_size` (`v1/core/kv_cache_utils.py`). Stock picks the smallest layer bucket — with the 5-layer DFlash2 drafter that is 5, padding 16 attn → 20 and 48 GDN → 50 layers (15.6 % of the pool per 32k request). `=8` gives 2 + 6 + 1 groups, +13 % KV tokens at the same pool size. Unset = stock |
| `patch_embed_int8.py` | Hooks `radiance_embed_int8` (embed_tokens as int8 rows, `RADIANCE_EMBED_INT8=1`, or int4 g128 with `RADIANCE_EMBED_BITS=4`; 1.18 / 1.74 GiB back for the KV cache) into `VocabParallelEmbedding.__init__` where the unquantized default is chosen — Qwen3Next and the DFlash drafter build embed_tokens without `quant_config`, so the `QuarkConfig.get_quant_method` hook point of the lm_head patches never sees them; adds `RADIANCE_EMBED_*` to `envs.compile_factors()`. Anchors on `patch_fused_norm_quant` output, runs last |
| `check_fused_norm.py` | Numerics of the three fusions against the unfused chain (eager + `torch.compile` native norm/activation + `scaled_fp8_quant` + the same pq GEMM) on real layer-3/4 weights for M ∈ {1, 8, 16, 64, 128, 512, 2048}: fp8 code/scale/residual diffs, GEMM output diff, `torch.compile(fullgraph)` + CUDA-graph replay smoke test, per-call timing. Throwaway container with the GPU exclusive |
| `mxfp4/bench_decode_cells.py` | Cell sweep of the HIP W4A8 decode kernel over M ∈ {16…64} on the six production shapes × (split-K 1/2/4, BK 64/128), DRAM-cold (weight rotation ≥ 160 MB), one worker process per `RADIANCE_MXFP4_DECODE_KS`/`_BK` combination, `--check` against the fp32 reference. Source of the 0.2.1 `RADIANCE_MXFP4_DECODE_TUNE16` table in `radiance_mxfp4_fp8.hip` `launch()` (2026-09-16, 252 cells in ~1 min): BK=64 at tm=4 was 1.33× slower on gate_up/qkvz, o_proj/out_proj M=16/24 want split-K 2 (−6 %). Throwaway container with the GPU exclusive (`PYTHONPATH` to a freshly built `.so` ahead of site-packages) |

**Kernel decision (Subagent C1, completed)**: not an either/or — both
paths are needed, layered:

1. **Mandatory base**: `patch_quark_mxfp4.py` hunks 2–5 enable AITER's
   Triton `gemm_afp4wfp4` (native MXFP4 W4A4) on gfx1201
   (`RADIANCE_MXFP4=1`). Without this, every MXFP4 linear layer falls
   back to `EmulationMxfp4LinearKernel` (BF16 dequant + F.linear).
2. **Optimization on top**: `RadianceMxfp4W4A8LinearKernel`
   (`mxfp4/radiance_mxfp4.py` + `.hip` kernel, `RADIANCE_MXFP4_W4A8=1`) is
   placed at the head of the ROCm kernel list via hunk 1 and takes over
   large-M (prefill) shapes with fp8 activations (1.6–1.9x faster than the
   tuned AITER path, per ggz14's `PERFORMANCE.md`); `can_implement()`/
   `is_supported()` reject everything else and the request falls back to (1).

All 5 hunks in `patch_quark_mxfp4.py` were tested against the real
vLLM 0.29.0 source (tag `v0.29.0`): anchor match + `ast.parse()` + idempotent
second run (NOOP) + `py_compile`, all green. Details/rationale per hunk are
in the file's docstring. Two of the five hunks (4 + 5) are **new relative to
ggz14's original** — between 0.27 and 0.29, vLLM added two additional
CDNA-only gates around the AITER custom op that ggz14's 0.27.1 target
didn't yet know about.

`patch_gfx1201.py` (top level, inherited from `StillDeadcode/vllm-radiance`,
byte-identical to ggz14's version) is already in the tree and was likewise
verified against vLLM 0.29.0 + Triton 3.8.0 — all four anchors (gcn-arch env,
AITER CDNA gate, Triton `HIPDriver.is_active`, AITER sampler gate) match
verbatim, no change needed.

**`patch_unified_attention_lds.py` (top level, LDS fix + bf16 tuning for
gfx1201) ported to aiter 0.1.21.post2**: between the old version (which
this patch originally targeted) and `0.1.21.post2`, AITER completely
rebuilt the config selection in `unified_attention.py` — `select_3d_config`/
`select_2d_config` (Python elif chains) are gone, replaced by table-driven
JSON configs (`get_unified_attention_config()` in
`unified_attention_utils.py`, loaded via `json.load()` — i.e. plain data,
not Python, so `_patchlib.apply()`/`ast.parse()` can't anchor there). The
patch was rewritten from scratch (6 hunks instead of 3) against the real
`0.1.21.post2` source, fetched via `raw.githubusercontent.com`:
- **LDS-fit clamp** (correctness, unconditional, 2D **and** 3D) now lives in
  `_unified_attention_2d_triton()`/`_unified_attention_3d_triton()`, right
  after the config lookup, before the respective kernel launch.
- **Consistency issue resolved**: `kernel_unified_attention_3d` and
  `reduce_segments` independently derive their segment split from the same
  `TILE_SIZE` (`tiles_per_segment = cdiv(seq_len, NUM_SEGMENTS *
  TILE_SIZE)`) — if `TILE_SIZE` were only clamped locally in
  `_unified_attention_3d_triton()`, `reduce_segments` would keep running
  with the old value and silently merge the wrong segments. Fix:
  `_unified_attention_3d_triton()` now returns its (clamped/tuned)
  `TILE_SIZE`, the single call site in `unified_attention()` catches it and
  passes the same value on to the subsequent `_reduce_segments_triton()`
  call.
- **bf16/fp16 3D decode tuning** (TILE16/warps4/stages2/waves2, plus a
  matching `num_warps=4` in the reduce kernel) structurally ported to the
  new location, gated behind `DEVICE_ARCH == "gfx1201"` (new relative to
  the old patch — the old RDNA branch was implicit, the new architecture
  code is arch-agnostic, and an ungated override would have hit other archs
  on this fork too).
Verified (without a GPU): anchor match (all 6 locations `count==1` against
the real `0.1.21.post2` file), `ast.parse()`, idempotent second run (NOOP),
`py_compile` — all green (`patch-verify-e2/` in the scratchpad, not part of
the commit). **Open**: whether TILE16/warps4/stages2/waves2 are still
do_bench-optimal in the new table structure needs to be re-measured on
real R9700 hardware — here it was only ported structurally/correctly to the
new location, not re-tuned.

**Open item for Subagent B**: the `.hip` kernel itself still needs
build wiring in the Dockerfile (compile step + extension load, analogous to
`radiance_kernels.py`'s pattern for the other `.hip` kernels in this repo,
e.g. R4D) — the pure Python side (hunk 1 + `radiance_mxfp4.py`) is done,
the compiled `.so` is still missing.

The `RADIANCE_MXFP4_KERNEL` env var from the original plan draft is
dropped — selection happens automatically via the kernel priority list
(`_POSSIBLE_MXFP4_KERNELS[ROCM]`), not via a manual switch.

## Git Branching / Upstream Merge

```
main                    ← integration (upstream + all sly patches), protected: PR + green `ci`
upstream/stilldeadcode  ← read-only mirror of StillDeadcode/vllm-radiance main (Codeberg)
upstream/ggz14          ← read-only mirror of ggz14/radiance-vllm-mxfp4 main (Codeberg)
archive/*, v0.1.x       ← tags: frozen pre-2026-09 branches / VERSION history
```

Remotes:
- `origin`        = `https://github.com/SlyBase/vllm-sly-radiance.git`
- `stilldeadcode` = `https://codeberg.org/StillDeadcode/vllm-radiance.git`
- `ggz14`         = `https://codeberg.org/ggz14/radiance-vllm-mxfp4.git`

The `upstream-sync` workflow (daily 04:00 UTC) fast-forwards the `upstream/*`
mirrors and opens or updates a PR `upstream/<name>` → `main` whenever a mirror
has commits `main` lacks (commit list, test-merge conflict status, image-relevant
files in the body). The mirrors are never edited by hand and never force-pushed.

Manual merge procedure (same thing the PR does):

```bash
git fetch origin
git switch -c merge/stilldeadcode origin/main
git merge origin/upstream/stilldeadcode
# expected conflicts: Dockerfile (our pins + patch loop stay), README.md, VERSION (bump)
# Patch anchor conflicts are hard failures (_patchlib.apply() uniqueness check) —
# re-verify every affected patch in sly/ against the new anchor string:
ci/patch_dryrun.sh
python3 ci/check_consistency.py --base origin/main
git push -u origin merge/stilldeadcode && gh pr create
# merge only when `ci` is green; never force-push main (branch protection)
```

## Build & Push

The `build` workflow (`.github/workflows/build.yml`, self-hosted runner
`rocm-build` in LXC 2408, CPU only) builds `vllm-sly-radiance:<VERSION>-rocm10.0`
on every `VERSION` change on `main` and pushes it to ghcr.io for `v*` tags or
`gh workflow run build.yml -f push_ghcr=true`. Manual equivalent:

```bash
docker build -t ghcr.io/slybase/vllm-sly-radiance:<VERSION>-rocm<ROCM_VERSION> .
docker push ghcr.io/slybase/vllm-sly-radiance:<VERSION>-rocm<ROCM_VERSION>
```

`<VERSION>` = content of `../VERSION` (own SemVer, independent of the
upstream radiance version). `docker login ghcr.io` must be set up locally
beforehand (on LXC 2408) with a PAT that has `write:packages` scope.
