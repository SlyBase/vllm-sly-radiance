# Changelog

Every image version (`VERSION`, image tag `vllm-sly-radiance:<version>-rocm10.1`) has a section here.
The section of a version is the body of its GitHub release: `ci/release_tag.sh` publishes it when the
`v<version>` tag is cut, and `ci/check_consistency.py` fails a PR that bumps `VERSION` without one.
Measurements are on one AMD Radeon AI PRO R9700 with `amd/Qwen3.8-27B-Quark-AWQ-MXFP4` + the DFlash2
W4A16 drafter and the production arguments; deep dives are in [docs/TECHNICAL.md](docs/TECHNICAL.md),
full benchmark tables in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions before 0.1.0 belong to the
upstream repositories (StillDeadcode/vllm-radiance, ggz14/radiance-vllm-mxfp4).

## [1.6.0] - 2026-10-18

Stack bump to vLLM 0.32.0 (released 2026-10-18). torch / triton / torchvision are unchanged: 0.32.0's own
`docker/Dockerfile.rocm_base` pins the same release/2.13 trio as 0.31.0, so everything below is what the
release itself moves.

### Added
- **Upstream 4-bit KV cache for ROCm (`--kv-cache-dtype ultraquant_4bit`, vllm#57057) is now available**
  to serve, on top of the repo's own TetraAttention (`KVCACHE_DTYPE=kv4`). UltraQuant is AMD's 4-bit KV
  format (Hadamard rotation, one power-of-two scale per 32 channels, about 2x the density of FP8;
  released with "ROCm gains the 4-bit ultraquant_4bit KV cache"), documented on MI355X. **Not yet measured
  on the R9700** and not yet A/B'd against fp8 or TETRA; `fp8` stays the default and `kv4` stays the
  measured 4-bit option until the gate says otherwise. The image needs no change: it is vLLM's own KV
  dtype, gated to the launch only.
- MiniCPM-V 4.7 support (`RADIANCE_MINICPMV4_7`) is native in 0.32.0 (vllm#58674): `sly/patch_minicpmv4_7.py`
  leaves the Dockerfile patch loop into `ci/unused_patches.txt`, and the patch now SKIPs on a tree that
  already carries the model instead of overwriting 0.32's newer files with its 0.31-era copies.
- The 0.31.0-era aiter hold is lifted: vLLM 0.32.0 pins AITER v0.1.24.post1, whose
  `aiter/ops/triton/__init__.py` carries the `unified_attention` alias again (verified in the
  v0.1.24.post1 tree), so `ROCM_AITER_UNIFIED_ATTN` stays usable at the new pin.
- Serving `ultraquant_4bit` together with DFlash drafting needs
  **`--kv-cache-dtype-skip-layers sliding_window`**: the run-wide KV dtype also reaches the
  drafter's per-layer backend selection, and the head-128 sliding-window / non-causal DFlash drafter
  geometry has no backend that accepts `ultraquant_4bit` (the crash is a loud backend-selection
  failure at engine init). This is upstream's own escape hatch (the same list TurboQuant uses for its
  dense-model boundary layers, which is empty for hybrids), so the 16 full-attention layers still take
  the UltraQuant backend while the drafter keeps `TRITON_ATTN` on its `auto` (bf16) window cache.

### Changed
- **vLLM 0.31.0 -> 0.32.0.** Every `sly/` patch re-anchored with identical behaviour (details in
  `sly/README.md` and the patch files): `patch_skinny_gemm` (0.32 added a `VLLM_BATCH_INVARIANT`
  early-return; the hook now sits after it, so a determinism serve keeps the deterministic GEMM),
  `patch_gdn_metadata` (0.32 removed the "all rows drafted nothing" early-out from the spec-mask block -
  the numpy fast path keeps that; and the pure-spec-block index slice is upstream now, so the 0.32 shape
  keeps only the cached arange/empty), `patch_gdn_lazy` (abstract.py's RecoverSSM guard renamed to
  `use_replayssm`; the `bt_row_idx` locals moved into `_copy_mamba_state_block`, both skip hooks
  re-anchored to the call sites), `patch_w4a16_tiles` (the stock W4A16 kernel takes `stride_bn`, the
  launcher gained stride asserts - the 0.32 `stride(1)` one guarded for tiled weights, see Fixed - the
  skinny limit grew 1.2x and moved into `pack_skinny_int4`; for the
  LAYOUT 0 path the stride is carried, the tiled layout is addressed absolutely as before),
  `patch_nvfp4_mxfp4` (a `_is_mxfp4_w4a8_fp8` branch moved between the NVFP4 return and the MXFP4
  check), `patch_cactus` (the rejection kernel gained `watermarking_skip_mask`; `cactus_delta` anchors on
  the new tail). One behaviour note: **MRV2 is now the default model runner on ROCm** (#56360) and cascade
  attention was removed upstream - the production launch used neither.
- aiter 0.1.23 -> 0.1.24.post1 (vLLM 0.32.0's AITER_BRANCH, exactly).
- transformers 5.17.0 -> 5.19.0 (newest inside 0.32.0's `>=5.16.1,<5.20.0`; also removes the old
  MiniCPM-V AutoProcessor caveat, which needed `>=5.18`). PyPI dependency set regenerated
  (`constraints.txt`, 78 changed pins) by `ci/resolve_stack.py`; python stays 3.12 (0.32 dropped 3.10).

### Fixed
- `sly/patch_w4a16_tiles`: guard `rdna_hybrid_w4a16`'s new 0.32 stride assert for tiled weights.
  0.32 added `assert b_q.stride(1) == 1` ("Weight rows must be contiguous") to
  `triton_w4a16_skinny_fmt_gemm`; the assert means innermost-contiguity and is written for 2-D
  weights. On a 4-D `radiance_w4a16_tile` weight stride(1) is the block stride (the tile addressing
  computes its indices and never reads `stride_bn`), so the assert fired at the first 1.6.0 UltraQuant
  start on the drafter's `fc` combine projection (M-skinny -> tiled store). The guard keeps the assert
  for 2-D weights only; the 4-D tile blocks stay checked by the patched `numel` equality. Found on the
  first live UltraQuant start, fixed in-patch and verified by patch dry run (151 hunks, pass 2 all NOOP).
- `patch_skinny_gemm`: restore the module-level `import radiance_gemm` that the 0.32 re-anchor of
  `patch_skinny_gemm.py` dropped. The 0.31 shape inserted the import by rewriting the
  `def rocm_unquantized_gemm_impl` anchor; the 0.32 body-only re-anchor shipped the hook text without
  the import, so the first indunitized-GEMM call (inside the compiled dflash2 candidate-selector
  graph at profile time) died with `NameError: _radiance_gemm is not defined`. The import is a second
  idempotent hunk now (anchored on the def signature, unique in 0.31 and 0.32 trees).

### Measured
- Night gate pending.

## [1.5.1] - 2026-10-16

### Added
- **Cactus acceptance transform for speculative decoding (`RADIANCE_CACTUS_DELTA`, default 0.5;
  `0` = bit-exact stock).** `sly/patch_cactus.py` raises the V2 rejection kernel's acceptance threshold
  from the target probability q_n to h_n = clamp(q_n + sqrt(2*delta*q_n*(1-q_n)), 0, 1)
  (Cactus, arxiv 2604.04987), trading a bounded KL budget for a higher acceptance rate. Only the
  non-greedy Leviathan branch is touched; greedy, synthetic-rate and block-verification paths stay
  untouched, and delta=0 keeps the stock comparison bit-exact (the exp/log round-trip is guarded).
  **Behaviour change: the image now samples with delta=0.5 unless `RADIANCE_CACTUS_DELTA=0` is set.**
  Greedy requests are unaffected; sampled requests (temperature > 0) see a bounded drift from the
  target distribution in exchange for more accepted draft tokens per step.
  CPU-tested in `sly/tests/test_cactus_cpu.py`; the GPU A/B (acceptance, tokens/step, GSM8K) runs
  through the accept gate with the env set on the gate unit.

## [1.5.0] - 2026-10-16

### Added
- **TetraAttention: a 4-bit KV cache for the full-attention layers (`KVCACHE_DTYPE=kv4`, opt-in; fp8
  stays the default).** The 16 attention layers' K and V are Walsh-Hadamard rotated and stored as
  signed int4 codes with one e4m3 scale per token (272 bytes per head and token instead of fp8's 512).
  Their attention runs on a new HIP kernel (`sly/kv4/radiance_kv4h.hip`, backend `TETRA`, working name
  `KV4H` kept as an alias) that feeds the stored nibbles straight into `v_wmma_i32_16x16x32_iu4` (S = K
  Q^T with an int4 query) and `v_wmma_i32_16x16x16_iu8` (P as uint8 against each tile's max, V's nibble
  as 16 v in int8): no unpacking, no zero point. The DFlash drafter's head-128 sliding-window, non-causal
  attention runs on it too. The modules, the backend registry entries and the int4 drafter patch are baked into the image
  (`sly/kv4/install_kv4.py` and `sly/kv4/patch_int4_pth_rocm.py` are in the Dockerfile patch loop;
  the kernel is built on first use); nothing changes with `KVCACHE_DTYPE=fp8`. Every launch / format parameter
  is a knob (`RADIANCE_KV4H_*`, launch arguments, `sly/bench_kv4_attn.py` flags, `sly/kv4/tune_kv4h.py`
  sweep with a knob table). The Triton implementation (`KV4`) and a fake-quant backend (`KV4FAKE`) stay
  as reference / A/B paths.
- `sly/kv4/kl_score.py`, `kl_compare.py`: accuracy of a KV format against a bf16 KV cache
  (teacher-forced top-20 logprobs, KL / top-1 / perplexity per context bin); `sly/kv4/decode_delta.py`:
  the pure decode rate by the 150/450-token difference.

Measured (one R9700; engine A/Bs in one GPU window, fp8 = R4D_HYBRID):
- KV pool **593k tokens instead of 342k** (~1.75x; a full 262k context needs 4.25 GiB of KV instead of 8).
- Accuracy against a bf16 KV cache, 5 texts x 120k tokens: KL **0.0126** (fp8 0.0123), top-1 agreement
  95.3 % (95.6 %), perplexity +0.54 % (+0.06 %), flat from 4k to 128k; GSM8K-200 0.80 (0.79); needle at
  97k found at 10 / 50 / 90 %.
- Decode, ms per step at an 18k prefix: 1 stream **33.5 vs 35.0**, 4 streams **41.9-48.1 vs 45.7-46.8**.
- Cold prefill with `--max-num-batched-tokens 3456` (3328-token chunks): 8k / 24k / 85k **2.27 / 7.47 /
  36.7 s vs 2.34 / 7.72 / 38.0 s**. Keep the budget >= 2 x 1664 plus the draft slots: with half-size
  cells vLLM sets the attention block to 1664 tokens (fp8: 880), and a smaller budget prefills one
  block per step.
- Trade-offs: slightly less precision (above); prefix-cache hits at 1664-token granularity (a turn
  recomputes up to 1,663 tokens instead of 879); an offload tier written by an fp8 serve must not be
  reused by a kv4 one (the bytes are format-specific).
- Re-measured on 1.5.0 against an fp8 control in one GPU window (engine A/B, 262k context): KV pool
  **747,492 vs 439,031 tokens** (+70 %), cold prefill 8k / 32k / 64k **+6 / +7 / +8 %** (the fp8 control ran
  with the same 3456-token chunk; with the production chunk of 2048 the gap is about 3 points wider),
  1-stream decode 34.2 vs 35.1 ms per step, BetterBench conc 1 / 2 / 4 / 8 within run-to-run noise,
  GSM8K-200 0.855 vs 0.845. Against a bf16 KV cache (5 texts x 120k tokens): KL 0.0208 (fp8 0.0245),
  top-1 agreement 94.5 % (94.4 %), true-token NLL +0.0057 (fp8 +0.0004), i.e. about +0.5 % perplexity. Two
  224k-token requests at once fit without preemption; 20 minutes of mixed load: 542 requests, no error.
  Also run with `KVCACHE=ram` and `KVCACHE=ram KVCACHE_EXACT=1` (2 GiB tier): multi-turn follow-up is
  token-identical to a cold run. Not tested: the disk tier, kv4 on other cards than the R9700.


## [1.4.0] - 2026-10-16

### Added
- **Tuned int4 W4A16 MoE config for the R9700** (`moe-configs/E=256,N=512,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json`, 256 experts, N 512, group 128 — MiniCPM-V 4.7 / Qwen3.5-MoE-A3B shape). The image only shipped fp8 configs, so int4 MoE ran the default tiles. *Measured:* 1.2-1.7x per MoE call, ~+8 % end-to-end decode (MiniCPM-V 4.7, 210 W).
- **docs/MINICPM-V47.md**: serving recipe for MiniCPM-V 4.7 + DSpark on one R9700 (MXFP4 at load for the bf16 linears and the drafter, k = 8, `--attention-backend ROCM_AITER_UNIFIED_ATTN`), and its dead ends in docs/NOT-ADOPTED.md. *Measured:* BetterBench `--decode` 169.4 tok/s at 210 W (reproduced 169.4 / 169.3; Qwen3.8 + DFlash2 production 132.4 in the same window); the attention backend alone 146.8 -> 169.4; GSM8K 200 0.835 (bf16 base 0.85).
- **MiniCPM-V 4.7 model support (`RADIANCE_MINICPMV4_7=1`, default `0`).** Backport of vllm-project/vllm#58674 (merged 2026-10-11, after the 0.31.0 pin) so `openbmb/MiniCPM-V-4.7-35B-A3B` loads and serves: `sly/patch_minicpmv4_7.py` ships the `minicpmv4_7.py` model file, the `minicpmv4_6.py` refactor (+272/−63) that 4.7's `super()` chain depends on, and the registry entry. The PR's parser/reasoning/chat-template files are not carried (only reached by a tool-calling/reasoning request, not by load/serve). Gated: default off leaves the Qwen3.8 production image byte-identical. The model file reads `config.json` directly (no `MiniCPMV4_7Config` import), so it loads on the image's transformers 5.17; only the HF `AutoProcessor` for `model_type: minicpmv4_7` needs `transformers>=5.18` (raise the pin, `pip install -U transformers` in the container before serving — the pin is metadata, not an import guard). GDN geometry (head_k/head_v 128/128, conv 4), full-attention geometry (head_dim 256, 2 KV heads) and MoE (256 experts / 8 active) match the R4D kernels. *Measured:* import of `MiniCPMV4_7ForConditionalGeneration` + the 4.6 refactor verified clean in a throwaway container (bind-mount of the two patched files, no GPU); end-to-end serving on the R9700 is the release gate, not the merge gate.

## [1.3.4] - 2026-10-13

### Added
- **`KVCACHE_EXACT=1` (default `0`; needs `KVCACHE=ram`/`disk`): bit-exact cached resumes + checkpoint retention.** Six new `kv-cache/` patches (canonical prefill chunks + checkpoint retention), applied at container start, all FATAL on a failed anchor and inert with their env gates at `0`. A prefix-hit prefill now runs on the same fixed chunk grid as a cold prefill (cached == uncached in the low bits), and the GPU/tier caches stop keeping GDN/drafter checkpoints at grid points nobody resumes from. Flag off: the generated serve command is byte-identical to before. Author's bench notes (not yet GPU-verified in this repo): follow-up turns served from cache 56 → 77 % (4 agents), 69 → 85 % (3 agents); a GPU A/B window is the release gate, not the merge gate.

## [1.3.3] - 2026-10-13

### Added
- **`STARTUP_CACHE=1` (default `0`): faster boots, self-invalidating build caches.** `serve-mxfp4.sh` now keys its cache dir on image **ID** + GPU arch (a re-pulled tag or another card gets a fresh dir instead of reusing stale graphs), persists the comgr / tvm-ffi / tilelang JIT caches across `--rm` boots, compiles `radiance_mxfp4_fp8.so` once per (source, flags, arch, image ID), and prunes stale cache trees when the disk passes 65 %. A miss is always safe (the slow path rebuilds). Measured (one R9700, approximate): comgr+tvm-ffi+tilelang 189 s → 151 s per boot; the `.so` ~30 s. Flag off: the generated command is unchanged.

## [1.3.2] - 2026-10-13

### Fixed
- **Lazy GDN without speculative decoding is deterministic again.** In 1.3.1, `RADIANCE_GDN_LAZY=1`
  without a drafter gave different logprobs for the same cold request from the second request on
  (eager: bit-identical). Cause: the pre-forward materialize (mode 0) zeroed the stash headers at
  block-table column `dst_col + 1`. Without a drafter there is no stash block, so that column lies past
  the request's blocks and can hold a stale id of a live block; the zeroing hit live GDN state.
  Proven on the GPU window: masking that column made lazy bit-identical to eager.
  - `radiance_gdn_lazy.temporal_enabled()`: the lazy temporal path (skip of vLLM's temporal copies +
    materialize) runs only when the mamba groups own a stash block (`num_speculative_blocks > 0`).
    Without a drafter vLLM's exact Triton copies run, as in eager.
  - libr4d `gdn_lazy_materialize` (rx10 patch): new mode bit `LZ_MODE_STASH`. The kernel reads or
    zeroes column `+1` only when the bit is set, so a wrong caller cannot hit a foreign block.
  - CPU tests: `sly/tests/test_gdn_lazy_cpu.py`.

### Measured
- GPU window 2026-10-13, image `1.3.2-rocm10.1`, production argv, bf16 SSM state, fp8 KV:
  - No spec, one cold 30,926-token chat request, 2 runs per arm: eager run 1 = run 2 (bit-identical);
    lazy runs 1 and 2 = eager (tokens and logprobs bit-identical, 148 tokens). 1.3.1 lazy: run 2
    diverged at token 0.
  - Spec (DFlash2, 7 tokens), lazy, turnbench EXACT, 3 sessions x 7 turns up to 110k tokens:
    HEALTH PASS, every cold turn exact, cached vs cold drift 16 of 18 GPU-path turns (eager 1.3.1 on
    the same bench: 18 of 18). The cached-path drift is the known prefix-cache drift of eager too,
    not a lazy defect.
- The acceptance night gate has not run on 1.3.2; this release is cut by hand (`release.yml`).

## [1.3.1] - 2026-10-12

### Fixed
- **The lazy GDN stash path no longer crashes without speculative decoding.** `RADIANCE_GDN_LAZY=1`
  read the stash column (`block_table_tensor[:, 1]`) unconditionally in `gdn_attn.py`. That column
  exists only when a drafter runs (`num_speculative_tokens > 0`); without DFlash the block table is
  one column wide and the index raised `IndexError: index 1 is out of bounds` at boot. The stash
  mechanism (column 2 / the stash block) is a speculative-decode concept, so it now activates only
  when `num_speculative_tokens > 0`; the lazy materialize path (temporal states) is independent and
  runs either way. *Measured: pending — the no-spec lazy arm crashes at boot before this fix and has
  not re-run on the GPU window; the lazy path stays default OFF.*

## [1.3.0] - 2026-10-11

### Added
- **bf16-state variants of the lazy GDN kernels** (`gdn_lazy_update` and `gdn_lazy_materialize`,
  `..._bf16_bf16state`). Production runs `--mamba-ssm-cache-dtype bfloat16`, but libr4d rx10 only
  shipped fp32/f16-state variants — so `RADIANCE_GDN_LAZY=1` died at boot with "no materialize
  kernel for a torch.bfloat16 state cache". The kernel bodies were already templated on the state
  dtype (the bf16 load/store primitives exist from the eager kernels); this adds the missing
  bf16state translation units, declarations, pybind bindings and registry rows. *Measured: the lazy
  arm boots and serves on the bf16 state cache (the multi-turn gate, `kv-cache/turnbench.py`, passes
  HEALTH — no empty replies, no loops — with logprob drift on the same order as the eager control);
  the no-spec lazy arm still crashed at boot (fixed in 1.3.1); the lazy path stays default OFF.*

## [1.2.2] - 2026-10-11

### Fixed
- **The lazy GDN state snapshots (`RADIANCE_GDN_LAZY=1`) are wired into the image again.** Since the
  revert on 2026-09-17 the feature was dead in every image: `radiance_gdn_lazy.py` was missing from
  the COPY allowlist and `patch_gdn_lazy` was not in the build loop, so the flag was a silent no-op.
  The patch itself was also written against the pre-0.30 tree and broke in three places: `abstract.py`'s
  `get_kv_cache_spec` (rebased onto `use_kda_recoverssm` / `num_speculative_tokens`), the shared-build
  fast path (dropped: `patch_gdn_shared_build` is not in the loop, so the full build path is the only
  one the image takes), and all nine `mamba_utils` hunks (rebased onto the 0.30 fused align kernels,
  which gained the `TEMPORAL_TILES` constexpr). Every hunk now carries its own idempotency sentinel.
  `radiance_gdn_lazy.py` was also written against the pre-0.30 `mamba_state_copy_funcs` (a flat
  per-state tuple); 0.30 restructured it into a per-`MambaAttentionBackendEnum` dict, so the
  `_Tables` lookup now indexes `copy_funcs[layer.mamba_type][state]` — the old code crashed with an
  `IndexError` on the first GDN layer. **The multi-turn correctness gate (`kv-cache/turnbench.py`)
  is not yet green on the GPU window; this release ships the wiring and the crash fix, with the
  lazy path still default OFF, so it changes no production behaviour.**

## [1.2.1] - 2026-10-10

### Fixed
- **The KV-cache offload modes (`KVCACHE=ram|disk`) work again on vLLM 0.30, i.e. on every image since
  0.4.0.** The six "always" patches were written against vLLM 0.27.1; four of them stopped part-way on
  the 0.30 tree, which left an offload tier that stored but never served (0 external hits). vLLM 0.30
  rebuilt the offload scheduler and carries part of the set itself: the load now starts at the local
  boundary (mixed-hit lookback + assertion), Mamba states are stored per retained checkpoint and
  `--prefix-cache-retention-interval` (default 0) keeps only the replay boundary and shared-prefix
  junctions (mamba-stride), store reachability is judged on the absolute grid (swa-align). Those hunks now
  report `SKIP` through the new `_patchlib.skip_if_upstream`; eagle-groups got a second anchor shape;
  reconcile-reask, touch-all and align-last-block apply unchanged; vLLM 0.31 (the pin on `main` since
  1.1.0: multi-line annotator call, no `_touch`, `_make_boundary_key` with request context) is carried
  too. The 0.27.1 shapes still apply, and `ci/patch_dryrun.sh` now runs the ram-mode list against the
  pinned vLLM so the next bump fails `ci` instead of a boot WARNING.
- **Tier hits landed two turns back.** With the eagle drafter annotated, the scheduler commits the Mamba
  boundary state one block before the aligned prompt end, while the drafter group's retained tail ended
  at the aligned end, so the next turn's lookup found no complete window there and fell back to the
  previous turn's junction (37-56k tokens recomputed per tier turn). `patch_swa_eagle_replay_tail.py`
  keeps one more drafter block per reachable boundary (`RADIANCE_SWA_EAGLE_REPLAY_TAIL=1`, a superset of
  the stock mask); the tier hit now lands where a GPU hit lands. Without the eagle annotation (upstream's
  default for this model) the hit lands 4-5 turns back, so the annotation stays on.

### Added
- `kv-cache/patch_offload_boundary_trace.py`: `RADIANCE_OFFLOAD_BOUNDARY_TRACE=1` logs every Mamba
  boundary-state hand-off and its fate (stored / dropped and why). Silent otherwise.
- `TIERBENCH_API_KEY` for `tierbench.py`, `equivbench.py`, `turnbench.py` against a server started with
  `VLLM_API_KEY`.
- `kv-cache/README.md`, "vLLM 0.30 and later": what each patch does on the new tree and why.

### Measured
- turnbench, 3 sessions x 7 turns, step 18k (398k tokens against a 331k GPU pool), `KVCACHE=ram` 16 GiB,
  production launch + vision, second start: tier hits C6/A7/B7/C7 at 94,080 / 112,896 / 117,376 / 113,792
  tokens (one block before the previous prompt end) against 77,952 / 94,976 / 100,352 / 77,952; 20-21k
  tokens recomputed per tier turn against 37-56k; tier turn wall 17-20 s against 29-41 s. Repeated on image
  1.1.1 (vLLM 0.31.0): tier hits at 94,160 / 112,640 / 117,920 / 114,400, 19-21k recomputed, 16.7-19.8 s,
  cold twins first divergent token 63 / 19 / 2 / none, max |dlogprob| 0.13-0.18. HEALTH pass on both images. Cold twins of the four tier turns (bf16 SSM cache): first divergent token 52 / none / 3 /
  none, max |dlogprob| 0.24 / 0.33 / 0.06 / 0.16 -- the same positions as without the patch and the same
  kind of drift as the GPU-hit turns of the run, so the tier adds none of its own; bit-exactness is not
  shown on this build. Single-stream decode with the tier on is unchanged. Image binaries unchanged:
  `kv-cache/` rides the `/patches` mount and is applied at container start.

## [1.2.0] - 2026-10-07

### Added
- **Sharded MXFP4 checkpoints (low-RAM loading) + local source reuse.** `fp8_mtp.py`
  (the one-time build behind `setup-mxfp4.sh`) now reads a source checkpoint in **either**
  layout — AMD's single `model.safetensors` **or** the standard HF sharded layout
  (`model.safetensors.index.json` + `model-0000X-of-0000Y.safetensors`) — and **writes sharded
  output by default** (5 GiB shards; `RADIANCE_SHARD_GIB` to resize, `RADIANCE_SINGLE_FILE=1` to
  restore the single-file output). A sharded checkpoint is what unblocks 16 GiB hosts: vLLM loads
  it shard-by-shard (one shard mapped at a time), so the peak host RAM for the weight load is the
  largest shard instead of the whole ~19 GiB model. The serve path (`serve-mxfp4.sh`) is already
  layout-agnostic — it hands vLLM the checkpoint directory — so a sharded build serves with no
  further change. Every copied body tensor is byte-identical to the source (verified), and the
  fp8 MTP head is unchanged (same `F8_E4M3` dtype, per-channel scale, ~2 % rel err).
- **`SRC_LOCAL` in `setup-mxfp4.sh`: reuse an already-downloaded source snapshot.** Point
  `SRC_LOCAL` at a directory holding `model.safetensors` or `model.safetensors.index.json` (any
  layout) and the ~19 GiB pull is skipped — machines that already have the weights via another
  workflow no longer re-download them. The container mount now bind-mounts the source at `/src`
  (was: the HF cache path), so a local directory anywhere on the host works, not just a snapshot
  under `~/.cache/huggingface`.

### Changed
- `setup-mxfp4.sh` step 3 reports "source checkpoint" (was: "AMD's MXFP4 release"); the
  reclaim-`~19 GiB` hint now only prints when a download actually happened (not for `SRC_LOCAL`).

Measured: no kernel change, so decode/prefill/KV pool are unaffected. Engine memory with a
16 GiB container cap, fresh boot per run, 4 runs: file-backed RSS of the engine 12.3-12.7 GiB
(single file) vs 5.0-5.3 GiB (5 GiB shards), anonymous RSS 2.2 GiB in both, no OOM kills, load
time unchanged (about 210-250 s to ready). The container cgroup still reaches its cap, and under
real host pressure the single file swapped heavily; see docs/TECHNICAL.md. The change moves the one-time
build and the serve-time weight load: a 16 GiB host loads a sharded checkpoint with peak RAM at one
5 GiB shard instead of the full model (minutes vs. hours of swapping); a `SRC_LOCAL` snapshot skips
the ~19 GiB pull. Copy fidelity is byte-exact (verified against a synthetic single-file **and**
sharded source); the fp8 MTP head is numerically unchanged (~2 % rel err, MXFP4 was ~11.6 %).

## [1.1.1] - 2026-10-08

### Fixed
- **Split-K decode GEMM could read stale partials.** The W4A8 decode kernel
  (`radiance_mxfp4_fp8_gemm_decode`, every MXFP4 linear at M <= `RADIANCE_MXFP4_DECODE_MAX_M` and, since 1.0,
  the wide band up to `RADIANCE_MXFP4_WIDE_MAX_M`) splits K over 2 or 4 blocks on narrow shapes (o_proj,
  out_proj, down, in_proj_ba) and the last block to arrive sums the partials. Only thread 0 fenced before the
  counter bump, and `__syncthreads` is workgroup scope, so the other threads' partial stores were not
  guaranteed visible to the reducing block on another CU. Now every thread fences, thread 0 bumps the
  counter with an agent-scope acq_rel atomic, and the reducing block fences again before it reads
  (the pattern libr4d's tiled W4A8 split tail documents for the same bug). Production runs this kernel
  on every decode/verify step. The same fix is in the opt-in AutoRound, EsCHA (both kernels) and the
  second ParoQuant decode kernel (the first already fenced on every thread) and in ggz14's root copy of
  the MXFP4 kernel. Written 2026-10-04 as 0.7.1 and never released; ported onto 1.1.0 unchanged.
- **Lazy GDN: a prefill step after a prefix hit could run from a stale state** (opt-in, only with
  `RADIANCE_GDN_LAZY=1`; production does not set it). `r4d_gdn_lazy_materialize` in `sly/r4d/r4d_extras_rx10.patch`
  returned without migrating the state when the stash column `bt[base_col + 1]` was null, even with no
  candidates to replay. That column is null when a prefill step spans more than one mamba block (block 896,
  `--max-num-batched-tokens 2048`) on a request's first step after a prefix-cache hit, so the step started from
  whatever the new running block held. The stash is now required only when count > 0; with none the kernel
  copies base -> dst. Decode (the lazy update kernel) is unchanged. Ported from libr4d-rx13 rx17 as published in
  zzpanic/qwen3.6-vllm-gfx1201-launchers `6147726` (validated there: offload-tier resume bit-identical, deep
  multi-turn gate 48/48 against 7/48); their stale-stash counters are not part of this fork's kernel.

### Added
- `sly/mxfp4/check_decode_splitk.py`: alternates two inputs launch after launch and requires every
  output to be bit-identical to its golden, per production shape x M x split, including the wide band.

### Measured
- Pending: `check_decode_splitk.py` on 1.1.0 (control) and 1.1.1, decode step A/B, GSM8K 200;
  lazy GDN: a prefix-hit multi-turn run with `RADIANCE_GDN_LAZY=1` against eager GDN.

## [1.1.0] - 2026-10-07

The newest stack that still runs on ROCm 10.1: vLLM 0.31.0 on the torch / triton / torchvision / aiter combination vLLM's own
ROCm base builds, all of it AMD's prebuilt rocm10.1 wheels. No new knob.

### Changed
- **vLLM 0.30.0 -> 0.31.0** (released 2026-10-05). Every `sly/` patch was re-anchored against v0.31.0 (details in
  docs/TECHNICAL.md, section 1.1.0): the DFlash context-KV hooks (`patch_dflash_fused_kv_fp8`, `patch_dflash_w4_packed`,
  `patch_w4a16_fuse`), `patch_gdn_metadata`, `patch_w4a16_tiles`, `patch_fused_norm_quant`, `patch_adaptive_width`,
  `radiance_lookup_draft` (DFlash2's `top_k` / `candidate_sampler`), `radiance_r4d_attn`. Behaviour is unchanged.
- **torch 2.12.0 -> 2.13.0+rocm10.1.0**, **triton 3.6.0 (PyPI) -> 3.8.0+git669b31ac.rocm10.1.0**, **torchvision 0.27.0
  (compiled) -> 0.28.0+rocm10.1.0 (AMD wheel, `amd-torchvision-device-gfx1201`)**: the exact pairing vLLM 0.31's
  `docker/Dockerfile.rocm_base` builds (release/2.13, Triton 669b31a, vision v0.28.0), from AMD's whl-next index. Triton is
  hash-pinned (`TRITON_BUILD`, `TRITON_SHA256`); torchvision is no longer compiled in the default build (still is for a
  source-built torch). torch 2.14 is on the index but is not what vLLM pins. The 0.5.0 - 0.5.4 GPU hang was torch 2.13 with
  triton 3.7.1 outside upstream's pairing; this release is upstream's pairing.
- **aiter 0.1.22.post1 -> 0.1.23** (vLLM 0.31's `AITER_BRANCH`). 0.1.23 still has the `aiter.ops.triton.unified_attention`
  alias vLLM's `rocm_aiter_unified_attn` imports; Renovate stays below 0.1.24, which dropped it. aiter 0.1.23 imports
  **flydsl** (pinned 0.3.4.1 in aiter's requirements.txt) at `import aiter`; aiter is installed `--no-deps`, so the first
  build failed at engine start with `No module named 'aiter.ops.triton.unified_attention'` (the compat finder swallows the
  real ImportError). flydsl 0.3.4.1 is now installed explicitly and the release-stage check asserts it.
- **transformers 5.18.0 -> 5.17.0**: vLLM 0.31 declares `transformers >= 5.10.4, < 5.18.0`; 5.17.0 is the newest it allows
  (Renovate capped below 5.18.0). `constraints.txt`: `openai-harmony` -> `oss-harmony` (vLLM 0.31's rename), nothing else moved.
- **libr4d 5dc6302 -> a3e4833** (Renovate #104): the new commit only touches libr4d's README (7 added lines), so the kernels and the `r4d_extras_rx10` patch chain are byte-identical to 1.0.0. The upstream-sync PR #103 (stilldeadcode README, no image file) is not taken. ROCm base and base-image digests are unchanged.

### Removed
- Four patches left the apply loop because vLLM 0.31.0 contains them: `sly/patch_short_prefill` (GDN 1-token prefill),
  `sly/patch_gdn_nonspec_mask`, `sly/patch_mamba_align_retire` (vllm#55450, already native in 0.30.0) and
  `sly/patch_rocm_load_max_split` (0.31 gates the load-time allocator scope on `is_cuda_alike()` and also applies it to
  `profile_run`). The files stay in the tree for the 0.30.x lineage (`ci/unused_patches.txt`).

### Measured
- pending: decode step gap, prefill 8k / 32k / 64k, c8 arrivals, TTFT 160 tokens, KV pool (reference 436,906), GSM8K 200,
  startup time (fresh and second start), A/B/A against 1.0.0 in one 300 W window.
- pending: DFlash acceptance with the context-KV precompute inside the FULL graph (0.31), `lookup draft: installed` marker,
  `perseq` adaptive width on 0.31's varlen graphs.

## [1.0.0] - 2026-10-08

Longer prefills are faster, the KV pool is 11.5 % larger, concurrent decode is ~5 % faster, and the image moves to
ROCm 10.1 with torch 2.12. All measured on one R9700 at 210 W with the production launch, A/B/A in one GPU window,
second start.

### Added
- **`--attention-backend R4D_HYBRID`**: libr4d's paged prefill attention for prefill runs of 512 tokens or more
  (`RADIANCE_R4D_PREFILL_MIN_Q`), AITER unified attention for everything else, so decode, graphs and KV pool stay as
  they were. Prefill 2k / 8k / 32k / 64k **+2 / +3 / +13 / +23 %** (2375 / 2410 / 1976 / 1555 -> 2428 / 2506 / 2229 /
  1919 tok/s). Plain `--attention-backend R4D` is not recommended: decode 60.9 instead of 34.9 ms per step.
- **`RADIANCE_ADAPTIVE_WIDTH=perseq`**: per-request verify width (port of the Radiance engine's `adaptive_k`) on
  varlen FULL decode graphs. 8 concurrent requests with arrivals **+4.7 %** (401.7 / 403.5 -> 421.4 tok/s); one stream is
  untouched (bit-identical greedy output). Why ggz14's dynamic width lost and this one does not: TECHNICAL.md.
- **`RADIANCE_MXFP4_WIDE_MAX_M=192`**: the split-K decode GEMM kernel for M 129..192 instead of the padded 256-row tile.
  Time to first token of a 160-token prompt **-13.8 %** (109.6 -> 94.4 ms), c8 with arrivals +1.7 %. 192 is the edge:
  the folded tile is faster again at 256.
- Test packages of this work in `sly/tests/lessons/`.

### Changed
- **KV pool 391,193 -> 436,097 tokens (+11.5 %)** at 262k context: `--gpu-memory-utilization 0.98` with
  `--compilation-config.cudagraph_mode FULL_DECODE_ONLY` (was 0.9655 with FULL_AND_PIECEWISE). CUDA graph memory 1.24 ->
  0.22 GiB, peak activation 1.69 -> 0.65 GiB. Decode step, c8 and TTFT unchanged; 64k prefill runs without OOM. Util
  0.98 alone gives 404,947.
- **ROCm 10.1** (`rocm/dev-ubuntu-24.04:10.1.0-full`, HIP 7.16) and AMD's **torch 2.12.0+rocm10.1.0** (AMD ships no 2.11
  for 10.1), torchvision 0.27.0; triton 3.6.0, aiter 0.1.22.post1, vLLM 0.30.0 unchanged. Image tag
  `1.0.0-rocm10.1`. The torch >= 2.12 CPU spin that held torch back is fixed in ROCm 10.1. Performance is neutral (step
  35.35 / 35.59 / 35.13 ms, prefill within 1-2 %). Fresh compile caches on first start; to stay on 10.0 see TECHNICAL.md.
- README reference launch, options table and docs layout updated; the KV pool reference in AGENTS.md is now 436,906 (the full 1.0 launch, warm compile cache).
- `tests-lessons/` moved to `sly/tests/lessons/`; the working notes are folded into TECHNICAL.md.

### Removed
- `NOTES-A/D/F/H.md` (content is in `docs/TECHNICAL.md`, section 1.0.0).

### Credits
- The adaptive verify width and the band findings (wide decode tile) come from the Radiance engine fork
  ([StillDeadcode/radiance](https://codeberg.org/StillDeadcode/radiance), our fork slydlake/radiance); the prefill
  kernel is libr4d.

### Measured
- Prefill (tok/s, control / R4D_HYBRID / control): 2k 2375 / 2428 / 2380, 8k 2410 / 2506 / 2450, 32k 1976 / 2229 / 1970,
  64k 1555 / 1919 / 1554. Greedy step 35.12 / 35.05 / 35.14 ms, c8 arrivals 395.7 / 391.8 / 395.9 tok/s, short-prompt
  greedy text identical, prompt NLL delta -0.0007.
- KV pool 436,097 (util 0.98 + FULL_DECODE_ONLY); greedy 34.89 ms (control 34.88 / 36.68), c8 401.9 (control 401.7 /
  414.4), TTFT 72 / 256 / 904 tokens unchanged, 64k prefill 1572.
- Wide band (edge 256 in the server run): TTFT 160 tokens 109.6 / 109.4 -> 94.4 ms, other prompts within 0.6 % except 256
  tokens (+9.7 %, hence the edge 192), c8 405.4 / 406.2 -> 412.8, greedy identical for prompts <= 160 tokens.
- Adaptive width: c8 arrivals off 401.7, uniform 408.8, perseq 421.4, off2 403.5; c4 steady unchanged; 30 % of decisions
  shortened, 4.15 verify rows saved per step; 8-way concurrent greedy diverges at near-ties (batch-shape numerics).
- ROCm 10.1: greedy 35.35 / 35.59 / 35.13 ms, prefill 2k 2357 / 2344 / 2385, 8k 2432 / 2382 / 2401 tok/s, KV 385,934 on
  the second start of the test run (fresh cache dirs), greedy text 4 of 8 identical (A/A: 8 of 8).
- **Release candidate as shipped** (ROCm 10.1 image, all 1.0 flags) against 0.7.0, same window, 210 W, base1 / rc / base2:
  KV pool 391,193 / **430,433** / 391,193 (+10 %; the varlen graphs of `perseq` and the 10.1 runtime take ~6k of the
  436k from the util/graph change alone); greedy 34.84 / 35.04 / 35.06 ms; TTFT 160 tokens 103.8 / **93.1** / 106.3 ms;
  prefill 8k / 32k / 64k 2666 / **2737** / 2556, 2128 / **2376** / 2083, 1645 / **2025** / 1627 tok/s (+5 / +13 / +24 %);
  c8 arrivals 418.3 / 411.0 / 407.9 (neutral in the combination; the isolated +4.7 % of `perseq` does not show here);
  multi-turn follow-up with a prefix hit 3.3–3.6 s on both.
- GSM8K 200 (cot zero-shot, greedy, flexible-extract): 0.84 (0.7.0 in the same window: 0.83).
- **BetterBench full at 300 W** (1.0.0 then 0.7.0, same window): weighted decode 135.8 / 137.4 tok/s (within noise);
  prefill 1.5k / 6k / 12k / 24k / 47k 3132 / 3136 / 3105 / 2949 / 2635 against 3108 / 3049 / 2946 / 2648 / 2164 tok/s
  (+1 / +3 / +5 / +11 / +22 %); concurrency 1 / 2 / 4 / 8 / 16 120 / 213 / 338 / 461 / 466 against 121 / 219 / 338 /
  446 / 449 tok/s; TTFT 47k 17.9 s against 21.7 s; KV pool 436,906 against 391,193 tokens.
- Not adopted (docs/NOT-ADOPTED.md): gated gate_up + SwiGLU fold, int4 lm_head LEAN configs, fp16 SSM state, draft refill
  after a prefix hit, plain R4D backend.

## [0.7.0] - 2026-10-04

### Fixed
- **0.4.6 – 0.6.2 did not start on the GPU.** aiter 0.1.24 (Renovate, `9d8c19d`) dropped the module
  alias `aiter.ops.triton.unified_attention`, and vLLM 0.30.0's `rocm_aiter_unified_attn` backend still
  imports it. EngineCore died during model construction with `ModuleNotFoundError`. The traceback
  surfaced in `qwen3_5.py` → `make_layers`, which is why transformers was suspected first. aiter is
  back on **0.1.22.post1**, the version production 0.4.1 runs, and Renovate holds it below 0.1.24 until
  vLLM imports the new path.

### Changed
- **PyTorch is AMD's prebuilt wheel instead of a 2-hour source build.** `torch==2.11.0+rocm10.0.0` and
  `amd-torch-device-gfx1201` come from `stable.repo.amd.com/rocm/whl-next`, built by AMD for exactly
  this ROCm release, and run against the image's own `/opt/rocm`. ROCm 10.0 is TheRock-based and
  loads the wheel's `.kpack` device code. Four pieces make it fit:
  - `fix_amd_torch_metadata.py` drops the wheels' `rocm[libraries]` / `rocm-bootstrap` /
    `triton==3.8` requirements. Otherwise pip silently replaces torch with the CUDA torch from PyPI.
  - a `rocm_sdk/` stand-in preloads torch's ROCm libraries from `/opt/rocm`, so the process has one
    HIP runtime.
  - `torch/lib` is not stripped; strip breaks the wheel's `libtorch_hip.so`.
  - the stack stage now fails if a non-ROCm torch ever ends up in the venv.
  
  Triton stays 3.6.0 (the PyPI wheel, as in 0.6.2), torchvision 0.24.1 is compiled against the new
  torch. The source build remains available as `--build-arg TORCH_FROM=torch-wheel` (or `build.yml`
  input `torch_from_source`) for a ROCm release AMD has no wheel for yet.

### Measured
- Accept gate, `full` (run 37202109954, image built from this change with transformers 5.17.0; this
  release keeps 5.18.0, which was cleared of the crash above): **PASS**. BetterBench conc 1/2/4/8
  120.1 / 203.1 / 308.1 / 374.5 t/s (0.97–1.05× baseline), acceptance 0.476 (base 0.468), GSM8K 0.835
  (base 0.845), all log markers. Soft misses: KV pool 379,461 tokens (0.98× of the baseline,
  −1.3 % against production 0.4.1's 384,316), startup 316 s (base 186 s).
- Cold build on the runner: ~30 min instead of ~2 h 45 min. Image 5.27 GB.

## [0.6.2] - 2026-10-04

Build only: the image contents are the same stack (torch 2.11.0, triton 3.6.0, torchvision 0.24.1,
aiter 0.1.24, vLLM 0.30.0). Nothing changes at runtime.

### Changed
- **The torch wheel comes from ghcr.io instead of the runner's build cache.** The torch compile
  (~2 h of a ~2 h 45 min cold build) moved into its own stages (`buildbase` → `torch-build` →
  `torch-wheel`). `build.yml` pushes that target once per torch definition to
  `ghcr.io/slybase/vllm-sly-radiance-torch:<tag>`, the tag from the new `ci/torch_key.py`, and
  passes it as `TORCH_FROM` from then on. Before, the wheel lived only in the runner's BuildKit
  cache, and that cache is lost whenever the disk fills or someone runs a prune. That happened on
  2026-10-04: the cache was gone, and a transformers-only PR compiled torch again. A plain
  `docker build .` still compiles torch locally, because `TORCH_FROM` defaults to the in-file stage.
- **triton is the PyPI wheel of the same tag**, hash-pinned with the new `TRITON_SHA256` ARG,
  instead of a 25-minute source build. The manylinux wheel ships the AMD backend with its own LLVM,
  HIP headers and device bitcode, and loads `libamdhip64.so` at runtime. That is what the source
  build against this base produced too. The anchor in `patch_gfx1201.py` (`triton/backends/amd/driver.py`)
  is byte-identical in the wheel. A triton bump now needs the new hash: the Renovate PR note says so,
  and a stale hash fails at the download instead of installing a different file.

### Measured
- Cold build on the runner (LXC 2408, 8 cores, empty BuildKit cache): torch 7,174 s + triton
  1,482 s of 2 h 46 min (run 37157999145). With the torch wheel on ghcr.io and triton from PyPI,
  the same cold build skips both, about 2 h 25 min less. The first build per torch key still compiles
  torch once.

## [0.6.0] - 2026-10-03

### Added
- **MXFP6-PARO (W6A8) serving**: `QUANT=mxfp6 ./serve.sh`, `./setup.sh --mxfp6`,
  `./setup-paroquant.sh --mxfp6` — OCP MXFP6 E2M3 weights on the existing fp8-WMMA kernels
  (`launch6_p` / `launch6_at_p`), ported from ggz14's radiance-vllm-mxfp4 (PR #57 / upstream
  mirror) onto this fork's diverged `sly/mxfp4/` kernel base via a 3-way merge against the
  common ancestor — this fork's own decode-band tuning (`decode_bk64`, `dec_tune16`, A-tiled
  multi-consumer tracking) and ggz14's MXFP6 staging land in the same files without conflict.
  `paroquant/radiance_paroquant_mxfp4.py` (`ParoQuantMXFP6Config`) taken wholesale from the PR —
  this file was untouched on `main` since the merge-base. The MXFP6 checkpoint is a build
  artifact of `paroquant/build_mxfp6.py` (0.5.0), not an external dependency.
- **`RADIANCE_MXFP6_FORCE_TP1`**: upstream validated MXFP6-PARO on TP>=2 only
  (`rad_require_tp2` gate); this override allows exploratory single-card testing. Deliberately
  undocumented in the README options table — not a supported configuration, for this fork's own
  GPU-window testing only.

## [0.5.0] - 2026-10-03

### Added
- **Opt-in KV-cache offload** (`KVCACHE=off|ram|disk`, off by default): a second-tier prefix cache
  behind the GPU prefix cache for long-context / agent-style sessions, where every turn re-sends the
  whole conversation and the prefix outgrows the GPU pool. `KVCACHE=ram` adds a RAM tier in `/dev/shm`
  plus the 6 behavioural patches (mixed-hit, eagle-groups, mamba-stride, reconcile-reask, swa-align/
  touch, align-last-block); `KVCACHE=disk` adds a filesystem secondary tier plus the full 15-patch
  instrumented set and wants a host `kvcache-reap.sh` reaper. Off by default: with `KVCACHE` unset the
  serve command is byte-identical to the unmodified script. The offload patches, `kvwatch.py` and the
  `turnbench`/`tierbench` benches ship in `kv-cache/` and are applied at container start via the
  existing `/patches` mount (not baked).
- **MXFP6-PARO build tooling**: `paroquant/build_mxfp6.py` (the RTN builder with the per-row
  exponent-spread clamp) and `requant.sh FORMAT=mxfp6` (E2M3 grid) — the host-side tools to produce a
  MXFP6 W6A8 checkpoint from a base model. The serving path (MXFP6 kernel + loader integration,
  `QUANT=mxfp6`) lands with the kernel port in a follow-up; this release ships the tooling and the
  KV-offload feature only.

### Measured
- KV offload, one R9700 (gfx1201), TP=1, `KVCACHE_RAM_GIB=16`, three ~110k-token agent-style
  sessions served round-robin: a ~81k-token request drops from **38.8 s** (no cache) to **13.2 s**
  (RAM tier), tokens served from the tier in brackets. Numbers from `kv-cache/README.md`.
## [0.4.7] - 2026-10-04

### Changed
- **Faster image builds, same image.** radiance's HIP extensions (R4D, MXFP4 fp8, GDN decode,
  autoround/escha/paroquant) are compiled in their own `kernels` stage that copies only their
  own sources: a Python- or patch-only change no longer recompiles them (~2 min per build), and
  BuildKit builds them beside the patch chain. The three quant plugin kernels compile in
  parallel. The builder stage keeps a ccache in a BuildKit cache mount for triton, torchvision,
  aiter and vLLM, so a stack patch bump recompiles only what changed (torch not yet wrapped).

## [0.4.6] - 2026-10-03

### Fixed
- **Ship `radiance_lmhead_int2.py` into the image** (issue #90): the module was wired into
  the Dockerfile patch loop (`patch_lmhead_int2`) but omitted from the release-stage
  site-packages `COPY` allowlist, so `import radiance_lmhead_int2` raised
  `ModuleNotFoundError` in every image and the int2 lm_head feature silently fell through to
  int4/fp8/stock — `RADIANCE_LMHEAD_INT2` was a no-op because the module never reached disk.
  Added the missing allowlist entry; the module now lands in `${SP}/` like its int4/fp8 siblings.
- **README quickstart version** (issue #90): `docker pull` and `docker run` examples pointed at
  the stale `0.4.0` tag; updated to `0.4.6`.

## [0.4.5] - 2026-10-02

### Added
- **Self-service HSA/KFD diagnostics** (issue #73): when GPU enumeration fails while `rocm-smi`
  still sees the card, the startup precheck now prints the host-side facts that triage it
  without a GPU window — host kernel, `/dev/kfd` access, container pids limit — plus the
  recovery order (reboot first, raise the pids limit, then capture `strace`/`dmesg` evidence
  before rebooting). `docs` and `radiance_preamble` only; the default GPU path is unchanged.

## [0.4.4] - 2026-10-02

### Added
- **Greedy int2 lm_head** (`RADIANCE_LMHEAD_INT2=1`, `sly/mxfp4/radiance_lmhead_int2.py` +
  `sly/patch_lmhead_int2.py`): two-stage vocabulary head — a 2-bit coarse pass over all 151,936 rows
  and a bf16 re-rank of the top-16 — with roughly 7.5× less lm_head weight traffic per decode step
  than the int4 head (≈1.26 GiB saved at [151936, 5120]). Greedy-only: it declines under sampling
  and whenever `RADIANCE_LMHEAD_INT4` is active, so the default path is byte-identical.
  Pre-flight: `sly/check_lmhead_int2.py` (real-weight coverage gate) and 11 CPU unit tests.
- **Drafter bf16 pre-expansion** (`RADIANCE_DFLASH_BF16=1`, default off): the DFlash2 drafter's W4A16
  rows are dequantized once at the post-load hook into per-layer bf16 weights, and its `apply_weights`
  runs plain bf16 GEMMs; +3.22 GiB VRAM for 3.88× the hot-path weight budget. The default path is
  byte-identical (template parse + packer round-trip validated).
- **W4A8 prefill cell sweep** (`sly/mxfp4/bench_prefill_cells.py`): the prefill counterpart of
  `bench_decode_cells.py` — M = 96…8192, row-major vs fragment-tiled arms, `--check` validates every
  cell against an independent fp32 reference before any timing counts.
- `sly/audit_kernel_binary.sh`: a CPU-only VGPR/spill audit of the compiled kernel binaries
  (`--vgpr-max 213 --spill-max 0 --strict`), run against the image's .so as a per-kernel record.

### Measured
- CPU only so far: all 11 int2 unit tests pass and the packer round-trip is bit-exact; the greedy
  coverage gate (≥ 99.5 % in the coarse top-16 on real weights) runs in a 300 W GPU window, and the
  production numbers follow at the acceptance gate.

## [0.4.3] - 2026-09-30

### Added
- **Weight-only NVFP4 (NVFP4A16) checkpoints load.** `patch_nvfp4_mxfp4.py` routes compressed-tensors'
  `CompressedTensorsW4A4Fp4(use_a16=True)` branch to the radiance requant scheme under
  `RADIANCE_NVFP4_MXFP4=1`; stock forces Marlin there and aborted the load on ROCm
  (`bottlecapai/ThinkingCap-Qwen3.8-27B-NVFP4`).
- `radiance_nvfp4_diag.py` + `RADIANCE_NVFP4_DIAG=native|requant|fold` (`RADIANCE_NVFP4_DIAG_A8=1`):
  measurement-only NVFP4 schemes (bf16 dequant per forward, `--enforce-eager`) -- the exact checkpoint as
  reference, the requant with bf16 activations, and an emulated native NVFP4 kernel path ("NV fold").
- `sly/bench_fidelity.py` / `sly/check_fidelity.py`: deterministic prompt-logprob quality gate (top-20,
  14k tokens, dNLL ±0.012, KL, top-1 agreement) for numerics changes; `sly/check_nvfp4_diag.py` CPU check.

### Measured
- ThinkingCap NVFP4A16 (docs/TECHNICAL.md, *What the requantization costs*): the NVFP4 -> MXFP4 requant
  costs +0.093 NLL against the exact checkpoint (top-1 agreement 91 %); fp8 activations +0.006, the GDN
  a/b gates in MXFP4 vs bf16 ±0.000. An emulated NV fold (e2m1 x e4m3 block scale folded into the e4m3
  weight byte, as the kernel folds the MX exponent) with fp8 activations: +0.005 (~95 % of the loss
  back) -- the candidate for a native NVFP4 kernel path. int4 lm_head + int4 embedding: +0.006.
- Speed of the ThinkingCap serve path (300 W, compiled + DFlash2, short prompt): step 34.5 ms, i.e. the
  Quark step gap; `RADIANCE_NVFP4_BF16_LAYERS=` (bf16 a/b gates) is +0.6 ms/step slower for no measurable
  accuracy, so the `in_proj_ba` default stays (its comment no longer cites the rejected GDN merge).
## [0.4.2] - 2026-09-30

### Changed
- Pin the PyTorch stack to the pre-ROCm-10.1 "sanctioned trio": **torch 2.11.0** + triton 3.6.0
  + torchvision 0.24.1, down from torch 2.14.0 / triton 3.8.0 / torchvision 0.29.0. Per
  [ROCm/ROCm#6406](https://github.com/ROCm/ROCm/issues/6406), torch >= 2.12 causes abnormally
  high CPU usage after the first GPU operation; downgrading to torch 2.11 resolves it (the
  bug is expected to be fixed in the ROCm 10.1 release, after which the stack can be bumped
  again). `torchaudio` is not part of this image's stack, so no pin is added for it.

## [0.4.1] - 2026-09-27

### Added
- Split-K for the W4A16 Triton GEMM (`sly/patch_w4a16_tiles.py`, knob `RADIANCE_W4A16_SPLITK`, default on):
  the stock kernel runs the whole K loop in one workgroup, so the N = 5120 projections at decode M get
  160–320 workgroups for 64 CUs and stay latency-bound (INT4 target `down_proj`: 139 µs against 82 µs
  for the MXFP4 split-K kernel on the same shape). Per-shape entries in `_GFX12X_SPLITK`; a shape
  without an entry runs exactly as in 0.4.0. Affects compressed-tensors INT4 targets and the DFlash2
  W4A16 drafter. The same kernel carries two cheaper dequant schemes (scale applied to the fp32 tile
  result; bf16 magic-number nibble conversion with the zero point folded into the tile result) and an
  interleave-free unpack, selectable per table entry. Split-K partials stay <= 1 MiB (the CUDA-graph
  memory estimate charges them to the KV pool); `RADIANCE_W4A16_SPLITK_TABLE=<json>` loads a swept
  table without a rebuild.
- Serial split-K reduce: the program that finishes a tile last sums the partials in split order
  (per-tile counter, self-resetting) -- deterministic, and one launch less per split call (134 per
  step on the INT4 target at one request).
- Tiled W4A16 weight layout (`RADIANCE_W4A16_TILED`, default on): 16-row x 128-K blocks of 1 KB, so
  every row group a tile reads is one contiguous burst (the MXFP4 kernel's `WPERM` idea). The HIP
  skinny path for M <= 5 reads the row layout and is skipped for tiled weights.
- Fusions for W4A16 layers (`sly/patch_w4a16_fuse.py`): MLP gate_up + silu(gate) * up in one GEMM
  (`RADIANCE_W4A16_SILU`); GDN in_proj_ba re-quantized to int4 rows of in_proj_qkvz, one GEMM with two
  contiguous outputs (`RADIANCE_GDN_BA_W4`, INT4 target; frees the bf16 ba weights); the DFlash
  context-KV projection on the drafter's own int4 rows instead of a 105 MB bf16 copy read every step
  (`RADIANCE_DFLASH_KV_W4`, exact, frees ~78 MB for the KV cache); the DFlash2 grouped-conv
  kernel_projection (bf16, 10 calls per step) re-quantized to int4 (`RADIANCE_DFLASH_CONV_W4`; changes
  proposals only, the target verifies every token).
- The int4 lm_head in the tiled layout too (`RADIANCE_LMHEAD_INT4_TILED`, default on with
  `RADIANCE_W4A16_TILED`); it is the largest W4A16 GEMM of a step (2 x 656 MB).
- `sly/check_w4a16_fuse.py`: the post-load transforms end to end on stand-in layers against the unfused
  computation (fused silu, merged split, context-KV, conv projection, plain calls into transformed layers).
- `sly/check_w4a16_splitk.py` (numerics against an fp32 reference and bit-identity of the new paths,
  CPU via the Triton interpreter or GPU) and `bench_w4a16_tiles.py --splitk`.

### Measured
- One R9700, 300 W, production arguments, one GPU window (2026-09-28), step gap from streamed greedy
  requests (median chunk interval, deterministic): **RedHatAI/Qwen3.8-27B-INT4 42.70 → 37.43 ms
  (−12.4 %)**, 36.69 ms (−14.1 %) together with `RADIANCE_SKINNY_GEMM=all`; the control arm repeated
  at 42.76 ms. **Quark MXFP4 (production) 34.47 → 34.15 ms (−0.9 %, drafter GEMMs).** At unchanged
  acceptance (4.29 / 4.21 tokens per step) INT4 decode is ~125 instead of 107.8 tok/s; the gap to
  MXFP4 shrinks from −19 % to −7 % step time. At 8 concurrent requests INT4 stays far behind
  (297 vs 426 tok/s: M = 40–64, where the W4A16 kernels reach only ~250 GB/s).
- GSM8K 200 (cot zero-shot, greedy): INT4 0.825 (0.820 before), MXFP4 0.850 (baseline 0.835–0.845).
- Window E (2026-09-29, 300 W, tiled weights, serial split-K, all fusions): kernel sweep at M = 8 --
  down_proj 150.8 -> 87.7 us, out/o 70 -> 40, GDN qkvz 110.6 -> 79.2, attention qkv 98 -> 72, gate_up
  206 -> 176, drafter context-KV 194 (bf16) -> 54, drafter qkv / o / fc 47 / 48 / 207 -> 37 / 32 / 126; the
  tiled layout alone is worth 1.1-1.6x at M <= 16. Quark MXFP4 (production), all new knobs off -> on:
  step gap **34.52 -> 33.72 ms (-2.3 %)**, KV pool **384,316** (the split-K loss of 405 tokens is
  offset now that the drafter's context-KV no longer needs a bf16 copy). The INT4 target arms of that
  window were not run (stopped early).
- Window D (2026-09-29, split-K table of window B): INT4 step gap 42.68 -> 36.53 ms, BetterBench decode
  (128 runs) **INT4 127.5 tok/s against MXFP4 128.1** in the same window; GSM8K INT4 0.83.
- Window F2 (2026-09-30, image 0.4.1-rc5 with the window-E table, second starts): **INT4 all knobs off ->
  on (with `RADIANCE_SKINNY_GEMM=all`) 42.73 -> 34.61 ms (-19.0 %)**, within 3 % of MXFP4 (33.61 ms);
  KV pool 370,561 -> 375,820; GSM8K 200 0.835. Tokens per step 4.52 -> 4.24 on the four fixed prompts is
  a different greedy text, not the drafter (MXFP4 moves as much between windows, 4.41 / 4.70).
  **MXFP4 (production) all knobs on: 33.61 ms, BetterBench decode 131.0 tok/s** (128 runs; 0.4.0 in
  production: 130.4), KV pool 384,316, concurrency 1 / 8: 121.5 / 416.1 tok/s. The tiled int4 lm_head
  alone: 33.74 -> 33.61 ms (-0.4 %, bit-identical output).
- Not adopted: merging GDN in_proj_ba into in_proj_qkvz (96 extra rows add a 129th tile -- 93.9 us
  merged vs 79.2 + 3.6 separate), so `RADIANCE_GDN_BA_W4` defaults to 0; a decode-attention retune
  at long context (the shipped rule is within 1 % of the best cell, ~470 GB/s at 32k); W4A8 int8.
- Kernel level (M = 8, DRAM-cold, `bench_w4a16_tiles.py --splitk`): down_proj 150 → 95 µs,
  out/o 61 → 48, gate_up 200 → 178, GDN qkvz 106 → 82, attention qkv 89 → 76, drafter qkv 53 → 39,
  drafter fc 197 → 131.

## [0.4.0] - 2026-09-25

### Changed
- **vLLM 0.29.0 → 0.30.0**, aiter 0.1.21.post2 → 0.1.22.post1, torchvision 0.24.1 → 0.29.0, ubuntu:24.04
  digest, all Python pins re-resolved for vLLM 0.30 (`constraints.txt`: 209 pins, 34 changed, new
  `mooncake-transfer-engine-rocm` and `msgpack`; `setuptools` held at 79.0.1 because vLLM requires `<80`).
- Four patch anchors moved in vLLM 0.30 and were re-anchored for both versions (`apply_any`):
  `sly/patch_quark_mxfp4` (multi-line `_resolve_backend_kernels`), `sly/patch_lmhead_fp8` / `_int4`
  (`get_quant_method` now opens with `get_quant_method_target`), `patch_autoround` / `patch_escha`
  (`__all__` gained `resolve_quant_method`). `patch_unpad`, `patch_qwen3_toolparse` and
  `sly/patch_mamba_align_retire` are no-ops now: vLLM ships the same fixes.
- CI: ruff 0.16.8, renovate action v46.3.3, `create-github-app-token` v3.
- Renovate: no hourly or concurrent PR cap any more (updates were piling up rate-limited).
- Docs: README cut down to stack, reference numbers, quickstart, changes, options; details moved to
  `docs/`; this changelog; GitHub releases carry the changelog section.

### Measured
- Performance-neutral against 0.3.6 (prefill ±0.3 %, step gap unchanged, KV pool 384,316), GSM8K 0.855.
  Reference run: 133.2 tok/s decode, 3,166 tok/s prefill at 1.5k, 410 tok/s at conc 8 (300 W);
  124.5 / 2,527 / 362 at 210 W ([details](docs/BENCHMARKS.md#040)).
- New recommendation for shorter contexts: `--max-model-len 131072 --max-num-batched-tokens 4096`,
  +3.5 … +4.8 % prefill on 8k–64k prompts.

## [0.3.6] - 2026-09-24

### Added
- **NVFP4 checkpoints** via load-time requantization to MXFP4 (`RADIANCE_NVFP4_MXFP4=1`, ggz14's
  `radiance_nvfp4`), onto the same W4A8 kernel: `unsloth/Qwen3.8-27B-NVFP4` serves at the Quark
  decode speed, KV 374,202 tokens, GSM8K 0.825. Fixed a meta-tensor crash at load on the way.
- **ParoQuant** (`RADIANCE_PAROQUANT=1`), **AutoRound** (`RADIANCE_AUTOROUND=1`) and **escha**
  (`RADIANCE_ESCHA=1`) quantization configs and HIP kernels (ggz14), all opt-in. ParoQuant works on
  one card (GSM8K 0.84) but is not tuned for it (decode 67.7 tok/s).

## [0.3.5] - 2026-09-24

### Added
- **libr4d extras** (ggz14's rx10 patch, rebased onto our pin): GDN decode/prefill kernels for the
  bf16 SSM cache now bind instead of the FLA fallback. **Prefill +4.9 … +6.0 %**, step gap −0.7 ms,
  decode +1 %, KV pool unchanged, GSM8K 0.845.
- Multi-GPU paths for other users (untested on the maintainer's single card): TP=3 via zero-weight
  dummy heads (`RADIANCE_TP_PAD=3`, needs `RADIANCE_MXFP4_WPERM=0`), TP=2 all-reduce knobs
  (`RADIANCE_AR_MAX_KB`, …); TP=4/8 use libr4d's N-rank kernels.

## [0.3.4] - 2026-09-24

### Added
- froggeric's fixed Qwen chat template v22.5 at `/opt/qwen-fixed.jinja` (`--chat-template`).

## [0.3.3] - 2026-09-23

### Added
- HIP port of vLLM's fused GDN MTP decode kernel (`RADIANCE_GDN_FUSED_DECODE=1`, default off).

## [0.3.2] - 2026-09-24

### Changed
- Image halved (9.4 → 4.8 GB: ROCm prune to gfx1201, venv split into a stable and a volatile layer,
  so a release pull is ~35 MB); every PyPI dependency pinned in `constraints.txt`.

## [0.3.1] - 2026-09-23

### Changed
- `RADIANCE_MXFP4_WPERM=1` (fragment-order weights) permutes into the weight's own storage: step gap
  38.5 → 37.2 ms, weighted decode +3.7 %, KV pool back to 384,316.

## [0.3.0] - 2026-09-23

### Added
- Fragment-tiled activations from the fused norm/quant producers feed the A-tiled prefill GEMM
  (`RADIANCE_MXFP4_A_TILED_MIN_M=513`): **prefill +11 … +13 %**, bit-identical.

## [0.2.9] - 2026-09-21

### Added
- Prompt lookup on top of the DFlash draft (`RADIANCE_LOOKUP_DRAFT`, default on): text that repeats
  something more than 2048 tokens back (edits, quotes) decodes **+57 … +116 %** faster, lossless.

## [0.2.8] - 2026-09-20

### Added
- Split-KV launch for the DFlash2 drafter's sliding-window attention: 1.8–3.5× per call, step
  −0.2 … −1.6 ms.

## [0.2.7] - 2026-09-20

### Fixed
- The fp8 verify-batch decode attention tune never applied since aiter 0.1.21; ported to the new config
  tables. **Long-context decode +56 % at 33k, +122 % at 65k, +147 % at 99k tokens.**

## [0.2.6] - 2026-09-19

### Changed
- Ledger: unused upstream patches documented with reasons.

## [0.2.5] - 2026-09-18

### Fixed
- Load-time `max_split_size_mb` allocator scope on ROCm: KV pool +10k tokens (MXFP4), +73k (INT4).

## [0.2.4] - 2026-09-17

### Added
- int4 lm_head for compressed-tensors targets, INT4 target tile table.

## [0.2.3] - 2026-09-17

### Fixed
- Backport of vllm#55450: Mamba align-mode state retirement across null gaps (a 258k prompt exhausted
  the KV pool and preempted itself).

## [0.2.2] - 2026-09-17

### Added
- KV group size override and int4/int8 `embed_tokens`: +1.2–1.7 GiB for the KV cache.

## [0.2.1] - 2026-09-16

### Changed
- Measured decode cell table for M in (8, 64] (2–8 concurrent sequences).

## [0.2.0] - 2026-09-16

### Changed
- Merged the full upstream/ggz14 history (243 commits); the active image stayed the same.

## [0.1.6] - 2026-09-16

### Added
- Fused add+RMSNorm / SiLU·mul / GDN gated norm + fp8 quant in front of the W4A8 GEMMs: conc 1 +3 %,
  conc 8 +5.6 %.

## [0.1.5] - 2026-09-15

### Added
- int4 lm_head (W4A16, group 128): step 42.0 → 39.7 ms at conc 1, +8k KV tokens.

## [0.1.4] - 2026-09-15

### Added
- gfx1201 tile table for the DFlash2 W4A16 drafter GEMMs: step 43.4 → 42.0 ms.

## [0.1.3] - 2026-09-15

### Added
- fp8 lm_head: step 48 → 44.5 ms, +1.27 GiB KV.

## [0.1.2] - 2026-09-15

### Added
- DFlash2 with a W4A16 drafter; `DEC_MAX_N` 36864 puts `gate_up` on the decode kernel: step 67 → 48 ms.

## [0.1.1] - 2026-09-14

### Added
- gfx1201 `gemm_afp4wfp4` config for AITER (the gfx950 default crashed on RDNA4's 64 KiB LDS).

## [0.1.0] - 2026-09-13

### Added
- Fork of StillDeadcode/vllm-radiance with ggz14's MXFP4 work, ported to vLLM 0.29.0.
