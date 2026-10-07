# NOTES-H: R4D_HYBRID attention backend (libr4d prefill for long prefill runs, AITER for everything else)

Status: code, CPU self test and patch dry run done; nothing has run on the GPU. Test script:
`/root/lessons/H/test.sh` on 2408 (copy: `tests-lessons/H/`).

## 1. Why is `--attention-backend R4D` decode slow here (60.9 vs 34.9 ms/step)? What is established

Measured (G1 vs control, 210 W): step 60.85 vs 34.9 ms, flat across all six prompt categories (60.3 to 61.6, so it
does not scale with acceptance or context), tokens/update 3.18 vs 3.27, TTFT +7 (72 tokens), +10 (256), +12 ms (904).
The constant offset of ~25 ms is the thing to explain.

Eliminated by reading the code and the G1 log:

* **The decode kernel itself.** `r4d_attn_decode_kernel` takes `ctx = seqused_k[seq]` from device memory and splits
  `ceil(tiles(ctx)/splits)`; the `max_ctx = max_model_len` baked at capture only picks the split COUNT (`128/(kv_heads*seqs)`
  = 32 for one request). At the prompt lengths of accprobe (a few hundred tokens) one attention call is microseconds, so
  25 ms / 16 layers = 1.6 ms per call cannot be kernel time. Radiance runs the same kernel at 32.4 ms/step.
* **Graph capture.** G1 log: `Capturing CUDA graphs (FULL) 8/8` for the target, same as the control. R4D's builder says
  `UNIFORM_BATCH` (control: `ALWAYS`), which is enough for the uniform 8-token verify batch; no "not supported ... setting
  cudagraph_mode=PIECEWISE" warning in the log.
* **Verify rows treated as prefill runs.** `_plan` cuts runs by equal q_len; a verify batch (q_len 8) is `<= MAX_DECODE_QLEN`
  (64/6 = 10) and goes to the decode kernel; the kernel itself rejects nothing.
* **Cascade attention, KV layout.** `use_cascade_attention` is False in both; R4D asks for LBHNC, which is also what
  AITER's `(LBHNC, LHBNC)` resolves to (first entry, no other worker backend declares layouts).

Differences that remain (R4D subclasses the TRITON backend, not the AITER one), in order of how well they fit a *constant*
25 ms cost:

1. **Host-side / launch-bound step instead of a GPU-bound one** (graph replay is the same, but the work around it is not:
   a different metadata builder, a different compile graph because `supports_quant_query_input` is False and
   `fused_output_quant_supported` is False, different custom-op nodes per layer). A step that is CPU-bound by 25 ms on
   this host would show exactly this: constant, independent of context and acceptance, and larger TTFT too.
2. **The 16-token kernel block** (`get_supported_kernel_block_sizes() == [16]`): manager page 880 instead of 896, block
   table with 16384 columns instead of 292 and a per-step 55x expansion of the block table in the model runner (G1 log:
   `Setting attention block size to 880`). Control shows `kernel_unified_attention_2d..._BLOCK_SIZE_896` in the E profile.
3. **No verify tune** (`radiance_attn_decode.py` is a wrapper on aiter's config, it never sees the R4D path). Not the
   cause of 25 ms (the tune is worth ~1-2 ms/step) but real.
4. bf16 queries instead of fp8 (control: fp8 q, folded into the previous op by torch.compile): small.

**What would settle it** (not done, needs the GPU): `/root/lessons/H/diag.sh` starts R4D and the control with the torch
profiler, decodes one request and prints, per arm, GPU busy vs span, idle share, kernels per step and the top kernels
(`trace_sum.py`). Reading: attention kernels dominate -> kernel/launch shape; large idle share -> host bound (1/2);
many more kernels per step -> not replayed. ~15 min, independent of test.sh.

The hybrid sidesteps all of this: the decode path is the AITER path, with AITER's builder, layout, block size and
graph class, so none of the R4D-specific decode differences exist in it.

## 2. KV layout and block size

Both backends read the same cache, no second layout is needed:

* Per layer a `(block, kv head, slot, 2*head_dim)` view (LBHNC), K and V packed per slot. R4D hard-requires it; AITER
  accepts `(LBHNC, LHBNC)` and picks LBHNC. The hybrid inherits AITER's tuple unchanged, so the KV pool is byte for byte
  the control's (manager block 896, kernel block 896, same page size, same 384k-token pool at the production setting).
* libr4d wants 16-slot blocks (the image's R4D backend therefore forces kernel block 16, which changes the physical
  layout and the page to 880). But libr4d takes the block and head strides as arguments:
  `offset = blk*block_stride + head*head_stride + (key % 16)*C`. A 896-slot block b is read as 56 sub-blocks (b, j) at
  element `b*S0 + j*16*C`; with `block_stride = 16*C` the kernel's block id is `b*(S0/(16*C)) + j`. The step builds that
  expanded block table once (first R4D layer of the step, cached on the metadata, 16 layers share it, full row width, a
  few tiny device ops), only on steps that contain a long run. Checked on CPU for LBHNC and an LHBNC-style stride
  permutation (`selftest_cpu.py`, 18 checks pass, run in the image with `HIP_VISIBLE_DEVICES=-1`).
* The geometry (stride(3)==1, stride(2)==C, N%16==0, S0%(16*C)==0) is verified at the first R4D call of each layer; a cache
  that fails it runs AITER only and logs why once.

## 3. What changed

* `radiance_r4d_hybrid_attn.py` (new): `R4DHybridAttentionBackend(RocmAiterUnifiedAttentionBackend)`, name `R4D_HYBRID`.
  Builder = AITER's builder plus a per-step run plan from `query_start_loc_cpu` (only computed when
  `max_query_len >= MIN_Q`, so a decode/verify step never pays for it and never differs from the control: `forward` is then
  literally `super().forward`). Impl = AITER impl; a step with long runs is cut into runs: equal-length requests with
  `q_len >= MIN_Q` -> one libr4d paged-prefill launch (fp8 or bf16 KV variant, bound through `radiance_r4d_attn`), the rest ->
  one AITER sub-batch (sliced `cu_seqlens`, `seq_lens`, block table; decodes riding along with a prefill chunk).
  q arrives fp8 from the layer (AITER needs that for decode), is widened to bf16 for libr4d (exact, times the static q scale).
  Fallback to AITER for the whole step on: no libr4d, geometry not head_dim 256 / GQA 6 / decoder, sliding window, alibi, sinks,
  soft cap, non-causal, fused output quantisation, non-bf16 output, graph capture, cache strides that do not fit. First reason logged once.
* `patch_r4d.py`: adds the `R4D_HYBRID` enum member (idempotent, anchored on the R4D line, runs in the existing loop entry).
* `Dockerfile`: `radiance_r4d_hybrid_attn.py` added to the module COPY list.
* `ci/patch_dryrun.sh`: OK (104 hunks, pass 2 all NOOP). `ci/check_consistency.py`: OK (no VERSION bump done, see below).

## 4. Knobs (all new, default off = unchanged image)

| Knob | Default | Recommended | Why |
|---|---|---|---|
| `--attention-backend R4D_HYBRID` | not selected | candidate for 1.0 if the test passes | the production change is this one argument |
| `RADIANCE_R4D_HYBRID_ROUTE` | 1 | 1 | 0 = never route to libr4d; the backend is then ROCM_AITER_UNIFIED_ATTN under another name (class A/B) |
| `RADIANCE_R4D_PREFILL_MIN_Q` | 512 | 512 (try 256, 1024) | smallest per-request q_len for libr4d; its prefill kernel owns 64 query rows per workgroup, a lone 256-token request fills 16 workgroups of 64 CUs; equal-length requests batch into one launch |
| `RADIANCE_USE_R4D` | 1 | 1 | existing master switch; 0 makes `R4D_HYBRID` refuse like `R4D` |
| `R4D_ATTN_FP8` | 0 | 0, see 5 | existing libr4d knob (QK8/PV8 legs), orthogonal |

README options table: add `--attention-backend R4D_HYBRID` plus `RADIANCE_R4D_PREFILL_MIN_Q` when adopted. VERSION/CHANGELOG
are left to the integration (minor bump: new backend).

## 5. R4D_ATTN_FP8 and accuracy

Kernel-level (libr4d patch comment, vs an fp32 oracle, row relRMSE): f16 legs 1.7e-3, QK8 1.7e-2, PV8 2.6e-2, both 3.1e-2;
speed +10 / +24 / +56 % at a hot 8k chunk, +18 % at a cold 40k chunk. fp8 KV caches only.

The control is NOT the accurate reference those numbers suggest: in aiter's `unified_attention.py` (read in the image) K and V
tiles are cast to `Q.dtype` and the probabilities with `P.to(V.dtype)`, and the layer hands the kernel an fp8 q
(`QuantFP8`, static scale 1.0). So the control already does QK^T in fp8 and P.V with fp8 P: the precision class of
`R4D_ATTN_FP8=3`. Consequences for the hybrid:

* Default (f16 legs): the long-prefill attention is MORE accurate than the control's, apart from the fp8 rounding of q that
  the layer applies before the kernel (unavoidable here; widened back to bf16 exactly).
* `R4D_ATTN_FP8=3` is then not a regression against the control, it matches its precision class, and is up to +56 % of WMMA issue
  rate. Candidate second step after the hybrid is verified; judge it by the prompt-NLL delta in `probe_h.py` and the GSM8K 200 gate.
  QK8 on top of the already-fp8 q rounds twice (e4m3 -> bf16 -> folded-scale e4m3), PV8 is the part that equals the control.

## 6. Expected effect

Prefill: the kernel is the one that gave +19 % at 64k and +9 % at 32k as the R4D backend (1874 vs 1572 and 2186 vs ~2000 at
210 W); the hybrid replaces exactly that kernel and adds a bf16 widening of q (25 MB per layer per 2048-token chunk) and one
block-table expansion per step, ~0.1 % of a chunk. Expect +15-19 % at 64k, +8-9 % at 32k, ~0 at 2k/8k (the attention share of the
profile is small there; R4D alone measured 2383 / 2479 tok/s at 2k / 8k). TTFT 72/256: identical (AITER path). TTFT 904: libr4d
(q_len 904 >= 512), expected equal or slightly better. Decode, c8: equal to control by construction (same builder, kernels,
graphs, tune). A verify batch riding with a prefill chunk now runs the tuned verify kernel in its own sub-batch (control: the
prefill config for the whole mixed batch), a small positive.

## 7. Risks

* **Not run on the GPU.** The libr4d launch ABI is copied from `R4DAttentionImpl.forward`; the stride/block-id trick and the run
  plan are verified on CPU only. First thing to look for in the log: `R4D_HYBRID: long prefill runs go to libr4d` and no
  `R4D_HYBRID: step runs AITER only (...)` line.
* The first long prefill after a start loads the libr4d prefill module (one-time, not covered by the warm-up; the test's first
  probe absorbs it).
* Greedy text of prompts that reach libr4d (>= MIN_Q tokens in one chunk) can differ from the control late in the text (different
  kernel, different rounding); prompts below the threshold take the control's path and must match exactly (tested).
* Prefix-cache hits shorten q_len: a long prompt with a big cached prefix may fall below MIN_Q and run AITER. Intended.
* Peak activation: ~50 MB of temporaries per long step that the memory profile run does not see; the production setting has
  ~1.6 GB headroom (and E2 shows 0.65 GiB with FULL_DECODE_ONLY).
* The new backend name changes the torch.compile cache key: first start compiles fresh (~6 min); start twice.
* The attention-output fusion (`output_scale`) and encoder layers fall back to AITER by design; if the production graph fuses the
  attention output quant (not observed), the hybrid would silently never route. The routing log line is the check.

## 8. Test: `/root/lessons/H/test.sh` (~45 min, GPU window)

Self test (CPU) -> C1 control -> H0 hybrid first start (compile, discarded) -> H hybrid -> C2 control; optional `H_OPT="fp8 minq256"`
(+9 min each, runtime env only). Per arm: `probe_h.py` first on a cold cache (short-prompt greedy text, long-prompt text, prompt
log-probs of ~3000 tokens), `accprobe3` greedy, TTFT 72/256/904, prefbench 2k/8k/32k/64k x2, c8 arrivals. Summary prints C1 vs C2
(noise floor) and C1 vs H text equality / NLL delta and the decision criteria (decode step and tok/upd equal within the C1/C2
spread, short text identical, prefill 32k/64k >= +10 %, 2k/8k and TTFT not worse, c8 not worse, KV pool equal).
`diag.sh` (~15 min, optional) answers section 1.
