# NOTES-B: gated fold (gate_up GEMM + SwiGLU + e4m3 quant), lever 4

## What changed

- `sly/mxfp4/radiance_mxfp4_fp8.hip`
  - `radiance_mxfp4_fp8_gemm_decode` gets a `GATED` template flag (+ `gI` argument). Requires DKS=1, DIRECT, no fp6.
    A block owns 64 output columns: waves 0-3 take gate rows `[g0, g0+64)`, waves 4-7 the matching up rows
    `[gI+g0, gI+g0+64)`. The weight is NOT re-laid-out (WPERM order or checkpoint order both work: a row->global-row
    map `grow(r)` replaces `n0 + r` in the two W staging paths). Epilogue: each half is rounded to bf16 exactly where
    the unfused C store rounds (`bf16((acc*rf)*As)`), parked in LDS (reuses `sW`, no extra LDS: 142 VGPR, 26112 B LDS,
    0 spill, same as the unfused kernel), then `bf16(bf16(g/(1+expf(-g))) * u)` is stored as bf16 `[M, I]`.
  - `radiance_silu_mul_quant` gets a `PRE` flag: input is already the product, so it only does amax + quant.
  - New pybind entry points: `launch_gated(a, w, ws, wref, as, p, M, N, K, stream) -> bool` (False = "the unfused path
    would not run the decode kernel at split-K 1 for this shape", nothing launched) and `launch_quant_rows(p, q, scale, M, N, stream)`.
    Served cell: same DWN=8 / BK=128 / DKS=1 / WPERM / NT statics the unfused `launch()` uses; M in 1..64, M <= RADIANCE_MXFP4_DECODE_MAX_M,
    N <= 36864, N % 128 == 0, K % 128 == 0, `split_k_for(N/128, M) == 1` (true for production gate_up, nblk 272, at every M).
- `sly/mxfp4/radiance_fused_norm.py`: knob `RADIANCE_MXFP4_GATED_FOLD`, custom op `radiance::mxfp4_gated_quant`
  (+ fake), gate `gated_ok(mlp)`, call site `gated_gemm_quant(mlp, x)`. The op owns the M branch (opaque to dynamo) and
  falls back to `mxfp4_linear_pq` + `silu_mul_quant` for every M the kernel declines (M > 64, tiled input, ...). All outputs are
  allocated inside the op (cudagraph capture safe, same as the existing ops); the `launch_gated` decision depends only on (M, N, K) and process env.
- `sly/patch_gated_fold.py` (in the Dockerfile loop after `patch_fused_norm_quant`): `Qwen2MoeMLP.forward` hook
  `if _rfn.GATED_FOLD and isinstance(x, tuple) and _rfn.gated_ok(self)` and the knob in `envs.compile_factors()`
  (only when set, so the default cache key is unchanged). Verified idempotent against the installed `src-070` tree.
- `tests-lessons/B/` = test.sh + check_gated.py + greedy_dump.py + cmp_greedy.py (also on 2408 in `/root/lessons/B/`).

## Knob

| name | default | recommended | why |
|---|---|---|---|
| `RADIANCE_MXFP4_GATED_FOLD` | 0 | 1 if the A/B confirms | needs `RADIANCE_FUSED_NORM_QUANT=1` (+ its SILU fusion) and the post-layernorm `(q, scale)` carrier; otherwise inert (logs `gated_fold=0`) |

## The design point that differs from Radiance

Radiance's `gemm_nt_q_gated` folds the activation quant into the epilogue because its fp8 activations carry a
GROUP scale (`group` parameter of `gated_quant_fp8`), so a block can quantise its own columns. vLLM's down_proj consumes a
PER-TOKEN scale (`As[M]`, one amax over all 17408 columns). That amax spans every block of the GEMM, so the quant cannot sit in
the epilogue without a cross-block reduction plus a second pass anyway. Done here: the epilogue folds the GEMM and the SwiGLU
(the bf16 `[M, 2I]` round trip becomes `[M, I]`), and quantisation stays a row-per-workgroup kernel (the existing one, `PRE`
mode). Launch count is unchanged (2 -> 2); saved traffic is the `gate_up` bf16 write+read halved plus the silu math moving off
the critical row kernel. Output is byte-identical to the unfused pair (same rounding points, same expf), so it is a pure
speed change; greedy and sampling trajectories cannot move. I could not read Radiance's `r4d_gemm_mxfp4a8_gated.hip`
(commit bb7d33f is not in the checked-out tree and git access to that repo was refused to this sandbox), so the
"why ks=1 / which M range" question is answered from vLLM's own tables: `split_k_for` gives gate_up (nblk 272) ks=1 at every M, which is
the only reason the epilogue can see the finished accumulator; narrower shapes (down, out_proj) want split-K and are not candidates.

## Expected effect

Radiance measured -0.43 ms (-1.3 %) per step single stream, which includes dropping a launch. Here the launch stays, so expect
roughly half to two thirds of that: ~64 layers x 3-7 us (M 8..64: 2.2-4.4 MB less traffic each way, mostly MALL-resident) =
-0.2 to -0.4 ms of a ~34 ms step (-0.5..-1 %). The microbench in check_gated.py gives the per-layer number directly; below ~2 us per layer the
knob is not worth the extra instantiations (16 new decode kernels, +~25 s hipcc).

## Risks

- A tiny numeric deviation would show as q byte mismatches in check_gated.py; the gate is bitwise, so nothing ships on a "close enough".
- 16 more decode-kernel instantiations (TM 1..4 x WPERM/NT) in the .so: image size/compile time only.
- Bandwidth: the epilogue's bf16 store is 128 B per row per block (64 columns), uncoalesced across rows; the same pattern as the
  unfused C store at DKS=1, but half as many bytes.
- Not covered: M in (64, 128] (decode band extension for c16: falls back to the unfused pair inside the op), TP > 1, fp6 (`wh`) layers.
- Could not run anything on the GPU (rules); compile-only verified on 2408 (no spills). Unverified until test.sh runs: bitwise
  equality, the microbench numbers, graph capture of the new op under FULL_AND_PIECEWISE, dynamo handling of the tuple-input branch in `Qwen2MoeMLP.forward`.
