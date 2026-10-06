# NOTES-D: adaptive verify width, graph-safe (lever 1)

Branch `worktree-agent-af5526795e4a04811`. New: `sly/radiance_adaptive_width.py` (policy, pure Python),
`sly/patch_adaptive_width.py` (wiring, in the Dockerfile loop and COPY list, row in `sly/README.md`),
`tests-lessons/D/{test.sh,greedy_eq.py}` (copy on 2408 in `/root/lessons/D/`). No VERSION/CHANGELOG/README
change (main session). Default `RADIANCE_ADAPTIVE_WIDTH=0`: the hooks return immediately, no extra graphs.

## 1. Why ggz14's `patch_dynwidth` lost (-3 % at conc 4/8/16)

Read from the installed 0.7.0 tree (`/Users/.../src-070/vllm`), not measured:

1. **FULL graph needs a uniform decode batch.** V2 runner: `gather_batch_req_state` ->
   `get_uniform_decode_token_count` returns the query length only if `num_tokens == max_query_len *
   num_reqs` and no row is a prefill. `CudaGraphManager.dispatch` -> `_is_compatible` then needs
   `desc.uniform_token_count == that count`. The FULL descriptors are captured for exactly one query
   length (`decode_query_len` = 8; `varlen_decode` is False unless upstream's `adaptive_verification`
   manager exists) and `_init_candidates` keys them by the token count, so even a uniform batch at width
   3 (4 rows each) has no graph. Anything else falls through to the PIECEWISE descriptor (or eager).
   ggz14 capped each request to `ceil(EMA)+2`: one capped request makes the batch non-uniform, i.e. almost
   every step.
2. **PIECEWISE is expensive on this model.** `splitting_ops` contains `unified_attention_with_output`
   and `qwen_gdn_attention_core` (compilation.py:773/782): 16 attention + 48 GDN layers = ~65 graph pieces
   with the attention and GDN custom ops run eagerly in between (CPU launch + metadata per piece, the
   per-layer FULL-graph fusion of launches is gone). That cost is of the same size as what trimming saves:
   a verify row costs ~0.39 ms (33 ms at M 8, 55 ms at M 64), ggz14's cap saved ~10 % of the rows
   (~2-3 ms at c8) and lost the FULL graph on top.
3. Upstream's own varlen route does not help: `AdaptiveVerificationManager` (DSpark) requires every
   target attention builder to report `AttentionCGSupport.ALWAYS` and `supports_device_cpu_query_lens_mismatch`
   (only FlashInfer / MLA indexer do). `ROCM_AITER_UNIFIED_ATTN` (via RocmAttentionMetadataBuilder) is
   ALWAYS, but the GDN builder is `UNIFORM_BATCH` and the unified backend lacks the mismatch flag; it
   also trims on the device (CPU keeps an upper bound), which this stack cannot take. Not usable.
4. **The drafter context pass does shrink** (good, same as Radiance): DFlash `propose` runs
   `precompute_and_store_context_kv(hidden_states[:num_target_tokens])` eagerly, with `num_target_tokens`
   = the target step's rows. The query pass is a fixed 1+7 uniform block (FULL graph, TRITON_ATTN is ALWAYS,
   independent of the verify width), so shortening costs the drafter nothing. Drafts stay 7 per request on
   the GPU; the worker consumes the first `len(scheduled_spec_decode_tokens[req])` (`combine_sampled_and_draft_tokens`,
   `cu_num_logits`), so a prefix verify is lossless.
5. **Async scheduling:** `AsyncScheduler._update_after_schedule` assigns every decode request the SAME
   shared placeholder list of `num_spec_tokens_to_schedule` entries; its length is what next `schedule()`
   turns into verify rows (`num_tokens_with_spec`). `update_draft_token_ids` is never called there, so a
   cap in that function (ggz14's first patch) does nothing under async; the second patch copies the list
   per request. Observations (`update_from_output`) lag one step behind the decision (step N+1 is scheduled
   before step N returns), which the exponential estimate absorbs.
6. Upstream already ships a static, batch-size keyed uniform width (`num_speculative_tokens_per_batch_size`)
   with FULL graphs per tier, but only for a fixed schedule and it only adds graphs at `round_up(size, q)`
   token counts (a batch of 5 at q=6 = 30 tokens has no graph). Not acceptance-driven; not used.

## 2. Design (as Radiance `adaptive_k`)

Objective per step: maximise `sum_i E_i(k_i) / T(M)`, `E_i(k) = 1 + sum_{j<=k} q_i^j`, `M = sum(1+k_i)`,
`T(M) = a + b*M` (env; default a 29.9, b 0.393 = 33 ms @ M 8, 55 ms @ M 64 at 210 W; optional table).
Widths from `{1,3,5,7}`, nothing shrunk while the unshrunk rows are < 32 (single stream bit-identical),
`q_i` censored (`seen = a + (a<k)`, decay 0.25, prior 2 pseudo-positions to the pool mean, pool decay 0.99).
The decision for step N+1 is taken where the placeholders of step N+1 are assigned (async:
`_update_after_schedule`; sync: end of `update_draft_token_ids`) over the decode requests of the step just
scheduled.

- `perseq`: marginal-gain greedy over the ladder, stop at the best `tokens/T`, total rows restricted to
  multiples of 8 (`RADIANCE_AW_QUANT`). Graphs: varlen FULL decode graphs (capture sizes 16..64, 8 request
  slots, promised `max_query_len` 8) next to the uniform ones. The uniform graph is first in the candidate
  order (full-width batches replay as today); the varlen graph catches every other pure decode batch in
  which each request verifies >= 2 rows. Dispatch guard: a batch with a prefill row or a row without drafts
  gets `max_query_len=None` and can never match a varlen graph (GDN's metadata for those batches is built
  on a different path). Capture uses the promised `max_query_len` in `mamba_hybrid.prepare_attn`
  (the hybrid model state used the dummy batch's own maximum, which would bake a too small q-grid).
  Why this should work: unified attention, `causal_conv1d_update`, the GDN recurrent kernels and the
  sampler read `query_start_loc` / `cu_seqlens` at run time; padded requests are zero-length rows with
  `NULL_BLOCK_ID` state, the case a partially filled graph already has.
- `uniform`: one width for the batch (same objective, common k) so the batch stays uniform. The patch
  captures a uniform FULL graph for each (n requests, q = w+1), n >= `RADIANCE_AW_GRAPH_MIN_REQS` (4)
  up to `max_num_seqs`: 15 extra graphs at max-num-seqs 8 and lens 1,3,5. Safe by construction: each graph
  is a normal uniform decode graph, only the query length differs (kernels see exactly the capture shapes).
  Captures only in the target's `ModelCudaGraphManager`, not the drafter's.

## 3. Knobs

| env | default | note |
|---|---|---|
| `RADIANCE_ADAPTIVE_WIDTH` | `0` | `0`, `uniform`, `perseq`. Recommend after the A/B (expected: `perseq` if its graphs hold, else `uniform`). |
| `RADIANCE_AW_LENS` | `1,3,5,7` | widths tried (depth always added). Fewer values = fewer uniform graphs. |
| `RADIANCE_AW_COST_A` / `_B` | `29.9` / `0.393` | ms, `T(M)=a+b*M`; `RADIANCE_AW_COST_TABLE="8:33,16:38,64:55"` overrides. |
| `RADIANCE_AW_MIN_ROWS` | `32` | unshrunk row floor. |
| `RADIANCE_AW_QUANT` | `8` | perseq total rows multiple (capture sizes). |
| `RADIANCE_AW_ALPHA` / `_PRIOR` | `0.25` / `2` | estimate decay / pool prior. |
| `RADIANCE_AW_VARLEN_GRAPH` | `1` | perseq: `0` = per-request widths on PIECEWISE (ggz14's setting; the diagnosis arm). |
| `RADIANCE_AW_GRAPH_MIN_REQS` | `4` | uniform: smallest batch with tier graphs. |
| `RADIANCE_AW_LOG_EVERY` | `500` | policy counter line every N decisions: shortened share, rows/step full/chosen/saved, width histogram. |
| `RADIANCE_AW_GRAPH_STATS` | `0` | diagnostic, independent of the mode: cumulative steps per route (decode-only vs prefill+ x FULL/PIECEWISE/NONE). |

## 4. Expected effect

Radiance: c8 arrivals 390.7/372.0 -> 420.6/415.9 tok/s (+9.8 %), steady c8 +5.7 %, step 60.5 -> 51.8 ms,
6.0-6.8 rows/sequence instead of 8, 954/2048 steps shortened, 7.1 rows saved per step. Here a row costs
~0.39 ms (Radiance 0.43) plus the drafter context rows, so `perseq` with working graphs should land at
+4...+8 % steady c8 and a bit more with arrivals (mixed prefill steps are PIECEWISE in any mode, so
shortening is pure gain there). `uniform` cannot exploit the 2x acceptance spread between requests;
expect roughly half of that. Single stream: identical output and step (floor of 32 rows).

## 5. Risks / not verified (no GPU used)

- Nothing was run on a GPU. Patch anchors verified: against the pinned upstream tree via
  `ci/patch_dryrun.sh` (112 hunks, pass 2 all NOOP) and applied twice inside the real image on 2408
  (CPU only). Policy logic unit-tested offline (monotone choice, quantisation, floor, uniform argmax).
- `perseq` varlen FULL graphs for the GDN + unified-attention stack are reasoned from the kernels' inputs,
  not proven: if the capture asserts, the server will not start (the test script reports it); if replay is
  wrong the concurrent-greedy common prefix will drop below the off-vs-off2 noise. Fallback: `uniform`, or
  `perseq` with `RADIANCE_AW_VARLEN_GRAPH=0` (correct, but PIECEWISE).
- Padded token rows of a varlen graph (real tokens < graph tokens) hold stale/garbage activations
  (GDN/attention write nothing past the last real row); rows are independent through the MLP/GEMMs, but
  a NaN there would not be caught. The greedy equality probes cover it only indirectly.
- Extra FULL graphs cost graph-pool memory that `profile_cudagraph_memory` extrapolates from the two
  largest graphs: check the KV pool stays 384,316 tokens (acceptance criterion 5 in `test.sh`).
- Decision (step N+1's widths) uses the request set of step N; a request finishing in between changes the
  batch (uniform: a batch of < 4 requests gets no tier graph -> one PIECEWISE step; perseq: still caught
  by varlen). New requests padded by the scheduler's "new decode request" pad (full-prefix-cache hit
  only) keep the full width, so that one step is non-uniform.
- Sync scheduling path (`update_draft_token_ids` hook) is implemented but untested; production runs async.
- Not compatible with data parallelism (per-rank widths) and not tested with structured output.
- A/B noise: use off / mode / off2 (the script does) and judge against |off - off2|.
