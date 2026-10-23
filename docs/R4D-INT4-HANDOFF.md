# Handoff: int4 libr4d prefill route (`R4D_KV4I`) — accuracy fails, cause open

Written 2026-10-23 for an agent that continues this work. Repo: `SlyBase/vllm-sly-radiance`.
Everything below was measured; items marked **(hypothesis)** were not.

## 1. Goal

TetraAttention (`--attention-backend TETRA --kv-cache-dtype int4_per_token_head`, "KV4") prefill is slower than
the fp8 path with libr4d prefill attention (1.8.0): -1.4 % (1.5k) to -11.4 % (47k) at 210 W. The idea: let the
libr4d prefill kernel read the rotated int4 cache directly (route for long prefill runs), keep Tetra for decode,
verify and the head-128 drafter.

## 2. Where the code is

| What | Where |
|---|---|
| Full experiment (kernel, route, tests, check switch) | branch `exp/kv4-int4-libr4d`, head `ff656a1` |
| Shippable part (start fix, contract check) | PR #183 (draft), branch `kv4-prefill-18x`, head `248ba5e` — route removed |
| Worktree on this host | `/opt/data/vllm-sly-radiance-kv4pf` (currently `kv4-prefill-18x`; `git worktree add` the exp branch separately) |
| Design note (pre-measurement) | `docs/R4D-INT4-PREFILL.md` (on both branches; exp has the original text) |
| HIP kernel `O_KV4I` | `sly/r4d/r4d_kv4i.patch` (applied third in the kernels stage; entry `r4d.attn_prefill_h256_gqa6_int4kv`, refuses to run without `R4D_KV4I=1`; K scale group 256, block 1664 only) |
| Route + guard | `sly/kv4/radiance_kv4h_attn.py` (`_kv4i_decline`, `_kv4i_launch`), pure decision logic `sly/kv4/kv4i_route.py` |
| CPU helpers/tests | `sly/kv4/kv4i_unpack.h`, `sly/tests/kv4/test_kv4i_{unpack,stage,route}_cpu.py`, `sly/tests/kv4/test_tetra_capture_safe_cpu.py` |
| Micro bench | `sly/bench_attn_r4d_vs_tetra.py` (`--check`, `--kv4i`, `--tile`, `--pipes`) |
| Scripts used on the GPU host | `/opt/data/analysis/r4d-int4-route/scripts/` (`run_acc.sh`, `watch_acc_v3.sh`, `unit.ACCV`, `unit.ACCT`, `mb.sh`) |
| Data | `/opt/data/analysis/r4d-int4-route/data/` (per-layer check log, KL table, KL npz + texts + token ids) |

Knobs (all env): `R4D_KV4I=1` (route on), `RADIANCE_KV4I_MIN_Q` (512), `RADIANCE_KV4I_MIN_CTX` (16384),
`RADIANCE_KV4I_QGAIN` (16), `R4D_KV4I_CHECK=N` (first N routed launches also run the Tetra kernel and log
`[kv4i-check] <layer> n= ctx= rel_l2= max_abs= ref_absmax= qrot_amax=`; host reads, debug only).

## 3. How the route computes

1. Cache holds K and V **Walsh-Hadamard rotated** (16x16) and quantised int4, one e4m3 scale per token per side
   (`sly/kv4/radiance_kv4h_format.py` is the layout oracle).
2. Route rotates the query (fp32, `F.rotate`), divides by `q_scale * QGAIN`, casts to e4m3, calls the kernel
   with `scale / HEAD` (both sides rotated, x16 each), then un-rotates the bf16 output (`F.unrotate`, 256x256 matmul).
3. Kernel unpack ("variant A" in the design note): nibble -> `rne_e4m3(code * s_token)` into LDS byte tiles,
   then the unchanged fp8 `O_QK8|O_PV8` path (P also e4m3). Variant B (exact code, scale folded into the score)
   was NOT built.

## 4. Measurements

All 210 W, one R9700, image `vllm-sly-radiance:1.8.1-kv4i-test` (built from the exp branch; production is 1.8.0).

**Kernel microbench (random K/V), tile 16, µs:** past 49152: libr4d fp8 30,726 / Tetra 55,235 / int4 31,099
(1.01x of fp8). Rel-L2 vs fp32 SDPA on random data: fp8 0.053-0.059, Tetra 0.186-0.238, int4 0.177-0.225.
=> random data cannot see the problem.

**Engine prefill (tok/s, chunk 4096, 65k ctx, prefix cache off):**

| depth | FP8 | KV4 no route | KV4 + route, MIN_CTX 16384 |
|---|--:|--:|--:|
| 1.5k | 3021 | 3020 | 3045 |
| 5.9k | 2977 | 2943 | 2971 |
| 11.8k | 2864 | 2778 | 2801 |
| 23.5k | 2713 | 2550 | 2620 |
| 47k | 2479 | 2220 | 2399 (+8.1 %) |

The speed is real. Note: this sweep ran BEFORE the `scale / HEAD` fix; compute time does not depend on it, but
re-run it once the route is correct.

**Accuracy (the gate): KL of the routed arm against KV4-without-route**, teacher-forced top-20 logprobs, 3 texts
x 60,000 tokens, `sly/kv4/kl_score.py` + `kl_compare.py`, same engine/args both arms (`--gpu-memory-utilization 0.90`):

| text | ppl no route | ppl route | KL mean | top-1 |
|---|--:|--:|--:|--:|
| cpp | 1.297 | 1.297 | 0.0034 | 99.4 % |
| mobydick | 7.475 | 7.494 | 0.0203 | 93.4 % |
| python | 1.750 | 18.4 | 1.84 | 59.1 % |

By position: 0-4k KL 0.000 (route does not fire), 4-16k 0.0014, 16-64k 0.75-0.91 (all docs pooled).
GSM8K-200: 0.835 vs 0.845 (prompts short, route never fires there — proves only that the engine runs).
Gate set in advance: KL <= 0.003, top-1 >= 99 %. Only cpp is near.

**Per-layer check** (`ACCV.check.log`, python text, ctx 20k-45k, 100 launches): rel-L2 of routed output vs the
Tetra kernel is 5-12 % in every full-attention layer; largest in layers 47, 55, 51 (10-12 %), 27-43 (8.6-9.8 %),
smallest in layers 3, 7 (5-7 %). Flat over ctx 20k-45k within a layer. `qrot_amax` 105-146, i.e. far below 448.
Reference: libr4d fp8 against fp32 is 5-6 %.

## 5. Bugs found on the way (all fixed on the exp branch, each invisible to the CPU tests of that moment)

1. `attn_metadata.use_cascade` assert (vLLM 0.31 dropped the field) — fixed on both branches, contract check added.
2. Route guard did `.tolist()` on a device tensor inside graph capture -> crash, and a stream sync per layer per
   step outside capture (-8 % at 1.5k). Guard now uses Python ints only.
3. Route passed `scale` instead of `scale / HEAD` -> softmax 256x too sharp (KL 2.1-3.1). Fixed `cb0327f`.
4. `kv4i_route.py` was missing from `install_kv4.py` copy list (would raise ModuleNotFoundError). Fixed.

## 6. Ruled out

- Wrong scale: fixed (item 3), improved KL 1.25 -> 0.02 (mobydick), python only 147 -> 18 ppl.
- e4m3 saturation of the rotated query: QGAIN 16 (store rotate(q)/16, kernel multiplies back) changed nothing
  (python ppl 18.34 -> 18.38).
- Chunk-boundary / ragged batches / tail tiles: error is flat over ctx and layers fire only on equal-length runs.
- GPU memory: first accuracy run died with OOM at `--gpu-memory-utilization 0.98` (prompt-logprob logits); 0.90 fixes it.

## 7. Open question and hypotheses (none tested)

Rel-L2 5-12 % per layer is only slightly above the fp8 kernel's own 5-6 % vs fp32, yet the model output breaks on
python text. So either the error is concentrated (a few heads/queries/keys with huge effect) or compounds across
16 full-attention layers on data with sharp, copy-like attention.
Unknown: **which of Tetra and the route is closer to the fp32 truth** — the check compares only the two.

- **(hypothesis)** Variant A double rounding `rne_e4m3(code * scale)`: up to 2^-4 relative error per element,
  correlated with the token scale; Tetra uses the exact code. Real K/V have outlier tokens (sinks) where it may matter,
  random K/V do not.
- **(hypothesis)** Hardware `v_cvt_pk_fp8_f32` rounding/saturation differs from torch RNE for scaled products
  (explicitly listed as "not covered" in the design note).
- **(hypothesis)** P in e4m3 with `PGROW` limited to 8 octaves (design note) hurts sharp attention.
- **(hypothesis)** V path: dim-major gather, scale for tokens >= ctx forced to 0, high/low nibble row mapping (d, d+128)
  — an off-by-one in a rarely hit case would be data dependent.

## 8. Suggested next steps (cheapest first)

1. **Offline reproduction, no GPU window afterwards.** In a debug build dump for ONE bad layer (e.g. layer 47,
   python text, one routed launch at ctx ~20k): `query`, the touched `kv_cache` blocks, `block_table`, `seq_lens`,
   routed `out_rot`, Tetra `ref`. Then on CPU build the fp32 truth from the int4 bytes (dequant per
   `radiance_kv4h_format.py`, plain SDPA) and compare **both** to it, per head, per query row, per key range.
   Tells: who is wrong, concentrated or spread, which heads.
2. Exact-code experiment: stage the bare code (LUT, exact) and apply `s_token` to V rows / q.k outside the e4m3
   rounding in a slow reference kernel (or emulate variant A vs B in torch on the dumped data first — cheap, tells
   whether double rounding explains the error before touching HIP).
3. Extend the micro bench with real dumped K/V instead of random (random data hid this completely).
4. Only then change the kernel. Re-run the gate with `scripts/run_acc.sh` (ARMS=V:route DOCS=python is a 10 min
   fast loop; full gate: both arms, 3 docs, ~20 min).

## 9. Hard requirements the continuation must keep

- Accuracy gate before any speed claim: KL <= 0.003 and top-1 >= 99 % against KV4-without-route, all 3 texts, then
  GSM8K-200 within 0.03. A throughput sweep does not see a wrong result (this cost three iterations).
- Route stays opt-in (`R4D_KV4I=1`), off by default; no image bump for the experiment.
- New `forward` paths need a CPU test with tensors that raise on any device->host read (see
  `test_tetra_capture_safe_cpu.py` on the exp branch) and must keep `ci/check_attn_metadata_contract.py` green.
- Open PRs as draft (`gh pr create --draft`, build.yml skips drafts).

## 10. GPU host procedure (do not improvise)

- Host `192.168.178.51`, ssh key `~/.ssh/ai1`, user root. Production LLM runs on `192.168.178.54` (`radiance-serve`)
  and must be restored: after the run `systemctl start radiance-serve` on .54 if `gpu-window release` did not, wait for
  health 200.
- Take the card only through `gpu-window acquire <name> <ttl>` WITHOUT `force`; if someone else holds it (e.g.
  `kore-v2ab`, TTL up to 4 h), wait. Check no container and VRAM < 2 GB afterwards.
- Power: 210 W production setting (LACT on pve3, `/root/splitk_power.sh`). Change only on explicit order, with backup.
- Unit template is `/root/docker-vllm7.service.orig` (restore it after every run). The accuracy units
  `unit.ACCT` (route off) / `unit.ACCV` (route on, `R4D_KV4I_CHECK=100`) bind-mount a source tree from
  `/root/kv4arm/w3` (tar the repo `sly/` dir, scp, untar there). Start via
  `gpu-window start vllm-sly-radiance:1.8.1-kv4i-test docker-vllm7.service`.
- Accuracy files on the host: `/root/kvroute/{kl_score.py,kl_compare.py,acc/{raw,ids,out/{bf16,route}}}`. The arm
  directory `out/bf16` actually holds KV4-WITHOUT-route (label from the older bf16 comparison); `out/route` is the routed arm.
  `kl_compare.py route` compares it (REF is env `KVACC_REF`, default `bf16`).
- GSM8K: `/root/gsm8k_any.sh TAG URL MODEL` (lm_eval venv `/root/lmeval-venv`).
- Never `pkill -f` with a self-matching pattern (killed the SSH session once): use `"[r]un_acc"` style.
- Docker is not reachable from the agent host; use ssh to .51.
- Subagents must not touch the GPU; the parent runs measurements.
- Sonnet subagents hit HTTP 429 session limits at times; Opus worked.
