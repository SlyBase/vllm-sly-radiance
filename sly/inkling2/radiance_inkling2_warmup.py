"""Inkling-2-Mini prefill-shape warmup: compile the Triton kernels of real prompt lengths BEFORE the API is up.

vLLM's `warmup_kernels` runs 9-token prefills only, so the first request of every new prompt length JIT-compiles
Triton kernels (2.8-9.5 s per first request measured on 2026-11-25). Per ladder size this runs one single-request
prefill plus one spec-decode step through the worker's own `execute_model` / `sample_tokens` (the same synthetic
`SchedulerOutput`s `warmup_kernels` builds), then retires the request. Single request on purpose: the serving case
is num_seqs == 1, and Triton specialises integer args on `== 1` and `% 16 == 0`.

Knob: RADIANCE_INKLING2_PREFILL_WARMUP (default 1 for an Inkling target, `0` = stock warmup only). A failing size is
logged and skipped: warmup must never stop an engine from starting. Hook: sly/patch_inkling2_warmup.py.
"""
import os
import time

KNOB = "RADIANCE_INKLING2_PREFILL_WARMUP"
# per tile bucket of the prefill kernels: a %16 and a non-%16 size, plus the edges; 512/513 bracket the GEMM switch
_BASE = (2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 46, 48, 63, 64, 65, 75, 100, 127, 128, 129, 200, 255, 256, 257, 400,
         511, 512, 513, 600, 1000, 1024, 1500, 2048, 2049, 3000)


def ladder(max_tokens, max_len):
    """Sorted prompt lengths to warm: the base ladder plus the chunk size, capped by the step budget and max_len - 1."""
    cap = min(max_tokens, max_len - 1)
    return sorted({n for n in (*_BASE, max_tokens) if 2 <= n <= cap})


def wanted(model_type, env=os.environ):
    """On for an Inkling target (model_type `inkling2_loop`); never for another model; `=0` switches it off."""
    return "inkling" in str(model_type or "").lower() and env.get(KNOB, "1") != "0"


def run_ladder(sizes, one, cleanup=lambda n: None, log=print):
    """Run `one(n)` per size; `cleanup(n)` always runs. Returns (ok_count, failed_sizes)."""
    ok, failed = 0, []
    for n in sizes:
        try:
            one(n)
            ok += 1
        except Exception as e:  # noqa: BLE001 - a warmup failure must not stop the engine
            failed.append(n)
            log(f"[radiance] inkling2 prefill warmup n={n} failed: {type(e).__name__}: {e}")
        finally:
            try:
                cleanup(n)
            except Exception as e:  # noqa: BLE001
                log(f"[radiance] inkling2 prefill warmup cleanup n={n} failed: {type(e).__name__}: {e}")
    return ok, failed


def warmup_prefill_ladder(worker):
    """Engine side (needs vllm + a GPU): called from Worker._compile_or_warm_up_model after warmup_kernels."""
    mr = worker.model_runner
    if not wanted(getattr(worker.model_config.hf_config, "model_type", None)) or mr.is_pooling_model:
        return
    import torch
    from vllm import SamplingParams
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
    from vllm.v1.worker.extensible_kv_cache import ensure_kv_cache_blocks
    from vllm.v1.worker.gpu.warmup import _fixed_draft_counts, _run_step, _warmup_block_counter, _warmup_token_ids

    spec_cfg = mr.speculative_config
    drafts = spec_cfg.max_num_new_slots_for_drafting if spec_cfg is not None else 0
    steps = mr.vllm_config.num_scheduled_spec_tokens
    dql = mr.decode_query_len if steps else 1
    groups = mr.kv_cache_config.kv_cache_groups
    count = _warmup_block_counter(mr)
    sizes = ladder(mr.max_num_tokens - drafts, mr.max_model_len)

    def retire(req_id):
        out = SchedulerOutput.make_empty()
        out.finished_req_ids = {req_id}
        _run_step(worker.execute_model, None, out)

    def one(n):
        req_id = f"_ttft_warmup_{n}_"
        first = [count(n, g) for g in groups]
        after = [count(n + dql, g) for g in groups]
        ensure_kv_cache_blocks(mr, 1 + sum(after))
        nxt = 1

        def alloc(c):
            nonlocal nxt
            ids = list(range(nxt, nxt + c))
            nxt += c
            return ids

        ids = _warmup_token_ids(mr, n)
        pre = SchedulerOutput.make_empty()
        pre.num_spec_tokens_to_schedule = steps
        pre.scheduled_new_reqs = [NewRequestData(
            req_id=req_id, prompt_token_ids=ids, mm_features=[], sampling_params=SamplingParams.for_sampler_warmup(),
            pooling_params=None, block_ids=tuple(alloc(c) for c in first), num_computed_tokens=0, lora_request=None,
            prefill_token_ids=ids)]
        pre.num_scheduled_tokens = {req_id: n}
        pre.total_num_scheduled_tokens = n
        _run_step(worker.execute_model, worker.sample_tokens, pre)
        cached = CachedRequestData.make_empty()
        cached.req_ids = [req_id]
        cached.num_computed_tokens = [n]
        cached.num_output_tokens = [1]
        deltas = [a - b for a, b in zip(after, first)]
        cached.new_block_ids = [tuple(alloc(d) for d in deltas) if any(deltas) else None]
        dec = SchedulerOutput.make_empty()
        dec.num_spec_tokens_to_schedule = steps
        dec.scheduled_cached_reqs = cached
        dec.num_scheduled_tokens = {req_id: dql}
        if steps:
            dec.scheduled_spec_decode_tokens = {req_id: [0] * steps}
        dec.total_num_scheduled_tokens = dql
        _run_step(worker.execute_model, worker.sample_tokens, dec)

    t0 = time.perf_counter()
    mr.kv_connector.set_disabled(True)
    try:
        with torch.inference_mode(), _fixed_draft_counts(mr):
            ok, failed = run_ladder(sizes, one, cleanup=lambda n: retire(f"_ttft_warmup_{n}_"))
    finally:
        mr.kv_connector.set_disabled(False)
        if mr.speculator is not None:
            mr.speculator.reset_state()
        if mr.kv_block_zeroer is not None:
            mr.kv_block_zeroer.zero_block_ids([0])
        torch.accelerator.synchronize()
    print(f"[radiance] inkling2 prefill warmup: {ok}/{len(sizes)} sizes in {time.perf_counter() - t0:.1f}s"
          f"{' failed=' + str(failed) if failed else ''}", flush=True)
