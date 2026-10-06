#!/usr/bin/env python3
"""Adaptive verify width for the DFlash2 block drafter, graph-safe (RADIANCE_ADAPTIVE_WIDTH).

Port of Radiance's adaptive_k (core/sched/adaptive_k.{h,cpp}) onto the V2 model runner. The policy
lives in radiance_adaptive_width.py; this patch only wires it in. Everything is inert unless
RADIANCE_ADAPTIVE_WIDTH=uniform|perseq (default 0: no behaviour change, no extra graphs).

WHY ggz14's patch_dynwidth lost (-3 %): it trimmed each request's spec list to ceil(EMA)+2, so one
capped request made the whole decode batch non-uniform. The V2 cudagraph manager hands the FULL graph
only to a uniform decode batch (num_tokens == max_query_len * num_reqs and a graph captured for exactly
that query length: cudagraph_utils.dispatch/_is_compatible, desc.uniform_token_count); every other
batch runs the PIECEWISE route, which for this hybrid model is ~65 graph pieces with the attention and
GDN custom ops (both splitting ops) executed eagerly in between. The trimmed verify rows saved about
0.4 ms each, the lost FULL graph cost about as much.

What this patch changes (each hunk is idempotent and anchored on the installed vLLM 0.29 sources):
  scheduler.py        update_from_output observes (drafted, accepted) per request; the batch-level
                      chooser trims the spec lists of the requests of the NEXT step (sync: after
                      update_draft_token_ids; async: after _update_after_schedule, where the
                      placeholder lists are set).
  async_scheduler.py  calls the chooser after the placeholders are assigned.
  cudagraph_utils.py  ModelCudaGraphManager only (not the drafter's):
                        uniform: captures a uniform-decode FULL graph for every (n requests,
                                 query length w+1) with n >= RADIANCE_AW_GRAPH_MIN_REQS, so a batch
                                 that drops to width w is still a uniform decode batch with a graph;
                        perseq : captures varlen FULL decode graphs (max_query_len = decode query
                                 length, uniform_token_count unset) next to the uniform ones; the
                                 uniform graph is tried first (candidate order), so full-width
                                 batches replay exactly as before. RADIANCE_AW_VARLEN_GRAPH=0 keeps
                                 per-request widths on PIECEWISE (the ggz14 setting, for the A/B).
  model_runner.py     dispatch passes max_query_len=None for a batch with a prefill row or a row
                      without draft tokens, so such a batch can never match a varlen graph (the GDN
                      metadata of those batches is built on a different path).
  mamba_hybrid.py     capture builds the attention metadata from the descriptor's PROMISED
                      max_query_len (the default model state already does; the hybrid one used the
                      dummy batch's actual maximum, which would bake a too small q-grid into a varlen
                      graph).

The kernels already read query_start_loc / cu_seqlens at run time (unified attention, the GDN
recurrent + conv-update kernels, the sampler's cu_num_logits), and padded requests have zero length,
the same shape a partially filled uniform graph already sees. Lossless: only fewer verify rows.
"""
import ast
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
V1 = SP / "vllm/v1"


def append_tail(path, tail, sentinel, label):
    if not path.exists():
        raise SystemExit(f"  FAIL  {label}: {path} missing")
    s = path.read_text()
    if sentinel in s:
        print(f"  NOOP  {label} already applied")
        return
    s = s.rstrip("\n") + "\n\n" + tail
    ast.parse(s)
    path.write_text(s)
    print(f"  OK    {label}")


# ------------------------------------------------------------------------------ scheduler.py
SCHED = V1 / "core/sched/scheduler.py"
SCHED_TAIL = '''# ---- RADIANCE adaptive verify width (patch_adaptive_width.py) ---------------------------------
try:
    import radiance_adaptive_width as _rad_aw
except Exception:  # pragma: no cover - module missing: stay inert
    _rad_aw = None
_RAD_AW_ON = _rad_aw is not None and _rad_aw.mode() != "0"


def _radiance_aw_obj(self):
    aw = getattr(self, "_rad_aw_obj", None)
    if aw is None:
        aw = self._rad_aw_obj = _rad_aw.AdaptiveWidth(self.num_spec_tokens)
    return aw


def _radiance_aw_observe(self, request, num_accepted, num_draft_tokens):
    if _RAD_AW_ON:
        _radiance_aw_obj(self).observe(request, num_draft_tokens, num_accepted)


def _radiance_aw_apply(self, reqs):
    """Trim the spec lists of `reqs` (the decode requests of the next step) to the batch choice."""
    if not _RAD_AW_ON or not reqs:
        return
    ns = [len(r.spec_token_ids) for r in reqs]
    ks = _radiance_aw_obj(self).decide(reqs, ns)
    for r, k, n in zip(reqs, ks, ns):
        if k < n:
            r.spec_token_ids = r.spec_token_ids[:k]


def _radiance_aw_sync(self, draft_token_ids):
    if not _RAD_AW_ON:
        return
    reqs = []
    for req_id in draft_token_ids.req_ids:
        r = self.requests.get(req_id)
        if r is None or r.is_finished() or r.is_prefill_chunk or not r.spec_token_ids:
            continue
        reqs.append(r)
    _radiance_aw_apply(self, reqs)


def _radiance_aw_after_schedule(self, scheduler_output):
    if not _RAD_AW_ON:
        return
    reqs = []
    for req_id in scheduler_output.num_scheduled_tokens:
        r = self.requests.get(req_id)
        if r is None or r.is_prefill_chunk or not r.spec_token_ids:
            continue
        reqs.append(r)
    _radiance_aw_apply(self, reqs)


Scheduler._radiance_aw_observe = _radiance_aw_observe
Scheduler._radiance_aw_sync = _radiance_aw_sync
Scheduler._radiance_aw_after_schedule = _radiance_aw_after_schedule
'''

# ------------------------------------------------------------------------------ cudagraph_utils
CG = V1 / "worker/gpu/cudagraph_utils.py"
CG_TAIL = '''# ---- RADIANCE adaptive verify width (patch_adaptive_width.py) ---------------------------------
def _radiance_aw_extra_descs(
    mgr, descs_by_mode, decode_mode, capture_sizes, max_decode_tokens, max_cg_size, separate
):
    """Extra FULL decode descriptors for RADIANCE_ADAPTIVE_WIDTH (target model's manager only)."""
    if type(mgr).__name__ != "ModelCudaGraphManager" or not separate or not decode_mode:
        return
    import os

    try:
        import radiance_adaptive_width as aw
    except Exception:
        return
    lst = descs_by_mode[decode_mode]
    added = []
    loras = list(mgr.lora_capture_cases)
    if aw.varlen_graph_enabled():
        # Varlen decode graphs (any split of the tokens, every request >= 2 rows). Inserted at the
        # FRONT: the candidate lists end up reversed, so the uniform graph of the same size is tried
        # first and a full-width batch replays exactly as without this knob. Below 16 tokens the
        # dummy batch has one row per request, which would capture the non-spec GDN path.
        for num_tokens in capture_sizes:
            if num_tokens < 16 or num_tokens > max_decode_tokens or num_tokens > max_cg_size:
                continue
            for nl in loras:
                desc = BatchExecutionDescriptor(
                    cg_mode=decode_mode,
                    num_tokens=num_tokens,
                    num_reqs=min(num_tokens, mgr.max_num_reqs),
                    max_query_len=mgr.decode_query_len,
                    num_active_loras=nl,
                )
                if desc not in lst:
                    lst.insert(0, desc)
                    added.append(desc)
    depth = mgr.decode_query_len - 1
    min_reqs = int(os.environ.get("RADIANCE_AW_GRAPH_MIN_REQS", "4"))
    for w in aw.tier_widths(depth):
        q = w + 1
        for n in range(max(min_reqs, 1), mgr.max_num_reqs + 1):
            num_tokens = n * q
            if num_tokens > max_decode_tokens or num_tokens > max_cg_size:
                continue
            for nl in loras:
                desc = BatchExecutionDescriptor(
                    cg_mode=decode_mode,
                    num_tokens=num_tokens,
                    num_reqs=n,
                    uniform_token_count=q,
                    num_active_loras=nl,
                )
                if desc not in lst:
                    lst.append(desc)
                    added.append(desc)
    if added:
        logger.info(
            "RADIANCE adaptive width (%s): %d extra FULL decode graphs",
            aw.mode(), len(added),
        )

'''

# ------------------------------------------------------------------------------ model_runner
MR = V1 / "worker/gpu/model_runner.py"
MR_TAIL = '''# ---- RADIANCE adaptive verify width (patch_adaptive_width.py) ---------------------------------
try:
    from radiance_adaptive_width import varlen_graph_enabled as _rad_aw_varlen_enabled

    _RAD_AW_VARLEN = bool(_rad_aw_varlen_enabled())
except Exception:  # pragma: no cover
    _RAD_AW_VARLEN = False


def _radiance_aw_dispatch_qlen(max_query_len, batch_req_state, scheduler_output):
    """max_query_len for the cudagraph dispatch. A varlen decode graph may only serve a pure decode
    batch in which every request verifies drafts (>= 2 rows); anything else gets None, which no
    varlen descriptor matches."""
    if not _RAD_AW_VARLEN or batch_req_state is None:
        return max_query_len
    if batch_req_state.has_prefill:
        return None
    if min(scheduler_output.num_scheduled_tokens.values()) < 2:
        return None
    return max_query_len


import os as _rad_os  # noqa: E402

_RAD_AW_STATS = _rad_os.environ.get("RADIANCE_AW_GRAPH_STATS", "0") == "1"
_rad_aw_cnt = {}


def _radiance_aw_stats(batch_desc, batch_req_state):
    """RADIANCE_AW_GRAPH_STATS=1: how many steps ran on which cudagraph route (diagnostic; works
    with RADIANCE_ADAPTIVE_WIDTH=0 too, so the control arm shows its own FULL/PIECEWISE split)."""
    if not _RAD_AW_STATS or batch_req_state is None:
        return
    key = ("prefill+" if batch_req_state.has_prefill else "decode-only ") + batch_desc.cg_mode.name
    _rad_aw_cnt[key] = _rad_aw_cnt.get(key, 0) + 1
    n = _rad_aw_cnt["steps"] = _rad_aw_cnt.get("steps", 0) + 1
    if n % 500 == 0:
        print("[radiance-aw] graph route (cumulative): "
              + " ".join(f"{k}={v}" for k, v in sorted(_rad_aw_cnt.items())), flush=True)
'''


def main():
    # observation hook: after the scheduler computed num_accepted / num_rejected
    apply(
        SCHED,
        "                num_rejected = num_draft_tokens - num_accepted\n",
        "                num_rejected = num_draft_tokens - num_accepted\n"
        "                self._radiance_aw_observe(request, num_accepted, num_draft_tokens)\n",
        "_radiance_aw_observe",
        "scheduler.py observe hook",
    )
    # sync scheduling: trim after the real drafts arrived
    apply(
        SCHED,
        "            request.spec_token_ids = self.structured_output_manager.validate_tokens(\n"
        "                request, spec_token_ids\n"
        "            )\n"
        "\n"
        "    def update_draft_token_ids_in_output(",
        "            request.spec_token_ids = self.structured_output_manager.validate_tokens(\n"
        "                request, spec_token_ids\n"
        "            )\n"
        "        self._radiance_aw_sync(draft_token_ids)\n"
        "\n"
        "    def update_draft_token_ids_in_output(",
        "_radiance_aw_sync(draft_token_ids)",
        "scheduler.py sync hook",
    )
    append_tail(SCHED, SCHED_TAIL, "def _radiance_aw_obj", "scheduler.py chooser")

    # async scheduling: placeholders are assigned in _update_after_schedule
    apply(
        V1 / "core/sched/async_scheduler.py",
        "                request.next_decode_eligible_step = self.current_step + self.pp_size\n",
        "                request.next_decode_eligible_step = self.current_step + self.pp_size\n"
        "        # RADIANCE (patch_adaptive_width.py): batch-level verify width for the next step.\n"
        "        self._radiance_aw_after_schedule(scheduler_output)\n",
        "_radiance_aw_after_schedule(scheduler_output)",
        "async_scheduler.py hook",
    )

    # cudagraph descriptors
    apply(
        CG,
        "        for mode, descs in descs_by_mode.items():\n",
        "        _radiance_aw_extra_descs(\n"
        "            self, descs_by_mode, decode_mode, capture_sizes, max_decode_tokens,\n"
        "            max_cg_capture_size, separate_decode_routine,\n"
        "        )\n"
        "        for mode, descs in descs_by_mode.items():\n",
        "_radiance_aw_extra_descs(\n",
        "cudagraph_utils.py descriptors",
    )
    append_tail(CG, CG_TAIL, "def _radiance_aw_extra_descs", "cudagraph_utils.py helper")

    # dispatch guard
    apply(
        MR,
        "            max_query_len=max_query_len,\n            need_eager=is_profile or skip_compiled,\n",
        "            max_query_len=_radiance_aw_dispatch_qlen(\n"
        "                max_query_len, batch_req_state, scheduler_output\n"
        "            ),\n"
        "            need_eager=is_profile or skip_compiled,\n",
        "_radiance_aw_dispatch_qlen(\n",
        "model_runner.py dispatch guard",
    )
    append_tail(MR, MR_TAIL, "def _radiance_aw_dispatch_qlen", "model_runner.py helper")
    apply(
        MR,
        "        if batch_desc.num_tokens == 0:\n            # All DP ranks have zero tokens to run.\n",
        "        _radiance_aw_stats(batch_desc, batch_req_state)\n"
        "        if batch_desc.num_tokens == 0:\n            # All DP ranks have zero tokens to run.\n",
        "        _radiance_aw_stats(batch_desc, batch_req_state)\n        if batch_desc.num_tokens == 0:",
        "model_runner.py graph-route stats",
    )

    # hybrid model state: capture with the promised max_query_len
    apply(
        V1 / "worker/gpu/model_states/mamba_hybrid.py",
        "        max_query_len = input_batch.num_scheduled_tokens.max().item()\n",
        "        max_query_len = input_batch.num_scheduled_tokens.max().item()\n"
        "        if for_capture and input_batch.max_query_len:\n"
        "            # RADIANCE (patch_adaptive_width.py): a varlen graph bakes the PROMISED q-grid\n"
        "            max_query_len = input_batch.max_query_len\n",
        "RADIANCE (patch_adaptive_width.py): a varlen graph",
        "mamba_hybrid.py promised max_query_len",
    )


main()
