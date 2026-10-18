#!/usr/bin/env python3
"""Lazy GDN state snapshots under speculative decode (RADIANCE_GDN_LAZY=1; radiance_gdn_lazy.py).

Three source patches, every one of them inert unless the env knob is set at runtime:
  1. mamba/abstract.py: the GDN MambaSpec asks for ONE speculative block (the candidate stash)
     instead of num_speculative_tokens -- this is where the per-request page count falls 9 -> 3.
  2. gdn_attn.py: the persistent spec-state-index buffer is 2 columns wide (running, stash), and the
     metadata carries the stash column for every row so a prefill can invalidate its stash.
  3. v1/worker/mamba_utils.py: the fused align copies skip TEMPORAL states (LAZY_TEMPORAL) and the
     lazy materialize kernel provides them (pre-forward migration after the Triton launch, post-step
     checkpoint BEFORE it, because the Triton kernel resets num_accepted in place). Only when the
     mamba groups own a stash block (radiance_gdn_lazy.temporal_enabled): without a drafter the
     stock Triton copies stay on (1.3.2; the stashless materialize broke determinism in 1.3.1).

Idempotency: each hunk's replacement carries its own sentinel string (SENT + suffix) in a comment,
so a second pass over an already-patched tree NOOPs every hunk. The sentinels are unique per hunk
(bare SENT would let an earlier hunk in the same file short-circuit the rest).
"""
import sysconfig
from pathlib import Path

from _patchlib import apply, apply_any

SP = Path(sysconfig.get_paths()["purelib"]) / "vllm"
SENT = "radiance lazy gdn"

# ---- 1. spec blocks -------------------------------------------------------------------------
# Re-anchored for vLLM 0.32.0: the RecoverSSM guard became `use_replayssm` (the new ReplaySSM
# Mamba2 decode kernel; `use_kda_recoverssm` stays as the separate Kimi-K3 KDA flag). The helper
# guard is renamed with the same shape so the lazy block count tracks whatever the stock count
# tracks on the pinned version.
apply_any(SP / "model_executor/layers/mamba/abstract.py",
[(
'''            num_speculative_blocks=(
                0
                if vllm_config.cache_config.use_replayssm
                else vllm_config.num_speculative_tokens
            ),
''',
'''            num_speculative_blocks=_radiance_lazy_spec_blocks(self, vllm_config),  # radiance lazy gdn
'''
), (
'''            num_speculative_blocks=(
                0
                if vllm_config.cache_config.use_kda_recoverssm
                else vllm_config.num_speculative_tokens
            ),
''',
'''            num_speculative_blocks=_radiance_lazy_spec_blocks(self, vllm_config),  # radiance lazy gdn
'''
)], SENT, "abstract.get_kv_cache_spec: one stash block under RADIANCE_GDN_LAZY")
apply(SP / "model_executor/layers/mamba/abstract.py",
'''class MambaBase(AttentionLayerBase):''',
'''def _radiance_lazy_spec_blocks(layer, vllm_config):
    """radiance lazy gdn helper: a lazy cache keeps one stash block per request instead of one block
    per draft token (radiance_gdn_lazy.py). GDN layers only; everything else keeps the stock count."""
    import os as _os
    if vllm_config.cache_config.use_replayssm:
        return 0
    n = vllm_config.num_speculative_tokens
    if n > 0 and _os.environ.get("RADIANCE_GDN_LAZY", "0") == "1" \\
            and "GDN" in str(getattr(layer.mamba_type, "name", layer.mamba_type)).upper():
        return 1
    return n


class MambaBase(AttentionLayerBase):''', SENT + " helper", "abstract: lazy spec-block helper")

# ---- 2. builder ------------------------------------------------------------------------------
G = SP / "v1/attention/backends/gdn_attn.py"
apply(G,
'''    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
''',
'''    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]
    # radiance lazy gdn field: the stash block of every batch row (window column 1), for prefill
    # invalidation. None unless RADIANCE_GDN_LAZY=1.
    radiance_stash_indices: torch.Tensor | None = None
''', SENT + " field", "gdn_attn: stash-index metadata field")
apply(G,
'''        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, self.num_spec + 1),
''',
'''        import os as _rl_os  # radiance lazy gdn width: 2 state columns (running, stash)
        self._rad_lazy = _rl_os.environ.get("RADIANCE_GDN_LAZY", "0") == "1"
        self.spec_state_indices_tensor: torch.Tensor = torch.empty(
            (self.decode_cudagraph_max_bs, 2 if (self._rad_lazy and self.num_spec > 0) else self.num_spec + 1),
''', SENT + " width", "gdn_attn: 2-wide spec-state buffer under lazy")
# the full build path is the only path the shipped image takes: patch_gdn_shared_build is not in
# the loop (ci/unused_patches.txt), so the shared-build fast path below never runs. The full path
# (GDNAttentionMetadata) populates radiance_stash_indices for every step.
apply(G,
'''        attn_metadata = GDNAttentionMetadata(
            num_prefills=num_prefills,
''',
'''        attn_metadata = GDNAttentionMetadata(
            radiance_stash_indices=(block_table_tensor[:, 1] if (self._rad_lazy and self.num_spec > 0) else None),  # radiance lazy gdn full build path
            num_prefills=num_prefills,
''', SENT + " full", "gdn_attn: stash indices on the full build path")

# ---- 3. copies -------------------------------------------------------------------------------
# Rebased for the vLLM 0.30 mamba_utils: the two fused align kernels gained a TEMPORAL_TILES
# constexpr (3D-grid partitioning of the temporal copy body), every launch passes it, the
# align postprocess snapshots the decision buffer before the reset, and get_mamba_groups now
# returns dict[MambaSpec, list[int]]. The LAZY_TEMPORAL skip sits directly before the shared
# _copy_mamba_state_block body; the materialize calls bracket the precopy / align postprocess.
M = SP / "v1/worker/mamba_utils.py"
apply(M,
'''    # PRECOMPUTED_NEW_COMPUTED: when True, num_computed_tokens_ptr already holds
    # the post-step new_num_computed value (V2 supplies the advanced count).
    PRECOMPUTED_NEW_COMPUTED: tl.constexpr = False,
    # TEMPORAL_TILES: when > 1, the temporal copy body is partitioned across
    # TEMPORAL_TILES CTAs along the u64 inner range. Callers must launch a
    # 3D grid (num_reqs, total_states, TEMPORAL_TILES). Default 1 preserves
    # the existing 2D-grid contract.
    TEMPORAL_TILES: tl.constexpr = 1,
):
''',
'''    # PRECOMPUTED_NEW_COMPUTED: when True, num_computed_tokens_ptr already holds
    # the post-step new_num_computed value (V2 supplies the advanced count).
    PRECOMPUTED_NEW_COMPUTED: tl.constexpr = False,
    # TEMPORAL_TILES: when > 1, the temporal copy body is partitioned across
    # TEMPORAL_TILES CTAs along the u64 inner range. Callers must launch a
    # 3D grid (num_reqs, total_states, TEMPORAL_TILES). Default 1 preserves
    # the existing 2D-grid contract.
    TEMPORAL_TILES: tl.constexpr = 1,
    LAZY_TEMPORAL: tl.constexpr = False,  # radiance lazy gdn post-sig: temporal states are materialised elsewhere
):
''', SENT + " post-sig", "mamba_utils: LAZY_TEMPORAL on the postprocess kernel")
# Re-anchored for vLLM 0.32.0: the kernels no longer compute a local `bt_row_idx` (the
# idx-mapping resolves straight into `req_idx` before the decision logic, and the call sites pass
# it directly). The skip sits directly before each kernel's _copy_mamba_state_block call, as on
# 0.30/0.31; the old anchors stay for the launchers patching older images.
apply_any(M,
[
(
'''    _copy_mamba_state_block(
        state_idx,
        req_idx,
        src_block_idx,
        dest_block_idx,
        accept_token_bias,
''',
'''    if LAZY_TEMPORAL:  # radiance lazy gdn post-skip
        if tl.load(state_conv_widths_ptr + state_idx) == 0:
            return
    _copy_mamba_state_block(
        state_idx,
        req_idx,
        src_block_idx,
        dest_block_idx,
        accept_token_bias,
'''
),
(
'''    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx
    _copy_mamba_state_block(
''',
'''    if LAZY_TEMPORAL:  # radiance lazy gdn post-skip
        if tl.load(state_conv_widths_ptr + state_idx) == 0:
            return
    bt_row_idx = batch_idx if HAS_IDX_MAPPING else req_idx
    _copy_mamba_state_block(
'''
)
], SENT + " post-skip", "mamba_utils: postprocess skips temporal states under lazy")
apply(M,
'''    CONV_STATE_DIM_FIRST: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr = True,
    # TEMPORAL_TILES: see postprocess_mamba_fused_kernel. Default 1 preserves
    # the 2D-grid contract; > 1 requires a 3D grid.
    TEMPORAL_TILES: tl.constexpr = 1,
):
    """Pre-copy mamba "align" state across block boundaries.
''',
'''    CONV_STATE_DIM_FIRST: tl.constexpr,
    HAS_IDX_MAPPING: tl.constexpr = True,
    # TEMPORAL_TILES: see postprocess_mamba_fused_kernel. Default 1 preserves
    # the 2D-grid contract; > 1 requires a 3D grid.
    TEMPORAL_TILES: tl.constexpr = 1,
    LAZY_TEMPORAL: tl.constexpr = False,  # radiance lazy gdn pre-sig
):
    """Pre-copy mamba "align" state across block boundaries.
''', SENT + " pre-sig", "mamba_utils: LAZY_TEMPORAL on the precopy kernel")
# vLLM 0.32.0: the precopy passes `req_idx` where 0.30/0.31 passed `batch_idx` (see post-skip
# note); old shape kept for the launchers.
apply_any(M,
[
(
'''    token_bias = tl.load(token_bias_ptr + req_idx)
    _copy_mamba_state_block(
        state_idx,
        req_idx,
        src_col,
        dst_col,
        token_bias,
''',
'''    if LAZY_TEMPORAL:  # radiance lazy gdn pre-skip
        if tl.load(state_conv_widths_ptr + state_idx) == 0:
            return
    token_bias = tl.load(token_bias_ptr + req_idx)
    _copy_mamba_state_block(
        state_idx,
        req_idx,
        src_col,
        dst_col,
        token_bias,
'''
),
(
'''    token_bias = tl.load(token_bias_ptr + req_idx)
    _copy_mamba_state_block(
        state_idx,
        batch_idx,
        src_col,
        dst_col,
        token_bias,
''',
'''    if LAZY_TEMPORAL:  # radiance lazy gdn pre-skip
        if tl.load(state_conv_widths_ptr + state_idx) == 0:
            return
    token_bias = tl.load(token_bias_ptr + req_idx)
    _copy_mamba_state_block(
        state_idx,
        batch_idx,
        src_col,
        dst_col,
        token_bias,
'''
)
], SENT + " pre-skip", "mamba_utils: precopy skips temporal states under lazy")
# remember what the tables need (initialize_from_forward_context owns the copy-funcs / ctx)
apply(M,
'''        if self.is_initialized:
            return
        # This only runs once per worker.
''',
'''        if self.is_initialized:
            return
        # radiance lazy gdn tables: the materialize tables are built from these on first use
        self._radiance_kv_cfg = kv_cache_config
        self._radiance_fwd_ctx = forward_context
        self._radiance_copy_funcs = mamba_state_copy_funcs
        # lazy temporal path only with a stash block (a drafter); else vLLM's exact copies run
        import radiance_gdn_lazy
        self._radiance_lazy_temporal = radiance_gdn_lazy.temporal_enabled(
            kv_cache_config, self.mamba_group_ids)
        # This only runs once per worker.
''', SENT + " tables", "mamba_utils: keep the forward context for the lazy tables")
# V1 postprocess: not supported under lazy
apply(M,
'''        if num_reqs == 0 or not self.is_initialized:
            return

        # Initialize output to current values (unchanged unless src==dst)
''',
'''        if num_reqs == 0 or not self.is_initialized:
            return
        if _radiance_lazy():  # radiance lazy gdn v1
            raise RuntimeError("RADIANCE_GDN_LAZY needs the V2 model runner (align postprocess)")

        # Initialize output to current values (unchanged unless src==dst)
''', SENT + " v1", "mamba_utils: V1 postprocess refuses lazy")
# precopy: Triton (conv only) then materialize
apply(M,
'''            idx_mapping,
            num_reqs,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
            HAS_IDX_MAPPING=idx_mapping is not None,
            TEMPORAL_TILES=_TEMPORAL_TILES,
        )
''',
'''            idx_mapping,
            num_reqs,
            COPY_BLOCK_SIZE=1024,
            CONV_STATE_DIM_FIRST=is_conv_state_dim_first(),
            HAS_IDX_MAPPING=idx_mapping is not None,
            TEMPORAL_TILES=_TEMPORAL_TILES,
            LAZY_TEMPORAL=self._radiance_lazy_temporal,
        )
        if self._radiance_lazy_temporal:  # radiance lazy gdn precopy: temporal migration = base + replay
            import radiance_gdn_lazy
            radiance_gdn_lazy.materialize(self, 0, num_reqs, state_idx_gpu, src_col_gpu,
                                          token_bias_gpu, idx_mapping)
''', SENT + " precopy", "mamba_utils: lazy materialize after the precopy")
apply(M,
'''        if num_reqs == 0 or not self.is_initialized:
            return

        # V2 reads non-contiguous idx_mapping positions, so snapshot the whole
        # decision buffer rather than only [:num_reqs].
''',
'''        if num_reqs == 0 or not self.is_initialized:
            return
        if self._radiance_lazy_temporal:  # radiance lazy gdn postalign: checkpoint = base + replay, BEFORE the reset
            import radiance_gdn_lazy
            radiance_gdn_lazy.materialize(self, 1, num_reqs, num_accepted_tokens_gpu,
                                          state_idx_gpu, new_num_computed_tokens_gpu, idx_mapping)
        # V2 reads non-contiguous idx_mapping positions, so snapshot the whole
        # decision buffer rather than only [:num_reqs].
''', SENT + " postalign", "mamba_utils: lazy materialize before the align postprocess")
apply(M,
'''            HAS_IDX_MAPPING=True,
            PRECOMPUTED_NEW_COMPUTED=True,
            TEMPORAL_TILES=_TEMPORAL_TILES,
        )
''',
'''            HAS_IDX_MAPPING=True,
            PRECOMPUTED_NEW_COMPUTED=True,
            TEMPORAL_TILES=_TEMPORAL_TILES,
            LAZY_TEMPORAL=self._radiance_lazy_temporal,  # radiance lazy gdn postalign-flag
        )
''', SENT + " postalign-flag", "mamba_utils: LAZY_TEMPORAL on the align postprocess launch")
apply(M,
'''def get_mamba_groups(kv_cache_config: KVCacheConfig) -> dict[MambaSpec, list[int]]:''',
'''def _radiance_lazy() -> bool:
    """radiance lazy gdn flag: RADIANCE_GDN_LAZY=1 (read once)."""
    import os as _os
    v = getattr(_radiance_lazy, "_v", None)
    if v is None:
        v = _radiance_lazy._v = _os.environ.get("RADIANCE_GDN_LAZY", "0") == "1"
    return v


def get_mamba_groups(kv_cache_config: KVCacheConfig) -> dict[MambaSpec, list[int]]:''', SENT + " flag", "mamba_utils: lazy flag helper")
print("patch_gdn_lazy: done")
