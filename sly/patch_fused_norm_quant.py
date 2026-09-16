#!/usr/bin/env python3
"""Wire sly/mxfp4/radiance_fused_norm.py into Qwen3.5 / Qwen3-Next (vLLM 0.29.0).

Behind RADIANCE_FUSED_NORM_QUANT=1 (default off; every hook is `_rfn.<FLAG> and <gate>`, which
dynamo folds to a constant, so the off state traces the stock graph). Hooks:

  qwen3_next.py           Qwen3NextDecoderLayer.forward (inherited by Qwen3_5DecoderLayer):
                          input_layernorm / post_attention_layernorm with a residual ->
                          radiance::add_rms_quant, hidden_states becomes the (q, scale) pair.
                          The residual == None branch (layer 0) and the final model.norm stay.
  qwen2_moe.py            Qwen2MoeMLP.forward (= Qwen3NextMLP): act_fn -> radiance::silu_mul_quant.
  qwen_gdn_linear_attn.py forward_hip (aiter branch) and forward_cuda: take the (q, scale) pair
                          into in_proj_qkvz / in_proj_ba (apply_weights' tuple branch) and carry
                          the bf16 projection for the .dtype/.device reads that follow;
                          _output_projection -> radiance::gdn_norm_quant straight into out_proj.
  envs.py                 compile_factors(): add RADIANCE_FUSED_NORM_QUANT* to the torch.compile
                          cache key (only VLLM_* are hashed; a knob flip must not replay a stale
                          AOT graph).

Anchors were taken from the patched 0.1.5 image, so this runs last in the Dockerfile loop.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

V = Path(sysconfig.get_paths()["purelib"]) / "vllm"
NEXT = V / "model_executor/models/qwen3_next.py"
MOE = V / "model_executor/models/qwen2_moe.py"
GDN = V / "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
ENVS = V / "envs.py"

IMPORT = (
    "try:  # --- radiance (sly/patch_fused_norm_quant.py): RADIANCE_FUSED_NORM_QUANT ---\n"
    "    import radiance_fused_norm as _rfn\n"
    "except Exception as _rfn_exc:  # never block model import on our own module\n"
    "    import sys as _rfn_sys\n"
    "\n"
    "    _rfn_sys.stderr.write(f\"[radiance.fused_norm] disabled, import failed: {_rfn_exc!r}\\n\")\n"
    "    _rfn = None\n"
)
IMPORT_SENTINEL = "    import radiance_fused_norm as _rfn\n"

# ---- qwen3_next.py ------------------------------------------------------------------------------
A = "from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP\n"
apply(NEXT, A, A + IMPORT, IMPORT_SENTINEL, "qwen3_next: import radiance_fused_norm")

A = ("        else:\n"
     "            hidden_states, residual = self.input_layernorm(hidden_states, residual)\n")
apply(NEXT, A,
      "        elif _rfn is not None and _rfn.ADD_RMS and _rfn.input_ok(self):\n"
      "            # radiance: add + rms_norm + per-token fp8 quant in one launch; hidden_states\n"
      "            # becomes the (q, scale) pair the W4A8 GEMMs consume without re-quantizing\n"
      "            hidden_states, residual = _rfn.add_rms_quant(\n"
      "                self.input_layernorm, hidden_states, residual\n"
      "            )\n" + A,
      "_rfn.input_ok(self)", "qwen3_next: input_layernorm -> radiance::add_rms_quant")

A = ("        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)\n"
     "        if self.use_attn_reduce_scatter_for_moe:\n")
apply(NEXT, A,
      "        if _rfn is not None and _rfn.ADD_RMS and _rfn.post_ok(self):\n"
      "            # radiance: add + rms_norm + fp8 quant -> (q, scale) into mlp.gate_up_proj\n"
      "            hidden_states, residual = _rfn.add_rms_quant(\n"
      "                self.post_attention_layernorm, hidden_states, residual\n"
      "            )\n"
      "        else:\n"
      "            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)\n"
      "        if self.use_attn_reduce_scatter_for_moe:\n",
      "_rfn.post_ok(self)", "qwen3_next: post_attention_layernorm -> radiance::add_rms_quant")

# ---- qwen2_moe.py -------------------------------------------------------------------------------
A = "logger = init_logger(__name__)\n"
apply(MOE, A, A + "\n" + IMPORT, IMPORT_SENTINEL, "qwen2_moe: import radiance_fused_norm")

A = "        out = self.act_fn(gate_up)\n"
apply(MOE, A,
      "        if _rfn is not None and _rfn.SILU and _rfn.mlp_ok(self):\n"
      "            # radiance: silu(gate) * up + per-token fp8 quant -> (q, scale) into down_proj\n"
      "            out = _rfn.silu_mul_quant(gate_up)\n"
      "        else:\n"
      "            out = self.act_fn(gate_up)\n",
      "_rfn.mlp_ok(self)", "qwen2_moe: Qwen2MoeMLP act_fn -> radiance::silu_mul_quant")

# ---- qwen_gdn_linear_attn.py --------------------------------------------------------------------
A = ("try:\n"
     "    import radiance_gdn as _radiance_gdn\n"
     "except Exception:\n"
     "    _radiance_gdn = None\n")
apply(GDN, A, A + IMPORT, IMPORT_SENTINEL, "gdn: import radiance_fused_norm")

A = ("        if GDN_AITER_TRITON_AVAILABLE:\n"
     "            num_tokens = hidden_states.size(0)\n"
     "            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
     "            projected_states_ba, _ = self.in_proj_ba(hidden_states)\n")
apply(GDN, A,
      "        if GDN_AITER_TRITON_AVAILABLE:\n"
      "            # radiance fused-norm carrier (forward_hip): hidden_states may be (q, scale)\n"
      "            _rfn_q = isinstance(hidden_states, tuple)\n"
      "            num_tokens = (hidden_states[0] if _rfn_q else hidden_states).size(0)\n"
      "            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
      "            projected_states_ba, _ = self.in_proj_ba(hidden_states)\n"
      "            if _rfn_q:  # only .dtype/.device are read below: carry the bf16 projection\n"
      "                hidden_states = projected_states_qkvz\n",
      "radiance fused-norm carrier (forward_hip)", "gdn: forward_hip takes the (q, scale) pair")

A = ("        num_tokens = hidden_states.size(0)\n"
     "        # ============================================================\n"
     "        # Part 1: Input Projection\n"
     "        # ============================================================\n"
     "        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
     "        ba, _ = self.in_proj_ba(hidden_states)\n")
apply(GDN, A,
      "        # radiance fused-norm carrier (forward_cuda): hidden_states may be (q, scale)\n"
      "        _rfn_q = isinstance(hidden_states, tuple)\n"
      "        num_tokens = (hidden_states[0] if _rfn_q else hidden_states).size(0)\n"
      "        # ============================================================\n"
      "        # Part 1: Input Projection\n"
      "        # ============================================================\n"
      "        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
      "        ba, _ = self.in_proj_ba(hidden_states)\n"
      "        if _rfn_q:  # only .dtype/.device are read below: carry the bf16 projection\n"
      "            hidden_states = mixed_qkvz\n",
      "radiance fused-norm carrier (forward_cuda)", "gdn: forward_cuda takes the (q, scale) pair")

A = ("        by the compilation pass when fuse_norm_quant is enabled.\n"
     "        \"\"\"\n"
     "        z_shape_og = z.shape\n")
apply(GDN, A,
      "        by the compilation pass when fuse_norm_quant is enabled.\n"
      "        \"\"\"\n"
      "        if _rfn is not None and _rfn.GDN and _rfn.gdn_ok(self):\n"
      "            # radiance: per-head gated rms_norm + fp8 quant in one launch -> out_proj\n"
      "            output, _ = self.out_proj(_rfn.gdn_norm_quant(self, core_attn_out, z))\n"
      "            return output\n"
      "        z_shape_og = z.shape\n",
      "_rfn.gdn_ok(self)", "gdn: _output_projection -> radiance::gdn_norm_quant")

# ---- envs.py ------------------------------------------------------------------------------------
A = "    return factors\n"
apply(ENVS, A,
      "    # --- radiance (sly/patch_fused_norm_quant.py) ---\n"
      "    # RADIANCE_FUSED_NORM_QUANT* change the traced graph (tuple activations, new custom\n"
      "    # ops); only VLLM_* are hashed above, so without this a knob flip would replay a stale\n"
      "    # AOT graph. Unset knobs add nothing: the default cache key is unchanged.\n"
      "    for _radiance_k in sorted(os.environ):\n"
      "        if _radiance_k.startswith(\"RADIANCE_FUSED_NORM_QUANT\"):\n"
      "            factors[_radiance_k] = os.environ[_radiance_k]\n" + A,
      "RADIANCE_FUSED_NORM_QUANT", "envs: RADIANCE_FUSED_NORM_QUANT* in compile_factors")
