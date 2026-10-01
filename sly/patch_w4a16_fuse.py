#!/usr/bin/env python3
"""Call sites for the W4A16 fusions of sly/patch_w4a16_tiles.py (compressed-tensors INT4 target and the
DFlash2 W4A16 drafter on gfx1201). The kernels, custom ops and the post-load transforms live in
rdna_hybrid_w4a16.py (patch_w4a16_tiles); this only wires them in:

  1. gpu/model_runner.py (V2 runner, the one this image runs): radiance_w4a16_postload(model, drafter)
     right after the drafter is loaded -- inside the load memory profile, before profiling and graph
     capture. It interleaves the gate/up rows of every W4A16 MLP (fused silu), merges each GDN layer's
     bf16 in_proj_ba into its W4A16 in_proj_qkvz as int4 rows, and re-quantizes the DFlash2 grouped-conv
     kernel_projection (bf16, 10 calls per step) to int4. With RADIANCE_DFLASH_BF16=1 the drafter arm
     instead dequantizes its W4A16 row sets to bf16 weights once (apply_weights runs plain GEMMs) and
     skips the silu / conv int4 transforms.
  2. qwen2_moe.py Qwen2MoeMLP.forward (the dense MLP of Qwen3.5/3.8) and qwen2.py Qwen2MLP.forward (the
     drafter's Qwen3MLP): gate_up GEMM + silu(gate) * up in one kernel (radiance_w4a16_silu) when the
     layer was interleaved. Sets _RADIANCE_W4_SILU so the post-load step knows the call site exists.
  3. qwen_gdn_linear_attn.py forward_cuda / forward_hip: in_proj_qkvz + in_proj_ba as one GEMM with two
     contiguous outputs (radiance_w4a16_split). Sets _RADIANCE_W4_QKVZ_BA.
  4. qwen3_dflash.py _project_context_kv: the context-KV projection on the drafter's own int4 rows
     (radiance_dflash_kv_project) instead of the 105 MB bf16 fused weight, which is then never built.

  5. envs.py compile_factors: RADIANCE_W4A16_*, RADIANCE_GDN_BA_W4, RADIANCE_DFLASH_*,
     RADIANCE_LMHEAD_INT4_TILED -- the transforms
     change the traced graph (fused ops instead of gate_up + act, one merged GEMM, a replaced module,
     or a plain GEMM on a load-time bf16 weight), so a knob flip must not replay an AOT graph compiled
     with the other setting.

Every site keeps the stock path when its layer was not transformed (MXFP4 target, bf16, other
quantizations). Knobs: RADIANCE_W4A16_SILU, RADIANCE_GDN_BA_W4, RADIANCE_DFLASH_CONV_W4,
RADIANCE_DFLASH_KV_W4 (default 1; RADIANCE_GDN_BA_W4 default 0 since window E), RADIANCE_DFLASH_BF16
(default 0: the load-time bf16 expansion of the drafter's W4A16 rows, see patch_w4a16_tiles).
Runs after sly/patch_fused_norm_quant (its anchors include the
fused-norm lines) and sly/patch_nvfp4_compile_key (the compile_factors line).
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
IMPORT = ("try:  # --- radiance (sly/patch_w4a16_fuse.py): W4A16 fused call sites ---\n"
          "    from vllm.model_executor.kernels.linear.mixed_precision import rdna_hybrid_w4a16 as _radiance_w4a16\n"
          "except Exception:  # never block model import on the fusion module\n"
          "    _radiance_w4a16 = None\n")

# --- 1. post-load hook in the V2 model runner ---
apply(SP / "vllm/v1/worker/gpu/model_runner.py",
      "                    eplb_models_added = self.eplb.maybe_register_speculator(\n"
      "                        self.speculator, self.speculative_config, load_dummy_weights\n"
      "                    )\n"
      "        time_after_load = time.perf_counter()\n",
      "                    eplb_models_added = self.eplb.maybe_register_speculator(\n"
      "                        self.speculator, self.speculative_config, load_dummy_weights\n"
      "                    )\n"
      "            # radiance (sly/patch_w4a16_fuse.py): W4A16 post-load fusions before profiling / capture\n"
      "            try:\n"
      "                from vllm.model_executor.kernels.linear.mixed_precision import (\n"
      "                    rdna_hybrid_w4a16 as _rw4,\n"
      "                )\n"
      "\n"
      "                _rw4.radiance_w4a16_postload(self.model, getattr(self.speculator, \"model\", None))\n"
      "            except Exception as _rw4_exc:  # a speed transform never fails the load\n"
      "                logger.warning(\"radiance w4a16 post-load skipped: %r\", _rw4_exc)\n"
      "        time_after_load = time.perf_counter()\n",
      "_rw4.radiance_w4a16_postload(self.model",
      "gpu/model_runner: W4A16 post-load hook")

# --- 2a. dense MLP of Qwen3.5 / 3.8 ---
F = SP / "vllm/model_executor/models/qwen2_moe.py"
apply(F,
      "try:  # --- radiance (sly/patch_fused_norm_quant.py): RADIANCE_FUSED_NORM_QUANT ---\n",
      IMPORT + "_RADIANCE_W4_SILU = _radiance_w4a16 is not None\n"
      "try:  # --- radiance (sly/patch_fused_norm_quant.py): RADIANCE_FUSED_NORM_QUANT ---\n",
      "_RADIANCE_W4_SILU = _radiance_w4a16 is not None",
      "qwen2_moe: W4A16 fusion import")
apply(F,
      "    def forward(self, x):\n"
      "        gate_up, _ = self.gate_up_proj(x)\n"
      "        if _rfn is not None and _rfn.SILU and _rfn.mlp_ok(self):\n",
      "    def forward(self, x):\n"
      "        if getattr(self.gate_up_proj, \"_radiance_silu\", False):\n"
      "            # radiance (sly/patch_w4a16_fuse.py): gate_up GEMM + silu(gate) * up in one kernel\n"
      "            out = _radiance_w4a16.radiance_mlp_forward(self, x)\n"
      "            out, _ = self.down_proj(out)\n"
      "            if self.expert_gate is not None:\n"
      "                out = F.sigmoid(self.expert_gate(x)[0]) * out\n"
      "            return out\n"
      "        gate_up, _ = self.gate_up_proj(x)\n"
      "        if _rfn is not None and _rfn.SILU and _rfn.mlp_ok(self):\n",
      "gate_up GEMM + silu(gate) * up in one kernel\n            out = _radiance_w4a16.radiance_mlp_forward",
      "qwen2_moe: Qwen2MoeMLP fused W4A16 silu")

# --- 2b. the drafter's MLP (qwen3.py: Qwen3MLP = Qwen2MLP) ---
F = SP / "vllm/model_executor/models/qwen2.py"
apply(F,
      "class Qwen2MLP(nn.Module):\n",
      IMPORT + "_RADIANCE_W4_SILU = _radiance_w4a16 is not None\n\n\nclass Qwen2MLP(nn.Module):\n",
      "_RADIANCE_W4_SILU = _radiance_w4a16 is not None",
      "qwen2: W4A16 fusion import")
apply(F,
      "    def forward(self, x):\n"
      "        gate_up, _ = self.gate_up_proj(x)\n"
      "        x = self.act_fn(gate_up)\n"
      "        x, _ = self.down_proj(x)\n"
      "        return x\n",
      "    def forward(self, x):\n"
      "        if getattr(self.gate_up_proj, \"_radiance_silu\", False):\n"
      "            # radiance (sly/patch_w4a16_fuse.py): gate_up GEMM + silu(gate) * up in one kernel\n"
      "            x, _ = self.down_proj(_radiance_w4a16.radiance_mlp_forward(self, x))\n"
      "            return x\n"
      "        gate_up, _ = self.gate_up_proj(x)\n"
      "        x = self.act_fn(gate_up)\n"
      "        x, _ = self.down_proj(x)\n"
      "        return x\n",
      "x, _ = self.down_proj(_radiance_w4a16.radiance_mlp_forward(self, x))",
      "qwen2: Qwen2MLP fused W4A16 silu")

# --- 3. GDN: in_proj_qkvz + in_proj_ba as one GEMM ---
F = SP / "vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
apply(F,
      "try:  # --- radiance (sly/patch_fused_norm_quant.py): RADIANCE_FUSED_NORM_QUANT ---\n",
      IMPORT + "_RADIANCE_W4_QKVZ_BA = _radiance_w4a16 is not None\n"
      "try:  # --- radiance (sly/patch_fused_norm_quant.py): RADIANCE_FUSED_NORM_QUANT ---\n",
      "_RADIANCE_W4_QKVZ_BA = _radiance_w4a16 is not None",
      "gdn: W4A16 fusion import")
apply(F,
      "        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
      "        ba, _ = self.in_proj_ba(hidden_states)\n"
      "        if _rfn_q:",
      "        if getattr(self.in_proj_qkvz, \"_radiance_ba\", 0):\n"
      "            # radiance (sly/patch_w4a16_fuse.py): qkvz + ba in one int4 GEMM, two contiguous outputs\n"
      "            mixed_qkvz, ba = _radiance_w4a16.radiance_qkvz_ba(self.in_proj_qkvz, hidden_states)\n"
      "        else:\n"
      "            mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
      "            ba, _ = self.in_proj_ba(hidden_states)\n"
      "        if _rfn_q:",
      "mixed_qkvz, ba = _radiance_w4a16.radiance_qkvz_ba(",
      "gdn: forward_cuda merged qkvz + ba")
apply(F,
      "            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
      "            projected_states_ba, _ = self.in_proj_ba(hidden_states)\n"
      "            if _rfn_q:",
      "            if getattr(self.in_proj_qkvz, \"_radiance_ba\", 0):\n"
      "                # radiance (sly/patch_w4a16_fuse.py): qkvz + ba in one int4 GEMM\n"
      "                projected_states_qkvz, projected_states_ba = _radiance_w4a16.radiance_qkvz_ba(\n"
      "                    self.in_proj_qkvz, hidden_states\n"
      "                )\n"
      "            else:\n"
      "                projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)\n"
      "                projected_states_ba, _ = self.in_proj_ba(hidden_states)\n"
      "            if _rfn_q:",
      "projected_states_qkvz, projected_states_ba = _radiance_w4a16.radiance_qkvz_ba(",
      "gdn: forward_hip merged qkvz + ba")

# --- 4. DFlash context-KV on the drafter's int4 rows ---
F = SP / "vllm/model_executor/models/qwen3_dflash.py"
apply(F,
      "_DFLASH_FP8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)\n",
      IMPORT + "\n_DFLASH_FP8 = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)\n",
      "rdna_hybrid_w4a16 as _radiance_w4a16",
      "qwen3_dflash: W4A16 fusion import")
apply(F,
      "        if self._fused_kv_weight is None:\n"
      "            self._fused_kv_weight = torch.cat(\n"
      "                [\n"
      "                    _dflash_kv_weight_rows(a.qkv_proj, a.q_size)\n"
      "                    for a in self._kv_source_attn\n"
      "                ],\n"
      "                dim=0,\n"
      "            )\n"
      "        all_kv_flat = F.linear(\n"
      "            normed_context_states, self._fused_kv_weight, self._fused_kv_bias\n"
      "        )\n",
      "        all_kv_flat = None\n"
      "        if _radiance_w4a16 is not None:\n"
      "            # radiance (sly/patch_w4a16_fuse.py): the drafter's own int4 K/V rows, no bf16 copy\n"
      "            all_kv_flat = _radiance_w4a16.radiance_dflash_kv_project(\n"
      "                self, normed_context_states\n"
      "            )\n"
      "        if all_kv_flat is None:\n"
      "            if self._fused_kv_weight is None:\n"
      "                self._fused_kv_weight = torch.cat(\n"
      "                    [\n"
      "                        _dflash_kv_weight_rows(a.qkv_proj, a.q_size)\n"
      "                        for a in self._kv_source_attn\n"
      "                    ],\n"
      "                    dim=0,\n"
      "                )\n"
      "            all_kv_flat = F.linear(\n"
      "                normed_context_states, self._fused_kv_weight, self._fused_kv_bias\n"
      "            )\n",
      "all_kv_flat = _radiance_w4a16.radiance_dflash_kv_project(",
      "qwen3_dflash: int4 context-KV projection")

# --- 5. the transform knobs in the torch.compile cache key ---
apply(SP / "vllm/envs.py",
      '        if _radiance_k.startswith(("RADIANCE_FUSED_NORM_QUANT", "RADIANCE_EMBED_", "RADIANCE_NVFP4_")):\n',
      '        # sly/patch_w4a16_fuse.py: the W4A16 post-load transforms change the traced graph\n'
      '        if _radiance_k.startswith(("RADIANCE_FUSED_NORM_QUANT", "RADIANCE_EMBED_", "RADIANCE_NVFP4_",\n'
      '                                   "RADIANCE_W4A16_", "RADIANCE_GDN_BA_W4", "RADIANCE_DFLASH_",\n'
      '                                   "RADIANCE_LMHEAD_INT4_TILED")):\n',
      '"RADIANCE_W4A16_", "RADIANCE_GDN_BA_W4", "RADIANCE_DFLASH_",',
      'envs: W4A16 transform knobs in compile_factors')
