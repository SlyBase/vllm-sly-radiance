#!/usr/bin/env python3
"""DFlash aux hidden states without a second residual add (RADIANCE_DFLASH_AUX_FROM_RESIDUAL=1, default off).

Qwen3NextModel.forward (models/qwen3_next.py, inherited by Qwen3.5/3.8) captures the drafter's aux
states with `_maybe_add_hidden_state`: `hidden_states + residual` after the layer before each aux
index (one Inductor add kernel, M x 5120 bf16 read twice and written, 5 per step for the DFlash2
drafter's target layers). The next layer's input norm computes the same sum again and returns it as
its residual: radiance_add_rms_quant writes res_out = bf16(f32(y) + f32(residual)), the stock
fused add-norm likewise, and that is exactly the Inductor add's result. With the knob the aux state
is that residual, taken from the layer whose input it is: the add kernel is gone, the drafter gets
the same bits. Measured in a rocprofv3 trace of 1.8.1 (prefill, chunk 4096): the adds are 0.11-0.13 %
of the GPU time at 1.5k-12k tokens (267 us each at M = 4089).

An aux index past this rank's last layer keeps the stock add (no next layer to take it from), and so
does any layer whose forward is not Qwen3NextDecoderLayer.forward. The knob changes the traced graph;
RADIANCE_DFLASH_* is already part of the compile cache key (sly/patch_fused_norm_quant.py's list).
Anchors on sly/patch_fused_norm_quant.py's decoder-layer hook, so it runs after that one.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

NEXT = (Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
        / "vllm" / "model_executor" / "models" / "qwen3_next.py")

apply(NEXT,
      "from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate\n",
      "from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate\n"
      "import os as _radiance_aux_os  # sly/patch_dflash_aux_residual.py\n"
      "\n"
      "_RADIANCE_DFLASH_AUX_FROM_RESIDUAL = (\n"
      "    _radiance_aux_os.environ.get(\"RADIANCE_DFLASH_AUX_FROM_RESIDUAL\", \"0\") == \"1\"\n"
      ")\n",
      "_RADIANCE_DFLASH_AUX_FROM_RESIDUAL = (",
      "qwen3_next: RADIANCE_DFLASH_AUX_FROM_RESIDUAL knob")

# the layer: remember its input residual (h + r) and hand it out on request
apply(NEXT,
      "        else:\n"
      "            hidden_states, residual = self.input_layernorm(hidden_states, residual)\n"
      "\n"
      "        if self.use_attn_reduce_scatter_for_moe:\n"
      "            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)\n",
      "        else:\n"
      "            hidden_states, residual = self.input_layernorm(hidden_states, residual)\n"
      "        _radiance_in_residual = residual  # sly/patch_dflash_aux_residual.py: h + r of the input\n"
      "\n"
      "        if self.use_attn_reduce_scatter_for_moe:\n"
      "            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)\n",
      "_radiance_in_residual = residual",
      "qwen3_next: layer keeps its input residual")

apply(NEXT,
      "        return hidden_states, residual\n"
      "\n"
      "\n"
      "@support_torch_compile\n"
      "class Qwen3NextModel(",
      "        if kwargs.get(\"_radiance_aux\"):  # sly/patch_dflash_aux_residual.py\n"
      "            return hidden_states, residual, _radiance_in_residual\n"
      "        return hidden_states, residual\n"
      "\n"
      "\n"
      "@support_torch_compile\n"
      "class Qwen3NextModel(",
      "return hidden_states, residual, _radiance_in_residual",
      "qwen3_next: layer returns its input residual on request")

# the model: take an aux state that feeds a layer of this rank from that layer
apply(NEXT,
      "        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)\n"
      "        for layer_idx, layer in enumerate(\n"
      "            islice(self.layers, self.start_layer, self.end_layer),\n"
      "            start=self.start_layer,\n"
      "        ):\n"
      "            hidden_states, residual = layer(\n"
      "                positions=positions,\n"
      "                hidden_states=hidden_states,\n"
      "                residual=residual,\n"
      "            )\n"
      "            self._maybe_add_hidden_state(\n"
      "                aux_hidden_states, layer_idx + 1, hidden_states, residual\n"
      "            )\n",
      "        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)\n"
      "        # radiance (sly/patch_dflash_aux_residual.py): an aux state that is the input of a layer of\n"
      "        # this rank is that layer's input residual (its input norm writes h + r): taken from there,\n"
      "        # in the same order, instead of a second add. Same bits: both are bf16(f32(h) + f32(r)).\n"
      "        _rad_aux = (\n"
      "            _RADIANCE_DFLASH_AUX_FROM_RESIDUAL\n"
      "            and bool(self.aux_hidden_state_layers)\n"
      "            and all(\n"
      "                type(lyr).forward is Qwen3NextDecoderLayer.forward\n"
      "                for lyr in islice(self.layers, self.start_layer, self.end_layer)\n"
      "            )\n"
      "        )\n"
      "        for layer_idx, layer in enumerate(\n"
      "            islice(self.layers, self.start_layer, self.end_layer),\n"
      "            start=self.start_layer,\n"
      "        ):\n"
      "            if _rad_aux and residual is not None and layer_idx in self.aux_hidden_state_layers:\n"
      "                hidden_states, residual, _rad_in = layer(\n"
      "                    positions=positions,\n"
      "                    hidden_states=hidden_states,\n"
      "                    residual=residual,\n"
      "                    _radiance_aux=True,\n"
      "                )\n"
      "                aux_hidden_states.append(_rad_in)\n"
      "            else:\n"
      "                hidden_states, residual = layer(\n"
      "                    positions=positions,\n"
      "                    hidden_states=hidden_states,\n"
      "                    residual=residual,\n"
      "                )\n"
      "            if not (\n"
      "                _rad_aux\n"
      "                and layer_idx + 1 in self.aux_hidden_state_layers\n"
      "                and layer_idx + 1 < self.end_layer\n"
      "            ):\n"
      "                self._maybe_add_hidden_state(\n"
      "                    aux_hidden_states, layer_idx + 1, hidden_states, residual\n"
      "                )\n",
      "_rad_in = layer(",
      "qwen3_next: aux states from the next layer's input residual")
