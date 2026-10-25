#!/usr/bin/env python3
"""Keep the attention output gate a strided view (RADIANCE_ATTN_GATE_VIEW=1, default off).

Qwen3NextAttention (models/qwen3_next.py, inherited by Qwen3.5/3.8) splits q_proj's output into
per-head [q | gate] halves. On ROCm the fused qk-norm/RoPE/gate kernel is off (cuda/xpu only), and
the eager path ends with `gate = gate.reshape(*orig_shape, -1)`: a copy of a strided view. Attention
is a graph split point, so Inductor has to materialize that copy as a piece output
(`triton_poi_fused_0` before the q/k-norm reduction, M x 6144 bf16 read + written per attention
layer), and the piece after attention reads it once more for `attn_output * sigmoid(gate)`.

With the knob, `_project_qkv_gate(..., gate_view=True)` returns the gate as the [M, heads, head_dim]
view of the projection, and `forward` multiplies per head: the gate copy is gone, and the gate
multiply reads the same bytes from the projection instead (512-byte runs). Same values in the same
order, so the o_proj input is bit-identical. Measured in a rocprofv3 trace of 1.8.1 (prefill, chunk
4096): the copy is 0.38 % of the GPU time at 1.5k, 6k and 12k tokens (246 us per layer at M = 4089).
`_project_qkv_gate`'s default is unchanged: radiance_paroquant's attention forward keeps its 2-D gate.

The knob changes the traced graph, so sly/patch_graph_knob_keys.py puts it in the compile cache key.
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
      "import os as _radiance_os  # sly/patch_attn_gate_view.py\n"
      "\n"
      "_RADIANCE_ATTN_GATE_VIEW = _radiance_os.environ.get(\"RADIANCE_ATTN_GATE_VIEW\", \"0\") == \"1\"\n",
      "_RADIANCE_ATTN_GATE_VIEW = ",
      "qwen3_next: RADIANCE_ATTN_GATE_VIEW knob")

apply(NEXT,
      "    def _project_qkv_gate(\n"
      "        self,\n"
      "        qkv: torch.Tensor,\n"
      "        positions: torch.Tensor,\n"
      "    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:\n",
      "    def _project_qkv_gate(\n"
      "        self,\n"
      "        qkv: torch.Tensor,\n"
      "        positions: torch.Tensor,\n"
      "        gate_view: bool = False,\n"
      "    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:\n",
      "        gate_view: bool = False,\n",
      "qwen3_next: _project_qkv_gate(gate_view=)")

apply(NEXT,
      "            q = q.reshape(*orig_shape, -1)\n"
      "            gate = gate.reshape(*orig_shape, -1)\n",
      "            q = q.reshape(*orig_shape, -1)\n"
      "            if not gate_view:  # radiance gate view: else the strided [..., heads, head_dim] view\n"
      "                gate = gate.reshape(*orig_shape, -1)\n",
      "radiance gate view: else the strided",
      "qwen3_next: gate view instead of the gate copy")

apply(NEXT,
      "        q, k, v, gate = self._project_qkv_gate(qkv, positions)\n"
      "        attn_output = self.attn(q, k, v)\n"
      "        if gate is not None:\n"
      "            attn_output = attn_output * torch.sigmoid(gate)\n",
      "        q, k, v, gate = self._project_qkv_gate(\n"
      "            qkv, positions, gate_view=_RADIANCE_ATTN_GATE_VIEW\n"
      "        )\n"
      "        attn_output = self.attn(q, k, v)\n"
      "        if gate is not None and gate.dim() == attn_output.dim() + 1:\n"
      "            # radiance gate view (sly/patch_attn_gate_view.py): the same multiply, per head\n"
      "            attn_output = (attn_output.view(gate.shape) * torch.sigmoid(gate)).view(\n"
      "                attn_output.shape\n"
      "            )\n"
      "        elif gate is not None:\n"
      "            attn_output = attn_output * torch.sigmoid(gate)\n",
      "gate_view=_RADIANCE_ATTN_GATE_VIEW",
      "qwen3_next: per-head gate multiply")
