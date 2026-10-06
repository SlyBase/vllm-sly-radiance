#!/usr/bin/env python3
"""Wire the gated-fold MLP (radiance_fused_norm.gated_gemm_quant) into Qwen2MoeMLP.forward.

RADIANCE_MXFP4_GATED_FOLD=1 (default 0): gate_up GEMM + SwiGLU + e4m3 quant for down_proj in one
custom op (`radiance::mxfp4_gated_quant`) instead of gate_up_proj -> silu_mul_quant. The hook is
`_rfn.GATED_FOLD and isinstance(x, tuple) and _rfn.gated_ok(self)`; with the knob off GATED_FOLD
is a module constant False and the traced graph is the stock one. The knob is also added to
vLLM's compile cache key (envs.compile_factors hashes VLLM_* only), so a flip never replays a
stale AOT graph. Runs after patch_fused_norm_quant (anchors on its hunk).
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

V = Path(sysconfig.get_paths()["purelib"]) / "vllm"
MOE = V / "model_executor/models/qwen2_moe.py"
ENVS = V / "envs.py"

A = ("        gate_up, _ = self.gate_up_proj(x)\n"
     "        if _rfn is not None and _rfn.SILU and _rfn.mlp_ok(self):\n")
apply(MOE, A,
      "        if (_rfn is not None and _rfn.GATED_FOLD and isinstance(x, tuple)\n"
      "                and _rfn.gated_ok(self)):\n"
      "            # radiance: gate_up GEMM + silu(gate) * up in the GEMM epilogue + fp8 quant\n"
      "            out, _ = self.down_proj(_rfn.gated_gemm_quant(self, x))\n"
      "            return out\n" + A,
      "_rfn.gated_gemm_quant(self, x)", "qwen2_moe: Qwen2MoeMLP gated fold")

A = ("            factors[_radiance_k] = os.environ[_radiance_k]\n"
     "    return factors\n")
apply(ENVS, A,
      "            factors[_radiance_k] = os.environ[_radiance_k]\n"
      "    # sly/patch_gated_fold.py: RADIANCE_MXFP4_GATED_FOLD swaps the MLP's traced custom ops\n"
      "    if \"RADIANCE_MXFP4_GATED_FOLD\" in os.environ:\n"
      "        factors[\"RADIANCE_MXFP4_GATED_FOLD\"] = os.environ[\"RADIANCE_MXFP4_GATED_FOLD\"]\n"
      "    return factors\n",
      "factors[\"RADIANCE_MXFP4_GATED_FOLD\"]", "envs: RADIANCE_MXFP4_GATED_FOLD in compile_factors")
