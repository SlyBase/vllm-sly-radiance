#!/usr/bin/env python3
"""Install the RADIANCE skinny-GEMM hook on vLLM's ROCm unquantized-linear chokepoint.

Any bf16 projection whose weight is small enough that rocBLAS lays it out as a handful of
workgroups goes to the R4D split-K kernel instead. radiance_gemm decides -- it holds the measured
(N, K) -> config table and declines everything else -- so adding a shape is an edit to that module,
not a re-patch. Gated by RADIANCE_USE_R4D through the same module.

The band starts at M = 6 because vLLM's own wvSplitK serves M <= 5, and it ends at 16 because that
is where the kernel holds its accumulators.
"""
import sysconfig
from pathlib import Path
from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
UTIL = SP / "vllm/model_executor/layers/utils.py"

# Re-anchored for vLLM 0.32.0: rocm_unquantized_gemm_impl gained a VLLM_BATCH_INVARIANT
# early-return at the top of the body, so the hook anchors on the platform import (unique in
# 0.31.0 and 0.32.0 alike) and sits after the batch-invariant branch: a determinism mode the
# user opts into keeps upstream's deterministic GEMM; the default path runs the hook first,
# exactly as on 0.31.0.
#
# The hook reads the module-global `_radiance_gemm`: 0.31-time inserted the `try: import`
# block by rewriting the `def` itself; the 0.32 re-anchor dropped that and shipped only the
# body hook -> `NameError: _radiance_gemm is not defined` at the first profile run
# (dflash2 candidate selector). The module import is its own hunk now, anchored on the def
# signature (unique in both versions), so fresh 0.31 and 0.32 trees, re-runs and
# already-patched cross-version copies all land here.
ANCHOR = (
    "    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx950, on_gfx1250\n"
)
NEW = (
    "    from vllm.platforms.rocm import on_gfx1x, on_gfx9, on_gfx950, on_gfx1250\n"
    "    # --- RADIANCE skinny GEMM kernel (patch_skinny_gemm.py) ---\n"
    "    if (_radiance_gemm is not None and _radiance_gemm.ENABLED and bias is None\n"
    "            and weight.dim() == 2 and x.dtype == torch.bfloat16\n"
    "            and weight.dtype == torch.bfloat16):\n"
    "        _out = _radiance_gemm.maybe_gemm(x, weight)\n"
    "        if _out is not None:\n"
    "            return _out\n"
)
# Module scope: `_radiance_gemm` must exist no matter which layer file body is patched in.
# (0.31-era patched trees already carry the block -> NOOP on the sentinel; a fresh tree
# gets it before the def. Note the try/except at the bottom of THIS script is only the
# patch-side probe -- the runtime name lives in utils.py, not here.)
IMPORT_ANCHOR = (
    "def rocm_unquantized_gemm_impl(\n"
    "    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None\n"
    ") -> torch.Tensor:\n"
)
IMPORT_NEW = (
    "# --- RADIANCE skinny GEMM module import (patch_skinny_gemm.py) ---\n"
    "try:\n"
    "    import radiance_gemm as _radiance_gemm\n"
    "except Exception:\n"
    "    _radiance_gemm = None\n"
    "\n"
    "def rocm_unquantized_gemm_impl(\n"
    "    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None\n"
    ") -> torch.Tensor:\n"
)


def main():
    apply(UTIL, ANCHOR, NEW, "RADIANCE skinny GEMM kernel (patch_skinny_gemm.py)",
          "route measured skinny bf16 GEMMs -> custom gfx1201 kernel")
    apply(UTIL, IMPORT_ANCHOR, IMPORT_NEW,
          "RADIANCE skinny GEMM module import (patch_skinny_gemm.py)",
          "radiance_gemm import at module scope")


if __name__ == "__main__":
    main()
