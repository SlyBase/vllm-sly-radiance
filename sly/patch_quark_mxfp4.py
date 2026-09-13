#!/usr/bin/env python3
"""Native MXFP4 (OCP micro-scaling) linear GEMM on gfx1201, for Quark W4A4 checkpoints such as
amd/Qwen3.8-27B-Quark-AWQ-MXFP4.

Ported from ggz14/radiance-vllm-mxfp4's patch_quark_mxfp4.py (targeting vLLM 0.27.1) to vLLM
0.29.0, per plan groovy-floating-twilight.md Subagent C1. vLLM's MXFP4 kernel-plugin machinery
(MxFp4LinearKernel ABC, _POSSIBLE_MXFP4_KERNELS[platform], init_mxfp4_linear_kernel()) is
unchanged in spirit between 0.27 and 0.29 -- three of the five hunks below are the same fix ggz14
wrote, re-anchored against source drift; two are new, closing a gap that did not exist at 0.27.1.

Anchor-by-anchor findings (vLLM 0.29.0 source, tag v0.29.0):

  1. REGISTER (kernels/linear/__init__.py, init_mxfp4_linear_kernel): DRIFTED. 0.29.0 refactored
     `linear_backend = _get_linear_backend()` out of this function body into a
     `_resolve_backend_kernels()` call; the original anchor no longer matches. Re-anchored below.

  2. SUPPORTS_MX gate (kernels/linear/mxfp4/aiter.py, AiterMxfp4LinearKernel.is_supported):
     VERBATIM MATCH, unchanged since 0.27. supports_mx() itself still allowlists gfx95/gfx1250
     only (gfx1250 was added since ggz14 wrote this patch; gfx1201 is still excluded either way).

  3. AITER GEMM import path (same file, gemm_with_dynamic_quant): DRIFTED further, not fixed.
     Confirmed via github.com/ROCm/aiter that `aiter.ops.triton.gemm_afp4wfp4` does NOT exist as a
     top-level module in EITHER aiter v0.1.19 (what vLLM 0.29.0's own Dockerfile.rocm_base pins)
     OR v0.1.21.post2 (this fork's target) -- only under `aiter.ops.triton.gemm.basic`. This
     means vLLM 0.29.0's own AiterMxfp4LinearKernel is import-broken against its own pinned aiter
     version, on ANY ROCm arch, not just gfx1201 -- an upstream vLLM/aiter version-matrix gap, not
     something specific to this fork. Also: the imported symbol was renamed upstream, from
     ggz14's `gemm_afp4wfp4_preshuffled_weight_scales` (0.27-era aiter) to `gemm_afp4wfp4_preshuffle`
     (current vLLM source and current aiter).

  4. NEW -- module-level custom-op registration gate: at 0.29.0, the entire
     `gemm_with_dynamic_quant` custom-op definition (incl. `direct_register_custom_op`) sits behind
     a MODULE-SCOPE `if is_aiter_found_and_supported():` guard that did not exist in this shape at
     0.27.1 (ggz14's patch never touches it). is_aiter_found_and_supported() is
     `get_cdna_version() > 2` -- always False on RDNA4. Without patching this too, hunks 2+3 make
     `AiterMxfp4LinearKernel.is_supported()` return True on gfx1201, but `apply_weights()`'s
     `torch.ops.vllm.gemm_with_dynamic_quant(...)` call then fails at runtime with an unknown-op
     error, because the op was never registered for this process.

  5. NEW -- second aiter-library gate inside is_supported() itself: after the supports_mx() check
     (hunk 2), 0.29.0 added an unconditional
         if is_aiter_found_and_supported(): return True, None
         return False, "AITER not found or not supported on the current platform"
     at the end of is_supported() -- again gated on the CDNA-only check, independent of
     supports_mx(). Same fix as hunk 4: OR in vLLM 0.29.0's own native
     `is_aiter_found_and_supported_on_rdna4()` (added upstream between 0.27 and 0.29 for a
     different call site in _aiter_ops.py; reused here).

Not touched here: patch_gfx1201.py's four hunks (gcn-arch env override, AITER CDNA-gate relax,
Triton HIPDriver.is_active, AITER sampler MI3xx-gate) are inherited unchanged from upstream
StillDeadcode/vllm-radiance and were separately verified (anchor-string diff against vLLM 0.29.0
tag + Triton 3.8.0 tag) to still apply verbatim -- no action needed, see sly/README.md.

The hand-written W4A8 fp8-WMMA kernel (radiance_mxfp4_fp8.hip + the RadianceMxfp4W4A8LinearKernel
plugin in sly/mxfp4/radiance_mxfp4.py) sits ON TOP of the AITER path this patch enables: it takes
priority when RADIANCE_MXFP4_W4A8=1, and its own can_implement()/is_supported() decline (falling
through to AiterMxfp4LinearKernel) for shapes/configs it does not cover. The two are not
alternatives -- both are needed, in this order.

Gated by RADIANCE_MXFP4=1 (default off), matching the upstream flag.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
KL = SP / "vllm/model_executor/kernels/linear/__init__.py"
KA = SP / "vllm/model_executor/kernels/linear/mxfp4/aiter.py"


# --- 1. register the radiance W4A8 plugin at the head of the ROCm MXFP4 list ----------------
# Re-anchored: 0.29.0 moved `linear_backend = _get_linear_backend()` out of this function and
# added a `_resolve_backend_kernels()` filtering call in its place. Insert after that call
# instead (same semantic point: right before kernel selection begins).
REGISTER_ANCHOR = (
    "    platform = current_platform._enum\n"
    "    possible = list(_POSSIBLE_MXFP4_KERNELS.get(platform, []))\n"
    "\n"
    '    # Apply --linear-backend filtering when set.\n'
    '    possible = _resolve_backend_kernels(possible, "MXFP4")\n'
)
REGISTER_NEW = (
    "    platform = current_platform._enum\n"
    "    possible = list(_POSSIBLE_MXFP4_KERNELS.get(platform, []))\n"
    "\n"
    '    # Apply --linear-backend filtering when set.\n'
    '    possible = _resolve_backend_kernels(possible, "MXFP4")\n'
    "\n"
    "    # --- radiance (patch_quark_mxfp4.py): the gfx1201 W4A8 fp8-WMMA kernel ---\n"
    "    # Imported here, not at module scope: this runs in the worker at model load, whereas the\n"
    "    # module is imported in the parent during config parsing, where initialising HIP would\n"
    "    # force the engine core to spawn instead of fork. It declines via is_supported() unless\n"
    "    # RADIANCE_MXFP4_W4A8=1 on gfx12x, so the list is unchanged everywhere else.\n"
    "    try:\n"
    "        import radiance_mxfp4 as _radiance_mxfp4\n"
    "\n"
    "        _radiance_cls = _radiance_mxfp4.kernel_class()\n"
    "        if _radiance_cls is not None:\n"
    "            possible.insert(0, _radiance_cls)\n"
    "    except Exception as _radiance_exc:  # never block model load on our own kernel\n"
    '        logger.warning_once("[radiance] MXFP4 W4A8 kernel unavailable: %r", _radiance_exc)\n'
)

# --- 2. relax aiter's CDNA4 gate so the Triton fp4 GEMM is reachable on gfx1201 -------------
# Verbatim vs. ggz14/0.27.1 -- this exact two-line block is unchanged at 0.29.0.
SUPPORTS_ANCHOR = (
    "        if not current_platform.supports_mx():\n"
    '            return False, "current platform does not support native MXFP4 computation"\n'
)
SUPPORTS_NEW = (
    "        # --- radiance (patch_quark_mxfp4.py): gfx1201 native MXFP4, RADIANCE_MXFP4=1 ---\n"
    "        # supports_mx() allowlists gfx95/gfx1250, but Triton 3.6+ lowers tl.dot_scaled on\n"
    "        # gfx12x too (upconvert + bf16 WMMA), verified bit-identical against the emulated path.\n"
    "        _radiance_mx = current_platform.supports_mx()\n"
    "        if not _radiance_mx:\n"
    "            import os as _radiance_os\n"
    '            if _radiance_os.environ.get("RADIANCE_MXFP4", "0") == "1":\n'
    "                from vllm.platforms.rocm import on_gfx12x\n"
    "\n"
    "                _radiance_mx = bool(on_gfx12x())\n"
    "                if _radiance_mx:\n"
    "                    logger.warning_once(\n"
    '                        "[radiance] native MXFP4 enabled on gfx12x "\n'
    '                        "(aiter gemm_afp4wfp4 via tl.dot_scaled); the emulation notice "\n'
    '                        "elsewhere in the log does not apply to mxfp4 x mxfp4 layers"\n'
    "                    )\n"
    "        if not _radiance_mx:\n"
    '            return False, "current platform does not support native MXFP4 computation"\n'
)

# --- 3. aiter moved the fp4 GEMM module (confirmed still moved at both v0.1.19 and the fork's
#        v0.1.21.post2 target), and the imported symbol was renamed upstream in vLLM itself -----
IMPORT_ANCHOR = (
    "        from aiter.ops.triton.gemm_afp4wfp4 import (\n"
    "            gemm_afp4wfp4,\n"
    "            gemm_afp4wfp4_preshuffle,\n"
    "        )\n"
)
IMPORT_NEW = (
    "        # --- radiance (patch_quark_mxfp4.py): aiter has no top-level gemm_afp4wfp4 module\n"
    "        # at either v0.1.19 (vLLM 0.29.0's own Dockerfile.rocm_base pin) or v0.1.21.post2\n"
    "        # (this fork's target) -- only under aiter.ops.triton.gemm.basic. Confirmed via the\n"
    "        # aiter GitHub tree at both tags; no backward-compat shim exists at the old path.\n"
    "        from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (\n"
    "            gemm_afp4wfp4,\n"
    "            gemm_afp4wfp4_preshuffle,\n"
    "        )\n"
    "\n"
    "        # aiter allowlists gfx950/gfx1250 for fp4; gfx1201 lowers tl.dot_scaled correctly\n"
    "        # (verified bit-identical against the emulated path), so relax the assert. Done\n"
    "        # here, lazily, to keep aiter out of the plugin-load import graph.\n"
    "        import aiter.ops.triton.utils._triton.arch_info as _radiance_arch\n"
    "\n"
    "        if not _radiance_arch.is_fp4_avail():\n"
    "            _radiance_arch.is_fp4_avail = lambda: True\n"
)

# --- 4. NEW at 0.29.0: the gemm_with_dynamic_quant custom op (which apply_weights() calls) is
#        only DEFINED -- let alone registered -- behind a module-scope CDNA-only gate. Hunks 2+3
#        alone make is_supported() return True on gfx1201, but the op itself would not exist. ---
MODULE_GATE_ANCHOR = (
    "# where HIP initialization is expected.\n"
    "if is_aiter_found_and_supported():\n"
    "    from vllm.utils.torch_utils import direct_register_custom_op\n"
)
MODULE_GATE_NEW = (
    "# where HIP initialization is expected.\n"
    "# --- radiance (patch_quark_mxfp4.py): also define+register this op on gfx1201/RDNA4. gfx12\n"
    "# has no aiter CK build (is_aiter_found_and_supported() stays CDNA-only for that reason) but\n"
    "# does have the Triton fp4 GEMM this op wraps -- is_aiter_found_and_supported_on_rdna4() is\n"
    "# vLLM 0.29.0's own native predicate for exactly that distinction.\n"
    "from vllm._aiter_ops import is_aiter_found_and_supported_on_rdna4\n"
    "\n"
    "if is_aiter_found_and_supported() or is_aiter_found_and_supported_on_rdna4():\n"
    "    from vllm.utils.torch_utils import direct_register_custom_op\n"
)

# --- 5. NEW at 0.29.0: is_supported()'s final aiter-library check, same CDNA-only gate, same fix.
FINAL_GATE_ANCHOR = (
    "        if is_aiter_found_and_supported():\n"
    "            return True, None\n"
    '        return False, "AITER not found or not supported on the current platform"\n'
)
FINAL_GATE_NEW = (
    "        # --- radiance (patch_quark_mxfp4.py): OR in the RDNA4 analog, same reasoning as\n"
    "        # the module-scope gate above this class.\n"
    "        from vllm._aiter_ops import is_aiter_found_and_supported_on_rdna4\n"
    "\n"
    "        if is_aiter_found_and_supported() or is_aiter_found_and_supported_on_rdna4():\n"
    "            return True, None\n"
    '        return False, "AITER not found or not supported on the current platform"\n'
)


def main():
    apply(KL, REGISTER_ANCHOR, REGISTER_NEW,
          "[radiance] MXFP4 W4A8 kernel unavailable",
          "mxfp4: register the radiance W4A8 kernel")
    apply(KA, SUPPORTS_ANCHOR, SUPPORTS_NEW,
          "[radiance] native MXFP4 enabled on gfx12x",
          "mxfp4: relax aiter's CDNA4 gate")
    apply(KA, IMPORT_ANCHOR, IMPORT_NEW,
          "aiter.ops.triton.gemm.basic.gemm_afp4wfp4",
          "mxfp4: aiter moved-module + fp4 arch allowlist")
    apply(KA, MODULE_GATE_ANCHOR, MODULE_GATE_NEW,
          "is_aiter_found_and_supported_on_rdna4",
          "mxfp4: register gemm_with_dynamic_quant op on RDNA4")
    apply(KA, FINAL_GATE_ANCHOR, FINAL_GATE_NEW,
          "OR in the RDNA4 analog, same reasoning as",
          "mxfp4: AiterMxfp4LinearKernel.is_supported() accepts RDNA4")


if __name__ == "__main__":
    main()
