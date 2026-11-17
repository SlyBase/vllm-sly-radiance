#!/usr/bin/env python3
"""gfx1201 (R9700 / RDNA4) logic patches for vLLM on ROCm. Idempotent string replacement on the
installed site-packages copies; re-running is safe.

The amdsmi-enumeration failures (platform undetected, device_count==0, get_device_name IndexError,
gcn-arch query) are not patched here: one root cause (amdsmi locked out after HIP init), fixed at
interpreter startup by radiance_amdsmi (amdsmi_init before HIP)."""
import sysconfig
from pathlib import Path
from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])


def main():
    # A0. _get_gcn_arch: honor RADIANCE_GFX_ARCH env. amdsmi's asic_info "target_graphics_version"
    #     is empty for gfx1201 so the query raises, and the torch.cuda fallback then crashes at import.
    #     A deterministic env read avoids both. VLLM_ROCM_GCN_ARCH is the pre-0.5.1 name for the same
    #     knob, still accepted; it is no longer the one the image sets, because vLLM warns about
    #     VLLM_*-prefixed variables it does not itself define.
    apply(
        SP / "vllm/platforms/rocm.py",
        "    try:\n        return _query_gcn_arch_from_amdsmi()",
        '    import os as _os\n'
        '    _env = _os.environ.get("RADIANCE_GFX_ARCH") or _os.environ.get("VLLM_ROCM_GCN_ARCH")\n'
        '    if _env:\n'
        '        return _env\n'
        "    try:\n        return _query_gcn_arch_from_amdsmi()",
        '_env = _os.environ.get("RADIANCE_GFX_ARCH")',
        "honor RADIANCE_GFX_ARCH env",
    )
    # A. AITER enablement: vLLM gates AITER on MI3xx; treat gfx12x as capable too.
    apply(
        SP / "vllm/_aiter_ops.py",
        "        from vllm.platforms.rocm import get_cdna_version\n\n"
        "        return get_cdna_version() > 2",
        "        from vllm.platforms.rocm import get_cdna_version, on_gfx12x\n\n"
        "        return get_cdna_version() > 2 or on_gfx12x()",
        "get_cdna_version() > 2 or on_gfx12x()",
        "is_aiter_found_and_supported: allow gfx12x",
    )
    # B. Triton HIPDriver.is_active(): stock gates on torch.cuda.is_available(), which is False in
    #    vLLM's GPU-less inspection subprocess where aiter touches the driver at import. A ROCm torch
    #    build always targets HIP, so gate on torch.version.hip.
    apply(
        SP / "triton/backends/amd/driver.py",
        "            return torch.cuda.is_available() and (torch.version.hip is not None)",
        "            return torch.version.hip is not None",
        "            return torch.version.hip is not None",
        "Triton HIPDriver.is_active: gate on torch.version.hip",
    )
    # C (dropped at the vLLM 0.34.0 bump): the AITER top-k/top-p sampler gate.
    #    v0.32.0 selected AITER's sampler in topk_topp_sampler.py via
    #    rocm_aiter_ops.is_enabled(); its C++/HIP kernel fails to build on RDNA4, so this hunk
    #    gated it to MI3xx. v0.34.0 removed the AITER sampler from the v1 sampling path entirely
    #    (no aiter.ops.sampling / aiter_sample reference in the vLLM tree; apply_top_k_top_p
    #    routes Triton/pytorch only), so the gate's behaviour is native now.

if __name__ == "__main__":
    main()
