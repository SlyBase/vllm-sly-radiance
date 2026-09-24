"""Register the ParoQuant quantization configs (ggz14's paroquant/) in every process, on request.

ParoQuant checkpoints carry `quant_method: paroquant` (int4 + learned rotations) or
`paroquant_mxfp4` (MXFP4 weights + rotations); vLLM knows neither, so the config classes have to be
registered before ModelConfig resolves the checkpoint's method. ggz14's launcher appends the two
imports to the stdlib sitecustomize.py; this image does the same from a .pth file
(radiance_quant_plugins.pth, read at interpreter start after radiance_amdsmi.pth), gated on
RADIANCE_PAROQUANT=1 so a Quark / NVFP4 / INT4 serve imports nothing. A failed import is reported
and swallowed: a broken side module must not take the other quantization methods down with it.
"""
import os
import sys

if os.environ.get("RADIANCE_PAROQUANT", "0") == "1":
    try:
        import radiance_paroquant  # noqa: F401  registers "paroquant"
        import radiance_paroquant_mxfp4  # noqa: F401  registers "paroquant_mxfp4"
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[radiance.paroquant] registration failed: {e!r}\n")
