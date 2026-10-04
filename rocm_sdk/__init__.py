"""Stand-in for TheRock's `rocm_sdk` package, for AMD's torch wheels (stable.repo.amd.com/rocm/whl-next).

Those wheels call `rocm_sdk.initialize_process(preload_shortnames=[...], check_version=...)` from
torch/_rocm_init.py to dlopen their ROCm libraries out of the pip-installed ROCm (`rocm[libraries]`).
This image ships the same ROCm release in /opt/rocm (the pruned base tree), so the stand-in preloads
from there instead: one ROCm, one HIP runtime in the process, and aiter/vLLM/radiance kernels built
by that tree's hipcc see the same libraries torch does. The source-built torch never imports this.
"""
import ctypes
import glob
import os
import warnings

_ROCM = os.environ.get("ROCM_PATH", "/opt/rocm")


def _rocm_version():
    try:
        with open(os.path.join(_ROCM, ".info", "version")) as f:
            return f.read().strip()
    except OSError:
        return None


def initialize_process(preload_shortnames=(), check_version=None, **_):
    have = _rocm_version()
    if check_version and have and not have.startswith(check_version):
        warnings.warn(f"torch wheel built for ROCm {check_version}, {_ROCM} is {have}", stacklevel=2)
    # /opt/rocm/lib is deliberately not in ld.so.conf (it carries its own copies of system
    # libraries), so torch's NEEDED entries resolve only against what is already loaded: dlopen
    # each library by full path, and glibc satisfies the later NEEDED by soname. Names are matched
    # case-insensitively (the shortname "miopen" is libMIOpen.so); host-math holds rocm-openblas.
    dirs = [os.path.join(_ROCM, "lib"), os.path.join(_ROCM, "lib", "host-math", "lib")]
    index = {}
    for d in dirs:
        for path in glob.glob(os.path.join(d, "lib*.so*")):
            stem = os.path.basename(path).split(".so")[0][3:].lower()
            index.setdefault(stem, []).append(path)
    missing = []
    for name in preload_shortnames:
        hits = sorted(index.get(name.lower(), []), key=len)
        if hits:
            ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
        else:
            missing.append(name)
    if missing and os.environ.get("RADIANCE_ROCM_SDK_DEBUG"):
        print(f"[rocm_sdk stand-in] not under {_ROCM}: {', '.join(missing)}")
