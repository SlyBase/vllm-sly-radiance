#!/usr/bin/env python3
"""Install the KV4 backend into a running image: copy the modules next to the other radiance_*.py
in site-packages and add the KV4 member to vLLM's attention backend registry.

Runs in the Dockerfile patch loop at image build (the kernel prebuild is skipped without /dev/kfd and
happens on first use), and still works from the /patches mount (serve-mxfp4.sh with KVCACHE_DTYPE=kv4). Idempotent: the copy is a plain
overwrite, the registry edit is anchored and skipped when present.
"""
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

from _patchlib import apply

HERE = Path(__file__).resolve().parent
SP = Path(sysconfig.get_paths()["purelib"])
REG = SP / "vllm/v1/attention/backends/registry.py"

MODULES = ("radiance_kv4_format.py", "radiance_kv4_store.py", "radiance_kv4_kernel.py",
           "radiance_kv4_attn.py", "radiance_kv4_fake.py",
           # KV4H: the HIP kernel's layout, store, launcher, backend and the kernel source itself
           # (radiance_kv4h_kernel builds it next to the installed module on first use)
           "radiance_kv4h_format.py", "radiance_kv4h_store.py", "radiance_kv4h_kernel.py",
           "radiance_kv4h_attn.py", "radiance_kv4h.hip", "kv4h_tune.json",
           "kv4i_route.py")          # the int4 libr4d route decision (imported by radiance_kv4h_attn)

for name in MODULES:
    shutil.copyfile(HERE / name, SP / name)
    print(f"  COPY  {name} -> {SP}")

# anchored on the R4D_HYBRID member so the fork's opt-in backends stay together; KV4 is as explicit
# an opt-in as those (never auto-selected)
apply(REG,
      '    R4D_HYBRID = "radiance_r4d_hybrid_attn.R4DHybridAttentionBackend"\n',
      '    R4D_HYBRID = "radiance_r4d_hybrid_attn.R4DHybridAttentionBackend"\n'
      '    KV4 = "radiance_kv4_attn.KV4AttentionBackend"\n',
      'KV4 = "radiance_kv4_attn.KV4AttentionBackend"', "registry: KV4 backend")
# KV4FAKE: fp8 path with the K/V rounded through a 4-bit format (accuracy checks, see the module)
apply(REG,
      '    KV4 = "radiance_kv4_attn.KV4AttentionBackend"\n',
      '    KV4 = "radiance_kv4_attn.KV4AttentionBackend"\n'
      '    KV4FAKE = "radiance_kv4_fake.KV4FakeAttentionBackend"\n',
      'KV4FAKE = "radiance_kv4_fake', "registry: KV4FAKE backend")
apply(REG,
      '    KV4FAKE = "radiance_kv4_fake.KV4FakeAttentionBackend"\n',
      '    KV4FAKE = "radiance_kv4_fake.KV4FakeAttentionBackend"\n'
      '    KV4H = "radiance_kv4h_attn.KV4HAttentionBackend"\n',
      'KV4H = "radiance_kv4h_attn', "registry: KV4H backend")
# TetraAttention: the name the kernel ships under; KV4H stays as the working-name alias
apply(REG,
      '    KV4H = "radiance_kv4h_attn.KV4HAttentionBackend"\n',
      '    KV4H = "radiance_kv4h_attn.KV4HAttentionBackend"\n'
      '    TETRA = "radiance_kv4h_attn.TetraAttentionBackend"\n',
      'TETRA = "radiance_kv4h_attn', "registry: TETRA backend")

# build TetraAttention's HIP kernel now (hipcc, ~20 s) so the engine's first forward does not wait for it;
# the launcher imports torch / aiter, hence a subprocess from the installed copies
if os.environ.get("RADIANCE_KV4H_PREBUILD", "1") != "0" and os.path.exists("/dev/kfd"):  # no GPU at image build
    # the format's default (query planes, K group) with both row-tile counts: the tune table picks
    # row_tiles 2 for decode and 4 for verify / prefill
    r = subprocess.run([sys.executable, "-c", "import radiance_kv4h_kernel as K; [K._ext(row_tiles=r) for r in (2, 4)]; "
                        "print('  BUILD radiance_kv4h (default format, row tiles 2 and 4) ok')"],
                       cwd=str(SP), capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr.strip()[-2000:])
    if r.returncode:
        print("[kv4] WARNING: KV4H kernel prebuild failed (the backend will retry on first use)")
