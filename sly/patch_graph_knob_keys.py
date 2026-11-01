#!/usr/bin/env python3
"""Add the graph-changing knobs of patch_attn_gate_view.py and patch_gdn_glue.py to the compile cache key.

RADIANCE_ATTN_GATE_VIEW (sly/patch_attn_gate_view.py), RADIANCE_GDN_EMPTY_OUT and
RADIANCE_GDN_STRIDED_GATES (patch_gdn_glue.py) change the traced graph, while envs.compile_factors()
hashes VLLM_* plus the RADIANCE_* prefixes listed there: without this a knob flip would replay a stale
AOT graph (serve-mxfp4.sh worked around it for the GDN pair with a cache-dir suffix; the prebuilt
image's launch has no such suffix). Same mechanism as sly/patch_nvfp4_compile_key.py: unset knobs add
nothing, so the default cache key is unchanged. Anchors on sly/patch_w4a16_fuse.py's line, so it runs
after that one in the Dockerfile loop.
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

ENVS = (Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
        / "vllm" / "envs.py")

apply(ENVS,
      '                                   "RADIANCE_LMHEAD_INT4_TILED")):\n',
      '                                   "RADIANCE_LMHEAD_INT4_TILED",\n'
      '                                   # sly/patch_graph_knob_keys.py: trace-time knobs\n'
      '                                   "RADIANCE_ATTN_GATE_VIEW", "RADIANCE_GDN_EMPTY_OUT",\n'
      '                                   "RADIANCE_GDN_STRIDED_GATES")):\n',
      '"RADIANCE_ATTN_GATE_VIEW", "RADIANCE_GDN_EMPTY_OUT"',
      "envs: gate-view / GDN glue knobs in compile_factors")
