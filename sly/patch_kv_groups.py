#!/usr/bin/env python3
"""RADIANCE_KV_GROUP_SIZE: override the hybrid KV-cache group size (vLLM 0.29.0, default off).

Why (vllm7, 2026-09-16, plan "Schritt 9"): `_get_kv_cache_groups_uniform_page_size` picks the
number of layers per KV cache group as the *smallest* layer bucket. Qwen3.8-27B + the DFlash2
drafter has three buckets: 16 full-attention layers, 48 gated-delta-net layers and the drafter's
5 sliding-window layers, so the stock heuristic uses group_size = 5 and pads 16 -> 20 and
48 -> 50 ("Add 4 padding layers, may waste at most 25.00%" / "Add 2 ... 4.17%" in the log).
Padding layers are not allocated, but every block of a padded group still costs the full
group's page count, so a 32k request takes 246 blocks x 5 pages = 1230 pages where 1064 are
in use (+15.6 %). With RADIANCE_KV_GROUP_SIZE=8 the buckets split into 2 + 6 + 1 groups
(only the drafter group is padded, 5 of 8 slots): 136 blocks x 8 pages = 1088 pages per
32k request, i.e. +13 % KV tokens at the same pool size and fewer groups than stock (9 vs 15).
The value is whatever divides the attention-type layer counts best; groups with fewer real
layers than group_size are the same code path the stock padding already exercises.

Unset = stock heuristic, byte-identical behaviour.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/core/kv_cache_utils.py"

A = ("        group_size = max_num_layers\n"
     "    grouped_layers = []\n")

apply(F, A,
      "        group_size = max_num_layers\n"
      "    # --- radiance (sly/patch_kv_groups.py): RADIANCE_KV_GROUP_SIZE overrides group_size ---\n"
      "    # The min-bucket heuristic pads 16 attn + 48 GDN layers to groups of 5 when a 5-layer\n"
      "    # drafter is present (15.6 % of the pool); see the patch's docstring for the numbers.\n"
      "    _radiance_gs = os.environ.get(\"RADIANCE_KV_GROUP_SIZE\", \"\")\n"
      "    if _radiance_gs:\n"
      "        logger.info(\n"
      "            \"[radiance] KV cache group size %s (stock heuristic: %d; layer buckets %s)\",\n"
      "            _radiance_gs, group_size, [len(_l) for _l in layer_buckets],\n"
      "        )\n"
      "        group_size = int(_radiance_gs)\n"
      "    grouped_layers = []\n",
      "RADIANCE_KV_GROUP_SIZE", "kv_cache_utils: RADIANCE_KV_GROUP_SIZE overrides the KV group size")
