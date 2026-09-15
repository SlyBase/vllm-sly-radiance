#!/usr/bin/env python3
"""patch_gdn_metadata's numpy fast path leaves `non_spec_sequence_masks_cpu` unbound.

patch_gdn_metadata (radiance) replaces the per-request torch loop in
vllm/v1/attention/backends/gdn_attn.py GDNAttentionMetadataBuilder.build()
with one numpy pass and drops the torch tensor
`non_spec_sequence_masks_cpu`. The mixed spec/non-spec branch further down
(spec decode + a prefill in the same batch, i.e. the cudagraph warmup and
every request arrival while others are decoding) still indexes with it ->
`UnboundLocalError: non_spec_sequence_masks_cpu` -> EngineCore dies at init
whenever `--speculative-config` is set. Not reachable without spec decode,
which is why the radiance patch was never hit before DFlash on vllm7.

Fix: rebuild the tensor from the numpy mask right where the numpy path
defines `_mask_np`; same values as before the patch.

Verified 2026-09-15 on vllm7 (DFlash2 k=7, MXFP4 target), see homelab plan
groovy-floating-twilight.md. Runs AFTER patch_gdn_metadata and
sly/patch_short_prefill (anchor is the post-patch text).
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

F = Path(sysconfig.get_paths()["purelib"]) / "vllm/v1/attention/backends/gdn_attn.py"

apply(F,
      '                _qlen_np = query_lens_cpu.numpy()\n'
      '                _nonspec_np = _qlen_np[~_mask_np]\n',
      '                _qlen_np = query_lens_cpu.numpy()\n'
      '                # --- SLY NONSPEC-MASK: the mixed spec/non-spec branch below indexes\n'
      '                # with non_spec_sequence_masks_cpu; the numpy path never defined it.\n'
      '                non_spec_sequence_masks_cpu = torch.from_numpy(~_mask_np)\n'
      '                _nonspec_np = _qlen_np[~_mask_np]\n',
      '                # --- SLY NONSPEC-MASK: the mixed spec/non-spec branch below indexes',
      'gdn metadata: define non_spec_sequence_masks_cpu on the numpy path')
