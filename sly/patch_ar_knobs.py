#!/usr/bin/env python3
"""Make the TP=2 all-reduce size gates and the compressed-path geometry env-tunable.

radiance_allreduce.py (StillDeadcode's N-rank module) hardcodes the P2P size gate at 48 MB. That
holds one prefill chunk's all-reduce at --max-num-batched-tokens 4096 (4096 x 5120 x bf16 = 40 MiB);
at chunk 8192 the message is 80 MiB and every prefill reduction silently falls back to RCCL.
ggz14 measured that fallback on 2x R9700 (TP=2, Qwen3.8-27B): 3.145 ms per call on RCCL vs 1.317 ms
on the kernel, +3.1-12.8 % prefill on MXFP4 once the cap fits the chunk (patch_ar_maxbytes.py).
Their patch anchors on the older two-rank file, so this is the same knob set for the module this
image ships:

  RADIANCE_AR_MAX_KB        P2P size cap at TP=2 (default 49152 = unchanged). Must hold
                            tokens x hidden x 2 of the largest chunk, e.g. 98304 at chunk 8192.
  RADIANCE_AR_QUANT_MIN_KB  smallest message on the 6-bit wire at TP=2 (default 128 = unchanged)
  RADIANCE_AR_QNT / _QNB    threads per block / block cap of the 6-bit path (default 1024 / 48)

TP=1 never builds the all-reduce, and TP 4/8 size their slots from the library's own ceiling, so
only a TP=2 serve with the env set behaves differently. Idempotent; fatal on drift.
"""
import sysconfig
from pathlib import Path

from _patchlib import apply

AR = Path(sysconfig.get_paths()["purelib"]) / "radiance_allreduce.py"

apply(AR,
      "_MAX_BYTES = 49152 * 1024\n",
      "# radiance (sly/patch_ar_knobs.py): has to track --max-num-batched-tokens, see the patch.\n"
      "_MAX_BYTES = int(os.environ.get(\"RADIANCE_AR_MAX_KB\") or 49152) * 1024\n",
      "RADIANCE_AR_MAX_KB",
      "all-reduce: RADIANCE_AR_MAX_KB")
apply(AR,
      "_QUANT_MIN_BYTES = 128 * 1024\n",
      "_QUANT_MIN_BYTES = int(os.environ.get(\"RADIANCE_AR_QUANT_MIN_KB\") or 128) * 1024\n",
      "RADIANCE_AR_QUANT_MIN_KB",
      "all-reduce: RADIANCE_AR_QUANT_MIN_KB")
apply(AR,
      "        self.qnt = 1024        # threads/block for the compressed push (wire-bound; not sensitive)\n"
      "        self.qmax_nb = 48      # block cap for the ws=2 compressed path\n",
      "        # radiance (sly/patch_ar_knobs.py): sweepable without a rebuild\n"
      "        self.qnt = int(os.environ.get(\"RADIANCE_AR_QNT\") or 1024)\n"
      "        self.qmax_nb = int(os.environ.get(\"RADIANCE_AR_QNB\") or 48)\n",
      "RADIANCE_AR_QNB",
      "all-reduce: RADIANCE_AR_QNT / RADIANCE_AR_QNB")
