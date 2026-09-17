#!/usr/bin/env python3
"""Hook radiance_embed_int8 into VocabParallelEmbedding.__init__ (vLLM 0.29.0) + compile key.

Qwen3Next builds `VocabParallelEmbedding(vocab_size, hidden_size)` without `quant_config`
(models/qwen3_next.py:667), and the DFlash drafter does the same, so `QuarkConfig.
get_quant_method` never sees embed_tokens -- the lm_head hook point does not work here.
The swap therefore sits where the unquantized default is chosen: with RADIANCE_EMBED_INT8=1
the bf16 table (248320 x 5120 = 2.37 GiB on Qwen3.8-27B, tie_word_embeddings=False) is
stored as int8 rows (1.18 GiB back for the KV cache) or, with RADIANCE_EMBED_BITS=4, as
group-128 nibbles (1.74 GiB back). Only plain VocabParallelEmbedding is touched, never a
ParallelLMHead; a preselected model-specific quant_method is left alone.

envs.compile_factors(): the knobs change the traced graph (int8 gather + rescale instead of
F.embedding), so they join the RADIANCE_FUSED_NORM_QUANT* entries in the torch.compile cache
key. Unset knobs add nothing.
"""

import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
VOCAB = SP / "vllm/model_executor/layers/vocab_parallel_embedding.py"
ENVS = SP / "vllm/envs.py"

A = ("        if quant_method is None:\n"
     "            quant_method = UnquantizedEmbeddingMethod()\n"
     "\n")
apply(VOCAB, A,
      A +
      "        # --- radiance (sly/patch_embed_int8.py): int8/int4 embed_tokens, RADIANCE_EMBED_INT8=1 ---\n"
      "        # Qwen3Next / DFlash build embed_tokens without quant_config, so the QuarkConfig\n"
      "        # hook never sees them; swap the unquantized default here instead.\n"
      "        if type(quant_method) is UnquantizedEmbeddingMethod:\n"
      "            try:\n"
      "                import radiance_embed_int8 as _radiance_embed_int8\n"
      "\n"
      "                _radiance_method = _radiance_embed_int8.quant_method_for(self, prefix)\n"
      "                if _radiance_method is not None:\n"
      "                    quant_method = _radiance_method\n"
      "            except Exception as _radiance_exc:  # never block model load on our own embedding\n"
      "                from vllm.logger import init_logger as _radiance_init_logger\n"
      "\n"
      "                _radiance_init_logger(__name__).warning_once(\n"
      "                    \"[radiance] int8 embedding unavailable: %r\", _radiance_exc\n"
      "                )\n"
      "\n",
      "RADIANCE_EMBED_INT8", "vocab_parallel_embedding: RadianceEmbedInt8 for VocabParallelEmbedding (RADIANCE_EMBED_INT8=1)")

B = '        if _radiance_k.startswith("RADIANCE_FUSED_NORM_QUANT"):\n'
apply(ENVS, B,
      '        # sly/patch_embed_int8.py: RADIANCE_EMBED_* change the embedding subgraph as well\n'
      '        if _radiance_k.startswith(("RADIANCE_FUSED_NORM_QUANT", "RADIANCE_EMBED_")):\n',
      'RADIANCE_EMBED_', 'envs: RADIANCE_EMBED_* in compile_factors')
