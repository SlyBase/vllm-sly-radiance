"""embed_tokens as int8 rows (or int4 group-128 nibbles) for the Quark MXFP4 target.

Off by default; RADIANCE_EMBED_INT8=1 switches it on, RADIANCE_EMBED_BITS=4 picks the
nibble variant. Read at import time, like the other RADIANCE_* knobs.

Why: the checkpoint excludes embed_tokens from quantisation, so the 248320 x 5120 bf16 table
(2.37 GiB, tie_word_embeddings=False) is the largest single tensor left after the int4
lm_head -- and it is only ever *gathered* (a few rows per token), never multiplied, so its
precision costs nothing at run time. Per-row symmetric int8 (scale = amax / 127, fp32 scale
per row) is well below the bf16 rounding of the activations that follow; int4 g128 RTN is
~16x the error and ships only behind the GSM8K gate. int8 hands 1.18 GiB, int4 1.74 GiB to
the KV-cache profiling that runs later (vllm7, 2026-09-16: +18k / +27k tokens at 32k).

How: `RadianceEmbedInt8` subclasses UnquantizedEmbeddingMethod so create_weights and the
bf16 weight loader stay untouched. process_weights_after_loading quantises in row chunks
and replaces `layer.weight` in place (same name; int8 [V, H] or uint8 [V, H//2]) and adds
`layer.weight_scale` (fp32 [V] or bf16 [V, H//128]). embedding() gathers the rows, rescales
and returns the model dtype -- plain torch ops, so torch.compile / CUDA graphs see an
ordinary index_select + mul. TP=1 only (the fused embedding op reads layer.weight directly).

Hooked in by sly/patch_embed_int8.py in VocabParallelEmbedding.__init__ (Qwen3Next builds
embed_tokens without quant_config, so QuarkConfig.get_quant_method never sees it); the knob
is part of the torch.compile cache key. The DFlash drafter shares the target's module
(llm_base_proposer._maybe_share_embeddings), so its lookups go through embedding() too.
"""

import os

import torch
from torch.nn import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
)

logger = init_logger("vllm.radiance.embed_int8")

ENABLED = os.environ.get("RADIANCE_EMBED_INT8", "0") == "1"
BITS = int(os.environ.get("RADIANCE_EMBED_BITS", "8"))
GROUP_SIZE = 128  # int4 only
CHUNK_ROWS = 8192  # 8192 x 5120 fp32 = 168 MB transient per chunk


def quant_method_for(layer: torch.nn.Module, prefix: str):
    """VocabParallelEmbedding.__init__ hook: our method for an enabled embedding, else None."""
    if not ENABLED or not isinstance(layer, VocabParallelEmbedding):
        return None
    if isinstance(layer, ParallelLMHead) or getattr(layer, "tp_size", 1) != 1:
        return None
    if BITS not in (4, 8):
        raise ValueError(f"RADIANCE_EMBED_BITS must be 4 or 8, got {BITS}")
    logger.info_once("[radiance] %s -> int%d embedding table", prefix, BITS)
    return RadianceEmbedInt8()


class RadianceEmbedInt8(UnquantizedEmbeddingMethod):
    """create_weights as the base class; the loaded weight and embedding() change."""

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w = layer.weight
        if w.dtype in (torch.int8, torch.uint8):  # second pass over a shared module
            return
        assert w.dim() == 2, f"embedding weight must be [vocab, hidden], got {tuple(w.shape)}"
        n, k = w.shape
        layer.radiance_out_dtype = w.dtype
        if BITS == 8:
            q = torch.empty((n, k), dtype=torch.int8, device=w.device)
            scale = torch.empty((n,), dtype=torch.float32, device=w.device)
            for i in range(0, n, CHUNK_ROWS):
                blk = w[i:i + CHUNK_ROWS].float()
                s = blk.abs().amax(dim=1).clamp_min_(1e-8) / 127.0
                q[i:i + CHUNK_ROWS] = torch.round(blk / s[:, None]).clamp_(-127, 127).to(torch.int8)
                scale[i:i + CHUNK_ROWS] = s
                del blk, s
        else:
            g = GROUP_SIZE
            assert k % g == 0 and k % 2 == 0, f"hidden {k} not divisible by group {g}"
            q = torch.empty((n, k // 2), dtype=torch.uint8, device=w.device)
            scale = torch.empty((n, k // g), dtype=torch.bfloat16, device=w.device)
            for i in range(0, n, CHUNK_ROWS):
                blk = w[i:i + CHUNK_ROWS].float().view(-1, k // g, g)
                s = blk.abs().amax(dim=2, keepdim=True).clamp_min_(1e-8) / 7.0
                nib = (torch.round(blk / s).clamp_(-8, 7) + 8).to(torch.uint8).view(-1, k)
                q[i:i + CHUNK_ROWS] = nib[:, 0::2] | (nib[:, 1::2] << 4)
                scale[i:i + CHUNK_ROWS] = s.view(-1, k // g).to(torch.bfloat16)
                del blk, s, nib
        layer.weight = Parameter(q, requires_grad=False)
        layer.weight_scale = Parameter(scale, requires_grad=False)
        logger.info(
            "[radiance] embed_tokens quantised to int%d: [%d, %d], %.2f GiB freed",
            BITS, n, k, (n * k * w.element_size() - q.numel() - scale.numel() * scale.element_size()) / 2**30,
        )

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        ids = input_.reshape(-1)
        rows = layer.weight.index_select(0, ids)
        out_dtype = layer.radiance_out_dtype
        if BITS == 8:
            out = rows.to(out_dtype) * layer.weight_scale.index_select(0, ids).to(out_dtype)[:, None]
        else:
            k = rows.shape[1] * 2
            nib = torch.stack((rows & 0xF, rows >> 4), dim=-1).view(-1, k // GROUP_SIZE, GROUP_SIZE)
            s = layer.weight_scale.index_select(0, ids).to(out_dtype)[:, :, None]
            out = ((nib.to(out_dtype) - 8) * s).view(-1, k)
        return out.view(*input_.shape, -1)
