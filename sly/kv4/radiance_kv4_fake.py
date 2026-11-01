"""KV4FAKE: the production fp8 path with the KV *rounded through a 4-bit format* on its way into
the cache -- accuracy of a format at fp8 speed, for GSM8K-level checks before a kernel exists.

Select with `--attention-backend KV4FAKE` (fp8 cache, everything else R4D_HYBRID) and name the
format in /patches/sly/kv4/FAKE (the serve scripts cannot pass -e):
  kernel   K uniform per half + V geo per token   (what radiance_kv4_kernel runs)
  asym32   asymmetric int4, scale + zero point per 32 rotated dims, K and V
  asym16   the same per 16 dims
  lloyd32  Lloyd-Max codebook, scale per 32 dims
  k4v8     K asymmetric int4 per 16 dims, V left at fp8 (which side carries the loss?)
  k8v4     K left at fp8, V asymmetric int4 per 16 dims
  none     nothing rounded: must reproduce the fp8 baseline (harness control)
The fused rope+cache path is disabled so every K/V goes through do_kv_cache_update.
"""
import os

import torch

import radiance_kv4_format as F
from radiance_r4d_hybrid_attn import R4DHybridAttentionBackend, R4DHybridAttentionImpl

_MARK = "/patches/sly/kv4/FAKE"
FAKE = os.environ.get("RADIANCE_KV4_FAKE") or (open(_MARK).read().strip() if os.path.exists(_MARK) else "kernel")


def fake_quant(key: torch.Tensor, value: torch.Tensor):
    n, h, _ = key.shape
    k = key.reshape(-1, F.HEAD).to(torch.float32)
    v = value.reshape(-1, F.HEAD).to(torch.float32)
    if FAKE == "none":          # harness control: the fp8 path with nothing rounded
        pass
    elif FAKE == "kernel":
        k = F.reconstruct(k, F.CODEBOOK_K, False, F.NGROUP_K)
        v = F.reconstruct(v, F.CODEBOOK_V, False, F.NGROUP_V)
    elif FAKE == "k4v8":
        k = F.asym_reconstruct(k, 16)
    elif FAKE == "k8v4":
        v = F.asym_reconstruct(v, 16)
    elif FAKE.startswith("asym"):
        g = int(FAKE[4:])
        k, v = F.asym_reconstruct(k, g), F.asym_reconstruct(v, g)
    elif FAKE.startswith("lloyd"):
        ng = F.HEAD // int(FAKE[5:])
        k, v = F.reconstruct(k, "lloyd", False, ng), F.reconstruct(v, "lloyd", False, ng)
    else:
        raise ValueError(f"unknown fake format {FAKE!r}")
    return k.reshape(n, h, F.HEAD).to(key.dtype), v.reshape(n, h, F.HEAD).to(value.dtype)


class KV4FakeAttentionBackend(R4DHybridAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "KV4FAKE"

    @staticmethod
    def get_impl_cls():
        return KV4FakeAttentionImpl


class KV4FakeAttentionImpl(R4DHybridAttentionImpl):
    def fused_rope_kvcache_supported(self):
        return False

    def fused_qk_norm_rope_kvcache_supported(self):
        return False

    def fused_qk_norm_mrope_kvcache_supported(self) -> bool:
        return False

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if key.shape[0] > 0:
            key, value = fake_quant(key, value)
        return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
