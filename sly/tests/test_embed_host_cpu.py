#!/usr/bin/env python3
"""CPU tests for sly/radiance_embed_host.py (embed_tokens host offload).

Pure mock: no torch, no vllm, no GPU. The refusal logic (_refusal) is the
safety gate that keeps the table in VRAM when the swap would break
correctness — every case must be caught.

Stubs for torch/vllm are installed per-test and restored after, so they do
not leak into other tests in the same pytest process (CI runs `pytest
sly/tests` with real torch-cpu; a fake torch there would poison them).

Run:  python3 -m pytest sly/tests/test_embed_host_cpu.py -q
"""
import sys
import types
import contextlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _stub_torch():
    torch_mod = types.ModuleType("torch")
    torch_mod.bfloat16 = "bf16"
    torch_mod.float16 = "fp16"
    torch_mod.float32 = "fp32"
    torch_mod.float64 = "fp64"
    torch_mod.Tensor = type("Tensor", (), {})
    torch_mod.nn = types.ModuleType("torch.nn")
    torch_mod.nn.functional = types.ModuleType("torch.nn.functional")
    torch_mod.nn.functional.embedding = lambda *a, **k: None
    torch_mod.cuda = types.ModuleType("torch.cuda")
    torch_mod.cuda.Event = MagicMock
    torch_mod.cuda.synchronize = lambda *a: None
    torch_mod.cuda.memory_allocated = lambda *a: 0
    torch_mod.cuda.empty_cache = lambda *a: None
    sys.modules["torch"] = torch_mod
    sys.modules["torch.nn"] = torch_mod.nn
    sys.modules["torch.nn.functional"] = torch_mod.nn.functional
    sys.modules["torch.cuda"] = torch_mod.cuda


_STUB_KEYS = ["torch", "torch.nn", "torch.nn.functional", "torch.cuda",
              "vllm", "vllm.model_executor", "vllm.model_executor.layers",
              "vllm.model_executor.layers.vocab_parallel_embedding",
              "radiance_embed_host"]


@contextlib.contextmanager
def _with_stubs():
    """Install fake torch+vllm, import radiance_embed_host, yield (module, ve),
    then restore sys.modules so the stubs do not leak into other tests."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if any(
        k == s or k.startswith(s + ".") for s in ["torch", "vllm", "radiance_embed_host"])}
    _stub_torch()
    vllm_mod = types.ModuleType("vllm")
    me = types.ModuleType("vllm.model_executor")
    layers = types.ModuleType("vllm.model_executor.layers")
    ve_mod = types.ModuleType("vllm.model_executor.layers.vocab_parallel_embedding")

    class VocabParallelEmbedding:
        pass

    class ParallelLMHead(VocabParallelEmbedding):
        pass

    class UnquantizedEmbeddingMethod:
        pass

    class QuarkEmbeddingMethod:
        pass

    ve_mod.VocabParallelEmbedding = VocabParallelEmbedding
    ve_mod.ParallelLMHead = ParallelLMHead
    ve_mod.UnquantizedEmbeddingMethod = UnquantizedEmbeddingMethod
    ve_mod.QuarkEmbeddingMethod = QuarkEmbeddingMethod

    vllm_mod.model_executor = me
    me.layers = layers
    layers.vocab_parallel_embedding = ve_mod

    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.model_executor"] = me
    sys.modules["vllm.model_executor.layers"] = layers
    sys.modules["vllm.model_executor.layers.vocab_parallel_embedding"] = ve_mod

    import radiance_embed_host
    try:
        yield radiance_embed_host, ve_mod
    finally:
        for k in _STUB_KEYS:
            sys.modules.pop(k, None)
        for k, v in saved.items():
            sys.modules[k] = v


def _cuda_weight(dim=2, contiguous=True, dtype="bf16", data_ptr=99):
    w = MagicMock()
    w.is_cuda = True
    w.dim.return_value = dim
    w.is_contiguous.return_value = contiguous
    w.dtype = dtype
    w.data_ptr.return_value = data_ptr
    return w


def _make_emb(ve, weight, quant_method=None, shared_with=False):
    emb = ve.VocabParallelEmbedding()
    emb.quant_method = quant_method if quant_method is not None else ve.UnquantizedEmbeddingMethod()
    emb.weight = weight
    model = MagicMock()
    if shared_with:
        named = [("embed_tokens", emb.weight), ("lm_head", emb.weight)]
    else:
        named = [("embed_tokens", emb.weight)]
    model.named_parameters.return_value = iter(named)
    return emb, model


# --- ENABLED env flag (read at import) ---
def test_enabled_default_off():
    import os
    os.environ.pop("RADIANCE_EMBED_HOST", None)
    with _with_stubs() as (mod, _):
        assert mod.ENABLED is False


def test_enabled_on():
    import os
    os.environ["RADIANCE_EMBED_HOST"] = "1"
    with _with_stubs() as (mod, _):
        assert mod.ENABLED is True
    os.environ.pop("RADIANCE_EMBED_HOST", None)


def test_enabled_explicit_off():
    import os
    os.environ["RADIANCE_EMBED_HOST"] = "0"
    with _with_stubs() as (mod, _):
        assert mod.ENABLED is False
    os.environ.pop("RADIANCE_EMBED_HOST", None)


# --- _view_fn (no vllm.utils -> None, no crash) ---
def test_view_fn_returns_none():
    with _with_stubs() as (mod, _):
        assert mod._view_fn() is None


# --- _target_embedding ---
def test_target_embedding_with_language_model():
    with _with_stubs() as (mod, _):
        model = MagicMock()
        lm = MagicMock()
        inner = MagicMock()
        inner.embed_tokens = "THE_EMBED"
        lm.model = inner
        model.get_language_model.return_value = lm
        got_lm, got_emb = mod._target_embedding(model)
        assert got_lm is lm
        assert got_emb == "THE_EMBED"


def test_target_embedding_without_language_model():
    with _with_stubs() as (mod, _):
        model = MagicMock(spec=[])  # no attributes -> no get_language_model
        inner = MagicMock()
        inner.embed_tokens = "THE_EMBED2"
        model.model = inner
        got_lm, got_emb = mod._target_embedding(model)
        assert got_lm is model
        assert got_emb == "THE_EMBED2"


# --- _refusal: every guard branch ---
def test_refusal_no_embed_tokens():
    with _with_stubs() as (mod, _):
        r = mod._refusal(MagicMock(), MagicMock(), None)
        assert r is not None and "no language_model.model.embed_tokens" in r


def test_refusal_wrong_type():
    with _with_stubs() as (mod, _):
        r = mod._refusal(MagicMock(), MagicMock(), object())
        assert r is not None and "not a VocabParallelEmbedding" in r


def test_refusal_parallel_lm_head():
    with _with_stubs() as (mod, ve):
        r = mod._refusal(MagicMock(), MagicMock(), ve.ParallelLMHead())
        assert r is not None and "not a VocabParallelEmbedding" in r


def test_refusal_quantized():
    with _with_stubs() as (mod, ve):
        emb, model = _make_emb(ve, _cuda_weight(), quant_method=ve.QuarkEmbeddingMethod())
        r = mod._refusal(model, MagicMock(), emb)
        assert r is not None and "quantized" in r


def test_refusal_weight_none():
    with _with_stubs() as (mod, ve):
        emb, model = _make_emb(ve, None)
        r = mod._refusal(model, MagicMock(), emb)
        assert r is not None and "not a contiguous 2-D" in r


def test_refusal_weight_not_cuda():
    with _with_stubs() as (mod, ve):
        w = MagicMock()
        w.is_cuda = False
        w.dim.return_value = 2
        w.is_contiguous.return_value = True
        w.dtype = "bf16"
        w.data_ptr.return_value = 99
        emb, model = _make_emb(ve, w)
        r = mod._refusal(model, MagicMock(), emb)
        assert r is not None and "not a contiguous 2-D" in r


def test_refusal_weight_not_2d():
    with _with_stubs() as (mod, ve):
        emb, model = _make_emb(ve, _cuda_weight(dim=3))
        r = mod._refusal(model, MagicMock(), emb)
        assert r is not None and "not a contiguous 2-D" in r


def test_refusal_weight_not_contiguous():
    with _with_stubs() as (mod, ve):
        emb, model = _make_emb(ve, _cuda_weight(contiguous=False))
        r = mod._refusal(model, MagicMock(), emb)
        assert r is not None and "not a contiguous 2-D" in r


def test_refusal_bad_dtype():
    with _with_stubs() as (mod, ve):
        emb, model = _make_emb(ve, _cuda_weight(dtype="fp64"))
        r = mod._refusal(model, MagicMock(), emb)
        assert r is not None and "dtype" in r


def test_refusal_tied_storage():
    with _with_stubs() as (mod, ve):
        emb, model = _make_emb(ve, _cuda_weight(), shared_with=True)
        r = mod._refusal(model, MagicMock(), emb)
        assert r is not None and "shared with" in r


def test_refusal_valid_accept():
    with _with_stubs() as (mod, ve):
        emb, model = _make_emb(ve, _cuda_weight(data_ptr=99))
        r = mod._refusal(model, MagicMock(), emb)
        assert r is None, f"expected OK, got refusal: {r}"
