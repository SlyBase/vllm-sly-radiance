"""CPU regression: TetraAttentionImpl.forward must not read a device tensor on the host while a CUDA
graph is being captured (R4D_KV4I route guard once did `.tolist()` first -> "Cannot copy between CPU and
CUDA tensors during CUDA graph capture", only seen on a GPU start). vllm/triton/HIP are stubbed; the
fake metadata tensors raise on every host read."""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

KV4 = Path(__file__).resolve().parents[2] / "kv4"


class _Any(types.ModuleType):
    """vllm.* stub: any attribute is a fresh class (usable as a base class or annotation)."""
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        cls = type(name, (), {"__init__": lambda self, *a, **k: None})
        setattr(self, name, cls)
        return cls


class HostRead(torch.Tensor):
    pass


def _boom(name):
    def f(self, *a, **k):
        raise AssertionError(f"host read during capture: {name}")
    return f


for _n in ("tolist", "item", "cpu", "numpy", "__int__", "__bool__", "__index__", "__float__"):
    setattr(HostRead, _n, _boom(_n))


@pytest.fixture
def attn(monkeypatch):
    for m in ("vllm", "vllm.config", "vllm.config.cache", "vllm.logger", "vllm.platforms",
              "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.quantization",
              "vllm.model_executor.layers.quantization.utils",
              "vllm.model_executor.layers.quantization.utils.quant_utils", "vllm.v1", "vllm.v1.attention",
              "vllm.v1.attention.backend", "vllm.v1.attention.backends",
              "vllm.v1.attention.backends.rocm_aiter_unified_attn", "vllm.v1.attention.backends.rocm_attn",
              "vllm.v1.kv_cache_interface"):
        monkeypatch.setitem(sys.modules, m, _Any(m))
    sys.modules["vllm.logger"].init_logger = lambda n: types.SimpleNamespace(
        info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    sys.modules["vllm.v1.attention.backend"].AttentionType = types.SimpleNamespace(DECODER="decoder")
    for m, attrs in {"radiance_kv4_attn": dict(KV_DTYPE="int4_per_token_head"), "r4d": {},
                     "radiance_kv4h_kernel": dict(kv4h_attention=None, LAST_LAUNCH={}),
                     "radiance_kv4h_store": dict(kv4h_store=None)}.items():
        mod = types.ModuleType(m)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, m, mod)
    monkeypatch.syspath_prepend(str(KV4))   # the real pure-python format + route modules
    spec = importlib.util.spec_from_file_location("tetra_under_test", KV4 / "radiance_kv4h_attn.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _meta(max_query_len, max_seq_len, n=8, nseq=2):
    t = lambda *s, dtype=torch.int32: torch.zeros(*s, dtype=dtype).as_subclass(HostRead)
    return types.SimpleNamespace(
        num_actual_tokens=n, max_query_len=max_query_len, max_seq_len=max_seq_len, causal=True,
        query_start_loc=t(nseq + 1), seq_lens=t(nseq), block_table=t(nseq, 4), slot_mapping=t(n))


def test_a_declined_launch_reads_no_device_value_at_all(attn, monkeypatch):
    """Not only during capture: every attention layer of every step goes through the guard, so one
    device->host read in it is a stream sync per layer (measured: -8 % prefill at 1.5k with the route
    ON but declining). The guard must decide from Python ints alone."""
    calls = []
    monkeypatch.setattr(attn, "kv4h_attention", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(attn, "KV4I_ON", True)
    monkeypatch.setattr(attn, "KV4I_ENTRY", object())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    impl = object.__new__(attn.TetraAttentionImpl)
    impl.head, impl.window, impl.num_heads, impl.num_kv_heads, impl.scale = 256, 0, 24, 4, 1.0
    kv = torch.zeros(1, 4, 1664, 272, dtype=torch.uint8)
    out = torch.zeros(4096, 24, 256, dtype=torch.bfloat16)
    # a long first chunk at shallow context (declined on context), and a deep ragged one (declined on shape)
    for meta in (_meta(4096, 4096, n=4096, nseq=1), _meta(4096, 49152, n=4100, nseq=2)):
        impl.forward(None, torch.zeros(meta.num_actual_tokens, 24, 256), None, None, kv, meta, out)
    assert len(calls) == 2


def test_forward_reads_no_host_value_during_capture(attn, monkeypatch):
    calls = []
    monkeypatch.setattr(attn, "kv4h_attention", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(attn, "KV4I_ON", True)
    monkeypatch.setattr(attn, "KV4I_ENTRY", object())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    impl = object.__new__(attn.TetraAttentionImpl)
    impl.head, impl.window, impl.num_heads, impl.num_kv_heads, impl.scale = 256, 0, 24, 4, 1.0
    t = lambda *s, dtype=torch.int32: torch.zeros(*s, dtype=dtype).as_subclass(HostRead)
    meta = types.SimpleNamespace(
        num_actual_tokens=8, max_query_len=8, max_seq_len=64, causal=True,
        query_start_loc=t(3), seq_lens=t(2), block_table=t(2, 4), slot_mapping=t(8))
    out = torch.zeros(8, 24, 256, dtype=torch.bfloat16)
    impl.forward(None, torch.zeros(8, 24, 256), None, None, None, meta, out)
    assert len(calls) == 1   # reached the Tetra kernel


def test_the_routed_launch_passes_the_rotation_folded_scale(attn, monkeypatch):
    """The cache holds ROTATED K and the query is rotated too, so q.k is inflated by HEAD (16 x 16):
    the kernel takes scale / HEAD, exactly as TetraAttention's own launch folds it
    (radiance_kv4h_kernel: scale_log2 = scale / head * log2e) and as the bench arm does. The route
    once passed `scale` unfolded: softmax 256x too sharp, KL 2.1 at >16k context, top-1 agreement 40 %,
    invisible to every CPU test and to the throughput sweep."""
    seen = []
    monkeypatch.setattr(attn, "KV4I_ON", True)
    monkeypatch.setattr(attn, "KV4I_MIN_CTX", 0)
    monkeypatch.setattr(attn, "KV4I_ENTRY", lambda *a: seen.append(a))
    monkeypatch.setattr(attn._r4d, "set_q_fp8", lambda s: None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: types.SimpleNamespace(cuda_stream=0))
    impl = object.__new__(attn.TetraAttentionImpl)
    impl.head, impl.window, impl.num_heads, impl.num_kv_heads, impl.scale = 256, 0, 24, 4, 0.0625
    impl._kv4i_scratch = None
    kv = torch.zeros(2, 4, 1664, 272, dtype=torch.uint8)
    meta = types.SimpleNamespace(
        num_actual_tokens=512, max_query_len=512, max_seq_len=1024, causal=True,
        seq_lens=torch.tensor([1024], dtype=torch.int32), block_table=torch.tensor([[1]], dtype=torch.int32))
    out = torch.zeros(512, 24, 256, dtype=torch.bfloat16)
    impl.forward(types.SimpleNamespace(), torch.randn(512, 24, 256), None, None, kv, meta, out)
    assert len(seen) == 1, "the route did not fire"
    assert seen[0][17] == pytest.approx(0.0625 / 256)      # the kernel `scale` argument
