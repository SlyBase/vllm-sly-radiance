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


def test_the_routed_query_has_headroom_below_the_e4m3_ceiling(attn, monkeypatch):
    """The rotated query is |sum of 256 signed terms| ~ ||q||_2, up to 16x the unrotated magnitude, and
    e4m3 saturates at 448 (a clipped head is a wrong head). The route stores rotate(q) / GAIN and tells
    the kernel q_scale = GAIN, which it multiplies back in fp32: same numbers, 16x more headroom.
    A head with ||q||_2 ~ 1000 must come through unclipped."""
    seen, qscale = [], []
    monkeypatch.setattr(attn, "KV4I_ON", True)
    monkeypatch.setattr(attn, "KV4I_MIN_CTX", 0)
    monkeypatch.setattr(attn, "KV4I_ENTRY", lambda *a: seen.append(a))
    monkeypatch.setattr(attn._r4d, "set_q_fp8", qscale.append, raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: types.SimpleNamespace(cuda_stream=0))
    stored = []
    real_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *a, **k: (
        stored.append(real_to(self, torch.float32).abs().max().item()) if a and a[0] == torch.float8_e4m3fn
        else None) or real_to(self, *a, **k))
    impl = object.__new__(attn.TetraAttentionImpl)
    impl.head, impl.window, impl.num_heads, impl.num_kv_heads, impl.scale = 256, 0, 24, 4, 0.0625
    impl._kv4i_scratch = None
    kv = torch.zeros(2, 4, 1664, 272, dtype=torch.uint8)
    meta = types.SimpleNamespace(
        num_actual_tokens=512, max_query_len=512, max_seq_len=1024, causal=True,
        seq_lens=torch.tensor([1024], dtype=torch.int32), block_table=torch.tensor([[1]], dtype=torch.int32))
    q = torch.zeros(512, 24, 256)
    q[:, 0, :] = 62.5            # ||q||_2 = 1000 on one head, all signs equal: rotated peak ~1000 > 448
    impl.forward(types.SimpleNamespace(), q, None, None, kv, meta, torch.zeros(512, 24, 256, dtype=torch.bfloat16))
    assert len(seen) == 1 and qscale, "the route did not fire"
    assert qscale[-1] == pytest.approx(attn.KV4I_QGAIN)
    assert stored[-1] * qscale[-1] > 448, "test input does not exceed the ceiling"
    assert stored[-1] <= 448, f"stored query {stored[-1]} saturates e4m3"


def test_the_dump_writes_one_launch_with_its_reference(attn, monkeypatch, tmp_path):
    """R4D_KV4I_DUMP=<dir> + R4D_KV4I_DUMP_LAYERS: the first DUMP_N routed launches of a matching layer
    save query, the e4m3 query, the touched cache blocks, the metadata, the routed output and the Tetra
    reference, so the accuracy hunt can rebuild the fp32 truth offline (debug only, host reads)."""
    monkeypatch.setattr(attn, "KV4I_ON", True)
    monkeypatch.setattr(attn, "KV4I_MIN_CTX", 0)
    monkeypatch.setattr(attn, "KV4I_ENTRY", lambda *a: None)
    monkeypatch.setattr(attn, "KV4I_DUMP", str(tmp_path))
    monkeypatch.setattr(attn, "KV4I_DUMP_LAYERS", ("layers.47.",))
    monkeypatch.setattr(attn, "KV4I_DUMP_N", 1)
    monkeypatch.setattr(attn._r4d, "set_q_fp8", lambda s: None, raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: types.SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(attn, "kv4h_attention", lambda *a, output, **k: output.fill_(3))
    impl = object.__new__(attn.TetraAttentionImpl)
    impl.head, impl.window, impl.num_heads, impl.num_kv_heads, impl.scale = 256, 0, 24, 4, 0.0625
    impl._kv4i_scratch = None
    kv = torch.arange(3 * 4 * 1664 * 272, dtype=torch.int64).remainder(251).to(torch.uint8).reshape(3, 4, 1664, 272)
    meta = types.SimpleNamespace(
        num_actual_tokens=512, max_query_len=512, max_seq_len=1700, causal=True,
        query_start_loc=torch.tensor([0, 512], dtype=torch.int32),
        seq_lens=torch.tensor([1700], dtype=torch.int32), block_table=torch.tensor([[2, 0, 1]], dtype=torch.int32))
    for name in ("m.layers.3.self_attn.attn", "m.layers.47.self_attn.attn", "m.layers.47.self_attn.attn"):
        out = torch.zeros(512, 24, 256, dtype=torch.bfloat16)
        impl.forward(types.SimpleNamespace(layer_name=name), torch.randn(512, 24, 256), None, None, kv, meta, out)
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["m.layers.47.self_attn.attn.0.pt"], files
    d = torch.load(tmp_path / files[0])
    assert torch.equal(d["kv"], kv[torch.tensor([2, 0])])          # ceil(1700 / 1664) = 2 blocks, in sequence order
    assert d["q8"].dtype == torch.float8_e4m3fn and d["query"].shape == (512, 24, 256)
    assert d["ref"].float().eq(3).all() and d["out_rot"].shape == (512, 24, 256)
    assert d["scale"] == pytest.approx(0.0625) and d["q_scale"] == pytest.approx(attn.KV4I_QGAIN)
    assert d["seq_lens"].tolist() == [1700] and d["query_start_loc"].tolist() == [0, 512]
