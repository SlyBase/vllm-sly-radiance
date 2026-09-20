#!/usr/bin/env python3
"""GPU-less check of radiance_attn_drafter: which calls it takes, which it leaves to vLLM, what it does on error.

torch, triton and vllm are stubbed and the tuned launch (`_run`) is replaced by a recorder, so this proves the
gate (drafter shape only), the split rule, the fallback on an exception, the kernel-signature guard, the env knobs
and idempotence -- not the kernel, which sly/bench_drafter_attn.py --check covers on the card.

  python3 sly/check_attn_drafter.py
"""
import importlib
import os
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

BF16, FP8, FP16 = "bf16", "fp8", "fp16"
failures = []


class T:
    """Just enough tensor for the gate: dtype, shape, strides."""

    def __init__(self, dtype, *shape):
        self.dtype, self.shape = dtype, shape
        self._strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            self._strides[i] = self._strides[i + 1] * shape[i + 1]

    def dim(self):
        return len(self.shape)

    def stride(self, i):
        return self._strides[i]

    def __len__(self):
        return self.shape[0]


class Mode:
    NONE = "none"
    FP8_PER_TENSOR = "fp8_per_tensor"


def stub_env(gfx12=True, drop_arg=None):
    """Fresh stub modules; returns (triton_attn module, orig-call recorder)."""
    for name in [n for n in sys.modules if n.startswith(("vllm", "triton", "radiance_attn_drafter"))]:
        del sys.modules[name]
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float8_e4m3fn = BF16, FP16, FP8
    sys.modules["torch"] = torch
    tr = types.ModuleType("triton")
    tr.jit = lambda f: f
    tl = types.ModuleType("triton.language")
    tl.constexpr = object
    tr.language = tl
    sys.modules.update({"triton": tr, "triton.language": tl})

    import radiance_attn_drafter as RAD                       # noqa: F401 (imported for its module-level state)
    names = list(RAD._KERNEL_ARGS)
    if drop_arg:
        names.remove(drop_arg)
    TU = types.ModuleType("vllm.v1.attention.ops.triton_unified_attention")
    TU.kernel_unified_attention = types.SimpleNamespace(arg_names=names)
    TU.reduce_segments = types.SimpleNamespace(arg_names=list(RAD._REDUCE_ARGS))
    orig_calls = []
    TA = types.ModuleType("vllm.v1.attention.backends.triton_attn")
    TA.unified_attention = lambda *a, **k: orig_calls.append((a, k)) or "orig"
    plat = types.ModuleType("vllm.platforms")
    plat.current_platform = types.SimpleNamespace(fp8_dtype=lambda: FP8)
    rocm = types.ModuleType("vllm.platforms.rocm")
    rocm.on_gfx12x = lambda: gfx12
    kvi = types.ModuleType("vllm.v1.kv_cache_interface")
    kvi.KVQuantMode = Mode
    for n, m in {"vllm": types.ModuleType("vllm"), "vllm.v1": types.ModuleType("vllm.v1"),
                 "vllm.v1.attention": types.ModuleType("vllm.v1.attention"),
                 "vllm.v1.attention.backends": types.ModuleType("vllm.v1.attention.backends"),
                 "vllm.v1.attention.ops": types.ModuleType("vllm.v1.attention.ops"),
                 "vllm.v1.attention.backends.triton_attn": TA,
                 "vllm.v1.attention.ops.triton_unified_attention": TU, "vllm.platforms": plat,
                 "vllm.platforms.rocm": rocm, "vllm.v1.kv_cache_interface": kvi}.items():
        sys.modules[n] = m
    sys.modules["vllm.v1.attention.backends"].triton_attn = TA
    sys.modules["vllm.v1.attention.ops"].triton_unified_attention = TU
    return TA, orig_calls


def load(gfx12=True, drop_arg=None, **env):
    for k in [k for k in os.environ if k.startswith("RADIANCE_ATTN_DRAFTER_")]:
        del os.environ[k]
    os.environ.update({f"RADIANCE_ATTN_DRAFTER_{k}": v for k, v in env.items()})
    TA, orig_calls = stub_env(gfx12, drop_arg)
    RAD = importlib.import_module("radiance_attn_drafter")
    return RAD, TA, orig_calls


def call_kw(nseq=1, q_len=8, head=128, q_heads=32, kv_heads=8, q_dtype=BF16, kv_dtype=FP8, causal=True,
            window=(2047, 0), mode=Mode.FP8_PER_TENSOR, **extra):
    kw = dict(q=T(q_dtype, nseq * q_len, q_heads, head), k=T(kv_dtype, 40, 896, kv_heads, head),
              v=T(kv_dtype, 40, 896, kv_heads, head), out=T(q_dtype, nseq * q_len, q_heads, head),
              cu_seqlens_q=T("i32", nseq + 1), max_seqlen_q=q_len, seqused_k=T("i32", nseq),
              max_seqlen_k=262144, softmax_scale=head ** -0.5, causal=causal, alibi_slopes=None,
              use_alibi_sqrt=False, window_size=window, block_table=T("i32", nseq, 300), softcap=0,
              q_descale=None, k_descale=T("f32", nseq, kv_heads), v_descale=T("f32", nseq, kv_heads),
              sinks=None, output_scale=None, mm_prefix_range=None, rswa_prefix_lens=None, rswa_window=None,
              kv_quant_mode=mode, k_scale_cache=None, v_scale_cache=None, chunk_lookback=-1, use_td=False)
    kw.update(extra)
    return kw


def check(cond, msg):
    if not cond:
        failures.append(msg)
        print("FAIL", msg)


def main():
    RAD, TA, orig_calls = load()
    tuned = []
    RAD._run = lambda kw, window, max_q, nseq, causal: tuned.append((window, max_q, nseq, causal))
    check(RAD.install() is True and TA.unified_attention is RAD.wrapper, "install() must rebind the backend's name")
    check(RAD.install() is True and TA.unified_attention is RAD.wrapper, "install() is not idempotent")

    def routed(kw=None, *args, **more):
        """'tuned' / 'orig' for one call through the installed wrapper."""
        n_t, n_o = len(tuned), len(orig_calls)
        TA.unified_attention(*args, **(kw if kw is not None else more))
        if len(tuned) > n_t:
            return "tuned"
        return "orig" if len(orig_calls) > n_o else "?"

    # 1. the drafter's call is taken, with the parameters the launch needs
    check(routed(call_kw()) == "tuned" and tuned[-1] == (2048, 8, 1, True), f"drafter call: {tuned[-1:]}")
    check(routed(call_kw(causal=False)) == "tuned" and tuned[-1] == (2048, 8, 1, False), "the drafter's real call is non-causal")
    check(routed(call_kw(nseq=8)) == "tuned" and tuned[-1] == (2048, 8, 8, True), "8 sequences")
    check(routed(call_kw(q_len=2)) == "tuned" and routed(call_kw(q_len=16)) == "tuned", "q_len 2 and 16")
    check(routed(call_kw(window=(4095, 0))) == "tuned" and tuned[-1][0] == 4096, "a wider window (swa_window_size)")

    # 2. everything else is vLLM's
    others = {
        "target-like head 256 / GQA 6": call_kw(head=256, q_heads=24, kv_heads=4),
        "GQA 8": call_kw(q_heads=64, kv_heads=8),
        "q_len 1 (decode)": call_kw(q_len=1),
        "q_len 32 (prefill chunk)": call_kw(q_len=32),
        "no sliding window": call_kw(window=(-1, -1)),
        "bf16 KV": call_kw(kv_dtype=BF16),
        "fp16 queries": call_kw(q_dtype=FP16),
        "kv_quant_mode none": call_kw(mode=Mode.NONE),
        "alibi": call_kw(alibi_slopes=T("f32", 32)),
        "sinks": call_kw(sinks=T("bf16", 32)),
        "softcap": call_kw(softcap=30.0),
        "fp8 query (q_descale)": call_kw(q_descale=T("f32", 1)),
        "output_scale": call_kw(output_scale=T("f32", 1)),
        "chunked": call_kw(chunk_lookback=2),
        "per-seq causal tensor": call_kw(causal=T("bool", 1)),
    }
    for name, kw in others.items():
        check(routed(kw) == "orig", f"{name} must stay on vLLM's launch")
    check(routed(None, *(1, 2)) == "orig", "positional call must stay on vLLM's launch")
    kw = call_kw()
    del kw["window_size"]
    check(routed(kw) == "orig", "a call without window_size must stay on vLLM's launch")

    # 3. split rule
    for nseq, want in ((1, 64), (2, 32), (3, 32), (4, 16), (5, 16), (8, 16)):
        check(RAD.segments_for(nseq) == want, f"segments_for({nseq}) = {RAD.segments_for(nseq)}, want {want}")

    # 4. an exception falls back once and stays there; the fallback returns vLLM's result
    RAD, TA, orig_calls = load()

    def boom(*a, **k):
        raise RuntimeError("boom")
    RAD._run = boom
    RAD.install()
    check(TA.unified_attention(**call_kw()) == "orig" and RAD._state["failed"], "an exception must fall back to vLLM")
    n = len(orig_calls)
    RAD._run = lambda *a: (_ for _ in ()).throw(AssertionError("must not be tried again"))
    check(TA.unified_attention(**call_kw()) == "orig" and len(orig_calls) == n + 1, "after a failure: vLLM only")

    # 5. knobs
    RAD, TA, orig_calls = load(TUNE="0")
    orig = TA.unified_attention
    check(RAD.install() is True and TA.unified_attention is orig, "TUNE=0 must not rebind")
    RAD, TA, orig_calls = load(SEGMENTS="32")
    check(RAD.segments_for(1) == 32 and RAD.segments_for(8) == 32, "SEGMENTS=32 must force the split count")
    RAD.ENABLED = False
    tuned = []
    RAD._run = lambda kw, window, max_q, nseq, causal: tuned.append(1)
    RAD.install()
    RAD.ENABLED = False
    TA.unified_attention(**call_kw())
    check(not tuned, "ENABLED=False at call time must go to vLLM")

    # 6. a vLLM / platform this was not written for is left alone
    RAD, TA, orig_calls = load(drop_arg="IS_3D")
    orig = TA.unified_attention
    check(RAD.install() is True and TA.unified_attention is orig, "a kernel without IS_3D must not be hooked")
    RAD, TA, orig_calls = load(gfx12=False)
    orig = TA.unified_attention
    check(RAD.install() is True and TA.unified_attention is orig, "not gfx12x: not hooked")

    if failures:
        print(f"FAIL: {len(failures)} finding(s)")
        sys.exit(1)
    print("radiance_attn_drafter check OK")


if __name__ == "__main__":
    main()
