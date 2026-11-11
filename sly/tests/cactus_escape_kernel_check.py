#!/usr/bin/env python3
"""Runs the PATCHED V2 rejection kernel (sly/patch_cactus.py) in the Triton interpreter on the CPU and
checks the escape mixture (issue #164): with probability CACTUS_ESCAPE the exact stock test runs
instead of the Cactus test, so acceptance = (1-eps)*min(1,h/p) + eps*min(1,q/p).

Not collected by pytest (no test_ prefix); sly/tests/test_cactus_escape_cpu.py starts it as a
subprocess with TRITON_INTERPRET=1 when `triton` and a pinned vLLM source file are available.

    TRITON_INTERPRET=1 python3 sly/tests/cactus_escape_kernel_check.py <stock.py> <patched.py>

Setup: every drafted token is the target's top token with q = 0.6 (point-mass draft, p = 1). Cactus
0.5 lifts that to h = 1 and accepts every draft; the stock rule accepts with chance 0.6. The mixture
accepts with chance (1-eps)*1 + eps*0.6 (cap off), i.e. it provably sits between the two. Checks:
  1. eps=0 -> plain Cactus (cap off: everything accepted; cap r=0.5: the capped reference).
  2. eps=1 -> bit-exact stock (same kernel, same uniforms).
  3. eps=0.5 / 0.25 -> float64 reference of the convex acceptance, with and without the cap.
  4. delta=0 with eps>0 -> still bit-exact stock (the mixture must not break the delta=0 guard).
  5. fade 8..16 with eps=0.5 -> per-position reference (mixture of the faded h and q).
  6. RADIANCE_CACTUS_ESCAPE outside [0, 1] -> refused at import.
"""
import importlib.util
import math
import os
import sys
import types

os.environ["TRITON_INTERPRET"] = "1"
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402


def _stub(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


for _n in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu", "vllm.v1.worker.gpu.sample",
           "vllm.v1.worker.gpu.spec_decode"):
    _stub(_n).__path__ = []


@triton.jit
def tl_rand32(seed, offset, includes_zero: tl.constexpr):
    u = tl.rand(seed, offset)
    if not includes_zero:
        u = tl.maximum(u, 4.6566127342e-10)
    return u


@triton.jit
def gumbel_block_argmax(logits, block, mask, resample_token_idx, expanded_idx_mapping_ptr, temp_ptr, seed_ptr,
                        pos_ptr, logits_cache_ptr, logits_cache_stride_0, logits_cache_stride_1,
                        logits_cache_col_ptr, logits_cache_source_ptr, logits_cache_source_stride, vocab_size,
                        IS_DRAFTING: tl.constexpr, APPLY_TEMPERATURE: tl.constexpr, USE_FP64: tl.constexpr,
                        PER_TOKEN_COL: tl.constexpr = False):
    # signature of vLLM 0.32.0 sample/gumbel.py
    # Stand-in: plain argmax; only the accepted counts are compared.
    return tl.max(tl.where(mask, logits, float("-inf")), axis=0, return_indices=True)


_stub("vllm.triton_utils", tl=tl, tldevice=tl.math, triton=triton)
_stub("vllm.v1.worker.gpu.sample.gumbel", gumbel_block_argmax=gumbel_block_argmax, tl_rand32=tl_rand32)
_stub("vllm.v1.worker.gpu.sample.watermark", philox_gumbel_block_argmax=None)

KNOBS = ("RADIANCE_CACTUS_DELTA", "RADIANCE_CACTUS_FADE_START", "RADIANCE_CACTUS_FADE_END",
         "RADIANCE_CACTUS_CAP_R", "RADIANCE_CACTUS_ESCAPE", "RADIANCE_NSD")
K, VOCAB, POS0 = 3, 9000, 100
Q = 0.6
# slot -> output length at the first verified position (prompt length = POS0 - out)
OUT0 = [0, 4, 9, 11, 13, 20, 30, 40]
SLOTS = len(OUT0)


@triton.jit
def _u_kernel(u_ptr, seed_ptr, pos_ptr):
    i = tl.program_id(0)
    tl.store(u_ptr + i, tl_rand32(tl.load(seed_ptr + i), tl.load(pos_ptr + i), False))


def load(path, name, **env):
    old = {k: os.environ.get(k) for k in KNOBS}
    for k in old:
        os.environ.pop(k, None)
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in old.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def batch():
    n, kp1 = SLOTS, K + 1
    logits = torch.zeros(n * kp1, VOCAB)
    logits[:, 17] = math.log(Q * (VOCAB - 1) / (1 - Q))      # q(top) = 0.6
    tokens = torch.full((n, kp1), 17, dtype=torch.int64)
    cu = torch.arange(n + 1, dtype=torch.int32) * kp1
    pos = (torch.arange(kp1, dtype=torch.int64) + POS0).repeat(n)
    idx = torch.arange(n, dtype=torch.int32)
    seed = torch.arange(SLOTS, dtype=torch.int64) * 7919 + 13
    prompt_len = torch.tensor([POS0 - o for o in OUT0], dtype=torch.int32)
    return (logits, tokens.reshape(-1), cu, pos, idx, idx.repeat_interleave(kp1),
            torch.arange(kp1, dtype=torch.int32).repeat(n), seed, prompt_len)


def run(mod, prompt_len=True):
    logits, tokens, cu, pos, idx, eidx, elp, seed, pl = batch()
    kw = {"cactus_prompt_len": pl} if prompt_len else {}
    _, num = mod.rejection_sample(logits, None, tokens, cu, pos, idx, eidx, elp, torch.ones(SLOTS), seed, K, **kw)
    return [int(x) - 1 for x in num]


def reference(start, end, delta=0.5, cap_r=0.0, eps=0.0):
    """float64: accepted drafts per slot. Point-mass draft (p=1): accept iff u < (1-eps)*min(1,h) + eps*Q."""
    _, _, _, pos, _, _, _, seed, _ = batch()
    u = torch.empty(SLOTS * (K + 1))
    _u_kernel[(SLOTS * (K + 1),)](u, seed.repeat_interleave(K + 1), pos)
    u = u.view(SLOTS, K + 1).double()
    out = []
    for s, o0 in enumerate(OUT0):
        acc = 0
        for i in range(K):
            # end == 0 is the "no fade" switch in the kernel, not a fade to zero
            fade = 1.0 if end == 0 else min(max((end - (o0 + i)) / max(end - start, 1), 0.0), 1.0)
            d = delta * fade
            h = min(max(Q + math.sqrt(2 * d * Q * (1 - Q)), 0.0), 1.0)
            h = min(h, 1.0 - cap_r * (1.0 - Q))
            a = (1.0 - eps) * min(1.0, h) + eps * Q
            if not float(u[s, i]) < a:
                break
            acc += 1
        out.append(acc)
    return out


def main(stock_path, patched_path):
    stock = run(load(stock_path, "rsu_stock"), prompt_len=False)
    assert any(a < K for a in stock), "stock accepted everything: setup cannot separate"
    full = [K] * SLOTS

    # eps=0: unchanged Cactus. Cap off -> h=1 accepts everything; cap r=0.5 -> the capped reference.
    assert run(load(patched_path, "rsu_e0", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_END=0,
                    RADIANCE_CACTUS_CAP_R=0, RADIANCE_CACTUS_ESCAPE=0)) == full
    e0cap = run(load(patched_path, "rsu_e0cap", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_END=0,
                     RADIANCE_CACTUS_CAP_R=0.5, RADIANCE_CACTUS_ESCAPE=0))
    assert e0cap == reference(0, 0, cap_r=0.5), f"eps=0 + cap differs: {e0cap}"
    print(f"eps=0: cap off {full}, cap r=0.5 {e0cap} == reference - OK")

    # eps=1: bit-exact stock (the kernel takes the stock log test, same uniforms).
    e1 = run(load(patched_path, "rsu_e1", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_END=0,
                  RADIANCE_CACTUS_CAP_R=0, RADIANCE_CACTUS_ESCAPE=1))
    assert e1 == stock, f"eps=1 must equal stock: {e1} vs {stock}"
    print("eps=1 == stock - OK")

    # eps=0.5 and 0.25, cap off: a = (1-eps)*1 + eps*0.6 must bite and match the reference.
    for eps in (0.5, 0.25):
        got = run(load(patched_path, f"rsu_e{eps}", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_END=0,
                       RADIANCE_CACTUS_CAP_R=0, RADIANCE_CACTUS_ESCAPE=eps))
        want = reference(0, 0, eps=eps)
        assert got == want, f"eps={eps} differs from the reference: {got} vs {want}"
        if eps == 0.5:
            assert any(0 < a < K for a in got), f"eps=0.5 did not bite: {got}"
        print(f"eps={eps}, cap off: {got} == reference - OK")

    # eps=0.5 with cap r=0.5: a = 0.5*0.8 + 0.5*0.6 = 0.7.
    got = run(load(patched_path, "rsu_e5cap", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_END=0,
                   RADIANCE_CACTUS_CAP_R=0.5, RADIANCE_CACTUS_ESCAPE=0.5))
    want = reference(0, 0, cap_r=0.5, eps=0.5)
    assert got == want, f"eps=0.5 + cap differs: {got} vs {want}"
    print(f"eps=0.5, cap r=0.5: {got} == reference - OK")

    # delta=0 with eps>0: the mixture must not break the bit-exact stock guard.
    d0 = run(load(patched_path, "rsu_d0e5", RADIANCE_CACTUS_DELTA=0, RADIANCE_CACTUS_FADE_END=0,
                  RADIANCE_CACTUS_CAP_R=0, RADIANCE_CACTUS_ESCAPE=0.5))
    assert d0 == stock, f"delta=0 + eps=0.5 must equal stock: {d0} vs {stock}"
    print("delta=0, eps=0.5 == stock - OK")

    # fade 8..16 with eps=0.5: before the fade a=0.8, after it a=0.6 (=stock), between: reference.
    faded = run(load(patched_path, "rsu_fade_e5", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_START=8,
                     RADIANCE_CACTUS_FADE_END=16, RADIANCE_CACTUS_CAP_R=0, RADIANCE_CACTUS_ESCAPE=0.5))
    want = reference(8, 16, eps=0.5)
    assert faded == want, f"fade + eps=0.5 differs: {faded} vs {want}"
    for s, o in enumerate(OUT0):
        if o >= 16:
            assert faded[s] == stock[s], f"slot {s} after the fade must equal stock: {faded} vs {stock}"
    print(f"fade 8..16, eps=0.5: {faded} == reference - OK")

    for bad in (1.5, -0.1):
        try:
            load(patched_path, f"rsu_bad{bad}", RADIANCE_CACTUS_ESCAPE=bad)
        except RuntimeError as e:
            print(f"refused RADIANCE_CACTUS_ESCAPE={bad}: {e}")
        else:
            raise AssertionError(f"RADIANCE_CACTUS_ESCAPE={bad} must not load")

    print("cactus escape kernel check OK")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
