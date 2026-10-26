#!/usr/bin/env python3
"""Runs the PATCHED V2 rejection kernel (sly/patch_cactus.py) in the Triton interpreter on the CPU and
checks the length fade (issue #164): delta_eff = delta * clamp((END - out) / (END - START), 0, 1), with
out = position - prompt length of the request.

Not collected by pytest (no test_ prefix); sly/tests/test_cactus_cpu.py starts it as a subprocess with
TRITON_INTERPRET=1 when `triton` and a pinned vLLM source file are available.

    TRITON_INTERPRET=1 python3 sly/tests/cactus_fade_kernel_check.py <stock.py> <patched.py>

Setup: every drafted token is the target's top token with q = 0.6 (point-mass draft). Cactus 0.5 lifts that
to h = 1 and accepts every draft; the stock rule accepts with chance 0.6. One slot per output range:
  1. fade 8..16: out < 8 -> every draft accepted; out >= 16 -> equals stock; in between -> equals a
     float64 reference of the fade formula.
  2. FADE_END=0 -> every draft accepted (1.8.1 behaviour); no prompt lengths passed -> the same.
  3. RADIANCE_CACTUS_DELTA=0 -> equals stock.
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
         "RADIANCE_CACTUS_CAP_R", "RADIANCE_CACTUS_CTX_FADE_START", "RADIANCE_CACTUS_CTX_FADE_END", "RADIANCE_NSD")
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


def reference(start, end, delta=0.5, cap_r=0.0, ctx_start=0, ctx_end=0):
    """float64: accepted drafts per slot under the fades (point-mass draft: accept iff h > u)."""
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
            if ctx_end > 0:  # context fade over the absolute position
                fade *= min(max((ctx_end - (POS0 + i)) / max(ctx_end - ctx_start, 1), 0.0), 1.0)
            h = min(Q + math.sqrt(2 * delta * fade * Q * (1 - Q)), 1.0)
            h = min(h, 1.0 - cap_r * (1.0 - Q))
            if not h > float(u[s, i]):
                break
            acc += 1
        out.append(acc)
    return out


def main(stock_path, patched_path):
    stock = run(load(stock_path, "rsu_stock"), prompt_len=False)
    assert any(a < K for a in stock), "stock accepted everything: setup cannot separate"
    full = [K] * SLOTS

    # cap off: the fade alone must stay bit-comparable with the reference that has no cap
    fade = run(load(patched_path, "rsu_fade", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_START=8,
                    RADIANCE_CACTUS_FADE_END=16, RADIANCE_CACTUS_CAP_R=0))
    for s, o in enumerate(OUT0):
        if o + K - 1 < 8:
            assert fade[s] == K, f"slot {s} (out {o}) before the fade must accept all: {fade}"
        if o >= 16:
            assert fade[s] == stock[s], f"slot {s} (out {o}) after the fade must equal stock: {fade} vs {stock}"
    want = reference(8, 16)
    assert fade == want, f"fade differs from the float64 reference: {fade} vs {want}"
    print(f"fade 8..16, cap off: {fade} (stock {stock}) == reference - OK")

    # context fade over the absolute position (agent turns: long prompt, short output). Positions are
    # POS0..POS0+K-1 in every slot, so 99..102 gives delta factors 2/3, 1/3 and 0.
    no_out = dict(RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_END=0, RADIANCE_CACTUS_CAP_R=0)
    ctx = run(load(patched_path, "rsu_ctx", RADIANCE_CACTUS_CTX_FADE_START=99, RADIANCE_CACTUS_CTX_FADE_END=102,
                   **no_out))
    want_ctx = reference(0, 0, ctx_start=99, ctx_end=102)
    assert ctx == want_ctx, f"context fade differs from the reference: {ctx} vs {want_ctx}"
    assert ctx != full, f"context fade did not bite: {ctx}"
    past = run(load(patched_path, "rsu_ctxpast", RADIANCE_CACTUS_CTX_FADE_START=50, RADIANCE_CACTUS_CTX_FADE_END=90,
                    **no_out))
    assert past == stock, f"past the context fade must equal stock: {past} vs {stock}"
    before = run(load(patched_path, "rsu_ctxbefore", RADIANCE_CACTUS_CTX_FADE_START=200,
                      RADIANCE_CACTUS_CTX_FADE_END=300, **no_out))
    assert before == full, f"before the context fade every draft must be accepted: {before}"
    # both fades multiply (output fade 8..16 and context fade 99..102)
    both = run(load(patched_path, "rsu_both", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_START=8,
                    RADIANCE_CACTUS_FADE_END=16, RADIANCE_CACTUS_CAP_R=0, RADIANCE_CACTUS_CTX_FADE_START=99,
                    RADIANCE_CACTUS_CTX_FADE_END=102))
    want_both = reference(8, 16, ctx_start=99, ctx_end=102)
    assert both == want_both, f"output x context fade differs from the reference: {both} vs {want_both}"
    print(f"context fade 99..102: {ctx}, past {past}, before {before}, both {both} == reference - OK")

    # cap on: q = 0.6, so h is limited to 1 - r*0.4 (0.8 at the default r = 0.5), i.e. only 80 % of the
    # drafts are accepted; the value must equal the float64 reference of the combined rule.
    cap = run(load(patched_path, "rsu_cap", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_END=0,
                   RADIANCE_CACTUS_CAP_R=0.5))
    want_cap = reference(0, 0, cap_r=0.5)
    assert cap == want_cap, f"cap differs from the reference: {cap} vs {want_cap}"
    assert any(0 < a < K for a in cap), f"cap did not bite: {cap}"
    print(f"cap r=0.5: {cap} == reference (0 < accepted < {K}) - OK")
    r1 = run(load(patched_path, "rsu_cap1", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_END=0,
                  RADIANCE_CACTUS_CAP_R=1.0))
    assert r1 == stock, f"cap r=1 must equal stock: {r1} vs {stock}"
    print("cap r=1 == stock - OK")

    try:
        load(patched_path, "rsu_badcap", RADIANCE_CACTUS_CAP_R=1.5)
    except RuntimeError as e:
        print(f"refused RADIANCE_CACTUS_CAP_R=1.5: {e}")
    else:
        raise AssertionError("RADIANCE_CACTUS_CAP_R outside [0, 1] must not load")

    assert run(load(patched_path, "rsu_off", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_FADE_END=0,
                    RADIANCE_CACTUS_CAP_R=0)) == full
    # no prompt lengths -> no fade; cap must be off for the "everything accepted" check
    assert run(load(patched_path, "rsu_nopl", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_CAP_R=0),
               prompt_len=False) == full
    print("fade off / no prompt lengths: every draft accepted - OK")

    assert run(load(patched_path, "rsu_zero", RADIANCE_CACTUS_DELTA=0)) == stock, "delta 0 must equal stock"
    print("delta 0 == stock - OK")
    print("cactus fade kernel check OK")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
