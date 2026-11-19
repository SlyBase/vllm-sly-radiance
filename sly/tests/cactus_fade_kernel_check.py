#!/usr/bin/env python3
"""Runs the PATCHED V2 rejection kernel (sly/patch_cactus.py) in the Triton interpreter on the CPU and
checks the length fade (issue #164): delta_eff = delta * clamp((END - pos) / (END - START), 0, 1), with
pos = the total sequence length (prompt + output) at the verified token. Every slot starts with output 0, so
a fade over the output length alone would keep the full delta everywhere and fail the reference.

Not collected by pytest (no test_ prefix); sly/tests/test_cactus_cpu.py starts it as a subprocess with
TRITON_INTERPRET=1 when `triton` and a pinned vLLM source file are available.

    TRITON_INTERPRET=1 python3 sly/tests/cactus_fade_kernel_check.py <stock.py> <patched.py>

Setup: every drafted token is the target's top token with q = 0.6 (point-mass draft). Cactus 0.5 lifts that
to h = 1 and accepts every draft; the stock rule accepts with chance 0.6. One slot per output range:
  1. fade 8..16: pos < 8 -> every draft accepted; pos >= 16 -> equals stock; in between -> equals a
     float64 reference of the fade formula. Default fade (4096..16384): prompt 130000 + output 10 ->
     equals stock; prompt 100 + output 100 -> full delta (every draft accepted).
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
                        logits_cache_col_ptr, logits_cache_source_ptr, logits_cache_source_stride,
                        logits_cache_row_mapping_ptr, vocab_size,
                        IS_DRAFTING: tl.constexpr, APPLY_TEMPERATURE: tl.constexpr, USE_FP64: tl.constexpr,
                        PER_TOKEN_COL: tl.constexpr = False):
    # signature of vLLM 0.34.0 sample/gumbel.py
    # Stand-in: plain argmax; only the accepted counts are compared.
    return tl.max(tl.where(mask, logits, float("-inf")), axis=0, return_indices=True)


_stub("vllm.triton_utils", tl=tl, tldevice=tl.math, triton=triton)
@triton.jit
def spec_step_seed(seed, step):
    # vLLM 0.34 sample/gumbel.py: mix the spec step into the seed's high bits
    return seed ^ (tl.cast(step, tl.int64) << 32)


_stub("vllm.v1.worker.gpu.sample.gumbel", gumbel_block_argmax=gumbel_block_argmax, tl_rand32=tl_rand32,
      spec_step_seed=spec_step_seed)
_stub("vllm.v1.worker.gpu.sample.watermark", philox_gumbel_block_argmax=None)
_stub("vllm.v1.worker.gpu.spec_decode.draft_logits", DraftLogitsCache=object, draft_logits_row=None)
_stub("vllm.v1.worker.gpu.spec_decode.fly", compute_fly_entropy=None)

KNOBS = ("RADIANCE_CACTUS_DELTA", "RADIANCE_CACTUS_FADE_START", "RADIANCE_CACTUS_FADE_END",
         "RADIANCE_CACTUS_CAP_R", "RADIANCE_CACTUS_ESCAPE", "RADIANCE_NSD")
K, VOCAB = 3, 9000
Q = 0.6
# slot -> total sequence length at the first verified position; prompt length PL0 (output = SEQ0 - PL0)
SEQ0 = [0, 4, 9, 11, 13, 20, 30, 40, 130010, 200]
PL0 = SEQ0[:8] + [130000, 100]
SLOTS = len(SEQ0)


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
    pos = (torch.arange(kp1, dtype=torch.int64) + torch.tensor(SEQ0).unsqueeze(1)).reshape(-1)
    idx = torch.arange(n, dtype=torch.int32)
    seed = torch.arange(SLOTS, dtype=torch.int64) * 7919 + 13
    prompt_len = torch.tensor(PL0, dtype=torch.int32)
    return (logits, tokens.reshape(-1), cu, pos, idx, idx.repeat_interleave(kp1),
            torch.arange(kp1, dtype=torch.int32).repeat(n), seed, prompt_len)


def run(mod, prompt_len=True):
    logits, tokens, cu, pos, idx, eidx, elp, seed, pl = batch()
    kw = {"cactus_prompt_len": pl} if prompt_len else {}
    _, num = mod.rejection_sample(logits, None, tokens, cu, pos, idx, eidx, elp, torch.ones(SLOTS), seed, K, **kw)
    return [int(x) - 1 for x in num]


def reference(start, end, delta=0.5, cap_r=0.0):
    """float64: accepted drafts per slot under the fade (point-mass draft: accept iff h > u)."""
    _, _, _, pos, _, _, _, seed, _ = batch()
    u = torch.empty(SLOTS * (K + 1))
    _u_kernel[(SLOTS * (K + 1),)](u, seed.repeat_interleave(K + 1), pos)
    u = u.view(SLOTS, K + 1).double()
    out = []
    for s, o0 in enumerate(SEQ0):
        acc = 0
        for i in range(K):
            # end == 0 is the "no fade" switch in the kernel, not a fade to zero
            fade = 1.0 if end == 0 else min(max((end - (o0 + i)) / max(end - start, 1), 0.0), 1.0)
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
    for s, o in enumerate(SEQ0):
        if o + K - 1 < 8:
            assert fade[s] == K, f"slot {s} (pos {o}) before the fade must accept all: {fade}"
        if o >= 16:
            assert fade[s] == stock[s], f"slot {s} (pos {o}) after the fade must equal stock: {fade} vs {stock}"
    want = reference(8, 16)
    assert fade == want, f"fade differs from the float64 reference: {fade} vs {want}"
    print(f"fade 8..16, cap off: {fade} (stock {stock}) == reference - OK")

    # default fade 4096..16384 (agent regime, #164): a 130k prompt is past FADE_END after 10 output
    # tokens -> bit-exact stock; prompt 100 + output 100 is far below FADE_START -> full delta.
    dflt = run(load(patched_path, "rsu_dflt", RADIANCE_CACTUS_DELTA=0.5, RADIANCE_CACTUS_CAP_R=0))
    assert stock[8] < K, f"stock accepted everything in the 130k slot: cannot separate {stock}"
    assert dflt[8] == stock[8], f"prompt 130000 + out 10 must equal stock: {dflt} vs {stock}"
    assert dflt[:8] == [K] * 8 and dflt[9] == K, f"short sequences must keep the full delta: {dflt}"
    print(f"default fade: 130k prompt == stock, short sequences full delta {dflt} - OK")

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
