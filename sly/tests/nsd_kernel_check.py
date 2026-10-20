#!/usr/bin/env python3
"""Runs the PATCHED V2 rejection kernel (sly/patch_nsd.py) in the Triton interpreter on the CPU.

Not collected by pytest (no test_ prefix); sly/tests/test_nsd_cpu.py starts it as a subprocess with
TRITON_INTERPRET=1 when `triton` and a pinned vLLM source file are available.

    TRITON_INTERPRET=1 python3 sly/tests/nsd_kernel_check.py <stock.py> <patched.py>

vLLM itself is not imported: the four modules the kernel file imports are stubbed. Checks:
  1. NSD off: output of the patched file == output of the stock file, bit for bit, and the
     nucleus kernel is never launched.
  2. NSD on: sampled tokens / accepted counts == a float64 re-implementation of the paper's rule
     (accept if u-test passes OR mass of strictly-higher target logits <= theta), over temperature,
     top-k masking, point-mass drafts (incl. a draft outside top-k and -1 placeholders), a vocabulary
     of several 8192-blocks, and greedy rows (must equal stock).
  3. NSD on + the Cactus knob at its default (0.5) or set, and theta outside [0, 1) -> the module
     refuses to import (the two acceptance rules are mutually exclusive).
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
                        logits_cache_col_ptr, vocab_size, IS_DRAFTING: tl.constexpr,
                        APPLY_TEMPERATURE: tl.constexpr, USE_FP64: tl.constexpr):
    # Stand-in: plain argmax. The resample path is the same in the stock and the patched file; only the
    # accepted prefix is compared (the bonus / resampled token is not what NSD changes).
    return tl.max(tl.where(mask, logits, float("-inf")), axis=0, return_indices=True)


_stub("vllm.triton_utils", tl=tl, tldevice=tl.math, triton=triton)
_stub("vllm.v1.worker.gpu.sample.gumbel", gumbel_block_argmax=gumbel_block_argmax, tl_rand32=tl_rand32)
_stub("vllm.v1.worker.gpu.sample.watermark", philox_gumbel_block_argmax=None)


@triton.jit
def _u_kernel(u_ptr, seed_ptr, pos_ptr):
    i = tl.program_id(0)
    tl.store(u_ptr + i, tl_rand32(tl.load(seed_ptr + i), tl.load(pos_ptr + i), False))


def load(path, name, **env):
    old = {k: os.environ.get(k) for k in ("RADIANCE_NSD", "RADIANCE_NSD_THETA", "RADIANCE_CACTUS_DELTA")}
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


class _Boom:
    def __getitem__(self, _):
        raise AssertionError("nucleus-mass kernel launched with RADIANCE_NSD off")


def make_batch(n, k, vocab, gen):
    """n requests x (k drafts + bonus). Returns processed target logits and everything the kernel takes."""
    temps = torch.tensor([0.0, 0.7, 1.0, 1.5], dtype=torch.float32)[torch.arange(n) % 4]   # row 0 mod 4 greedy
    topk = torch.tensor([0, 5, 40, 0])[torch.arange(n) % 4]                              # 0 = no top-k
    raw = torch.randn(n, k + 1, vocab, generator=gen) * 2.5
    logits = raw / torch.where(temps == 0, torch.ones_like(temps), temps).view(n, 1, 1)
    for r in range(n):                                         # top-k: the sampler masks the rest to -inf
        if int(topk[r]) > 0:
            thr = logits[r].topk(int(topk[r]), dim=-1).values[..., -1:]
            logits[r] = torch.where(logits[r] >= thr, logits[r], torch.full_like(logits[r], -float("inf")))
    order = logits.argsort(-1, descending=True)
    draft = torch.empty(n, k, dtype=torch.int64)
    for r in range(n):
        for s in range(k):
            c = int(torch.randint(0, 6, (1,), generator=gen))
            if c <= 3:
                draft[r, s] = order[r, s, c * 3]                # top token, rank 3, 6, 9: in and around the nucleus
            elif c == 4:
                draft[r, s] = order[r, s, -1]                   # last: -inf under top-k, tiny otherwise
            else:
                draft[r, s] = -1 if (s > 0 and r % 3 == 0) else order[r, s, 1]
    return temps, logits, draft


def run(mod, temps, logits, draft, seed, k):
    n, kp1, vocab = logits.shape
    tokens = torch.full((n, kp1), 7, dtype=torch.int64)
    tokens[:, 1:] = draft                                       # [last sampled, d1..dk] per request
    cu = (torch.arange(n + 1, dtype=torch.int32) * kp1)
    pos = (torch.arange(kp1, dtype=torch.int64) + 100).repeat(n)
    idx = torch.arange(n, dtype=torch.int32)
    eidx = idx.repeat_interleave(kp1)
    elp = torch.arange(kp1, dtype=torch.int32).repeat(n)
    out, num = mod.rejection_sample(logits.reshape(n * kp1, vocab), None, tokens.reshape(-1), cu, pos, idx, eidx,
                                    elp, temps, seed, k)
    valid = torch.arange(kp1) < num.view(n, 1)               # `sampled` is new_empty: only the first num are defined
    return torch.where(valid, out, torch.full_like(out, -2)), num


def reference(logits, temps, draft, seed, k, theta, nsd):
    """float64 re-implementation of the Leviathan branch with a point-mass draft, plus the NSD OR."""
    n, kp1, vocab = logits.shape
    pos = (torch.arange(kp1, dtype=torch.int64) + 100).repeat(n)
    u = torch.empty(n * kp1, dtype=torch.float32)
    _u_kernel[(n * kp1,)](u, seed.repeat_interleave(kp1), pos)
    u = u.view(n, kp1).double()
    acc = torch.zeros(n, dtype=torch.int64)
    border = torch.zeros(n, dtype=torch.bool)
    for r in range(n):
        if float(temps[r]) == 0.0:
            continue                                            # greedy: argmax test, compared against stock
        for s in range(k):
            t = int(draft[r, s])
            if t < 0:
                break
            row = logits[r, s].double()
            logp = row - torch.logsumexp(row, 0)
            std = bool(logp[t] > math.log(float(u[r, s])))
            above = float(torch.exp(logp)[row > row[t]].sum()) if row[t] > -float("inf") else 2.0
            border[r] |= abs(above - theta) < 2e-4 or abs(float(logp[t]) - math.log(float(u[r, s]))) < 2e-4
            ok = std or (nsd and above <= theta)
            if not ok:
                break
            acc[r] += 1
    return acc, border


def main(stock_path, patched_path):
    gen = torch.Generator().manual_seed(11)
    n, k, vocab = 24, 3, 20000                                  # 20000 -> 3 vocab blocks of 8192
    temps, logits, draft = make_batch(n, k, vocab, gen)
    seed = torch.randint(1, 2 ** 62, (n,), generator=gen, dtype=torch.int64)

    stock = load(stock_path, "rsu_stock")
    off = load(patched_path, "rsu_off")
    assert off.NSD_ENABLED is False
    off._nsd_nucleus_mass_kernel = _Boom()
    s_out, s_num = run(stock, temps, logits, draft, seed, k)
    o_out, o_num = run(off, temps, logits, draft, seed, k)
    assert torch.equal(s_num, o_num) and torch.equal(s_out, o_out), "NSD off differs from stock"
    print(f"off == stock: {n} requests, accepted {s_num.sum().item() - n} of {(draft >= 0).sum().item()} drafts")

    for theta in (0.0, 0.5, 0.95, 0.99):
        # RADIANCE_CACTUS_DELTA=0: Cactus is ON by default (0.5) and the two rules exclude each
        # other, so both knobs are set explicitly here (sly/tests/test_spec_knobs_cpu.py).
        on = load(patched_path, f"rsu_on_{theta}", RADIANCE_NSD=1, RADIANCE_NSD_THETA=theta,
                  RADIANCE_CACTUS_DELTA=0)
        assert on.NSD_ENABLED and on.NSD_THETA == theta
        out, num = run(on, temps, logits, draft, seed, k)
        want, border = reference(logits, temps, draft, seed, k, theta, True)
        greedy = temps == 0
        assert torch.equal(num[greedy], s_num[greedy]) and torch.equal(out[greedy], s_out[greedy]), "greedy changed"
        got = (num - 1).long()
        bad = (got != want) & ~greedy & ~border
        assert not bool(bad.any()), f"theta={theta}: accepted counts differ at rows {bad.nonzero().flatten().tolist()}"
        ref_off, _ = reference(logits, temps, draft, seed, k, theta, False)
        gain = int(((num - 1).long() - s_num.long() + 1)[~greedy].sum())
        assert gain >= 0, "NSD must never accept fewer tokens than the standard rule"
        # the standard-only reference must reproduce the stock kernel (validates the reference itself)
        bad0 = (ref_off != (s_num - 1).long()) & ~greedy & ~border
        assert not bool(bad0.any()), "reference disagrees with the stock kernel"
        # accepted draft tokens are written, the first rejected position gets a resampled token != the draft
        for r in range(n):
            a = int(got[r])
            assert torch.equal(out[r, :a], draft[r, :a])
        print(f"theta={theta}: on == reference ({int((~greedy).sum())} sampled rows, {int(border.sum())} borderline "
              f"skipped), extra accepted tokens vs stock: {gain}")

    for env in ({"RADIANCE_NSD": 1},                                  # Cactus left at its 0.5 default
                {"RADIANCE_NSD": 1, "RADIANCE_CACTUS_DELTA": 0.5},
                {"RADIANCE_NSD": 1, "RADIANCE_CACTUS_DELTA": 0, "RADIANCE_NSD_THETA": 1.0},
                {"RADIANCE_NSD": 1, "RADIANCE_CACTUS_DELTA": 0, "RADIANCE_NSD_THETA": -0.1}):
        try:
            load(patched_path, "rsu_bad", **env)
        except RuntimeError as e:
            print(f"refused {env}: {e}")
        else:
            raise AssertionError(f"{env} must not import")
    print("nsd kernel check OK")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
