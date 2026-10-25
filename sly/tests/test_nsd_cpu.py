#!/usr/bin/env python3
"""CPU tests for sly/patch_nsd.py (Nucleus Speculative Decoding, arxiv 2610.07822).

The rule (reference: EIT-NLP/Nucleus-Speculative-Decoding, deepspec/eval/base_evaluator.py
`_target_top_p_accept_mask`): a draft token x is accepted if the standard test passes OR the target
probability mass of the tokens with a STRICTLY higher logit than x is <= theta.

  * the math: the streaming form the kernel uses (per-block partials of exp(logit - lse), no sort) against
    the reference's mask form, over theta edge cases, ties, temperature, top-k and a multi-block vocabulary
  * the point-mass draft the repo uses (q = 1 on the draft token): closed form of the output distribution
  * the patch: applies to the pinned source, twice (idempotent), parses, and the knob-off path is the stock
    kernel (USE_NSD constexpr); if `triton` and the pinned source are present the patched Triton kernel is
    also run in the interpreter (sly/tests/nsd_kernel_check.py: off == stock bit for bit, on == float64
    reference, exclusivity with the Cactus knob)

Run:  python3 sly/tests/test_nsd_cpu.py [path-to-pinned-rejection_sampler_utils.py]   or   pytest sly/tests
      (under pytest the pinned source is .dryrun/src/... from ci/patch_dryrun.sh; absent -> those tests skip)
"""
import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "sly" / "patch_nsd.py"
BLOCK = 8192   # the kernel's VOCAB_BLOCK_SIZE


def processed_logits(raw, temperature=1.0, top_k=0):
    """What reaches rejection_sample: temperature applied, then top-k masked to -inf."""
    x = raw / temperature
    if top_k:
        thr = x.topk(top_k, dim=-1).values[..., -1:]
        x = torch.where(x >= thr, x, torch.full_like(x, -float("inf")))
    return x


def nucleus_reference(logits, tokens, theta):
    """The reference implementation (mask form, one vocabulary-sized temporary): float64."""
    logits = logits.double()
    probs = torch.softmax(logits, -1)
    tok = logits.gather(-1, tokens.unsqueeze(-1))
    higher = logits > tok
    return probs.masked_fill(~higher, 0.0).sum(-1) <= theta


def nucleus_streaming(logits, tokens, theta, block=BLOCK):
    """The kernel's form: block stats -> global lse, then per-block partials of exp(l - lse) where l > l[x]."""
    logits = logits.float()
    n, v = logits.shape
    nb = -(-v // block)
    pad = torch.full((n, nb * block - v), -float("inf"))
    blocks = torch.cat([logits, pad], -1).view(n, nb, block)
    bmax = blocks.max(-1).values
    bsum = torch.where(bmax > -float("inf"), torch.exp(blocks - bmax.unsqueeze(-1)).sum(-1), torch.zeros_like(bmax))
    gmax = bmax.max(-1).values
    lse = gmax + torch.log((bsum * torch.exp(bmax - gmax.unsqueeze(-1))).sum(-1))
    tok = logits.gather(-1, tokens.unsqueeze(-1))
    partial = torch.where(blocks > tok.unsqueeze(-1), torch.exp(blocks - lse.view(n, 1, 1)), torch.zeros(1)).sum(-1)
    mass = partial.sum(-1)
    mass = torch.where(tok.squeeze(-1) > -float("inf"), mass, torch.full_like(mass, 2.0))
    return mass <= theta


def test_streaming_matches_reference():
    g = torch.Generator().manual_seed(3)
    for vocab in (50, 8192, 20000):                         # 1, 1 and 3 vocabulary blocks
        for temp in (0.6, 1.0, 1.7):
            for top_k in (0, 7, 40):
                raw = torch.randn(64, vocab, generator=g) * 2.0
                logits = processed_logits(raw, temp, top_k)
                tokens = torch.randint(0, vocab, (64,), generator=g)
                tokens[:8] = logits.argmax(-1)[:8]          # the top token
                for theta in (0.0, 0.5, 0.9, 0.95, 0.99):
                    ref = nucleus_reference(logits, tokens, theta)
                    got = nucleus_streaming(logits, tokens, theta)
                    # fp32 partial sums vs fp64: only a token whose mass sits within 1e-4 of theta may flip
                    p = torch.softmax(logits.double(), -1)
                    above = p.masked_fill(~(logits.double() > logits.double().gather(-1, tokens.unsqueeze(-1))), 0).sum(-1)
                    near = (above - theta).abs() < 1e-4
                    assert bool(((ref == got) | near).all()), (vocab, temp, top_k, theta)


def test_theta_edges_and_ties():
    logits = torch.tensor([[3.0, 2.0, 2.0, 1.0, 0.0, -float("inf")]])
    p = torch.softmax(logits.double(), -1)[0]
    tokens = torch.arange(6)
    for theta, want in (
        # Never test theta == the mass exactly: fp32 partial sums vs fp64 decide that case by rounding, and the
        # difference (1 ulp of a probability) does not matter for a sampler. Test just above / below instead.
        (0.0, [True, False, False, False, False, False]),                  # only the argmax: nothing strictly above
        (float(p[0]) - 1e-5, [True, False, False, False, False, False]),
        (float(p[0]) + 1e-5, [True, True, True, False, False, False]),     # ties: tokens 1 and 2 both see only token 0 above
        (float(p[0] + p[1] + p[2]) + 1e-5, [True, True, True, True, False, False]),
        (0.999999, [True, True, True, True, True, False]),                 # a token the sampler removed is never accepted
    ):
        got = nucleus_streaming(logits.expand(6, -1), tokens, theta).tolist()
        ref = nucleus_reference(logits.expand(6, -1), tokens, theta).tolist()
        assert got == want == ref, (theta, got, want, ref)


def test_point_mass_draft_distribution():
    """Deterministic draft d (q = 1 on d): standard acceptance prob = p(d); with NSD it is 1 inside the nucleus.
    The residual (p without d, renormalised) is unchanged, so the output distribution is exactly p when the
    rule is off, and deviates from p only by moving mass onto d when d is in the nucleus."""
    logits = processed_logits(torch.randn(1, 400, generator=torch.Generator().manual_seed(9)) * 2.0, 1.0, 40)
    p = torch.softmax(logits.double(), -1)[0]
    for d in (int(p.argmax()), int(p.argsort(descending=True)[3]), int(p.argsort(descending=True)[39]), 5):
        for theta, nsd in ((0.95, False), (0.95, True), (0.5, True)):
            in_nucleus = bool(nucleus_reference(logits, torch.tensor([d]), theta)[0]) and nsd
            acc = 1.0 if in_nucleus else float(p[d])
            resid = p.clone()
            resid[d] = 0.0
            resid = resid / resid.sum() if float(resid.sum()) > 0 else resid
            out = (1.0 - acc) * resid
            out[d] += acc
            assert abs(float(out.sum()) - 1.0) < 1e-12
            if not in_nucleus:
                assert torch.allclose(out, p, atol=1e-12), "knob off / outside the nucleus must reproduce p exactly"
            else:
                assert float(out[d]) >= float(p[d]) - 1e-12
                tv = 0.5 * float((out - p).abs().sum())
                assert tv <= 1.0 - float(p[d]) + 1e-12   # moves at most the residual mass onto d


def _scratch_site(src: Path):
    td = tempfile.TemporaryDirectory()
    site = Path(td.name) / "site-packages"
    pkg = site / "vllm" / "v1" / "worker" / "gpu" / "spec_decode"
    pkg.mkdir(parents=True)
    target = pkg / "rejection_sampler_utils.py"
    shutil.copy(src, target)
    # patch_cactus.py also patches the caller next to it (prompt lengths for the length fade)
    shutil.copy(Path(src).with_name("rejection_sampler.py"), target.with_name("rejection_sampler.py"))
    return td, site, target


def _run_patch(site):
    env = dict(os.environ, PYTHONPATH=str(ROOT), RADIANCE_PATCH_TARGET_DIR=str(site))
    return subprocess.run([sys.executable, str(PATCH)], env=env, capture_output=True, text=True)


_ARGV_SOURCE = None   # set by the __main__ block only: under pytest argv[1] is the test file itself


def _pinned_source():
    if _ARGV_SOURCE is not None:
        return _ARGV_SOURCE
    cand = ROOT / ".dryrun" / "src" / "vllm" / "vllm" / "v1" / "worker" / "gpu" / "spec_decode" / "rejection_sampler_utils.py"
    return cand if cand.exists() else None


def _skip(msg):
    try:
        import pytest
        pytest.skip(msg)
    except ImportError:
        print("SKIP", msg)


def test_patch_structure_and_idempotence():
    src = _pinned_source()
    if src is None:
        return _skip("pinned vLLM source not available (ci/patch_dryrun.sh covers apply + idempotence)")
    td, site, target = _scratch_site(src)
    with td:
        r1 = _run_patch(site)
        assert r1.returncode == 0 and r1.stdout.count("OK ") >= 6, f"pass 1:\n{r1.stdout}\n{r1.stderr}"
        r2 = _run_patch(site)
        assert r2.returncode == 0 and "OK " not in r2.stdout and r2.stdout.count("NOOP") == r1.stdout.count("OK "), r2.stdout
        s = target.read_text()
        ast.parse(s)
        assert s.count('"RADIANCE_NSD"') == 1 and s.count('"RADIANCE_NSD_THETA"') == 1, "env reads must be module-level, once"
        assert "USE_NSD: tl.constexpr" in s and "if USE_NSD:" in s, "knob-off path must be a constexpr branch"
        assert "tl.where(logits > token_logit" in s, "strictly-greater mask missing"
        assert "sort" not in s.split("_nsd_nucleus_mass_kernel")[1].split("_rejection_kernel")[0], "no sort allowed"


def test_patch_order_independent_of_cactus():
    """The Cactus patch (PR #153) hooks the same function; both orders must give the same file."""
    src = _pinned_source()
    cactus = ROOT / "sly" / "patch_cactus.py"
    if src is None or not cactus.exists():
        return _skip("needs the pinned source and sly/patch_cactus.py (PR #153)")
    results = []
    for order in ((PATCH, cactus), (cactus, PATCH)):
        td, site, target = _scratch_site(src)
        with td:
            env = dict(os.environ, PYTHONPATH=str(ROOT), RADIANCE_PATCH_TARGET_DIR=str(site))
            for p in order:
                r = subprocess.run([sys.executable, str(p)], env=env, capture_output=True, text=True)
                assert r.returncode == 0, r.stdout + r.stderr
            ast.parse(target.read_text())
            results.append(sorted(l for l in target.read_text().splitlines() if not l.strip().startswith("#")))
    assert results[0] == results[1]


def test_patched_kernel_in_triton_interpreter():
    src = _pinned_source()
    try:
        import triton  # noqa: F401
    except ImportError:
        return _skip("triton not installed (the interpreter check runs where it is)")
    if src is None:
        return _skip("pinned vLLM source not available")
    td, site, target = _scratch_site(src)
    with td:
        assert _run_patch(site).returncode == 0
        env = dict(os.environ, TRITON_INTERPRET="1")
        r = subprocess.run([sys.executable, str(ROOT / "sly" / "tests" / "nsd_kernel_check.py"), str(src), str(target)],
                           env=env, capture_output=True, text=True, timeout=1200)
        assert r.returncode == 0 and "nsd kernel check OK" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


if __name__ == "__main__":
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        _ARGV_SOURCE = Path(sys.argv[1])
    test_streaming_matches_reference()
    test_theta_edges_and_ties()
    test_point_mass_draft_distribution()
    test_patch_structure_and_idempotence()
    test_patch_order_independent_of_cactus()
    test_patched_kernel_in_triton_interpreter()
    print("nsd CPU tests OK")
