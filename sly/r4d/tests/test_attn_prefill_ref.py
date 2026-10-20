#!/usr/bin/env python3
"""CPU tests for the R4D prefill attention kernel and its A1 option bits (O_CFOLD, O_MAX3).

Two jobs:

  1. A float32 reference of what `r4d_attn_prefill_kernel` computes -- causal, GQA 6, head_dim 256,
     paged fp8-e4m3 KV, permuted block table -- checked against torch's own
     `scaled_dot_product_attention` on the unpacked tensors. This is the oracle the GPU variant is
     judged against (sly/r4d/tests/gpu_attn_prefill_opt2.py), so it is validated independently here.

  2. The patch: apply `sly/r4d/r4d_extras_rx10.patch` to the pinned libr4d sources and check that the
     A1 bits exist, that the launcher reads R4D_ATTN_OPT2, and that the default arm stays the shipped
     OPT mask (bit 22 / 24 clear, bit 23 never touched).

Run:  python3 sly/r4d/tests/test_attn_prefill_ref.py      or   pytest sly/r4d/tests
      The patch test needs the pinned sources: R4D_PIN_SRC, default
      /opt/data/cache/scratch/arch-r4d/pin (pin a3e4833 + this patch). Absent -> that test skips;
      the patch-text checks always run.
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:                                   # the patch checks run anywhere; the reference needs torch
    import torch
    import torch.nn.functional as F
except ImportError:                    # pragma: no cover
    torch = None

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3] if len(_HERE.parents) > 3 else _HERE.parent   # repo root, or wherever it sits
PATCH = Path(os.environ.get("R4D_PATCH", ROOT / "sly" / "r4d" / "r4d_extras_rx10.patch"))
PIN = Path(os.environ.get("R4D_PIN_SRC", "/opt/data/cache/scratch/arch-r4d/pin"))

HEAD_DIM, GQA, BS = 256, 6, 16
SCALE = HEAD_DIM ** -0.5
O_CFOLD, O_MAX3 = 1 << 22, 1 << 24


# --------------------------------------------------------------------------- fp32 reference

def unpack_kv(kv, bt, ctx, kvh, kvheads):
    """[nblk, kvheads, BS, 2*D] fp8 bytes + block table -> [ctx, 2*D] float, block order undone."""
    k = torch.empty(ctx, 2 * HEAD_DIM)
    for kk in range(ctx):
        blk = int(bt[0, kk // BS])
        k[kk] = kv[blk, kvh, kk % BS].float()
    return k[:, :HEAD_DIM], k[:, HEAD_DIM:]


def prefill_reference(q, kv, bt, ctx, kvh, kdesc, vdesc):
    """What the kernel computes, in float32: o[i] = softmax(q_i K^T) V, keys <= ctx - q_len + i.

    q: [M, H, D] bf16 (H = kvheads * GQA); kv/bt/ctx as the launcher takes them.
    """
    M = q.shape[0]
    K, V = unpack_kv(kv, bt, ctx, kvh, kvheads=q.shape[1] // GQA)
    out = torch.empty(M, GQA, HEAD_DIM)
    for h in range(GQA):
        qh = q[:, kvh * GQA + h].float() * (SCALE * kdesc)          # FOLDQ: scale in the query
        s = qh @ K.T                                                 # ... so K stays raw
        limits = torch.arange(M).unsqueeze(1) + (ctx - M)            # causal, offset by ctx_prior
        keys = torch.arange(ctx).unsqueeze(0)
        s = s.masked_fill(keys > limits, -float("inf"))
        p = torch.softmax(s, dim=-1)
        out[:, h] = p @ (V * vdesc)
    return out


def sdpa_reference(q, kv, bt, ctx, kvh, kdesc, vdesc):
    """The same thing by torch.nn.functional.scaled_dot_product_attention on the unpacked tensors."""
    M = q.shape[0]
    K, V = unpack_kv(kv, bt, ctx, kvh, kvheads=q.shape[1] // GQA)
    out = torch.empty(M, GQA, HEAD_DIM)
    limits = torch.arange(M).unsqueeze(1) + (ctx - M)
    mask = torch.arange(ctx).unsqueeze(0) <= limits
    for h in range(GQA):
        o = F.scaled_dot_product_attention(q[:, kvh * GQA + h].float().unsqueeze(0),
                                           K.unsqueeze(0), (V * vdesc).unsqueeze(0),
                                           attn_mask=mask.unsqueeze(0), scale=SCALE * kdesc)
        out[:, h] = o[0]
    return out


def make_case(q_len, ctx_prior, kvheads, seed=0):
    ctx = ctx_prior + q_len
    nblk = (ctx + BS - 1) // BS
    g = torch.Generator().manual_seed(seed)
    q = (torch.randn(q_len, kvheads * GQA, HEAD_DIM, generator=g) * 0.5).to(torch.bfloat16)
    kv = torch.randn(nblk, kvheads, BS, 2 * HEAD_DIM, generator=g).to(torch.float8_e4m3fn)
    bt = torch.randperm(nblk, generator=g).to(torch.int32).view(1, -1)
    return q, kv, bt, ctx


def _need_torch():
    if torch is None:
        raise Skip("torch not installed (the reference tests need it)")


def test_reference_matches_sdpa():
    _need_torch()
    for q_len, ctx_prior, kvheads in [(64, 0, 1), (64, 37, 4), (100, 1000, 4), (513, 0, 1),
                                      (513, 1000, 4)]:
        q, kv, bt, ctx = make_case(q_len, ctx_prior, kvheads)
        for kdesc, vdesc in [(1.0, 1.0), (0.5, 2.0)]:
            ref = prefill_reference(q, kv, bt, ctx, 0, kdesc, vdesc)
            sdpa = sdpa_reference(q, kv, bt, ctx, 0, kdesc, vdesc)
            err = (ref - sdpa).abs().max().item()
            assert err <= 1e-5, f"ref vs sdpa {q_len}/{ctx_prior}/{kvheads} {kdesc}: {err:.3e}"
    # the tolerance has to bite: a 1% scale error must not slip through
    q, kv, bt, ctx = make_case(100, 37, 4)
    bad = prefill_reference(q, kv, bt, ctx, 0, 1.01, 1.0)
    good = sdpa_reference(q, kv, bt, ctx, 0, 1.0, 1.0)
    assert (bad - good).abs().max().item() > 1e-5, "reference is insensitive to a scale error"


def test_reference_is_causal_at_the_offset():
    """The first query row may only see the ctx_prior keys it inherits, not its own tile's tail."""
    _need_torch()
    q, kv, bt, ctx = make_case(64, 37, 1)
    ref = prefill_reference(q, kv, bt, ctx, 0, 1.0, 1.0)
    K, V = unpack_kv(kv, bt, ctx, 0, 1)
    qh = q[:, 0].float() * SCALE
    s = qh[0] @ K.T
    p = torch.softmax(s[:38], dim=-1)                     # row 0 sees keys 0..37 (ctx_prior + itself)
    want = p @ V[:38]
    assert (ref[0, 0] - want).abs().max().item() <= 1e-5


# --------------------------------------------------------------------------- the patch

def _patch_text():
    return PATCH.read_text()


def _patch_sections():
    """[(target file, patch section)] -- one entry per `diff --git`."""
    out = []
    for sec in re.split(r"(?m)^(?=diff --git )", _patch_text()):
        m = re.match(r"diff --git a/(\S+) b/(\S+)", sec)
        if m:
            out.append((m.group(2), sec))
    return out


def test_patch_text_defines_the_a1_bits():
    """The two bits exist, sit on 22/24, and the launcher reads R4D_ATTN_OPT2."""
    t = _patch_text()
    added = [ln[1:] for ln in t.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    body = "\n".join(added)
    assert re.search(r"#define O_CFOLD\s+%d\b" % O_CFOLD, body), "O_CFOLD (bit 22) missing"
    assert re.search(r"#define O_MAX3\s+%d\b" % O_MAX3, body), "O_MAX3 (bit 24) missing"
    assert 'getenv("R4D_ATTN_OPT2")' in body, "launcher does not read R4D_ATTN_OPT2"
    # bit 23 is unexplained in the source; nothing may claim it
    used = {int(m.group(2)) for m in re.finditer(r"#define (O_[A-Z0-9]+)\s+(\d+)", body)}
    assert 8388608 not in used, "bit 23 is taken by something unexplained"
    assert O_CFOLD in used and O_MAX3 in used


class Skip(Exception):
    """not a failure: the input this test needs is not here."""


def _skip(msg):
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg)
    raise Skip(msg)


def _sections(text, tmp):
    """[(file name, section)] for the sections whose target file this pin tree actually has.

    The patch also CREATES files; a partial pin cannot carry those.
    """
    out = []
    for sec in re.split(r"(?m)^(?=diff --git )", text):
        m = re.match(r"diff --git a/(\S+) b/(\S+)", sec)
        if m and (tmp / m.group(2)).exists() and "new file mode" not in sec:
            out.append((m.group(2), sec))
    return out


def _apply(tmp, sections, reverse=False):
    p = tmp / "chk.patch"
    p.write_text("".join(s for _, s in sections))
    cmd = ["git", "apply"] + (["-R"] if reverse else []) + [str(p)]
    return subprocess.run(cmd, cwd=tmp, capture_output=True).returncode == 0


def _walk_back(tmp, todo):
    """Undo the patch the pin tree already carries, so the forward apply below is a real one.

    The pin directory is a cache and holds pin + *some* revision of this patch. Reverse the working
    patch first; failing that, reverse the revision that git has (HEAD, then its parents), which is
    what a stale pin was made from. False = the pin has drifted from every known revision.
    """
    if _apply(tmp, todo, reverse=True):
        return True
    try:
        rel = PATCH.resolve().relative_to(ROOT)
    except ValueError:
        return False
    for rev in ("HEAD", "HEAD~1", "HEAD~2"):
        r = subprocess.run(["git", "show", f"{rev}:{rel}"], cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            continue
        cand = _sections(r.stdout, tmp)
        if cand and _apply(tmp, cand, reverse=True):
            return True
    return False


def test_patch_applies_and_default_is_the_shipped_mask():
    """git apply on the pinned sources; the default arm carries neither A1 bit."""
    if not PIN.is_dir():
        return _skip(f"pinned libr4d sources not found at {PIN} (set R4D_PIN_SRC)")
    if shutil.which("git") is None:
        return _skip("git not on PATH (the pin check applies the patch with git apply)")
    tmp = Path(tempfile.mkdtemp(prefix="r4d-pin-"))
    try:
        for f in list(PIN.glob("*.hip")) + list(PIN.glob("*.h")):
            shutil.copy(f, tmp / f.name)
        todo = _sections(_patch_text(), tmp)
        acted = [n for n, _ in todo]
        assert "r4d_attn_prefill_h256_gqa6.hip" in acted, f"pin has no prefill source in {PIN}"

        if not _walk_back(tmp, todo):
            return _skip(f"the pin cache at {PIN} is neither pin nor pin+patch; "
                         f"refresh it from the pinned libr4d revision")
        base_opt = re.search(r"#define P_OPT\s+(\d+)",
                             (tmp / "r4d_attn_paged_h256_gqa6.hip").read_text()).group(1)
        fwd = tmp / "forward.patch"
        fwd.write_text("".join(s for _, s in todo))
        subprocess.run(["git", "apply", str(fwd)], cwd=tmp, check=True)

        pre = (tmp / "r4d_attn_prefill_h256_gqa6.hip").read_text()
        pag = (tmp / "r4d_attn_paged_h256_gqa6.hip").read_text()

        cfold = int(re.search(r"#define O_CFOLD\s+(\d+)", pre).group(1))
        max3 = int(re.search(r"#define O_MAX3\s+(\d+)", pre).group(1))
        assert (cfold, max3) == (O_CFOLD, O_MAX3)

        # P_OPT is context, not an added line: the base arm must come through untouched
        assert re.search(r"#define P_OPT\s+" + base_opt + r"\b", pag), "the patch edits P_OPT"
        base = int(base_opt)
        assert base & (cfold | max3) == 0, "default OPT mask already carries an A1 bit"
        assert base & 8388608 == 0, "default OPT mask carries the unexplained bit 23"

        # default 0 -> the bare P_OPT arm is still reachable and unchanged
        assert re.search(r"P_LAUNCH2\(P_OPT\)", pag), "default arm missing"
        assert re.search(r"else\s+P_LAUNCH\(OPTX\)", pag), "the OPT2 knob must fall back to OPTX"
        assert 'getenv("R4D_ATTN_OPT2")' in pag
        assert re.search(r"atoi\(e\)\s*:\s*0", pag), "R4D_ATTN_OPT2 must default to 0"

        # both bits are guarded by if constexpr on the kernel side, so the base path cannot see them
        assert pre.count("if constexpr (CFOLD)") + pre.count("if constexpr (MAX3)") > 0
        assert "O_CFOLD | O_MAX3" in pag, "launcher does not OR both A1 bits"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Skip as e:
                print(f"skip {name}: {e}")
            except Exception as e:                                  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print("all tests passed" if not fails else f"{fails} test(s) failed")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
