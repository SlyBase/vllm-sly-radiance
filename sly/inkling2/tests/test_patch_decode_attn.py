"""sly/patch_inkling2_decode_attn.py + sly/patch_inkling2_prefill_attn.py: Inkling 2 attention on gfx12.

Checks:
  - text & two-pass idempotency of both patches alone and stacked in Dockerfile order;
  - numerics in the Triton CPU interpreter against a per-key fp32 reference (VEC_KV on and off, global & local SWA);
  - offline gfx1201 ISA compile: 0 VGPR spills and 16-byte `global_load_b128` loads (no `global_load_u8`), both before
    and after stacking `patch_inkling2_prefill_attn.py`.
"""
import ast
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[3]
OPS = "vllm/models/inkling2/nvidia/ops"


@pytest.fixture()
def tree(tmp_path):
    src = os.environ.get("INKLING_VLLM_SRC") or "/opt/data/analysis/inkling2-mini/vsrc"
    if not Path(src, "models/inkling2").exists():
        pytest.skip("INKLING_VLLM_SRC not set")
    shutil.copytree(src, tmp_path / "vllm", ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def run(target, name):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(target), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(REPO / "sly" / name)], capture_output=True, text=True, env=env)


def patched(tree):
    if (tree / "vllm/platforms/rocm.py").exists():
        assert run(tree, "patch_inkling2_gfx12.py").returncode == 0
    r = run(tree, "patch_inkling2_decode_attn.py")
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def prefill_patched(tree):
    patched(tree)
    r = run(tree, "patch_inkling2_prefill_attn.py")
    assert r.returncode == 0, r.stdout + r.stderr
    return r


def test_hunks_land_and_parse(tree):
    patched(tree)
    plan = (tree / OPS / "attention.py").read_text()
    kern = (tree / OPS / "attention_sm90.py").read_text()
    ast.parse(plan)
    ast.parse(kern)
    assert "_ROCM_CTAS_PER_SM * num_sms // work" in plan
    assert "_RAD_LOC_BN) + 1)" in plan
    assert "BULK_ON: tl.constexpr" in kern
    assert "for i in range(num_bulk, num_iters):" in kern
    assert "def _load_fp8_kv_block(" in kern
    assert "[radiance.inkling2] decode_attn:" in kern


def test_decode_launch_is_decode_only_and_gated_by_the_splits_knob(tree):
    patched(tree)
    kern = (tree / OPS / "attention_sm90.py").read_text()
    assert "torch.version.hip and mode == MODE_PARTIAL and block_m <= 32 and _RAD_DEC_SPLITS" in kern
    assert "BULK=_RAD_DEC_SPLITS and _BULK and mode != MODE_OUT_LSE" in kern
    assert "VEC_KV=vec" in kern
    assert "_RAD_VEC_KV = _RAD_DEC_SPLITS and" in kern
    plan = (tree / OPS / "attention.py").read_text()
    assert "if torch.version.hip and _RAD_DEC_SPLITS else 0" in plan
    assert "if torch.version.hip and _RAD_DEC_SPLITS:" in plan


def test_second_run_is_a_noop(tree):
    first = patched(tree)
    assert "OK " in first.stdout and "FAIL" not in first.stdout
    before = {p: p.read_text() for p in (tree / OPS).glob("attention*.py")}
    r = run(tree, "patch_inkling2_decode_attn.py")
    assert r.returncode == 0 and "OK " not in r.stdout.replace("NOOP", "") and "NOOP" in r.stdout
    assert before == {p: p.read_text() for p in (tree / OPS).glob("attention*.py")}


def test_absent_model_is_a_noop(tmp_path):
    r = run(tmp_path, "patch_inkling2_decode_attn.py")
    assert r.returncode == 0 and "NOOP" in r.stdout


def test_prefill_hunks_land_parse_and_both_patches_stay_noop_on_pass2(tree):
    prefill_patched(tree)
    kern = (tree / OPS / "attention_sm90.py").read_text()
    ast.parse(kern)
    assert "RADIANCE_INKLING2_PREFILL_BULK" in kern
    assert "RADIANCE_INKLING2_PREFILL_BLOCK_N" in kern
    assert "RADIANCE_INKLING2_PREFILL_LAZY" in kern and "            if LAZY:" in kern
    assert "BULK=_BULK and (_PREFILL_BULK if mode == MODE_OUT_LSE else _RAD_DEC_SPLITS)" in kern
    before = {p: p.read_text() for p in (tree / OPS).glob("attention*.py")}
    # Both decode and prefill patches must be pure NOOPs on pass 2 (exact ci/patch_dryrun.sh contract!)
    for name in ("patch_inkling2_decode_attn.py", "patch_inkling2_prefill_attn.py"):
        r = run(tree, name)
        assert r.returncode == 0 and "OK " not in r.stdout.replace("NOOP", ""), f"{name}: {r.stdout}"
    assert before == {p: p.read_text() for p in (tree / OPS).glob("attention*.py")}


def test_prefill_absent_model_is_a_noop(tmp_path):
    r = run(tmp_path, "patch_inkling2_prefill_attn.py")
    assert r.returncode == 0 and "NOOP" in r.stdout


def load_kernels(path):
    import torch  # noqa: F401
    import triton
    import triton.language as tl

    shim = types.ModuleType("vllm.triton_utils")
    shim.tl, shim.triton = tl, triton
    with mock.patch.dict(sys.modules, {"vllm": types.ModuleType("vllm"), "vllm.triton_utils": shim}):
        spec = importlib.util.spec_from_file_location("attention_sm90_patched", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


H, HKV, D, PAGE = 6, 2, 128, 128
G = H // HKV


def reference(q, kc, vc, rel, bt, past, window, E):
    import torch

    T = q.shape[0]
    K = kc[bt.long()].reshape(-1, HKV, D).float()
    V = vc[bt.long()].reshape(-1, HKV, D).float()
    out = torch.zeros(T, H, D)
    for t in range(T):
        pos = past + t
        lo = max(0, pos - window + 1) if window else 0
        keys = torch.arange(lo, pos + 1)
        rel_base = (past + min((t // 128 + 1) * 128, T) + 127) // 128 * 128 - (E + 256)
        for h in range(H):
            s = (q[t, h].float() @ K[keys, h // G].T) * D**-0.5
            near = (pos - keys) < E
            s[near] += rel[t, h, keys[near] - rel_base].float()
            out[t, h] = torch.softmax(s, 0) @ V[keys, h // G]
    return out


@pytest.mark.parametrize("rel_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("window,E", [(0, 2048), (1024, 1024)])
def test_vector_block_load_matches_the_per_key_reference(tree, monkeypatch, window, E, rel_dtype):
    pytest.importorskip("triton")
    import torch

    prefill_patched(tree)
    S = load_kernels(tree / OPS / "attention_sm90.py")
    monkeypatch.setattr(torch.version, "hip", "6.0", raising=False)
    monkeypatch.setattr(S, "_RAD_DEC_BN", 32)
    monkeypatch.setattr(S, "_RAD_LOC_BN", 32)
    seen = []
    kern = S._rel_attention_kernel

    class Spy:
        def __getitem__(self, grid):
            return lambda *a, **kw: (seen.append(kw), kern[grid](*a, **kw))[1]

    monkeypatch.setattr(S, "_rel_attention_kernel", Spy())
    dt = getattr(torch, rel_dtype)
    tol = 1e-5 if dt == torch.float32 else 2e-3
    torch.manual_seed(0)
    for q_len, past in ((1, 300), (6, 700), (8, 2600) if window == 0 else (8, 2500)):
        T, seq = q_len, past + q_len
        nblk = -(-seq // PAGE)
        kv = (torch.randn(nblk, HKV, PAGE, 2 * D) * 2).to(torch.float8_e4m3fn)
        kc, vc = kv.transpose(1, 2).split(D, dim=-1)
        bt = torch.randperm(nblk, dtype=torch.int32)
        q = torch.randn(T, H, D)
        rel = (torch.randn(T + 128, H, E + 256) * 0.5).to(dt)
        ref = reference(q, kc, vc, rel, bt, past, window, E)
        for vec in (True, False):
            monkeypatch.setattr(S, "_RAD_VEC_KV", vec)
            splits = 3
            out = torch.zeros(splits, T, H, D)
            lse = torch.zeros(splits, T, H)
            S.rel_attention(q, kc, vc, out, lse, rel, block_table=bt[None], cache_seqlens=torch.tensor([seq], dtype=torch.int32),
                            cu_seqlens_q=torch.tensor([0, T], dtype=torch.int32), max_query_len=T, softmax_scale=D**-0.5,
                            window=window, num_splits=splits, mode=S.MODE_PARTIAL)
            kw = seen[-1]
            assert kw["VEC_KV"] is vec and kw["BLOCK_N"] == 32 and kw["LAZY"] is False, kw
            w = torch.exp2(lse - lse.max(0).values)
            got = (w[..., None] * out).sum(0) / w.sum(0)[..., None]
            err = (got - ref).abs().max().item()
            assert err <= tol, f"window={window} q_len={q_len} past={past} vec={vec}: max_abs={err}"


POINTERS = {"Q": "*bf16", "K": "*fp8e4nv", "V": "*fp8e4nv", "Rel": "*bf16", "Out": "*fp32", "Lse": "*fp32", "Gate": "*bf16",
            "Sink": "*bf16", "BlockTable": "*i32", "SeqLens": "*i32", "CuSeqlensQ": "*i32", "softmax_scale": "fp32"}


def gfx1201_asm(path, window, num_warps, block_n, vec):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    kern = load_kernels(path)._rel_attention_kernel
    consts = dict(H=24, G=3, D=128, E=1024, PAGE=128, WINDOW=window, DRAFT=False, MODE=2, BLOCK_M=32, BLOCK_N=block_n,
                  BULK=True, VEC_KV=vec)
    if "LAZY" in kern.arg_names:
        consts["LAZY"] = False
    sig = {n: POINTERS.get(n, "i32") for n in kern.arg_names if n not in consts}
    code = triton.compile(ASTSource(kern, sig, consts), target=GPUTarget("hip", "gfx1201", 32),
                          options={"num_warps": num_warps, "num_stages": 1})
    return code.asm["amdgcn"]


@pytest.mark.parametrize("window,num_warps", [(0, 8), (1024, 4)])
def test_gfx1201_isa_has_16_byte_kv_loads_and_no_spills(tree, window, num_warps):
    pytest.importorskip("triton")
    prefill_patched(tree)
    env = dict(os.environ, TRITON_INTERPRET="0")
    r = subprocess.run([sys.executable, __file__, str(tree / OPS / "attention_sm90.py"), str(window), str(num_warps)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr[-3000:]
    asm = r.stdout
    assert int(re.search(r"\.vgpr_spill_count:\s*(\d+)", asm).group(1)) == 0
    assert len(re.findall(r"global_load_b128", asm)) >= 4
    assert "global_load_u8" not in asm


if __name__ == "__main__":
    print(gfx1201_asm(Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), 32, True))
