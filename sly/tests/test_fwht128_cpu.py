"""RADIANCE_MXFP4_HAD=128: `had_rot` runs a fast Walsh-Hadamard transform (7 butterfly stages),
not the dense fp32 GEMM `x.float() @ had_matrix`. CPU only.

Convention (radiance_mxfp4.had_matrix): R = H128 diag(s) with s = had_signs(128) (already / sqrt(128)),
row vector x -> x @ R, blockwise over K. The reference here is that dense product.
The Triton kernel runs in the Triton interpreter (subprocess: TRITON_INTERPRET must be set before
triton is imported); the GPU launch is checked in a GPU window (see PR text).
"""
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MXFP4 = os.path.join(HERE, "..", "mxfp4")
sys.path.insert(0, MXFP4)
import radiance_mxfp4 as R  # noqa: E402

MS = (1, 4, 16, 17)
KS = (128, 640, 5120)


def dense(x):
    M, K = x.shape
    return (x.float().view(M, K // 128, 128) @ R.had_matrix(128, x.device)).view(M, K)


def rand(M, K, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(M, K, generator=g).to(dtype)


@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("K", KS)
def test_fwht128_matches_dense_gemm_fp32(M, K):
    x = rand(M, K)
    y = R.fwht128(x)
    assert y.dtype == torch.float32 and y.shape == (M, K)
    torch.testing.assert_close(y, dense(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_fwht128_keeps_input_dtype(dtype):
    x = rand(17, 640, dtype)
    y = R.fwht128(x)
    assert y.dtype == dtype
    # the dense path rounds the fp32 result to x.dtype once; so does this one
    torch.testing.assert_close(y.float(), dense(x).to(dtype).float(), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M", MS)
def test_fwht128_is_orthogonal(M):
    """R R^T = I: the inverse rotation (dense R^T) recovers x, and the norm is kept."""
    x = rand(M, 640, seed=1)
    y = R.fwht128(x)
    back = (y.view(M, 5, 128) @ R.had_matrix(128, x.device).T).view(M, 640)
    torch.testing.assert_close(back, x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y.norm(dim=1), x.norm(dim=1), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("M", MS)
def test_unsigned_transform_applied_twice_is_identity(M):
    """With s = 1/sqrt(128) the transform is the symmetric orthonormal H128: H H = I.
    (With the seed-0 signs it is not an involution: (H D)(H D) != I, so the signed one is
    checked against its inverse above.)"""
    import radiance_fwht
    x = rand(M, 640, seed=2)
    ones = torch.full((128,), 128 ** -0.5)
    torch.testing.assert_close(radiance_fwht.fwht128(radiance_fwht.fwht128(x, ones), ones), x,
                               rtol=1e-5, atol=1e-5)


def test_had_rot_uses_fwht128(monkeypatch):
    calls = []
    real = R.fwht128
    monkeypatch.setattr(R, "fwht128", lambda x: calls.append(tuple(x.shape)) or real(x))
    monkeypatch.setattr(R, "HAD", 128)
    x = rand(17, 640, torch.bfloat16)
    y = R.had_rot(x)
    assert calls == [(17, 640)]
    # same result as before this change: dense fp32 GEMM, rounded once to bf16
    torch.testing.assert_close(y.float(), dense(x).to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)


def test_had_rot_does_not_build_the_dense_gemm(monkeypatch):
    """`x.float() @ had_matrix` is the 835 us fp32 hipBLASLt kernel this replaces."""
    monkeypatch.setattr(R, "HAD", 128)
    monkeypatch.setattr(R, "had_matrix", lambda *a: (_ for _ in ()).throw(AssertionError("dense GEMM")))
    R.had_rot(rand(4, 256))


@pytest.mark.skipif(subprocess.run([sys.executable, "-c", "import triton"], capture_output=True).returncode != 0,
                    reason="triton not installed")
def test_triton_kernel_matches_butterfly_in_interpreter():
    """The GPU kernel's code, run by the Triton interpreter on CPU tensors."""
    code = (
        "import torch, radiance_fwht as F\n"
        "s = torch.where(torch.rand(128, generator=torch.Generator().manual_seed(0)) < .5, -1., 1.) / 128 ** .5\n"
        "for M in (1, 4, 16, 17):\n"
        "  for K in (128, 640):\n"
        "    for dt in (torch.float32, torch.bfloat16, torch.float16):\n"
        "      x = torch.randn(M, K, generator=torch.Generator().manual_seed(M * K)).to(dt)\n"
        "      a, b = F._fwht128_triton(x, s), F._fwht128_torch(x, s)\n"
        "      assert a.dtype == dt\n"
        "      tol = 1e-5 if dt == torch.float32 else 1e-2\n"
        "      torch.testing.assert_close(a.float(), b.float(), rtol=tol, atol=tol)\n"
        "print('INTERPRETER OK')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600,
                       env={**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": MXFP4})
    assert "INTERPRETER OK" in r.stdout, r.stdout + r.stderr
