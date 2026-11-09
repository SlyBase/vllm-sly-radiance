"""Block-128 fast Walsh-Hadamard transform for RADIANCE_MXFP4_HAD=128 (the unfused `had_rot` path).

y[:, b*128:(b+1)*128] = (H128 x[:, b*128:(b+1)*128]) * signs, H128 the Sylvester matrix. That is
`x.float() @ (H128 * signs)` (radiance_mxfp4.had_matrix) without the GEMM: 7 butterfly stages, fp32 math,
one rounding to the input dtype at the end.

Triton kernel: one program handles ROWS 128-blocks. Each stage pairs the lowest index bit
([ROWS, 64, 2] -> split -> a+b, a-b -> join -> [ROWS, 2, 64] -> [ROWS, 128]) and moves that bit to the
top, so after 7 stages the order is natural again and every bit was transformed once.
"""
import torch

try:
    import triton
    import triton.language as tl
except ImportError:   # CPU-only test environments; the GPU path needs it
    triton = None

N = 128
ROWS, WARPS = 2, 1   # ponytail: untuned guess (256 elements per 1-wave program), sweep in a GPU window

if triton is not None:
    @triton.jit
    def _fwht128_kernel(x_ptr, s_ptr, y_ptr, n_blocks, ROWS: tl.constexpr):
        blk = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
        col = tl.arange(0, 128)
        offs = blk[:, None] * 128 + col[None, :]
        v = tl.load(x_ptr + offs, mask=blk[:, None] < n_blocks, other=0.0).to(tl.float32)
        for _ in tl.static_range(7):
            a, b = tl.split(tl.reshape(v, [ROWS, 64, 2]))
            v = tl.reshape(tl.permute(tl.join(a + b, a - b), (0, 2, 1)), [ROWS, 128])
        v = v * tl.load(s_ptr + col)[None, :]
        tl.store(y_ptr + offs, v.to(y_ptr.dtype.element_ty), mask=blk[:, None] < n_blocks)


def _fwht128_triton(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    y = torch.empty_like(x)
    n_blocks = x.numel() // N
    _fwht128_kernel[(triton.cdiv(n_blocks, ROWS),)](x, signs, y, n_blocks, ROWS=ROWS, num_warps=WARPS)
    return y


def _fwht128_torch(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    v = x.float().reshape(-1, N)
    for h in (1, 2, 4, 8, 16, 32, 64):
        v = v.view(-1, N // (2 * h), 2, h)
        v = torch.stack((v[:, :, 0] + v[:, :, 1], v[:, :, 0] - v[:, :, 1]), 2)
    return (v.reshape(-1, N) * signs).reshape(x.shape).to(x.dtype)


def fwht128(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    """[M, K] (K % 128 == 0), bf16/fp16/fp32 -> same shape and dtype. `signs`: fp32 [128] on x.device."""
    assert x.shape[-1] % N == 0, "fwht128 needs K % 128 == 0"
    return (_fwht128_triton if x.is_cuda else _fwht128_torch)(x, signs)
