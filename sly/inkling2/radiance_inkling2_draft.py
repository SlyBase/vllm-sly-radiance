"""DSpark draft kernels for Inkling 2 on gfx12 (RDNA4): the launches around the draft's matmuls.

Per verify step the draft runs ~10 RMSNorms and 7 Markov steps. As vLLM writes them (and as our
torch RMSNorm replacement for flashinfer wrote it) that is ~10 elementwise launches per norm and ~10
launches plus a 103 MB hipBLASLt GEMM per Markov step. Same math, folded:

  rmsnorm(x, w, eps, residual)    one kernel: residual += x (rounded once), x = norm(fp32 sum)
  markov_step(...)                cat + embedding + rnn_proj addmm + markov_gate in one kernel
  fold_softcap(base, fw, fb, ..)  bias = addmm(fb, latent, fw^T) and add_softcap in one pass over the
                                  (vocab, rank) fold weight; one launch instead of two

Only the fp32 summation order differs from the ops they replace; every bf16 rounding point is kept
(raw -> gate/cand/out, bias, base + bias, the logits). tl.dot operand dtype follows the weight dtype
(bf16 on the GPU; fp32 in the CPU tests, the interpreter's bf16 dot is wrong). Each function returns
None where its input is outside the kernel's range and the caller keeps the original ops.
"""
import os

import torch
from triton.language.extra import libdevice as tldevice
from vllm.triton_utils import tl, triton

_LIB = os.environ.get("TRITON_INTERPRET") != "1"  # the Triton interpreter has no libdevice
MAX_ROWS = 16  # dot M tile of fold_softcap
_CB = 8        # Markov channels per program
_BV = 128      # vocab rows per program of fold_softcap
_BM = 32       # shortlist entries per program of fold_softcap_gather
# Drafter shortlist: vocabulary entries whose base logit is among the draft row's SHORTLIST_K largest get
# the Markov correction, the rest are -inf (0 = off, dense). Measured on code / prose / json draft rows:
# the correction moves the draft's argmax out of the base top-4096 in ~0.3 % of the rows.
SHORTLIST_K = int(os.environ.get("RADIANCE_INKLING2_DRAFT_SHORTLIST", "0"))


@triton.jit
def _bf16_round(x):
    # fp32 -> bf16 -> fp32 on the bits (round to nearest even): the compiler folds extf(truncf(x)),
    # and the interpreter's fp32 -> bf16 cast truncates.
    bits = x.to(tl.uint32, bitcast=True)
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _tanh(x, LIB: tl.constexpr):
    if LIB:
        return tldevice.tanh(x)
    else:
        e = tl.exp(-2.0 * tl.abs(x))
        return tl.where(x < 0, -1.0, 1.0) * (1.0 - e) / (1.0 + e)


@triton.jit
def _exp(x, LIB: tl.constexpr):
    if LIB:
        return tldevice.exp(x)
    else:
        return tl.exp(x)


# ------------------------------------------------------------------ RMSNorm


@triton.jit
def _rmsnorm_kernel(X, Res, W, Out, stride_x, stride_r, stride_o, N: tl.constexpr, eps,
                    HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    v = tl.load(X + row * stride_x + cols, mask, other=0).to(tl.float32)
    if HAS_RES:
        v += tl.load(Res + row * stride_r + cols, mask, other=0).to(tl.float32)
        tl.store(Res + row * stride_r + cols, _bf16_round(v).to(tl.bfloat16), mask)
    rstd = tl.rsqrt(tl.div_rn(tl.sum(v * v, 0), N) + eps)
    w = tl.load(W + cols, mask, other=0).to(tl.float32)
    tl.store(Out + row * stride_o + cols, (v * rstd * w).to(Out.dtype.element_ty), mask)


def rmsnorm(x, weight, eps, residual=None):
    """flashinfer's rmsnorm / fused_add_rmsnorm contract on 2D bf16 rows: fp32 norm, bf16 out. Without
    `residual` returns a new tensor; with it, `residual += x` (rounded to bf16 once) and
    `x = norm(unrounded fp32 sum)`, both in place, returns x."""
    rows, n = x.shape
    assert x.stride(1) == 1 and weight.is_contiguous()
    if residual is not None:
        assert residual.shape == x.shape and residual.stride(1) == 1
    out = x if residual is not None else torch.empty((rows, n), dtype=x.dtype, device=x.device)
    if rows:
        _rmsnorm_kernel[(rows,)](x, residual if residual is not None else x, weight, out,
                                 x.stride(0), residual.stride(0) if residual is not None else 0,
                                 out.stride(0), n, eps, residual is not None,
                                 triton.next_power_of_2(n), num_warps=4)
    return out


# ------------------------------------------------------------------ Markov step


@triton.jit
def _proj(W, base_row, c, k, st, emb, stride_w, R: tl.constexpr):
    # (CB,) dot of [state, emb] with rows `base_row + c` of W (columns [0, R) | [R, 2R))
    wr = W + (base_row + c[:, None]) * stride_w
    return (tl.sum(tl.load(wr + k[None, :]).to(tl.float32) * st[None, :], 1)
            + tl.sum(tl.load(wr + R + k[None, :]).to(tl.float32) * emb[None, :], 1))


@triton.jit
def _step_kernel(State, Prev, Ctx, W1, W, NewState, Latent, stride_w1, stride_w,
                 R: tl.constexpr, CB: tl.constexpr, LIB: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1) * CB + tl.arange(0, CB)
    k = tl.arange(0, R)
    st = tl.load(State + row * R + k).to(tl.float32)
    emb = tl.load(W1 + tl.load(Prev + row) * stride_w1 + k).to(tl.float32)

    gate = _bf16_round(tl.load(Ctx + row * 3 * R + c) + _proj(W, 0, c, k, st, emb, stride_w, R))
    cand = _bf16_round(tl.load(Ctx + row * 3 * R + R + c) + _proj(W, R, c, k, st, emb, stride_w, R))
    out = _bf16_round(tl.load(Ctx + row * 3 * R + 2 * R + c) + _proj(W, 2 * R, c, k, st, emb, stride_w, R))
    old = tl.load(State + row * R + c).to(tl.float32)
    # Each torch op of the reference rounds to bf16; its sigmoid is 1 / (1 + expf(-x)), IEEE division.
    gate = _bf16_round(tl.div_rn(1.0, 1.0 + _exp(-gate, LIB)))
    cand = _bf16_round(_tanh(cand, LIB))
    keep = _bf16_round(gate * old)
    update = _bf16_round(_bf16_round(1.0 - gate) * cand)
    tl.store(NewState + row * R + c, _bf16_round(keep + update).to(tl.bfloat16))
    tl.store(Latent + row * R + c, _bf16_round(_tanh(out, LIB)).to(tl.bfloat16))


def markov_step(state, prev_tokens, context, w1, w_rec):
    """One recurrence step of the Markov head: `raw = context + [state, w1[prev]] @ w_rec^T` (fp32),
    then the bf16 gating. state (rows, r) bf16, prev_tokens (rows,) int, context (rows, 3r) fp32,
    w1 (vocab, r), w_rec (3r, 2r) (a column slice of rnn_proj.weight is fine). Returns
    `(new_state, latent)`, both bf16 (rows, r) -- `markov_gate(raw, state)` of vLLM."""
    rows, r = state.shape
    assert r & (r - 1) == 0 and r % _CB == 0 and state.is_contiguous()
    assert context.shape == (rows, 3 * r) and context.is_contiguous()
    assert w_rec.shape == (3 * r, 2 * r) and w_rec.stride(1) == 1 and w1.stride(1) == 1
    new_state, latent = torch.empty_like(state), torch.empty_like(state)
    if rows:
        _step_kernel[(rows, r // _CB)](state, prev_tokens, context, w1, w_rec, new_state, latent,
                                       w1.stride(0), w_rec.stride(0), r, _CB, _LIB, num_warps=4)
    return new_state, latent


# ------------------------------------------------------------------ fold + softcap


@triton.jit
def _fold_softcap_kernel(Base, FW, FB, Lat, Out, stride_base, rows, V, softcap,
                         R: tl.constexpr, TP: tl.constexpr, BV: tl.constexpr, DOT: tl.constexpr,
                         LIB: tl.constexpr):
    v = tl.program_id(0) * BV + tl.arange(0, BV)
    vm = v < V
    r = tl.arange(0, TP)
    rm = r < rows
    k = tl.arange(0, R)
    lat = tl.load(Lat + r[:, None] * R + k[None, :], rm[:, None], other=0).to(DOT)
    fw = tl.load(FW + v[None, :].to(tl.int64) * R + k[:, None], vm[None, :], other=0).to(DOT)
    acc = tl.dot(lat, fw, input_precision="ieee")
    bias = _bf16_round(acc + tl.load(FB + v, vm, other=0).to(tl.float32)[None, :])
    m2 = rm[:, None] & vm[None, :]
    a = tl.load(Base + r[:, None].to(tl.int64) * stride_base + v[None, :], m2, other=0)
    x = _bf16_round(a.to(tl.float32) + bias) / softcap
    x = _tanh(x, LIB) * softcap
    tl.store(Out + r[:, None].to(tl.int64) * V + v[None, :], x.to(tl.bfloat16), m2)


def fold_softcap(base, fold_weight, fold_bias, latent, softcap):
    """bf16 (rows, V) `softcap * tanh(bf16(base + bias) / softcap)` with
    `bias = bf16(fold_bias + latent @ fold_weight^T)`: `addmm(fold_bias, latent, fold_weight.T)`
    followed by `add_softcap`, in one pass over fold_weight (V, r). base (rows, V) bf16 (a row-strided
    view is fine), latent (rows, r) bf16. None when rows > MAX_ROWS."""
    rows, V = base.shape
    r = fold_weight.shape[1]
    if rows > MAX_ROWS or rows == 0 or r & (r - 1) or base.stride(1) != 1:
        return None
    assert fold_weight.shape[0] == V and fold_weight.is_contiguous() and latent.shape == (rows, r)
    assert latent.is_contiguous()
    out = torch.empty((rows, V), dtype=torch.bfloat16, device=base.device)
    dot = tl.bfloat16 if fold_weight.dtype == torch.bfloat16 else tl.float32
    _fold_softcap_kernel[(triton.cdiv(V, _BV),)](
        base, fold_weight, fold_bias, latent, out, base.stride(0), rows, V, softcap, r,
        MAX_ROWS, _BV, dot, _LIB, num_warps=4)
    return out


# ------------------------------------------------------------------ drafter shortlist


def shortlist(base_logits, k):
    """(values, indices), each (rows, n_spec, k): the k largest base logits of every draft row of
    `base_logits` (rows, n_spec, V) and where they are. None when k is off (<= 0) or no smaller than
    the vocabulary."""
    if k <= 0 or k >= base_logits.shape[-1]:
        return None
    return torch.topk(base_logits, k, dim=-1)


@triton.jit
def _gather_fold_kernel(Vals, Idx, FW, FB, Lat, Out, stride_vals, stride_idx, K, V, softcap,
                        R: tl.constexpr, BM: tl.constexpr, LIB: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    mk = m < K
    idx = tl.load(Idx + row * stride_idx + m, mk, other=0).to(tl.int64)
    val = tl.load(Vals + row * stride_vals + m, mk, other=0).to(tl.float32)
    k = tl.arange(0, R)
    fw = tl.load(FW + idx[:, None] * R + k[None, :], mk[:, None], other=0).to(tl.float32)
    lat = tl.load(Lat + row * R + k).to(tl.float32)
    bias = _bf16_round(tl.sum(fw * lat[None, :], 1) + tl.load(FB + idx, mk, other=0).to(tl.float32))
    x = _bf16_round(val + bias) / softcap
    x = _tanh(x, LIB) * softcap
    tl.store(Out + row * V + idx, x.to(tl.bfloat16), mk)


def fold_softcap_gather(vals, idx, fold_weight, fold_bias, latent, softcap, V):
    """`fold_softcap` for a shortlist: vals (rows, k) the base logits at the vocabulary positions idx
    (rows, k); returns bf16 (rows, V), -inf outside the shortlist. Only the fold-weight rows of the
    shortlist are read (k x r instead of V x r: ~2 MB instead of 103 MB per draft step). vals / idx may be
    row-strided views (draft step i of a (rows, n_spec, k) topk). None when r is not a power of two."""
    rows, k = vals.shape
    r = fold_weight.shape[1]
    if r & (r - 1) or vals.stride(1) != 1 or idx.stride(1) != 1:
        return None
    assert fold_weight.shape[0] == V and fold_weight.is_contiguous() and latent.shape == (rows, r)
    assert latent.is_contiguous() and idx.shape == vals.shape
    out = torch.full((rows, V), float("-inf"), dtype=torch.bfloat16, device=vals.device)
    _gather_fold_kernel[(rows, triton.cdiv(k, _BM))](
        vals, idx, fold_weight, fold_bias, latent, out, vals.stride(0), idx.stride(0), k, V, softcap,
        r, _BM, _LIB, num_warps=4)
    return out
