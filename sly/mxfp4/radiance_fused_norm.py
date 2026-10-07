"""Fused norm / activation + per-token fp8 quant in front of the W4A8 MXFP4 GEMMs.

RADIANCE_FUSED_NORM_QUANT=1 (default off). The .hip already carries three kernels that end in the
exact per-token e4m3 quant `radiance::mxfp4_linear` runs internally, so the (q, scale) pair can go
straight onto `radiance::mxfp4_linear_pq` (apply_weights' tuple branch) instead of a bf16 tensor
that the op re-quantizes:

  add_rms_quant   Qwen3.5 decoder input_layernorm / post_attention_layernorm (residual != None):
                  f32 add -> Gemma rms_norm (1+w) -> bf16 -> e4m3, plus the bf16 residual out.
                  Consumers: qkv_proj (attention), in_proj_qkvz + in_proj_ba (GDN), gate_up_proj.
  silu_mul_quant  Qwen2MoeMLP act_fn (= Qwen3NextMLP): silu(gate) * up -> e4m3 for down_proj.
  gdn_norm_quant  QwenGatedDeltaNetAttention._output_projection: per-head RMSNormGated
                  (norm_before_gate, silu gate) -> e4m3 for out_proj. z stays a column view of the
                  fused qkvz projection (row stride passed to the kernel, no copy).

Per-fusion switches (only read when the master knob is on, default 1 each):
RADIANCE_FUSED_NORM_QUANT_ADD_RMS, RADIANCE_FUSED_NORM_QUANT_SILU, RADIANCE_FUSED_NORM_QUANT_GDN.

Rules this module follows (same as radiance_mxfp4.py): the gates below run in traced code, so they
are plain attribute / dtype / shape reads that dynamo folds into constants per layer; no stderr in
traced code (the one-shot activation log lives inside the op bodies); every output is allocated
with torch.empty inside the op (CUDA-graph capture). A site is only fused when EVERY consumer of
the quantized activation is a folded radiance W4A8 layer -- anything else keeps the stock chain.
The env knobs are added to vLLM's compile cache key by sly/patch_fused_norm_quant.py (envs.py
compile_factors hashes VLLM_* only), so toggling them never replays a stale AOT graph.
"""

import os
import sys

import torch


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default) == "1"


ENABLED = _flag("RADIANCE_FUSED_NORM_QUANT", "0")
ADD_RMS = ENABLED and _flag("RADIANCE_FUSED_NORM_QUANT_ADD_RMS", "1")
SILU = ENABLED and _flag("RADIANCE_FUSED_NORM_QUANT_SILU", "1")
GDN = ENABLED and _flag("RADIANCE_FUSED_NORM_QUANT_GDN", "1")

# Kernel launcher limits (radiance_mxfp4_fp8.hip): add_rms K <= 10240, silu N <= 18432 (MAXG 9,
# 0.1.6), gdn N <= 10240 with 128-wide heads; all 8-aligned.
_ADD_RMS_MAX_K = 10240
_SILU_MAX_N = 18432
_GDN_MAX_N = 10240

_ext = None
_rmx_tiled_wanted = _rmx_tiled_register = None
_ROCM = False
_GemmaRMSNorm = _RMSNormGated = _SiluAndMul = ()

if ENABLED:
    import radiance_mxfp4 as _rmx

    if not _rmx.ENABLED or _rmx._ext is None:
        sys.stderr.write("[radiance.fused_norm] RADIANCE_FUSED_NORM_QUANT=1 but the W4A8 kernel is "
                         "not active (RADIANCE_MXFP4_W4A8 / extension) -- fusions OFF\n")
        ADD_RMS = SILU = GDN = False
    else:
        if _rmx.MIN_M > 0:
            raise RuntimeError("RADIANCE_FUSED_NORM_QUANT=1 needs RADIANCE_MXFP4_W4A8_MIN_M=0: below "
                               "MIN_M mxfp4_linear hands the layer to aiter's W4A4 path, which cannot "
                               "consume a pre-quantized fp8 activation.")
        if _rmx.SANITIZE_X:
            raise RuntimeError("RADIANCE_FUSED_NORM_QUANT=1 is incompatible with "
                               "RADIANCE_MXFP4_SANITIZE=1 (the fused kernels quantize before any "
                               "nan_to_num could run).")
        _ext = _rmx._ext
        if getattr(_rmx, "A_TILED_MIN_M", 0):
            _rmx_tiled_wanted, _rmx_tiled_register = _rmx.a_tiled_wanted, _rmx.a_tiled_register
        from vllm.model_executor.layers.activation import SiluAndMul as _SiluAndMul
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm as _GemmaRMSNorm
        from vllm.model_executor.layers.layernorm import RMSNormGated as _RMSNormGated
        from vllm.platforms import current_platform

        _ROCM = bool(current_platform.is_rocm())
        sys.stderr.write(f"[radiance.fused_norm] armed: add_rms_quant={int(ADD_RMS)} "
                         f"silu_mul_quant={int(SILU)} gdn_norm_quant={int(GDN)} "
                         f"a_tiled_min_m={getattr(_rmx, 'A_TILED_MIN_M', 0)}\n")

_seen: set = set()


def _once(tag: str, msg: str) -> None:
    # Op bodies only (opaque to dynamo). Proves the fused path actually ran.
    if tag not in _seen:
        _seen.add(tag)
        sys.stderr.write(f"[radiance.fused_norm] {tag} active: {msg}\n")
        sys.stderr.flush()


def _check_bf16(name: str, *ts: torch.Tensor) -> None:
    for t in ts:
        if t.dtype != torch.bfloat16:
            raise RuntimeError(f"radiance::{name}: expected bfloat16, got {t.dtype}")


# ---- fragment-tiled output (RADIANCE_MXFP4_A_TILED_MIN_M) ----------------------------------------
# At prefill-class M the two producers whose every consumer is a folded W4A8 GEMM write q in the
# WMMA-fragment-tiled layout radiance_mxfp4_fp8_gemm_atiled reads straight into registers (no A tile
# in LDS; measured 12-16% faster than the folded kernel at M >= 2048, radiance_mxfp4_fp8.hip). The
# decision is per call inside the opaque op body (eager ints, no traced guard); the consumer finds
# it through radiance_mxfp4's data_ptr registry. The storage is padded to whole 16-row fragments and
# q is the [M, K] view of it, so real and fake shapes agree. gdn_norm_quant has no tiled variant and
# stays row-major (out_proj takes the folded kernel). Values are identical in both layouts.

def _tiled(M: int, K: int) -> bool:
    return (_rmx_tiled_wanted is not None and _rmx_tiled_wanted(M) and K % 128 == 0)


def _q_alloc(M: int, K: int, device, tiled: bool) -> torch.Tensor:
    if tiled:
        return torch.empty(((M + 15) // 16 * 16, K), device=device, dtype=torch.float8_e4m3fn)[:M]
    return torch.empty((M, K), device=device, dtype=torch.float8_e4m3fn)


# ---- custom ops -------------------------------------------------------------------------------

@torch.library.custom_op("radiance::add_rms_quant", mutates_args=())
def _add_rms_quant_op(y: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                      eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """y, residual: [M, K] bf16; weight: [K] bf16 (Gemma convention, the kernel adds the 1).
    Returns (q [M, K] e4m3, scale [M] f32, residual_out [M, K] bf16)."""
    _check_bf16("add_rms_quant", y, residual, weight)
    M, K = y.shape
    tiled = _tiled(M, K)
    q = _q_alloc(M, K, y.device, tiled)
    s = torch.empty((M,), device=y.device, dtype=torch.float32)
    r = torch.empty((M, K), device=y.device, dtype=torch.bfloat16)
    if M:
        y = y.contiguous()
        residual = residual.contiguous()
        weight = weight.contiguous()
        _once("add_rms_quant", f"M={M} K={K}")
        _ext.launch_add_rms_quant(y.data_ptr(), residual.data_ptr(), weight.data_ptr(),
                                  q.data_ptr(), s.data_ptr(), r.data_ptr(), M, K, float(eps),
                                  torch.cuda.current_stream().cuda_stream, 1 if tiled else 0)
        if tiled:
            _rmx_tiled_register(q, M, K)
    return q, s, r


@_add_rms_quant_op.register_fake
def _(y, residual, weight, eps):
    M, K = y.shape
    return (torch.empty((M, K), device=y.device, dtype=torch.float8_e4m3fn),
            torch.empty((M,), device=y.device, dtype=torch.float32),
            torch.empty((M, K), device=y.device, dtype=torch.bfloat16))


@torch.library.custom_op("radiance::silu_mul_quant", mutates_args=())
def _silu_mul_quant_op(gate_up: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """gate_up: [M, 2N] bf16 -> (q [M, N] e4m3 of silu(gate)*up, scale [M] f32)."""
    _check_bf16("silu_mul_quant", gate_up)
    M, N2 = gate_up.shape
    N = N2 // 2
    tiled = _tiled(M, N)
    q = _q_alloc(M, N, gate_up.device, tiled)
    s = torch.empty((M,), device=gate_up.device, dtype=torch.float32)
    if M:
        gate_up = gate_up.contiguous()
        _once("silu_mul_quant", f"M={M} N={N}")
        _ext.launch_silu_mul_quant(gu=gate_up.data_ptr(), q=q.data_ptr(), scale=s.data_ptr(),
                                   M=M, N=N, stream=torch.cuda.current_stream().cuda_stream,
                                   tiled=1 if tiled else 0)
        if tiled:
            _rmx_tiled_register(q, M, N)
    return q, s


@_silu_mul_quant_op.register_fake
def _(gate_up):
    M, N2 = gate_up.shape
    return (torch.empty((M, N2 // 2), device=gate_up.device, dtype=torch.float8_e4m3fn),
            torch.empty((M,), device=gate_up.device, dtype=torch.float32))


@torch.library.custom_op("radiance::gdn_norm_quant", mutates_args=())
def _gdn_norm_quant_op(x: torch.Tensor, z: torch.Tensor, weight: torch.Tensor,
                       eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """x: [M, N] bf16 (N = heads * 128); z: [M, N] bf16, any row stride as long as each row is a
    contiguous run; weight: [128] bf16. Returns (q [M, N] e4m3, scale [M] f32)."""
    _check_bf16("gdn_norm_quant", x, z, weight)
    M, N = x.shape
    q = torch.empty((M, N), device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty((M,), device=x.device, dtype=torch.float32)
    if M:
        x = x.contiguous()
        weight = weight.contiguous()
        if M > 1 and (z.stride(1) != 1 or z.stride(0) < N or z.stride(0) % 8):
            z = z.contiguous()
        elif M == 1 and z.stride(1) != 1:
            z = z.contiguous()
        zs = int(z.stride(0)) if M > 1 else N
        _once("gdn_norm_quant", f"M={M} N={N} z_row_stride={zs}")
        _ext.launch_gdn_norm_quant(x.data_ptr(), z.data_ptr(), zs, weight.data_ptr(), q.data_ptr(),
                                   s.data_ptr(), M, N, float(eps),
                                   torch.cuda.current_stream().cuda_stream)
    return q, s


@_gdn_norm_quant_op.register_fake
def _(x, z, weight, eps):
    M, N = x.shape
    return (torch.empty((M, N), device=x.device, dtype=torch.float8_e4m3fn),
            torch.empty((M,), device=x.device, dtype=torch.float32))


# ---- gates (traced; plain attribute reads only) -------------------------------------------------

def mx_ok(lin) -> bool:
    """A folded radiance W4A8 linear whose apply_weights takes the (q, scale) tuple branch."""
    wref = getattr(lin, "radiance_wref", None)
    w = getattr(lin, "weight", None)
    return (wref is not None and w is not None and wref.numel() == w.shape[0]
            and not getattr(lin, "_rad_ar_overlap", False)
            and getattr(lin, "tp_size", 1) == 1)


def _gemma_ok(norm) -> bool:
    w = getattr(norm, "weight", None)
    return (isinstance(norm, _GemmaRMSNorm) and w is not None and w.dtype == torch.bfloat16
            and w.shape[0] % 8 == 0 and w.shape[0] <= _ADD_RMS_MAX_K)


def input_ok(layer) -> bool:
    """Qwen3NextDecoderLayer.input_layernorm (residual != None) -> the attention's projections."""
    if getattr(layer, "use_attn_reduce_scatter_for_moe", True):
        return False
    if not _gemma_ok(getattr(layer, "input_layernorm", None)):
        return False
    kind = getattr(layer, "layer_type", None)
    if kind == "linear_attention":
        a = layer.linear_attn
        # only forward_hip / forward_cuda carry the tuple (sly/patch_fused_norm_quant.py)
        return (_ROCM and getattr(a, "tp_size", 0) == 1
                and mx_ok(getattr(a, "in_proj_qkvz", None))
                and mx_ok(getattr(a, "in_proj_ba", None)))
    if kind == "full_attention":
        return mx_ok(getattr(layer.self_attn, "qkv_proj", None))
    return False


def post_ok(layer) -> bool:
    """Qwen3NextDecoderLayer.post_attention_layernorm -> dense MLP gate_up_proj."""
    if getattr(layer, "use_attn_reduce_scatter_for_moe", True):
        return False
    if not _gemma_ok(getattr(layer, "post_attention_layernorm", None)):
        return False
    mlp = getattr(layer, "mlp", None)
    # expert_gate would read x as a tensor; the sparse MoE block has no gate_up_proj at all
    return getattr(mlp, "expert_gate", 1) is None and mx_ok(getattr(mlp, "gate_up_proj", None))


def mlp_ok(mlp) -> bool:
    """Qwen2MoeMLP.act_fn -> down_proj."""
    d = getattr(mlp, "down_proj", None)
    if not (isinstance(getattr(mlp, "act_fn", None), _SiluAndMul) and mx_ok(d)):
        return False
    n = d.weight.shape[1] * 2
    return n % 8 == 0 and n <= _SILU_MAX_N


def gdn_ok(attn) -> bool:
    """QwenGatedDeltaNetAttention._output_projection -> out_proj."""
    n = getattr(attn, "norm", None)
    if not (_ROCM and isinstance(n, _RMSNormGated) and mx_ok(getattr(attn, "out_proj", None))):
        return False
    return (getattr(attn, "tp_size", 0) == 1 and getattr(attn, "head_v_dim", 0) == 128
            and n.group_size is None and n.norm_before_gate
            and n.activation in ("silu", "swish") and n.weight.dtype == torch.bfloat16
            and n.weight.shape[0] == 128 and attn.value_dim <= _GDN_MAX_N)


# ---- call sites (traced) ----------------------------------------------------------------------

def add_rms_quant(norm, hidden_states: torch.Tensor, residual: torch.Tensor):
    """GemmaRMSNorm(hidden_states, residual) + quant -> ((q, scale), residual_out)."""
    q, s, r = torch.ops.radiance.add_rms_quant(hidden_states, residual, norm.weight,
                                               norm.variance_epsilon)
    return (q, s), r


def silu_mul_quant(gate_up: torch.Tensor):
    q, s = torch.ops.radiance.silu_mul_quant(gate_up)
    return (q, s)


def gdn_norm_quant(attn, core_attn_out: torch.Tensor, z: torch.Tensor):
    """_output_projection's norm(core_attn_out, z) + flatten + quant -> (q, scale)."""
    M = core_attn_out.shape[0]
    q, s = torch.ops.radiance.gdn_norm_quant(core_attn_out.reshape(M, -1), z.reshape(M, -1),
                                             attn.norm.weight, attn.norm.eps)
    return (q, s)
