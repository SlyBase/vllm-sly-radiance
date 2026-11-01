"""NVFP4 diagnostics: serve NVFP4 linears WITHOUT the radiance W4A8 kernel, dequantized to bf16 per forward.

Measurement only (--enforce-eager, slow: every forward dequantizes every weight). Selected by
RADIANCE_NVFP4_DIAG (radiance_nvfp4.scheme_class / bf16_scheme_class delegate here when it is set) and
compared with sly/bench_fidelity.py + sly/check_fidelity.py. Modes:

  native   the checkpoint's weights exactly (e2m1 * e4m3 block scale / global), bf16 activations; the
           GDN in_proj_a/b stay bf16 (RADIANCE_NVFP4_BF16_LAYERS is ignored). The reference arm.
  requant  radiance_nvfp4's NVFP4 -> MXFP4 requantization (incl. RADIANCE_NVFP4_BF16_LAYERS), but bf16
           activations: separates the requant loss from the fp8-activation loss of the W4A8 kernel.
  fold     "NV fold": e2m1 * e4m3 block scale rounded to ONE e4m3 byte, the way the W4A8 kernel folds the
           MX block exponent into the weight byte (kMag LUT), row factor 2^-max(block exponent) exact.
           Emulates a native NVFP4 kernel path without requantization; in_proj_a/b bf16.
RADIANCE_NVFP4_DIAG_A8=1 adds per-token e4m3 activations (amax/448) as the W4A8 kernel feeds them.
Measured on ThinkingCap-Qwen3.8-27B-NVFP4 (2026-09-29, 14,125 tokens, vs native): requant +0.088 NLL,
W4A8 serve path +0.093, top-1 agreement 91 %. Offline weight error vs native: requant 0.11, fold 0.021.
"""
import os
import re
import sys

import torch
import torch.nn.functional as F
from torch.nn import Parameter

import radiance_nvfp4 as R

MODE = os.environ.get("RADIANCE_NVFP4_DIAG", "native")
A8 = os.environ.get("RADIANCE_NVFP4_DIAG_A8", "0") == "1"
ROWS = 4096
_CLS = {}
_N = {"layers": 0}


def _log(msg):
    sys.stderr.write(f"[radiance.nvfp4diag] {msg}\n")
    sys.stderr.flush()


def _deq_nv(packed, scale, rowdiv, dtype, rowshift=None):
    """rowshift [n] (fold only): 2^-max(block exponent) per row; e2m1 * scale * rowshift is rounded to e4m3
    as a kernel LUT would store it, then scaled back exactly."""
    n, k = packed.shape[0], packed.shape[1] * 2
    out = torch.empty(n, k, dtype=dtype, device=packed.device)
    for a in range(0, n, ROWS):
        b = min(n, a + ROWS)
        v = R.unpack_e2m1(packed[a:b]).reshape(b - a, k // 16, 16)
        sc = scale[a:b].to(torch.float32)
        if rowshift is None:
            w = v * (sc / rowdiv[a:b].unsqueeze(-1)).unsqueeze(-1)
        else:
            sh = rowshift[a:b].view(-1, 1, 1)
            w = (v * sc.unsqueeze(-1) * sh).to(torch.float8_e4m3fn).float() / sh / rowdiv[a:b].view(-1, 1, 1)
        out[a:b] = w.reshape(b - a, k).to(dtype)
    return out


def _a8(x):
    s = x.abs().amax(-1, keepdim=True).float().clamp(min=1e-12) / 448.0
    return ((x.float() / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)


def _deq_mx(packed, e8m0, dtype):
    n = packed.shape[0]
    out = torch.empty(n, packed.shape[1] * 2, dtype=dtype, device=packed.device)
    for a in range(0, n, ROWS):
        b = min(n, a + ROWS)
        out[a:b] = R.dequant_mxfp4(packed[a:b], e8m0[a:b]).to(dtype)
    return out


def _make():
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import CompressedTensorsScheme
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter, ModelWeightParameter, PerTensorScaleParameter)

    class DiagNvfp4(CompressedTensorsScheme):
        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        def create_weights(self, layer, output_partition_sizes, input_size_per_partition,
                           params_dtype, weight_loader, **kwargs):
            n = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            layer.input_size_per_partition = input_size_per_partition
            layer.output_size_per_partition = n
            layer.params_dtype = params_dtype
            layer.register_parameter("weight_packed", ModelWeightParameter(
                data=torch.empty(n, input_size_per_partition // 2, dtype=torch.uint8),
                input_dim=1, output_dim=0, weight_loader=weight_loader))
            layer.register_parameter("weight_global_scale", PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader))
            layer.register_parameter("weight_scale", GroupQuantScaleParameter(
                data=torch.empty(n, input_size_per_partition // 16, dtype=torch.float8_e4m3fn),
                input_dim=1, output_dim=0, weight_loader=weight_loader))
            layer.register_parameter("input_global_scale", PerTensorScaleParameter(
                data=torch.empty(len(output_partition_sizes), dtype=torch.float32),
                weight_loader=weight_loader))

        @torch.no_grad()
        def process_weights_after_loading(self, layer) -> None:
            packed = layer.weight_packed.data
            scale = layer.weight_scale.data
            gs = layer.weight_global_scale.data.float().tolist()
            widths = list(layer.logical_widths)
            rowdiv = torch.cat([torch.full((w,), g, dtype=torch.float32, device=packed.device)
                                for w, g in zip(widths, gs)])
            del layer.weight_global_scale, layer.input_global_scale
            if MODE in ("native", "fold"):
                layer.diag_rowdiv = Parameter(rowdiv, requires_grad=False)
                layer.weight = layer.weight_packed          # as radiance: .weight = packed uint8
                layer.diag_kind = "nv"
                if MODE == "fold":
                    kexp = torch.floor(torch.log2(scale.float().clamp(min=2.0 ** -20)))
                    layer.diag_rowshift = Parameter(torch.exp2(-kexp.amax(1)), requires_grad=False)
                    layer.diag_kind = "fold"
            else:
                w = _deq_nv(packed, scale, rowdiv, torch.float32)
                p, s, rel = R.requant_rows(w)
                del w, layer.weight_packed, layer.weight_scale
                layer.diag_mx = Parameter(p, requires_grad=False)
                layer.diag_e8 = Parameter(s, requires_grad=False)
                layer.weight = layer.diag_mx
                layer.diag_kind = "mx"
            _N["layers"] += 1
            if _N["layers"] in (1, 64, 200) or _N["layers"] % 100 == 0:
                _log(f"{MODE}: {_N['layers']} layers prepared")

        def apply_weights(self, layer, x, bias=None):
            if A8:
                x = _a8(x)
            if layer.diag_kind == "nv":
                w = _deq_nv(layer.weight_packed, layer.weight_scale, layer.diag_rowdiv, x.dtype)
            elif layer.diag_kind == "fold":
                w = _deq_nv(layer.weight_packed, layer.weight_scale, layer.diag_rowdiv, x.dtype,
                            layer.diag_rowshift)
            else:
                w = _deq_mx(layer.diag_mx, layer.diag_e8, x.dtype)
            return F.linear(x, w, bias)

    class DiagBf16Requant(CompressedTensorsScheme):
        """bf16 linear (in_proj_a/b) -> MXFP4 as radiance does, forward with bf16 activations."""

        @classmethod
        def get_min_capability(cls) -> int:
            return 80

        def create_weights(self, layer, output_partition_sizes, input_size_per_partition,
                           params_dtype, weight_loader, **kwargs):
            n = sum(output_partition_sizes)
            layer.logical_widths = output_partition_sizes
            layer.input_size_per_partition = input_size_per_partition
            layer.output_size_per_partition = n
            layer.params_dtype = params_dtype
            layer.register_parameter("weight", ModelWeightParameter(
                data=torch.empty(n, input_size_per_partition, dtype=params_dtype),
                input_dim=1, output_dim=0, weight_loader=weight_loader))

        @torch.no_grad()
        def process_weights_after_loading(self, layer) -> None:
            p, s, rel = R.requant_rows(layer.weight.data)
            del layer.weight
            layer.diag_mx = Parameter(p, requires_grad=False)
            layer.diag_e8 = Parameter(s, requires_grad=False)
            layer.weight = layer.diag_mx

        def apply_weights(self, layer, x, bias=None):
            return F.linear(x, _deq_mx(layer.diag_mx, layer.diag_e8, x.dtype), bias)

    return DiagNvfp4, DiagBf16Requant


def scheme_class():
    if "nv" not in _CLS:
        _CLS["nv"], _CLS["bf16"] = _make()
        _log(f"mode={MODE} a8={A8} (dequant per forward)")
    return _CLS["nv"]


def bf16_scheme_class(layer_name):
    if MODE != "requant" or not R.BF16_LAYERS or not layer_name or not re.search(R.BF16_LAYERS, layer_name):
        return None
    scheme_class()
    return _CLS["bf16"]

