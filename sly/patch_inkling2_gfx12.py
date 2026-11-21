#!/usr/bin/env python3
"""Inkling 2 on gfx12 (RDNA4): run the model's sm120 (Triton) path on ROCm.

vLLM's Inkling 2 (vllm/models/inkling2/nvidia, upstream main / PR #69028, in no release yet) picks
its kernels by CUDA capability family: sm100 (tcgen05, TRT-LLM-gen), sm90 (Humming) or sm120
(plain Triton kernels, no tcgen05). gfx1201 is none of them, so this patch makes the sm120 path the
ROCm path, with four kinds of edits:

  1. gates      `is_device_capability_family(120)` also accepts ROCm (not in ops/nvfp4.py: there it
                selects the native e2m1 PTX conversion, which gfx12 lacks; the software path stays).
  2. PDL        `tl.extra.cuda.gdc_*()` -> `pass`, `launch_pdl=True` dropped (NVIDIA programmatic
                dependent launch; ROCm's Triton has neither).
  3. PTX        every `tl.inline_asm_elementwise` that is hot-path math (ex2/lg2/rcp/rsqrt, sigmoid,
                tanh, bf16 rounding, a no-op register move, an L2 prefetch hint) -> the equivalent
                Triton expression (fast-math approximations become exact ones: a few ulp apart, not a
                semantic change). Left alone on purpose: the fp8-KV unpack (attention_sm90.py, only
                compiled with --kv-cache-dtype fp8), the e2m1 hardware path (dead behind the constexpr
                `_HW_E2M1`) and the Gluon calls (`gl.inline_asm_elementwise`, sm100 only).
  4. experts    ops/moe_sm120.py: NVFP4 expert GEMM -> radiance_inkling2_moe.nvfp4_expert_gemm
                (dequantize tiles + bf16 dot; gfx12 has no FP4 MMA / `tl.dot_scaled`).

  4b. libdevice `tldevice.rcp_rn(x)` -> `tl.div_rn(1.0, x)`, `tldevice.div_rn` -> `tl.div_rn` (hip libdevice lacks both).
  4c. launch kwarg `maxnreg` dropped; `if not _HW_E2M1:` functions keep only the software path.
  4d. MXFP8 expert tiles shrunk on ROCm (gfx12 LDS is 64 KB).
  4e. rel_attention num_stages 2 -> 1 on ROCm.
  4f. linear(): `not use_sm100_kernels()` before `is_flashinfer_autotuning()` (imports flashinfer).
  4g. DSpark draft `_RMSNorm`: torch implementation of flashinfer's rmsnorm contract.
  4h. DSpark draft KV write -> radiance_inkling2_kv.kv_scatter (Triton; reshape_and_cache_flash is CUDA-only).
  5. platform   platforms/rocm.py `supported_quantization` gains "inkling2" (the quantization name
                Inkling's ModelConfig resolves to; otherwise ModelConfig refuses it on ROCm).

A vLLM without models/inkling2 (every release up to 0.34) is a NOOP.
"""
import ast
import os
import re
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(os.environ.get("RADIANCE_PATCH_TARGET_DIR") or sysconfig.get_paths()["purelib"])
ROOT = SP / "vllm/models/inkling2"
GATE = "current_platform.is_device_capability_family(120)"
GATE_ROCM = f"({GATE} or current_platform.is_rocm())"

# PTX text (normalised: whitespace squeezed) -> Triton expression over the asm's first argument `{0}`
_ASM = [
    (r"^ex2\.approx(\.ftz)?\.f32 \$0, \$1;$", "tl.exp2({0})"),
    (r"^lg2\.approx(\.ftz)?\.f32 \$0, \$1;$", "tl.log2({0})"),
    (r"^rcp\.approx(\.ftz)?\.f32 \$0, \$1;$", "(1.0 / ({0}))"),
    (r"^rsqrt\.approx\.f32 \$0, \$1;$", "tl.rsqrt({0})"),
    (r"^tanh\.approx\.f32 \$0, \$1;$", "(1.0 - 2.0 / (tl.exp2(({0}) * 2.8853900817779268) + 1.0))"),
    (r"^\{ \.reg \.f32 t; mul\.rn\.f32 t, \$1, 0fBFB8AA3B; .*rcp\.approx\.ftz\.f32 \$0, t; \}$",
     "(1.0 / (1.0 + tl.exp2(({0}) * -1.4426950408889634)))"),
    (r"^cvt\.rn\.bf16\.f32 \$0, \$1;$", "({0}).to(tl.bfloat16)"),
    (r"^mov\.b32 \$0, \$1;$", "({0})"),
]
_PREFETCH = r"prefetch\.global\.L2"  # statement-level hint, result unused -> `pass`


def _norm(s):
    return re.sub(r"\s+", " ", s).strip()


def _asm_text(node, consts):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _norm(node.value)
    if isinstance(node, ast.Name) and node.id in consts:
        return _norm(consts[node.id])
    return None


def _module_str_consts(tree):
    out = {}
    for n in tree.body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            v = n.value
            if isinstance(v, ast.Call) and v.args:  # tl.constexpr("...")
                v = v.args[0]
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                out[n.targets[0].id] = v.value
    return out


def replace_ptx(src):
    """Source with the known `tl.inline_asm_elementwise` calls replaced; (new source, count)."""
    tree = ast.parse(src)
    consts = _module_str_consts(tree)
    stmts = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Expr)}
    lines = src.splitlines(keepends=True)
    edits = []
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr == "inline_asm_elementwise"
                and isinstance(n.func.value, ast.Name) and n.func.value.id == "tl"):
            continue
        asm = _asm_text(n.args[0] if n.args else next(k.value for k in n.keywords
                                                       if k.arg == "asm_str"), consts)
        if asm is None:
            continue
        args = n.args[2] if len(n.args) > 2 else next(
            (k.value for k in n.keywords if k.arg == "args"), None)
        if id(n) in stmts and re.search(_PREFETCH, asm):
            new = "pass"
        else:
            tmpl = next((t for pat, t in _ASM if re.search(pat, asm)), None)
            if tmpl is None or not isinstance(args, ast.List) or len(args.elts) != 1:
                continue
            new = tmpl.format(ast.get_source_segment(src, args.elts[0]))
        edits.append((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset, new))
    for l0, c0, l1, c1, new in sorted(edits, reverse=True):
        head = lines[l0 - 1].encode()[:c0].decode()
        tail = lines[l1 - 1].encode()[c1:].decode()
        lines[l0 - 1:l1] = [head + new + tail]
    return "".join(lines), len(edits)


def drop_hw_e2m1(src):
    """In every function whose body is `if not _HW_E2M1: <software path> return ...` followed by the
    PTX path, keep only the software path (Triton compiles the dead tail too)."""
    tree = ast.parse(src)
    lines = src.splitlines(keepends=True)
    edits = []
    for fn in (n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)):
        for i, st in enumerate(fn.body):
            if (isinstance(st, ast.If) and isinstance(st.test, ast.UnaryOp)
                    and isinstance(st.test.operand, ast.Name) and st.test.operand.id == "_HW_E2M1"
                    and i + 1 < len(fn.body)):
                body = "".join(lines[st.body[0].lineno - 1:st.body[-1].end_lineno])
                dedented = "".join(l[4:] if l.startswith("    ") else l
                                   for l in body.splitlines(keepends=True))
                edits.append((st.lineno, fn.body[-1].end_lineno, dedented))
    for a, b, new in sorted(edits, reverse=True):
        lines[a - 1:b] = [new]
    return "".join(lines)


def patch_file(p):
    s = o = p.read_text()
    if p.name != "nvfp4.py":  # there the gate means "native e2m1 conversions": keep it false
        s = re.sub(re.escape(GATE) + r"(?! or current_platform\.is_rocm)", GATE_ROCM, s)
    s = re.sub(r"^(\s*)tl\.extra\.cuda\.gdc_(?:wait|launch_dependents)\(\)[ \t]*$", r"\1pass", s,
               flags=re.M)
    s = re.sub(r"^[ \t]*launch_pdl=True,[ \t]*\n", "", s, flags=re.M)
    s = s.replace(", launch_pdl=True", "")
    # hip's libdevice lacks rcp_rn / div_rn; tl.div_rn (core) is the same correctly rounded division
    s = re.sub(r"tldevice\.rcp_rn\(((?:[^()]|\([^()]*\))*)\)", r"tl.div_rn(1.0, \1)", s)
    s = s.replace("tldevice.div_rn(", "tl.div_rn(")
    s = re.sub(r"^[ \t]*maxnreg=maxnreg,[ \t]*\n", "", s, flags=re.M)
    s, _ = replace_ptx(s)
    if p.name == "nvfp4.py":
        s = drop_hw_e2m1(s)
    if s == o:
        return False
    ast.parse(s)  # never write a file that would not parse
    p.write_text(s)
    return True


def main():
    if not ROOT.is_dir():
        print("  NOOP  inkling2 gfx12: this vLLM has no models/inkling2")
        return
    changed = [p.name for p in sorted(ROOT.rglob("*.py")) if patch_file(p)]
    print(f"  OK    inkling2 gfx12: gates/PDL/PTX rewritten in {len(changed)} files" if changed
          else "  NOOP  inkling2 gfx12 gates/PDL/PTX already applied")
    apply(
        ROOT / "nvidia/ops/moe_sm120.py",
        "    assert weight.dtype == torch.uint8\n    n, k = weight.shape[1], weight.shape[2] * 2\n",
        "    assert weight.dtype == torch.uint8\n    n, k = weight.shape[1], weight.shape[2] * 2\n"
        "    if torch.version.hip:  # gfx12 has no FP4 MMA: dequantize tiles, bf16 dot\n"
        "        from radiance_inkling2_moe import nvfp4_expert_gemm\n\n"
        "        return nvfp4_expert_gemm(act, act_scale, row_scale, weight, weight_scale, ones,\n"
        "                                 tiles, plan)\n",
        "from radiance_inkling2_moe import nvfp4_expert_gemm",
        "inkling2 moe_sm120: NVFP4 experts -> dequant GEMM on ROCm",
    )
    apply(
        SP / "vllm/platforms/rocm.py",
        '    supported_quantization: list[str] = [\n        "awq",\n',
        '    supported_quantization: list[str] = [\n        "inkling2",\n        "awq",\n',
        '"inkling2",\n        "awq"',
        "rocm platform: inkling2 in supported_quantization",
    )
    apply(
        ROOT / "nvidia/ops/moe_sm120.py",
        "            block_n, block_t, num_warps = 256, 128, 8\n",
        "            block_n, block_t, num_warps = 256, 128, 8\n"
        "        if torch.version.hip:  # 64 KB LDS (sm120 tiles need 99 KB)\n"
        "            block_n, block_t, num_warps = min(block_n, 64), min(block_t, 16), 4\n",
        "if torch.version.hip:  # 64 KB LDS",
        "inkling2 moe_sm120: MXFP8 tiles for 64 KB LDS",
    )
    apply(
        ROOT / "nvidia/ops/attention_sm90.py",
        "        num_warps=8 if block_m >= 128 else 4,\n        num_stages=2,\n",
        "        num_warps=8 if block_m >= 128 else 4,\n        num_stages=1 if torch.version.hip else 2,\n",
        "num_stages=1 if torch.version.hip else 2,",
        "inkling2 rel_attention: 1 stage on ROCm (LDS)",
    )
    apply(
        ROOT / "nvidia/ops/linear.py",
        "        x.shape[0] > LOW_M_MAX_TOKENS\n        or is_flashinfer_autotuning()\n        or not use_sm100_kernels()\n",
        "        x.shape[0] > LOW_M_MAX_TOKENS\n        or not use_sm100_kernels()\n        or is_flashinfer_autotuning()\n",
        "        or not use_sm100_kernels()\n        or is_flashinfer_autotuning()",
        "inkling2 linear: short-circuit before the flashinfer probe",
    )
    apply(
        ROOT / "nvidia/dspark.py",
        "        from flashinfer.norm import fused_add_rmsnorm, rmsnorm\n\n"
        "        if residual is None:\n            return rmsnorm(x, self.weight, self.eps)\n"
        "        # In place: residual += x; x = norm(residual), from the unrounded sum.\n"
        "        fused_add_rmsnorm(x, residual, self.weight, self.eps)\n        return x\n",
        "        # CUDA-only library's rmsnorm / fused-add contract, in torch:\n"
        "        # fp32 norm, bf16 out; with a residual: residual += x (rounded once), x = norm(fp32 sum).\n"
        "        v = x.float() if residual is None else x.float() + residual.float()\n"
        "        if residual is not None:\n            residual.copy_(v)\n"
        "        y = (v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)\n"
        "             * self.weight.float()).to(x.dtype)\n"
        "        if residual is None:\n            return y\n"
        "        x.copy_(y)\n        return x\n",
        "fp32 norm, bf16 out; with a residual",
        "inkling2 dspark draft: torch RMSNorm instead of flashinfer",
    )
    apply(
        ROOT / "nvidia/dspark.py",
        "        impl = cast(FlashAttentionImpl, self.attn.impl)\n"
        "        impl.do_kv_cache_update(self.attn, k, v, self.attn.kv_cache, slot_mapping)\n",
        "        # the CUDA flash backend's cache op is not importable on ROCm: one Triton scatter\n"
        "        # (graph safe; cache = (blocks, heads, page, 2 * head_dim) = K | V, slot < 0 = padding)\n"
        "        from radiance_inkling2_kv import kv_scatter\n\n"
        "        kv_scatter(self.attn.kv_cache, k, v, slot_mapping)\n",
        "from radiance_inkling2_kv import kv_scatter",
        "inkling2 dspark draft: Triton KV scatter instead of the CUDA cache op",
    )


if __name__ == "__main__":
    main()
