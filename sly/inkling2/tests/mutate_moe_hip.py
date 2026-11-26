"""Mutation check of the HIP MoE expert GEMM tests (kernel, Python wrapper, vLLM patch hook).

  python3 mutate_moe_hip.py kernel <radiance_inkling2_moe.hip> <outdir>
      writes outdir/m_<name>.hip for the GPU window: build each like the real kernel, put it in
      its own dir as radiance_inkling2_moe_hip.so and run tests/test_moe_hip.py with that dir first
      on PYTHONPATH.  Every mutant must turn a test red.  Only mutations that stay inside the
      allocated buffers are listed (no out-of-bounds access on the shared GPU): the expert-id and
      `q < nq` guards are covered by tests but deliberately not mutated.
  INKLING_VLLM_SRC=<vllm src dir> python3 mutate_moe_hip.py host
      mutates a COPY of the Python wrapper / patch script, runs the CPU tests, prints caught/total.
      Every restore is asserted (byte compare) and the repo files are hashed before and after.

Each mutation is an exact str.replace with a unique anchor (asserted).
"""
import filecmp
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

KERNEL = {
    "even_mag_shift": ("((x & 0x07070707u) << 2) | ((x & 0x08080808u) << 4)",
                       "((x & 0x07070707u) << 1) | ((x & 0x08080808u) << 4)"),
    "even_sign_shift": ("((x & 0x07070707u) << 2) | ((x & 0x08080808u) << 4)",
                        "((x & 0x07070707u) << 2) | ((x & 0x08080808u) << 3)"),
    "odd_mag_shift": ("((x & 0x70707070u) >> 2) | (x & 0x80808080u)",
                      "((x & 0x70707070u) >> 3) | (x & 0x80808080u)"),
    "odd_sign": ("((x & 0x70707070u) >> 2) | (x & 0x80808080u)",
                 "((x & 0x70707070u) >> 2) | ((x & 0x80808080u) >> 1)"),
    "e8m0_zero_exponent": ("e ? (e << 23) : 0x00400000u", "e ? (e << 23) : 0u"),
    "epilogue_2pow12": ("(NVFP4 ? 4096.0f : 1.0f)", "(NVFP4 ? 2048.0f : 1.0f)"),
    "act_scale_tile_stride": ("(row >> 3) * (cols >> 2) + (g >> 2)", "(row >> 3) * (cols >> 3) + (g >> 2)"),
    "weight_scale_byte1": ("__builtin_amdgcn_cvt_f32_fp8(ws[j], 1)", "__builtin_amdgcn_cvt_f32_fp8(ws[j], 0)"),
    "act_scale_second_group": ("acc[r] = fmaf(p1, sw1 * sa.y, acc[r]);", "acc[r] = fmaf(p1, sw1 * sa.x, acc[r]);"),
    "cta_count_guard": ("    if (cta >= tiles[2 * max_ctas]) return;\n", ""),
    "row_end_plus_one": ("int nr = tiles[max_ctas + cta] - cta * MAX_ROWS;",
                         "int nr = tiles[max_ctas + cta] - cta * MAX_ROWS + 1;"),
    "no_row_scale": ("(NVFP4 ? row_scale[row] : 1.0f)", "1.0f"),
    "no_expert_scale": ("const float es = expert_scale[expert] * (NVFP4", "const float es = 1.0f * (NVFP4"),
    "no_dpp_half_mirror": ("    v += __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x141, 0xF, 0xF, true));  // i <-> 7 - i\n", ""),
    "dpp_xor1_is_xor2": ("0xB1, 0xF, 0xF, true));   // xor 1", "0x4E, 0xF, 0xF, true));   // xor 1"),
    "store_lane": ("if (nok && r == li) {", "if (nok && r == (li & 3)) {"),  # rows 4-7 never stored
    "second_pass_rows": ("            LOAD_ACT(r0, prow)\n", "            LOAD_ACT(0, prow)\n"),
    "mx_no_weight_scale": ("fmaf(p * sw, s_sa[(size_t)r * nq + q], acc[r])", "fmaf(p, s_sa[(size_t)r * nq + q], acc[r])"),
    "mx_last_dot": ("p = dot4(w1.w, a1.w, p);", "p = dot4(w1.w, a1.z, p);"),
    "steps_round_down": ("const int nj = (nq + LPR - 1) / LPR;", "const int nj = nq / LPR;"),
}

# (file under sly/inkling2 or sly/, old, new, count, tests to run)
HOST_TESTS = ["tests/test_moe_hip_host.py"]
PATCH_TESTS = ["tests/test_moe_hip_host.py", "tests/test_patch_gfx12.py"]
HOST = {
    "tile_8_to_16": ("inkling2/radiance_inkling2_moe.py", "_HIP_TILE = 8 ", "_HIP_TILE = 16 ", 1, HOST_TESTS),
    "lds_rows_floor": ("inkling2/radiance_inkling2_moe.py", "-(-max_ctas // _TOPK)", "max_ctas // _TOPK", 1, HOST_TESTS),
    "no_k_max": ("inkling2/radiance_inkling2_moe.py", ' and k <= GEOMETRY["k_max"]', "", 1, HOST_TESTS),
    "k_multiple": ("inkling2/radiance_inkling2_moe.py", "k % (64 if nvfp4 else 128) == 0", "k % 64 == 0", 1, HOST_TESTS),
    "no_size_check": ("inkling2/radiance_inkling2_moe.py", "if ext[name] and t.numel()", "if False and t.numel()", 1, HOST_TESTS),
    "extent_last_expert": ("inkling2/radiance_inkling2_moe.py", "wrow = (experts - 1) * n + nn", "wrow = (experts - 2) * n + nn", 1, HOST_TESTS),
    "extent_act_scale": ("inkling2/radiance_inkling2_moe.py", 'ext["act_scale"] = int((off + sb).max())', 'ext["act_scale"] = int(off.max())', 1, HOST_TESTS),
    "no_alignment_check": ("inkling2/radiance_inkling2_moe.py", "if tensors[name].data_ptr() % 16:", "if tensors[name].data_ptr() % 1:", 1, HOST_TESTS),
    "no_dtype_check": ("inkling2/radiance_inkling2_moe.py", "if t.element_size() != _ITEMSIZE[name]:", "if False:", 1, HOST_TESTS),
    "debug_expert_range": ("inkling2/radiance_inkling2_moe.py", "if not 0 <= t[cta] < experts:", "if not 0 <= t[cta] <= experts:", 1, HOST_TESTS),
    "debug_count_range": ("inkling2/radiance_inkling2_moe.py", "if not 0 <= count <= plan.max_ctas:", "if False:", 1, HOST_TESTS),
    "env_knob": ("inkling2/radiance_inkling2_moe.py", 'RADIANCE_INKLING2_HIP_MOE") == "0"', 'RADIANCE_INKLING2_HIP_MOE") == "1"', 1, HOST_TESTS),
    "hip_args_order": ("inkling2/radiance_inkling2_moe.py", "plan.max_ctas, weight.shape[0], n, k, plan.tile,", "plan.max_ctas, n, weight.shape[0], k, plan.tile,", 2, HOST_TESTS),
    "no_hip_when_rocm_gate": ("patch_inkling2_gfx12.py", "        if torch.version.hip:  # gfx12: radiance_inkling2_moe", "        if True:  # gfx12: radiance_inkling2_moe", 1, PATCH_TESTS),
    "hook_drops_tiles": ("patch_inkling2_gfx12.py", '                                     tiles, plan)\\n"', '                                     plan)\\n"', 1, PATCH_TESTS),
    "hook_wrong_scale": ("patch_inkling2_gfx12.py", "return mxfp8_expert_gemm(act, act_scale, weight, weight_scale, ones,", "return mxfp8_expert_gemm(act, act_scale, weight, weight_scale, weight_scale,", 1, PATCH_TESTS),
}


def write_kernel_mutants(src: Path, outdir: Path):
    text = Path(src).read_text()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    for name, (old, new) in KERNEL.items():
        assert text.count(old) == 1, f"mutation anchor not unique: {name}"
        (Path(outdir) / f"m_{name}.hip").write_text(text.replace(old, new, 1))
    return list(KERNEL)


def sha(paths):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_host():
    src = os.environ.get("INKLING_VLLM_SRC")
    assert src, "set INKLING_VLLM_SRC"
    watched = [REPO / "sly/inkling2/radiance_inkling2_moe.py", REPO / "sly/patch_inkling2_gfx12.py"]
    before = sha(watched)
    caught, rows = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copytree(REPO / "sly/inkling2", tmp / "sly/inkling2",
                        ignore=shutil.ignore_patterns("__pycache__", "*.so"))
        shutil.copy(REPO / "sly/patch_inkling2_gfx12.py", tmp / "sly/patch_inkling2_gfx12.py")
        shutil.copy(REPO / "_patchlib.py", tmp / "_patchlib.py")
        pristine = tmp / "pristine"
        pristine.mkdir()
        for rel in {"inkling2/radiance_inkling2_moe.py", "patch_inkling2_gfx12.py"}:
            shutil.copy(tmp / "sly" / rel, pristine / Path(rel).name)

        def pytest(tests):
            return subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *tests],
                                  cwd=tmp / "sly/inkling2", capture_output=True, text=True).returncode

        assert pytest(PATCH_TESTS) == 0, "baseline must be green"
        for name, (rel, old, new, count, tests) in HOST.items():
            target = tmp / "sly" / rel
            text = target.read_text()
            assert text.count(old) == count, f"mutation anchor count != {count}: {name} ({text.count(old)})"
            target.write_text(text.replace(old, new))
            rc = pytest(tests)
            shutil.copy(pristine / target.name, target)
            assert filecmp.cmp(pristine / target.name, target, shallow=False), f"restore failed: {name}"
            rows.append((name, "caught" if rc else "SURVIVED"))
            caught += bool(rc)
            print(f"{name:28s} {'caught' if rc else 'SURVIVED'}", flush=True)
    assert sha(watched) == before, "repo files changed"
    print(f"host mutants: caught {caught}/{len(HOST)}")
    return caught == len(HOST)


if __name__ == "__main__":
    if sys.argv[1] == "kernel":
        print(" ".join(write_kernel_mutants(Path(sys.argv[2]), Path(sys.argv[3]))))
    else:
        sys.exit(0 if run_host() else 1)
