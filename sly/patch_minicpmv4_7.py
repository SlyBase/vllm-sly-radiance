#!/usr/bin/env python3
"""MiniCPM-V 4.7 model support, backported from vllm-project/vllm#58674.

0.31.0 predates #58674 (merged 2026-10-11, into main), so `MiniCPMV4_7ForConditionalGeneration`
is not in the registry and the model file does not exist. The PR ships four production files:
`minicpmv4_7.py` (new), a `minicpmv4_6.py` refactor (+272/-63) that makes 4.6 inheritable and
adds the `_scoped_mm_kwarg` / two-arg `_get_mm_fields_config` helpers 4.7 imports, the registry
entry, and a tool-calling/reasoning parser.

This patch carries the three files the model LOADS (4.7, the 4.6 refactor, the registry entry)
and skips the parser/reasoning/chat-template files: they are only reached when a request enables
tool calling or reasoning, not when the model is loaded and served.

The 4.6 file is replaced wholesale, not hunked: the refactor restructures the multi-modal
processor and 4.7's `super()` chain (get_hf_processor, get_image_max_slice_num,
_get_mm_fields_config) assumes the new signatures. Hunking a 63-line deletion into 0.31.0's
file is more fragile than shipping the upstream post-merge file, which is byte-identical to
what a fresh 0.31.0+#58674 install would have.

transformers note: 0.31.0 pins `transformers<5.18`. The model file does NOT import a
`MiniCPMV4_7Config` from transformers (it reads `config.json` fields directly and inherits the
4.6 weight loader), so the model loads on 5.17. Only the HF `AutoProcessor` for
`model_type: minicpmv4_7` needs 5.18 -- raise the pin and `pip install -U transformers` in the
container before serving; the pin is metadata, not a hard import guard.

Gated on RADIANCE_MINICPMV4_7=1 so the production Qwen3.8 image is untouched when the knob is
off (the file copy and registry entry are then skipped).
"""
import os
import sysconfig
from pathlib import Path

from _patchlib import apply

SP = Path(sysconfig.get_paths()["purelib"])
MODELS = SP / "vllm/model_executor/models"
REG = MODELS / "registry.py"
SRC = Path(__file__).parent / "minicpmv4_7"

ENABLED = os.environ.get("RADIANCE_MINICPMV4_7", "0") == "1"


def _install_file(dst: Path, src_name: str, sentinel: str, label: str) -> None:
    """Copy the asset over `dst`, idempotently: skip when `sentinel` is already in the file."""
    if dst.exists() and sentinel in dst.read_text():
        print(f"  NOOP  {label} already applied")
        return
    data = (SRC / src_name).read_text()
    dst.write_text(data)
    import ast
    ast.parse(data)
    print(f"  OK    {label}")


REG_OLD = (
    '    "MiniCPMV4_6ForConditionalGeneration": (\n'
    '        "minicpmv4_6",\n'
    '        "MiniCPMV4_6ForConditionalGeneration",\n'
    '    ),\n'
)
REG_NEW = (
    '    "MiniCPMV4_6ForConditionalGeneration": (\n'
    '        "minicpmv4_6",\n'
    '        "MiniCPMV4_6ForConditionalGeneration",\n'
    '    ),\n'
    '    "MiniCPMV4_7ForConditionalGeneration": (\n'
    '        "minicpmv4_7",\n'
    '        "MiniCPMV4_7ForConditionalGeneration",\n'
    '    ),\n'
)


def main() -> None:
    if not ENABLED:
        print("  SKIP  minicpmv4_7: RADIANCE_MINICPMV4_7=0")
        return
    # vLLM 0.32.0 contains vllm-project/vllm#58674 natively (registry entry, minicpmv4_7.py and
    # the minicpmv4_6.py refactor). The shipped copies here are the 0.31-era files -- installing
    # them over 0.32+'s own would clobber upstream's newer model code, so the patch stays off the
    # loop from 1.7.0 and refuses to run on a tree that already carries the model natively.
    if (MODELS / "minicpmv4_7.py").exists():
        print("  SKIP  minicpmv4_7: native in this vLLM (0.32.0+); the 0.31-era copies are not installed")
        return
    _install_file(MODELS / "minicpmv4_6.py", "minicpmv4_6.py",
                  "_scoped_mm_kwarg", "minicpmv4_7: 4.6 refactor")
    _install_file(MODELS / "minicpmv4_7.py", "minicpmv4_7.py",
                  "canvas 3D M-RoPE", "minicpmv4_7: model file")
    apply(REG, REG_OLD, REG_NEW, "MiniCPMV4_7ForConditionalGeneration",
          "minicpmv4_7: registry entry")


if __name__ == "__main__":
    main()
