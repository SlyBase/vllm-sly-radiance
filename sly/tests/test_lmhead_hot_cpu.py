"""CPU tests for the lm_head hot rows (RADIANCE_LMHEAD_HOT in sly/mxfp4/radiance_lmhead_int4.py).

Plain torch only: no vllm, no GPU. The int4 GEMV itself (torch.ops.vllm.rdna_hybrid_w4a16_apply) is
replaced by a plain matmul stand-in, so these tests pin the part this feature adds: row-file
validation, the fp8/bf16 row storage, and the scatter over the int4 logits.

Run:  python3 -m pytest sly/tests/test_lmhead_hot_cpu.py -q
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mxfp4"))
for _k in ("RADIANCE_LMHEAD_HOT", "RADIANCE_LMHEAD_HOT_FMT", "RADIANCE_LMHEAD_INT4_CODES"):
    os.environ.pop(_k, None)

import radiance_lmhead_int4 as L

N, K = 96, 256


class FakeLayer(torch.nn.Module):
    tp_size = 1


def fp8_rows_ref(x):
    """The rule of fq_eval.fp8_rows (the simulation the 14 % KLD gain was measured with)."""
    x = x.float()
    s = (x.abs().amax(1, keepdim=True) / 448.0).clamp(min=1.0 / (448.0 * 512.0))
    return ((x / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s).to(torch.bfloat16)


@pytest.fixture
def head():
    torch.manual_seed(0)
    return (torch.randn(N, K) * 0.05).to(torch.bfloat16)


@pytest.fixture
def x():
    torch.manual_seed(1)
    return torch.randn(3, K).to(torch.bfloat16)


def rows_file(tmp_path, rows):
    p = tmp_path / "hot.json"
    p.write_text(json.dumps(rows))
    return str(p)


def test_unset_changes_nothing(head, x):
    layer = FakeLayer()
    L.attach_hot(layer, head)  # env unset: no attributes, no memory
    assert not hasattr(layer, "radiance_hot")
    out = x.float() @ head.float().T
    out = out.to(torch.bfloat16)
    before = out.clone()
    assert L.apply_hot(layer, x, out) is out
    assert torch.equal(out, before)


@pytest.mark.parametrize("fmt", ["bf16", "fp8"])
def test_listed_rows_match_reference_others_untouched(tmp_path, head, x, fmt):
    rows = [5, 17, 18, 95, 0]
    layer = FakeLayer()
    L.attach_hot(layer, head, rows_file(tmp_path, rows), fmt)
    int4_logits = (torch.randn(3, N) * 3).to(torch.bfloat16)  # stand-in for the int4 GEMV result
    out = int4_logits.clone()
    L.apply_hot(layer, x, out)

    idx = torch.tensor(rows)
    ref_w = head[idx] if fmt == "bf16" else fp8_rows_ref(head[idx])
    ref = x.float() @ ref_w.float().T
    assert torch.allclose(out[:, idx].float(), ref, rtol=2e-2, atol=2e-2)
    if fmt == "bf16":
        assert torch.equal(out[:, idx], (x @ ref_w.T))
    other = torch.ones(N, dtype=torch.bool)
    other[idx] = False
    assert torch.equal(out[:, other], int4_logits[:, other])  # bit-identical


@pytest.mark.parametrize("draft_on", [False, True])
def test_drafter_head_call_skips_hot_rows_unless_enabled(tmp_path, head, x, monkeypatch, draft_on):
    """The DFlash2 drafter shares the target head and calls it via LogitsProcessor.get_top_k_tokens."""
    monkeypatch.setattr(L, "HOT_DRAFT", draft_on)
    layer = FakeLayer()
    L.attach_hot(layer, head, rows_file(tmp_path, [5, 17]), "bf16")
    int4_logits = (torch.randn(3, N) * 3).to(torch.bfloat16)

    def head_call(out):  # stand-in for get_top_k_tokens -> _apply_head -> quant_method.apply
        return L.apply_hot(layer, x, out)

    drafted = L.draft_head(head_call)(int4_logits.clone())
    assert torch.equal(drafted, int4_logits) is (not draft_on)
    target = head_call(int4_logits.clone())  # the target path keeps the hot rows either way
    assert not torch.equal(target, int4_logits)


def test_drafter_flag_reset_after_error(tmp_path, head, x, monkeypatch):
    monkeypatch.setattr(L, "HOT_DRAFT", False)
    layer = FakeLayer()
    L.attach_hot(layer, head, rows_file(tmp_path, [5]), "bf16")

    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        L.draft_head(boom)()
    out = (torch.randn(3, N) * 3).to(torch.bfloat16)
    assert not torch.equal(L.apply_hot(layer, x, out.clone()), out)


def test_fp8_rows_storage_is_fp8_plus_row_scale(tmp_path, head):
    layer = FakeLayer()
    L.attach_hot(layer, head, rows_file(tmp_path, [1, 2, 3]), "fp8")
    _idx, w, s = layer.radiance_hot
    assert w.dtype == torch.float8_e4m3fn and s.dtype == torch.float32 and s.shape == (3,)
    deq = (w.float() * s[:, None]).to(torch.bfloat16)
    assert torch.equal(deq, fp8_rows_ref(head[[1, 2, 3]]))


def test_rows_sorted_and_deduplicated_input_refused(tmp_path, head):
    layer = FakeLayer()
    L.attach_hot(layer, head, rows_file(tmp_path, [9, 3, 7]), "bf16")
    assert layer.radiance_hot[0].tolist() == [3, 7, 9]  # order of the file does not matter
    with pytest.raises(RuntimeError, match="duplicate"):
        L.attach_hot(FakeLayer(), head, rows_file(tmp_path, [3, 3]), "bf16")


@pytest.mark.parametrize("bad", [[], [-1, 2], [0, N], [1.5], ["a"], {"rows": [1]}, [True], "x"])
def test_bad_row_files_refused(tmp_path, head, bad):
    with pytest.raises(RuntimeError, match="RADIANCE_LMHEAD_HOT"):
        L.attach_hot(FakeLayer(), head, rows_file(tmp_path, bad), "fp8")


def test_missing_file_and_bad_fmt_and_tp_refused(tmp_path, head):
    with pytest.raises(RuntimeError, match="RADIANCE_LMHEAD_HOT"):
        L.attach_hot(FakeLayer(), head, str(tmp_path / "nope.json"), "fp8")
    with pytest.raises(RuntimeError, match="FMT"):
        L.attach_hot(FakeLayer(), head, rows_file(tmp_path, [1]), "int8")
    tp2 = FakeLayer()
    tp2.tp_size = 2
    with pytest.raises(RuntimeError, match="tensor parallel"):
        L.attach_hot(tp2, head, rows_file(tmp_path, [1]), "fp8")
