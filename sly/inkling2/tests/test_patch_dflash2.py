"""sly/patch_inkling2_dflash2.py + sly/inkling2/radiance_inkling2_dflash2.py: vLLM's native DFlash2
speculator on the Inkling-2-Mini target (drafter `incoai/Inkling-2-Mini-DFlash2`).

CPU only. The patch runs against a VERBATIM copy of the three vLLM files it edits, taken from the
shipped image `ink-k9` after its whole patch chain (fixtures/dflash2_ink_k9), so the anchors are checked
against the real text. The helper module is checked against independent reference maths.
"""
import ast
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PATCH = REPO / "sly" / "patch_inkling2_dflash2.py"
FIXTURE = HERE / "fixtures" / "dflash2_ink_k9"
sys.path.insert(0, str(REPO / "sly" / "inkling2"))

import radiance_inkling2_dflash2 as df2  # noqa: E402

EAGLE3 = "vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py"
QWEN3_DF2 = "vllm/model_executor/models/qwen3_dflash2.py"
LOADER = "vllm/v1/worker/gpu/spec_decode/dflash/utils.py"
FILES = (EAGLE3, QWEN3_DF2, LOADER)


@pytest.fixture()
def tree(tmp_path):
    shutil.copytree(FIXTURE, tmp_path, dirs_exist_ok=True)
    (tmp_path / "vllm/models/inkling2").mkdir(parents=True)  # marks a vLLM that carries Inkling 2
    return tmp_path


def run(tree):
    env = dict(os.environ, RADIANCE_PATCH_TARGET_DIR=str(tree), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(PATCH)], capture_output=True, text=True, env=env)


def text(tree, rel):
    return (tree / rel).read_text()


# ---------------------------------------------------------------- the source patch


def test_hooks_land_and_parse(tree):
    r = run(tree)
    assert r.returncode == 0, r.stdout + r.stderr
    for rel in FILES:
        ast.parse(text(tree, rel))
    # captured layer ids: Inkling visits, no DFlash +1 shift
    assert "aux_layer_ids(" in text(tree, EAGLE3)
    assert "[i + 1 for i in (dflash_config.get" not in text(tree, EAGLE3)
    q = text(tree, QWEN3_DF2)
    assert "selector_softcaps(draft_config)" in q  # unary cap / post-correction cap split
    assert "softcap_after(" in q  # softcap on unary + edges
    assert "self.post_softcap = post_softcap" in q and "post_softcap=selector_softcaps(" in q
    assert "candidate_head_cls(vllm_config)(" in q  # target head incl. its MXFP4 dense route
    assert "embed_with_norm(" in q
    assert "attach_embed_norm(draft_inner, target_inner)" in text(tree, LOADER)


def test_softcap_runs_after_the_edge_scores(tree):
    run(tree)
    q = text(tree, QWEN3_DF2)
    fwd = q[q.index("class CandidateSelector") :]
    assert fwd.index("_score_edges(") < fwd.index("softcap_after(")


def test_drafter_mxfp4_hook_lands_after_the_head_wiring(tree):
    run(tree)
    t = text(tree, LOADER)
    ast.parse(t)
    wired, hook = t.index("dflash_model.lm_head = target_lm_head"), t.index("process_dflash_draft(dflash_model)")
    assert wired < hook < t.index("return dflash_model")
    assert "if torch.version.hip:" in t[wired:hook] and "maybe_release_shared_head(dflash_model, target_lm_head)" in t[hook:]


def test_drafter_mxfp4_hook_lands_without_inkling2(tree):  # RADIANCE_DFLASH_MXFP4=1 serves a Qwen target too
    shutil.rmtree(tree / "vllm/models/inkling2")
    r = run(tree)
    assert r.returncode == 0 and "NOOP" in r.stdout
    assert "process_dflash_draft(dflash_model)" in text(tree, LOADER)
    assert "attach_embed_norm" not in text(tree, LOADER)


def test_original_vocab_clamp_is_still_upstream(tree):
    """Candidates index the selector codebooks (draft vocab); the clamp is upstream already."""
    run(tree)
    q = text(tree, QWEN3_DF2)
    assert "min(vllm_config.model_config.get_vocab_size(), self.config.vocab_size)" in q


def test_idempotent(tree):
    run(tree)
    first = {rel: text(tree, rel) for rel in FILES}
    r2 = run(tree)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    assert {rel: text(tree, rel) for rel in FILES} == first
    assert "NOOP" in r2.stdout and "OK " not in r2.stdout


@pytest.mark.parametrize("rel,old", [
    (EAGLE3, "layer_ids = [i + 1 for i in (dflash_config.get"),
    (QWEN3_DF2, "top_k=int(draft_config[\"selector_top_k\"]),"),
    (QWEN3_DF2, "self.candidate_logits_processor = LogitsProcessor("),
    (LOADER, "maybe_share_target_embed(dflash_model, draft_inner, target_inner)"),
])
def test_fatal_on_anchor_drift(tree, rel, old):
    p = tree / rel
    p.write_text(p.read_text().replace(old, old.replace("i + 1", "i + 2").replace("(", "( ", 1)))
    r = run(tree)
    assert r.returncode != 0
    assert "FAIL" in r.stdout + r.stderr


def test_noop_without_inkling2(tree):
    shutil.rmtree(tree / "vllm/models/inkling2")
    r = run(tree)
    assert r.returncode == 0 and "NOOP" in r.stdout
    assert text(tree, EAGLE3) == (FIXTURE / EAGLE3).read_text()


# ---------------------------------------------------------------- capture ids


def spec(model_type, visits=48):
    text_cfg = types.SimpleNamespace(num_layer_visits=visits)
    hf = types.SimpleNamespace(model_type=model_type, text_config=text_cfg)
    return types.SimpleNamespace(target_model_config=types.SimpleNamespace(hf_config=hf))


def test_aux_ids_inkling_are_visits_with_final_norm_last():
    assert df2.aux_layer_ids(spec("inkling2"), [5, 23, 41, 48]) == [5, 23, 41, 48]
    assert df2.aux_layer_ids(spec("inkling2_loop"), [5, 23, 41, 48]) == [5, 23, 41, 48]


def test_aux_ids_other_targets_keep_dflash_shift():
    assert df2.aux_layer_ids(spec("qwen3_5"), [1, 16, 31]) == [2, 17, 32]


@pytest.mark.parametrize("ids", [[], [23, 5, 48], [5, 5, 48], [-1, 48], [5, 23, 41], [5, 23, 41, 49]])
def test_aux_ids_inkling_reject_bad_ids(ids):
    with pytest.raises(ValueError):
        df2.aux_layer_ids(spec("inkling2"), ids)


# ---------------------------------------------------------------- selector softcap


def test_softcap_mode_split():
    cfg = {"final_logit_softcapping": 30.0, "selector_softcap_mode": "after_correction"}
    assert df2.selector_softcaps(cfg) == (None, 30.0)
    cfg["selector_softcap_mode"] = "before_correction"
    assert df2.selector_softcaps(cfg) == (30.0, None)
    assert df2.selector_softcaps({"final_logit_softcapping": 30.0}) == (30.0, None)  # Qwen default
    assert df2.selector_softcaps({}) == (None, None)
    assert df2.selector_softcaps({"final_logit_softcapping": 0.0, "selector_softcap_mode": "after_correction"}) == (None, None)
    with pytest.raises(ValueError):
        df2.selector_softcaps({"selector_softcap_mode": "sideways"})


def reference_scores(fixture_src, unary, hidden, cand, anchor, pred, succ, top_k):
    """The fixture's own pure-torch edge scorer (what the Triton kernel implements)."""
    tree_ = ast.parse(fixture_src)
    fn = next(n for n in tree_.body if isinstance(n, ast.FunctionDef) and n.name == "_score_edges_reference")
    ns = {"torch": torch}
    exec(compile(ast.Module([fn], []), "ref", "exec"), ns)
    return ns["_score_edges_reference"](pred, succ, cand, unary, hidden, anchor, top_k)


def test_softcap_after_correction_matches_reference():
    g = torch.Generator().manual_seed(0)
    b, steps, k, r, vocab, cap = 3, 7, 16, 32, 100, 30.0
    pred, succ = torch.randn(vocab, r, generator=g), torch.randn(vocab, r, generator=g)
    cand = torch.randint(0, vocab, (b, steps, k), generator=g)
    unary = torch.randn(b, steps, k, generator=g) * 40  # large enough that the cap bites
    hidden = torch.randn(b, steps, r, generator=g)
    anchor = torch.randint(0, vocab, (b,), generator=g)
    src = (FIXTURE / QWEN3_DF2).read_text()
    raw = reference_scores(src, unary, hidden, cand, anchor, pred, succ, k)
    # independent after-correction reference: cap * tanh((unary_c + edge_pc) / cap)
    expected = cap * torch.tanh(raw / cap)
    got = df2.softcap_after(raw, cap)
    torch.testing.assert_close(got, expected)
    assert got.abs().max() < cap
    # it is NOT the before-correction rule (cap the unary logit only, then add the edge)
    unary_capped = cap * torch.tanh(unary / cap)
    before = raw - unary.float()[:, :, None, :] + unary_capped[:, :, None, :]
    assert not torch.allclose(got, before, atol=1e-3)
    assert raw.shape == (b, steps, k, k)


def test_softcap_after_without_cap_is_identity():
    x = torch.randn(2, 7, 16, 16)
    assert df2.softcap_after(x, None) is x


# ---------------------------------------------------------------- head class + embedding


def test_candidate_head_cls_follows_the_target(monkeypatch):
    base, ink = type("LogitsProcessor", (), {}), type("Inkling2LogitsProcessor", (), {})
    for name, attr, cls in (
        ("vllm.model_executor.layers.logits_processor", "LogitsProcessor", base),
        ("vllm.models.inkling2.nvidia.model", "Inkling2LogitsProcessor", ink),
    ):
        mod = types.ModuleType(name)
        setattr(mod, attr, cls)
        monkeypatch.setitem(sys.modules, name, mod)
    assert df2.candidate_head_cls(types.SimpleNamespace(speculative_config=spec("inkling2"))) is ink
    assert df2.candidate_head_cls(types.SimpleNamespace(speculative_config=spec("qwen3_5"))) is base


def test_embed_norm_attach_and_apply(monkeypatch):
    norm = torch.nn.RMSNorm(4)
    target = types.SimpleNamespace(embed_norm=norm)
    draft = torch.nn.Module()
    monkeypatch.delenv("RADIANCE_INKLING2_DFLASH2_EMBED_NORM", raising=False)
    df2.attach_embed_norm(draft, target)
    assert draft.embed_norm is norm  # shared, not copied
    x = torch.randn(3, 4) * 7
    torch.testing.assert_close(df2.embed_with_norm(draft, x), norm(x))
    # knob off: raw target embedding
    plain = torch.nn.Module()
    monkeypatch.setenv("RADIANCE_INKLING2_DFLASH2_EMBED_NORM", "0")
    df2.attach_embed_norm(plain, target)
    assert df2.embed_with_norm(plain, x) is x
    # a target without embed_norm (Qwen) is left alone
    other = torch.nn.Module()
    monkeypatch.delenv("RADIANCE_INKLING2_DFLASH2_EMBED_NORM")
    df2.attach_embed_norm(other, types.SimpleNamespace())
    assert df2.embed_with_norm(other, x) is x
