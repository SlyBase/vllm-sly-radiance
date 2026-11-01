"""CPU test for ci/check_attn_metadata_contract.py: the pre-PR#183 backends read
`attn_metadata.use_cascade`, which vLLM 0.31's RocmAttentionMetadata no longer has -- the checker must
flag exactly that, and the current files must pass. Pure Python (ast), no torch/vllm."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci"))

import check_attn_metadata_contract as C  # noqa: E402

# a RocmAttentionMetadata WITHOUT use_cascade, as in the pinned vLLM
META = '''
from dataclasses import dataclass
@dataclass
class RocmAttentionMetadata:
    num_actual_tokens: int
    max_query_len: int
    query_start_loc: object
    max_seq_len: int
    seq_lens: object
    block_table: object
    slot_mapping: object
    scheduler_metadata: object = None
    causal: bool = True
    @property
    def n(self): return 1
'''
FILES = ["sly/kv4/radiance_kv4h_attn.py", "sly/kv4/radiance_kv4_attn.py"]


def _git(rev, path):
    r = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        pytest.skip("git history unavailable")
    return r.stdout


@pytest.mark.parametrize("path", FILES)
def test_prefix_file_fails_on_use_cascade(path):
    found = C.scan(_git("96f39ee", path), path, C.index([META]))
    assert any("use_cascade" in f and f.startswith(path + ":") for f in found), found


@pytest.mark.parametrize("path", FILES)
def test_current_file_passes(path):
    assert C.scan((ROOT / path).read_text(), path, C.index([META])) == []


def test_guarded_reads_and_assigned_attrs_pass():
    src = ("def f(attn_metadata: RocmAttentionMetadata):\n"
           "    getattr(attn_metadata, 'use_cascade', False)\n"
           "    attn_metadata.hyb = 1\n    return attn_metadata.hyb\n")
    assert C.scan(src, "x.py", C.index([META])) == []
