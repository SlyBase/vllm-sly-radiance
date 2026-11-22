"""ci/check_new_names.py: a patch must not introduce a name its function cannot resolve.

The anchor of a hunk can survive an upstream refactor while a local variable the hunk reads is gone
(patch_adaptive_width read `max_decode_tokens` after vLLM main dropped it: dry run green, every graph
start died with NameError). Run: python3 -m unittest ci/tests/test_check_new_names.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import check_new_names as c  # noqa: E402

PRISTINE = """import os
def f(self):
    a = 1
    return a + len(os.sep)
"""


class NewNames(unittest.TestCase):
    def test_hunk_reading_a_removed_local_is_reported(self):
        patched = PRISTINE.replace("    return", "    g(max_decode_tokens)\n    return")
        patched += "def g(x):\n    return x\n"
        self.assertEqual(c.new_undefined(PRISTINE, patched), {("f", "max_decode_tokens")})

    def test_names_bound_locally_globally_or_builtin_pass(self):
        patched = PRISTINE.replace("    return", "    b = a\n    print(b, os.name)\n    return")
        self.assertEqual(c.new_undefined(PRISTINE, patched), set())

    def test_preexisting_undefined_names_are_not_blamed_on_the_patch(self):
        base = PRISTINE.replace("    return", "    upstream_dyn\n    return")
        patched = base.replace("    return", "    b = 2\n    return")
        self.assertEqual(c.new_undefined(base, patched), set())

    def test_nested_function_sees_enclosing_locals(self):
        patched = PRISTINE.replace("    return", "    def h():\n        return a\n    return")
        self.assertEqual(c.new_undefined(PRISTINE, patched), set())

    def test_tree_mode_compares_changed_files_only(self):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d, "src"), Path(d, "dst")
            for root in (src, dst):
                (root / "pkg").mkdir(parents=True)
                (root / "pkg/same.py").write_text("def k():\n    return undefined_upstream\n")
            (src / "pkg/m.py").write_text(PRISTINE)
            (dst / "pkg/m.py").write_text(PRISTINE.replace("    return", "    zz\n    return"))
            self.assertEqual(c.check_trees(src, dst), [("pkg/m.py", "f", "zz")])


if __name__ == "__main__":
    unittest.main()
