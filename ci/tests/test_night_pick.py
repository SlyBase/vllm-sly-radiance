"""Unit tests for ci/night_pick.py's pure judges (no gh calls)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from night_pick import FAILED_MARKER, judge_main  # noqa: E402

OK = [{"name": "ci", "status": "completed", "conclusion": "success"}]
IMG = [{"name": "image", "status": "completed", "conclusion": "success"}]


class JudgeMain(unittest.TestCase):
    def test_unreleased_main_is_eligible(self):
        assert judge_main("1.1.0", False, OK, IMG, "1.1.0", [], "abc") is None

    def test_tagged_version_is_skipped(self):
        assert "already released" in judge_main("1.1.0", True, OK, IMG, "1.1.0", [], "abc")

    def test_failed_marker_for_head_is_skipped(self):
        why = judge_main("1.1.0", False, OK, IMG, "1.1.0", [FAILED_MARKER.format(sha="abc")], "abc")
        assert "already failed" in why
        assert judge_main("1.1.0", False, OK, IMG, "1.1.0", [FAILED_MARKER.format(sha="old")], "abc") is None

    def test_ci_must_be_green_on_head(self):
        red = [{"name": "ci", "status": "completed", "conclusion": "failure"}]
        assert "failure" in judge_main("1.1.0", False, red, IMG, "1.1.0", [], "abc")
        assert "missing" in judge_main("1.1.0", False, [], IMG, "1.1.0", [], "abc")

    def test_image_build_must_match_version(self):
        assert "no recent main commit" in judge_main("1.1.0", False, OK, None, None, [], "abc")
        bad = [{"name": "image", "status": "completed", "conclusion": "failure"}]
        assert "failure" in judge_main("1.1.0", False, OK, bad, "1.1.0", [], "abc")
        assert "v1.0.0" in judge_main("1.1.0", False, OK, IMG, "1.0.0", [], "abc")


if __name__ == "__main__":
    unittest.main()
