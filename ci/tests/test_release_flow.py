"""Tests for the collect-then-release flow: [Unreleased] changelog section, version rule, cut_release."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_consistency import version_rule  # noqa: E402
from cut_release import cut  # noqa: E402

IMG = ["sly/patch_x.py"]
LOG = "# Changelog\n\n## [Unreleased]\n\n### Fixed\n- a fix\n\n## [1.0.0] - 2026-01-01\n\nold\n"


class VersionRule(unittest.TestCase):
    def test_no_image_files_is_fine(self):
        assert version_rule("1.0.0", "1.0.0", [], False, True) is None

    def test_same_version_with_unreleased_entry_is_fine(self):
        assert version_rule("1.0.0", "1.0.0", IMG, True, True) is None

    def test_same_version_without_unreleased_entry_fails_when_tagged(self):
        assert "VERSION is still 1.0.0" in version_rule("1.0.0", "1.0.0", IMG, False, True)

    def test_same_version_untagged_frame_is_fine(self):
        assert version_rule("1.0.0", "1.0.0", IMG, False, False) is None

    def test_bump_is_fine(self):
        assert version_rule("1.0.0", "1.1.0", IMG, False, True) is None

    def test_bump_with_leftover_unreleased_fails(self):
        assert "[Unreleased]" in version_rule("1.0.0", "1.1.0", IMG, True, True)


class CutRelease(unittest.TestCase):
    def test_renames_heading_and_dates_it(self):
        out = cut(LOG, "1.1.0", "2026-10-28")
        assert "## [1.1.0] - 2026-10-28\n\n### Fixed" in out
        assert "[Unreleased]" not in out
        assert out.index("[1.1.0]") < out.index("[1.0.0]")

    def test_empty_unreleased_is_an_error(self):
        with self.assertRaises(SystemExit):
            cut("## [Unreleased]\n\n## [1.0.0] - x\n\nold\n", "1.1.0", "d")

    def test_missing_unreleased_is_an_error(self):
        with self.assertRaises(SystemExit):
            cut("## [1.0.0] - x\n\nold\n", "1.1.0", "d")


if __name__ == "__main__":
    unittest.main()
