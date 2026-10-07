"""Offline tests for ci/resolve_stack.py: recorded vLLM v0.30.0 / v0.31.0 files, a recorded AMD wheel
index listing (cp312 linux wheels) and a recorded list of transformers releases. No network.

    python3 -m unittest discover -s ci/tests -v
"""
import json
import re
import shutil
import sys
import tempfile
import unittest
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import resolve_stack as rs  # noqa: E402

FIX = HERE / "fixtures"
KNOWN_SHA = "2f745c62aebb33167bfb6d586f569fbea67f8fc5181b03798b91a6b9187a625e"  # triton 3.8.0+git669b31ac.rocm10.1.0


class FixtureSources:
    """Same interface as rs.LiveSources, served from ci/tests/fixtures."""

    def vllm_file(self, ver, path):
        f = FIX / f"vllm-{ver}" / path.replace("/", "__")
        if not f.exists():
            raise rs.Incompatible(f"no fixture for vLLM v{ver} {path}")
        return f.read_text()

    def index_project(self, project):
        f = FIX / "amd-index" / f"{project}.html"
        if not f.exists():
            return []
        return [urllib.parse.unquote(h) for h in re.findall(r'href="([^"]+)"', f.read_text())]

    def pypi_versions(self, pkg):
        return json.loads((FIX / f"pypi-{pkg}-versions.json").read_text())["versions"]

    def wheel_sha256(self, project, filename):
        assert "669b31ac" in filename
        return KNOWN_SHA


SRC = FixtureSources()
V1_0_0 = (FIX / "Dockerfile-1.0.0.args").read_text()
V1_1_0 = (FIX / "Dockerfile-1.1.0.args").read_text()
OVERRIDES_1_0_0 = [
    {"vllm": "0.30.0", "pins": {"TRITON_VERSION": "3.6.0"}, "reason": "PyPI triton on purpose"},
    {"vllm": "0.30.0", "pins": {"AITER_VERSION": "0.1.22.post1"}, "reason": "aiter hold"},
]


class Derivation(unittest.TestCase):
    def test_vllm_0_31_0_on_rocm_10_1_0(self):
        res = rs.resolve("0.31.0", "10.1.0", SRC)
        self.assertEqual(res["pins"], {
            "TORCH_AMD_ROCM": "10.1.0",
            "TORCH_VERSION": "2.13.0",
            "TORCHVISION_VERSION": "0.28.0",
            "TRITON_VERSION": "3.8.0",
            "TRITON_BUILD": "git669b31ac.rocm10.1.0",
            "AITER_VERSION": "0.1.23",
            "TRANSFORMERS_VERSION": "5.17.0",  # newest below vLLM 0.31's `< 5.18.0` (5.18.0 and 5.19.0 exist)
        })
        self.assertEqual(res["warnings"], [])

    def test_vllm_0_30_0_without_overrides_has_no_amd_triton(self):
        with self.assertRaises(rs.Incompatible) as cm:
            rs.resolve("0.30.0", "10.1.0", SRC)
        self.assertIn("f0b55c0", str(cm.exception))

    def test_vllm_0_30_0_with_the_1_0_0_overrides(self):
        res = rs.resolve("0.30.0", "10.1.0", SRC, has_triton_build=False, overrides=OVERRIDES_1_0_0)
        self.assertEqual(res["pins"]["TORCH_VERSION"], "2.12.0")
        self.assertEqual(res["pins"]["TORCHVISION_VERSION"], "0.27.0")  # vLLM says v0.27.1, AMD ships 0.27.0
        self.assertEqual(res["pins"]["TRITON_VERSION"], "3.6.0")
        self.assertEqual(res["pins"]["AITER_VERSION"], "0.1.22.post1")
        self.assertNotIn("TRITON_BUILD", res["pins"])

    def test_override_expires_with_the_vllm_version(self):
        res = rs.resolve("0.31.0", "10.1.0", SRC, overrides=OVERRIDES_1_0_0)
        self.assertEqual(res["active_overrides"], [])
        self.assertEqual(res["pins"]["AITER_VERSION"], "0.1.23")

    def test_unsupported_rocm_major(self):
        with self.assertRaises(rs.Incompatible) as cm:
            rs.resolve("0.31.0", "11.0.0", SRC)
        self.assertIn("major 11", str(cm.exception))

    def test_no_wheel_for_the_rocm_version(self):
        with self.assertRaises(rs.Incompatible) as cm:
            rs.resolve("0.31.0", "10.2.0", SRC)
        self.assertIn("rocm10.2.0", str(cm.exception))

    def test_dockerfile_without_triton_build_cannot_take_amd_triton(self):
        with self.assertRaises(rs.Incompatible) as cm:
            rs.resolve("0.31.0", "10.1.0", SRC, has_triton_build=False)
        self.assertIn("TRITON_BUILD", str(cm.exception))

    def test_unparseable_vllm_pins(self):
        text = (FIX / "vllm-0.31.0" / "docker__Dockerfile.rocm_base").read_text()
        with self.assertRaises(rs.Incompatible):
            rs.parse_rocm_base(text.replace('ARG PYTORCH_BRANCH="733fca1" # release/2.13 as of 09/17',
                                            'ARG PYTORCH_BRANCH="733fca1"'))
        with self.assertRaises(rs.Incompatible):
            rs.parse_rocm_base(text.replace('ARG AITER_BRANCH="v0.1.23"', 'ARG AITER_BRANCH="main"'))
        with self.assertRaises(rs.Incompatible):
            rs.parse_rocm_base(text.replace('ARG TRITON_BRANCH="669b31a" # release/internal/3.8.x as of 09/17',
                                            'ARG TRITON_BRANCH="release/3.8.x"'))

    def test_specifiers(self):
        self.assertTrue(rs.spec_ok(">=5.10.4,<5.18.0", "5.17.0"))
        self.assertFalse(rs.spec_ok(">=5.10.4,<5.18.0", "5.18.0"))
        self.assertTrue(rs.spec_ok(">=5.10.4", "5.19.0"))
        self.assertTrue(rs.spec_ok("!=5.3.*,>=5.0", "5.4.0"))
        self.assertFalse(rs.spec_ok("!=5.3.*,>=5.0", "5.3.2"))
        self.assertEqual(rs.transformers_spec("a\ntransformers >= 5.10.4, < 5.18.0  # c\nb"), ">=5.10.4,<5.18.0")


class Check(unittest.TestCase):
    def run_check(self, args_text, vllm, overrides=None):
        pins = rs.pins_of(args_text)
        res = rs.resolve(vllm, rs.rocm_version(pins["ROCM_BASE"]), SRC, has_triton_build="TRITON_BUILD" in pins,
                         overrides=overrides)
        return rs.check(pins, res, SRC, verify_hash=True)

    def test_1_0_0_is_consistent_with_its_documented_overrides(self):
        self.assertEqual(self.run_check(V1_0_0, "0.30.0", OVERRIDES_1_0_0), [])

    def test_1_1_0_is_consistent_without_overrides(self):
        self.assertEqual(self.run_check(V1_1_0, "0.31.0"), [])

    def test_vllm_bump_without_apply_is_a_mismatch(self):
        stale = V1_1_0.replace("ARG TRANSFORMERS_VERSION=5.17.0", "ARG TRANSFORMERS_VERSION=5.18.0")
        problems = self.run_check(stale, "0.31.0")
        self.assertEqual(len(problems), 1)
        self.assertIn("TRANSFORMERS_VERSION=5.18.0", problems[0])

    def test_transformers_only_has_to_be_in_range(self):
        older = V1_1_0.replace("ARG TRANSFORMERS_VERSION=5.17.0", "ARG TRANSFORMERS_VERSION=5.16.1")
        self.assertEqual(self.run_check(older, "0.31.0"), [])

    def test_each_pin_is_compared(self):
        bad = (V1_1_0.replace("ARG TORCH_VERSION=2.13.0", "ARG TORCH_VERSION=2.12.0")
               .replace("ARG AITER_VERSION=0.1.23", "ARG AITER_VERSION=0.1.22.post1")
               .replace("ARG TRITON_SHA256=2f74", "ARG TRITON_SHA256=0000"))
        problems = "\n".join(self.run_check(bad, "0.31.0"))
        self.assertIn("TORCH_VERSION", problems)
        self.assertIn("AITER_VERSION", problems)
        self.assertIn("TRITON_SHA256", problems)

    def test_torch_amd_rocm_must_follow_the_base_image(self):
        bad = V1_1_0.replace("ARG TORCH_AMD_ROCM=10.1.0", "ARG TORCH_AMD_ROCM=10.0.0")
        self.assertIn("TORCH_AMD_ROCM", "\n".join(self.run_check(bad, "0.31.0")))


class Apply(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # the base: the 1.1.0 ARG block as it was before the vLLM bump (stack of 0.30.0, AMD triton install)
        self.base_df = (V1_1_0.replace("ARG VLLM_VERSION=0.31.0", "ARG VLLM_VERSION=0.30.0")
                        .replace("ARG TORCH_VERSION=2.13.0", "ARG TORCH_VERSION=2.12.0")
                        .replace("ARG TORCHVISION_VERSION=0.28.0", "ARG TORCHVISION_VERSION=0.27.0")
                        .replace("ARG AITER_VERSION=0.1.23", "ARG AITER_VERSION=0.1.22.post1")
                        .replace("ARG TRANSFORMERS_VERSION=5.17.0", "ARG TRANSFORMERS_VERSION=5.18.0"))
        # what Renovate leaves in the working tree: only VLLM_VERSION moved
        (self.tmp / "Dockerfile").write_text(self.base_df.replace("ARG VLLM_VERSION=0.30.0", "ARG VLLM_VERSION=0.31.0"))
        (self.tmp / "VERSION").write_text("1.0.0\n")
        (self.tmp / "CHANGELOG.md").write_text("# Changelog\n\nintro\n\n## [1.0.0] - 2026-10-08\n\nold section\n")

    def run_apply(self):
        res = rs.resolve("0.31.0", "10.1.0", SRC)
        return rs.apply(self.tmp, res, {"dockerfile": self.base_df, "version": "1.0.0"}, "10.1.0", "0.31.0",
                        "2026-10-07", SRC, constraints=False)

    def test_apply_writes_pins_bumps_version_and_changelog(self):
        log = self.run_apply()
        self.assertEqual(rs.pins_of((self.tmp / "Dockerfile").read_text()), rs.pins_of(V1_1_0))
        self.assertEqual((self.tmp / "VERSION").read_text(), "1.1.0\n")
        cl = (self.tmp / "CHANGELOG.md").read_text()
        self.assertIn("## [1.1.0] - 2026-10-07", cl)
        self.assertIn("vLLM 0.30.0 -> 0.31.0", cl)
        self.assertIn("torch 2.12.0 -> 2.13.0", cl)
        self.assertIn("transformers 5.18.0 -> 5.17.0", cl)
        self.assertIn(rs.PENDING, cl)
        self.assertLess(cl.index("## [1.1.0]"), cl.index("## [1.0.0]"))
        self.assertTrue(any("minor bump" in line for line in log))

    def test_apply_is_idempotent(self):
        self.run_apply()
        snap = {p: (self.tmp / p).read_text() for p in ("Dockerfile", "VERSION", "CHANGELOG.md")}
        self.run_apply()
        self.assertEqual(snap, {p: (self.tmp / p).read_text() for p in snap})

    def test_apply_keeps_a_hand_written_changelog_section(self):
        (self.tmp / "CHANGELOG.md").write_text("# C\n\n## [1.1.0] - 2026-10-07\n\nwritten by a human\n\n## [1.0.0] - x\n\nold\n")
        self.run_apply()
        self.assertIn("written by a human", (self.tmp / "CHANGELOG.md").read_text())

    def test_bump_kinds(self):
        self.assertEqual(rs.bump_kind("0.30.0", "0.31.0", "10.1.0", "10.1.0"), "minor")
        self.assertEqual(rs.bump_kind("0.31.0", "0.31.1", "10.1.0", "10.1.0"), "patch")
        self.assertEqual(rs.bump_kind("0.31.0", "0.31.0", "10.1.0", "10.1.1"), "patch")
        self.assertEqual(rs.bump_kind("0.31.0", "0.31.0", "10.1.0", "10.2.0"), "minor")
        self.assertEqual(rs.bump_kind("0.31.0", "0.31.0", "10.1.0", "11.0.0"), "minor")
        self.assertEqual(rs.bump_kind("0.31.0", "0.31.1", "10.1.0", "10.2.0"), "minor")
        self.assertIsNone(rs.bump_kind("0.31.0", "0.31.0", "10.1.0", "10.1.0"))
        self.assertEqual(rs.bump_version("1.0.0", "minor"), "1.1.0")
        self.assertEqual(rs.bump_version("1.1.3", "patch"), "1.1.4")
        self.assertEqual(rs.bump_version("1.1.3", "minor"), "1.2.0")


class RepoFiles(unittest.TestCase):
    def test_override_file_is_well_formed(self):
        for o in rs.load_overrides(HERE.parent / "stack_overrides.json"):
            self.assertTrue(o["reason"].strip())


if __name__ == "__main__":
    unittest.main()
