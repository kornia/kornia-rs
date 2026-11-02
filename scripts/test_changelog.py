#!/usr/bin/env python3
"""Tests for the internal-only detection in scripts/changelog.py."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import changelog  # noqa: E402


class TestInternalOnly(unittest.TestCase):
    def assert_internal(self, paths):
        self.assertTrue(
            changelog.internal_only(set(paths)),
            f"expected internal-only: {paths!r}",
        )

    def assert_not_internal(self, paths):
        self.assertFalse(
            changelog.internal_only(set(paths)),
            f"expected NOT internal-only: {paths!r}",
        )

    def test_internal_prefixes(self):
        for p in (
            ".github/workflows/changelog.yml",
            ".claude/skills/kornia-workflow/SKILL.md",
            ".agents/skills/foo/SKILL.md",
            ".devcontainer/devcontainer.json",
            "scripts/changelog.py",
            "ci/run.sh",
        ):
            with self.subTest(path=p):
                self.assert_internal([p])

    def test_internal_root_files(self):
        for p in (
            ".coderabbit.yaml",
            ".gitignore",
            ".gitmodules",
            ".pre-commit-config.yaml",
            "AGENTS.md",
            "AI_POLICY.md",
            "CLAUDE.md",
            "CONTRIBUTING.md",
            "SECURITY.md",
            "best_practices.md",
            "Cross.toml",
            "pixi.toml",
            "pixi.lock",
            "Cargo.lock",
            "devel-aarch64.Dockerfile",
            "devel-i686.Dockerfile",
            "devel-x86_64.Dockerfile",
        ):
            with self.subTest(path=p):
                self.assert_internal([p])

    def test_tests_and_benches_anywhere(self):
        for p in (
            "tests/integration.rs",
            "crates/kornia-imgproc/tests/resize.rs",
            "crates/kornia-imgproc/benches/resize.rs",
            "kornia-py/tests/test_api.py",
        ):
            with self.subTest(path=p):
                self.assert_internal([p])

    def test_pr_1236_ci_only_change(self):
        # The files changed by PR #1236 (ci: check and test kornia-algebra with
        # approx enabled), which previously failed the fragment check.
        self.assert_internal(
            [
                ".claude/skills/kornia-workflow/SKILL.md",
                ".github/workflows/rust_lint.yml",
                ".github/workflows/rust_test.yml",
                "AGENTS.md",
                "pixi.toml",
            ]
        )

    def test_user_visible_paths_are_not_internal(self):
        for p in (
            "crates/kornia-imgproc/src/lib.rs",
            "crates/kornia/Cargo.toml",
            "Cargo.toml",
            "kornia-py/src/lib.rs",
            "kornia-cpp/src/kornia.cpp",
            "examples/resize.rs",
            "docs/guide.md",
            "README.md",
            "README.zh-CN.md",
        ):
            with self.subTest(path=p):
                self.assert_not_internal([p])

    def test_mixed_change_is_not_internal(self):
        self.assert_not_internal(
            ["crates/kornia-imgproc/src/lib.rs", ".github/workflows/ci.yml"]
        )

    def test_empty_change_is_not_internal(self):
        self.assertFalse(changelog.internal_only(set()))


if __name__ == "__main__":
    unittest.main()
