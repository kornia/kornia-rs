#!/usr/bin/env python3
"""Tests for when a stale review eval case fails scripts/check_skills.py."""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

import check_skills  # noqa: E402


class TestEvalSelfReview(unittest.TestCase):
    review = (
        "## Self-review\nVerdict: approve\n"
        "§4 — no finding — checked the diff — [inferred]\n"
        "Reuse searches:\n  $ rg helper crates/\nSkill gaps: none\n"
    )

    def test_accepts_a_library_self_review_and_tooling_without_searches(self):
        self.assertIsNone(check_skills.self_review_error(self.review, library=True))
        tooling = self.review.split("Reuse searches:")[0] + "Skill gaps: none"
        self.assertIsNone(check_skills.self_review_error(tooling, library=False))
        self.assertIsNotNone(check_skills.self_review_error(tooling, library=True))

    def test_rejects_the_same_incomplete_or_unresolved_reviews_as_ci(self):
        for old, new in (
            ("## Self-review", "Self-review:"),
            ("Verdict: approve", "Verdict: pending"),
            ("approve", "request changes"),
            ("Skill gaps:", "Gaps:"),
            ("[inferred]", "<!-- [measured] -->"),
            ("[inferred]", "\n## Tests\n[measured]"),
            ("no finding", "blocking"),
            ("  $ rg helper crates/", "No searches."),
            ("Reuse searches:", "Checks:"),
        ):
            with self.subTest(new=new):
                body = self.review.replace(old, new)
                self.assertIsNotNone(check_skills.self_review_error(body, library=True))


class TestEvalsStrict(unittest.TestCase):
    def strict(self, paths, env=None):
        with mock.patch.dict(os.environ, env or {}, clear=False), \
             mock.patch.object(check_skills.changelog, "changed", return_value=set(paths)):
            if env is None:
                os.environ.pop("SKILLS_STRICT_EVALS", None)
            return check_skills.evals_strict("origin/main")[0]

    def test_a_change_to_the_skills_or_their_checkers_fails_on_a_stale_case(self):
        for p in (
            ".claude/skills/kornia-code-review/evals/clean-empty-image/merged.diff",
            ".claude/skills/kornia-code-review/SKILL.md",
            ".agents/skills/kornia-code-review/SKILL.md",
            "scripts/check_skills.py",
            "scripts/skill_evals.py",
        ):
            self.assertTrue(self.strict({"crates/kornia-imgproc/src/lib.rs", p}), p)

    def test_any_other_change_only_warns(self):
        # The case that motivated this: a rustdoc fix in code a control quotes.
        self.assertFalse(self.strict({"crates/kornia-imgproc/src/distance_transform.rs", "changelog.d/1.fixed.md"}))
        self.assertFalse(self.strict({".github/workflows/ci.yml", "docs/index.md"}))
        self.assertFalse(self.strict(set()), "nothing changed against the base: main itself")

    def test_a_lookalike_path_is_not_the_skills(self):
        self.assertFalse(self.strict({"docs/.claude/skills/x.md", "scripts/check_skills_old.py"}))

    def test_an_unknown_change_set_fails_closed(self):
        err = subprocess.CalledProcessError(128, ["git", "merge-base"])
        with mock.patch.object(check_skills.changelog, "changed", side_effect=err):
            os.environ.pop("SKILLS_STRICT_EVALS", None)
            strict, why = check_skills.evals_strict("origin/main")
        self.assertTrue(strict)
        self.assertIn("origin/main", why)

    def test_the_environment_overrides_the_diff(self):
        self.assertTrue(self.strict({"crates/x.rs"}, {"SKILLS_STRICT_EVALS": "1"}))
        self.assertFalse(self.strict({".claude/skills/x/SKILL.md"}, {"SKILLS_STRICT_EVALS": "0"}))
        self.assertFalse(self.strict({"crates/x.rs"}, {"SKILLS_STRICT_EVALS": ""}), "empty means: decide from the diff")


if __name__ == "__main__":
    unittest.main()
