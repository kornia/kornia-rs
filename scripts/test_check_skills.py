#!/usr/bin/env python3
"""Tests for when a stale review eval case fails scripts/check_skills.py."""
import os
import subprocess
import sys
import re
import tempfile
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
            ("no finding", "blocking"),
            ("  $ rg helper crates/", "No searches."),
            ("Reuse searches:", "Checks:"),
        ):
            with self.subTest(new=new):
                body = self.review.replace(old, new)
                self.assertIsNotNone(check_skills.self_review_error(body, library=True))

    def test_tag_after_self_review_does_not_count(self):
        body = self.review.replace("[inferred]", "") + "\n## Tests\n[measured]"
        self.assertEqual(check_skills.self_review_error(body, library=True), "missing [measured] or [inferred] line")


class TestSelfReviewFreshness(unittest.TestCase):
    RULE = (
        "---\ndescription: Use for a self-review in a fresh session\n---\n\n"
        "- **Self-review (mandatory before opening a PR).** Run in a fresh session"
    )

    def test_repository_docs_pass(self):
        self.assertEqual(check_skills.self_review_doc_errors(), [])

    def test_each_mention_needs_a_fresh_session(self):
        good = "Self-review with `kornia-code-review` in a fresh session."
        old = "| `kornia-code-review` | mandatory self-review before every PR |"
        self.assertFalse(check_skills.missing_fresh_session(good))
        for probe in (
            "Then run the `kornia-code-review` skill on your diff",
            "Run the `kornia-code-review` skill and paste its output in the PR description.",
            "Self review your diff with `kornia-code-review`.",
            "The author's agent runs `kornia-code-review` on the diff.",
            "Run the skill in `.claude/skills/kornia-code-review` and paste its output.",
            "Before opening a PR, run `.claude/skills/kornia-code-review` on your diff.",
            "Use `kornia-code-review` on your diff before opening a PR.",
        ):
            self.assertTrue(check_skills.missing_fresh_session(probe), probe)
        self.assertTrue(check_skills.missing_fresh_session(old))
        self.assertTrue(check_skills.missing_fresh_session(good + "\n\n" + old))
        self.assertTrue(check_skills.missing_fresh_session(good + "\n" + old, per_line=True))

    def test_each_instruction_needs_it_not_just_each_paragraph(self):
        probe = (
            "Self-review with `kornia-code-review` in a fresh session. "
            "Before opening the next PR, run `kornia-code-review` on your diff."
        )
        self.assertTrue(check_skills.missing_fresh_session(probe))

    def test_the_mention_may_span_sentences_only_with_the_phrase(self):
        spans = (
            "Read `.claude/skills/kornia-code-review/SKILL.md`. "
            "Do a self-review before every PR."
        )
        self.assertTrue(check_skills.missing_fresh_session(spans))

    def test_a_qualifier_in_the_next_sentence_counts(self):
        for good in (
            "Self-review with `kornia-code-review`. Do it in a fresh session.",
            "Self-review with `kornia-code-review`, e.g. in a fresh session.",
        ):
            self.assertFalse(check_skills.missing_fresh_session(good), good)

    def test_a_qualifier_in_the_previous_sentence_does_not_count(self):
        # The CodeRabbit case: a compliant first instruction must not excuse a
        # stale second one that follows it.
        probe = "Run the skill in a fresh session. Self-review with `kornia-code-review`."
        self.assertTrue(check_skills.missing_fresh_session(probe))

    def test_rule_needs_it_in_the_description_and_the_bullet(self):
        self.assertFalse(check_skills.self_review_rule_stale(self.RULE))
        no_description = self.RULE.replace("a self-review in a fresh session", "a self-review")
        no_bullet = self.RULE.replace("Run in a fresh session", "The author runs this")
        self.assertTrue(check_skills.self_review_rule_stale(no_description))
        self.assertTrue(check_skills.self_review_rule_stale(no_bullet))

    def test_the_real_rule_without_the_phrase_is_stale(self):
        rule = (check_skills.ROOT / check_skills.SELF_REVIEW_SKILL).read_text(encoding="utf-8")
        reverted = re.sub(r"fresh\s+session", "session", rule)
        self.assertNotEqual(reverted, rule)
        self.assertTrue(check_skills.self_review_rule_stale(reverted))

    def test_errors_are_wired_to_the_docs_and_the_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in check_skills.SELF_REVIEW_DOCS:
                (root / rel).parent.mkdir(parents=True, exist_ok=True)
                (root / rel).write_text("`kornia-code-review` self-review in a fresh session", encoding="utf-8")
            rule = root / check_skills.SELF_REVIEW_SKILL
            rule.parent.mkdir(parents=True, exist_ok=True)
            rule.write_text(self.RULE, encoding="utf-8")
            with mock.patch.object(check_skills, "ROOT", root):
                self.assertEqual(check_skills.self_review_doc_errors(), [])
                (root / "AGENTS.md").write_text("`kornia-code-review` self-review", encoding="utf-8")
                self.assertTrue(any("AGENTS.md" in e for e in check_skills.self_review_doc_errors()))
                rule.write_text("- **Self-review (mandatory before opening a PR).** The author runs this", encoding="utf-8")
                self.assertTrue(any("rule does not require" in e for e in check_skills.self_review_doc_errors()))


class TestLibraryDiff(unittest.TestCase):
    def test_rename_and_copy_check_both_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            diff = Path(tmp) / "patch.diff"
            for operation in ("rename", "copy"):
                for source, target, library in (
                    ("docs/a", "crates/x/b", True),
                    ("kornia-py/src/a", "docs/b", True),
                    ("docs/a", "kornia-cpp/src/b", True),
                    ("docs/a", "docs/b", False),
                ):
                    with self.subTest(operation=operation, source=source, target=target):
                        diff.write_text(
                            f"diff --git a/{source} b/{target}\nsimilarity index 100%\n"
                            f"{operation} from {source}\n{operation} to {target}\n",
                            encoding="utf-8",
                        )
                        self.assertEqual(check_skills.diff_touches_library(diff), library)


class TestEvalsStrict(unittest.TestCase):
    def strict(self, paths, env=None):
        with mock.patch.object(os, "environ", env or {}), \
             mock.patch.object(check_skills.changelog, "changed", return_value=set(paths)):
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
