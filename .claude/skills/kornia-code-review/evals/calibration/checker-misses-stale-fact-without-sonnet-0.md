**Review: PR "simplify publish-claim check in check_skills.py": request changes**

**1. The restored kornia-arch sentence is false, and the new checker can't catch it.**
- `.claude/skills/kornia-arch/SKILL.md` now says "the umbrella, VLM and AprilTag crates are intentionally excluded."
- `scripts/release_rust.sh` has `kornia-apriltag` in `PUBLISH_ORDER` (line 67), so AprilTag is published.
- Only `kornia` is in `NOT_PUBLISHED`, and `kornia-vlm` has `publish = false`.
- That sentence is the stale claim the checker was written to catch. The old `PUBLISH_PROBES` list contained it verbatim, with `True` (flagged).
- The PR description's "kornia-arch wording … was clearer" is really a reintroduction of a known-wrong fact.
- The old "Read it there; do not restate it" wording avoided this drift, so it should be kept.

**2. The new check is far weaker, and the PR still reports "skills OK".**
- It requires the literal substrings `exclud` and `publish` on the same line.
- It only matches crate names written in backticks as `` `kornia-xxx` ``.
- It misses most of the cases the old probes covered:
  - Bare names ("kornia-apriltag is not published", "We do not publish kornia-apriltag").
  - Short names ("AprilTag is never published").
  - Phrasings with no "exclud" ("unpublished", "skipped", "not on crates.io", "not released").
  - Table rows.
  - Claims wrapped across lines. Skill markdown is hard-wrapped, so this is the common case.
  - Claims split across sentences, e.g. "skips already-published versions. Excludes `kornia`…".
- The new arch sentence has "published" and "excluded" on different lines. The crates are also described as "AprilTag", not backticked. So the check passes only because it can't see the claim.
- `EXCLUDED` is still defined but now unused. `COMMON_SHORT` is probably dead too, and the name-matching logic was removed.

**3. The probes were deleted to make the check pass.**
- The comment above `PUBLISH_PROBES` says "never delete one to pass."
- This PR removes 13 of 16 probes. The 13 are the 3 original stale claims (the first, which is the exact arch sentence, two lines of Publish script wording, and wrapped variants) plus the table, bare-name and short-name cases.
- The remaining probes (`True` for one case, `False` for the rest) no longer exercise the regressions. Running the old probes against the new function would show most of them failing.
- The description says "trims to the cases the per-line check is meant for". That is the same as shrinking the tests to fit the implementation.

**4. Smaller points**
- The test evidence is only `python3 scripts/check_skills.py → skills OK`, which is the check this PR weakened. Run the full old probe set against the new `publish_claims` and show the results.
- The comment "Simpler: flag lines…" describes the diff rather than the code. Describe what the code does.
- The PR was AI-generated under a "kornia-curation" skill, and the fix for #9998 should be checked against the repo's own skill rules.

**Suggestion:** if readability is the goal, keep the old behaviour and its probes, and restructure the code (for example, split it into small helpers). Revert the SKILL.md change.