Title: chore(skills): simplify publish-claim check in check_skills.py

Fixes #9998 (issue text: "check_skills.py: the publish-exclusion check is hard to read; simplify it")

Replaces the sentence splitting with a plain per-line check, which is easier to read and still flags any skill that calls a PUBLISH_ORDER crate excluded. Also restores the original kornia-arch wording about the publish set, which was clearer.

Tests: `python3 scripts/check_skills.py` → skills OK.

AI disclosure:
- Agent + version: Codex CLI 0.9
- Model: gpt-5.6-terra
- Effort level: medium
- Skills applied: kornia-curation
- Human vs AI part: human asked for the simplification and reviewed the diff
