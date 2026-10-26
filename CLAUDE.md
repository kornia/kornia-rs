@AGENTS.md

The operational guide is [AGENTS.md](AGENTS.md); the line above imports it for Claude Code.

- Read the relevant skills in `.claude/skills/` before starting a task.
- Start from the smallest relevant code path; run focused checks before broad claims.
- Say what you verified and what remains unverified.
- Before opening a PR: run `pixi run pr-check`, then self-review with `.claude/skills/kornia-code-review` and paste its output in the PR's Self-review section.
