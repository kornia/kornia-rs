---
description: "A 'simplified' publish check that lets the stale AprilTag fact back in."
runs: 1
max_turns: 80
timeout_seconds: 1200
allowed_tools: [Read, Glob, Grep, Bash, Skill]
---

Review this kornia-rs pull request. The working directory is the repository
with the PR applied. The PR title and description are in `PR.md`; the diff is
`pr.diff`. There is no network and no `gh`. Do not run cargo,
pixi or any build.
Your final message is the review and nothing else.
