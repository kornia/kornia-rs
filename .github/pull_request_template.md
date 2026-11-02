## What does this pull request do?

<!-- Explain the problem and the change. A few clear sentences are enough. -->

<!-- For user-visible changes, add changelog.d/<PR>.<type>.md (breaking, added, perf, fixed); see changelog.d/README.md.
For internal changes without a fragment, say so and ask for the no-changelog label. -->

## Related issue

<!-- Required: the issue this PR addresses, as `Fixes #<number>`. -->

## What did you check?

<!-- List the tests, benches, or manual checks you ran, and include useful results.
Say what failed or was not run. Perf-sensitive change: before/after `cargo bench` numbers. -->

```text
$ pixi run pr-check
```

## Self-review (required)

<!-- Run the kornia-code-review skill from .claude/skills on `git diff main...HEAD`
(e.g. tell your agent: "self-review my diff with .claude/skills/kornia-code-review"),
fix what it finds, and paste its final output here. The PR check fails while this
section has no verdict. -->
```text
Verdict:

Reuse searches: (library code only)

Skill gaps:
```

## AI usage disclosure

<!-- Exact names, e.g. `Claude Code 2.x`, `claude-opus-5-5`, `high`. No AI used: write
`none` for Agent + version and leave Model, Effort level and Skills applied empty. -->
- **Agent + version:**
- **Model:**
- **Effort level:**
- **Skills applied:** (names from `.claude/skills/`)
- **Human vs AI part:** (e.g. "human diagnosed the bug and reviewed the diff, agent wrote the fix and tests")
