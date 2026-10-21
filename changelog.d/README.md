# Changelog fragments

Each PR with a user-visible change adds one file here instead of editing
`CHANGELOG.md`, so PRs no longer conflict on the `[Unreleased]` block.
`scripts/changelog.py` assembles them at release.

## Adding an entry

Name the file `<PR>.<type>.md` with the **PR** number (links point to
`/pull/<PR>`), e.g. `1234.fixed.md`. No PR number yet: use `+short-name.fixed.md`
and rename it before merge (CI rejects `+` names).

| Type | Heading | Use for |
|---|---|---|
| `breaking` | Breaking changes | Incompatible changes; say what changed from and to, and how to migrate |
| `added` | Added | New APIs, features, bindings, user-facing docs |
| `perf` | Performance | Measured speedups or memory savings; quote the number |
| `fixed` | Bug fixes | Corrections to existing behaviour |

Write for users of the crates and the `kornia-rs` wheels: what changed and why
it matters. No heading or outer bullet (the build adds them and the PR link);
inner lists, tables and code blocks are fine. A PR may add one file per type. To
credit several PRs for one change, give each the same text in the same type.

Internal refactors, tests and CI-only changes need no fragment: say so in the
PR and ask a maintainer for the `no-changelog` label.

```bash
pixi run changelog-preview   # print pending notes, changes nothing
```

## Releasing

On the release branch (see the `release` skill):

```bash
python3 scripts/changelog.py build --version X.Y.Z --date YYYY-MM-DD
```

This moves `[Unreleased]` (legacy prose plus rendered fragments) into a dated
section, resets `[Unreleased]` and deletes the consumed fragments. Add the
compare link, review `git diff`, and commit in the release PR under the
`no-changelog` label. Only release PRs edit `CHANGELOG.md`.
