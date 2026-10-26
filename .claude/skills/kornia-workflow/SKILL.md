---
name: kornia-workflow
description: Use before committing, opening a PR, or debugging CI in kornia-rs — the CI-aligned pixi/cargo commands for format, lint, test, Python and C++ bindings, plus the PR requirements (linked issue, test logs, AI disclosure). Trigger on "run tests", "lint", "before PR", "CI failed".
---

# kornia-rs build / test workflow

All commands are mirrored in CI. Run them through `pixi` (preferred) so the
toolchain matches CI.

## Mandatory pre-PR gate

```bash
pixi run pr-check       # skills-check + rust-fmt-check + rust-clippy + rust-check + rust-test
```

Then self-review: run `kornia-code-review` on `git diff main...HEAD`, fix what
it finds, and paste its final output (with `Verdict:` and `Skill gaps:`) in
the PR's **Self-review** section; `pr-validation.yml` fails while it is empty.
Be ready to explain any line when a reviewer asks.

Equivalent pieces:

```bash
pixi run skills-check                 # validate .claude/skills (scripts/check_skills.py)
pixi run rust-fmt-check               # cargo fmt --all -- --check
pixi run rust-clippy                  # clippy --workspace --no-deps --all-targets --features ci -D warnings
pixi run rust-test                    # cargo test --features ci
pixi run rust-test-package <crate>    # single crate (cargo test -p)
cargo test --doc -p <crate>           # doctests for public docs
```

Conditional gates:

- Python code touched: `pixi run py-test` (builds with maturin first).
- C++ code touched: `pixi run cpp-test` (`cpp-fmt-check` for formatting).
- Cargo.toml version/deps touched: `python3 scripts/check_version_pins.py`
  and `bash scripts/release_rust.sh --check` (also pre-commit hooks).
- New dependency: check advisories (`security_audit.yml` runs in CI).

Without pixi: `cargo test --features ci`, `cargo test -p kornia-imgproc`,
`cargo clippy --workspace --no-deps --all-targets --features ci -- -D warnings`.

## PR rules (checked in review; `pr-validation.yml` fails on a missing Self-review and warns on missing fields)

1. Start the PR body from `.github/pull_request_template.md` and fill every
   section (`gh pr create --body-file`; `--body` text skips the template).
   Linked, approved issue assigned to you: `Fixes #123` / `Closes #123`.
2. Small, one concern, only what the issue asks. Out-of-scope hunks block
   review (`kornia-code-review`); split them into their own PRs.
3. Paste the commands you actually ran and their output, including failures
   and what you did not run.
4. Fill in the AI usage disclosure (see `AI_POLICY.md`): agent + version,
   model, effort level, skills applied, and which part a human did. You are
   the sole responsible author of every line.
5. Perf-sensitive change ⇒ before/after bench numbers (`kornia-code-review`).
6. Changed a fact the skills describe, hit a non-obvious trap, or found something
   reusable (a helper, a measured number, a pattern) ⇒ one line in the owning
   skill (`kornia-curation` §1, §3).
   User-visible change ⇒ add `changelog.d/<PR>.<type>.md` (see
   `changelog.d/README.md`); never edit `CHANGELOG.md` outside a release PR.
7. Reference the existing library implementation (OpenCV, a Rust crate) when
   porting an algorithm; state deliberate departures from it in the rustdoc.
8. Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
9. New/changed public API needs doc comments; bug fixes need a regression test.
10. An agent posting a comment or review on someone's behalf ends it with
    `Posted on behalf of @<handle> by <Agent> (<actual model>).` (not
    `Co-Authored-By:`). The PR body is never committed (`/PR.md` is ignored).

## Debugging CI

Reproduce with the exact job command from `.github/workflows/rust_lint.yml`
or `rust_test.yml`. Clippy warnings are errors.
