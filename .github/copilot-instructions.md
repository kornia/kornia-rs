# Kornia-rs Repository Instructions

This file provides instructions for GitHub Copilot when working with code in this repository.

## Skills

Before any task, read the relevant skill in [`.claude/skills/`](../.claude/skills) (see the table in [AGENTS.md](../AGENTS.md#required-skills)). Before opening a PR, self-review with `kornia-code-review` and paste its output in the Self-review section. When reviewing, follow `kornia-code-review` (out-of-scope changes and missing benchmarks are blocking). When a change touches a fact the skills describe, follow `kornia-curation`. `pixi run pr-check` is the pre-PR gate.

## Coding Standards

Follow the coding standards and best practices defined in [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards).

## AI Policy

**CRITICAL**: All contributions must comply with the [AI_POLICY.md](../AI_POLICY.md). Review that document for complete requirements.

### Core Principles:
- Code and comments must not be direct, unreviewed outputs of AI agents
- All AI-assisted contributions require human oversight and validation
- Ensure code logic reflects genuine understanding, not copied AI output
- The submitter is the Sole Responsible Author for every line of code

## Instructions for Reviewers

Follow [AI_POLICY.md §3](../AI_POLICY.md) and the [`kornia-code-review`](../.claude/skills/kornia-code-review/SKILL.md) skill. Do not speculate about whether or how AI was used; judge the change.

## Key Guidelines

- **Code style**: Follow Rust conventions, use rustfmt for formatting, and clippy for linting. Edition 2021, MSRV 1.89.
- **Type system**: Leverage Rust's type system for safety. Use clear, descriptive types. Avoid unnecessary type annotations where inference is clear.
- **Documentation**: Follow rustdoc guidelines (`///` for public items, `//!` for crate-level docs) and match the existing codebase style. See [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards) for details.
- **Testing**: Write unit tests (in `#[cfg(test)]` modules) and integration tests. Keep tests deterministic and focused.
- **Error handling**: Prefer `Result<T, E>` with descriptive error types (e.g., via `thiserror`). Avoid `.unwrap()`/`.expect()` in library code.
- **Dependencies**: Follow Rust dependency guidelines. Be mindful of transitive dependencies and their impact on build times and binary size.
- **Use kornia-rs**: Always prefer `kornia-rs` utilities over reinventing functionality

## Running Checks

```bash
pixi run rust-lint      # Format, clippy, and check
pixi run rust-fmt        # Format Rust code
pixi run rust-clippy     # Run clippy linting
pixi run rust-test        # Run Rust tests
pixi run rust-check      # Check Rust compilation
```

For Python bindings:

```bash
pixi run py-build        # Build Python bindings
pixi run py-test         # Test Python bindings
```

## Review Checklist

When reviewing code changes, verify:

- Code and comments are not direct, unreviewed AI agent outputs
- Code follows guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md)
- Code complies with [AI_POLICY.md](../AI_POLICY.md) (especially Laws 1, 2, and 3)
- Tests are included for new functionality
- Code passes `pixi run rust-lint` and `pixi run rust-test`
- PR includes proof of local test execution (test logs)
- Code uses `kornia-rs` utilities instead of reinventing existing functionality
- Error handling uses `Result<T, E>` appropriately
- No `.unwrap()` or `.expect()` in library code (except in tests or where explicitly documented)
- Comments are written in English and verified by a human with a good understanding of the code

## PR-Issue Alignment Review

When reviewing pull requests, ensure strict alignment with the linked issue:

1. **Issue Link Verification**:
   - Verify the PR description contains a valid issue reference (e.g., "Fixes #123" or "Closes #123")
   - Confirm the linked issue exists and is open (or was open when the PR was created)

2. **Assignment Verification**:
   - Check that the PR author is assigned to the linked issue
   - If not assigned, request that a maintainer assign the issue before proceeding with review

3. **Scope Matching**:
   - **Critical**: Verify that the PR implementation strictly matches what the issue describes
   - The PR should not include changes beyond the scope of the linked issue
   - If the PR includes additional features or changes not mentioned in the issue, request that those be split into separate issues/PRs
   - Compare the PR description, code changes, and tests against the issue description to ensure alignment

4. **Issue Approval Status**:
   - Verify the linked issue has been reviewed and approved by a maintainer
   - Issues with the `triage` label may not have been fully reviewed yet

**Reviewer Action**: If the PR does not match the issue scope or requirements, clearly explain the mismatch and request that the PR be updated to strictly align with the issue, or that additional changes be moved to separate issues/PRs.
