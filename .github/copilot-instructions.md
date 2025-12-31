# Kornia-rs Repository Instructions

This file provides instructions for GitHub Copilot when working with code in this repository.

## Coding Standards

Follow the coding standards and best practices defined in [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards).

## AI-Generated Content Policy

- Code and comments must not be direct, unreviewed outputs of AI agents
- All AI-assisted contributions require human oversight and validation
- Ensure code logic reflects genuine understanding, not copied AI output

## Key Guidelines

- **Code style**: Follow Rust conventions, use rustfmt for formatting, and clippy for linting. Edition 2021, MSRV 1.76.
- **Type system**: Leverage Rust's type system for safety. Use clear, descriptive types. Avoid unnecessary type annotations where inference is clear.
- **Documentation**: Follow rustdoc guidelines (`///` for public items, `//!` for crate-level docs) and match the existing codebase style. See [CONTRIBUTING.md](../CONTRIBUTING.md#coding-standards) for details.
- **Testing**: Write unit tests (in `#[cfg(test)]` modules) and integration tests. Keep tests deterministic and focused.
- **Error handling**: Prefer `Result<T, E>` with descriptive error types (e.g., via `thiserror`). Avoid `.unwrap()`/`.expect()` in library code.
- **Dependencies**: Follow Rust dependency guidelines. Be mindful of transitive dependencies and their impact on build times and binary size.

## Running Checks

```bash
pixi run rust-lint      # Format, clippy, and check
pixi run rust-fmt        # Format Rust code
pixi run rust-clippy     # Run clippy linting
pixi run rust-test       # Run Rust tests
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
- Tests are included for new functionality
- Code passes `pixi run rust-lint` and `pixi run rust-test`
- Error handling uses `Result<T, E>` appropriately
- No `.unwrap()` or `.expect()` in library code (except in tests or where explicitly documented)
