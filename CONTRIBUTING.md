# Contributing to kornia-rs

Thank you for your interest in contributing! This document explains how to set up your environment, the coding standards we follow, and the checks you should run locally before opening a pull request.

This project is part of the Kornia ecosystem. Join the community on [Discord](https://discord.gg/HfnywwpBnD) to discuss features and get help.

## Policies and Guidelines

- **AI-Assisted Development**: AI tools (e.g., GitHub Copilot, ChatGPT, Cursor) may be used to assist with coding, but all contributions must be authored and reviewed by humans. PRs that are fully AI-generated without human understanding, oversight, or ability to explain the code will be rejected. Contributors must understand all code they submit and be able to respond to review feedback.
  - **Automated Detection**: Automated review systems (including GitHub Copilot) analyze PRs for AI-generated content. PRs flagged as having excessive AI-generated content without sufficient human authorship will be rejected. Ensure your contributions reflect genuine human understanding and modification of any AI-assisted code.
- **15-Day Rule**: PRs with no activity for 15+ days will be automatically closed.
- **Transparency**: All discussions must be public.

We're all volunteers. These policies help us focus on high-impact work.

## Ways to Contribute

1. **Ask/Answer questions:**
   - [GitHub Discussions](https://github.com/kornia/kornia-rs/discussions)
   - [Discord](https://discord.gg/HfnywwpBnD)
   - Don't use GitHub issues for Q&A.

2. **Report bugs** via [GitHub issues](https://github.com/kornia/kornia-rs/issues):
   - Search for existing issues first.
   - Use the bug report template.
   - Include: clear description, reproduction steps, toolchain versions, and code sample.

3. **Fix bugs or add features:**
   - Check [help wanted issues](https://github.com/kornia/kornia-rs/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22help%20wanted%22) for starting points.
   - **PRs must be linked to an issue** (use "Closes #123" or "Fixes #123").
   - Follow the [development setup](#developing-kornia-rs) below.
   - Run local tests before submitting.

4. **Donate resources:**
   - [Open Collective](https://opencollective.com/kornia)
   - [GitHub Sponsors](https://github.com/sponsors/kornia)

# Developing kornia-rs

## Setup

1. **Fork** the [repository](https://github.com/kornia/kornia-rs/fork)

2. **Clone your fork** and add upstream:
   ```bash
   $ git clone git@github.com:<your Github username>/kornia-rs.git
   $ cd kornia-rs
   $ git remote add upstream https://github.com/kornia/kornia-rs.git
   ```

3. **Create a branch** (don't work on `main`):
   ```bash
   git checkout upstream/main -b feat/foo_feature
   # or
   git checkout upstream/main -b fix/bar_bug
   ```

4. **Development environment**

   We use [pixi](https://pixi.sh) for package and environment management.

   **Install Pixi:**

   ```bash
   # On Linux/macOS
   curl -fsSL https://pixi.sh/install.sh | bash

   # On Windows (PowerShell)
   irm https://pixi.sh/install.ps1 | iex

   # Or using conda/mamba
   conda install -c conda-forge pixi
   ```

   **Set up the development environment:**

   ```bash
   # Install all dependencies (default environment)
   pixi install

   # For development tools (includes additional dev dependencies)
   pixi install -e dev

   # For CUDA development (Linux only)
   pixi install -e cuda
   ```

   **Available tasks:**

   kornia-rs provides several tasks via pixi for common development workflows:

   ```bash
   # Rust development
   pixi run rust-check        # Check Rust compilation (all targets)
   pixi run rust-clippy       # Run clippy (all targets, warnings as errors)
   pixi run rust-fmt          # Format Rust code
   pixi run rust-fmt-check    # Check Rust formatting
   pixi run rust-lint         # Run all Rust lints (fmt + clippy + check)
   pixi run rust-test         # Run Rust tests
   pixi run rust-test-release # Run Rust tests (release mode)
   pixi run rust-clean        # Clean Rust build artifacts

   # Python bindings
   pixi run py-build          # Build kornia-py for development
   pixi run py-build-release  # Build kornia-py for release
   pixi run py-test           # Run pytest
   pixi run py-test-threaded  # Run pytest with free-threading
   pixi run py-clean          # Clean Python build artifacts

   # C++ bindings
   pixi run cpp-build         # Build C++ library (debug)
   pixi run cpp-build-release # Build C++ library (release)
   pixi run cpp-test          # Build and run C++ tests
   pixi run cpp-fmt           # Format C++ code
   pixi run cpp-clean         # Clean C++ build artifacts

   # CUDA development (requires cuda environment)
   pixi run -e cuda rust-build-cuda  # Build Rust with CUDA support
   pixi run -e cuda rust-test-cuda   # Run Rust tests with CUDA support

   # Utilities
   pixi run fmt-all           # Format all code (Rust, TOML, C++)
   pixi run test-all          # Run all tests (Rust, Python, C++)
   pixi run clean-all         # Clean all build artifacts
   ```

   **Pre-commit hooks:**

   This repository uses `pre-commit` for code quality. Install it with:

   ```bash
   pipx install pre-commit  # or: pip install --user pre-commit
   pre-commit install
   ```

   The hooks include:
   - whitespace and EOF fixers
   - YAML validation
   - Rust formatting (`pre-commit-rust` fmt)

   Run manually with:
   ```bash
   pre-commit run -a
   ```

5. **Develop and test:**

   **Requirements:**
   - AI tools may assist with coding, but you must understand and review all code before submission
   - **All local tests must pass before submitting PRs**

   Create test cases for your code. Run tests with:
   ```bash
   # Run all Rust tests
   pixi run rust-test

   # Run tests for a specific package
   pixi run rust-test-package <package-name>

   # Run Python tests
   pixi run py-test

   # Run C++ tests
   pixi run cpp-test
   ```

# Coding Standards

- Use meaningful names for variables, functions, and types.

- **Write small incremental changes:**
  - Commit small, logical changes
  - Write clear commit messages
  - Avoid large files

- **Add tests:**
  - Write unit tests for each functionality (in `#[cfg(test)]` modules)
  - Add integration tests where appropriate
  - Keep tests deterministic and focused
  - Test error cases and edge conditions

- **Formatting:**
  - Use `rustfmt` for formatting (run with `pixi run rust-fmt` or `cargo fmt --all`)
  - Follow Rust conventions and style guide

- **Linting:**
  - Use `clippy` with warnings denied. CI and local checks should pass:
    - `pixi run rust-clippy` (workspace, all targets, all features, `-D warnings`)
  - Address all clippy warnings before submitting PRs

- **Edition and MSRV:**
  - Rust edition 2021
  - Minimum supported Rust version (MSRV): 1.76 (as declared in the workspace `Cargo.toml`)

- **Error handling:**
  - Prefer `Result<T, E>` with descriptive error types (e.g., via `thiserror`)
  - Avoid `.unwrap()`/`.expect()` in library code (except in tests or where explicitly documented)
  - Use `?` operator for error propagation where appropriate

- **Naming and clarity:**
  - Choose descriptive names
  - Prefer early returns
  - Keep functions short and cohesive
  - Use Rust's type system for safety

- **Documentation:**
  - Use rustdoc comments (`///` for public items, `//!` for crate-level docs)
  - Document public APIs thoroughly
  - Include examples in documentation where helpful
  - Run `cargo doc --no-deps` to verify documentation builds

- **Dependencies:**
  - Be mindful of transitive dependencies and their impact on build times and binary size
  - Follow Rust dependency guidelines
  - Lock dependency resolution: use `--locked` flag when appropriate

# Pull Request

**Requirements:**
- Link PR to an issue (use "Closes #123" or "Fixes #123")
- Pass all local tests before submission
- 15-Day Rule: Inactive PRs (>15 days) will be closed
- AI-Assisted Development: AI tools may assist, but PRs must be human-authored and reviewed. Fully AI-generated PRs without human understanding will be rejected. Automated systems (including GitHub Copilot) detect excessive AI-generated content and may reject PRs.
- Transparency: Keep discussions public

**Note:** Tickets may be closed during cleanup. Feel free to reopen if you plan to finish the work.

**CI checks:**
- All tests pass (Rust, Python, C++ as applicable)
- Code formatting (rustfmt, clang-format)
- Linting (clippy with `-D warnings`)
- Documentation builds successfully (`cargo doc`)
- Pre-commit hooks pass

Fix any failing checks before your PR will be considered.

## Git and PR workflow

- Create feature branches from `main` and open PRs against `main`.
- Keep PRs focused and small when possible; include tests and documentation updates.
- Ensure all local checks pass before pushing:
  - `pre-commit run -a`
  - `pixi run rust-lint`
  - `pixi run rust-test` (and `pixi run test-all` for all features if relevant)
- Commit style: conventional commits are recommended (e.g., `feat:`, `fix:`, `docs:`). This helps with changelog and release notes.

## Cross-compilation

We configure `cross` via `Cross.toml` with Dockerfiles for targets:
- `x86_64-unknown-linux-gnu` → `devel-x86_64.Dockerfile`
- `aarch64-unknown-linux-gnu` → `devel-aarch64.Dockerfile`
- `i686-unknown-linux-gnu` → `devel-i686.Dockerfile`

You can build or test for a target with `cross` (install via `cargo install cross`):

```bash
cross build --target x86_64-unknown-linux-gnu
cross test  --target x86_64-unknown-linux-gnu
```

Or use pixi tasks for cross-compilation:

```bash
pixi run rust-cross-build-aarch64
pixi run rust-cross-test-aarch64
```

## Versioning and workspace changes

- The workspace uses a shared version in the root `Cargo.toml` and each crate carries its own version.
- When bumping versions, ensure:
  - The workspace package version is updated
  - Each published crate version is updated
  - Intra-workspace dependency versions are aligned (see `[workspace.dependencies]` with pinned versions)

## Release (maintainers)

The script `scripts/release_rust.sh` publishes all crates using `cross publish` and runs in dry-run mode by default.

Steps:
1. Update versions across crates and workspace; update dependency pins in `[workspace.dependencies]` accordingly.
2. Verify locally:
   ```bash
   pixi run rust-lint
   pixi run rust-test
   ./scripts/release_rust.sh  # dry-run
   ```
3. Perform the real publish when ready:
   ```bash
   ./scripts/release_rust.sh --no-dry-run
   ```

## Python bindings (`kornia-py`)

- Build wheels: `pixi run py-build`
- Run tests: `pixi run py-test`
- See `kornia-py/README.md` for additional details

## Development containers

The repository includes a devcontainer configuration (see README) for a reproducible environment.

## Reporting issues

Please include:
- OS and toolchain versions (`rustc -V`, `cargo -V`)
- Reproduction steps and minimal code
- Backtraces or logs as applicable

# License

Licensed under Apache-2.0. By contributing, you agree that your contributions will be licensed under the same license.
