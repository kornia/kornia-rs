# Contributing to kornia-rs

Thank you for your interest in contributing! This document explains how to set up your environment, the coding standards we follow, and the checks you should run locally before opening a pull request.

This project is part of the Kornia ecosystem. Join the community on [Discord](https://discord.gg/HfnywwpBnD) to discuss features and get help.

## Policies and Guidelines

- **AI Policy & Authorship**: See [AI_POLICY.md](AI_POLICY.md) for the complete policy. Summary:
    - Kornia-rs accepts AI-assisted code but strictly rejects AI-generated contributions where the submitter acts as a proxy.
    - **Proof of Verification**: PRs must include local test logs proving execution (e.g., `pixi run rust-test` or `cargo test`).
    - **Pre-Discussion**: All PRs must be discussed in Discord or via a GitHub issue before implementation.
    - **Library References**: Implementations must be based on existing library references (Rust crates, OpenCV, etc.).
    - **Hallucination & Redundancy Ban**: Use existing `kornia-rs` utilities and never reinvent the wheel, except when the utility is not available.
    - **The "Explain It" Standard**: You must be able to explain any code you submit.
    - Violations result in immediate closure or rejection.

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
   - Follow the [development setup](#developing-kornia-rs) below.
   - See [Pull Request](#pull-request) section for PR requirements.

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
  - See [Best Practices](#best-practices) for more detailed error handling guidance

# Best Practices

This section provides comprehensive guidance for contributing to kornia-rs, with a focus on Rust best practices, performance, and maintainability.

## Before You Start

1. **Discuss First**: Always discuss your proposed changes in Discord or via a GitHub issue before starting implementation (see [Policies and Guidelines](#policies-and-guidelines)). This ensures your work aligns with project goals and avoids duplicate effort.

2. **Start Small**: If you're new to the project, start with small bug fixes or documentation improvements to familiarize yourself with the codebase and contribution process.

3. **Understand the Codebase**: Take time to explore existing code patterns, architecture, and conventions before implementing new features.

4. **Review Existing Utilities**: Before implementing new functionality, search the codebase for existing utilities in `kornia-rs` crates. This aligns with the AI Policy's Hallucination & Redundancy Ban (see [Policies and Guidelines](#policies-and-guidelines)).

## Development Workflow

1. **Keep PRs Focused**: Each PR should address a single concern. If you're working on multiple features, create separate PRs for each.

2. **Test Locally First**: Always run all relevant tests locally before submitting (see [Pull Request](#pull-request) for requirements):
   ```bash
   pixi run rust-lint    # Check formatting and linting
   pixi run rust-test    # Run all tests
   pixi run rust-check   # Verify compilation
   ```

3. **Update Documentation**: When adding new features or changing behavior, update rustdoc comments for public APIs (see [Coding Standards](#coding-standards) and [Rust-Specific Best Practices](#rust-specific-best-practices) for documentation guidelines).

4. **Follow Rust Idioms**:
   - Use pattern matching effectively (`match`, `if let`, `while let`)
   - Prefer composition over inheritance
   - Use `Option` and `Result` types appropriately
   - Prefer iterator chains over manual loops where idiomatic

## Code Quality

1. **Ownership and Performance**:
   - Prefer borrowing (`&T`, `&mut T`) over owned values; use `&[T]` over `Vec<T>` in parameters when ownership isn't needed
   - Avoid unnecessary allocations and clones (especially for large data structures like images and tensors)
   - Use `Cow<T>` for conditional cloning scenarios
   - Consider zero-copy operations (references, slices, views)
   - Use `Arc<T>` or `Rc<T>` only when shared ownership is truly needed
   - Prefer `&str` over `String` in function parameters
   - Profile before optimizing (use `cargo bench` and profiling tools)
   - Consider SIMD optimizations for numerical computations when available
   - Use appropriate data structures (e.g., `HashMap` vs `BTreeMap` based on access patterns)

2. **Code Clarity**:
   - Use descriptive variable and function names that convey intent
   - Keep functions focused and single-purpose
   - Prefer clear code over comments; when comments are needed, explain "why" not "what"
   - Avoid over-engineering; start simple and refactor when needed

3. **Memory Safety**:
   - Avoid `unsafe` code unless absolutely necessary
   - If using `unsafe`, document why it's safe with `// SAFETY:` comments
   - Prefer safe abstractions over raw pointers
   - Use `MaybeUninit` for uninitialized memory when needed

## Testing Best Practices

- Write tests for happy paths, error cases, edge conditions, boundary conditions, and integration scenarios
- Keep unit tests in `#[cfg(test)]` modules close to the code they test (see [Coding Standards](#coding-standards) for test structure)
- Create integration tests in `tests/` directory
- Make tests deterministic, fast, and independent
- Use descriptive test names; consider property-based testing (`proptest`) for numerical algorithms

## Review Process

- Review your own PR first: check for typos/formatting, verify tests pass, ensure documentation is updated, and confirm AI policy compliance
- Respond promptly to review feedback
- Be open to feedback and explain your decisions when questioned
- See [Pull Request](#pull-request) section for review requirements

## AI-Assisted Development

- Understand every line of code you submit; you must be able to explain it during review (see [AI Policy](AI_POLICY.md))
- Review AI output thoroughly: check for unnecessary complexity, verify it follows project conventions, ensure it uses existing utilities, and test it
- Be transparent in PR descriptions about what was AI-assisted and what you manually reviewed (see [Pull Request](#pull-request) for AI Usage Disclosure requirements)

## Communication

- Write clear, concise PR descriptions (see [Pull Request](#pull-request) for requirements)
- Always link to related issues or discussions in your PR description
- Ask questions in Discord or PR comments if unsure; it's better to clarify early than to rework later

## Rust-Specific Best Practices

1. **Error Handling**:
   - Use `Result<T, E>` with descriptive error types (prefer `thiserror` for library code)
   - Use `?` operator for error propagation
   - Avoid `unwrap()` and `expect()` in library code (except in tests or where explicitly documented)
   - Provide context in error messages; consider error conversion with `From` trait implementations

2. **Type Safety**:
   - Use newtype patterns for domain-specific types (e.g., `Image`, `Tensor`)
   - Leverage Rust's type system to prevent invalid states
   - Prefer enums over boolean flags for state representation

3. **Documentation**:
   - Document all public APIs with rustdoc comments (`///`)
   - Include examples, document panics/errors/safety requirements, and performance characteristics when relevant

4. **Dependencies**:
   - Minimize dependencies; prefer standard library when possible
   - Use feature flags for optional dependencies
   - Document why each dependency is needed
   - Keep dependency versions up to date (within MSRV constraints)

# Pull Request

## Issue Approval and Assignment Workflow

**Before submitting a PR, you must:**

1. **Open an issue first**: All PRs must be linked to an existing issue. If no issue exists for your work, create one using the appropriate template (bug report or feature request).

2. **Wait for maintainer approval**: A maintainer must review and approve the issue before you start working on it. New issues are automatically labeled with `triage` and will receive a welcome message explaining this process.

3. **Wait for assignment**: You must be assigned to the issue by a maintainer before submitting a PR. This ensures:
   - The issue aligns with project goals
   - No duplicate work is being done
   - Proper coordination of contributions

4. **Do not start work until assigned**: PRs submitted without prior issue approval and assignment may be closed or receive warnings during automated validation.

This workflow helps maintain quality, avoid conflicts, and ensure contributions align with the project's direction. The automated PR validation workflow will check these requirements and post warnings if they're not met.

**Requirements:**
- **Issue approval and assignment**: The linked issue must be approved by a maintainer and you must be assigned to it (see workflow above)
- Link PR to an issue (use "Closes #123" or "Fixes #123")
- Pass all local tests before submission
- For first time contributors, provide proof of local test execution in the PR description
- **AI Policy Compliance**: Must comply with [AI_POLICY.md](AI_POLICY.md). This includes:
  - Using existing `kornia-rs` utilities instead of reinventing
  - Using `Result<T, E>` for error handling (avoid `unwrap()`/`expect()` in library code)
  - Being able to explain all submitted code
  - Providing proof of local test execution (test logs)
  - Linking to pre-discussion (Discord/GitHub issue)
  - Providing library reference for implementations
- 15-Day Rule: Inactive PRs (>15 days) will be closed
- Transparency: Keep discussions public

**Code review:**
- By default, @copilot will check the PR against the AI Policy and the coding standards.
- Code must be reviewed by the repository owner or a senior contributor to finally decide over the quality of the PR.
- The project owners have the final say on whether the PR is accepted or not.

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
- Keep PRs focused and small when possible (see [Best Practices](#best-practices)); include tests and documentation updates.
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
