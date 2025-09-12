### Contributing to kornia-rs

Thank you for your interest in contributing! This document explains how to set up your environment, the coding standards we follow, and the checks you should run locally before opening a pull request.

This project is part of the Kornia ecosystem. Join the community on [Discord](https://discord.gg/HfnywwpBnD) to discuss features and get help.

## Prerequisites

- Rust toolchain (edition 2021, MSRV: 1.76 as declared in the workspace `Cargo.toml`)
- Python 3 if you work on `kornia-py`
- just (command runner)
- pre-commit (for local checks)

Recommended:
- Docker if you plan to use `cross` for cross-compilation

## Repository structure

- Workspace with multiple crates under `crates/*` and examples under `examples/*`
- Python bindings in `kornia-py/` (excluded from the Rust workspace by default)

## Quick start

Install the basics (see README for details), then:

```bash
cargo install just
pipx install pre-commit  # or: pip install --user pre-commit

# In repo root
pre-commit install
just check-environment
just lint      # format + clippy (warnings as errors) + check
just test      # run Rust tests (workspace, default features)
```

For Python bindings:

```bash
just py-install
just py-test
```

## Coding standards

- Formatting: rustfmt (run with `cargo fmt --all` or `just format`).
- Linting: clippy with warnings denied. CI and local checks should pass:
  - `just clippy` (workspace, all targets, all features, `-D warnings`)
  - `just clippy-default` (default features)
- Edition and MSRV: Rust 2021; minimum supported Rust version 1.76.
- Error handling: prefer `Result<T, E>` with descriptive error types (e.g., via `thiserror`). Avoid `.unwrap()`/`.expect()` in library code.
- Naming and clarity: choose descriptive names; prefer early returns; keep functions short and cohesive.
- Tests: add unit tests and examples where applicable; keep tests deterministic.

### Pre-commit hooks

This repository uses `pre-commit` with:
- whitespace and EOF fixers
- YAML validation
- Rust formatting (`pre-commit-rust` fmt)

Install and run:

```bash
pre-commit install
pre-commit run -a
```

## Build, test, and checks

- Fast local checks: `just lint` and `just test`
- Strict clippy (no warnings): enforced by `just clippy`
- Locked dependency resolution: our recipes use `--locked`. If dependency updates are needed, run `cargo update` and commit `Cargo.lock` when appropriate.

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

## Git and PR workflow

- Create feature branches from `main` and open PRs against `main`.
- Keep PRs focused and small when possible; include tests and documentation updates.
- Ensure all local checks pass before pushing:
  - `pre-commit run -a`
  - `just lint`
  - `just test` (and `just test-all` for all features if relevant)
- Commit style: conventional commits are recommended (e.g., `feat:`, `fix:`, `docs:`). This helps with changelog and release notes.

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
   just lint
   just test
   ./scripts/release_rust.sh  # dry-run
   ```
3. Perform the real publish when ready:
   ```bash
   ./scripts/release_rust.sh --no-dry-run
   ```

## Python bindings (`kornia-py`)

- Build wheels: `just py-build`
- Run tests: `just py-test`
- See `kornia-py/README.md` for additional details

## Development containers

The repository includes a devcontainer configuration (see README) for a reproducible environment.

## Reporting issues

Please include:
- OS and toolchain versions (`rustc -V`, `cargo -V`)
- Reproduction steps and minimal code
- Backtraces or logs as applicable

## License

Licensed under Apache-2.0. By contributing, you agree that your contributions will be licensed under the same license.
