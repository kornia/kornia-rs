---
name: release
description: Use when cutting a kornia-rs release or pre-release (rc) — bumps the workspace version, updates CHANGELOG.md (moving [Unreleased] into a dated section and referencing the previous release), opens the release PR, and triggers the crates.io / wheel publish workflows. Trigger on "cut a release", "prepare a release", "pre-release", "publish to crates.io", "bump version".
---

# Releasing kornia-rs

Publishing is **irreversible** (crates.io only yanks, never deletes). Never run
the publish step without explicit human confirmation of version + scope.

## Facts about this repo

- Workspace version: single source in root `Cargo.toml` (`version = "..."`, all
  crates inherit via `version.workspace = true`).
- Publish script: `scripts/release_rust.sh` — topological, idempotent, skips
  already-published versions. Excludes `kornia`, `kornia-vlm`, `kornia-apriltag`.
- CI publish: `.github/workflows/rust_release.yml` (`workflow_dispatch`, guarded
  by typing `publish`) and `.github/workflows/python_release.yml` (wheels).
- Tag convention: `vX.Y.Z`, pre-releases `vX.Y.Z-rc.N`.
- User-facing notes live in **`CHANGELOG.md`** (newest first) AND the GitHub
  release body. Keep them consistent.

## Checklist

Create one todo per step.

1. **Confirm with the human** (blocking): version string, whether it is an rc,
   and scope (Rust crates / Python wheels / both). Do not proceed on a guess.
2. **Gather what changed.** `git log <last-tag>..main --oneline --no-merges`.
   Group by theme (not per-commit). Read the previous CHANGELOG.md entries first
   to match tone and altitude — user-impact prose, not a commit dump.
3. **Update `CHANGELOG.md`:**
   - Move the curated items from `## [Unreleased]` into a new dated section
     `## [X.Y.Z] — YYYY-MM-DD` (for an rc, keep it labelled `(pre-release)`).
   - Reset `## [Unreleased]` to empty.
   - Add/refresh the compare link at the bottom **referencing the previous
     release tag**: `[X.Y.Z]: https://github.com/kornia/kornia-rs/compare/<prev-tag>...vX.Y.Z`.
     This is what keeps the changelog navigable across releases.
   - Dates: ask the human or read from the environment; never invent one.
4. **Bump version** in root `Cargo.toml`. Run `cargo update -p <workspace crate>`
   / `cargo check` so `Cargo.lock` reflects the bump.
5. **Release branch + PR:** `release/vX.Y.Z`, commit `chore(release): vX.Y.Z`
   with the version bump + CHANGELOG. Open PR, let CI pass, get it merged. Never
   force-push a contributor's fork branch — release commits go on a repo branch.
6. **Tag** the merged commit `vX.Y.Z` and push the tag.
7. **Publish** (only after human types the go-ahead):
   - Rust: run `rust_release.yml` (`workflow_dispatch`, confirm field `publish`),
     or locally `scripts/release_rust.sh` (plan) → `--execute`.
   - Python: run `python_release.yml`.
8. **GitHub release:** `gh release create vX.Y.Z` (add `--prerelease` for rc)
   with the body copied from the new CHANGELOG section.
9. **Verify:** crate versions on crates.io, `pip install --pre kornia-rs==X.Y.Z`,
   release marked pre-release if rc.

## Red flags

- "Just publish real quick" → stop, run the checklist; publish is irreversible.
- CHANGELOG untouched → a release without a changelog entry is incomplete.
- No compare link / wrong previous tag → changelog history breaks.
- Pushing to a fork branch → release work belongs on a repo `release/*` branch.
