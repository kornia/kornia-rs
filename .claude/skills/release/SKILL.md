---
name: release
description: Use when cutting a kornia-rs release or pre-release (rc) — bumps the workspace version, updates CHANGELOG.md (moving [Unreleased] into a dated section and referencing the previous release), opens the release PR, and triggers the crates.io / wheel publish workflows. Trigger on "cut a release", "prepare a release", "pre-release", "publish to crates.io", "bump version".
---

# Releasing kornia-rs

Publishing is **irreversible** (crates.io only yanks, never deletes). Never run
the publish step without explicit human confirmation of version + scope.

## CUDA — always in scope (unless the human opts out)

**Every release must include the CUDA/GPU path.** The default `python_release.yml`
builds wheels **without** `--features cuda`, so `pip install kornia-rs` ships a
**CPU-only** wheel (no `kornia_rs.cuda`). Do NOT ship a release calling out GPU
features from a wheel that lacks them.

Before publishing wheels, confirm the GPU story:
- kornia-py's `cuda` feature uses cudarc **dynamic loading** — a `--features cuda`
  wheel still imports on machines without CUDA and should fall back to CPU; the
  GPU path activates at runtime when `libcuda` + `libnvrtc` are present. So a
  single cuda-built wheel is the goal (universal), not a separate variant.
- Add `--features cuda` to the linux wheel `args:` in `python_release.yml` (and
  verify import still works with no CUDA present) so pip users get GPU.
- Runtime requirements to put in the release notes: NVIDIA driver (`libcuda.so`)
  **and** `nvrtc` (kernels are JIT-compiled), matching arch (x86_64 / aarch64 /
  Jetson Tegra). Without nvrtc the GPU path can't compile kernels.
- Verify on a GPU box before announcing: `bench_cuda_warp_affine --features
  gpu-cuda` (Rust) and `import kornia_rs; kornia_rs.cuda...` (Python).

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
4. **Bump version** in root `Cargo.toml` — BOTH places: `workspace.package.version`
   AND every kornia-* pin in `[workspace.dependencies]` (11 lines; Cargo cannot
   inherit those, and locally `path` wins so a stale pin builds fine and only
   ships a wrong requirement at publish — this bit the rc.5 cut).
   `sed -i 's/version = "<old>"/version = "<new>"/g' Cargo.toml` covers both;
   `python3 scripts/check_version_pins.py` (also a pre-commit hook on
   Cargo.toml) verifies. Then `cargo check` so `Cargo.lock` reflects the bump.
   Sanity: the release commit's Cargo.toml diff should touch ~12 version lines,
   not 1 — compare against the previous release commit's diff shape.
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
