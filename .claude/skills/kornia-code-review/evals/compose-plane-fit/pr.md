Title: feat(3d): add least-squares plane fitting

Fixes #9997 (issue text: "Add a plane fit for 3D point sets (total least squares), needed for ground-plane estimation.")

Adds `kornia_3d::plane::fit_plane`: centroid, covariance, and a small cyclic Jacobi eigen-solver; the normal is the eigenvector of the smallest eigenvalue. No new dependencies.

Tests: cargo test -p kornia-3d plane passes.

AI disclosure:
- Agent + version: Claude Code 2.1
- Model: claude-sonnet-5-5
- Effort level: medium
- Skills applied: kornia-coding-style, kornia-api-flavor
- Human vs AI part: human wrote the issue and reviewed the diff; agent wrote the code and tests

Self-review:
```text
Verdict: approve
Skill gaps: none
```

Author walkthrough:
- Per file: see description.
- Riskiest part and how I checked it: numerical edge cases, covered by the unit tests.
