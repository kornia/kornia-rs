Title: feat(imgproc): add unwarp_points_perspective

Fixes #9998 (issue text: "Add a function that maps keypoints detected in a perspective-warped (rectified) image back to the original image's pixel coordinates, using the same homography given to warp_perspective.")

Adds `warp::unwarp_points_perspective`, which inverts the forward homography and applies it to a slice of points, writing into a caller-provided output slice (no allocation). Small self-contained helpers keep the module independent of the warp kernels.

Tests: cargo test -p kornia-imgproc warp passes (identity, translation, singular).

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
