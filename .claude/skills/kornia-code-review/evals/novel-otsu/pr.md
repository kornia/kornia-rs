Title: feat(imgproc): add Otsu threshold

Fixes #9996 (issue text: "Implement Otsu's automatic threshold for u8 grayscale images (see TODO in threshold.rs).")

Adds `threshold_otsu`, which computes the histogram, picks the threshold that maximises between-class variance, and applies `threshold_binary`. Returns the chosen threshold.

Tests: cargo test -p kornia-imgproc threshold passes.

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
