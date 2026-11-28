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

## Self-review
```text
Verdict: approve
§4 — no finding — the bimodal test checks that the selected threshold separates the intensity groups. — [inferred]
Reuse searches:
  $ rg -l 'pub fn threshold_binary<' crates/kornia-imgproc/src
  crates/kornia-imgproc/src/threshold.rs
  The final pass uses threshold_binary.
Skill gaps: none
```

Author walkthrough:
- Per file: see description.
- Riskiest part and how I checked it: numerical edge cases, covered by the unit tests.
