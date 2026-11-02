A passing review approves (negative control: measures false positives):

1. No blocking or major finding, except optionally these true findings on the merged PR: a scope finding (blocking is fine) on the rustdoc paragraph about the no-foreground `1e10` value and f32 precision, which #1163 calls separate; a finding that the paragraph's "inexact above 4096 px" claim is wrong (squared distances lose integer exactness in f32 once width² + height² passes 2^24, e.g. 3000x3000); and a request for bench numbers on the changed rayon splits (section 3 is arguable: the PR shows the split is unchanged for non-empty input). Every model flagged the scope finding on 2026-10-24 and Opus found the 4096 error, so #1166 is a control with two known, listed defects.
2. No blocking or major finding is false (`graders/precision.md`, reported, not scored).

Origin: #1166 (`ead741e778`) as merged; `pr.md` is its description rewritten to the current template with the outputs re-run on this tree, and `patch.diff` adds the changelog fragment it predates. `issue.md` is #1163 as it was when the PR opened.

2026-10-24, automated run 1 (`claude plugin eval`, one run per arm, judge Opus, or Sonnet for Opus): Opus pass / W/OUT pass; Sonnet pass / W/OUT pass; Haiku fail (no-blocking) / W/OUT pass.

2026-10-24, automated run 2 (eval files hidden from the reviewed tree): Opus pass / W/OUT pass; Sonnet pass / W/OUT pass; Haiku run error / W/OUT pass.
