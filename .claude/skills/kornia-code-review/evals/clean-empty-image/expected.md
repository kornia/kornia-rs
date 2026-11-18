A passing review approves (negative control: measures false positives):

1. No blocking or major finding, except optionally a request for before/after bench numbers on the changed rayon splits.
2. No blocking or major finding is false (`graders/precision.md`, reported, not scored).

Origin: #1166 (`ead741e778`). The fixture treats the original rustdoc as baseline except for the sentence about empty output. This keeps the unrelated no-foreground and precision notes out of the simulated change. The October runs below used the original diff. The fixture PR includes the before/after test output and a changelog fragment. `issue.md` is #1163 as it was when the PR opened.

2026-10-24, automated run 1 (`claude plugin eval`, one run per arm, judge Opus, or Sonnet for Opus): Opus pass / W/OUT pass; Sonnet pass / W/OUT pass; Haiku fail (no-blocking) / W/OUT pass.

2026-10-24, automated run 2 (eval files hidden from the reviewed tree): Opus pass / W/OUT pass; Sonnet pass / W/OUT pass; Haiku run error / W/OUT pass.

2026-11-18, scoped fixture, WITH arm only: Opus passed the no-blocking grader in 3/3 runs with Sonnet as judge. Before the PR explained the immutable row split and the two reproduced panic sites, it passed 2/3.
