A passing review approves (negative control: measures false positives):

1. No blocking finding.
2. No major finding, except optionally a request for a unit test that exercises `==`. The original regression is a compile failure, and the CI check and test tasks enable `kornia-algebra/approx`.
3. No blocking or major finding is false (`graders/precision.md`, reported, not scored).

Origin: #746 (`f048f48724`) as merged; `pr.md` is its description rewritten to the current template with the outputs re-run on this tree, and `patch.diff` adds the changelog fragment it predates. `issue.md` is #728 as it was when the PR opened.

2026-10-24, automated run 1 (`claude plugin eval`, one run per arm, judge Opus, or Sonnet for Opus): Opus pass / W/OUT fail (no-blocking); Sonnet pass / W/OUT pass; Haiku fail (no-blocking) / W/OUT pass.

2026-10-24, automated run 2 (eval files hidden from the reviewed tree): Opus pass / W/OUT pass; Sonnet fail (no-blocking) / W/OUT pass; Haiku fail (no-blocking) / W/OUT pass.

2026-11-18, updated fixture, WITH arm only: Opus passed the no-blocking grader in 3/3 runs with Sonnet as judge.
