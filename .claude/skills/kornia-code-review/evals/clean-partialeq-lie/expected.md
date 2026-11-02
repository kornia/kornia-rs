A passing review approves (negative control: measures false positives):

1. No blocking finding.
2. No major finding, except optionally a request for a unit test that exercises `==` (the regression check is the all-features build), and a finding that CI does not build with `approx` so the fix has no CI regression check (true; the PR says so).
3. No blocking or major finding is false (`graders/precision.md`, reported, not scored).

Origin: #746 (`f048f48724`) as merged; `pr.md` is its description rewritten to the current template with the outputs re-run on this tree, and `patch.diff` adds the changelog fragment it predates. `issue.md` is #728 as it was when the PR opened.

2026-10-24, automated run 1 (`claude plugin eval`, one run per arm, judge Opus, or Sonnet for Opus): Opus pass / W/OUT fail (no-blocking); Sonnet pass / W/OUT pass; Haiku fail (no-blocking) / W/OUT pass.

2026-10-24, automated run 2 (eval files hidden from the reviewed tree): Opus pass / W/OUT pass; Sonnet fail (no-blocking) / W/OUT pass; Haiku fail (no-blocking) / W/OUT pass.
