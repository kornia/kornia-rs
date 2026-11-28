---
type: llm
---
Negative control: a merged PR with no real blocking or major defect beyond those listed. PASS if every finding has a recognized severity and the review has no finding rated blocking and no finding rated major, except optionally a request for a unit test that exercises `==`. The original regression is a compile failure, and the CI check and test tasks enable `kornia-algebra/approx`. Minor findings and nits are fine. FAIL if any other blocking or major finding is present, quoting it. FAIL if any finding has no severity or uses an unrecognized severity such as "critical" or "must fix". A checked section reported as "no finding" is not a finding.
