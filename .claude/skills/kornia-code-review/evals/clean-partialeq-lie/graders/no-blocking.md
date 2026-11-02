---
type: llm
---
Negative control: a merged PR with no real blocking or major defect beyond those listed. PASS if the review has no finding rated blocking and no finding rated major, except optionally a request for a unit test that exercises `==` (the regression check is the all-features build), and a finding that CI does not build with `approx` so the fix has no CI regression check (true; the PR says so). Minor findings and nits are fine. FAIL if any other blocking or major finding is present, quoting it. A finding with no severity word (blocking/major/minor/nit), or another word such as "must fix", counts as unrated and does not meet a severity requirement.
