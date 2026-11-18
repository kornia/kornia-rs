---
type: llm
---
Negative control: a merged PR with no real blocking or major defect beyond those listed. PASS if the review has no finding rated blocking and no finding rated major, except optionally a request for a unit test that exercises `==`. The original regression is a compile failure, and the CI check and test tasks enable `kornia-algebra/approx`. Minor findings and nits are fine. FAIL if any other blocking or major finding is present, quoting it. A finding with no severity word (blocking/major/minor/nit), or another word such as "must fix", counts as unrated and does not meet a severity requirement.
