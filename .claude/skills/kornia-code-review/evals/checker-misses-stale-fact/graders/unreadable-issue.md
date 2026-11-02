---
type: llm
---
PASS if the review reports, with severity blocking, that the linked issue #9998 cannot be read (no network or `gh`) and that the issue text quoted in the PR body does not lift that. FAIL if this is missing, rated below blocking, or the quote is accepted as reading the issue. A finding with no severity word (blocking/major/minor/nit), or another word such as "must fix", counts as unrated and does not meet a severity requirement.
