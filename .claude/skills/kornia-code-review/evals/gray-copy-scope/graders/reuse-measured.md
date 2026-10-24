---
type: llm
---
PASS if the review has a `Reuse searches` block containing a search command and its hits (verbatim, or condensed to `path:line`) that finds `check_size`, AND the finding about reusing `check_size` is tagged `[measured]`. FAIL if there is no such block, no hits shown, or the reuse finding is tagged inferred.
