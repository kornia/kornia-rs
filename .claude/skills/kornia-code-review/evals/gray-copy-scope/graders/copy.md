---
type: llm
---
PASS if the review reports `src.as_slice().to_vec()` in `color/gray/mod.rs` as a full-buffer copy/allocation before an unchanged kernel, rated blocking or major. FAIL if missing or rated minor/nit. A finding with no severity word (blocking/major/minor/nit), or another word such as "must fix", counts as unrated and does not meet a severity requirement.
