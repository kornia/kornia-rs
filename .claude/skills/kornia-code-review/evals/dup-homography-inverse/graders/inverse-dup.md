---
type: llm
---
PASS if the review reports, as major or blocking and tagged `[measured]`, that `mat3_inv` in `warp/points.rs` duplicates `invert_homography`/`inverse_perspective_matrix` in `warp/perspective.rs` and should call it. FAIL if missing, rated minor/nit, or tagged inferred. A finding with no severity word (blocking/major/minor/nit), or another word such as "must fix", counts as unrated and does not meet a severity requirement.
