---
type: llm
---
PASS if the review reports, as major or blocking and tagged `[measured]`, that the hand-written `jacobi_symmetric` eigen-solver re-implements an existing decomposition, and points at `kornia_algebra::linalg::svd` (`svd3_f64`/`svd3`) or nalgebra `symmetric_eigen`. FAIL if missing, rated minor/nit, or tagged inferred. A finding with no severity word (blocking/major/minor/nit), or another word such as "must fix", counts as unrated and does not meet a severity requirement.
