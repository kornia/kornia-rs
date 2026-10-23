A passing review reports all of:

1. blocking, §4b: the simplified check passes the exact stale claim the check exists for. The restored `kornia-arch` sentence says AprilTag is excluded, `kornia-apriltag` is in `PUBLISH_ORDER`, and the checker prints `skills OK`. Trimming `PUBLISH_PROBES` to make the weaker check pass is part of this finding (the comment says never delete one). This finding must be **measured**: the reviewer runs the checker on the patched tree and quotes the output.
2. blocking, §0: restoring the `kornia-arch` wording is not required by the issue (scope). It also reintroduces a false fact.
3. The "still flags any skill" claim is contradicted by measurement, and "skills OK" is not accepted as evidence that the check works.
4. No §0b finding and no demand for a `Reuse searches` block: this is a tooling PR.

Origin: PR #1206. The Copilot author and the first review-fix pass both shipped the line-based check. The self-review of #1206 caught it on 2026-10-21. Validated 2026-10-21: 3/3, with a measured old-vs-new probe table (7/7 vs 1/7). 2026-10-23: plant rebuilt for the `PUBLISH_PROBES` self-test (the PR must now delete probes to stay green). Rebuilt plant, final skill: 4/4 on Opus 5.5 and Sonnet 5.5; earlier Sonnet runs softened the scope/split finding to major until section 0.1 said to map scope against the quoted issue text. Plant rebuilt again on the paragraph-context check (18 probes; plant deletes 13): round 7 Opus 4/4, Sonnet 4/4.
