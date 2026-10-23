A passing review reports all of:

1. blocking, §0 — `crates/kornia-imgproc/src/flip.rs`: `cols` → `width` rename is not required by the issue.
2. blocking, §4 — `color/gray/mod.rs`: the size check compares pixel count, not shape (e.g. 2x6 src, 3x4 dst passes) and no longer reuses `check_size`.
3. blocking or major, §1 — `color/gray/mod.rs`: `src.as_slice().to_vec()` is a full-buffer copy before an unchanged kernel.
4. §3 — perf claim without `bench_color` before/after numbers.
5. The "faster" / "cache locality" claim is treated as unmeasured (tagged inferred or challenged), and "tests pass" is not accepted as evidence for the shape check.
6. §0b — a `Reuse searches` block with a pasted search that finds `check_size`; the reuse part of item 2 is tagged measured.

Validated 2026-10-21: 5/5 on the first skill draft, and 5/5 on the revised skill (scope first, missing Self-review and walkthrough flagged).
