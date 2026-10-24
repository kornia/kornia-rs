Title: fix(imgproc): no panic on zero-size images in distance_transform and compute_histogram

**Fixes/Relates to:** #1163

User-visible fix: `changelog.d/1166.fixed.md`.

## Changes Made
- `distance_transform.rs`: the four rayon splits (`grid` and `src` rows, `scratch` columns, the `transpose_map` destination) go through the crate's `parallel::par_rows_exact_mut` / `par_row_chunks` helpers, which yield no rows for a zero-size image instead of panicking on a zero chunk size. Added the missing Arguments/Returns/Errors rustdoc.
- `histogram.rs`: `par_chunks(ROWS_PER_TASK * width)` becomes `par_row_chunks(src_slice, width, ROWS_PER_TASK)`, the same split for non-empty input.

## How Was This Tested?
- Unit tests `distance_transform_empty_image` and `test_compute_histogram_empty_image` cover 0x5, 5x0 and 0x0. They fail on main (tests applied to main's code, source hunks left out) and pass here:
```text
$ cargo test -p kornia-imgproc --lib empty_image   # main + the two new tests
test distance_transform::tests::distance_transform_empty_image ... FAILED
test histogram::tests::test_compute_histogram_empty_image ... FAILED
thread 'distance_transform::tests::distance_transform_empty_image' panicked at crates/kornia-imgproc/src/distance_transform.rs:75:14:
chunk_size must not be zero
thread 'histogram::tests::test_compute_histogram_empty_image' panicked at crates/kornia-imgproc/src/histogram.rs:74:10:
chunk_size must not be zero
test result: FAILED. 6 passed; 2 failed; 0 ignored; 0 measured; 395 filtered out; finished in 0.01s

$ cargo test -p kornia-imgproc --lib empty_image   # this PR
test histogram::tests::test_compute_histogram_empty_image ... ok
test distance_transform::tests::distance_transform_empty_image ... ok
test result: ok. 8 passed; 0 failed; 0 ignored; 0 measured; 395 filtered out; finished in 0.00s
```
- Non-empty images: `par_row_chunks(d, w, n)` is `d.par_chunks(w * n)` and `par_rows_exact_mut(d, w)` is `d.par_chunks_exact_mut(w)` once `w > 0` (`parallel.rs:26,36,46`), and the buffers are exactly `width * height` long, so the split is unchanged. No perf change is claimed and no bench was run.

## Self-review
```text
Verdict: approve
- §0 scope: the two functions #1163 names. The rustdoc paragraph on the no-foreground value (1e10) and f32 precision documents existing behaviour of the function being changed; it is not the fix.
- §1-3: no new allocation or copy; same rayon split for non-empty input (parallel.rs:26,36,46); no perf claim, so no bench.
- §4: the new tests fail on main with the #1163 panic and pass here (output above).
- §5: no unwrap in library code; rustdoc has Arguments/Returns/Errors.
- §6: changelog fragment added; outputs pasted from this tree.
Reuse searches:
- zero-size-safe row split (split a buffer into rows without panicking on empty images)
  $ rg -n "fn par_row|fn par_rows" crates/
  crates/kornia-imgproc/src/parallel.rs:21:pub(crate) fn par_row_chunks_mut<T: Send>(
  crates/kornia-imgproc/src/parallel.rs:31:pub(crate) fn par_row_chunks<T: Sync>(
  crates/kornia-imgproc/src/parallel.rs:42:pub(crate) fn par_rows_exact_mut<T: Send>(
  -> reused; no new helper.
Skill gaps: none
```

## Author walkthrough
- **Per file:** `distance_transform.rs` splits buffers with rayon at four sites (`grid` and `src` rows, `scratch` columns, and the `transpose_map` destination). With width or height 0 the chunk size is 0 and rayon panics, which is the bug in #1163. The crate already has row helpers that skip empty buffers, so I switched all four to them rather than adding an early return. `histogram.rs` had the same split once.
- **Riskiest part and how I checked it:** changing the split in a hot loop. For non-empty input the helpers reduce to the same `par_chunks` / `par_chunks_exact_mut` calls (they only clamp a zero chunk size to 1), and `chunks_exact` matches `chunks` because every buffer is exactly `width * height`. The existing distance-transform and histogram tests still pass.

## AI Usage Disclosure
- **Agent + version:** Claude Code 2.1
- **Model:** claude-sonnet-5-5
- **Effort level:** medium
- **Skills applied:** kornia-workflow, kornia-coding-style, kornia-code-review
- **Human vs AI part:** human diagnosed the panic and chose the helpers; agent wrote the tests and rustdoc
