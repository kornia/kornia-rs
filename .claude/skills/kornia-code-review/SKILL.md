---
name: kornia-code-review
description: Use when reviewing a kornia-rs PR or diff, or self-reviewing before opening one — scope and size against the linked issue (blocking), performance (copies, allocations, latency, system load), benchmark obligations, correctness traps, skill compliance, and how to report findings as measured vs inferred. Trigger on "review this PR", "review my diff", "is this ready", "self-review".
---

# Reviewing kornia-rs changes

Review the diff against what the PR claims. Every finding is
`path:line — severity — problem — fix — [measured|inferred]`.
**Measured** = you ran a command and quote its output. **Inferred** = from
reading code. Never write "should be faster" or "tests pass" without output.

## 0. Scope and size (check first, blocking)

PRs are small and do exactly what the linked issue asks. Before reviewing
code:

1. Read the linked issue. No linked issue ⇒ report **out of scope: no issue**.
2. Map every changed file/hunk to the issue. Any hunk that does not serve it
   (drive-by refactors, renames, formatting of untouched code, unrelated deps,
   extra features, "while I was here" fixes) ⇒ report
   **out of scope: `path:line` — not required by #<issue>; move to its own PR**.
3. More than one concern (e.g. feature + refactor, two ops) ⇒ report
   **split required** with the proposed split.
4. Large diffs (hundreds of non-generated lines) need a stated reason why they
   cannot be split; without one ⇒ **split required**.

Scope findings are blocking: list them at the top of the review and do not
approve until they are resolved, whatever the code quality.

## 1. Copies and allocations (most common regression)

- Full-buffer `.to_vec()`, `.clone()`, `collect::<Vec<_>>()`, `Image::new`/
  `Tensor::from_shape_vec` on image/tensor data inside an op is a copy. Ask
  why it cannot borrow (`as_slice()`, views, `&mut dst`).
- Ops write into a caller-provided `&mut dst`; no allocation per call, per
  frame or per pixel. Scratch buffers are hoisted or owned by a reusable struct.
- A "refactor" or "simplify" PR that adds a copy before an unchanged kernel is a
  perf regression even when tests pass — ask for the bench (section 3).

## 2. Latency and system load

- rayon: no nested `par_iter`, no parallelism over tiny inputs, row chunking
  sized to the image (see `crates/kornia-imgproc/src/parallel.rs`).
- No blocking I/O, locks or logging in per-pixel/per-frame loops.
- CUDA: no extra host syncs or device↔host copies per call; host reads only
  after the stream sync.
- New threads, busy loops or polling in library code need justification.

## 3. Benchmark obligations

If the diff touches a hot path (`crates/*/src/**` kernels, color, resize, warp,
filters, features, SIMD, CUDA, tensor ops):

1. The matching bench exists: `crates/<crate>/benches/bench_<module>.rs` (e.g.
   `src/resize/` → `bench_resize`). New perf-critical op → new bench in the PR.
2. Before/after numbers are in the PR, same machine, criterion baseline:
   ```bash
   git checkout main && cargo bench -p <crate> --bench bench_<module> -- --save-baseline main
   git checkout -   && cargo bench -p <crate> --bench bench_<module> -- --baseline main
   ```
3. Benches with `required-features` (e.g. `cuda`) are **silently skipped**
   without the feature — check the output shows the bench actually ran.
4. If the change alters what a bench measures (signature, input size, dtype),
   the bench is updated in the same PR, not left measuring the old path.
5. A regression beyond noise needs an explicit reason in the PR, or it blocks.

## 4. Correctness traps

- Size validation compares **shape** (width, height, channels), not pixel
  count, and runs before `dst` is written.
- SIMD path matches the scalar path in a test (see `kornia-simd`); tail
  elements handled.
- Bug fix ⇒ regression test that fails on `main`.
- Public API change ⇒ bindings updated or the PR says it is not exposed
  (see `kornia-bindings`).
- Docs, units and coordinate frames match the code.

## 5. Skill compliance

Check the diff against `kornia-coding-style` (naming, `thiserror`, no
`unwrap` in library code, `// SAFETY:`, rustdoc) and `kornia-api-flavor`.
Cite the rule; don't restate it here.

## 6. PR hygiene

- AI disclosure filled with agent + version, model, effort and skills applied
  (see `kornia-workflow`). Verification section quotes real commands and output.
- If the PR changes a fact a skill states, the skill is updated in the same PR
  (see `kornia-curation`).
