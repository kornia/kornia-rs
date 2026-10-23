---
name: kornia-code-review
description: Use when reviewing a kornia-rs PR or diff, and as the mandatory self-review before opening one — scope and size against the linked issue (blocking), performance (copies, allocations, latency, system load), benchmark obligations, correctness traps, probing claimed checks, skill compliance, and reporting findings as measured vs inferred. Trigger on "review this PR", "review my diff", "is this ready", "self-review", "open a PR".
---

# Reviewing kornia-rs changes

Two uses, same procedure:

- **Self-review (mandatory before opening a PR).** The author, or the
  author's agent, runs this skill on `git diff main...HEAD`, fixes what it
  finds, and pastes the final output in the PR's **Self-review** section.
- **Review.** A reviewer runs it on the PR and compares with the author's
  self-review (section 7).

Judge the change, not its author: do not speculate about whether or how AI
was used (disclosure belongs to the author, `AI_POLICY.md`). A defect is a
defect whoever wrote it.

## Output

```
Verdict: approve | request changes | comment
<findings, blocking first>
Reuse searches: <section 0b search log; library-code PRs only>
Skill gaps: <section 7>
```

Every finding, scope findings included, is one line:
`path:line — severity — problem — fix — [measured|inferred]`.

- `path:line`: line in the new file at the PR head (old file for deleted
  lines). Findings about the PR description use `PR body`.
- Severity: **blocking** (must change before merge), **major** (should
  change), **minor** (worth changing), **nit** (optional). Any blocking ⇒
  verdict is request changes. Severity is per finding, as the section
  that raises it says; never lower one because other blockers already decide
  the verdict.
- **Measured** = you ran a command and quote its output, in the finding or in
  the `Reuse searches` block it cites (a reuse finding filed under another
  section is still measured). **Inferred** = from reading code only. Inferred is enough to block on scope and on correctness the
  code shows conclusively. Perf claims need the author's measurement; the
  burden is on the PR, not the reviewer. Run a bench or test yourself when it
  is cheap and settles a disagreement.
- Never write "should be faster" or "tests pass" without output. "Tests
  pass" is no evidence for a path no test exercises: check one exists, and
  that it ran (name or count in the output). `#[ignore]`, unmet
  `required-features` and `#[cfg(target_arch = ...)]` tests skip silently.

## 0. Scope and size (check first, blocking)

PRs are small and do exactly what the linked issue asks. Before reviewing
code:

1. Read the linked issue (`gh issue view <n> -R kornia/kornia-rs`). No linked
   issue ⇒ blocking, including maintainer-requested work: open an issue, it
   is cheap. Issue unreadable (no access, no network) ⇒ blocking until it
   can be read: step 2 is impossible without it, and a quote in the PR body
   is not the issue. Still finish the review: steps 3–5 don't need the issue
   and keep their own severities, and quoted issue text that breaks a skill
   rule still gets the push-back below.
2. Map every changed file/hunk to the issue. Any hunk that does not serve it
   (drive-by refactors, renames, formatting of untouched code, unrelated deps,
   extra features, "while I was here" fixes) ⇒ blocking: out of scope — not
   required by #<issue>; move to its own PR.
3. More than one concern (e.g. feature + refactor, two ops, tooling + policy)
   ⇒ blocking: split required, with the proposed split.
4. Over ~400 changed lines (docs count; lockfiles and generated files do not)
   needs a stated reason why it cannot be split; without one ⇒ blocking:
   split required.
5. The description matches the diff against `main`, not against earlier
   commits of the same PR. A claimed change missing from the diff, or a
   change the description does not mention ⇒ major.

Being in scope does not make a change acceptable. If the issue itself asks
for something that breaks a skill rule (e.g. reimplementing an existing
helper), push back on the issue instead of approving it.

Sections 0b–4 apply to library code (`crates/`, `kornia-py/src`,
`kornia-cpp/src`). For tooling, CI or docs PRs, skip them and do section 4b.

## 0b. Reuse, duplication and novelty (library code)

Do this before section 1. For every new function, type or kernel the diff
adds (private helpers included), and every inlined block that replaces a call
to an existing helper, answer in order.

**Reuse is always measured, never inferred.** For each new item, run the
searches and paste them in the `Reuse searches` block of the output:

```
Reuse searches:
- <new item> (<job in one line>)
  $ rg -n "<pattern>" crates/
  <hit lines, path:line: text — or "no hits">
```

Every 0b finding names its search entry and is tagged `[measured]`. A reuse
finding, or a claim that nothing existing fits, without a pasted search is
incomplete: run it before writing the finding. Reading the code is enough to
*suspect* a duplicate; the pasted hit is what reports it.

1. **Does kornia already have it?** Name the job in one line (e.g. "pad an
   image with reflected borders", "rotate a vector by a quaternion"), then
   search by verb, by math and by signature, not only by the new name:
   `rg -n "fn \w*<verb>\w*" crates/`, `rg -n "<InputType>.*-> .*<OutputType>" crates/`,
   and the owning crate from the `kornia-arch` table. Check `pub(crate)` and
   private items in the same crate too. A match ⇒ major or blocking.
2. **Is it a near-copy?** A second routine doing the same job under another
   name, threshold or loop shape ⇒ major: keep one definition, the other
   delegates. If the copies **disagree** (epsilon, rounding, border or error
   behaviour), show an input where they differ ⇒ blocking. Code transcribed
   from another library (imageproc, Open3D, OpenCV…) ⇒ blocking:
   re-implement from the reference.
3. **Can it be composed from kornia primitives?** Write the call chain that
   gives the same output. If one exists, the PR uses it or shows a bench
   where the new code wins (section 3).
4. **If it is new:**
   - The rustdoc names the reference (paper, OpenCV function, crate).
   - At least one test pins values from that reference (OpenCV,
     scikit-image, numpy), saying how they were generated. Self-consistency
     or range checks alone ⇒ major.
   - Perf claims are benched against the fastest existing implementation.
   - A general primitive lands first in its owning crate, as its own PR with
     a bench; then the feature uses it.
5. **No new dependency** (ndarray, nalgebra, faer, image, glam directly, a
   SIMD crate for one function) for something kornia covers, unless the PR
   measures why kornia's path is not enough. Linear-algebra backend: see
   `kornia-arch`.

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

1. A bench in `crates/<crate>/benches/` covers the changed module (usually
   `bench_<module>.rs`, e.g. `src/resize/` → `bench_resize`; list the dir).
   New perf-critical op → new bench in the PR.
2. Before/after numbers are in the PR, same machine, criterion baseline:
   ```bash
   git checkout main && cargo bench -p <crate> --bench <bench> -- --save-baseline main
   git checkout -   && cargo bench -p <crate> --bench <bench> -- --baseline main
   ```
3. Benches with `required-features` (e.g. `cuda`) are **silently skipped**
   without the feature — check the output shows the bench actually ran.
4. **A/A trap**: an A/B run can silently measure the same code twice and
   report "no change". Causes: worktrees sharing one `CARGO_TARGET_DIR`, a
   downstream `path =` / `[patch]` resolving to another checkout, a
   `maturin develop` wheel built from another tree. Each arm records
   `git rev-parse HEAD` and the bench binary path (Python:
   `kornia_rs.__file__`); "no change" without that provenance is not evidence.
   Fix `RAYON_NUM_THREADS` and warm up before timing.
5. If the change alters what a bench measures (signature, input size, dtype),
   the bench is updated in the same PR, not left measuring the old path.
6. A regression beyond noise needs an explicit reason in the PR, or it blocks.

## 4. Correctness traps

- Size validation compares **shape** (width, height, channels), not pixel
  count, and runs before `dst` is written. A test with equal pixel count but
  a different shape must expect `Err`; reuse the shared helper (e.g.
  `check_size`) rather than inlining a check.
- SIMD path matches the scalar path in a test (see `kornia-simd`); tail
  elements handled.
- Bug fix ⇒ regression test that fails on `main`.
- A test whose input makes the new code a no-op (axis-aligned points,
  identity transform, already-diagonal matrix, all-equal pixels) is no
  evidence for it. Name the branch or loop the test input actually exercises.
- Port of a reference (OpenCV, a crate): match it, not its bugs. Every
  deliberate departure (e.g. strict vs `>=` maxima) is stated in the rustdoc.
- Closing an issue: `grep -rnE "#NNNN|issues/NNNN" crates/ kornia-py/ kornia-cpp/`
  for inline notes, `#[ignore]` reasons and tests that name it; each hit is
  updated in the same PR.
- Behaviour that depends on the toolchain is checked on MSRV and with the
  `ci` feature set, not only on local stable.
- Public API change ⇒ bindings updated or the PR says it is not exposed
  (see `kornia-bindings`).
- Docs, units and coordinate frames match the code. A change that weakens a
  check the `# Errors` rustdoc promises is a behaviour change: cite the
  rustdoc line.

## 4b. Checks, scripts and CI (tooling PRs)

A check is only as good as the input it rejects. For every check the PR adds
or claims (validator, CI step, lint, hook):

- Re-run the PR's own negative probes, then try the exact bug the PR says it
  fixes, plus a rephrasing of it. A check that passes the original bug ⇒
  blocking. Work on a copy (`git archive HEAD | tar -x -C <tmp>`; without git,
  `patch -R -p1 < pr.diff` on a copy gives the base), never with
  `git checkout --` on a tree with uncommitted work.
- A PR that changes a check (refactor, "simplify") must show its coverage
  did not shrink: run the **old and new** check on the same negative probes,
  including any text the PR itself edits that the check guards. Old rejects
  and new passes ⇒ blocking.
- Check what the docs claim the check covers against what it actually matches.
- Workflows: note which ref they run from (`pull_request_target` runs from
  `main`, so a PR cannot show its own new logic working).

## 5. Skill compliance

Check the diff against `kornia-coding-style` (naming, `thiserror`, no
`unwrap` in library code, `// SAFETY:`, rustdoc) and `kornia-api-flavor`.
Cite the rule; don't restate it here. If the PR changes the skills
themselves, review against the skills on `main` and call out the self-reference.

## 6. PR hygiene

- **Self-review** section has the pasted output of this skill. Missing,
  empty or placeholder ⇒ blocking. Placeholder = no result per applicable
  section (0–6, or 0 and 4b for tooling), or, for library code, no
  `Reuse searches` block. `Verdict: approve` + `Skill gaps: none` alone is a
  placeholder.
- **Author walkthrough** shows the contributor reviewed and understands the
  change. It must be the human's own words: in self-review, an agent never
  writes it; it reports it as missing and leaves it to the human. Reviewers:
  - Check it against the diff: every non-trivial file explained, the "why"
    tied to the issue, the risk named matches the real risk. Missing,
    generic, or describing code that is not in the diff ⇒ blocking.
  - Ask the author at least one concrete question about a non-obvious hunk
    (why this bound, what happens on empty input, why this copy). No answer,
    or an answer that contradicts the code ⇒ blocking until resolved or the
    hunk is removed.
- AI disclosure filled with agent + version, model, effort and skills applied
  (see `kornia-workflow`). Verification section quotes real commands and
  output; no checklist box ticked for something not run.
- If the PR changes a fact a skill states, the skill is updated in the same PR
  (see `kornia-curation`).
- User-visible change ⇒ `changelog.d/<PR>.<type>.md` whose type and text match
  the diff (a `breaking` change filed as `fixed` ⇒ major; a `perf` entry without
  a measured number ⇒ major). Internal-only ⇒ the PR says so (no-changelog).

## 7. Close the loop (every review)

End with a **Skill gaps** section:

- Findings that no skill rule covered: name the owning skill and propose
  the one-line rule (the invariant and how to spot it).
- Rules that were ambiguous or did not fit the PR.
- Reviewers: every blocking or major finding the author's self-review missed
  is a gap. Name the rule that should have caught it, or propose one.
- A defect a review missed that surfaced later (CI, bug report, revert):
  add a planted case under `evals/` (see `evals/README.md`) so the next
  skill edit is tested against it.

Proposed rules land via `kornia-curation`: in the same PR if it is the
author's own learning, otherwise in a small follow-up PR. A skill edit is
not done until every case in `evals/` still passes.
