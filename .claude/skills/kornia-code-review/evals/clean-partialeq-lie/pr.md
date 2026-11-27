Title: fix(algebra): derive PartialEq on Lie groups so --all-features builds

Fixes #728

User-visible fix: `changelog.d/746.fixed.md`.

## Changes Made
- `lie/{so3,rxso3,se3,sim3}.rs`: add `PartialEq` to the derives. #728 lists six structs; `SO2F32` and `SE2F32` already derive it, so this adds the other four. `SO3F32` and `SE3F32` have `impl approx::AbsDiffEq`, which requires `PartialEq`, so `--all-features` failed to compile; `RxSO3F32` and `Sim3F32` have no approx impl and are added because #728 lists them.
- Each struct's rustdoc says the comparison is exact and member-wise, so `q` and `-q` (the same rotation) compare unequal.

## How Was This Tested?
The bug is a build failure, so the regression check is an all-features build. The first local build reproduces #728 without the derives. The second includes them. CI coverage came later in #1236: on today's main, both `rust-check` and `rust-test` explicitly enable `kornia-algebra/approx`.
```text
$ cargo check --all-targets --all-features -p kornia-algebra   # before this PR
error[E0277]: can't compare `SE3F32` with `SE3F32`
   --> crates/kornia-algebra/src/lie/se3.rs:380:5
error[E0277]: can't compare `SO3F32` with `SO3F32`
error: could not compile `kornia-algebra` (lib) due to 14 previous errors
$ cargo check --all-targets --all-features -p kornia-algebra   # this PR
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 54.27s
```
Not run: `cargo test`; the change adds derives only.
Public API: `PartialEq` is a new trait impl on the four types. Bindings: not exposed; kornia-py uses `SE3F32`/`SO3F32` only inside `pgo.rs` and has no Python class or `__eq__` for them, kornia-cpp does not use them.

## Self-review
```text
Verdict: approve
§0 — no finding — the diff adds the missing derive to four of the six types named in #728. The other two already derive it. — [inferred]
§0b — no finding — the existing SO2 and SE2 types use derived equality. — [measured]
§1–3 — no finding — deriving equality adds no work to existing operations. — [inferred]
§4 — no finding — the all-features build above went from E0277 to “Finished”. The CI tasks now compile those impls too (#1236, after this fix). — [measured]
§5 — no finding — the docs explain why quaternions of opposite sign can compare unequal. — [inferred]
§6 — no finding — the changelog names the newly supported comparison. — [inferred]
Reuse searches:
  $ rg -B1 'pub struct (SE2F32|SO2F32)' crates/kornia-algebra/src/lie/{se2,so2}.rs
crates/kornia-algebra/src/lie/so2.rs-#[derive(Debug, Clone, Copy, PartialEq)]
crates/kornia-algebra/src/lie/so2.rs:pub struct SO2F32 {
--
crates/kornia-algebra/src/lie/se2.rs-#[derive(Debug, Clone, Copy, PartialEq)]
crates/kornia-algebra/src/lie/se2.rs:pub struct SE2F32 {
Skill gaps: none
```

## Author walkthrough
- **Per file:** each of the four files gets the same one-word derive change plus a doc note. `so3.rs` and `se3.rs` are the ones the `approx` impls need, which is why the feature build broke; `rxso3.rs` and `sim3.rs` are the other two #728 lists.
- **Riskiest part and how I checked it:** users treating `==` as rotation equality. The doc note on each type says the comparison is exact and that `q` and `-q` compare unequal; I did not add a custom `PartialEq` that normalises sign, because that would make `==` disagree with the stored data.

## AI Usage Disclosure
- **Agent + version:** none
- **Model:**
- **Effort level:**
- **Skills applied:**
- **Human vs AI part:** written by a human
