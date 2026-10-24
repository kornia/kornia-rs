---
type: llm
arm: with-only
---
Judge only the review's blocking and major findings, against the PR below.
FAIL if any of them is false: its conclusion is wrong because it says the
diff does something it does not, cites a function or line in the diff that
does not exist, calls correct code wrong, or states the opposite of a fact
in the key. Severity is never a contradiction: rating a key item higher or
lower than the key is graded elsewhere. A wrong
detail in a finding whose conclusion still holds is not false. Line numbers
refer to the whole file, which the diff shows only in part: never fail a finding
for its line number. A claim about code
outside the diff is taken as true unless the key contradicts it. Findings that
are not in the key are fine when true; missing findings, severity choices and
style are graded elsewhere. PASS when no blocking or major finding is false,
including when the review has none.
When failing, quote the false finding.

## Key (expected.md)

A passing review approves (negative control: measures false positives):
1. No blocking finding.
2. No major finding, except optionally a request for a unit test that exercises `==` (the regression check is the all-features build), and a finding that CI does not build with `approx` so the fix has no CI regression check (true; the PR says so).
3. No blocking or major finding is false (`graders/precision.md`, reported, not scored).

## PR description

Title: fix(algebra): derive PartialEq on Lie groups so --all-features builds

**Fixes/Relates to:** #728

User-visible fix: `changelog.d/746.fixed.md`.

## Changes Made
- `lie/{so3,rxso3,se3,sim3}.rs`: add `PartialEq` to the derives. #728 lists six structs; `SO2F32` and `SE2F32` already derive it, so this adds the other four. `SO3F32` and `SE3F32` have `impl approx::AbsDiffEq`, which requires `PartialEq`, so `--all-features` failed to compile; `RxSO3F32` and `Sim3F32` have no approx impl and are added because #728 lists them.
- Each struct's rustdoc says the comparison is exact and member-wise, so `q` and `-q` (the same rotation) compare unequal.

## How Was This Tested?
The bug is a build failure, so the regression check is an all-features build. CI's `ci` feature set does not enable `approx`, so CI does not build this path; this run is local:
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
- §0 scope: the four of #728's six structs that lacked the derive; no other change.
- §1-3: derive only; no runtime path changes; no perf claim.
- §4: regression check is the all-features build (fails on main, passes here; local only, CI does not enable approx); rustdoc states the q/-q caveat; bindings not exposed.
- §5: derive only; no unwrap, unsafe or new error types.
- §6: changelog fragment added.
Reuse searches:
  - equality for Lie groups
    $ rg -n "impl.*PartialEq|impl approx::AbsDiffEq" crates/kornia-algebra/src/lie
    so2.rs:180, se2.rs:319, se3.rs:380, so3.rs:320/601: `impl approx::AbsDiffEq`; no hand-written PartialEq
    -> derive, as se2.rs/so2.rs already do.
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

## Linked issue

title:	[Bug]: Missing PartialEq derive on Lie group structs breaks compilation with --all-features
state:	OPEN
author:	namanguptagit
labels:	bug, help wanted
assignees:	namanguptagit
projects:	
milestone:	
number:	728
--
### 🐛 Describe the bug

The kornia-algebra crate currently fails to compile when the --all-features flag is passed. This is because the optional approx crate dependency is enabled, and the approx trait AbsDiffEq requires PartialEq. However, PartialEq is missing from several Lie group structs in crates/kornia-algebra/src/lie/*.rs.

### 🔄 Steps to Reproduce

```bash
1. Clone the repository
2. Navigate to crates/kornia-algebra or the workspace root.
3. Run the following command:
cargo check --all-targets --all-features -p kornia-algebra
```

### 💻 Minimal Code Example

```rust
// N/A - This is a compilation error in the library itself.
```

### ✅ Expected behavior

The crate kornia-algebra should compile successfully with --all-features enabled.

### ❌ Actual behavior

Compilation fails with errors similar to the following:

```
error[E0277]: can't compare `SO3F32` with `SO3F32`
   --> crates/kornia-algebra/src/lie/so3.rs:267:29
    |
267 | impl approx::AbsDiffEq for SO3F32 {
    |                            ^^^^^^ no implementation for `SO3F32 == SO3F32`
    |
    = help: the trait `PartialEq` is not implemented for `SO3F32`
```

### 🔧 Environment

```shell
- kornia-rs version:main (v0.1.10-rc.5-121-gc317f04 approx)
- Rust version (`rustc -V`):rustc 1.93.1 
- Cargo version (`cargo -V`):cargo 1.93.1 
- OS (e.g., Linux, macOS, Windows): macOS
- Target architecture (if cross-compiling):N/A
- Python version (if using Python bindings):N/A
```

### 📝 Additional context

The proposed fix is to simply add PartialEq to the #[derive(Debug, Clone, Copy)] list for the following structs in crates/kornia-algebra/src/lie/:

SO2F32 (so2.rs)
SO3F32 (so3.rs)
RxSO3F32 (rxso3.rs)
Sim3F32 (sim3.rs)
SE2F32 (se2.rs)
SE3F32 (se3.rs)

### 🤝 Contribution Intent

- [x] I plan to submit a PR to fix this bug
- [ ] I'm reporting this bug but not planning to fix it

## Diff

```diff
diff --git a/crates/kornia-algebra/src/lie/rxso3.rs b/crates/kornia-algebra/src/lie/rxso3.rs
index d3a3ce78..56fef8d7 100644
--- a/crates/kornia-algebra/src/lie/rxso3.rs
+++ b/crates/kornia-algebra/src/lie/rxso3.rs
@@ -18,7 +18,12 @@ use super::so3::SO3F32;
 /// and 3D rotations. Internally stored as a quaternion where:
 /// - Rotation: normalized part of quaternion
 /// - Scale: ||quaternion||²
-#[derive(Debug, Clone, Copy)]
+///
+/// Note regarding `PartialEq`:
+///   Quaternions form a double cover for SO3, meaning `q` and `-q` represent the
+///   same rotation. However, `PartialEq` performs an exact, member-wise comparison
+///   and will return `false` for `q` and `-q`.
+#[derive(Debug, Clone, Copy, PartialEq)]
 pub struct RxSO3F32 {
     /// Quaternion representing rotation, scale = ||q||²
     pub quaternion: QuatF32,
diff --git a/crates/kornia-algebra/src/lie/se3.rs b/crates/kornia-algebra/src/lie/se3.rs
index 3567c6bb..514c5314 100644
--- a/crates/kornia-algebra/src/lie/se3.rs
+++ b/crates/kornia-algebra/src/lie/se3.rs
@@ -7,7 +7,11 @@ use rand::Rng;
 
 const SMALL_ANGLE_EPSILON: f32 = 1.0e-8;
 
-#[derive(Debug, Clone, Copy)]
+/// Note regarding `PartialEq`:
+/// This struct derives `PartialEq` which performs an exact element-wise comparison.
+/// Because quaternions form a double cover for SO3 (`q` and `-q` represent the same rotation),
+/// this means that two `SE3F32` instances representing the same transformation may evaluate as not equal.
+#[derive(Debug, Clone, Copy, PartialEq)]
 pub struct SE3F32 {
     pub r: SO3F32,
     pub t: Vec3AF32,
diff --git a/crates/kornia-algebra/src/lie/sim3.rs b/crates/kornia-algebra/src/lie/sim3.rs
index c95338f2..da671c07 100644
--- a/crates/kornia-algebra/src/lie/sim3.rs
+++ b/crates/kornia-algebra/src/lie/sim3.rs
@@ -26,7 +26,11 @@ use super::so3::SO3F32;
 /// - R³ is the 3D Euclidean vector space (translation)
 ///
 /// 7 degrees of freedom: 3 for rotation, 1 for scale, 3 for translation
-#[derive(Debug, Clone, Copy)]
+/// Note regarding `PartialEq`:
+/// This struct derives `PartialEq` which performs an exact element-wise comparison.
+/// Because quaternions form a double cover for SO3 (`q` and `-q` represent the same rotation),
+/// this means that two `Sim3F32` instances representing the same transformation may evaluate as not equal.
+#[derive(Debug, Clone, Copy, PartialEq)]
 pub struct Sim3F32 {
     /// Scaling and rotation component
     pub rxso3: RxSO3F32,
diff --git a/crates/kornia-algebra/src/lie/so3.rs b/crates/kornia-algebra/src/lie/so3.rs
index 9b019de5..b6b9ca21 100644
--- a/crates/kornia-algebra/src/lie/so3.rs
+++ b/crates/kornia-algebra/src/lie/so3.rs
@@ -57,7 +57,12 @@ const SMALL_ANGLE_EPSILON: f32 = 1.0e-8;
 ///   There is no globally continuous way to make this choice — discontinuities
 ///   (quaternion sign flips) are unavoidable over the full rotation group.
 /// - After repeated multiplications, call `q.normalize()` to prevent drift off S³.
-#[derive(Debug, Clone, Copy)]
+///
+/// Note regarding `PartialEq`:
+///   Quaternions form a double cover for SO3, meaning `q` and `-q` represent the
+///   same rotation. However, `PartialEq` performs an exact, member-wise comparison
+///   and will return `false` for `q` and `-q`.
+#[derive(Debug, Clone, Copy, PartialEq)]
 pub struct SO3F32 {
     pub q: QuatF32,
 }
diff --git a/changelog.d/746.fixed.md b/changelog.d/746.fixed.md
new file mode 100644
--- /dev/null
+++ b/changelog.d/746.fixed.md
@@ -0,0 +1 @@
+`SO3F32`, `RxSO3F32`, `SE3F32` and `Sim3F32` derive `PartialEq`, so `kornia-algebra` builds with `--all-features`.
```

<!-- skill_evals.py sync: 430f018c3a2f06a7 -->
