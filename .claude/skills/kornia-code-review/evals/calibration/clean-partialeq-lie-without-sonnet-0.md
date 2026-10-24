**Review: derive `PartialEq` on `SO3F32`, `RxSO3F32`, `SE3F32`, `Sim3F32`**

The fix looks correct and minimal. I found no blocking problems, but the docs have one misleading claim and some formatting slips. I only read the code; I ran no builds.

**What's right**
- `SO3F32` and `SE3F32` have `approx::AbsDiffEq` impls that need `PartialEq` (`so3.rs:320`, `se3.rs:380`). Deriving it fixes the `--all-features` build in #728.
- The derive follows what `so2.rs` and `se2.rs` already do. The `SO3F64` type at `so3.rs:361` also has it.

**What to fix**
1. **The "use `approx` instead" advice is wrong.**
   - The PR description says to use the `approx` comparisons for rotation equality. The `approx` impls don't give that either.
   - `SO3F32::abs_diff_eq` is `self.q.abs_diff_eq(&other.q, eps)` (`so3.rs:329-331`), which is also sign-sensitive. `q` and `-q` compare unequal there too.
   - The rustdoc added in the diff doesn't make this claim, so the code is fine. Correct the PR text, and consider a doc sentence or a helper that compares rotations up to sign, such as `|q1·q2|` close to 1.
2. **Doc formatting problems.**
   - In `sim3.rs`, the new "Note regarding `PartialEq`" lines follow the "7 degrees of freedom…" line with no blank `///` between them. rustdoc will merge them into one paragraph. Add a `///` separator, as `so3.rs` and `rxso3.rs` do.
   - The doc comment on `SE3F32` is only the note, with no type description. It reads oddly as the first line of the docs.
   - `so3.rs` and `rxso3.rs` use a 3-space indent after `///`, which renders as a code block in some markdown settings. The `se3.rs` and `sim3.rs` notes are unindented, so the four are inconsistent.
3. **The notes for `SE3F32` and `Sim3F32` overstate the problem.** Two instances "representing the same transformation may evaluate as not equal" is only true for the sign flip. It would be clearer to say the rotation quaternion is compared exactly.
4. **Derive scope.** The issue also lists `SO2F32` and `SE2F32`, which already derive it, so nothing is missing there.
   - `RxSO3F32` and `Sim3F32` have no `AbsDiffEq` impl in this tree, so they don't need the derive for this build fix. `Sim3F32` needs `RxSO3F32: PartialEq` anyway, so keeping both is reasonable.
   - Float `PartialEq` without `Eq` is expected here.
5. **Changelog.** The fragment is `746.fixed.md`, while the issue is #728. Check that 746 is this PR's number and not a typo.
6. **Tests.** No test was added. A small test for `a == a` and `q != -q` would document the semantics the new docs describe. The author's claim that the build check passes is plausible but unverified here.

Once the misleading `approx` advice and the doc formatting are fixed, this can merge.
