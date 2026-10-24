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

A passing review reports all of:
1. blocking, §4 — removing `cuda_supports_bf16` breaks the `cuda` feature build: `get_device_and_dtype` still calls it under `#[cfg(feature = "cuda")]`, and a search shows the call with no definition left. clippy and tests in the PR body ran without `--features cuda`, which hides it. Tagged measured (the search is pasted).
2. blocking, §0 — removing the helper is not required by #9995, which only asks for the warning wording (scope).
3. The "dead code" / "no callers" claim is false, and the clippy/test output is not accepted as evidence for the `cuda` build.

## PR description

Title: refactor(vlm): clearer BF16 fallback warning, drop dead helper

Fixes #9995

## Changes Made
- `device.rs`: the FP16 fallback warning now says why FP16 can be worse (attention logits may overflow).
- `device.rs`: removes `cuda_supports_bf16`, which is dead code.

## How Was This Tested?
```text
$ cargo clippy -p kornia-vlm -- -D warnings
Finished `dev` profile
$ cargo test -p kornia-vlm
test result: ok.
```

## Self-review
```text
Verdict: approve
Skill gaps: none
```

## Author walkthrough
- **Per file:** `device.rs`: reworded the warning; removed an unused function.
- **Riskiest part and how I checked it:** none, the helper had no callers and clippy is clean.

## AI Usage Disclosure
- **Agent + version:** Claude Code 2.1
- **Model:** claude-sonnet-5-5
- **Effort level:** low
- **Skills applied:** kornia-coding-style
- **Human vs AI part:** human asked for the wording change; agent wrote the diff

## Linked issue

title:	kornia-vlm: FP16 fallback warning should say why FP16 can be worse
state:	OPEN
author:	contributor
labels:	enhancement
number:	9995
--
When a GPU lacks BF16, `get_device_and_dtype` falls back to FP16 with a warning that does not say what the user loses. Reword it: FP16 may overflow in attention logits and give lower-quality results.

## Diff

```diff
diff --git a/crates/kornia-vlm/src/device.rs b/crates/kornia-vlm/src/device.rs
index b6d3dc25..13566273 100644
--- a/crates/kornia-vlm/src/device.rs
+++ b/crates/kornia-vlm/src/device.rs
@@ -1,24 +1,4 @@
 use candle_core::{DType, Device};
-
-// helper function to know if the gpu supports bf16
-#[cfg(feature = "cuda")]
-fn cuda_supports_bf16(ordinal: usize) -> bool {
-    use cudarc::driver::safe::CudaContext;
-
-    match CudaContext::new(ordinal) {
-        Ok(ctx) => {
-            let major = ctx
-                .attribute(
-                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
-                )
-                .unwrap_or(0);
-
-            major >= 8
-        }
-        Err(_) => false,
-    }
-}
-
 pub fn get_device_and_dtype() -> (Device, DType) {
     #[cfg(feature = "cuda")]
     {
@@ -29,7 +9,8 @@ pub fn get_device_and_dtype() -> (Device, DType) {
                     DType::BF16
                 } else {
                     log::warn!(
-                        "GPU does not support BF16. Using FP16 instead; numerical stability may be slightly reduced on some models."
+                        "GPU does not support BF16, falling back to FP16. \
+                        Note: FP16 may overflow in attention logits and produce lower-quality results."
                     );
                     DType::F16
                 };
```

<!-- skill_evals.py sync: 624caa96f39a2d0e -->
