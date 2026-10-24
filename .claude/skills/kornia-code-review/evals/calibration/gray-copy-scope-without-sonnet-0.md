**Review: refactor(imgproc): simplify gray_from_rgb u8 validation (#9999)**

I'd request changes. I only read the diff and the surrounding code; I didn't build or run anything.

**1. The PR adds a full-image copy on the hot path (`color/gray/mod.rs`)**
- `let src_buf = src.as_slice().to_vec();` allocates and copies the whole RGB image on every call. For a 1080p image that is about 6 MB per call.
- The description says this gives "better cache locality" and is "slightly faster". That is the wrong way round. The kernel already reads `src.as_slice()` directly, and a copy only adds an allocation and memory traffic. It also undoes the point of having a caller-provided `dst` buffer.
- No benchmark is included to support the claim. This should be removed. The kernel call should stay `kernels::gray_from_rgb_u8(src.as_slice(), dst.as_slice_mut(), npixels)`.

**2. The inlined check is weaker than `check_size` and duplicates it**
- `check_size` in `color/kernel_common.rs` compares `src.size() != dst.size()`, meaning width and height. The new code compares only `rows*cols` products. A 4×6 source and a 6×4 destination, or 2×12 and 3×8, now pass validation. The kernel then runs on a transposed or reshaped layout without any error.
- That is a behavior change in validation, and the PR presents it as a pure refactor.
- The error construction is copy-pasted from `check_size`. The other impls in the same file (lines 44, 52, 178, 203 and 211) still call the helper, so this one is now inconsistent with them.
- The issue asks for "readability", but inlining a shared helper makes the code longer and harder to maintain. I'd drop this change. If the maintainers do want it, keep the exact `src.size() != dst.size()` comparison.

**3. Unrelated change in `flip.rs`**
- The rename of `cols` to `width` in `hflip_rgb_u8_scalar` has nothing to do with the PR title or the issue. It's harmless, but it is scope creep and should go in a separate PR or be dropped. The function is `#[allow(dead_code)]`, and its callers pass the arguments positionally, so nothing breaks.

**4. PR hygiene**
- The description says the validation is "simpler", but the diff adds lines and loses the shared helper.
- "cargo test -p kornia-imgproc passes" doesn't cover the new gap. No test checks a mismatched-shape, same-pixel-count pair, and none could pass the old `check_size`. A regression test for that case is needed if any validation change stays.
- The AI disclosure says "Skills applied: kornia-coding-style" and "human reviewed". The `to_vec` and the claimed speedup suggest the code wasn't reviewed against the performance rules for this crate.

**Recommendation:** Close this, or reduce it to nothing. The original code was already correct and efficient, and both functional changes in the diff make it worse: a needless allocation and a looser size check.
