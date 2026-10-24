title:	[Bug]: distance_transform and compute_histogram panic on zero-width images
state:	OPEN
author:	Aatman09
labels:	bug, help wanted
assignees:	Aatman09
projects:
milestone:
number:	1163
--
### 🐛 Describe the bug

`DistanceTransformExecutor::execute` and `compute_histogram` in `kornia-imgproc` panic when the input image has `width == 0`.

`Image::new` accepts a zero-width image without error, so the panic can be reached through the public API. Both functions split the pixel buffer with rayon's `par_chunks` using a chunk size derived from the width, and rayon panics when the chunk size is 0:

- `crates/kornia-imgproc/src/distance_transform.rs:75`: `par_chunks_mut(width)`
- `crates/kornia-imgproc/src/histogram.rs:74`: `par_chunks(ROWS_PER_TASK * width)`


### 🔄 Steps to Reproduce

```bash
1. Create an Image<f32, 1> or Image<u8, 1> with ImageSize { width: 0, height: 5 } and an empty data vector. Image::new returns Ok.
2. Pass it to DistanceTransformExecutor::new().execute(&img) or to compute_histogram(&img, &mut hist, 4).
3. Both calls panic with `chunk_size must not be zero` instead of returning.
```

### 💻 Minimal Code Example

```rust
use kornia_image::{Image, ImageSize};
use kornia_imgproc::{distance_transform::DistanceTransformExecutor, histogram::compute_histogram};

#[test]
fn dt_zero_width() {
    let img = Image::<f32, 1>::new(ImageSize { width: 0, height: 5 }, vec![]).unwrap();
    let _ = DistanceTransformExecutor::new().execute(&img); // panics
}

#[test]
fn hist_zero_width() {
    let img = Image::<u8, 1>::new(ImageSize { width: 0, height: 5 }, vec![]).unwrap();
    let mut hist = vec![0usize; 4];
    let _ = compute_histogram(&img, &mut hist, 4); // panics
}
```

### ✅ Expected behavior

No panic. Either:
- return early: `compute_histogram` leaves `hist` unchanged, and `execute` returns an empty image of the same size; or
- return an `ImageError` for empty input.

Happy to go with whichever the maintainers prefer. Returning early seems more natural, since `Image::new` treats an empty image as valid.


### ❌ Actual behavior

thread dt_zero_width panicked at crates/kornia-imgproc/src/distance_transform.rs:75:14:
chunk_size must not be zero

thread 'hist_zero_width' panicked at crates/kornia-imgproc/src/histogram.rs:74:10:
chunk_size must not be zero


### 🔧 Environment

```shell
- kornia-rs version: 0.2.0 (main @ be0ac0cb)
- Rust version (`rustc -V`): rustc 1.96.0 (ac68faa20 2026-05-25)
- Cargo version (`cargo -V`): cargo 1.96.0 (30a34c682 2026-05-25)
- OS (e.g., Linux, macOS, Windows): Linux
- Target architecture (if cross-compiling): x86_64 (not cross-compiling)
- Python version (if using Python bindings): N/A
```

### 📝 Additional context

- Other functions split rows by width the same way and may have the same issue: `features/fast.rs`, `features/responses.rs`, `filter/ops.rs`, `color/yuv/kernels.rs`. I haven't checked whether they already guard against `width == 0`; I can check them as part of the fix.
- Related but separate: for an image with no foreground pixels, `DistanceTransformExecutor::execute` returns `1e10`, while `distance_transform_vanilla` returns `f32::MAX`. Not part of this fix; mentioning it in case it should be tracked.


### 🤝 Contribution Intent

- [x] I plan to submit a PR to fix this bug
- [ ] I'm reporting this bug but not planning to fix it
