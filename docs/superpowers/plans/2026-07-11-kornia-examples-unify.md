# kornia-rs Examples & README Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the two AI-looking examples (`color_spaces`, `image_api`) simple and human, showcase the unified host/GPU residency dispatch, and fix + extend the README's CUDA story — all verified building and running.

**Architecture:** Three independent edit targets (two example crates + the shared README), then one verification task that builds/runs every CUDA example and every README snippet. Each edit stands alone and is committed separately.

**Tech Stack:** Rust (kornia-image, kornia-imgproc, kornia-io, cudarc), Python (kornia_rs wheel in `venv995`), CUDA on Jetson Orin (aarch64).

## Global Constraints

- Repo lives at `/mnt/data/kornia-rs` (symlinked from `/home/nvidia/kornia-rs`); use either path.
- Branch before committing — currently on `main`. Create `feat/examples-readme-unify` first.
- **README prose must be clear, concise, and human — not AI-generated.** No marketing lines, no emoji beyond the existing section-header glyphs already in the file, no over-explaining. Short declarative sentences.
- Examples must contain **no emoji, no `print_separator`, no ASCII banners, no ✓/✨/📊 status narration.**
- Every code snippet in the README must reference **only shipped symbols**. These removed/renamed symbols must not appear anywhere in README.md: `cuda.upload`, `CudaImage`, `.download(`, `cuda.gray_from_rgb`, `cuda.from_dlpack`, `CudaPreprocessor`.
- New public Rust fns return named structs; new Python bindings return `#[pyclass(frozen)]` — N/A here (no new API), but do not regress.
- `cudarc` direct-dependency spec (copy verbatim, matches `examples/cuda_imgproc/Cargo.toml`):
  ```toml
  cudarc = { version = "0.19", default-features = false, features = [
    "cuda-version-from-build-system", "driver", "fallback-dynamic-loading",
    "fallback-latest", "nvrtc", "std",
  ], optional = true }
  ```
- venv activate: `source /tmp/claude-1000/-home-nvidia-kornia-rs/b6aebf82-b845-4858-a5c9-b2b01dc947bc/scratchpad/venv995/bin/activate`

---

### Task 0: Branch

- [ ] **Step 1: Create the working branch**

Run:
```bash
cd /home/nvidia/kornia-rs && git checkout -b feat/examples-readme-unify
```
Expected: `Switched to a new branch 'feat/examples-readme-unify'`

---

### Task 1: Rewrite `examples/color_spaces` (de-vibe + GPU residency dispatch)

**Files:**
- Modify: `examples/color_spaces/src/main.rs` (full replace, 53 → ~40 lines)
- Modify: `examples/color_spaces/Cargo.toml` (add optional `cuda` feature)

**Interfaces:**
- Consumes (all shipped, verified in the API audit): `Rgb8::to_cuda(&Arc<CudaStream>) -> Result<Rgb8, ImageError>`, `Gray8::zeros_cuda(ImageSize, &Arc<CudaStream>) -> Result<Gray8, ImageError>`, `Gray8::to_host(&Arc<CudaStream>) -> Result<Gray8, ImageError>`, `ConvertColor::convert(&self, &mut Dst)` (residency-dispatching), `kornia_io::functional::read_image_any_rgb8`.
- Produces: nothing downstream depends on it.

- [ ] **Step 1: Replace `examples/color_spaces/src/main.rs`**

```rust
//! Type-safe color conversions with residency dispatch.
//!
//! `convert` runs on the CPU, or on the GPU with `--features cuda`, selected by
//! where the images live. The call site is identical either way.

use kornia_image::color_spaces::Rgb8;
use kornia_imgproc::color::{ConvertColor, Gray8};
use kornia_io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rgb = F::read_image_any_rgb8("../../tests/data/dog.jpeg")?;

    // Host images -> CPU path.
    let mut gray = Gray8::from_size_val(rgb.size(), 0)?;
    rgb.convert(&mut gray)?;
    println!(
        "host: {}x{} rgb -> gray, first pixel {}",
        rgb.width(),
        rgb.height(),
        gray.as_slice()[0]
    );

    // The same convert() on device images runs the CUDA kernel.
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;

        let stream = CudaContext::new(0)?.default_stream();
        let rgb_gpu = rgb.to_cuda(&stream)?;
        let mut gray_gpu = Gray8::zeros_cuda(rgb.size(), &stream)?;
        rgb_gpu.convert(&mut gray_gpu)?;

        let gray_gpu_host = gray_gpu.to_host(&stream)?;
        assert_eq!(gray.as_slice(), gray_gpu_host.as_slice());
        println!("gpu:  same convert() on device, output matches the CPU result");
    }

    Ok(())
}
```

- [ ] **Step 2: Replace `examples/color_spaces/Cargo.toml`**

```toml
[package]
name = "color_spaces"
version = { workspace = true }
authors = { workspace = true }
edition = { workspace = true }
rust-version = { workspace = true }
description = "Example demonstrating type-safe color space conversions"
homepage = { workspace = true }
repository = { workspace = true }
license = { workspace = true }
publish = false

[features]
cuda = ["dep:cudarc", "kornia-image/cuda", "kornia-imgproc/cuda"]

[dependencies]
kornia-image = { workspace = true }
kornia-imgproc = { workspace = true }
kornia-io = { workspace = true }
cudarc = { version = "0.19", default-features = false, features = [
  "cuda-version-from-build-system", "driver", "fallback-dynamic-loading",
  "fallback-latest", "nvrtc", "std",
], optional = true }
```

- [ ] **Step 3: Build + run the host path**

Run:
```bash
cd /home/nvidia/kornia-rs/examples/color_spaces && cargo run
```
Expected: one `host: 258x195 rgb -> gray, first pixel <N>` line, no emoji, exit 0.

- [ ] **Step 4: Build + run the CUDA path**

Run:
```bash
cd /home/nvidia/kornia-rs/examples/color_spaces && cargo run --features cuda
```
Expected: the `host:` line plus `gpu:  same convert() on device, output matches the CPU result`. If the `assert_eq!` fails the process panics — that is the correctness gate.

- [ ] **Step 5: fmt + clippy**

Run:
```bash
cd /home/nvidia/kornia-rs && cargo fmt -p color_spaces && cargo clippy -p color_spaces --features cuda
```
Expected: no diff from fmt, no clippy warnings.

- [ ] **Step 6: Commit**

```bash
cd /home/nvidia/kornia-rs
git add examples/color_spaces/src/main.rs examples/color_spaces/Cargo.toml
git commit -m "docs(examples): rewrite color_spaces — human style + GPU residency dispatch"
```

---

### Task 2: Rewrite `examples/image_api` (de-vibe, trim 301 → ~55 lines)

**Files:**
- Modify: `examples/image_api/src/main.rs` (full replace)

**Interfaces:**
- Consumes (shipped): `Image::<T,C>::from_size_val`, `Image::<T,C>::new`, `Image::as_slice`, `Image::width/height`, `ImageError`, `ImageSize`.
- Produces: nothing downstream.

- [ ] **Step 1: Replace `examples/image_api/src/main.rs`**

```rust
//! Tour of the kornia-image `Image` API: construction, zero-copy access,
//! typed channels, and construction-time validation.
//!
//! Run with: cargo run -p image_api

use kornia_image::{Image, ImageError, ImageSize};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fill a 640x480 RGB image with a constant.
    let img = Image::<u8, 3>::from_size_val(ImageSize { width: 640, height: 480 }, 128)?;
    println!("filled {}x{}x3, first byte {}", img.width(), img.height(), img.as_slice()[0]);

    // Build a 10x10 grayscale gradient from a data vector.
    let gradient: Vec<u8> = (0..100).map(|i| (i * 255 / 100) as u8).collect();
    let grad = Image::<u8, 1>::new(ImageSize { width: 10, height: 10 }, gradient)?;
    println!("gradient {}x{}, last pixel {}", grad.width(), grad.height(), grad.as_slice()[99]);

    // as_slice() borrows the backing buffer — no copy. Pixels are interleaved.
    let rgb = Image::<u8, 3>::from_size_val(ImageSize { width: 5, height: 5 }, 42)?;
    let data = rgb.as_slice();
    let px = (2 * rgb.width() + 3) * 3; // row 2, col 3
    println!("pixel (2,3) = ({}, {}, {})", data[px], data[px + 1], data[px + 2]);

    // f32 and 4-channel images use the same API through the type parameters.
    let rgba = Image::<f32, 4>::from_size_val(ImageSize { width: 8, height: 8 }, 0.25)?;
    println!("rgba f32 {}x{}x4, {} elements", rgba.width(), rgba.height(), rgba.as_slice().len());

    // Construction checks the buffer length against the shape.
    let wrong = vec![0u8; 100]; // a 10x10x3 image needs 300 bytes
    match Image::<u8, 3>::new(ImageSize { width: 10, height: 10 }, wrong) {
        Ok(_) => unreachable!("wrong-sized data must not construct"),
        Err(e) => println!("rejected wrong-sized data: {e}"),
    }

    Ok(())
}
```

- [ ] **Step 2: Build + run**

Run:
```bash
cd /home/nvidia/kornia-rs && cargo run -p image_api
```
Expected: five plain lines (`filled ...`, `gradient ...`, `pixel (2,3) ...`, `rgba f32 ...`, `rejected wrong-sized data: ...`), no banner, no emoji, exit 0.

- [ ] **Step 3: fmt + clippy**

Run:
```bash
cd /home/nvidia/kornia-rs && cargo fmt -p image_api && cargo clippy -p image_api
```
Expected: no fmt diff, no clippy warnings. (`ImageError` is used by the type parameter on `match`; if clippy flags it unused, keep the import — it names the error type in the doc line and Result.)

- [ ] **Step 4: Commit**

```bash
cd /home/nvidia/kornia-rs
git add examples/image_api/src/main.rs
git commit -m "docs(examples): trim image_api to a concise, human API tour"
```

---

### Task 3: Fix + extend the README CUDA story

**Files:**
- Modify: `README.md` — replace the `### GPU / CUDA` section (currently lines 315–346, ending just before `## 🧑‍💻 Development`).

**Interfaces:**
- Consumes (shipped Python): `kornia_rs.cuda.is_available()`, `kornia_rs.cuda.Stream.default([device])`, `Image.from_numpy(...).to_cuda(stream)`, `Image.cpu()`, `Image.numpy()`, `kornia_rs.imgproc.gray_from_rgb(image)`, `kornia_rs.Preprocessor(...)`, `kornia_rs.IMAGENET_MEAN/STD`, `Tensor.data_ptr`, `torch.from_dlpack`.
- Consumes (shipped Rust): `Rgb8::to_cuda`, `Gray8::zeros_cuda`, `ConvertColor::convert`.

- [ ] **Step 1: Replace lines 315–346 of `README.md`**

Delete the existing `### GPU / CUDA` block (from the `### GPU / CUDA` heading through the `(`K.cuda.from_dlpack`) ...` paragraph, i.e. up to but not including `## 🧑‍💻 Development`) and insert the following markdown in its place:

~~~markdown
### GPU / CUDA

The published wheels are GPU-capable but load CUDA lazily: the same wheel runs on
CPU when no GPU is present and uses the GPU when one is. The GPU path needs an
NVIDIA driver (`libcuda`) and `nvrtc` from the CUDA toolkit; without them the CPU
ops keep working.

Device pixels use the same `Image` type. `.device` reads `"cpu"` or `"cuda:{id}"`,
`.to_cuda(stream)` uploads, `.cpu()` downloads. Color ops live under
`kornia_rs.imgproc` and dispatch on residency: a device `Image` runs the CUDA
kernel, a host `Image` or numpy array runs the CPU kernel.

```python
import numpy as np
import kornia_rs as K
from kornia_rs.image import Image
from kornia_rs.cuda import Stream

if K.cuda.is_available():
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    img = Image.from_numpy(rgb).to_cuda(Stream.default())  # -> "cuda:0"
    gray = K.imgproc.gray_from_rgb(img)                    # runs on the GPU
    out = gray.cpu().numpy()                               # -> host, (480, 640, 1)
```

GPU color conversions (`gray_from_rgb`, `bgr_from_rgb`, `hsv_from_rgb`,
`lab_from_rgb`, `ycbcr_from_rgb`, `sepia_from_rgb`, `apply_colormap`, …) and the
fused `Preprocessor` are the GPU entry points. Tensors cross to PyTorch with no
copy through DLPack (`torch.from_dlpack`) and `__cuda_array_interface__`.

### Production: GPU-resident camera → model

`Preprocessor` fuses resize, normalize and HWC→CHW into one CUDA kernel per
frame. It emits a device tensor that feeds an inference engine with no host copy
— the path for real-time camera pipelines.

```python
import torch
from kornia_rs import Preprocessor, IMAGENET_MEAN, IMAGENET_STD
from kornia_rs.cuda import Stream

# One kernel per frame: NV12 -> normalized fp16 [1, 3, 640, 640] on the GPU.
pre = Preprocessor(mode="letterbox", format="nv12", f16=True,
                   mean=IMAGENET_MEAN, std=IMAGENET_STD, stream=Stream.default(0))

t = pre.run(nv12_frame, 1920, 1080, 640, 640)  # device Tensor
x = torch.from_dlpack(t)                        # zero-copy handoff to PyTorch
# TensorRT: ctx.set_tensor_address("images", t.data_ptr)
```

The same one-call-per-residency model holds in Rust — `convert` picks CPU or GPU
from where the images live:

```rust
let stream = CudaContext::new(0)?.default_stream();
let rgb = Rgb8::from_size_vec(size, data)?.to_cuda(&stream)?;  // device image
let mut gray = Gray8::zeros_cuda(size, &stream)?;
rgb.convert(&mut gray)?;                                       // runs on the GPU
```

Full pipelines: [`examples/cuda_camera_preprocess`](examples/cuda_camera_preprocess)
(V4L2 camera → fused CUDA preprocess) and
[`kornia-py/examples/preprocess_to_inference.py`](kornia-py/examples/preprocess_to_inference.py)
(NV12 → fused preprocess → ResNet-18 / TensorRT, GPU-resident end to end).
~~~

- [ ] **Step 2: Verify no removed symbol survives**

Run:
```bash
cd /home/nvidia/kornia-rs && grep -nE 'cuda\.upload|CudaImage|\.download\(|cuda\.gray_from_rgb|cuda\.from_dlpack|CudaPreprocessor' README.md
```
Expected: no output (exit 1). Any hit is a failure.

- [ ] **Step 3: Run the GPU/CUDA snippet (3a) verbatim**

Create `/tmp/claude-1000/-home-nvidia-kornia-rs/b6aebf82-b845-4858-a5c9-b2b01dc947bc/scratchpad/readme_gpu.py` with the exact 3a Python block above, then:
```bash
source /tmp/claude-1000/-home-nvidia-kornia-rs/b6aebf82-b845-4858-a5c9-b2b01dc947bc/scratchpad/venv995/bin/activate
python /tmp/claude-1000/-home-nvidia-kornia-rs/b6aebf82-b845-4858-a5c9-b2b01dc947bc/scratchpad/readme_gpu.py && echo SNIPPET_OK
```
Expected: `SNIPPET_OK` (no output from the snippet itself; add `print(out.shape)` locally if you want to eyeball `(480, 640, 1)`). If `K.imgproc` is not resolvable as an attribute, change the call to `from kornia_rs import imgproc` + `imgproc.gray_from_rgb(img)` in BOTH the scratch file and the README, mirroring `kornia-py/examples/image_cuda_basics.py`.

- [ ] **Step 4: Run the production snippet (3b) with a synthetic frame**

Create `/tmp/.../scratchpad/readme_prod.py` = the 3b Python block with two edits so it runs standalone: add `import numpy as np` and, before `pre.run`, `nv12_frame = np.random.randint(0, 255, (1920 * 1080 * 3 // 2,), dtype=np.uint8)`. Then:
```bash
python /tmp/claude-1000/-home-nvidia-kornia-rs/b6aebf82-b845-4858-a5c9-b2b01dc947bc/scratchpad/readme_prod.py && echo PROD_OK
```
Expected: `PROD_OK`. This proves the `Preprocessor(...)`, `run(...)`, and `torch.from_dlpack(t)` calls are valid against the shipped API. (The README keeps the undefined `nv12_frame` for readability — it is illustrative, like the existing examples; the run only validates the API surface.)

- [ ] **Step 5: Commit**

```bash
cd /home/nvidia/kornia-rs
git add README.md
git commit -m "docs(readme): fix removed CUDA API, add GPU-resident production section"
```

---

### Task 4: Verify the untouched CUDA examples still build & run

**Files:** none modified (verification only; if something is broken, stop and report before editing).

- [ ] **Step 1: cuda_imgproc (Rust) build + run**

Run:
```bash
cd /home/nvidia/kornia-rs/examples/cuda_imgproc && cargo run
```
Expected: RGB→gray output, exit 0. (This example already builds `kornia-image`/`kornia-tensor` with `cuda` and was fixed earlier in the session.)

- [ ] **Step 2: cuda_camera_preprocess (Rust) build only**

Run:
```bash
cd /home/nvidia/kornia-rs && cargo build -p cuda-camera-preprocess --features cuda
```
Expected: `Finished`. Running needs a `/dev/video*` V4L2 camera; if none is present, note "run skipped: no camera" — build success is the gate here.

- [ ] **Step 3: Python CUDA examples run**

Run:
```bash
source /tmp/claude-1000/-home-nvidia-kornia-rs/b6aebf82-b845-4858-a5c9-b2b01dc947bc/scratchpad/venv995/bin/activate
cd /home/nvidia/kornia-rs/kornia-py
python examples/image_cuda_basics.py && python examples/cuda_preprocess_tensorrt.py
```
Expected: each prints its `OK` line, exit 0. (`preprocess_to_inference.py` needs torchvision/TensorRT; run it too if those import, else note "skipped: torchvision/TRT not installed".)

- [ ] **Step 4: Final formatting gate across the touched crates**

Run:
```bash
cd /home/nvidia/kornia-rs && cargo fmt --all --check
```
Expected: no output (clean).

- [ ] **Step 5: Push the branch**

```bash
cd /home/nvidia/kornia-rs && git push -u origin feat/examples-readme-unify
```
Expected: branch pushed; open a PR if desired.

---

## Self-Review

**Spec coverage:**
- Change 1 (color_spaces de-vibe + GPU arm) → Task 1. ✓
- Change 2 (image_api trim) → Task 2. ✓
- Change 3a (fix broken README CUDA) → Task 3 Step 1–2. ✓
- Change 3b (production subsection + Rust snippet) → Task 3 Step 1. ✓
- Verification (5 CUDA examples + README snippets + grep guard + fmt/clippy) → Tasks 1.3–1.5, 2.2–2.3, 3.2–3.4, 4. ✓
- README human/concise constraint → Global Constraints + Task 3 prose. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete content; the one runtime unknown (`K.imgproc` attribute vs `from kornia_rs import imgproc`) has an explicit fallback instruction, not a placeholder.

**Type consistency:** `to_cuda`/`zeros_cuda`/`to_host`/`convert` signatures match the API audit; `Image::new`/`from_size_val`/`as_slice` match the original example's usage; README symbols all cross-checked against the shipped stubs.
