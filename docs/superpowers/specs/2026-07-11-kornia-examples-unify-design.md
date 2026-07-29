# Design: Simplify kornia-rs examples & showcase the unified CUDA API

## Context

PR #995 unified kornia-rs's device API (Rust `Image::to_cuda` now returns a
device `Image`; `ConvertColor::convert` dispatches host↔device by residency;
Python `Image`/`Tensor`/`Preprocessor` with `stream=` selecting the device).

An audit of all 47 Rust examples + 3 Python CUDA examples + 20 benchmarks found:

- **No stale/deprecated API anywhere** — every CUDA example is already on the
  post-#995 surface (`to_cuda`/`.cpu()`/`.numpy()`, `imgproc.*` residency color
  ops, `Preprocessor(stream=, ..., consumer_stream=)`). No migration needed.
- **Two examples read as AI-generated ("vibe coded")**:
  - `examples/color_spaces` — emoji + marketing narration
    (`println!("🎨 Type-Safe Color Space API Demo")`, `"✨ All conversions
    type-safe at compile time!"`, `"// now returns Rgb8 directly!"`).
  - `examples/image_api` — 301 lines with a `print_separator()` helper and
    ✓/✨/📊 status prints on every trivial step.
- **A demonstration gap**: no Rust example shows the headline unified feature —
  the *same* `convert()` call running on GPU by virtue of the operands being
  device-resident. `cuda_imgproc` uses a hand-written NVRTC kernel (low-level);
  `color_spaces` uses `convert()` but host-only.

Goal: make the two offending examples simple and human, close the demonstration
gap in `color_spaces`, fix the README's broken CUDA section and give it a
production showcase (fused preprocess → zero-copy inference), and verify every
CUDA example and README snippet actually builds and runs on this box.

## Scope

**In scope** (edit): `examples/color_spaces`, `examples/image_api`, and
`README.md` (the top-level readme; `kornia-py/README.md` is a symlink to it).
**Verify only** (no edit unless broken): `examples/cuda_imgproc`,
`examples/cuda_camera_preprocess`, and the 3 Python CUDA examples
(`image_cuda_basics.py`, `cuda_preprocess_tensorrt.py`,
`preprocess_to_inference.py`).
**Out of scope**: the other 45 Rust examples and all 20 benchmarks — already
clean and on the new API. No unrelated refactoring.

### README is currently broken (motivates Change 3)

The README's **"GPU / CUDA"** Python section uses the exact API that PR #995
**removed or renamed** — every snippet there fails against the shipped library:

- `K.cuda.upload(rgb)` → `CudaImage` (type removed)
- `cu_img.download()` (method removed)
- `K.cuda.gray_from_rgb(...)` (GPU color ops no longer under `cuda.*`)
- `K.cuda.from_dlpack(...)` (removed)
- `CudaPreprocessor` (renamed `Preprocessor`)

A user copy-pasting the "production GPU" section gets `AttributeError`. Fixing
this is a correctness issue, not just polish.

## Change 1 — `examples/color_spaces` (de-vibe + unified host/GPU dispatch)

Rewrite `src/main.rs` (53 → ~35 lines). Remove every emoji and narration line.
Keep the typed `ConvertColor` demo (Rgb8 → Gray8 → Rgb8) with plain output, then
add a `#[cfg(feature = "cuda")]` block that runs the **identical `convert()`
call** on device images and checks it bit-exactly matches the host result:

```rust
//! Type-safe color conversions with residency dispatch: the same `convert()`
//! call runs on the CPU or, with `--features cuda`, on the GPU — residency is
//! chosen by where the image lives, not by a different API.

use kornia_image::color_spaces::Rgb8;
use kornia_imgproc::color::{ConvertColor, Gray8};
use kornia_io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rgb = F::read_image_any_rgb8("../../tests/data/dog.jpeg")?;

    let mut gray = Gray8::from_size_val(rgb.size(), 0)?;
    rgb.convert(&mut gray)?; // host operands -> CPU path
    println!("host gray {}x{}, first px {}", gray.width(), gray.height(), gray.as_slice()[0]);

    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::CudaContext;
        let stream = CudaContext::new(0)?.default_stream();
        let rgb_dev = rgb.to_cuda(&stream)?;
        let mut gray_dev = Gray8::zeros_cuda(rgb.size(), &stream)?;
        rgb_dev.convert(&mut gray_dev)?; // identical call, device operands -> GPU kernel
        let gray_host = gray_dev.to_host(&stream)?;
        assert_eq!(gray.as_slice(), gray_host.as_slice());
        println!("gpu  gray matches host bit-for-bit");
    }

    Ok(())
}
```

`Cargo.toml`: add an optional `cuda` feature mirroring `examples/cuda_imgproc`:

```toml
[features]
cuda = ["kornia-image/cuda", "kornia-imgproc/cuda", "dep:cudarc"]

[dependencies]
cudarc = { workspace = true, optional = true, default-features = false, features = ["cuda-version-from-build-system"] }
```

(Exact cudarc feature spec copied from `examples/cuda_imgproc/Cargo.toml` during
implementation so it matches the toolchain the repo already builds against.)

## Change 2 — `examples/image_api` (de-vibe, trim)

Rewrite `src/main.rs` (301 → ~80 lines). Remove `print_separator()`, all emoji,
and the per-step narration. Keep a focused, human tour with sparse comments:

- create from a fill value (`Image::<u8,3>::from_size_val`)
- create from a data `Vec` (`Image::<u8,1>::new`)
- zero-copy access (`as_slice`) + one indexed-pixel read
- one error case (wrong-sized data → matched `ImageError`)
- an `f32` / 4-channel variant to show the type parameters

Plain `println!` reporting values, no status glyphs. Keep `main()` returning
`Result<(), ImageError>` and `?`-propagating; drop the one intentional error into
a small `match` as today, minus the narration.

## Change 3 — `README.md` (fix broken CUDA section + production showcase)

Keep the README's existing structure, tone, and emoji headers (intentional
project branding — do **not** strip). Two edits:

**3a. Replace the broken "GPU / CUDA" Python section** with the shipped unified
API. Every snippet must run against the current wheel:

```python
import numpy as np
import kornia_rs as K
from kornia_rs.image import Image
from kornia_rs.cuda import Stream

if K.cuda.is_available():
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Host -> GPU: one Image type, residency read via .device.
    img = Image.from_numpy(rgb).to_cuda(Stream.default())   # "cuda:0"

    # GPU color op: same imgproc.* entry point as the CPU path, dispatched
    # on residency (a device Image runs the CUDA kernel).
    gray = K.imgproc.gray_from_rgb(img)                      # device Image
    out = gray.cpu().numpy()                                 # GPU -> host
    assert out.shape == (480, 640, 1)
```

**3b. Add a short "Production: GPU-resident camera → model" subsection** that
shows the headline capability — a fused preprocess feeding an inference engine
with zero host copy — anchored to the real `preprocess_to_inference.py` /
`cuda_preprocess_tensorrt.py` examples:

```python
from kornia_rs import Preprocessor, IMAGENET_MEAN, IMAGENET_STD
from kornia_rs.cuda import Stream
import torch

# Fused resize + normalize + HWC->CHW (fp16) in one CUDA kernel, per frame.
pre = Preprocessor(mode="letterbox", format="nv12", f16=True,
                   mean=IMAGENET_MEAN, std=IMAGENET_STD, stream=Stream.default(0))

t = pre.run(nv12_frame, 1920, 1080, 640, 640)   # -> device Tensor [1,3,640,640]
x = torch.from_dlpack(t)                          # zero-copy handoff to PyTorch
# or bind straight to TensorRT: ctx.set_tensor_address("images", t.data_ptr)
```

Add a compact Rust CUDA snippet too (mirrors the rewritten `color_spaces`),
showing residency dispatch — the same `convert()` on host or device:

```rust
let stream = CudaContext::new(0)?.default_stream();
let rgb = Rgb8::from_size_vec(size, data)?.to_cuda(&stream)?;   // device Image
let mut gray = Gray8::zeros_cuda(size, &stream)?;
rgb.convert(&mut gray)?;                                        // runs on the GPU
```

Keep it tight: 3a replaces the stale block in place; 3b is one new subsection of
~25 lines. Link to the `examples/` CUDA programs for the full pipelines rather
than inlining them.

## Verification (end-to-end, on this box)

Repo now lives at `/mnt/data/kornia-rs` (symlinked from `/home/nvidia/kornia-rs`).

1. `cargo run -p color_spaces` (host path).
2. `cargo run -p color_spaces --features cuda` — prints "gpu gray matches host
   bit-for-bit"; the `assert_eq!` is the correctness gate.
3. `cargo run -p image_api` — runs clean, no glyphs in output.
4. `cargo run -p cuda_imgproc --features cuda` (already fixed this session; run
   to confirm the RGB→gray output).
5. `cargo build -p cuda-camera-preprocess --features cuda` — build only (running
   needs a `/dev/video*` V4L2 camera; note if absent).
6. Python (in `venv995`, cuda wheel already installed):
   `python examples/image_cuda_basics.py`, `cuda_preprocess_tensorrt.py`,
   `preprocess_to_inference.py` — each prints its OK line.
7. `cargo fmt -p color_spaces -p image_api` + `cargo clippy` on both.
8. **README snippets actually run**: extract each Python snippet from the edited
   README into a scratch script and run it in `venv995` (the `is_available()`
   guard + `imgproc.gray_from_rgb` + `Preprocessor` + `torch.from_dlpack` paths).
   Compile the Rust README snippet as a throwaway `cargo` target (or fold it into
   the `color_spaces` build). No snippet may reference a removed symbol
   (`grep -nE 'cuda\.upload|CudaImage|\.download\(|cuda\.gray_from_rgb|cuda\.from_dlpack|CudaPreprocessor' README.md` → zero hits).

## Success criteria

- `color_spaces` and `image_api` contain no emoji, no `print_separator`, no
  marketing narration; each is short and reads as human-written.
- `color_spaces --features cuda` demonstrates the unified dispatch and asserts
  GPU==CPU output.
- All five CUDA examples build; all that can run without extra hardware run
  green.
- README's GPU/CUDA section references **only** shipped symbols (grep guard
  above passes) and every snippet runs; the new production subsection shows the
  fused-preprocess → zero-copy-inference path. Project branding/emoji headers
  preserved.
