# kornia-rs: low level computer vision library in Rust

English | [简体中文](README.zh-CN.md)

![Crates.io Version](https://img.shields.io/crates/v/kornia)
[![PyPI version](https://badge.fury.io/py/kornia-rs.svg)](https://badge.fury.io/py/kornia-rs)
[![Documentation](https://img.shields.io/badge/docs.rs-kornia-orange)](https://docs.rs/kornia)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/HfnywwpBnD)

The `kornia` crate is a low level library for Computer Vision written in [Rust](https://www.rust-lang.org/) 🦀

Use the library to perform image I/O, visualization and other low level operations in your machine learning and data-science projects in a thread-safe and efficient way.

## 📚 Table of Contents

- [Getting Started](#getting-started)
- [Features](#features)
- [Installation](#️-installation)
- [Examples](#examples-image-processing)
- [Python Usage](#python-usage)
- [Development](#-development)
- [Contributing](#-contributing)
- [Citation](#citation)

## Getting Started

### Quick Example

The following example demonstrates how to read and display image information:

```rust
use kornia::image::Image;
use kornia::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image: Image<u8, 3, _> = F::read_image_any_rgb8("tests/data/dog.jpeg")?;

    println!("Hello, world! 🦀");
    println!("Loaded Image size: {:?}", image.size());
    println!("\nGoodbyte!");

    Ok(())
}
```

```bash
Hello, world! 🦀
Loaded Image size: ImageSize { width: 258, height: 195 }

Goodbyte!
```

## Features

- 🦀 The library is primarily written in [Rust](https://www.rust-lang.org/).
- 🚀 Multi-threaded and efficient image I/O, image processing and advanced computer vision operators.
- 🔢 Efficient Tensor and Image API for deep learning and scientific computing.
- 🐍 Python bindings are created with [PyO3/Maturin](https://github.com/PyO3/maturin).
- 📦 We package with support for Linux [amd64/arm64], macOS and Windows.
- Supported Python versions are 3.7/3.8/3.9/3.10/3.11/3.12/3.13, including the free-threaded build.

### Supported image formats

- Read images from AVIF, BMP, DDS, Farbeld, GIF, HDR, ICO, JPEG (libjpeg-turbo), OpenEXR, PNG, PNM, TGA, TIFF, WebP.

### Image processing

- Convert images to grayscale, resize, crop, rotate, flip, pad, normalize, denormalize, and other image processing operations.

### Video processing

- Capture video frames from a camera and video writers.

## 🛠️ Installation

### 🦀 Rust

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia = "0.1"
```

Alternatively, you can use each sub-crate separately:

```toml
[dependencies]
kornia-tensor = "0.1"
kornia-tensor-ops = "0.1"
kornia-io = "0.1"
kornia-image = "0.1"
kornia-imgproc = "0.1"
kornia-icp = "0.1"
kornia-3d = "0.1"
kornia-apriltag = "0.1"
kornia-vlm = "0.1"
kornia-nn = "0.1"
kornia-algebra = "0.1"
```

### 🐍 Python

```bash
pip install kornia-rs
```

A subset of the full rust API is exposed. See the [kornia documentation](https://kornia.readthedocs.io/en/stable/) for more detail about the API for python functions and objects exposed by the `kornia-rs` Python module.

The `kornia-rs` library is thread-safe for use under the free-threaded Python build.

### System Dependencies (Optional)

Depending on the features you want to use, you might need to install the following dependencies in your system:

#### v4l (Video4Linux camera support)

```bash
sudo apt-get install clang
```

#### turbojpeg

```bash
sudo apt-get install nasm
```

#### gstreamer

```bash
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

**Note:** Check the [gstreamer installation guide](https://docs.rs/gstreamer/latest/gstreamer/#installation) for more details.

## Examples: Image Processing

The following example shows how to read an image, convert it to grayscale and resize it. The image is then logged to a [`rerun`](https://github.com/rerun-io/rerun) recording stream for visualization.

For more examples and use cases, check out the [`examples`](https://github.com/kornia/kornia-rs/tree/main/examples) directory, which includes:
- Image processing operations (resize, rotate, normalize, filters)
- Video capture and processing
- AprilTag detection
- Feature detection (FAST)
- Visual language models (VLM) integration
- And more...

```rust
use kornia::{image::{Image, ImageSize}, imgproc};
use kornia::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image: Image<u8, 3, _> = F::read_image_any_rgb8("tests/data/dog.jpeg")?;
    let image_viz = image.clone();

    let image_f32: Image<f32, 3, _> = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // convert the image to grayscale
    let mut gray = Image::<f32, 1, _>::from_size_val(image_f32.size(), 0.0)?;
    imgproc::color::gray_from_rgb(&image_f32, &mut gray)?;

    // resize the image
    let new_size = ImageSize {
        width: 128,
        height: 128,
    };

    let mut gray_resized = Image::<f32, 1, _>::from_size_val(new_size, 0.0)?;
    imgproc::resize::resize_native(
        &gray, &mut gray_resized,
        imgproc::interpolation::InterpolationMode::Bilinear,
    )?;

    println!("gray_resize: {:?}", gray_resized.size());

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").spawn()?;

    rec.log(
        "image",
        &rerun::Image::from_elements(
            image_viz.as_slice(),
            image_viz.size().into(),
            rerun::ColorModel::RGB,
        ),
    )?;

    rec.log(
        "gray",
        &rerun::Image::from_elements(gray.as_slice(), gray.size().into(), rerun::ColorModel::L),
    )?;

    rec.log(
        "gray_resize",
        &rerun::Image::from_elements(
            gray_resized.as_slice(),
            gray_resized.size().into(),
            rerun::ColorModel::L,
        ),
    )?;

    Ok(())
}
```

![Screenshot from 2024-03-09 14-31-41](https://github.com/kornia/kornia-rs/assets/5157099/afdc11e6-eb36-4fcc-a6a1-e2240318958d)

## Python Usage

### Reading Images

Load an image, which is converted directly to a numpy array to ease the integration with other libraries.

```python
import kornia_rs as K
import numpy as np
import torch

# load an image with using libjpeg-turbo
img: np.ndarray = K.read_image_jpeg("dog.jpeg")

# alternatively, load other formats
# img: np.ndarray = K.read_image_any("dog.png")

assert img.shape == (195, 258, 3)

# convert to dlpack to import to torch
img_t = torch.from_dlpack(img)
assert img_t.shape == (195, 258, 3)
```

### Writing Images

Write an image to disk:

```python
import kornia_rs as K
import numpy as np

# load an image with using libjpeg-turbo
img: np.ndarray = K.read_image_jpeg("dog.jpeg")

# write the image to disk
K.write_image_jpeg("dog_copy.jpeg", img)
```

### Encoding and Decoding

Encode or decode image streams using the `turbojpeg` backend:

```python
import kornia_rs as K

# load image with kornia-rs
img = K.read_image_jpeg("dog.jpeg")

# encode the image with jpeg
image_encoder = K.ImageEncoder()
image_encoder.set_quality(95)  # set the encoding quality

# get the encoded stream
img_encoded: list[int] = image_encoder.encode(img)

# decode back the image
image_decoder = K.ImageDecoder()

decoded_img: np.ndarray = image_decoder.decode(bytes(img_encoded))
```

### Image Resizing

Resize an image using the `kornia-rs` backend with SIMD acceleration:

```python
import kornia_rs as K

# load image with kornia-rs
img = K.read_image_jpeg("dog.jpeg")

# resize the image
resized_img = K.resize(img, (128, 128), interpolation="bilinear")

assert resized_img.shape == (128, 128, 3)
```

## 🧑‍💻 Development

### Prerequisites

Before you begin, ensure you have `rust` and `python3` installed on your system.

### Setting Up Your Development Environment

1. **Install Rust** using rustup:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

2. **Install [`uv`](https://docs.astral.sh/uv/)** to manage Python dependencies:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install [`just`](https://github.com/casey/just)** command runner for managing development tasks:
   ```bash
   cargo install just
   ```

4. **Clone the repository** to your local directory:
   ```bash
   git clone https://github.com/kornia/kornia-rs.git
   ```

### Available Commands

You can check all available development commands by running `just` in the root directory of the project:

```bash
$ just
Available recipes:
    check-environment                 # Check if the required binaries for the project are installed
    clean                             # Clean up caches and build artifacts
    clippy                            # Run clippy with all features
    clippy-default                    # Run clippy with default features
    fmt                               # Run autoformatting and linting
    py-build py_version='3.9'         # Create virtual environment, and build kornia-py
    py-build-release py_version='3.9' # Create virtual environment, and build kornia-py for release
    py-install py_version='3.9'       # Create virtual environment, and install dev requirements
    py-test                           # Test the kornia-py code with pytest
    test name=''                      # Test the code or a specific test
```
### 🐳 Development Container

This project includes a development container configuration for a consistent development environment across different machines.

**Using the Dev Container:**

1. Install the `Remote - Containers` extension in Visual Studio Code
2. Open the project folder in VS Code
3. Press `F1` and select `Remote-Containers: Reopen in Container`
4. VS Code will build and open the project in the containerized environment

The devcontainer includes all necessary dependencies and tools for building and testing `kornia-rs`.

### 🦀 Rust Development

Compile the project and run all tests:

```bash
just test
```

To run specific tests:

```bash
just test image
```

To run clippy linting:

```bash
just clippy
```

### 🐍 Python Development

Build Python wheels using `maturin`:

```bash
just py-build
```

Run Python tests:

```bash
just py-test
```

## 💜 Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Coding standards and style guidelines
- Development workflow
- How to run local checks before submitting PRs

### AI Policy

Kornia-rs accepts AI-assisted code but strictly rejects AI-generated contributions where the submitter acts as a proxy. All contributors must be the **Sole Responsible Author** for every line of code. Please review our [AI Policy](AI_POLICY.md) before submitting pull requests. Key requirements include:

- **Proof of Verification**: PRs must include local test logs proving execution (e.g., `pixi run rust-test` or `cargo test`)
- **Pre-Discussion**: All PRs must be discussed in Discord or via a GitHub issue before implementation
- **Library References**: Implementations must be based on existing library references (Rust crates, OpenCV, etc.)
- **Use Existing Utilities**: Use existing `kornia-rs` utilities instead of reinventing the wheel
- **Error Handling**: Use `Result<T, E>` for error handling (avoid `unwrap()`/`expect()` in library code)
- **Explain It**: You must be able to explain any code you submit

Automated AI reviewers (e.g., @copilot) will check PRs against these policies. See [AI_POLICY.md](AI_POLICY.md) for complete details.

### Community

This is a child project of [Kornia](https://github.com/kornia/kornia).

- 💬 Join our community on [Discord](https://discord.gg/HfnywwpBnD)
- 💖 Support the project on [OpenCollective](https://opencollective.com/kornia)
- 📖 Read the full [documentation](https://kornia.readthedocs.io/en/stable/)
- 🦀 Browse the [Rust API docs](https://docs.rs/kornia)

## Citation

If you use kornia-rs in your research, please cite:

```bibtex
@misc{2505.12425,
Author = {Edgar Riba and Jian Shi and Aditya Kumar and Andrew Shen and Gary Bradski},
Title = {Kornia-rs: A Low-Level 3D Computer Vision Library In Rust},
Year = {2025},
Eprint = {arXiv:2505.12425},
}
```
