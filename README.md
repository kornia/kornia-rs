# kornia-rs: low level computer vision library in Rust

![Crates.io Version](https://img.shields.io/crates/v/kornia)
[![PyPI version](https://badge.fury.io/py/kornia-rs.svg)](https://badge.fury.io/py/kornia-rs)
[![Documentation](https://img.shields.io/badge/docs.rs-kornia-orange)](https://docs.rs/kornia)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENCE)
[![Discord](https://img.shields.io/badge/Discord-5865F2?logo=discord&logoColor=white)](https://discord.gg/HfnywwpBnD)

The `kornia` crate is a low level library for Computer Vision written in [Rust](https://www.rust-lang.org/) 🦀

Use the library to perform image I/O, visualisation and other low level operations in your machine learning and data-science projects in a thread-safe and efficient way.

## Getting Started

`cargo run --bin hello_world -- --image-path path/to/image.jpg`

```rust
use kornia::image::Image;
use kornia::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image: Image<u8, 3> = F::read_image_any_rgb8("tests/data/dog.jpeg")?;

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

- 🦀The library is primarly written in [Rust](https://www.rust-lang.org/).
- 🚀 Multi-threaded and efficient image I/O, image processing and advanced computer vision operators.
- 🔢 Efficient Tensor and Image API for deep learning and scientific computing.
- 🐍 Python bindings are created with [PyO3/Maturin](https://github.com/PyO3/maturin).
- 📦 We package with support for Linux [amd64/arm64], Macos and WIndows.
- Supported Python versions are 3.7/3.8/3.9/3.10/3.11/3.12/3.13, including the free-threaded build.

### Supported image formats

- Read images from AVIF, BMP, DDS, Farbeld, GIF, HDR, ICO, JPEG (libjpeg-turbo), OpenEXR, PNG, PNM, TGA, TIFF, WebP.

### Image processing

- Convert images to grayscale, resize, crop, rotate, flip, pad, normalize, denormalize, and other image processing operations.

### Video processing

- Capture video frames from a camera and video writers.

## 🛠️ Installation

### >_ System dependencies

Dependeing on the features you want to use, you might need to install the following dependencies in your system:

#### turbojpeg

```bash
sudo apt-get install nasm
```

#### gstreamer

```bash
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
```

** Check the gstreamr installation guide: <https://docs.rs/gstreamer/latest/gstreamer/#installation>

### 🦀 Rust

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia = "v0.1.9"
```

Alternatively, you can use each sub-crate separately:

```toml
[dependencies]
kornia-tensor = { git = "https://github.com/kornia/kornia-rs", tag = "v0.1.9" }
kornia-tensor-ops = { git = "https://github.com/kornia/kornia-rs", tag = "v0.1.9" }
kornia-io = { git = "https://github.com/kornia/kornia-rs", tag = "v0.1.9" }
kornia-image = { git = "https://github.com/kornia/kornia-rs", tag = "v0.1.9" }
kornia-imgproc = { git = "https://github.com/kornia/kornia-rs", tag = "v0.1.9" }
kornia-icp = { git = "https://github.com/kornia/kornia-rs", tag = "v0.1.9" }
kornia-linalg = { git = "https://github.com/kornia/kornia-rs", tag = "v0.1.9" }
kornia-3d = { git = "https://github.com/kornia/kornia-rs", tag = "v0.1.9" }
```

### 🐍 Python

```bash
pip install kornia-rs
```

A subset of the full rust API is exposed. See the [kornia documentation](https://kornia.readthedocs.io/en/stable/) for more detail about the API for python functions and objects exposed by the `kornia-rs` Python module.

The `kornia-rs` library is thread-safe for use under the free-threaded Python build.

## Examples: Image processing

The following example shows how to read an image, convert it to grayscale and resize it. The image is then logged to a [`rerun`](https://github.com/rerun-io/rerun) recording stream.

Checkout all the examples in the [`examples`](https://github.com/kornia/kornia-rs/tree/main/examples) directory to see more use cases.

```rust
use kornia::{image::{Image, ImageSize}, imgproc};
use kornia::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image: Image<u8, 3> = F::read_image_any_rgb8("tests/data/dog.jpeg")?;
    let image_viz = image.clone();

    let image_f32: Image<f32, 3> = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // convert the image to grayscale
    let mut gray = Image::<f32, 1>::from_size_val(image_f32.size(), 0.0)?;
    imgproc::color::gray_from_rgb(&image_f32, &mut gray)?;

    // resize the image
    let new_size = ImageSize {
        width: 128,
        height: 128,
    };

    let mut gray_resized = Image::<f32, 1>::from_size_val(new_size, 0.0)?;
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

## Python usage

Load an image, that is converted directly to a numpy array to ease the integration with other libraries.

```python
    import kornia_rs as K
    import numpy as np

    # load an image with using libjpeg-turbo
    img: np.ndarray = K.read_image_jpeg("dog.jpeg")

    # alternatively, load other formats
    # img: np.ndarray = K.read_image_any("dog.png")

    assert img.shape == (195, 258, 3)

    # convert to dlpack to import to torch
    img_t = torch.from_dlpack(img)
    assert img_t.shape == (195, 258, 3)
```

Write an image to disk

```python
    import kornia_rs as K
    import numpy as np

    # load an image with using libjpeg-turbo
    img: np.ndarray = K.read_image_jpeg("dog.jpeg")

    # write the image to disk
    K.write_image_jpeg("dog_copy.jpeg", img)
```

Encode or decode image streams using the `turbojpeg` backend

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

decoded_img: np.ndarray = image_decoder.decode(bytes(image_encoded))
```

Resize an image using the `kornia-rs` backend with SIMD acceleration

```python
import kornia_rs as K

# load image with kornia-rs
img = K.read_image_jpeg("dog.jpeg")

# resize the image
resized_img = K.resize(img, (128, 128), interpolation="bilinear")

assert resized_img.shape == (128, 128, 3)
```

## 🧑‍💻 Development

Pre-requisites: install `rust` and `python3` in your system.

Install rustup in your system
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Install [`uv`](https://docs.astral.sh/uv/) to manage python dependencies
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the [`just`](https://github.com/casey/just) command runner. This tool is used to manage the development tasks.
```bash
cargo install just
```

Clone the repository in your local directory
```bash
git clone https://github.com/kornia/kornia-rs.git
```

You can check the available commands by running `just` in the root directory of the project.

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
### 🐳 Devcontainer

This project includes a development container to provide a consistent development environment.

The devcontainer is configured to include all necessary dependencies and tools required for building and testing the `kornia-rs` project. It ensures that the development environment is consistent across different machines and setups.

**How to use**

1. **Install Remote - Containers extension**: In Visual Studio Code, install the `Remote - Containers` extension from the Extensions view (`Ctrl+Shift+X`).

2. **Open the project in the container**:
    - Open the `kornia-rs` project folder in Visual Studio Code.
    - Press `F1` and select `Remote-Containers: Reopen in Container`.

Visual Studio Code will build the container and open the project inside it. You can now develop, build, and test the project within the containerized environment.

### 🦀 Rust

Compile the project and run the tests

```bash
just test
```

For specific tests, you can run the following command:

```bash
just test image
```

### 🐍 Python

To build the Python wheels, we use the `maturin` package. Use the following command to build the wheels:

```bash
just py-build
```

To run the tests, use the following command:

```bash
just py-test
```

## 💜 Contributing

This is a child project of [Kornia](https://github.com/kornia/kornia). Join the community to get in touch with us, or just sponsor the project: <https://opencollective.com/kornia>
