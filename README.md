# kornia-rs: low level computer vision library in Rust

![Crates.io Version](https://img.shields.io/crates/v/kornia-rs)
[![PyPI version](https://badge.fury.io/py/kornia-rs.svg)](https://badge.fury.io/py/kornia-rs)
[![Documentation](https://img.shields.io/badge/docs.rs-kornia_rs-orange)](https://docs.rs/kornia-rs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENCE)
[![Slack](https://img.shields.io/badge/Slack-4A154B?logo=slack&logoColor=white)](https://join.slack.com/t/kornia/shared_invite/zt-csobk21g-CnydWe5fmvkcktIeRFGCEQ)

The `kornia-rs` crate is a low level library for Computer Vision written in [Rust](https://www.rust-lang.org/) 🦀

Use the library to perform image I/O, visualisation and other low level operations in your machine learning and data-science projects in a thread-safe and efficient way.

## Getting Started

`cargo run --example hello_world`

```rust
use kornia_rs::image::Image;
use kornia_rs::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image: Image<u8, 3> = F::read_image_jpeg(image_path)?;

    println!("Hello, world!");
    println!("Loaded Image size: {:?}", image.size());
    println!("\nGoodbyte!");

    Ok(())
}
```

```bash
Hello, world!
Loaded Image size: ImageSize { width: 258, height: 195 }

Goodbyte!
```

## Features

- 🦀The library is primarly written in [Rust](https://www.rust-lang.org/).
- 🚀 Multi-threaded and efficient image I/O, image processing and advanced computer vision operators.
- 🔢 The n-dimensional backend is based on the [`ndarray`](https://crates.io/crates/ndarray) crate.
- 🐍 Pthon bindings are created with [PyO3/Maturin](https://github.com/PyO3/maturin).
- 📦 We package with support for Linux [amd64/arm64], Macos and WIndows.
- Supported Python versions are 3.7/3.8/3.9/3.10/3.11

### Supported image formats

- Read images from AVIF, BMP, DDS, Farbeld, GIF, HDR, ICO, JPEG (libjpeg-turbo), OpenEXR, PNG, PNM, TGA, TIFF, WebP.

### Image processing

- Convert images to grayscale, resize, crop, rotate, flip, pad, normalize, denormalize, and other image processing operations.

## 🛠️ Installation

### >_ System dependencies

You need to install the following dependencies in your system:

```bash
sudo apt-get install nasm
```

### 🦀 Rust

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia-rs = "0.1.0"
```

Alternatively, you can use the `cargo` command to add the dependency:

```bash
cargo add kornia-rs
```

### 🐍 Python

```bash
pip install kornia-rs
```

## Examples: Image processing

The following example shows how to read an image, convert it to grayscale and resize it. The image is then logged to a [`rerun`](https://github.com/rerun-io/rerun) recording stream.

Checkout all the examples in the [`examples`](https://github.com/kornia/kornia-rs/tree/main/examples) directory to see more use cases.

```rust
use kornia_rs::image::Image;
use kornia_rs::io::functional as F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // read the image
    let image_path = std::path::Path::new("tests/data/dog.jpeg");
    let image: Image<u8, 3> = F::read_image_jpeg(image_path)?;
    let image_viz = image.clone();

    let image_f32: Image<f32, 3> = image.cast_and_scale::<f32>(1.0 / 255.0)?;

    // convert the image to grayscale
    let gray: Image<f32, 1> = kornia_rs::color::gray_from_rgb(&image_f32)?;

    let gray_resize: Image<f32, 1> = kornia_rs::resize::resize_native(
        &gray,
        kornia_rs::image::ImageSize {
            width: 128,
            height: 128,
        },
        kornia_rs::resize::InterpolationMode::Bilinear,
    )?;

    println!("gray_resize: {:?}", gray_resize.size());

    // create a Rerun recording stream
    let rec = rerun::RecordingStreamBuilder::new("Kornia App").connect()?;

    // log the images
    let _ = rec.log("image", &rerun::Image::try_from(image_viz.data)?);
    let _ = rec.log("gray", &rerun::Image::try_from(gray.data)?);
    let _ = rec.log("gray_resize", &rerun::Image::try_from(gray_resize.data)?);

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

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Clone the repository in your local directory

```bash
git clone https://github.com/kornia/kornia-rs.git
```

### 🦀 Rust

Compile the project and run the tests

```bash
cargo test
```

For specific tests, you can run the following command:

```bash
cargo test image
```

### 🐍 Python

To build the Python wheels, we use the `maturin` package. Use the following command to build the wheels:

```bash
make build-python
```

To run the tests, use the following command:

```bash
make test-python
```

## 💜 Contributing

This is a child project of [Kornia](https://github.com/kornia/kornia). Join the community to get in touch with us, or just sponsor the project: https://opencollective.com/kornia
