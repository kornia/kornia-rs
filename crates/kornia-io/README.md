# Kornia: kornia-io

[![Crates.io](https://img.shields.io/crates/v/kornia-io.svg)](https://crates.io/crates/kornia-io)
[![Documentation](https://docs.rs/kornia-io/badge.svg)](https://docs.rs/kornia-io)
[![License](https://img.shields.io/crates/l/kornia-io.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Image and Video Input/Output library for the Kornia ecosystem.**

## 🚀 Overview

`kornia-io` provides high-performance utilities for reading and writing images and video streams. It abstracts over common formats and libraries to provide a unified, type-safe API for getting visual data into your Rust applications. It supports standard image formats (JPEG, PNG, TIFF) and integrates with GStreamer and V4L2 for advanced video capture. The generic `read_image`/`decode_image`/`write_image` helpers exposed by the Python bindings (`kornia-py`) are Python-only; Rust users should call the typed per-format functions (`jpeg`, `png`, `tiff`) in this crate directly.

## 🔑 Key Features

*   **Image I/O:** Read and write support for JPEG, PNG, and TIFF formats.
*   **In-memory encode + decode:** Symmetric `encode_image_{jpeg,png}_*` ↔ `decode_image_{jpeg,png}_*` for round-tripping pixels through `Vec<u8>` without touching disk — including lossless PNG-16 (`encode_image_png_gray16`) for depth maps.
*   **TurboJPEG Support:** Optional integration with `turbojpeg` for high-performance JPEG encoding and decoding.
*   **Video Capture (GStreamer):** Access generic video streams (files, IP cameras, webcams) via GStreamer integration.
*   **Camera Access (V4L2):** Direct low-latency access to V4L2 devices on Linux.
*   **Type Integration:** Returns `kornia-image` structs directly, ensuring seamless interoperability with the rest of the ecosystem.

## 📦 Installation

Add the following to your `Cargo.toml`. Select features based on your needs:

```toml
[dependencies]
kornia-io = { version = "0.1.0", features = ["turbojpeg", "gstreamer"] }
```

## 🛠️ Usage

### Reading an Image

```rust
use kornia_io::functional as F;
use kornia_image::color_spaces::Rgb8;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read an image from disk
    // automatically detects format (JPEG, PNG, TIFF)
    let image = F::read_image_any_rgb8("path/to/image.jpg")?;

    println!("Image loaded: {}x{}", image.width(), image.height());

    Ok(())
}
```

### Video Capture (GStreamer)

*Requires `gstreamer` and `v4l` features.*

```rust
use kornia_io::gstreamer::{CameraCapture, V4L2CameraConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a camera capture object
    let mut capture = CameraCapture::new(
      &V4L2CameraConfig::new().with_camera_id(0).with_fps(30)
    )?;

    // Start the capture pipeline
    capture.start()?;

    // Grab a frame
    if let Some(img) = capture.grab_rgb8()? {
        println!("Image captured: {:?}", img.size());
    }

    Ok(())
}
```

## 🧩 Modules

*   **`error`**: Error types for the I/O module.
*   **`functional`**: High-level helper functions like `read_image_any_rgb8`.
*   **`jpeg` / `png` / `tiff`**: Format-specific implementations.
*   **`jpegturbo`**: (Feature: `turbojpeg`) High-performance JPEG encoding and decoding via TurboJPEG.
*   **`gstreamer`**: (Feature: `gstreamer`) Video capture and streaming via GStreamer.
*   **`v4l`**: (Feature: `v4l`, Linux only) Direct Video4Linux2 camera access.
*   **`fps_counter`**: Utilities for measuring frame rates.

## 💡 Related Examples

You can find comprehensive examples in the `examples` folder of the repository:

*   [`rtspcam`](../../examples/rtspcam): RTSP camera streaming using GStreamer.
*   [`v4l`](../../examples/v4l): Video capture using V4L2.
*   [`video_player`](../../examples/video_player): Simple video player example.
*   [`video_write`](../../examples/video_write): Video recording and writing.
*   [`foxglove`](../../examples/foxglove): Integration with Foxglove Studio for visualization.
*   [`exif_auto_orient`](../../examples/exif_auto_orient): Compare raw JPEG decode vs EXIF auto-oriented output.

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.
