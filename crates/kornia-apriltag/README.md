# Kornia: kornia-apriltag

[![Crates.io](https://img.shields.io/crates/v/kornia-apriltag.svg)](https://crates.io/crates/kornia-apriltag)
[![Documentation](https://docs.rs/kornia-apriltag/badge.svg)](https://docs.rs/kornia-apriltag)
[![License](https://img.shields.io/crates/l/kornia-apriltag.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **High-performance AprilTag marker detection and decoding.**

## 🚀 Overview

`kornia-apriltag` is a pure Rust implementation of the AprilTag fiducial marker system. It provides robust and efficient algorithms for detecting and decoding tags from images, suitable for robotics, augmented reality, and camera calibration tasks.

## 🔑 Key Features

*   **Multi-Family Support:** Detects various tag families including `Tag16H5`, `Tag25H9`, `Tag36H11`, and Circle tags (`TagCircle21H7`, `TagCircle49H12`).
*   **High Performance:** Optimized implementation with efficient image thresholding, segmentation, and quad fitting.
*   **Configurable Pipeline:** Fine-tune detection parameters such as sharpening, edge refinement, and quad decimation.
*   **Pure Rust:** No external C/C++ dependencies, ensuring easy compilation and cross-platform support.
*   **Strongly Typed:** Integrates seamlessly with `kornia-image` for type-safe image processing.

## 📦 Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia-apriltag = "0.1.0"
```

## 🛠️ Usage

### Detecting Tags

```rust
use kornia_apriltag::{AprilTagDecoder, DecodeTagsConfig, family::TagFamilyKind};
use kornia_image::{Image, ImageSize, allocator::CpuAllocator};
// use kornia_io::functional as F; // Assuming you have an image reader

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure the decoder for Tag36h11
    let config = DecodeTagsConfig::new(vec![TagFamilyKind::Tag36H11])?;

    // 2. Load an image (grayscale required)
    // let image = F::read_image_any_rgb8("tag.jpg")?;
    // let gray_image = kornia_imgproc::color::rgb_to_gray(&image)?;

    // Create dummy image for example
    let gray_image = Image::<u8, 1, _>::new(
        ImageSize { width: 100, height: 100 },
        vec![0u8; 10000],
        CpuAllocator
    )?;

    // 3. Initialize decoder
    let mut decoder = AprilTagDecoder::new(config, gray_image.size())?;

    // 4. Decode
    let detections = decoder.decode(&gray_image)?;

    println!("Detected {} tags", detections.len());

    for detection in detections {
        println!("Found tag ID: {}", detection.id);
        println!("Center: {:?}", detection.center);
    }

    Ok(())
}
```

## 🧩 Modules

*   **`decoder`**: Main decoding logic and `AprilTagDecoder` struct.
*   **`errors`**: Error types for AprilTag detection.
*   **`family`**: Tag family definitions.
*   **`quad`**: Low-level quad detection algorithms.
*   **`segmentation`**: Image segmentation and clustering.
*   **`threshold`**: Thresholding utilities for AprilTag detection.
*   **`union_find`**: Union-find data structure for connected components.
*   **`utils`**: Utility functions and types.

**Note:** `DecodeTagsConfig` is a struct exported from the crate root (`lib.rs`), not a separate module.

## 💡 Related Examples

You can find comprehensive examples in the `examples` folder of the repository:

*   [`apriltag`](../../examples/apriltag): Demonstration of detecting AprilTags in images using `kornia-apriltag`.

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.
