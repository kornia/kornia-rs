# Kornia: kornia-imgproc

[![Crates.io](https://img.shields.io/crates/v/kornia-imgproc.svg)](https://crates.io/crates/kornia-imgproc)
[![Documentation](https://docs.rs/kornia-imgproc/badge.svg)](https://docs.rs/kornia-imgproc)
[![License](https://img.shields.io/crates/l/kornia-imgproc.svg)](https://github.com/kornia/kornia/blob/main/LICENSE)

> **Image processing algorithms for the Kornia ecosystem.**

## 🚀 Overview

`kornia-imgproc` provides a collection of standard and advanced image processing algorithms. It is designed to work seamlessly with `kornia-image` and `kornia-tensor`, offering high-performance implementations of common computer vision tasks.

## 🔑 Key Features

*   **Geometric Transformations:** Resize, crop, flip, rotate, and warp images.
*   **Color Space Conversions:** Convert between RGB, BGR, Grayscale, HSV, YUV, and other color spaces.
*   **Filtering:** Box blur, Gaussian blur, Median filter, Sobel, and custom kernels.
*   **Morphological Operations:** Dilation, Erosion, Opening, and Closing for structure aware image transformations.
*   **Calibration:** Camera calibration and image undistortion.
*   **Normalization:** Min/max scaling, mean/std normalization.
*   **Histograms:** Compute and manipulate image histograms.
*   **Feature Detection:** (Experimental) Corner detection and feature extraction.
*   **Contours:** Find image contours and compute their properties.

## 📦 Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
kornia-imgproc = "0.1.0"
```

## 🛠️ Usage

### Normalizing an Image

```rust
use kornia_image::{Image, ImageSize, allocator::CpuAllocator};
use kornia_imgproc::normalize::normalize_mean_std;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create a dummy f32 image
    let src = Image::<f32, 3, _>::new(
        ImageSize { width: 100, height: 100 },
        vec![0.5f32; 100 * 100 * 3],
        CpuAllocator
    )?;

    // 2. Normalize with mean and std
    let mut dst = Image::<f32, 3, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
    let mean = [0.485, 0.456, 0.406];
    let std = [0.229, 0.224, 0.225];

    normalize_mean_std(&src, &mut dst, &mean, &std)?;

    println!("Normalized image size: {:?}", dst.size());
    Ok(())
}
```

### Color Conversion

```rust
use kornia_image::{Image, ImageSize, allocator::CpuAllocator};
use kornia_imgproc::color::{Rgb8, Gray8, ConvertColor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rgb = Rgb8::new(
        ImageSize { width: 10, height: 10 },
        vec![0u8; 10 * 10 * 3],
        CpuAllocator
    )?;

    // Convert RGB to Grayscale
    let mut gray = Gray8::from_size_val(rgb.size(), 0, CpuAllocator)?;
    rgb.convert(&mut gray)?;

    assert_eq!(gray.num_channels(), 1);
    Ok(())
}
```

### Morphological Opening

```rust
use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
use kornia_imgproc::morphology::{open, Kernel, KernelShape};
use kornia_imgproc::padding::PaddingMode;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let src = Image::<u8, 1, _>::new(
        ImageSize { width: 5, height: 5 },
        vec![
            0, 0, 0, 0, 0,
            0, 255, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
        ],
        CpuAllocator,
    )?;

    let kernel = Kernel::try_new(KernelShape::Box { size: 3 })?;
    let mut dst = Image::<u8, 1, _>::from_size_val(src.size(), 0, CpuAllocator)?;

    open(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;

    assert!(dst.as_slice().iter().all(|&value| value == 0));
    Ok(())
}
```

## 🧩 Modules

*   **`calibration`**: Lens distortion correction.
*   **`color`**: Color space conversions.
*   **`contours`**: Contour finding and properties.
*   **`core`**: Basic image operations.
*   **`crop`**: Image cropping.
*   **`draw`**: Drawing utilities for images.
*   **`enhance`**: Image enhancement.
*   **`features`**: Feature detection.
*   **`filter`**: Convolutions and blurring.
*   **`flip`**: Image flipping operations.
*   **`histogram`**: Histogram computation.
*   **`interpolation`**: Interpolation utilities.
*   **`metrics`**: Image processing metrics.
*   **`morphology`**: Morphological operations.
*   **`normalize`**: Image normalization utilities.
*   **`padding`**: Image padding.
*   **`parallel`**: Parallelization utilities.
*   **`pyramid`**: Pyramid operations.
*   **`resize`**: Image resizing with various interpolation methods.
*   **`threshold`**: Image thresholding operations.
*   **`warp`**: Affine and perspective transformations.

## 💡 Related Examples

You can find comprehensive examples in the `examples` folder of the repository:

*   [`imgproc`](../../examples/imgproc): General image processing operations.
*   [`rotate`](../../examples/rotate): Image rotation and warping.
*   [`color_spaces`](../../examples/color_spaces): Color space conversion.
*   [`filters`](../../examples/filters): Image filtering (e.g., Sobel, Gaussian).
*   [`morphology`](../../examples/morphology/): Morphological operations (e.g., Dilation, Erosion, Opening, and Closing).
*   [`histogram`](../../examples/histogram): Histogram computation.
*   [`normalize`](../../examples/normalize): Image normalization.
*   [`normalize_ii`](../../examples/normalize_ii): Another normalization example.
*   [`undistort_image`](../../examples/undistort_image): Lens distortion correction.
*   [`undistort_points_image`](../../examples/undistort_points_image): Point undistortion.
*   [`pnp_demo`](../../examples/pnp_demo): Uses image processing for PnP.
*   [`ros-z-nodes`](../../examples/ros-z-nodes): ROS nodes using image processing.

## 🤝 Contributing

Contributions are welcome! This crate is part of the Kornia workspace. Please refer to the main repository for contribution guidelines.

## 📄 License

This crate is licensed under the Apache-2.0 License.
