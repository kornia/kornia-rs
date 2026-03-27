//! Canny edge detection.
//!
//! Finds the edges in a grayscale image using the classic Canny algorithm
//! (J. Canny, 1986). Under the hood it:
//!
//! 1. Computes horizontal and vertical gradients with a normalised Sobel filter
//!    (via [`spatial_gradient_float`]).
//! 2. Thins edges to one-pixel width through non-maximum suppression.
//! 3. Links edges using double-threshold hysteresis — strong edges are kept
//!    immediately, and weak edges are promoted only if they touch a strong one.
//!
//! ## Choosing thresholds
//!
//! Because this implementation uses a **normalised** Sobel kernel, gradient
//! magnitudes are roughly in the range `[0, 125]` for `0–255` input. Typical
//! values are `low_threshold ≈ 10` and `high_threshold ≈ 40`. These are lower
//! than the classic OpenCV defaults (`50 / 150`) which assume an un-normalised
//! Sobel.
//!
//! # Example
//!
//! ```rust,no_run
//! use kornia_image::{Image, ImageSize};
//! use kornia_tensor::CpuAllocator;
//! use kornia_imgproc::filter::canny;
//!
//! let size = ImageSize { width: 640, height: 480 };
//! let src = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator).unwrap();
//! let mut edges = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator).unwrap();
//! canny(&src, &mut edges, 10.0, 40.0).unwrap();
//! ```

use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use kornia_tensor::CpuAllocator;

use super::spatial_gradient_float;

/// Detect edges in a single-channel `f32` grayscale image.
///
/// Pass in your grayscale image and two thresholds, and you get back a clean
/// binary edge map (`255` = edge, `0` = background). The algorithm is the
/// standard three-stage Canny pipeline: Sobel gradients → non-maximum
/// suppression → hysteresis linking.
///
/// # Arguments
///
/// * `src` - Single-channel `f32` source image (e.g. after grayscale conversion).
/// * `dst` - Pre-allocated single-channel `u8` output image, same size as `src`.
/// * `low_threshold` - Gradient magnitude below this is definitely **not** an edge.
/// * `high_threshold` - Gradient magnitude above this is definitely an edge.
///   Values in between are kept only if they are connected to a strong edge.
///
/// # Returns
///
/// `Ok(())` on success, or an [`ImageError`] if `src` and `dst` sizes differ.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] when `src` and `dst` have different
/// dimensions.
///
/// # Example
///
/// ```rust
/// use kornia_imgproc::filter::canny;
/// use kornia_image::{Image, ImageSize};
/// use kornia_tensor::CpuAllocator;
///
/// let size = ImageSize { width: 64, height: 64 };
/// let src = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator).unwrap();
/// let mut dst = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator).unwrap();
/// canny(&src, &mut dst, 10.0, 40.0).unwrap();
/// ```
pub fn canny<A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, 1, A1>,
    dst: &mut Image<u8, 1, A2>,
    low_threshold: f32,
    high_threshold: f32,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let rows = src.rows();
    let cols = src.cols();

    // Prevent underflow since algorithms ignore 1-pixel borders
    if rows < 3 || cols < 3 {
        dst.as_slice_mut().fill(0);
        return Ok(());
    }

    // 1. Compute Sobel gradients dx, dy
    let mut dx = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
    let mut dy = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
    spatial_gradient_float(src, &mut dx, &mut dy)?;

    let dx_data = dx.as_slice();
    let dy_data = dy.as_slice();

    // 2. Compute magnitude and direction
    let num_pixels = rows * cols;
    let mut magnitude = vec![0.0f32; num_pixels];
    let mut direction = vec![0.0f32; num_pixels];

    for i in 0..num_pixels {
        let gx = dx_data[i];
        let gy = dy_data[i];
        magnitude[i] = (gx * gx + gy * gy).sqrt();
        direction[i] = gy.atan2(gx);
    }

    // 3. Non-maximum suppression
    let mut nms = vec![0.0f32; num_pixels];

    for r in 1..rows - 1 {
        for c in 1..cols - 1 {
            let idx = r * cols + c;
            let mag = magnitude[idx];
            let angle = direction[idx];

            // Quantise the angle to one of four directions (0°, 45°, 90°, 135°).
            // We map the continuous angle to the nearest neighbour pair.
            let angle_deg = angle.to_degrees();
            // Normalise into [0, 180)
            let angle_norm = ((angle_deg % 180.0) + 180.0) % 180.0;

            let (n1, n2) = if !(22.5..157.5).contains(&angle_norm) {
                // 0° direction → compare East / West
                (magnitude[idx - 1], magnitude[idx + 1])
            } else if angle_norm < 67.5 {
                // 45° direction (gradient points South-East to North-West since image Y is down)
                // Compare NW / SE
                (
                    magnitude[(r - 1) * cols + (c - 1)],
                    magnitude[(r + 1) * cols + (c + 1)],
                )
            } else if angle_norm < 112.5 {
                // 90° direction → compare North / South
                (magnitude[(r - 1) * cols + c], magnitude[(r + 1) * cols + c])
            } else {
                // 135° direction (gradient points South-West to North-East since image Y is down)
                // Compare NE / SW
                (
                    magnitude[(r - 1) * cols + (c + 1)],
                    magnitude[(r + 1) * cols + (c - 1)],
                )
            };

            if mag >= n1 && mag >= n2 {
                nms[idx] = mag;
            }
        }
    }

    // 4. Double-threshold hysteresis
    //    - Strong pixels (>= high_threshold) are immediate edges.
    //    - Weak pixels (>= low_threshold) become edges only if connected to a strong pixel.

    const STRONG: u8 = 255;
    const WEAK: u8 = 128;

    let dst_data = dst.as_slice_mut();

    // Classify pixels
    for i in 0..num_pixels {
        if nms[i] >= high_threshold {
            dst_data[i] = STRONG;
        } else if nms[i] >= low_threshold {
            dst_data[i] = WEAK;
        } else {
            dst_data[i] = 0;
        }
    }

    // Hysteresis: iteratively promote WEAK pixels that are 8-connected to STRONG ones.
    let mut changed = true;
    while changed {
        changed = false;
        for r in 1..rows - 1 {
            for c in 1..cols - 1 {
                let idx = r * cols + c;
                if dst_data[idx] != WEAK {
                    continue;
                }
                // Check 8-connected neighbours for a STRONG pixel.
                let has_strong = dst_data[(r - 1) * cols + (c - 1)] == STRONG
                    || dst_data[(r - 1) * cols + c] == STRONG
                    || dst_data[(r - 1) * cols + (c + 1)] == STRONG
                    || dst_data[r * cols + (c - 1)] == STRONG
                    || dst_data[r * cols + (c + 1)] == STRONG
                    || dst_data[(r + 1) * cols + (c - 1)] == STRONG
                    || dst_data[(r + 1) * cols + c] == STRONG
                    || dst_data[(r + 1) * cols + (c + 1)] == STRONG;

                if has_strong {
                    dst_data[idx] = STRONG;
                    changed = true;
                }
            }
        }
    }

    // Suppress remaining WEAK pixels
    for val in dst_data.iter_mut() {
        if *val != STRONG {
            *val = 0;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;

    /// A white square on a black background should produce edges exactly on the perimeter.
    #[test]
    fn test_canny_white_square() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 32,
            height: 32,
        };

        // Create a 32×32 black image with a 10×10 white square at (11..21, 11..21).
        let mut data = vec![0.0f32; 32 * 32];
        for r in 11..21 {
            for c in 11..21 {
                data[r * 32 + c] = 255.0;
            }
        }

        let src = Image::<f32, 1, _>::new(size, data, CpuAllocator)?;
        let mut dst = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;

        canny(&src, &mut dst, 30.0, 100.0)?;

        let dst_data = dst.as_slice();

        // Interior pixels (fully inside the square) should be 0 (not an edge).
        for r in 13..19 {
            for c in 13..19 {
                assert_eq!(
                    dst_data[r * 32 + c],
                    0,
                    "interior pixel ({r},{c}) should not be an edge"
                );
            }
        }

        // Exterior pixels (well outside the square) should be 0.
        for r in 0..5 {
            for c in 0..5 {
                assert_eq!(
                    dst_data[r * 32 + c],
                    0,
                    "exterior pixel ({r},{c}) should not be an edge"
                );
            }
        }

        // At least some border pixels should be detected as edges.
        let mut edge_count = 0;
        for r in 10..22 {
            for c in 10..22 {
                if dst_data[r * 32 + c] == 255 {
                    edge_count += 1;
                }
            }
        }
        assert!(
            edge_count > 10,
            "expected at least 10 edge pixels on the square border, got {edge_count}"
        );

        Ok(())
    }

    /// Uniform images should produce no edges.
    #[test]
    fn test_canny_uniform() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 16,
            height: 16,
        };
        let src = Image::<f32, 1, _>::from_size_val(size, 128.0, CpuAllocator)?;
        let mut dst = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;

        canny(&src, &mut dst, 10.0, 50.0)?;

        // No edges expected at all.
        assert!(
            dst.as_slice().iter().all(|&v| v == 0),
            "uniform image should have no edges"
        );

        Ok(())
    }
}
