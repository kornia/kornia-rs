//! Hough Line Transform — detect straight lines in edge images.
//!
//! Given a binary edge map (e.g. the output of [`canny`](crate::filter::canny)),
//! this module finds straight lines by letting every edge pixel "vote" for all
//! the lines that could pass through it. Lines that collect enough votes are
//! returned as `(rho, theta)` pairs in Hesse normal form.
//!
//! Internally the transform precomputes sin/cos lookup tables so the voting
//! step is fast even for large images.
//!
//! ## Typical workflow
//!
//! ```rust,no_run
//! use kornia_image::{Image, ImageSize};
//! use kornia_tensor::CpuAllocator;
//! use kornia_imgproc::features::hough_lines;
//!
//! // Start from a Canny edge map (u8, single-channel, 255 = edge).
//! let size = ImageSize { width: 640, height: 480 };
//! let edge_map = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator).unwrap();
//!
//! // Detect lines — 1° angular resolution, 1-pixel distance resolution.
//! let lines = hough_lines(&edge_map, 80, 1.0, std::f32::consts::PI / 180.0).unwrap();
//!
//! for line in &lines {
//!     println!("rho={:.1}  theta={:.1}°", line.rho, line.theta.to_degrees());
//! }
//! ```

use kornia_image::{allocator::ImageAllocator, Image, ImageError};

/// A detected line, stored in Hesse normal form.
///
/// Every point `(x, y)` on the line satisfies:
///
/// ```text
/// x * cos(theta) + y * sin(theta) = rho
/// ```
///
/// This is the same parameterisation used by OpenCV's `HoughLines`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    /// Perpendicular distance (in pixels) from the image origin to the line.
    /// Can be negative depending on the line's position.
    pub rho: f32,
    /// Angle of the line's normal, in radians (`0` = vertical line,
    /// `π/2` = horizontal line).
    pub theta: f32,
}

/// Detect straight lines in a binary edge map.
///
/// Feed in the output of [`crate::filter::canny`] (or any `u8` image where
/// non-zero means "edge") and this function returns every line that received
/// at least `threshold` votes in the Hough accumulator, sorted from strongest
/// to weakest.
///
/// # Arguments
///
/// * `edge_map` - Single-channel `u8` edge image (`255` = edge, `0` = background).
/// * `threshold` - How many edge pixels must agree before a line is reported.
///   Must be `>= 1`. Higher values give fewer, more confident lines.
/// * `rho_resolution` - Distance bucket size in pixels (typically `1.0`).
///   Must be positive and finite.
/// * `theta_resolution` - Angle bucket size in radians (typically `π / 180`,
///   i.e. 1° steps). Must be positive and finite.
///
/// # Returns
///
/// A `Vec<Line>` sorted by descending vote count (strongest line first).
///
/// # Errors
///
/// * [`ImageError::InvalidImageSize`] if the image has zero width or height.
/// * [`ImageError::InvalidSigmaValue`] if `rho_resolution` or `theta_resolution`
///   is not positive and finite.
/// * [`ImageError::InvalidHistogramBins`] if `threshold` is zero.
pub fn hough_lines<A: ImageAllocator>(
    edge_map: &Image<u8, 1, A>,
    threshold: u32,
    rho_resolution: f32,
    theta_resolution: f32,
) -> Result<Vec<Line>, ImageError> {
    let rows = edge_map.rows();
    let cols = edge_map.cols();

    if rows == 0 || cols == 0 {
        return Err(ImageError::InvalidImageSize(cols, rows, 0, 0));
    }

    // Validate resolution parameters to avoid NaN/inf accumulator dimensions.
    if !rho_resolution.is_finite() || rho_resolution <= 0.0 {
        return Err(ImageError::InvalidSigmaValue(rho_resolution, 0.0));
    }
    if !theta_resolution.is_finite() || theta_resolution <= 0.0 {
        return Err(ImageError::InvalidSigmaValue(0.0, theta_resolution));
    }

    // Reject threshold == 0 to avoid returning every accumulator cell.
    if threshold == 0 {
        return Err(ImageError::InvalidHistogramBins(0));
    }

    // Compute the image diagonal using f64 to avoid usize overflow on
    // large images (especially on 32-bit targets).
    let max_rho = ((rows as f64 * rows as f64 + cols as f64 * cols as f64).sqrt()) as f32;

    // Accumulator dimensions
    let num_rho = (2.0 * max_rho / rho_resolution).ceil() as usize + 1;
    let num_theta = (std::f32::consts::PI / theta_resolution).ceil() as usize;

    // Precompute sin/cos lookup tables
    let mut cos_table = vec![0.0f32; num_theta];
    let mut sin_table = vec![0.0f32; num_theta];
    for t in 0..num_theta {
        let theta = t as f32 * theta_resolution;
        cos_table[t] = theta.cos();
        sin_table[t] = theta.sin();
    }

    // Allocate and fill the accumulator
    let mut accumulator = vec![0u32; num_rho * num_theta];

    let data = edge_map.as_slice();
    for r in 0..rows {
        for c in 0..cols {
            if data[r * cols + c] == 0 {
                continue;
            }
            let x = c as f32;
            let y = r as f32;
            for t in 0..num_theta {
                let rho = x * cos_table[t] + y * sin_table[t];
                let rho_idx = ((rho + max_rho) / rho_resolution).round() as usize;
                if rho_idx < num_rho {
                    accumulator[rho_idx * num_theta + t] += 1;
                }
            }
        }
    }

    // Extract peaks above threshold, storing (votes, rho, theta)
    let mut peaks: Vec<(u32, f32, f32)> = Vec::new();
    for rho_idx in 0..num_rho {
        for t in 0..num_theta {
            let votes = accumulator[rho_idx * num_theta + t];
            if votes >= threshold {
                let rho = rho_idx as f32 * rho_resolution - max_rho;
                let theta = t as f32 * theta_resolution;
                peaks.push((votes, rho, theta));
            }
        }
    }

    // Sort by descending vote count
    peaks.sort_by(|a, b| b.0.cmp(&a.0));

    let lines = peaks
        .into_iter()
        .map(|(_, rho, theta)| Line { rho, theta })
        .collect();

    Ok(lines)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;
    use kornia_tensor::CpuAllocator;

    /// A perfect horizontal line of white pixels should be detected.
    #[test]
    fn test_hough_horizontal_line() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 64,
            height: 64,
        };

        // Draw a horizontal edge at row 32 spanning the full width.
        let mut data = vec![0u8; 64 * 64];
        for c in 0..64 {
            data[32 * 64 + c] = 255;
        }

        let edge_map = Image::<u8, 1, _>::new(size, data, CpuAllocator)?;
        let lines = hough_lines(&edge_map, 30, 1.0, std::f32::consts::PI / 180.0)?;

        // We should detect at least one line.
        assert!(!lines.is_empty(), "expected at least one line");

        // The strongest line should be approximately at theta ≈ π/2 (90°) and rho ≈ 32.
        let best = &lines[0];
        let theta_deg = best.theta.to_degrees();
        assert!(
            (theta_deg - 90.0).abs() < 5.0,
            "expected theta ≈ 90°, got {theta_deg:.1}°"
        );
        assert!(
            (best.rho - 32.0).abs() < 3.0,
            "expected rho ≈ 32, got {:.1}",
            best.rho
        );

        Ok(())
    }

    /// A perfect vertical line of white pixels should be detected.
    #[test]
    fn test_hough_vertical_line() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 64,
            height: 64,
        };

        // Draw a vertical edge at col 20 spanning the full height.
        let mut data = vec![0u8; 64 * 64];
        for r in 0..64 {
            data[r * 64 + 20] = 255;
        }

        let edge_map = Image::<u8, 1, _>::new(size, data, CpuAllocator)?;
        let lines = hough_lines(&edge_map, 30, 1.0, std::f32::consts::PI / 180.0)?;

        assert!(!lines.is_empty(), "expected at least one line");

        // The strongest line should be approximately at theta ≈ 0° and rho ≈ 20.
        let best = &lines[0];
        let theta_deg = best.theta.to_degrees();
        assert!(
            !(5.0..=175.0).contains(&theta_deg),
            "expected theta ≈ 0°, got {theta_deg:.1}°"
        );
        assert!(
            (best.rho - 20.0).abs() < 3.0,
            "expected rho ≈ 20, got {:.1}",
            best.rho
        );

        Ok(())
    }

    /// An empty image should produce no lines.
    #[test]
    fn test_hough_empty_image() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 32,
            height: 32,
        };
        let edge_map = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        let lines = hough_lines(&edge_map, 1, 1.0, std::f32::consts::PI / 180.0)?;

        assert!(lines.is_empty(), "empty image should have no lines");
        Ok(())
    }
}
