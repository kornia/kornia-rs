use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use kornia_tensor::CpuAllocator;
use std::collections::VecDeque;

use crate::filter::{gaussian_blur, spatial_gradient_float};

/// Apply Canny edge detection to a grayscale image.
///
/// Implements the standard 5-stage pipeline:
///
/// 1. Gaussian blur to suppress noise
/// 2. Sobel gradient computation (magnitude and direction)
/// 3. Non-maximum suppression to thin edges to 1 pixel width
/// 4. Double thresholding to classify pixels as strong, weak, or suppressed
/// 5. BFS hysteresis to connect weak edges adjacent to strong ones
///
/// # Arguments
///
/// * `src` - Input single-channel u8 grayscale image.
/// * `magnitude` - Output f32 image holding gradient magnitudes after NMS.
/// * `edges` - Output binary u8 image: 255 = edge, 0 = non-edge.
/// * `low_threshold` - Lower hysteresis threshold.
/// * `high_threshold` - Upper hysteresis threshold (strong edges).
/// * `kernel_size` - Gaussian blur kernel size (odd, e.g. 3 or 5).
///
/// # Errors
///
/// Returns `ImageError::InvalidImageSize` when dimensions do not match.
///
/// # References
///
/// - Canny (1986), IEEE TPAMI 8(6):679-698
/// - OpenCV: <https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html>
/// - Kornia Python: <https://github.com/kornia/kornia/blob/main/kornia/filters/canny.py>
pub fn canny<A1: ImageAllocator, A2: ImageAllocator, A3: ImageAllocator>(
    src: &Image<u8, 1, A1>,
    magnitude: &mut Image<f32, 1, A2>,
    edges: &mut Image<u8, 1, A3>,
    low_threshold: f32,
    high_threshold: f32,
    kernel_size: usize,
) -> Result<(), ImageError> {
    if src.size() != magnitude.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(), src.rows(), magnitude.cols(), magnitude.rows(),
        ));
    }
    if src.size() != edges.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(), src.rows(), edges.cols(), edges.rows(),
        ));
    }

    let rows = src.rows();
    let cols = src.cols();
    let n = rows * cols;

    // Stage 1: convert u8 to f32, scale to [0, 1].
    let src_f32_data: Vec<f32> = src.as_slice().iter().map(|&p| p as f32 / 255.0).collect();
    let src_f32 = Image::<f32, 1, _>::new(src.size(), src_f32_data, CpuAllocator)?;

    // Stage 2: Gaussian blur.
    // sigma = 0 lets gaussian_blur auto-compute from kernel_size using the SciPy convention.
    let mut blurred = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
    gaussian_blur(&src_f32, &mut blurred, (kernel_size, kernel_size), (0.0, 0.0))?;

    // Stage 3: Sobel gradients. spatial_gradient_float uses the normalized 3x3 Sobel kernel.
    let mut gx = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
    let mut gy = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
    spatial_gradient_float(&blurred, &mut gx, &mut gy)?;

    // Compute gradient magnitude and quantized direction for NMS.
    // Directions: 0=horizontal, 1=45deg, 2=vertical, 3=135deg.
    let mut mag = vec![0.0f32; n];
    let mut dir = vec![0u8; n];
    let gx_s = gx.as_slice();
    let gy_s = gy.as_slice();
    for i in 0..n {
        let gx_v = gx_s[i];
        let gy_v = gy_s[i];
        mag[i] = (gx_v * gx_v + gy_v * gy_v).sqrt();
        // Map angle to [0, 180) then bin.
        let angle = gy_v.atan2(gx_v).to_degrees();
        let angle = if angle < 0.0 { angle + 180.0 } else { angle };
        dir[i] = if angle < 22.5 || angle >= 157.5 {
            0 // horizontal
        } else if angle < 67.5 {
            1 // 45 degrees
        } else if angle < 112.5 {
            2 // vertical
        } else {
            3 // 135 degrees
        };
    }

    // Stage 4: Non-maximum suppression.
    // Suppress pixels not locally maximal along their gradient direction.
    let mut nms = vec![0.0f32; n];
    for r in 1..(rows - 1) {
        for c in 1..(cols - 1) {
            let idx = r * cols + c;
            let m = mag[idx];
            let (n1, n2) = match dir[idx] {
                0 => (mag[r * cols + c - 1], mag[r * cols + c + 1]),
                1 => (mag[(r - 1) * cols + c + 1], mag[(r + 1) * cols + c - 1]),
                2 => (mag[(r - 1) * cols + c], mag[(r + 1) * cols + c]),
                _ => (mag[(r - 1) * cols + c - 1], mag[(r + 1) * cols + c + 1]),
            };
            if m >= n1 && m >= n2 {
                nms[idx] = m;
            }
        }
    }
    magnitude.as_slice_mut().copy_from_slice(&nms);

    // Stage 5: Double thresholding. 0=suppressed, 1=weak, 2=strong.
    let mut class = vec![0u8; n];
    for i in 0..n {
        if nms[i] >= high_threshold {
            class[i] = 2;
        } else if nms[i] >= low_threshold {
            class[i] = 1;
        }
    }

    // Stage 6: Hysteresis edge tracking via BFS.
    // Confirm weak edges only when 8-connected to a strong edge.
    let mut edge = vec![0u8; n];
    let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

    for r in 0..rows {
        for c in 0..cols {
            if class[r * cols + c] == 2 {
                edge[r * cols + c] = 255;
                queue.push_back((r, c));
            }
        }
    }

    while let Some((r, c)) = queue.pop_front() {
        for dr in -1i32..=1 {
            for dc in -1i32..=1 {
                let nr = r as i32 + dr;
                let nc = c as i32 + dc;
                if nr >= 0 && nr < rows as i32 && nc >= 0 && nc < cols as i32 {
                    let nr = nr as usize;
                    let nc = nc as usize;
                    let nidx = nr * cols + nc;
                    if class[nidx] == 1 && edge[nidx] == 0 {
                        edge[nidx] = 255;
                        queue.push_back((nr, nc));
                    }
                }
            }
        }
    }

    edges.as_slice_mut().copy_from_slice(&edge);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    fn make_image(w: usize, h: usize, data: Vec<u8>) -> Result<Image<u8, 1, CpuAllocator>, ImageError> {
        Image::new(ImageSize { width: w, height: h }, data, CpuAllocator)
    }

    #[test]
    fn test_canny_uniform_image_produces_no_edges() -> Result<(), ImageError> {
        let w = 10; let h = 10;
        let src = make_image(w, h, vec![128u8; w * h])?;
        let mut mag = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
        let mut edges = Image::<u8, 1, _>::from_size_val(src.size(), 0, CpuAllocator)?;
        canny(&src, &mut mag, &mut edges, 0.01, 0.05, 3)?;
        assert!(edges.as_slice().iter().all(|&v| v == 0));
        Ok(())
    }

    #[test]
    fn test_canny_vertical_step_edge_detected() -> Result<(), ImageError> {
        let w = 20; let h = 10;
        let data: Vec<u8> = (0..h).flat_map(|_| (0..w).map(|c| if c < w / 2 { 0u8 } else { 255u8 })).collect();
        let src = make_image(w, h, data)?;
        let mut mag = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
        let mut edges = Image::<u8, 1, _>::from_size_val(src.size(), 0, CpuAllocator)?;
        canny(&src, &mut mag, &mut edges, 0.01, 0.05, 3)?;
        assert!(edges.as_slice().iter().any(|&v| v == 255), "Expected at least one edge pixel");
        Ok(())
    }

    #[test]
    fn test_canny_size_mismatch_returns_error() -> Result<(), ImageError> {
        let src = make_image(5, 5, vec![0u8; 25])?;
        let mut mag = Image::<f32, 1, _>::from_size_val(ImageSize { width: 6, height: 5 }, 0.0, CpuAllocator)?;
        let mut edges = Image::<u8, 1, _>::from_size_val(src.size(), 0, CpuAllocator)?;
        assert!(canny(&src, &mut mag, &mut edges, 0.01, 0.05, 3).is_err());
        Ok(())
    }

    #[test]
    fn test_canny_output_is_binary() -> Result<(), ImageError> {
        let w = 15; let h = 15;
        let data: Vec<u8> = (0..h).flat_map(|r| (0..w).map(move |c| if r == h/2 || c == w/2 { 255u8 } else { 0u8 })).collect();
        let src = make_image(w, h, data)?;
        let mut mag = Image::<f32, 1, _>::from_size_val(src.size(), 0.0, CpuAllocator)?;
        let mut edges = Image::<u8, 1, _>::from_size_val(src.size(), 0, CpuAllocator)?;
        canny(&src, &mut mag, &mut edges, 0.01, 0.05, 3)?;
        assert!(edges.as_slice().iter().all(|&v| v == 0 || v == 255));
        Ok(())
    }
  }
