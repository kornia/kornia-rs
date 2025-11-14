//! Image enhancement and adjustment operations.
//!
//! This module provides functions for enhancing and adjusting image properties such as
//! brightness, contrast, and combining multiple images with weighted blending.
//!
//! # Available Operations
//!
//! * **Weighted Addition** ([`add_weighted`]) - Blend two images with adjustable weights
//!
//! # Example: Image Blending
//!
//! ```
//! use kornia_image::{Image, ImageSize};
//! use kornia_imgproc::enhance::add_weighted;
//!
//! let img1 = Image::<f32, 3>::from_size_val(
//!     ImageSize { width: 100, height: 100 },
//!     0.3,
//! ).unwrap();
//!
//! let img2 = Image::<f32, 3>::from_size_val(
//!     ImageSize { width: 100, height: 100 },
//!     0.7,
//! ).unwrap();
//!
//! let mut result = Image::<f32, 3>::from_size_val(
//!     img1.size(),
//!     0.0,
//! ).unwrap();
//!
//! // Blend 50% of each image
//! add_weighted(&img1, 0.5, &img2, 0.5, 0.0, &mut result).unwrap();
//! ```

use kornia_image::{allocator::ImageAllocator, Image, ImageError};

use crate::parallel;

/// Blend two images using weighted addition with an optional offset.
///
/// Performs a pixel-wise weighted combination of two images according to:
///
/// ```text
/// dst(x,y,c) = src1(x,y,c) · α + src2(x,y,c) · β + γ
/// ```
///
/// This operation is useful for:
/// * Image blending and cross-dissolve effects
/// * Alpha compositing when combined with masks
/// * Brightness/contrast adjustment when one image is constant
/// * Creating image pyramids and multi-resolution blending
///
/// # Arguments
///
/// * `src1` - The first input image.
/// * `alpha` - Multiplicative weight for the first image (α).
/// * `src2` - The second input image (must match src1 dimensions).
/// * `beta` - Multiplicative weight for the second image (β).
/// * `gamma` - Scalar offset added to each pixel (γ).
/// * `dst` - The output image (must match src1/src2 dimensions).
///
/// # Example: 50/50 Blend
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::enhance::add_weighted;
///
/// let img1 = Image::<f32, 1>::from_size_val(
///     ImageSize { width: 10, height: 10 },
///     100.0,
/// ).unwrap();
///
/// let img2 = Image::<f32, 1>::from_size_val(
///     ImageSize { width: 10, height: 10 },
///     200.0,
/// ).unwrap();
///
/// let mut result = Image::<f32, 1>::from_size_val(img1.size(), 0.0).unwrap();
///
/// add_weighted(&img1, 0.5, &img2, 0.5, 0.0, &mut result).unwrap();
/// // result pixels are now 150.0 = 0.5 * 100.0 + 0.5 * 200.0
/// ```
///
/// # Performance
///
/// This function is parallelized and processes images row-by-row for efficiency.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if:
/// * `src1` and `src2` have different dimensions
/// * `dst` dimensions don't match `src1`/`src2`
///
/// # See also
///
/// * Standard alpha blending sets α + β = 1.0
/// * For contrast/brightness: use constant `src2` with β and γ
pub fn add_weighted<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator, A3: ImageAllocator>(
    src1: &Image<T, C, A1>,
    alpha: T,
    src2: &Image<T, C, A2>,
    beta: T,
    gamma: T,
    dst: &mut Image<T, C, A3>,
) -> Result<(), ImageError>
where
    T: num_traits::Float + num_traits::FromPrimitive + std::fmt::Debug + Send + Sync + Copy,
{
    if src1.size() != src2.size() {
        return Err(ImageError::InvalidImageSize(
            src1.cols(),
            src1.rows(),
            src2.cols(),
            src2.rows(),
        ));
    }

    if src1.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src1.width(),
            src1.height(),
            dst.width(),
            dst.height(),
        ));
    }

    // compute the weighted sum
    parallel::par_iter_rows_val_two(src1, src2, dst, |&src1_pixel, &src2_pixel, dst_pixel| {
        *dst_pixel = (src1_pixel * alpha) + (src2_pixel * beta) + gamma;
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_add_weighted() -> Result<(), ImageError> {
        let src1_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let src1 = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            src1_data,
            CpuAllocator,
        )?;
        let src2_data = vec![4.0f32, 5.0, 6.0, 7.0];
        let src2 = Image::<f32, 1, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            src2_data,
            CpuAllocator,
        )?;
        let alpha = 2.0f32;
        let beta = 2.0f32;
        let gamma = 1.0f32;
        let expected = [11.0, 15.0, 19.0, 23.0];

        let mut weighted = Image::<f32, 1, _>::from_size_val(src1.size(), 0.0, CpuAllocator)?;

        super::add_weighted(&src1, alpha, &src2, beta, gamma, &mut weighted)?;

        weighted
            .as_slice()
            .iter()
            .zip(expected.iter())
            .for_each(|(a, b)| {
                assert!((a - b).abs() < 1e-6);
            });

        Ok(())
    }
}
