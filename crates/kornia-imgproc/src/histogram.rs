use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::prelude::*;

/// Computes the pixel intensity histogram of a grayscale image.
///
/// Bins pixel intensities into the specified number of histogram bins.
/// The computation is parallelized for efficiency using Rayon.
///
/// **Note:** Currently limited to 8-bit single-channel images.
///
/// # Arguments
///
/// * `src` - The input grayscale image (`u8` pixels)
/// * `hist` - Output histogram slice (must have length `num_bins`)
/// * `num_bins` - Number of histogram bins (1-256)
///
/// # Returns
///
/// Returns `Ok(())` on success. The histogram counts are written to `hist`.
///
/// # Errors
///
/// Returns [`ImageError::InvalidHistogramBins`] if:
/// - `num_bins` is 0 or greater than 256
/// - `hist.len()` does not equal `num_bins`
///
/// # Examples
///
/// Computing a 3-bin histogram:
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::histogram::compute_histogram;
///
/// let image = Image::<u8, 1, _>::new(
///   ImageSize {
///     width: 3,
///     height: 3,
///   },
///   vec![0, 2, 4, 128, 130, 132, 254, 255, 255],
///   CpuAllocator
/// ).unwrap();
///
/// let mut histogram = vec![0; 3];
///
/// compute_histogram(&image, &mut histogram, 3).unwrap();
/// // Pixels are binned: [0-85] = 3, [86-170] = 3, [171-255] = 3
/// assert_eq!(histogram, vec![3, 3, 3]);
/// ```
///
/// Standard 256-bin histogram for analysis:
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_image::allocator::CpuAllocator;
/// use kornia_imgproc::histogram::compute_histogram;
///
/// let image = Image::<u8, 1, _>::from_size_val(
///     ImageSize { width: 100, height: 100 },
///     128,
///     CpuAllocator
/// ).unwrap();
///
/// let mut hist = vec![0; 256];
/// compute_histogram(&image, &mut hist, 256).unwrap();
/// // All 10,000 pixels have value 128
/// assert_eq!(hist[128], 10000);
/// ```
pub fn compute_histogram<A: ImageAllocator>(
    src: &Image<u8, 1, A>,
    hist: &mut [usize],
    num_bins: usize,
) -> Result<(), ImageError> {
    if num_bins == 0 || num_bins > 256 {
        return Err(ImageError::InvalidHistogramBins(num_bins));
    }

    if hist.len() != num_bins {
        return Err(ImageError::InvalidHistogramBins(num_bins));
    }

    // we assume 8-bit images for now and range [0, 255]
    let scale = 256.0 / num_bins as f32;

    let width = src.width();
    let src_slice = src.as_slice();

    // parallaized computation of histogram on local threads
    let partial_hist = src_slice
        .par_chunks_exact(width)
        .map(|row| {
            let mut local_hist = vec![0_usize; num_bins];
            for &pixel in row {
                let bin = (pixel as f32 / scale).floor() as usize;
                local_hist[bin] += 1;
            }
            local_hist
        })
        .reduce(
            || vec![0; num_bins],
            |mut a, b| {
                for (i, val) in b.into_iter().enumerate() {
                    a[i] += val;
                }
                a
            },
        );

    for (i, val) in partial_hist.into_iter().enumerate() {
        hist[i] += val;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use kornia_image::{Image, ImageError, ImageSize};
    use kornia_tensor::CpuAllocator;
    #[test]
    fn test_compute_histogram() -> Result<(), ImageError> {
        let image = Image::new(
            ImageSize {
                width: 3,
                height: 3,
            },
            vec![0, 2, 4, 128, 130, 132, 254, 255, 255],
            CpuAllocator,
        )?;

        let mut histogram = vec![0; 3];

        super::compute_histogram(&image, &mut histogram, 3)?;
        assert_eq!(histogram, vec![3, 3, 3]);

        Ok(())
    }
}
