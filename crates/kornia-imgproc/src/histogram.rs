use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use rayon::prelude::*;

/// Compute the pixel intensity histogram of an image.
///
/// NOTE: this is limited to 8-bit 1-channel images.
///
/// # Arguments
///
/// * `src` - The input image to compute the histogram.
/// * `hist` - The output histogram.
/// * `num_bins` - The number of bins to use for the histogram.
///
/// # Returns
///
/// A vector of size `num_bins` containing the histogram.
///
/// # Errors
///
/// Returns an error if the number of bins is invalid.
///
/// # Example
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
/// assert_eq!(histogram, vec![3, 3, 3]);
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

    let mut bin_lut = [0usize; 256];
    for i in 0..256 {
        bin_lut[i] = (i * num_bins) >> 8;
    }

    let counts = src.as_slice()
        .par_chunks(4096)
        .fold(
            || vec![0usize; num_bins],
            |mut local, chunk| {
                for &px in chunk {
                    let idx = bin_lut[px as usize];
                    local[idx] += 1;
                }
                local
            },
        )
        .reduce(
            || vec![0usize; num_bins],
            |mut a, b| {
                for (i, val) in b.iter().enumerate() {
                    a[i] += val;
                }
                a
            },
        );

    for i in 0..num_bins {
        hist[i] += counts[i];
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
