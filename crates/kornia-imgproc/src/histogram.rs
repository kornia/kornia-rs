use kornia_image::{Image, ImageError};
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
/// use kornia_imgproc::histogram::compute_histogram;
///
/// let image = Image::<u8, 1>::new(
///   ImageSize {
///     width: 3,
///     height: 3,
///   },
///   vec![0, 2, 4, 128, 130, 132, 254, 255, 255],
/// ).unwrap();
///
/// let mut histogram = vec![0; 3];
///
/// compute_histogram(&image, &mut histogram, 3).unwrap();
/// assert_eq!(histogram, vec![3, 3, 3]);
/// ```
pub fn compute_histogram(
    src: &Image<u8, 1>,
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

    #[cfg(feature = "cuda")]
    if let crate::cuda::dispatch::Residency::Device(exec) =
        crate::cuda::dispatch::single_residency(src)?
    {
        return exec.run(|stream| {
            cuda_adapters::compute_histogram_cuda(src, hist, num_bins, scale, stream)
        });
    }

    let width = src.width();
    let src_slice = src.as_slice();

    // Coarse chunks so we allocate O(cores) local histograms rather than
    // O(rows). Each task accumulates its rows into one local histogram before
    // the reduce step merges them.
    const ROWS_PER_TASK: usize = 16;
    let partial_hist = src_slice
        .par_chunks(ROWS_PER_TASK * width)
        .map(|chunk| {
            let mut local_hist = vec![0_usize; num_bins];
            for &pixel in chunk {
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

    #[test]
    fn test_compute_histogram() -> Result<(), ImageError> {
        let image = Image::new(
            ImageSize {
                width: 3,
                height: 3,
            },
            vec![0, 2, 4, 128, 130, 132, 254, 255, 255],
        )?;

        let mut histogram = vec![0; 3];

        super::compute_histogram(&image, &mut histogram, 3)?;
        assert_eq!(histogram, vec![3, 3, 3]);

        Ok(())
    }
}

/// OpenCV-exact histogram-equalization LUT from a 256-bin histogram.
///
/// `lut[i] = rint((cdf[i] − cdf_min) · (255f32 / (N − cdf_min)))` with f32
/// scale and round-half-to-even — byte-for-byte what `cv2.equalizeHist`
/// computes (verified empirically, including its tie rounding). A constant
/// image (`N == cdf_min`) gets the identity LUT, matching cv2. The CUDA
/// `equalize_lut_u8` kernel is the textual twin of this function.
pub(crate) fn equalize_lut(hist: &[usize; 256], total: usize) -> [u8; 256] {
    let mut cdf = [0usize; 256];
    let mut acc = 0usize;
    let mut cdf_min = 0usize;
    let mut found = false;
    for i in 0..256 {
        acc += hist[i];
        cdf[i] = acc;
        if !found && acc > 0 {
            cdf_min = acc;
            found = true;
        }
    }
    let mut lut = [0u8; 256];
    if total == cdf_min {
        for (i, l) in lut.iter_mut().enumerate() {
            *l = i as u8;
        }
        return lut;
    }
    let scale = 255.0f32 / (total - cdf_min) as f32;
    for i in 0..256 {
        let v = (cdf[i] as i64 - cdf_min as i64) as f32 * scale;
        lut[i] = (v.round_ties_even() as i32).clamp(0, 255) as u8;
    }
    lut
}

/// Histogram equalization for 8-bit single-channel images — byte-for-byte
/// with `cv2.equalizeHist` (see [`equalize_lut`]). Device pairs run the
/// CUDA histogram → LUT → apply chain, byte-identical to the CPU path.
pub fn equalize_hist(src: &Image<u8, 1>, dst: &mut Image<u8, 1>) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    #[cfg(feature = "cuda")]
    {
        use crate::try_device;
        try_device!(src, dst, |stream| cuda_adapters::equalize_hist_cuda(
            src, dst, stream
        ));
    }

    let mut hist_vec = vec![0usize; 256];
    compute_histogram(src, &mut hist_vec, 256)?;
    let mut hist = [0usize; 256];
    hist.copy_from_slice(&hist_vec);
    let lut = equalize_lut(&hist, src.as_slice().len());

    dst.as_slice_mut()
        .par_iter_mut()
        .zip(src.as_slice().par_iter())
        .for_each(|(d, &s)| {
            *d = lut[s as usize];
        });
    Ok(())
}

#[cfg(feature = "cuda")]
mod cuda_adapters {
    use super::*;
    use crate::cuda::dispatch::{device_slices, untyped_device_err};
    use crate::cuda::histogram::{
        launch_apply_lut_u8, launch_equalize_lut_u8, launch_histogram_u8,
    };
    use cudarc::driver::CudaStream;
    use std::sync::Arc;

    fn err(e: impl std::fmt::Display) -> ImageError {
        ImageError::Cuda(e.to_string())
    }

    pub(super) fn compute_histogram_cuda(
        src: &Image<u8, 1>,
        hist: &mut [usize],
        num_bins: usize,
        scale: f32,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let ctx = stream.context();
        let s = src
            .0
            .as_cudaslice()
            .ok_or_else(|| untyped_device_err("source"))?;
        let n = src.rows() * src.cols();
        let mut d_hist = stream.alloc_zeros::<u32>(num_bins).map_err(err)?;
        launch_histogram_u8(ctx, stream, s, &mut d_hist, n, num_bins as u32, scale).map_err(err)?;
        let host: Vec<u32> = stream.clone_dtoh(&d_hist).map_err(err)?;
        stream.synchronize().map_err(err)?;
        for (h, v) in hist.iter_mut().zip(host) {
            *h = v as usize;
        }
        Ok(())
    }

    pub(super) fn equalize_hist_cuda(
        src: &Image<u8, 1>,
        dst: &mut Image<u8, 1>,
        stream: &Arc<CudaStream>,
    ) -> Result<(), ImageError> {
        let ctx = stream.context();
        let (s, d) = device_slices!(src, dst);
        let n = src.rows() * src.cols();
        let mut d_hist = stream.alloc_zeros::<u32>(256).map_err(err)?;
        let mut d_lut = stream.alloc_zeros::<u8>(256).map_err(err)?;
        launch_histogram_u8(ctx, stream, s, &mut d_hist, n, 256, 1.0).map_err(err)?;
        launch_equalize_lut_u8(ctx, stream, &d_hist, &mut d_lut, n).map_err(err)?;
        launch_apply_lut_u8(ctx, stream, s, d, &d_lut, n).map_err(err)?;
        Ok(())
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::cuda::color::test_utils::{default_stream, pattern_u8};
    use kornia_image::ImageSize;

    fn sz(w: usize, h: usize) -> ImageSize {
        ImageSize {
            width: w,
            height: h,
        }
    }

    /// Device histogram counts must EXACTLY equal the CPU's (same binning
    /// expression, commutative integer atomics), for 256 and coarse bins,
    /// odd sizes and tiny images.
    #[test]
    fn histogram_device_equals_host_exact() {
        let stream = default_stream();
        for (w, h) in [(64usize, 48usize), (67, 43), (5, 4), (1, 1)] {
            for num_bins in [256usize, 64, 10, 1] {
                let src = Image::<u8, 1>::new(sz(w, h), pattern_u8(w * h)).unwrap();
                let mut cpu = vec![0usize; num_bins];
                compute_histogram(&src, &mut cpu, num_bins).unwrap();

                let d_src = src.to_cuda(&stream).unwrap();
                let mut gpu = vec![0usize; num_bins];
                compute_histogram(&d_src, &mut gpu, num_bins).unwrap();
                assert_eq!(cpu, gpu, "{w}x{h} bins={num_bins}");
                assert_eq!(gpu.iter().sum::<usize>(), w * h);
            }
        }
    }

    /// equalize_hist device output must be byte-exact vs the CPU path.
    #[test]
    fn equalize_device_equals_host_byte_exact() {
        let stream = default_stream();
        for (w, h) in [(64usize, 48usize), (67, 43), (5, 4), (1, 1)] {
            let src = Image::<u8, 1>::new(sz(w, h), pattern_u8(w * h)).unwrap();
            let mut cpu = Image::<u8, 1>::from_size_val(sz(w, h), 0).unwrap();
            equalize_hist(&src, &mut cpu).unwrap();

            let d_src = src.to_cuda(&stream).unwrap();
            let mut d_dst = Image::<u8, 1>::zeros_cuda(sz(w, h), &stream).unwrap();
            equalize_hist(&d_src, &mut d_dst).unwrap();
            let back = d_dst.to_host_owned().unwrap();
            assert_eq!(back.as_slice(), cpu.as_slice(), "{w}x{h}");
        }

        // Constant image: identity (cv2 behavior), both paths.
        let src = Image::<u8, 1>::from_size_val(sz(16, 16), 77).unwrap();
        let mut cpu = Image::<u8, 1>::from_size_val(sz(16, 16), 0).unwrap();
        equalize_hist(&src, &mut cpu).unwrap();
        assert!(cpu.as_slice().iter().all(|&v| v == 77));
        let d_src = src.to_cuda(&stream).unwrap();
        let mut d_dst = Image::<u8, 1>::zeros_cuda(sz(16, 16), &stream).unwrap();
        equalize_hist(&d_src, &mut d_dst).unwrap();
        assert!(d_dst
            .to_host_owned()
            .unwrap()
            .as_slice()
            .iter()
            .all(|&v| v == 77));
    }
}

#[cfg(test)]
mod equalize_tests {
    use super::*;
    use kornia_image::ImageSize;

    /// cv2's tie rounding is round-half-to-even (f32 rint), calibrated
    /// empirically against cv2.equalizeHist: a 512-px image crafted so two
    /// LUT products land exactly on .5 — 0.5 must round to 0 and 1.5 to 2.
    #[test]
    fn equalize_lut_matches_cv2_tie_rounding() {
        // hist: value 0 x255, value 1 x1, value 2 x1, value 255 x255 → N=512.
        // cdf_min=255, N-cdf_min=257... craft simpler: direct LUT check.
        let mut hist = [0usize; 256];
        // N = 1020: cdf_min = 2 at bin 10; scale = 255/1018.
        // bin 12: cdf-cdf_min = 2 -> 2*255/1018 = 0.50098 -> 1 (not a tie)
        // Craft exact ties instead: N - cdf_min = 510 -> scale = 0.5 exactly.
        // cdf-cdf_min = 1 -> 0.5 -> rint -> 0 ; = 3 -> 1.5 -> rint -> 2.
        hist[10] = 2; // cdf_min = 2
        hist[20] = 1; // cdf = 3 -> diff 1 -> 0.5 -> 0
        hist[30] = 2; // cdf = 5 -> diff 3 -> 1.5 -> 2
        hist[255] = 507; // cdf = 512 = N -> diff 510 -> 255
        let lut = equalize_lut(&hist, 512);
        assert_eq!(lut[10], 0);
        assert_eq!(lut[20], 0, "0.5 must round-half-to-even to 0");
        assert_eq!(lut[30], 2, "1.5 must round-half-to-even to 2");
        assert_eq!(lut[255], 255);
        // Below cdf_min: signed diff saturates to 0.
        assert_eq!(lut[0], 0);
    }

    /// Constant image → identity LUT (cv2 returns the image unchanged).
    #[test]
    fn equalize_constant_image_is_identity() {
        let sz = ImageSize {
            width: 8,
            height: 8,
        };
        let src = Image::<u8, 1>::from_size_val(sz, 200).unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(sz, 0).unwrap();
        equalize_hist(&src, &mut dst).unwrap();
        assert_eq!(src.as_slice(), dst.as_slice());
    }

    /// Full-range ramp equalizes to (approximately) itself.
    #[test]
    fn equalize_uniform_ramp_is_near_identity() {
        let sz = ImageSize {
            width: 256,
            height: 1,
        };
        let data: Vec<u8> = (0..=255).collect();
        let src = Image::<u8, 1>::new(sz, data).unwrap();
        let mut dst = Image::<u8, 1>::from_size_val(sz, 0).unwrap();
        equalize_hist(&src, &mut dst).unwrap();
        for (i, &v) in dst.as_slice().iter().enumerate() {
            let expect = ((i as f32) * 255.0 / 255.0).round_ties_even() as i32;
            assert!((v as i32 - expect).abs() <= 1, "i={i} v={v}");
        }
    }
}
