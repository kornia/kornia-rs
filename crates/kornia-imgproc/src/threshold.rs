use num_traits::Zero;
use std::cmp::PartialOrd;

use kornia_image::{Image, ImageError};

use crate::parallel;

/// Adaptive mean binary thresholding for grayscale `u8` images.
///
/// Compares each pixel with the rounded mean of its centred square window
/// minus an integer offset. Pixels strictly above the threshold become the
/// configured maximum; the rest become zero. Borders replicate the nearest
/// edge pixel, including when the window is larger than the image.
///
/// Reference: OpenCV `adaptiveThreshold` with `ADAPTIVE_THRESH_MEAN_C` and
/// `THRESH_BINARY`. This API accepts an integer offset and a `u8` maximum.
/// Exact sums take O(width × height) time and at most
/// O(width × min(block_size, height)) active scratch space. Internal buffers
/// retain capacity between calls; use one instance per concurrent operation.
pub struct AdaptiveThresholdMean {
    block_size: usize,
    offset: i16,
    max_value: u8,
    area: u64,
    workspace: Workspace,
}

#[derive(Default)]
struct Workspace {
    ring: Vec<u32>,
    columns: Vec<u32>,
    prefix: Vec<u32>,
    wide_columns: Vec<u64>,
}

impl AdaptiveThresholdMean {
    /// Validate a configuration without allocating scratch.
    ///
    /// # Arguments
    ///
    /// * `block_size` - Odd window side in pixels, greater than one.
    /// * `offset` - Integer brightness offset subtracted from the rounded mean.
    /// * `max_value` - Output value for pixels above the local threshold.
    ///
    /// # Returns
    ///
    /// A configured threshold with empty internal scratch buffers.
    ///
    /// # Errors
    ///
    /// [`ImageError::InvalidKernelLength`] if the window is even, less than
    /// three, or too large for its sum and rounding term to fit in `u64`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_imgproc::threshold::AdaptiveThresholdMean;
    /// let threshold = AdaptiveThresholdMean::new(31, 10, 255)?;
    /// # Ok::<(), kornia_image::ImageError>(())
    /// ```
    pub fn new(block_size: usize, offset: i16, max_value: u8) -> Result<Self, ImageError> {
        let invalid_kernel = || ImageError::InvalidKernelLength(block_size, block_size);
        if block_size < 3 || block_size.is_multiple_of(2) {
            return Err(invalid_kernel());
        }
        let area = (block_size as u64)
            .checked_mul(block_size as u64)
            .filter(|&area| {
                area.checked_mul(u64::from(u8::MAX))
                    .and_then(|v| v.checked_add(area / 2))
                    .is_some()
            })
            .ok_or_else(invalid_kernel)?;
        Ok(Self {
            block_size,
            offset,
            max_value,
            area,
            workspace: Workspace::default(),
        })
    }

    /// Apply this threshold, reusing its internal scratch buffers.
    ///
    /// Buffers grow only when more capacity is needed. The 3 × 3 path uses
    /// no scratch. Empty images succeed without writing output.
    ///
    /// # Arguments
    ///
    /// * `src` - Host-resident single-channel `u8` image.
    /// * `dst` - Host-resident output with the same dimensions as `src`.
    ///
    /// # Returns
    ///
    /// `Ok(())` with the binary image written to `dst`.
    ///
    /// # Errors
    ///
    /// * [`ImageError::InvalidImageSize`] if source and destination sizes differ.
    /// * [`ImageError::UnsupportedDevice`] if either image is device-resident.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_image::{Image, ImageSize};
    /// use kornia_imgproc::threshold::AdaptiveThresholdMean;
    ///
    /// let size = ImageSize { width: 3, height: 3 };
    /// let src = Image::<u8, 1>::new(size, vec![200, 200, 200, 200, 40, 200, 200, 200, 200])?;
    /// let mut dst = Image::from_size_val(size, 0)?;
    /// let mut threshold = AdaptiveThresholdMean::new(3, 10, 255)?;
    /// threshold.apply(&src, &mut dst)?;
    /// assert_eq!(dst.as_slice(), &[255, 255, 255, 255, 0, 255, 255, 255, 255]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn apply(&mut self, src: &Image<u8, 1>, dst: &mut Image<u8, 1>) -> Result<(), ImageError> {
        crate::color::kernel_common::check_size(src, dst)?;
        let Self {
            block_size,
            offset,
            max_value,
            area,
            workspace,
        } = self;
        let (block_size, offset, max_value, area) = (*block_size, *offset, *max_value, *area);
        #[cfg(feature = "cuda")]
        if crate::cuda::dispatch::is_device(src) || crate::cuda::dispatch::is_device(dst) {
            return Err(ImageError::UnsupportedDevice);
        }

        let width = src.width();
        let height = src.height();
        if width == 0 || height == 0 {
            return Ok(());
        }
        if block_size == 3 {
            adaptive_mean_3x3(src.as_slice(), dst.as_slice_mut(), width, offset, max_value);
            return Ok(());
        }
        // The clamped pixel-plus-offset level is at most 256, so both its
        // product and the rounded window sum must fit in u32.
        if area <= u64::from(u32::MAX / 256) {
            adaptive_mean_streaming_u32(
                src,
                dst,
                workspace,
                block_size,
                offset,
                max_value,
                area as u32,
            );
            return Ok(());
        }
        let radius = block_size / 2;
        let src_pixels = src.as_slice();
        // The existing u8 box blur uses quantized weights, and the integral image
        // uses f32 sums. Neither preserves the exact rounded mean on large pages.
        workspace.wide_columns.resize(width, 0);
        let columns = &mut workspace.wide_columns;
        for (x, column) in columns.iter_mut().enumerate() {
            *column = u64::from(src_pixels[x]) * (radius + 1) as u64;
            for row in src_pixels
                .chunks_exact(width)
                .take(height.min(radius + 1))
                .skip(1)
            {
                *column += u64::from(row[x]);
            }
            *column += u64::from(src_pixels[(height - 1) * width + x])
                * radius.saturating_sub(height - 1) as u64;
        }

        for (y, dst_row) in dst.as_slice_mut().chunks_exact_mut(width).enumerate() {
            if y > 0 {
                let leaving = y.saturating_sub(radius + 1) * width;
                let entering = (y + radius).min(height - 1) * width;
                for (x, column) in columns.iter_mut().enumerate() {
                    *column = *column - u64::from(src_pixels[leaving + x])
                        + u64::from(src_pixels[entering + x]);
                }
            }
            let mut sum = columns[0] * (radius + 1) as u64
                + columns
                    .iter()
                    .take(width.min(radius + 1))
                    .skip(1)
                    .sum::<u64>()
                + columns[width - 1] * radius.saturating_sub(width - 1) as u64;
            for (x, output) in dst_row.iter_mut().enumerate() {
                if x > 0 {
                    sum = sum - columns[x.saturating_sub(radius + 1)]
                        + columns[(x + radius).min(width - 1)];
                }
                let mean = ((sum + area / 2) / area) as i32;
                *output = if i32::from(src_pixels[y * width + x]) > mean - i32::from(offset) {
                    max_value
                } else {
                    0
                };
            }
        }
        Ok(())
    }
}

/// Compute replicated-border horizontal sums, using wrapping prefixes for wide rows.
fn adaptive_horizontal_sum(source: &[u8], output: &mut [u32], radius: usize, prefix: &mut [u32]) {
    let width = source.len();
    if radius == 2 && width >= 5 {
        let p = |i| u32::from(source[i]);
        output[0] = 3 * p(0) + p(1) + p(2);
        output[1] = 2 * p(0) + p(1) + p(2) + p(3);
        for (window, sum) in source.windows(5).zip(&mut output[2..width - 2]) {
            *sum = window.iter().map(|&v| u32::from(v)).sum();
        }
        output[width - 2] = p(width - 4) + p(width - 3) + p(width - 2) + 2 * p(width - 1);
        output[width - 1] = p(width - 3) + p(width - 2) + 3 * p(width - 1);
        return;
    }
    prefix[0] = 0;
    let mut total = 0u32;
    // Prefixes may wrap on very wide images. Their wrapping difference is
    // exact because each window sum is bounded by the u32 fast-path check.
    for (out, &pixel) in prefix[1..].iter_mut().zip(source) {
        total = total.wrapping_add(u32::from(pixel));
        *out = total;
    }
    let border = |x: usize| {
        let low = x.saturating_sub(radius);
        let high = (x + radius + 1).min(width);
        prefix[high].wrapping_sub(prefix[low])
            + u32::from(source[0]) * radius.saturating_sub(x) as u32
            + u32::from(source[width - 1]) * (x + radius).saturating_sub(width - 1) as u32
    };
    if width > 2 * radius {
        for (x, out) in output[..radius].iter_mut().enumerate() {
            *out = border(x);
        }
        for ((out, &end), &start) in output[radius..width - radius]
            .iter_mut()
            .zip(&prefix[2 * radius + 1..])
            .zip(prefix.iter())
        {
            *out = end.wrapping_sub(start);
        }
        for (i, out) in output[width - radius..].iter_mut().enumerate() {
            *out = border(width - radius + i);
        }
    } else {
        for (x, out) in output.iter_mut().enumerate() {
            *out = border(x);
        }
    }
}

/// Retain horizontal sums in a ring and update exact vertical sums one row at a time.
fn adaptive_mean_streaming_u32(
    src: &Image<u8, 1>,
    dst: &mut Image<u8, 1>,
    workspace: &mut Workspace,
    block_size: usize,
    offset: i16,
    max_value: u8,
    area: u32,
) {
    let width = src.width();
    let height = src.height();
    let radius = block_size / 2;
    let pixels = src.as_slice();
    let ring_rows = block_size.min(height);
    let Workspace {
        ring,
        columns,
        prefix,
        ..
    } = workspace;
    ring.resize(width * ring_rows, 0);
    columns.resize(width, 0);
    columns.fill(0);
    prefix.resize(width + 1, 0);
    let mut latest = radius.min(height - 1);
    for y in 0..=latest {
        let row = &mut ring[y * width..][..width];
        adaptive_horizontal_sum(&pixels[y * width..][..width], row, radius, prefix);
        let weight = if y == 0 { radius + 1 } else { 1 } as u32;
        for (sum, &value) in columns.iter_mut().zip(row.iter()) {
            *sum += value * weight;
        }
    }
    let extra = radius.saturating_sub(height - 1) as u32;
    for (sum, &value) in columns.iter_mut().zip(&ring[latest * width..][..width]) {
        *sum += value * extra;
    }
    for (y, output) in dst.as_slice_mut().chunks_exact_mut(width).enumerate() {
        let entering = if y == 0 {
            0
        } else {
            (y + radius).min(height - 1)
        };
        if y > 0 {
            let leaving = y.saturating_sub(radius + 1) % ring_rows;
            // Evict the old sum before its ring slot can be overwritten.
            for (sum, &value) in columns.iter_mut().zip(&ring[leaving * width..][..width]) {
                *sum -= value;
            }
            if entering > latest {
                let row = &mut ring[(entering % ring_rows) * width..][..width];
                adaptive_horizontal_sum(&pixels[entering * width..][..width], row, radius, prefix);
                latest = entering;
            }
        }
        let additions = &ring[(entering % ring_rows) * width..][..width];
        for (((sum, &add), &pixel), out) in columns
            .iter_mut()
            .zip(additions)
            .zip(&pixels[y * width..][..width])
            .zip(output)
        {
            if y > 0 {
                *sum += add;
            }
            let level = (i32::from(pixel) + i32::from(offset)).clamp(0, 256) as u32;
            // q > floor((sum + area/2)/area) iff q*area > sum + area/2.
            // Clamping q to [0, 256] preserves comparison with any u8 mean.
            *out = if level * area > *sum + area / 2 {
                max_value
            } else {
                0
            };
        }
    }
}

// Independent windows let the compiler vectorize the small-kernel path without
// a rolling-sum dependency or scratch allocation. At most 9 * 255 fits in u16.
/// Classify a 3 × 3 neighbourhood directly without allocating scratch.
fn adaptive_mean_3x3(src: &[u8], dst: &mut [u8], width: usize, offset: i16, max_value: u8) {
    let height = src.len() / width;
    let classify = |sum: u16, pixel: u8| {
        if i32::from(pixel) > i32::from((sum + 4) / 9) - i32::from(offset) {
            max_value
        } else {
            0
        }
    };
    for (y, output) in dst.chunks_exact_mut(width).enumerate() {
        let top = &src[y.saturating_sub(1) * width..][..width];
        let middle = &src[y * width..][..width];
        let bottom = &src[(y + 1).min(height - 1) * width..][..width];
        let column_sum = |x| u16::from(top[x]) + u16::from(middle[x]) + u16::from(bottom[x]);
        output[0] = classify(2 * column_sum(0) + column_sum(1.min(width - 1)), middle[0]);
        if width > 2 {
            for (((t, m), b), pixel) in top
                .windows(3)
                .zip(middle.windows(3))
                .zip(bottom.windows(3))
                .zip(output[1..width - 1].iter_mut())
            {
                let sum = u16::from(t[0])
                    + u16::from(t[1])
                    + u16::from(t[2])
                    + u16::from(m[0])
                    + u16::from(m[1])
                    + u16::from(m[2])
                    + u16::from(b[0])
                    + u16::from(b[1])
                    + u16::from(b[2]);
                *pixel = classify(sum, m[1]);
            }
        }
        if width > 1 {
            output[width - 1] = classify(
                column_sum(width - 2) + 2 * column_sum(width - 1),
                middle[width - 1],
            );
        }
    }
}

/// Apply a binary threshold to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `dst` - The output image of an arbitrary number of channels and type.
/// * `threshold` - The threshold value. Must be the same type as the image.
/// * `max_value` - The maximum value to use when the input value is greater than the threshold.
///
/// # Returns
///
/// The thresholded image with the same number of channels as the input image.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::threshold::threshold_binary;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1>::new(ImageSize { width: 2, height: 3 }, data).unwrap();
///
/// let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0).unwrap();
///
/// threshold_binary(&image, &mut thresholded, 100, 255).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_binary<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    threshold: T,
    max_value: T,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // run the thresholding operation in parallel
    parallel::par_iter_rows_val(src, dst, |src_pixel, dst_pixel| {
        *dst_pixel = if *src_pixel > threshold {
            max_value
        } else {
            T::zero()
        };
    });

    Ok(())
}

/// Apply an inverse binary threshold to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `dst` - The output image of an arbitrary number of channels and type.
/// * `threshold` - The threshold value. Must be the same type as the image.
/// * `max_value` - The maximum value to use when the input value is less than the threshold.
///
/// # Returns
///
/// The thresholded image with the same number of channels as the input image.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::threshold::threshold_binary_inverse;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1>::new(ImageSize { width: 2, height: 3 }, data).unwrap();
///
/// let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0).unwrap();
///
/// threshold_binary_inverse(&image, &mut thresholded, 100, 255).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_binary_inverse<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    threshold: T,
    max_value: T,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // run the thresholding operation in parallel
    parallel::par_iter_rows_val(src, dst, |src_pixel, dst_pixel| {
        *dst_pixel = if *src_pixel > threshold {
            T::zero()
        } else {
            max_value
        };
    });

    Ok(())
}

/// Apply a truncated threshold to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `threshold` - The threshold value. Must be the same type as the image.
///
/// # Returns
///
/// The thresholded image with the same number of channels as the input image.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::threshold::threshold_truncate;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 1>::new(ImageSize { width: 2, height: 3 }, data).unwrap();
///
/// let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0).unwrap();
///
/// threshold_truncate(&image, &mut thresholded, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 3);
/// ```
pub fn threshold_truncate<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    threshold: T,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // run the thresholding operation in parallel
    parallel::par_iter_rows_val(src, dst, |src_pixel, dst_pixel| {
        *dst_pixel = if *src_pixel > threshold {
            threshold
        } else {
            *src_pixel
        };
    });

    Ok(())
}

/// Apply a threshold to an image, setting values below the threshold to zero.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `threshold` - The threshold value. Must be the same type as the image.
///
/// # Returns
///
/// The thresholded image with the same number of channels as the input image.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::threshold::threshold_to_zero;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 3>::new(ImageSize { width: 2, height: 1 }, data).unwrap();
///
/// let mut thresholded = Image::<_, 3>::from_size_val(image.size(), 0).unwrap();
///
/// threshold_to_zero(&image, &mut thresholded, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 3);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 1);
/// ```
pub fn threshold_to_zero<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    threshold: T,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // run the thresholding operation in parallel
    parallel::par_iter_rows_val(src, dst, |src_pixel, dst_pixel| {
        *dst_pixel = if *src_pixel > threshold {
            *src_pixel
        } else {
            T::zero()
        };
    });

    Ok(())
}

/// Apply a threshold to an image, setting values above the threshold to zero.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `threshold` - The threshold value. Must be the same type as the image.
///
/// # Returns
///
/// The thresholded image with the same number of channels as the input image.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::threshold::threshold_to_zero_inverse;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
/// let image = Image::<_, 3>::new(ImageSize { width: 2, height: 1 }, data).unwrap();
///
/// let mut thresholded = Image::<_, 3>::from_size_val(image.size(), 0).unwrap();
///
/// threshold_to_zero_inverse(&image, &mut thresholded, 150).unwrap();
/// assert_eq!(thresholded.num_channels(), 3);
/// assert_eq!(thresholded.size().width, 2);
/// assert_eq!(thresholded.size().height, 1);
/// ```
pub fn threshold_to_zero_inverse<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<T, C>,
    threshold: T,
) -> Result<(), ImageError>
where
    T: Copy + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // run the thresholding operation in parallel
    parallel::par_iter_rows_val(src, dst, |src_pixel, dst_pixel| {
        *dst_pixel = if *src_pixel > threshold {
            T::zero()
        } else {
            *src_pixel
        };
    });

    Ok(())
}

/// Apply a range threshold to an image.
///
/// # Arguments
///
/// * `src` - The input image of an arbitrary number of channels and type.
/// * `lower_bound` - The lower bound for each channel.
/// * `upper_bound` - The upper bound for each channel.
///
/// # Returns
///
/// The thresholded image with a single channel as byte values.
///
/// Precondition: the input image must have the same number of channels as the lower and upper bounds.
/// Precondition: the input image range must be 0-255.
///
/// # Examples
///
/// ```
/// use kornia_image::{Image, ImageSize};
/// use kornia_imgproc::threshold::in_range;
///
/// let data = vec![100u8, 200, 50, 150, 200, 250];
///
/// let image = Image::<u8, 3>::new(
///    ImageSize {
///       width: 2,
///       height: 1,
///    },
///    data,
/// )
/// .unwrap();
///
/// let mut thresholded = Image::<u8, 1>::from_size_val(image.size(), 0).unwrap();
///
/// in_range(&image, &mut thresholded, &[100, 150, 0], &[200, 200, 200]).unwrap();
/// assert_eq!(thresholded.num_channels(), 1);
/// assert_eq!(thresholded.size().width, 2);
///
/// assert_eq!(thresholded.get_pixel(0, 0, 0).unwrap(), &255);
/// assert_eq!(thresholded.get_pixel(1, 0, 0).unwrap(), &0);
/// ```
pub fn in_range<T, const C: usize>(
    src: &Image<T, C>,
    dst: &mut Image<u8, 1>,
    lower_bound: &[T; C],
    upper_bound: &[T; C],
) -> Result<(), ImageError>
where
    T: Clone + Send + Sync + PartialOrd + Zero,
{
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    // parallelize the operation by rows
    parallel::par_iter_rows(src, dst, |src_pixel, dst_pixel| {
        let mut is_in_range = true;
        src_pixel
            .iter()
            .zip(lower_bound.iter().zip(upper_bound.iter()))
            .for_each(|(src_val, (lower, upper))| {
                is_in_range &= src_val >= lower && src_val <= upper;
            });
        dst_pixel[0] = if is_in_range { 255 } else { 0 };
    });

    Ok(())
}

// TODO: otsu, triangle

#[cfg(test)]
mod tests {
    use super::{AdaptiveThresholdMean, Workspace};
    use kornia_image::{Image, ImageError, ImageSize};

    #[test]
    fn adaptive_mean_reuses_workspace_across_shapes() -> Result<(), ImageError> {
        let allocations = |w: &Workspace| {
            [
                (w.ring.as_ptr() as usize, w.ring.capacity()),
                (w.columns.as_ptr() as usize, w.columns.capacity()),
                (w.prefix.as_ptr() as usize, w.prefix.capacity()),
                (w.wide_columns.as_ptr() as usize, w.wide_columns.capacity()),
            ]
        };
        for block_size in [3, 5, 51, 4097] {
            let mut threshold = AdaptiveThresholdMean::new(block_size, 1, 201)?;
            assert!(allocations(&threshold.workspace)
                .iter()
                .all(|&(_, capacity)| capacity == 0));
            let mut previous = None;
            for (width, height) in [(37, 19), (7, 4), (13, 17), (37, 19)] {
                let mut src = Image::<u8, 1>::new(
                    ImageSize { width, height },
                    (0..width * height).map(|i| (i % 256) as u8).collect(),
                )?;
                let mut dst = Image::from_size_val(src.size(), 99)?;
                threshold.apply(&src, &mut dst)?;
                let before = allocations(&threshold.workspace);
                if block_size == 3 {
                    assert!(before.iter().all(|&(_, capacity)| capacity == 0));
                }
                if let Some(previous) = previous {
                    assert_eq!(before, previous);
                }
                previous = Some(before);
                src.as_slice_mut().fill(128);
                threshold.apply(&src, &mut dst)?;
                assert_eq!(allocations(&threshold.workspace), before);
                assert!(dst.as_slice().iter().all(|&p| p == 201));
            }
        }
        Ok(())
    }

    #[test]
    fn adaptive_horizontal_sums_match_direct_windows() {
        for width in [1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 31, 32, 33, 64, 129, 257] {
            let source: Vec<u8> = (0..width)
                .map(|i| ((i * 73 + i / 3 * 11) % 256) as u8)
                .collect();
            let mut output = vec![0u32; width];
            let mut prefix = vec![0u32; width + 1];
            for block_size in [5, 15, 31, 51, 67, 101, 4095] {
                let radius = block_size / 2;
                super::adaptive_horizontal_sum(&source, &mut output, radius, &mut prefix);
                for (x, &actual) in output.iter().enumerate() {
                    let expected: u32 = (0..block_size)
                        .map(|j| u32::from(source[(x + j).saturating_sub(radius).min(width - 1)]))
                        .sum();
                    assert_eq!(
                        actual, expected,
                        "width={width}, window={block_size}, x={x}"
                    );
                }
            }
        }
    }

    #[test]
    fn adaptive_mean_large_windows_and_wide_fallback() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 3,
            height: 2,
        };
        for value in [0, 128, 255] {
            let src = Image::<u8, 1>::from_size_val(size, value)?;
            let mut dst = Image::from_size_val(size, 99)?;
            for window in [4095, 4097, 1_000_001] {
                for offset in [i16::MIN, -1, 0, 1, i16::MAX] {
                    AdaptiveThresholdMean::new(window, offset, 173)?.apply(&src, &mut dst)?;
                    assert_eq!(
                        dst.as_slice(),
                        &[if offset > 0 { 173 } else { 0 }; 6],
                        "value={value}, window={window}, offset={offset}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn adaptive_mean_large_windows_on_nonuniform_image() -> Result<(), ImageError> {
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: 3,
                height: 2,
            },
            vec![0, 100, 255, 240, 130, 5],
        )?;
        let mut dst = Image::from_size_val(src.size(), 99)?;
        for block_size in [4095, 4097, 1_000_001] {
            let radius = block_size / 2;
            let area = (block_size as u64).pow(2);
            // Count each source pixel's multiplicity in the replicated window,
            // independently of the production rolling and prefix sums.
            let multiplicity = |centre: usize, sample: usize, length: usize| {
                u64::from(sample >= centre.saturating_sub(radius) && sample <= centre + radius)
                    + if sample == 0 {
                        radius.saturating_sub(centre) as u64
                    } else {
                        0
                    }
                    + if sample == length - 1 {
                        (centre + radius).saturating_sub(length - 1) as u64
                    } else {
                        0
                    }
            };
            for offset in [-127, -7, 0, 9, 127] {
                AdaptiveThresholdMean::new(block_size, offset, 173)?.apply(&src, &mut dst)?;
                for y in 0..src.height() {
                    for x in 0..src.width() {
                        let mut sum = 0u64;
                        for sy in 0..src.height() {
                            for sx in 0..src.width() {
                                sum += u64::from(src.as_slice()[sy * src.width() + sx])
                                    * multiplicity(x, sx, src.width())
                                    * multiplicity(y, sy, src.height());
                            }
                        }
                        let mean = ((sum + area / 2) / area) as i32;
                        let expected = if i32::from(src.as_slice()[y * src.width() + x])
                            > mean - i32::from(offset)
                        {
                            173
                        } else {
                            0
                        };
                        assert_eq!(
                            dst.as_slice()[y * src.width() + x],
                            expected,
                            "window={block_size}, offset={offset}, ({x},{y})"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn adaptive_mean_opencv_reference() -> Result<(), ImageError> {
        // Generated with OpenCV 4.11.0:
        // cv2.adaptiveThreshold(src, 173, cv2.ADAPTIVE_THRESH_MEAN_C,
        //                       cv2.THRESH_BINARY, block_size, offset)
        let src = Image::<u8, 1>::new(
            ImageSize {
                width: 5,
                height: 4,
            },
            vec![
                195, 201, 204, 9, 210, 20, 170, 190, 199, 212, 5, 88, 150, 200, 223, 10, 65, 189,
                230, 240,
            ],
        )?;
        let cases = [
            (
                3,
                0,
                [
                    173, 173, 173, 0, 173, 0, 173, 173, 173, 173, 0, 0, 0, 0, 173, 0, 0, 173, 173,
                    173,
                ],
            ),
            (
                3,
                7,
                [
                    173, 173, 173, 0, 173, 0, 173, 173, 173, 173, 0, 0, 0, 173, 173, 0, 0, 173,
                    173, 173,
                ],
            ),
            (
                7,
                -3,
                [
                    173, 173, 173, 0, 173, 0, 173, 173, 173, 173, 0, 0, 0, 173, 173, 0, 0, 173,
                    173, 173,
                ],
            ),
        ];
        let mut dst = Image::from_size_val(src.size(), 0)?;
        for (block_size, offset, expected) in cases {
            AdaptiveThresholdMean::new(block_size, offset, 173)?.apply(&src, &mut dst)?;
            assert_eq!(
                dst.as_slice(),
                expected,
                "block={block_size}, offset={offset}"
            );
        }
        Ok(())
    }

    #[test]
    fn adaptive_mean_matches_direct_window_sum() -> Result<(), ImageError> {
        for (width, height) in [(1, 1), (1, 9), (9, 1), (2, 3), (13, 17), (33, 5), (64, 3)] {
            let src = Image::<u8, 1>::new(
                ImageSize { width, height },
                (0..width * height)
                    .map(|i| ((i * 73 + i / 3 * 11) % 256) as u8)
                    .collect(),
            )?;
            let mut dst = Image::from_size_val(src.size(), 0)?;
            for block_size in [3, 5, 31] {
                for offset in [i16::MIN, -7, 0, 9, i16::MAX] {
                    AdaptiveThresholdMean::new(block_size, offset, 201)?.apply(&src, &mut dst)?;
                    let radius = block_size as isize / 2;
                    for y in 0..height {
                        for x in 0..width {
                            let mut sum = 0u64;
                            for dy in -radius..=radius {
                                for dx in -radius..=radius {
                                    let sy =
                                        (y as isize + dy).clamp(0, height as isize - 1) as usize;
                                    let sx =
                                        (x as isize + dx).clamp(0, width as isize - 1) as usize;
                                    sum += u64::from(src.as_slice()[sy * width + sx]);
                                }
                            }
                            let area = (block_size * block_size) as u64;
                            let mean = ((sum + area / 2) / area) as i32;
                            let expected = if i32::from(src.as_slice()[y * width + x])
                                > mean - i32::from(offset)
                            {
                                201
                            } else {
                                0
                            };
                            assert_eq!(
                                dst.as_slice()[y * width + x],
                                expected,
                                "{width}x{height}, block={block_size}, offset={offset}, ({x},{y})"
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn adaptive_mean_uniform_and_empty() -> Result<(), ImageError> {
        for value in [0, 128, 255] {
            let src = Image::<u8, 1>::from_size_val(
                ImageSize {
                    width: 3,
                    height: 2,
                },
                value,
            )?;
            let mut dst = Image::from_size_val(src.size(), 99)?;
            for (offset, max_value, expected) in
                [(0, 255, 0), (1, 173, 173), (-1, 255, 0), (1, 0, 0)]
            {
                AdaptiveThresholdMean::new(31, offset, max_value)?.apply(&src, &mut dst)?;
                assert!(dst.as_slice().iter().all(|&p| p == expected));
            }
        }
        for (width, height) in [(0, 0), (0, 3), (3, 0)] {
            let src = Image::<u8, 1>::from_size_val(ImageSize { width, height }, 0)?;
            let mut dst = Image::from_size_val(src.size(), 0)?;
            AdaptiveThresholdMean::new(3, 0, 255)?.apply(&src, &mut dst)?;
            assert!(dst.as_slice().is_empty());
        }
        Ok(())
    }

    #[test]
    fn adaptive_mean_rejects_invalid_arguments_without_writing() -> Result<(), ImageError> {
        let src = Image::<u8, 1>::from_size_val(
            ImageSize {
                width: 3,
                height: 2,
            },
            100,
        )?;
        for block_size in [0, 1, 2, 4, usize::MAX] {
            assert!(matches!(
                AdaptiveThresholdMean::new(block_size, 0, 255),
                Err(ImageError::InvalidKernelLength(_, _))
            ));
        }
        let mut wrong_shape = Image::from_size_val(
            ImageSize {
                width: 2,
                height: 3,
            },
            99,
        )?;
        assert!(matches!(
            AdaptiveThresholdMean::new(3, 0, 255)?.apply(&src, &mut wrong_shape),
            Err(ImageError::InvalidImageSize(..))
        ));
        assert_eq!(wrong_shape.as_slice(), &[99; 6]);
        Ok(())
    }

    #[test]
    fn adaptive_mean_preserves_strokes_across_lighting_gradient() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 60,
            height: 15,
        };
        let src = Image::<u8, 1>::new(
            size,
            (0..size.width * size.height)
                .map(|i| {
                    let x = i % size.width;
                    let paper = 90 + x as u8 * 2;
                    if x % 10 == 5 {
                        paper - 40
                    } else {
                        paper
                    }
                })
                .collect(),
        )?;
        let mut dst = Image::from_size_val(size, 0)?;
        AdaptiveThresholdMean::new(9, 10, 255)?.apply(&src, &mut dst)?;
        for (i, &pixel) in dst.as_slice().iter().enumerate() {
            assert_eq!(pixel, if i % size.width % 10 == 5 { 0 } else { 255 });
        }
        Ok(())
    }

    #[test]
    fn threshold_binary() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [0u8, 255, 0, 255, 255, 255];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0)?;

        super::threshold_binary(&image, &mut thresholded, 100, 255)?;

        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 3);

        assert_eq!(thresholded.as_slice(), data_expected);

        Ok(())
    }

    #[test]
    fn threshold_binary_inverse() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [255u8, 0, 255, 0, 0, 0];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0)?;

        super::threshold_binary_inverse(&image, &mut thresholded, 100, 255)?;

        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 3);

        assert_eq!(thresholded.as_slice(), data_expected);

        Ok(())
    }

    #[test]
    fn threshold_truncate() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [100u8, 150, 50, 150, 150, 150];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 2,
                height: 3,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0)?;

        super::threshold_truncate(&image, &mut thresholded, 150)?;

        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 3);

        assert_eq!(thresholded.as_slice(), data_expected);

        Ok(())
    }

    #[test]
    fn threshold_to_zero() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [0u8, 200, 0, 0, 200, 250];
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 3>::from_size_val(image.size(), 0)?;

        super::threshold_to_zero(&image, &mut thresholded, 150)?;

        assert_eq!(thresholded.num_channels(), 3);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 1);

        assert_eq!(thresholded.as_slice(), data_expected);

        Ok(())
    }

    #[test]
    fn threshold_to_zero_inverse() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let data_expected = [100u8, 0, 50, 150, 0, 0];
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 3>::from_size_val(image.size(), 0)?;

        super::threshold_to_zero_inverse(&image, &mut thresholded, 150)?;

        assert_eq!(thresholded.num_channels(), 3);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 1);

        assert_eq!(thresholded.as_slice(), data_expected);

        Ok(())
    }

    #[test]
    fn test_threshold_binary_equal_to_threshold() -> Result<(), ImageError> {
        // The implementation uses strict `>` (not `>=`), so a pixel whose value
        // exactly equals the threshold is treated as NOT exceeding it and maps to
        // zero rather than max_val.  Pin that boundary behaviour here.
        let data = vec![100u8, 101, 99];
        let image = Image::<_, 1>::new(
            ImageSize {
                width: 3,
                height: 1,
            },
            data,
        )?;

        let mut thresholded = Image::<_, 1>::from_size_val(image.size(), 0)?;

        super::threshold_binary(&image, &mut thresholded, 100, 255)?;

        // 100 == threshold → 0 (not strictly greater)
        // 101 >  threshold → 255
        // 99  <  threshold → 0
        assert_eq!(thresholded.as_slice(), &[0u8, 255, 0]);

        Ok(())
    }

    #[test]
    fn test_in_range() -> Result<(), ImageError> {
        let data = vec![100u8, 200, 50, 150, 200, 250];
        let image = Image::<_, 3>::new(
            ImageSize {
                width: 2,
                height: 1,
            },
            data,
        )?;

        let mut thresholded = Image::<u8, 1>::from_size_val(image.size(), 0)?;

        super::in_range(&image, &mut thresholded, &[100, 150, 0], &[200, 200, 200])?;

        assert_eq!(thresholded.num_channels(), 1);
        assert_eq!(thresholded.size().width, 2);
        assert_eq!(thresholded.size().height, 1);

        assert_eq!(thresholded.get([0, 0, 0]), Some(&255));
        assert_eq!(thresholded.get([0, 1, 0]), Some(&0));

        Ok(())
    }
}
