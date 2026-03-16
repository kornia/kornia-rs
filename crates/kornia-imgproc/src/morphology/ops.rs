use super::kernels::Kernel;
use crate::padding::{spatial_padding, Padding2D, PaddingMode};
use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;
use rayon::prelude::*;

#[derive(Clone, Copy)]
enum MorphologyOp {
    Dilate,
    Erode,
}

/// Dilate an image using a [`Kernel`].
///
/// Dilation replaces each pixel with the maximum value over the active kernel
/// support. This expands bright regions and can connect nearby structures.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image. It must have the same size as `src`.
/// * `kernel` - The structuring element defining the neighborhood to scan.
/// * `padding_mode` - The border handling mode used before scanning.
/// * `constant_value` - The fill value used when `padding_mode` is
///   [`PaddingMode::Constant`].
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` sizes differ.
/// Returns [`ImageError::InvalidKernelShape`] if `kernel` is not valid for
/// morphology operations.
///
/// # Example
///
/// ```rust
/// use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
/// use kornia_imgproc::morphology::{dilate, Kernel, KernelShape};
/// use kornia_imgproc::padding::PaddingMode;
///
/// let src = Image::<u8, 1, _>::new(
///     ImageSize { width: 3, height: 3 },
///     vec![0, 0, 0, 0, 255, 0, 0, 0, 0],
///     CpuAllocator,
/// )?;
/// let mut dst = Image::<u8, 1, _>::from_size_val(src.size(), 0, CpuAllocator)?;
/// let kernel = Kernel::try_new(KernelShape::Box { size: 3 })?;
///
/// dilate(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
/// assert!(dst.as_slice().iter().all(|&value| value == 255));
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
pub fn dilate<
    T: Copy + Send + Sync + Ord,
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError> {
    morphology_op(
        src,
        dst,
        kernel,
        padding_mode,
        constant_value,
        MorphologyOp::Dilate,
    )
}

/// Erode an image using a [`Kernel`].
///
/// Erosion replaces each pixel with the minimum value over the active kernel
/// support. This shrinks bright regions and removes isolated bright pixels.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image. It must have the same size as `src`.
/// * `kernel` - The structuring element defining the neighborhood to scan.
/// * `padding_mode` - The border handling mode used before scanning.
/// * `constant_value` - The fill value used when `padding_mode` is
///   [`PaddingMode::Constant`].
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` sizes differ.
/// Returns [`ImageError::InvalidKernelShape`] if `kernel` is not valid for
/// morphology operations.
///
/// # Example
///
/// ```rust
/// use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
/// use kornia_imgproc::morphology::{erode, Kernel, KernelShape};
/// use kornia_imgproc::padding::PaddingMode;
///
/// let src = Image::<u8, 1, _>::new(
///     ImageSize { width: 3, height: 3 },
///     vec![255; 9],
///     CpuAllocator,
/// )?;
/// let mut dst = Image::<u8, 1, _>::from_size_val(src.size(), 0, CpuAllocator)?;
/// let kernel = Kernel::try_new(KernelShape::Box { size: 3 })?;
///
/// erode(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
/// assert_eq!(dst.as_slice()[4], 255);
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
pub fn erode<
    T: Copy + Send + Sync + Ord,
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError> {
    morphology_op(
        src,
        dst,
        kernel,
        padding_mode,
        constant_value,
        MorphologyOp::Erode,
    )
}

/// Apply an opening operation to an image.
///
/// Opening performs erosion followed by dilation using the same structuring
/// element. It is commonly used to remove small bright noise while preserving
/// larger structures.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image. It must have the same size as `src`.
/// * `kernel` - The structuring element defining the neighborhood to scan.
/// * `padding_mode` - The border handling mode used before scanning.
/// * `constant_value` - The fill value used when `padding_mode` is
///   [`PaddingMode::Constant`].
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` sizes differ.
/// Returns [`ImageError::InvalidKernelShape`] if `kernel` is not valid for
/// morphology operations.
///
/// # Example
///
/// ```rust
/// use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
/// use kornia_imgproc::morphology::{open, Kernel, KernelShape};
/// use kornia_imgproc::padding::PaddingMode;
///
/// let src = Image::<u8, 1, _>::new(
///     ImageSize { width: 5, height: 5 },
///     vec![
///         0, 0, 0, 0, 0,
///         0, 255, 0, 0, 0,
///         0, 0, 0, 0, 0,
///         0, 0, 0, 0, 0,
///         0, 0, 0, 0, 0,
///     ],
///     CpuAllocator,
/// )?;
/// let mut dst = Image::<u8, 1, _>::from_size_val(src.size(), 0, CpuAllocator)?;
/// let kernel = Kernel::try_new(KernelShape::Box { size: 3 })?;
///
/// open(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
/// assert!(dst.as_slice().iter().all(|&value| value == 0));
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
pub fn open<
    T: Copy + Send + Sync + Ord + Default,
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError> {
    validate_inputs(src, dst, kernel)?;

    let mut temp_img = Image::from_size_val(src.size(), T::default(), CpuAllocator)?;
    erode(src, &mut temp_img, kernel, padding_mode, constant_value)?;
    dilate(&temp_img, dst, kernel, padding_mode, constant_value)?;
    Ok(())
}

/// Apply a closing operation to an image.
///
/// Closing performs dilation followed by erosion using the same structuring
/// element. It is commonly used to fill small dark holes inside bright regions.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image. It must have the same size as `src`.
/// * `kernel` - The structuring element defining the neighborhood to scan.
/// * `padding_mode` - The border handling mode used before scanning.
/// * `constant_value` - The fill value used when `padding_mode` is
///   [`PaddingMode::Constant`].
///
/// # Returns
///
/// `Ok(())` on success.
///
/// # Errors
///
/// Returns [`ImageError::InvalidImageSize`] if `src` and `dst` sizes differ.
/// Returns [`ImageError::InvalidKernelShape`] if `kernel` is not valid for
/// morphology operations.
///
/// # Example
///
/// ```rust
/// use kornia_image::{allocator::CpuAllocator, Image, ImageSize};
/// use kornia_imgproc::morphology::{close, Kernel, KernelShape};
/// use kornia_imgproc::padding::PaddingMode;
///
/// let src = Image::<u8, 1, _>::new(
///     ImageSize { width: 5, height: 5 },
///     vec![
///         0, 0, 0, 0, 0,
///         0, 255, 255, 255, 0,
///         0, 255, 0, 255, 0,
///         0, 255, 255, 255, 0,
///         0, 0, 0, 0, 0,
///     ],
///     CpuAllocator,
/// )?;
/// let mut dst = Image::<u8, 1, _>::from_size_val(src.size(), 0, CpuAllocator)?;
/// let kernel = Kernel::try_new(KernelShape::Box { size: 3 })?;
///
/// close(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
/// assert_eq!(dst.as_slice()[12], 255);
/// # Ok::<(), kornia_image::ImageError>(())
/// ```
pub fn close<
    T: Copy + Send + Sync + Ord + Default,
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError> {
    validate_inputs(src, dst, kernel)?;

    let mut temp_img = Image::from_size_val(src.size(), T::default(), CpuAllocator)?;
    dilate(src, &mut temp_img, kernel, padding_mode, constant_value)?;
    erode(&temp_img, dst, kernel, padding_mode, constant_value)?;
    Ok(())
}

fn validate_inputs<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &Image<T, C, A2>,
    kernel: &Kernel,
) -> Result<(), ImageError> {
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.width(),
            src.height(),
            dst.width(),
            dst.height(),
        ));
    }

    kernel.validate()
}

fn morphology_op<
    T: Copy + Send + Sync + Ord,
    const C: usize,
    A1: ImageAllocator,
    A2: ImageAllocator,
>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
    op: MorphologyOp,
) -> Result<(), ImageError> {
    validate_inputs(src, dst, kernel)?;

    let width = src.width();
    let height = src.height();
    let padded = pad_image(src, kernel, padding_mode, constant_value)?;
    let padded_width = padded.width();
    let padded_data = padded.as_slice();
    let active_offsets = kernel_offsets(kernel);

    dst.as_slice_mut()
        .par_chunks_exact_mut(width * C)
        .enumerate()
        .for_each(|(y, row_chunk)| {
            for x in 0..width {
                for c in 0..C {
                    let first = active_offsets[0];
                    let mut acc =
                        padded_data[((y + first.1) * padded_width + (x + first.0)) * C + c];

                    for &(kx, ky) in &active_offsets[1..] {
                        let value = padded_data[((y + ky) * padded_width + (x + kx)) * C + c];
                        acc = match op {
                            MorphologyOp::Dilate => acc.max(value),
                            MorphologyOp::Erode => acc.min(value),
                        };
                    }

                    row_chunk[x * C + c] = acc;
                }
            }
        });

    debug_assert_eq!(dst.height(), height);
    Ok(())
}

fn pad_image<T: Copy + Send + Sync, const C: usize, A: ImageAllocator>(
    src: &Image<T, C, A>,
    kernel: &Kernel,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<Image<T, C, CpuAllocator>, ImageError> {
    let (pad_h, pad_w) = kernel.pad();
    let padded_size = ImageSize {
        width: src.width() + 2 * pad_w,
        height: src.height() + 2 * pad_h,
    };
    let mut padded = Image::from_size_val(padded_size, constant_value[0], CpuAllocator)?;

    let padding = Padding2D {
        top: pad_h,
        bottom: pad_h,
        left: pad_w,
        right: pad_w,
    };
    spatial_padding(src, &mut padded, padding, padding_mode, constant_value)?;

    Ok(padded)
}

fn kernel_offsets(kernel: &Kernel) -> Vec<(usize, usize)> {
    let mut offsets = Vec::new();

    for y in 0..kernel.height() {
        for x in 0..kernel.width() {
            if kernel.data()[y * kernel.width() + x] != 0 {
                offsets.push((x, y));
            }
        }
    }

    offsets
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::kernels::{try_box_kernel, KernelShape};

    #[test]
    fn test_box_kernel() {
        let kernel = Kernel::try_new(KernelShape::Box { size: 3 }).unwrap();
        assert_eq!(kernel.width(), 3);
        assert_eq!(kernel.height(), 3);
        assert!(kernel.data().iter().all(|&x| x == 1));
    }

    #[test]
    fn test_cross_kernel() {
        let kernel = Kernel::try_new(KernelShape::Cross { size: 3 }).unwrap();
        let data = kernel.data();
        assert_eq!(data[3], 1);
        assert_eq!(data[4], 1);
        assert_eq!(data[5], 1);
        assert_eq!(data[1], 1);
        assert_eq!(data[7], 1);
        assert_eq!(data[0], 0);
    }

    #[test]
    fn test_ellipse_kernel() {
        let kernel = Kernel::try_new(KernelShape::Ellipse {
            width: 5,
            height: 5,
        })
        .unwrap();
        assert_eq!(kernel.width(), 5);
        assert_eq!(kernel.height(), 5);
        assert_eq!(kernel.data()[12], 1);
    }

    #[test]
    fn test_kernel_padding() {
        let kernel = Kernel::try_new(KernelShape::Box { size: 5 }).unwrap();
        let (pad_h, pad_w) = kernel.pad();
        assert_eq!(pad_h, 2);
        assert_eq!(pad_w, 2);
    }

    #[test]
    fn test_invalid_even_kernel_size() {
        let err = Kernel::try_new(KernelShape::Box { size: 4 }).unwrap_err();
        assert!(matches!(err, ImageError::InvalidKernelShape(_)));
    }

    #[test]
    fn test_invalid_zero_kernel_size() {
        let err = try_box_kernel(0).unwrap_err();
        assert!(matches!(err, ImageError::InvalidKernelShape(_)));
    }

    #[test]
    fn test_dilate_3x3() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 3,
            height: 3,
        };
        let data = vec![0u8, 0, 0, 0, 255, 0, 0, 0, 0];
        let src = Image::new(size, data, CpuAllocator)?;
        let mut dst = Image::new(size, vec![0u8; 9], CpuAllocator)?;

        let kernel = Kernel::try_new(KernelShape::Box { size: 3 })?;
        dilate(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;

        assert!(dst.as_slice().iter().all(|&x| x == 255));
        Ok(())
    }

    #[test]
    fn test_erode_3x3() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 3,
            height: 3,
        };
        let src = Image::new(size, vec![255u8; 9], CpuAllocator)?;
        let mut dst = Image::new(size, vec![0u8; 9], CpuAllocator)?;

        let kernel = Kernel::try_new(KernelShape::Box { size: 3 })?;
        erode(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;

        assert_eq!(dst.as_slice()[4], 255);
        assert_eq!(dst.as_slice()[0], 0);
        Ok(())
    }

    #[test]
    fn test_open_remove_noise() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut data = vec![0u8; 25];
        data[6] = 255;

        let src = Image::new(size, data, CpuAllocator)?;
        let mut dst = Image::new(size, vec![0u8; 25], CpuAllocator)?;

        let kernel = Kernel::try_new(KernelShape::Box { size: 3 })?;
        open(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;

        assert!(dst.as_slice().iter().all(|&x| x == 0));
        Ok(())
    }

    #[test]
    fn test_close_fill_hole() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut data = vec![0u8; 25];
        for y in 1..=3 {
            for x in 1..=3 {
                data[y * 5 + x] = 255;
            }
        }
        data[12] = 0;

        let src = Image::new(size, data, CpuAllocator)?;
        let mut dst = Image::new(size, vec![0u8; 25], CpuAllocator)?;

        let kernel = Kernel::try_new(KernelShape::Box { size: 3 })?;
        close(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;

        assert_eq!(dst.as_slice()[12], 255);
        assert_eq!(dst.as_slice()[6], 255);
        assert_eq!(dst.as_slice()[18], 255);
        Ok(())
    }

    #[test]
    fn test_cross_kernel_dilation() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let mut data = vec![0u8; 25];
        data[12] = 255;

        let src = Image::new(size, data, CpuAllocator)?;
        let mut dst = Image::new(size, vec![0u8; 25], CpuAllocator)?;
        let kernel = Kernel::try_new(KernelShape::Cross { size: 3 })?;

        dilate(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;

        let expected = [
            0, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(dst.as_slice(), expected);
        Ok(())
    }

    #[test]
    fn test_multichannel_dilate() -> Result<(), ImageError> {
        let src = Image::<u8, 3, _>::new(
            ImageSize {
                width: 3,
                height: 3,
            },
            vec![
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            ],
            CpuAllocator,
        )?;
        let mut dst = Image::<u8, 3, _>::from_size_val(src.size(), 0, CpuAllocator)?;
        let kernel = Kernel::try_new(KernelShape::Box { size: 3 })?;

        dilate(&src, &mut dst, &kernel, PaddingMode::Constant, [0, 0, 0])?;

        assert_eq!(dst.get_pixel(0, 0, 0), Ok(&10));
        assert_eq!(dst.get_pixel(0, 0, 1), Ok(&20));
        assert_eq!(dst.get_pixel(0, 0, 2), Ok(&30));
        Ok(())
    }
}
