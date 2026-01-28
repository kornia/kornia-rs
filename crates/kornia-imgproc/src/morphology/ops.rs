use super::kernels::Kernel;
use crate::padding::{spatial_padding, Padding2D, PaddingMode};
use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use kornia_tensor::CpuAllocator;
use rayon::prelude::*;

/// Dilate an image using a [`Kernel`].
///
/// Dilation expands white regions in the image. Each pixel is replaced
/// by the maximum value in the neighborhood defined by the kernel.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image (will be overwritten).
/// * `kernel` - The morphological structuring element ([`Kernel`]).
/// * `padding_mode` - The border handling mode ([`PaddingMode`]).
/// * `constant_value` - The fill value for constant padding.
///
/// # Returns
///
/// Ok(()) on success, or [`ImageError`] if shapes don't match.
pub fn dilate<
    T: Copy + Default + Send + Sync + Ord,
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
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            src.width(),
            src.height(),
        ));
    }

    let width = src.width();
    let height = src.height();
    let (pad_h, pad_w) = kernel.pad();
    let k_height = kernel.height();
    let k_width = kernel.width();
    let k_data = kernel.data();

    let padded_size = ImageSize {
        width: width + 2 * pad_w,
        height: height + 2 * pad_h,
    };
    let padded_buffer = vec![T::default(); padded_size.width * padded_size.height * C];
    let mut padded = Image::new(padded_size, padded_buffer, CpuAllocator)?;

    let padding = Padding2D {
        top: pad_h,
        bottom: pad_h,
        left: pad_w,
        right: pad_w,
    };
    spatial_padding(src, &mut padded, padding, padding_mode, constant_value)?;

    // dilation
    let dst_slice = dst.as_slice_mut();
    let dst_chunks: Vec<_> = dst_slice.chunks_mut(width * C).collect();

    dst_chunks
        .into_par_iter()
        .enumerate()
        .for_each(|(h, row_chunk)| {
            for c in 0..C {
                for w in 0..width {
                    let mut max_val = T::default();

                    for kh in 0..k_height {
                        for kw in 0..k_width {
                            if k_data[kh * k_width + kw] == 1 {
                                let px = w + kw;
                                let py = h + kh;
                                if let Ok(pixel) = padded.get_pixel(px, py, c) {
                                    max_val = max_val.max(*pixel);
                                }
                            }
                        }
                    }

                    let idx = w * C + c;
                    row_chunk[idx] = max_val;
                }
            }
        });

    Ok(())
}

/// Erode an image using a [`Kernel`].
///
/// Erosion shrinks white regions in the image. Each pixel is replaced
/// by the minimum value in the neighborhood defined by the kernel.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image (will be overwritten).
/// * `kernel` - The morphological structuring element ([`Kernel`]).
/// * `padding_mode` - The border handling mode ([`PaddingMode`]).
/// * `constant_value` - The fill value for constant padding.
///
/// # Returns
///
/// Ok(()) on success, or [`ImageError`] if shapes don't match.
pub fn erode<
    T: Copy + Default + Send + Sync + Ord,
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
    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            src.width(),
            src.height(),
        ));
    }

    let width = src.width();
    let height = src.height();
    let (pad_h, pad_w) = kernel.pad();
    let k_height = kernel.height();
    let k_width = kernel.width();
    let k_data = kernel.data();

    let padded_size = ImageSize {
        width: width + 2 * pad_w,
        height: height + 2 * pad_h,
    };
    let padded_buffer = vec![T::default(); padded_size.width * padded_size.height * C];
    let mut padded = Image::new(padded_size, padded_buffer, CpuAllocator)?;

    let padding = Padding2D {
        top: pad_h,
        bottom: pad_h,
        left: pad_w,
        right: pad_w,
    };
    spatial_padding(src, &mut padded, padding, padding_mode, constant_value)?;

    // erosion
    let dst_slice = dst.as_slice_mut();
    let dst_chunks: Vec<_> = dst_slice.chunks_mut(width * C).collect();

    dst_chunks
        .into_par_iter()
        .enumerate()
        .for_each(|(h, row_chunk)| {
            for c in 0..C {
                for w in 0..width {
                    let mut min_val: Option<T> = None;

                    for kh in 0..k_height {
                        for kw in 0..k_width {
                            if k_data[kh * k_width + kw] == 1 {
                                let px = w + kw;
                                let py = h + kh;
                                if let Ok(pixel) = padded.get_pixel(px, py, c) {
                                    min_val = Some(match min_val {
                                        None => *pixel,
                                        Some(v) => v.min(*pixel),
                                    });
                                }
                            }
                        }
                    }

                    let idx = w * C + c;
                    row_chunk[idx] = min_val.unwrap_or_default();
                }
            }
        });

    Ok(())
}

/// Opening: erosion followed by dilation.
///
/// Removes small objects and smooths object boundaries.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image (will be overwritten).
/// * `kernel` - The morphological structuring element ([`Kernel`]).
/// * `padding_mode` - The border handling mode ([`PaddingMode`]).
/// * `constant_value` - The fill value for constant padding.
///
/// # Returns
///
/// Ok(()) on success, or [`ImageError`] if shapes don't match.
pub fn open<
    T: Copy + Default + Send + Sync + Ord,
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
    let mut temp_img = src.clone();
    erode(src, &mut temp_img, kernel, padding_mode, constant_value)?;
    dilate(&temp_img, dst, kernel, padding_mode, constant_value)?;
    Ok(())
}

/// Closing: dilation followed by erosion.
///
/// Fills small holes and smooths object boundaries.
///
/// # Arguments
///
/// * `src` - The source image.
/// * `dst` - The destination image (will be overwritten).
/// * `kernel` - The morphological structuring element ([`Kernel`]).
/// * `padding_mode` - The border handling mode ([`PaddingMode`]).
/// * `constant_value` - The fill value for constant padding.
///
/// # Returns
///
/// Ok(()) on success, or [`ImageError`] if shapes don't match.
pub fn close<
    T: Copy + Default + Send + Sync + Ord,
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
    let mut temp_img = src.clone();
    dilate(src, &mut temp_img, kernel, padding_mode, constant_value)?;
    erode(&temp_img, dst, kernel, padding_mode, constant_value)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::kernels::KernelShape;

    #[test]
    fn test_box_kernel() {
        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        assert_eq!(kernel.width(), 3);
        assert_eq!(kernel.height(), 3);
        assert!(kernel.data().iter().all(|&x| x == 1));
    }

    #[test]
    fn test_cross_kernel() {
        let kernel = Kernel::new(KernelShape::Cross { size: 3 });
        let data = kernel.data();
        // center row
        assert_eq!(data[3], 1);
        assert_eq!(data[4], 1);
        assert_eq!(data[5], 1);
        // center column
        assert_eq!(data[1], 1);
        assert_eq!(data[7], 1);
        // corners
        assert_eq!(data[0], 0);
    }

    #[test]
    fn test_ellipse_kernel() {
        let kernel = Kernel::new(KernelShape::Ellipse {
            width: 5,
            height: 5,
        });
        assert_eq!(kernel.width(), 5);
        assert_eq!(kernel.height(), 5);
        // center
        assert_eq!(kernel.data()[12], 1);
    }

    #[test]
    fn test_kernel_padding() {
        let kernel = Kernel::new(KernelShape::Box { size: 5 });
        let (pad_h, pad_w) = kernel.pad();
        assert_eq!(pad_h, 2);
        assert_eq!(pad_w, 2);
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

        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        dilate(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        let result = dst.as_slice();

        assert!(
            result.iter().all(|&x| x == 255),
            "All pixels should be 255 after dilation"
        );
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

        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        erode(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        let result = dst.as_slice();

        assert_eq!(result[4], 255, "Center pixel should survive erosion");
        assert_eq!(
            result[0], 0,
            "Corner pixel should be eroded due to zero padding"
        );
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

        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        open(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        let result = dst.as_slice();

        assert!(
            result.iter().all(|&x| x == 0),
            "Opening should remove the isolated noise pixel"
        );
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

        let kernel = Kernel::new(KernelShape::Box { size: 3 });
        close(&src, &mut dst, &kernel, PaddingMode::Constant, [0])?;
        let result = dst.as_slice();

        assert_eq!(
            result[12], 255,
            "Closing should fill the hole in the center"
        );
        assert_eq!(result[6], 255);
        assert_eq!(result[18], 255);
        Ok(())
    }
}
