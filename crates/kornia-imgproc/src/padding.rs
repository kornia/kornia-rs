use kornia_image::{allocator::ImageAllocator, Image, ImageError};

/// A border type for the spatial padding.
#[derive(Debug, Clone, Copy)]
pub enum BorderType {
    /// This border type fills the border with a single, constant color value.
    ///
    /// Example: ...d c b a | 0 0 0 0...
    Constant,

    /// This border type takes the outermost row or column of pixels and repeats it into the padded region.
    ///
    /// Example: ...d c b a | a a a a...
    Replicate,

    /// This border type reflects the pixel values at the boundary, starting with the pixel 'next' to the edge.
    ///
    /// Example: ...d c b a | b c d e...
    Reflect,

    /// This border type reflects the pixel values at the boundary, starting with the edge pixel itself.
    ///
    /// Example: ...d c b a | a b c d...
    Reflect101,

    /// This border type wraps the content from the opposite side to fill the border.
    ///
    /// Example: ...d c b a | w x y z...
    Wrap,
}

/// Creates a new image with spatial padding applied to reach target size,
/// centering the original image and using the specified fill value and type.
///
/// # Arguments
///
/// * `src` - The source image to pad.
/// * `dst` - The destination image where the padded output will be stored.
/// * `top` - The number of pixels to pad on the top edge.
/// * `bottom` - The number of pixels to pad on the bottom edge.
/// * `left` - The number of pixels to pad on the left edge.
/// * `right` - The number of pixels to pad on the right edge.
/// * `border_type` - The type of border handling to use (e.g., Constant, Replicate, Reflect, Reflect101, Wrap).
/// * `constant_value` - The pixel value used for constant padding, specified as an array of length `C` (one value per channel).
///
/// # Example
///
/// ```rust
/// use kornia_image::{allocator::CpuAllocator, ImageSize, Image};
/// use kornia_imgproc::padding::{BorderType, spatial_padding};
///
/// // Create a 2x2 RGB image filled with 1s
/// let src = Image::<u8, 3, _>::new(
///     ImageSize { width: 2, height: 2 },
///     vec![1u8; 2 * 2 * 3],
///     CpuAllocator,
/// ).unwrap();
///
/// // Create destination image
/// let mut dst = Image::<u8, 3, _>::new(
///     ImageSize { width: 4, height: 4 },
///     vec![0u8; 4 * 4 * 3],
///     CpuAllocator,
/// ).unwrap();
///
/// // Apply 1-pixel constant padding with black (0) border
/// spatial_padding(
///     &src,
///     &mut dst,
///     1, 1, 1, 1,
///     BorderType::Constant,
///     [0u8; 3],
/// ).unwrap();
///
/// // The resulting image should now be 4x4 in size
/// assert_eq!(dst.size().width, 4);
/// assert_eq!(dst.size().height, 4);
/// ```
#[allow(clippy::too_many_arguments)]
pub fn spatial_padding<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    top: usize,
    bottom: usize,
    left: usize,
    right: usize,
    border_type: BorderType,
    constant_value: [T; C],
) -> Result<(), ImageError>
where
    T: Copy + Clone + Default,
{
    let old_width = src.size().width;
    let old_height = src.size().height;
    let new_width = dst.size().width;
    let new_height = dst.size().height;

    if new_width != old_width + left + right || new_height != old_height + top + bottom {
        return Err(ImageError::InvalidImageSize(
            new_width,
            new_height,
            old_width + left + right,
            old_height + top + bottom,
        ));
    }

    let old_data = src.storage.as_slice();
    let new_data = dst.storage.as_mut_slice();

    match border_type {
        // if constant padding, fill with constant value
        BorderType::Constant => {
            for chunk in new_data.chunks_exact_mut(C) {
                chunk.copy_from_slice(&constant_value);
            }
        }
        _ => {
            for v in new_data.iter_mut() {
                *v = T::default();
            }
        }
    }

    // copy old image data as center of new image data
    for y in 0..old_height {
        let dst_row_start = (y + top) * new_width * C + left * C;
        let src_row_start = y * old_width * C;
        let row_len = old_width * C;

        new_data[dst_row_start..dst_row_start + row_len]
            .copy_from_slice(&old_data[src_row_start..src_row_start + row_len]);
    }

    match border_type {
        BorderType::Constant => {
            // constant padding was already handled when initializing new_data
        }

        BorderType::Replicate | BorderType::Reflect | BorderType::Reflect101 | BorderType::Wrap => {
            let reflect = |i: isize, len: usize| -> usize {
                if len == 1 {
                    return 0;
                }
                let len = len as isize;
                let mut i = i;

                // for out of range padding, iterate till within range
                while i < 0 || i >= len {
                    if i < 0 {
                        i = -i - 1;
                    } else if i >= len {
                        i = 2 * len - i - 1;
                    }
                }
                i as usize
            };

            let reflect101 = |i: isize, len: usize| -> usize {
                if len == 1 {
                    return 0;
                }
                let len = len as isize;
                let mut i = i;

                // for out of range padding, iterate till within range
                while i < 0 || i >= len {
                    if i < 0 {
                        i = -i;
                    } else if i >= len {
                        i = 2 * len - i - 2;
                    }
                }
                i as usize
            };

            let wrap = |i: isize, len: usize| -> usize {
                ((i % len as isize + len as isize) % len as isize) as usize
            };

            // helper function for mapping coordinate transformation
            let map_index = |i: isize, len: usize| -> usize {
                match border_type {
                    BorderType::Replicate => i.clamp(0, len as isize - 1) as usize,
                    BorderType::Reflect => reflect(i, len),
                    BorderType::Reflect101 => reflect101(i, len),
                    BorderType::Wrap => wrap(i, len),
                    _ => 0,
                }
            };

            // top
            for y in 0..top {
                let src_y = map_index(y as isize - top as isize, old_height);
                let dst_row_start = y * new_width * C;
                let src_row_start = (src_y + top) * new_width * C;
                let row_len = new_width * C;

                let temp_src_row = new_data[src_row_start..src_row_start + row_len].to_vec();
                new_data[dst_row_start..dst_row_start + row_len].copy_from_slice(&temp_src_row);
            }

            // bottom
            for y in (new_height - bottom)..new_height {
                let src_y = map_index(y as isize - top as isize, old_height);
                let dst_row_start = y * new_width * C;
                let src_row_start = (src_y + top) * new_width * C;
                let row_len = new_width * C;

                let temp_src_row = new_data[src_row_start..src_row_start + row_len].to_vec();
                new_data[dst_row_start..dst_row_start + row_len].copy_from_slice(&temp_src_row);
            }

            for y in 0..new_height {
                let row_start = y * new_width * C;
                let row_end = row_start + new_width * C;
                let row = &mut new_data[row_start..row_end];

                // left
                for x in 0..left {
                    let src_x = map_index(x as isize - left as isize, old_width);
                    let src_idx = (left + src_x) * C;
                    let dst_idx = x * C;

                    let temp_row = row[src_idx..src_idx + C].to_vec();
                    row[dst_idx..dst_idx + C].copy_from_slice(&temp_row);
                }

                // right
                for x in (new_width - right)..new_width {
                    let src_x = map_index(x as isize - left as isize, old_width);
                    let src_idx = (left + src_x) * C;
                    let dst_idx = x * C;

                    let temp_row = row[src_idx..src_idx + C].to_vec();
                    row[dst_idx..dst_idx + C].copy_from_slice(&temp_row);
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{allocator::CpuAllocator, Image, ImageSize};

    #[test]
    fn test_spatial_padding_constant_and_replicate() {
        // original 2x2 RGB image (C = 3)
        let src_data = vec![1u8, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4];
        let src = Image::<u8, 3, _>::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            src_data,
            CpuAllocator,
        )
        .unwrap();

        // destination image (4x4 RGB)
        let mut dst = Image::<u8, 3, _>::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![0u8; 4 * 4 * 3],
            CpuAllocator,
        )
        .unwrap();

        // constant padding
        spatial_padding(
            &src,
            &mut dst,
            1,
            1,
            1,
            1,
            BorderType::Constant,
            [9u8, 9, 9],
        )
        .unwrap();

        // top-left pixel in padded image should be constant (9,9,9)
        assert_eq!(&dst.as_slice()[0..3], &[9, 9, 9]);
        // center (1,1) should correspond to original (0,0)
        assert_eq!(
            &dst.as_slice()[(4 + 1) * 3..(4 + 1) * 3 + 3],
            &[1, 1, 1]
        );

        // replicate padding
        spatial_padding(
            &src,
            &mut dst,
            1,
            1,
            1,
            1,
            BorderType::Replicate,
            [0u8, 0, 0],
        )
        .unwrap();

        // top-left should replicate src(0,0)
        assert_eq!(&dst.as_slice()[0..3], &[1, 1, 1]);
        // top-right should replicate src(0,1)
        assert_eq!(&dst.as_slice()[3 * 3..3 * 3 + 3], &[2, 2, 2]);
        // bottom-left should replicate src(1,0)
        assert_eq!(
            &dst.as_slice()[(3 * 4) * 3..(3 * 4) * 3 + 3],
            &[3, 3, 3]
        );
    }
}
