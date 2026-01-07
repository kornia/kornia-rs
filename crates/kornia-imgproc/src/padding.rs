use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};
use rayon::prelude::*;

/// A border type for the spatial padding.
#[derive(Debug, Clone, Copy)]
pub enum PaddingMode {
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
    Reflect101,

    /// This border type reflects the pixel values at the boundary, starting with the edge pixel itself.
    ///
    /// Example: ...d c b a | a b c d...
    Reflect,

    /// This border type wraps the content from the opposite side to fill the border.
    ///
    /// Example: ...d c b a | w x y z...
    Wrap,
}
impl PaddingMode {
    #[inline]
    fn reflect(i: isize, len: usize) -> usize {
        if len == 1 {
            return 0;
        }
        let len = len as isize;
        let mut i = i;
        while i < 0 || i >= len {
            if i < 0 {
                i = -i - 1;
            } else if i >= len {
                i = 2 * len - i - 1;
            }
        }
        i as usize
    }

    #[inline]
    fn reflect101(i: isize, len: usize) -> usize {
        if len == 1 {
            return 0;
        }
        let len = len as isize;
        let mut i = i;
        while i < 0 || i >= len {
            if i < 0 {
                i = -i;
            } else if i >= len {
                i = 2 * len - i - 2;
            }
        }
        i as usize
    }

    #[inline]
    fn wrap(i: isize, len: usize) -> usize {
        ((i % len as isize + len as isize) % len as isize) as usize
    }

    /// Maps index `i` to a valid index i.e. within `[0, len)` according to the padding mode.
    ///
    /// - `Replicate`: clamp to edge
    /// - `Reflect`: mirror including edge
    /// - `Reflect101`: mirror excluding edge
    /// - `Wrap`: circular wrap
    /// - `Constant`: returns 0 (not used directly)
    ///
    /// # Arguments
    /// - `i`: The (possibly out-of-range) coordinate index.
    /// - `len`: The valid length of the dimension.
    ///
    /// # Returns
    /// A valid mapped index within `[0, len)`.
    #[inline]
    pub fn map_index(&self, i: isize, len: usize) -> usize {
        match self {
            PaddingMode::Replicate => i.clamp(0, len as isize - 1) as usize,
            PaddingMode::Reflect => Self::reflect(i, len),
            PaddingMode::Reflect101 => Self::reflect101(i, len),
            PaddingMode::Wrap => Self::wrap(i, len),
            PaddingMode::Constant => 0,
        }
    }

    /// Applies the selected padding mode to fill image borders in `new_data`.
    ///
    /// # Arguments
    /// - `new_data`: Target image buffer (already containing the original image in the center).
    /// - `old_width`, `old_height`: Dimensions of the original image.
    /// - `new_width`, `new_height`: Dimensions of the padded image.
    /// - `padding`: `left`, `right`, `top` and `bottom` padding extents in pixels.
    ///
    /// # Notes
    /// - [`PaddingMode::Constant`] is assumed to be already applied when initializing `new_data`.
    /// - Other modes (`Replicate`, `Reflect`, `Reflect101`, `Wrap`) will fill the outer border areas.
    pub fn apply_padding<T: Copy + Send + Sync, const C: usize>(
        &self,
        new_data: &mut [T],
        old_width: usize,
        old_height: usize,
        new_width: usize,
        new_height: usize,
        padding: &Padding2D,
    ) {
        if let PaddingMode::Constant = self {
            return; // already filled
        }

        let top = padding.top;
        let bottom = padding.bottom;
        let left = padding.left;
        let right = padding.right;
        let row_stride = new_width * C;

        // top
        {
            let (top_section, rest) = new_data.split_at_mut(top * row_stride);

            top_section
                .par_chunks_exact_mut(row_stride)
                .enumerate()
                .for_each(|(y, dst_row)| {
                    let src_y = self.map_index(y as isize - top as isize, old_height);
                    let src_row = &rest[src_y * row_stride..(src_y + 1) * row_stride];
                    dst_row.copy_from_slice(src_row);
                });
        }

        // bottom
        {
            let split_point = (new_height - bottom) * row_stride;
            let (rest, bottom_section) = new_data.split_at_mut(split_point);

            bottom_section
                .par_chunks_exact_mut(row_stride)
                .enumerate()
                .for_each(|(idx, dst_row)| {
                    let y = new_height - bottom + idx;
                    let src_y = self.map_index(y as isize - top as isize, old_height);
                    let src_start = (src_y + top) * row_stride;
                    let src_row = &rest[src_start..src_start + row_stride];
                    dst_row.copy_from_slice(src_row);
                });
        }

        new_data.par_chunks_exact_mut(row_stride).for_each(|row| {
            // left
            for x in 0..left {
                let src_x = self.map_index(x as isize - left as isize, old_width);
                let src_idx = (left + src_x) * C;
                let dst_idx = x * C;
                row.copy_within(src_idx..src_idx + C, dst_idx);
            }

            // right
            for x in (new_width - right)..new_width {
                let src_x = self.map_index(x as isize - left as isize, old_width);
                let src_idx = (left + src_x) * C;
                let dst_idx = x * C;
                row.copy_within(src_idx..src_idx + C, dst_idx);
            }
        });
    }
}

/// Represents 2D padding with top, bottom, left, and right values (in pixels).
pub struct Padding2D {
    /// Amount of padding to add on the top side.
    pub top: usize,
    /// Amount of padding to add on the bottom side.
    pub bottom: usize,
    /// Amount of padding to add on the left side.
    pub left: usize,
    /// Amount of padding to add on the right side.
    pub right: usize,
}
impl Padding2D {
    /// Validates that a new image size correctly matches the expected dimensions
    /// after applying this padding to an existing image.
    ///
    /// # Arguments
    /// - `old_size`: The original image size before padding.
    /// - `new_size`: The resulting image size after padding.
    ///
    /// # Returns
    /// - `true` if the `new_size` width and height are equal to
    ///   `old_size.width + left + right` and `old_size.height + top + bottom`, respectively.
    /// - `false` otherwise.
    ///
    /// # Example
    /// ```rust
    /// use kornia_image::ImageSize;
    /// use kornia_imgproc::padding::Padding2D;
    /// let padding = Padding2D { top: 1, bottom: 1, left: 2, right: 2 };
    /// let old_size = ImageSize { width: 4, height: 4 };
    /// let new_size = ImageSize { width: 8, height: 6 };
    ///
    /// assert!(padding.validate_size(old_size, new_size));
    /// ```
    pub fn validate_size(&self, old_size: ImageSize, new_size: ImageSize) -> bool {
        new_size.width == old_size.width + self.left + self.right
            && new_size.height == old_size.height + self.top + self.bottom
    }
}

/// Creates a new image with spatial padding applied to reach target size,
/// centering the original image and using the specified fill value and type.
///
/// # Arguments
///
/// * `src` - The source image to pad.
/// * `dst` - The destination image where the padded output will be stored.
/// * `padding` - The amount of padding (in pixels) for all four sides defined in [`Padding2D`] (top, bottom, left, right).
/// * `padding_mode` - The type of border handling to use defined in [`PaddingMode`] (e.g., Constant, Replicate, Reflect, Reflect101, Wrap).
/// * `constant_value` - The pixel value used for constant padding, specified as an array of length `C` (one value per channel).
///
/// # Errors
///
/// Returns an error if the size of `dst` does not match with the expected size
/// i.e. after applying padding specified in argument `padding` on `src`.
///
/// # Example
///
/// ```rust
/// use kornia_image::{allocator::CpuAllocator, ImageSize, Image};
/// use kornia_imgproc::padding::{PaddingMode, Padding2D, spatial_padding};
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
///     Padding2D { top: 1, bottom: 1, left: 1, right: 1 },
///     PaddingMode::Constant,
///     [0u8; 3],
/// ).unwrap();
///
/// // The resulting image should now be 4x4 in size
/// assert_eq!(dst.size().width, 4);
/// assert_eq!(dst.size().height, 4);
/// ```
pub fn spatial_padding<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    padding: Padding2D,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError>
where
    T: Copy + Default + Send + Sync,
{
    if !padding.validate_size(src.size(), dst.size()) {
        return Err(ImageError::InvalidImageSize(
            dst.width(),
            dst.height(),
            src.width() + padding.left + padding.right,
            src.height() + padding.top + padding.bottom,
        ));
    }

    let old_width = src.width();
    let old_height = src.height();
    let new_width = dst.width();
    let new_height = dst.height();

    let old_data = src.as_slice();
    let new_data = dst.as_slice_mut();

    match padding_mode {
        // if constant padding, fill with constant value
        PaddingMode::Constant => {
            new_data
                .chunks_exact_mut(C)
                .for_each(|chunk| chunk.copy_from_slice(&constant_value));
        }
        _ => {
            new_data.fill(T::default());
        }
    }

    // copy old image data as center of new image data
    let new_stride = new_width * C;
    let old_stride = old_width * C;

    let row_offset = padding.top * new_stride + padding.left * C;

    for (src_row, dst_row) in old_data
        .chunks_exact(old_stride)
        .zip(new_data[row_offset..].chunks_exact_mut(new_stride))
    {
        dst_row[..old_stride].copy_from_slice(src_row);
    }

    padding_mode.apply_padding::<T, C>(
        new_data, old_width, old_height, new_width, new_height, &padding,
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::{allocator::CpuAllocator, Image, ImageError, ImageSize};

    // helper functions
    fn make_src_2x2_rgb() -> Result<Image<u8, 3, CpuAllocator>, ImageError> {
        Image::new(
            ImageSize {
                width: 2,
                height: 2,
            },
            vec![1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            CpuAllocator,
        )
    }

    fn make_dst_4x4_rgb() -> Result<Image<u8, 3, CpuAllocator>, ImageError> {
        Image::new(
            ImageSize {
                width: 4,
                height: 4,
            },
            vec![0u8; 48],
            CpuAllocator,
        )
    }

    const PAD_1: Padding2D = Padding2D {
        top: 1,
        bottom: 1,
        left: 1,
        right: 1,
    };

    #[test]
    fn test_spatial_padding_constant() -> Result<(), ImageError> {
        let src = make_src_2x2_rgb()?;
        let mut dst = make_dst_4x4_rgb()?;

        spatial_padding(&src, &mut dst, PAD_1, PaddingMode::Constant, [9, 9, 9])?;

        let d = dst.as_slice();

        // corners
        assert_eq!(&d[0..3], &[9, 9, 9]);
        assert_eq!(&d[45..48], &[9, 9, 9]);

        // top edge
        assert_eq!(&d[3..6], &[9, 9, 9]);

        // actual image
        assert_eq!(&d[15..18], &[1, 1, 1]);
        assert_eq!(&d[30..33], &[4, 4, 4]);

        Ok(())
    }

    #[test]
    fn test_spatial_padding_replicate() -> Result<(), ImageError> {
        let src = make_src_2x2_rgb()?;
        let mut dst = make_dst_4x4_rgb()?;

        spatial_padding(&src, &mut dst, PAD_1, PaddingMode::Replicate, [0, 0, 0])?;

        let d = dst.as_slice();

        // corners
        assert_eq!(&d[0..3], &[1, 1, 1]);
        assert_eq!(&d[45..48], &[4, 4, 4]);

        // edges
        assert_eq!(&d[3..6], &[1, 1, 1]);
        assert_eq!(&d[21..24], &[2, 2, 2]);

        Ok(())
    }

    #[test]
    fn test_spatial_padding_reflect101() -> Result<(), ImageError> {
        let src = make_src_2x2_rgb()?;
        let mut dst = make_dst_4x4_rgb()?;

        spatial_padding(&src, &mut dst, PAD_1, PaddingMode::Reflect101, [0, 0, 0])?;

        let d = dst.as_slice();

        // corners
        assert_eq!(&d[0..3], &[4, 4, 4]);
        assert_eq!(&d[9..12], &[3, 3, 3]);

        // top edge
        assert_eq!(&d[3..6], &[3, 3, 3]);

        // actual image
        assert_eq!(&d[15..18], &[1, 1, 1]);

        Ok(())
    }

    #[test]
    fn test_spatial_padding_reflect() -> Result<(), ImageError> {
        let src = make_src_2x2_rgb()?;
        let mut dst = make_dst_4x4_rgb()?;

        spatial_padding(&src, &mut dst, PAD_1, PaddingMode::Reflect, [0, 0, 0])?;

        let d = dst.as_slice();

        // corners
        assert_eq!(&d[0..3], &[1, 1, 1]);
        assert_eq!(&d[9..12], &[2, 2, 2]);

        // edges
        assert_eq!(&d[6..9], &[2, 2, 2]);
        assert_eq!(&d[39..42], &[3, 3, 3]);

        Ok(())
    }

    #[test]
    fn test_spatial_padding_wrap() -> Result<(), ImageError> {
        let src = make_src_2x2_rgb()?;
        let mut dst = make_dst_4x4_rgb()?;

        spatial_padding(&src, &mut dst, PAD_1, PaddingMode::Wrap, [0, 0, 0])?;

        let d = dst.as_slice();

        // corners
        assert_eq!(&d[0..3], &[4, 4, 4]);
        assert_eq!(&d[9..12], &[3, 3, 3]);
        assert_eq!(&d[36..39], &[2, 2, 2]);
        assert_eq!(&d[45..48], &[1, 1, 1]);

        // edges
        assert_eq!(&d[12..15], &[2, 2, 2]);

        Ok(())
    }

    #[test]
    fn test_spatial_padding_dst_size_mismatch() -> Result<(), ImageError> {
        let src = make_src_2x2_rgb()?;
        let mut dst = Image::<u8, 3, _>::new(
            ImageSize {
                width: 3,
                height: 4,
            },
            vec![0u8; 36],
            CpuAllocator,
        )?;

        let res = spatial_padding(&src, &mut dst, PAD_1, PaddingMode::Replicate, [0, 0, 0]);
        assert!(res.is_err());

        Ok(())
    }

    #[test]
    fn test_spatial_padding_larger_than_image_replicate() -> Result<(), ImageError> {
        let src = Image::<u8, 3, _>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![7, 7, 7],
            CpuAllocator,
        )?;

        let padding = Padding2D {
            top: 3,
            bottom: 3,
            left: 4,
            right: 4,
        };

        let mut dst = Image::<u8, 3, _>::new(
            ImageSize {
                width: 9,
                height: 7,
            },
            vec![0u8; 189],
            CpuAllocator,
        )?;

        spatial_padding(&src, &mut dst, padding, PaddingMode::Replicate, [0, 0, 0])?;

        for px in dst.as_slice().chunks_exact(3) {
            assert_eq!(px, &[7, 7, 7]);
        }

        Ok(())
    }

    #[test]
    fn test_spatial_padding_larger_than_image_wrap() -> Result<(), ImageError> {
        let src = Image::<u8, 3, _>::new(
            ImageSize {
                width: 1,
                height: 1,
            },
            vec![5, 5, 5],
            CpuAllocator,
        )?;

        let padding = Padding2D {
            top: 2,
            bottom: 2,
            left: 2,
            right: 2,
        };

        let mut dst = Image::<u8, 3, _>::new(
            ImageSize {
                width: 5,
                height: 5,
            },
            vec![0u8; 75],
            CpuAllocator,
        )?;

        spatial_padding(&src, &mut dst, padding, PaddingMode::Wrap, [0, 0, 0])?;

        for px in dst.as_slice().chunks_exact(3) {
            assert_eq!(px, &[5, 5, 5]);
        }

        Ok(())
    }
}
