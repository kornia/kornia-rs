use kornia_image::{allocator::ImageAllocator, Image, ImageError, ImageSize};

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
    /// - `Reflect`: mirror excluding edge  
    /// - `Reflect101`: mirror including edge  
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
    /// - `left`, `right`, `top`, `bottom`: Padding extents in pixels.
    ///
    /// # Notes
    /// - [`PaddingMode::Constant`] is assumed to be already applied when initializing `new_data`.
    /// - Other modes (`Replicate`, `Reflect`, `Reflect101`, `Wrap`) will fill the outer border areas.
    pub fn apply_padding<T: Copy, const C: usize>(
        &self,
        new_data: &mut [T],
        old_width: usize,
        old_height: usize,
        new_width: usize,
        new_height: usize,
        padding: &Padding2D,
    ) {
        let top = padding.top;
        let bottom = padding.bottom;
        let left = padding.left;
        let right = padding.right;

        if matches!(self, PaddingMode::Constant) {
            // constant padding was already handled when initializing new_data
            return;
        }

        // top
        for y in 0..top {
            let src_y = self.map_index(y as isize - top as isize, old_height);
            let dst_row_start = y * new_width * C;
            let src_row_start = (src_y + top) * new_width * C;
            let row_len = new_width * C;

            let temp_src_row = new_data[src_row_start..src_row_start + row_len].to_vec();
            new_data[dst_row_start..dst_row_start + row_len].copy_from_slice(&temp_src_row);
        }

        // bottom
        for y in (new_height - bottom)..new_height {
            let src_y = self.map_index(y as isize - top as isize, old_height);
            let dst_row_start = y * new_width * C;
            let src_row_start = (src_y + top) * new_width * C;
            let row_len = new_width * C;

            let temp_src_row = new_data[src_row_start..src_row_start + row_len].to_vec();
            new_data[dst_row_start..dst_row_start + row_len].copy_from_slice(&temp_src_row);
        }

        // left and right
        for y in 0..new_height {
            let row_start = y * new_width * C;
            let row_end = row_start + new_width * C;
            let row = &mut new_data[row_start..row_end];

            // left
            for x in 0..left {
                let src_x = self.map_index(x as isize - left as isize, old_width);
                let src_idx = (left + src_x) * C;
                let dst_idx = x * C;

                let temp_row = row[src_idx..src_idx + C].to_vec();
                row[dst_idx..dst_idx + C].copy_from_slice(&temp_row);
            }

            // right
            for x in (new_width - right)..new_width {
                let src_x = self.map_index(x as isize - left as isize, old_width);
                let src_idx = (left + src_x) * C;
                let dst_idx = x * C;

                let temp_row = row[src_idx..src_idx + C].to_vec();
                row[dst_idx..dst_idx + C].copy_from_slice(&temp_row);
            }
        }
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
        if new_size.width != old_size.width + self.left + self.right
            || new_size.height != old_size.height + self.top + self.bottom
        {
            return false;
        }
        true
    }
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
#[allow(clippy::too_many_arguments)]
pub fn spatial_padding<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    padding: Padding2D,
    padding_mode: PaddingMode,
    constant_value: [T; C],
) -> Result<(), ImageError>
where
    T: Copy + Default,
{
    if !padding.validate_size(src.size(), dst.size()) {
        return Err(ImageError::InvalidImageSize(
            dst.size().width,
            dst.size().height,
            src.size().width + padding.left + padding.right,
            src.size().height + padding.top + padding.bottom,
        ));
    }

    let old_width = src.size().width;
    let old_height = src.size().height;
    let new_width = dst.size().width;
    let new_height = dst.size().height;

    let old_data = src.as_slice();
    let new_data = dst.storage.as_mut_slice();

    match padding_mode {
        // if constant padding, fill with constant value
        PaddingMode::Constant => {
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
    let new_stride = new_width * C;
    let old_stride = old_width * C;
    let row_len = old_stride;

    let mut old_row_start = 0;
    let mut new_row_start = padding.top * new_stride + padding.left * C;

    for _ in 0..old_height {
        new_data[new_row_start..new_row_start + row_len]
            .copy_from_slice(&old_data[old_row_start..old_row_start + row_len]);

        old_row_start += old_stride;
        new_row_start += new_stride;
    }

    padding_mode.apply_padding::<T, C>(
        new_data, old_width, old_height, new_width, new_height, &padding,
    );

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
            vec![0u8; 48],
            CpuAllocator,
        )
        .unwrap();

        // constant padding
        spatial_padding(
            &src,
            &mut dst,
            Padding2D {
                top: 1,
                bottom: 1,
                left: 1,
                right: 1,
            },
            PaddingMode::Constant,
            [9u8, 9, 9],
        )
        .unwrap();

        // top-left pixel in padded image should be constant (9,9,9)
        assert_eq!(&dst.as_slice()[0..3], &[9, 9, 9]);
        // center (1,1) should correspond to original (0,0)
        assert_eq!(&dst.as_slice()[15..18], &[1, 1, 1]);

        // replicate padding
        spatial_padding(
            &src,
            &mut dst,
            Padding2D {
                top: 1,
                bottom: 1,
                left: 1,
                right: 1,
            },
            PaddingMode::Replicate,
            [0u8, 0, 0],
        )
        .unwrap();

        // top-left should replicate src(0,0)
        assert_eq!(&dst.as_slice()[0..3], &[1, 1, 1]);
        // top-right should replicate src(0,1)
        assert_eq!(&dst.as_slice()[9..12], &[2, 2, 2]);
        // bottom-left should replicate src(1,0)
        assert_eq!(&dst.as_slice()[36..39], &[3, 3, 3]);
    }
}
