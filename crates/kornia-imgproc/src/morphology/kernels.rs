use kornia_image::ImageError;

/// Shapes of morphological `Kernels`.
///
/// Defines the geometry of the kernel used in morphological operations.
/// All kernels are centered at their geometric center.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelShape {
    /// A rectangular box structuring element.
    ///
    /// All pixels within the box are included in the morphological operation.
    Box {
        /// `size` - The side length of the square kernel (size x size).
        size: usize,
    },

    /// A cross (plus) shaped structuring element.
    ///
    /// Only pixels along the horizontal and vertical center lines are included,
    /// forming a plus/cross pattern.
    Cross {
        /// `size` - The side length of the square cross kernel (size x size).
        size: usize,
    },

    /// An ellipse (or circle) shaped structuring element.
    ///
    /// Pixels inside the elliptical boundary are included, defined by the
    /// equation: (x^2/ rx^2) + (y^2 / ry^2) <= 1.
    Ellipse {
        /// * `width` - The width of the ellipse.
        width: usize,
        /// * `height` - The height of the ellipse.
        height: usize,
    },
}

/// A morphological structuring element.
///
/// The kernel defines the neighborhood structure used in morphological operations
/// (dilate, erode, open, close). It stores a binary mask where 1 indicates pixels
/// included in the operation and 0 indicates excluded pixels.
///
/// # Arguments
///
/// * `data` - A flat vector containing the kernel values (0 or 1).
/// * `width` - The width of the kernel in pixels.
/// * `height` - The height of the kernel in pixels.
///
/// # Example
///
/// ```rust
/// use kornia_imgproc::morphology::{Kernel, KernelShape};
///
/// // Create a 3x3 box kernel
/// let kernel = Kernel::new(KernelShape::Box { size: 3 });
/// assert_eq!(kernel.width(), 3);
/// assert_eq!(kernel.height(), 3);
/// assert_eq!(kernel.pad(), (1, 1));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kernel {
    pub(crate) data: Vec<u8>,
    pub(crate) width: usize,
    pub(crate) height: usize,
}

impl Kernel {
    /// Create a morphological kernel from a shape, validating dimensions.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the kernel (Box, Cross, or Ellipse).
    ///
    /// # Returns
    ///
    /// A [`Kernel`] struct on success, or [`ImageError::InvalidKernelLength`]
    /// if dimensions are zero or overflow.
    pub fn try_new(shape: KernelShape) -> Result<Self, ImageError> {
        match shape {
            KernelShape::Box { size } => try_box_kernel(size),
            KernelShape::Cross { size } => try_cross_kernel(size),
            KernelShape::Ellipse { width, height } => try_ellipse_kernel(width, height),
        }
    }

    /// Create a morphological kernel from a shape.
    ///
    /// # Arguments
    ///
    /// * `shape` - The shape of the kernel (Box, Cross, or Ellipse).
    ///
    /// # Returns
    ///
    /// A [`Kernel`] struct with the appropriate data.
    ///
    /// # Panics
    ///
    /// Panics if dimensions are 0 or overflow. For a fallible constructor, use [`Kernel::try_new`].
    pub fn new(shape: KernelShape) -> Self {
        Self::try_new(shape).expect("Invalid kernel length")
    }

    /// Check whether the kernel is valid (non-empty dimensions, matched data length,
    /// and at least one active tap).
    pub fn is_valid(&self) -> bool {
        if self.width == 0 || self.height == 0 {
            return false;
        }
        match self.width.checked_mul(self.height) {
            Some(len) if len == self.data.len() => self.data.contains(&1),
            _ => false,
        }
    }

    /// Get a reference to the kernel data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the width of the kernel.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the height of the kernel.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the padding for the kernel (offset from center).
    pub fn pad(&self) -> (usize, usize) {
        (self.height / 2, self.width / 2)
    }
}

/// Fallibly create a box structuring element.
///
/// # Arguments
///
/// * `size` - The size of the kernel (size x size).
///
/// # Returns
///
/// A [`Kernel`] filled with 1s, or [`ImageError::InvalidKernelLength`] if `size == 0`
/// or multiplication overflows.
pub fn try_box_kernel(size: usize) -> Result<Kernel, ImageError> {
    if size == 0 {
        return Err(ImageError::InvalidKernelLength(size, size));
    }
    let len = size
        .checked_mul(size)
        .ok_or(ImageError::InvalidKernelLength(size, size))?;
    let data = vec![1u8; len];
    Ok(Kernel {
        data,
        width: size,
        height: size,
    })
}

/// Create a box structuring element.
///
/// # Arguments
///
/// * `size` - The size of the kernel (size x size).
///
/// # Returns
///
/// A [`Kernel`] filled with 1s.
///
/// # Panics
///
/// Panics if `size == 0` or multiplication overflows. Use [`try_box_kernel`] for fallible creation.
pub fn box_kernel(size: usize) -> Kernel {
    try_box_kernel(size).expect("Invalid kernel length")
}

/// Fallibly create a cross structuring element.
///
/// # Arguments
///
/// * `size` - The size of the kernel (size x size).
///
/// # Returns
///
/// A [`Kernel`] with 1s along the horizontal and vertical center lines,
/// or [`ImageError::InvalidKernelLength`] if `size == 0` or multiplication overflows.
pub fn try_cross_kernel(size: usize) -> Result<Kernel, ImageError> {
    if size == 0 {
        return Err(ImageError::InvalidKernelLength(size, size));
    }
    let len = size
        .checked_mul(size)
        .ok_or(ImageError::InvalidKernelLength(size, size))?;
    let mut data = vec![0u8; len];
    let mid = size / 2;

    // fill horizontal line
    for j in 0..size {
        data[mid * size + j] = 1;
    }

    // fill vertical line
    for i in 0..size {
        data[i * size + mid] = 1;
    }

    Ok(Kernel {
        data,
        width: size,
        height: size,
    })
}

/// Create a cross structuring element.
///
/// # Arguments
///
/// * `size` - The size of the kernel (size x size).
///
/// # Returns
///
/// A [`Kernel`] with 1s along the horizontal and vertical center lines.
///
/// # Panics
///
/// Panics if `size == 0` or multiplication overflows. Use [`try_cross_kernel`] for fallible creation.
pub fn cross_kernel(size: usize) -> Kernel {
    try_cross_kernel(size).expect("Invalid kernel length")
}

/// Fallibly create an ellipse structuring element.
///
/// # Arguments
///
/// * `width` - The width of the ellipse.
/// * `height` - The height of the ellipse.
///
/// # Returns
///
/// A [`Kernel`] with 1s inside the ellipse boundary, or [`ImageError::InvalidKernelLength`]
/// if width or height is 0 or multiplication overflows.
pub fn try_ellipse_kernel(width: usize, height: usize) -> Result<Kernel, ImageError> {
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidKernelLength(width, height));
    }
    let len = width
        .checked_mul(height)
        .ok_or(ImageError::InvalidKernelLength(width, height))?;

    let mut data = vec![0u8; len];

    // Edge case: 1x1 ellipse acts as identity with a single active element
    if width == 1 && height == 1 {
        data[0] = 1;
        return Ok(Kernel {
            data,
            width,
            height,
        });
    }

    let cx = width as f32 / 2.0;
    let cy = height as f32 / 2.0;
    let rx = width as f32 / 2.0;
    let ry = height as f32 / 2.0;

    for i in 0..height {
        for j in 0..width {
            let x = j as f32 - cx;
            let y = i as f32 - cy;
            // Ellipse equation: (x^2 / rx^2) + (y^2 / ry^2) <= 1
            if (x * x) / (rx * rx) + (y * y) / (ry * ry) <= 1.0 {
                data[i * width + j] = 1;
            }
        }
    }

    // Ensure kernel has at least one active tap
    if !data.contains(&1) {
        let mid_y = height / 2;
        let mid_x = width / 2;
        data[mid_y * width + mid_x] = 1;
    }

    Ok(Kernel {
        data,
        width,
        height,
    })
}

/// Create an ellipse structuring element.
///
/// # Arguments
///
/// * `width` - The width of the ellipse.
/// * `height` - The height of the ellipse.
///
/// # Returns
///
/// A [`Kernel`] with 1s inside the ellipse boundary.
///
/// # Panics
///
/// Panics if width or height is 0 or multiplication overflows. Use [`try_ellipse_kernel`] for fallible creation.
pub fn ellipse_kernel(width: usize, height: usize) -> Kernel {
    try_ellipse_kernel(width, height).expect("Invalid kernel length")
}
