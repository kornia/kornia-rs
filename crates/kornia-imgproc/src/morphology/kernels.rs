use kornia_image::ImageError;

/// Shapes of morphological `Kernels`.
///
/// Defines the geometry of the kernel used in morphological operations.
/// All kernels are centered at their geometric center.
#[derive(Debug, Clone)]
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
/// let kernel = Kernel::try_new(KernelShape::Box { size: 3 }).unwrap();
/// assert_eq!(kernel.width(), 3);
/// assert_eq!(kernel.height(), 3);
/// assert_eq!(kernel.pad(), (1, 1));
/// ```
pub struct Kernel {
    data: Vec<u8>,
    width: usize,
    height: usize,
}

impl Kernel {
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
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidKernelShape`] if the kernel dimensions are
    /// zero, even-sized, or if the generated kernel does not contain any active
    /// elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_imgproc::morphology::{Kernel, KernelShape};
    ///
    /// let kernel = Kernel::try_new(KernelShape::Cross { size: 3 }).unwrap();
    /// assert_eq!(kernel.pad(), (1, 1));
    /// ```
    pub fn try_new(shape: KernelShape) -> Result<Self, ImageError> {
        match shape {
            KernelShape::Box { size } => try_box_kernel(size),
            KernelShape::Cross { size } => try_cross_kernel(size),
            KernelShape::Ellipse { width, height } => try_ellipse_kernel(width, height),
        }
    }

    /// Get a reference to the kernel data.
    ///
    /// # Returns
    ///
    /// The flattened binary mask of the structuring element.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_imgproc::morphology::{Kernel, KernelShape};
    ///
    /// let kernel = Kernel::try_new(KernelShape::Box { size: 3 }).unwrap();
    /// assert_eq!(kernel.data().len(), 9);
    /// ```
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Get the width of the kernel.
    ///
    /// # Returns
    ///
    /// The kernel width in pixels.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_imgproc::morphology::{Kernel, KernelShape};
    ///
    /// let kernel = Kernel::try_new(KernelShape::Box { size: 5 }).unwrap();
    /// assert_eq!(kernel.width(), 5);
    /// ```
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the height of the kernel.
    ///
    /// # Returns
    ///
    /// The kernel height in pixels.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_imgproc::morphology::{Kernel, KernelShape};
    ///
    /// let kernel = Kernel::try_new(KernelShape::Ellipse { width: 3, height: 5 }).unwrap();
    /// assert_eq!(kernel.height(), 5);
    /// ```
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the padding for the kernel (offset from center).
    ///
    /// # Returns
    ///
    /// The symmetric `(pad_h, pad_w)` padding required around the source image.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_imgproc::morphology::{Kernel, KernelShape};
    ///
    /// let kernel = Kernel::try_new(KernelShape::Box { size: 7 }).unwrap();
    /// assert_eq!(kernel.pad(), (3, 3));
    /// ```
    pub fn pad(&self) -> (usize, usize) {
        (self.height / 2, self.width / 2)
    }

    /// Validate that the kernel is well-formed for morphological operations.
    ///
    /// # Returns
    ///
    /// `Ok(())` if the kernel can be used by morphology operators.
    ///
    /// # Errors
    ///
    /// Returns [`ImageError::InvalidKernelShape`] if the kernel has zero-sized
    /// dimensions, even-sized dimensions, inconsistent storage, or no active
    /// elements.
    ///
    /// # Example
    ///
    /// ```rust
    /// use kornia_imgproc::morphology::{Kernel, KernelShape};
    ///
    /// let kernel = Kernel::try_new(KernelShape::Box { size: 3 }).unwrap();
    /// kernel.validate().unwrap();
    /// ```
    pub fn validate(&self) -> Result<(), ImageError> {
        validate_kernel_dimensions(self.width, self.height)?;

        if self.data.len() != self.width * self.height {
            return Err(ImageError::InvalidKernelShape(
                "kernel storage does not match its dimensions".to_string(),
            ));
        }

        if !self.data.iter().any(|&value| value != 0) {
            return Err(ImageError::InvalidKernelShape(
                "kernel must contain at least one active element".to_string(),
            ));
        }

        Ok(())
    }
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
/// # Errors
///
/// Returns [`ImageError::InvalidKernelShape`] if `size` is zero or even.
///
/// # Example
///
/// ```rust
/// use kornia_imgproc::morphology::try_box_kernel;
///
/// let kernel = try_box_kernel(3).unwrap();
/// assert_eq!(kernel.data(), &[1, 1, 1, 1, 1, 1, 1, 1, 1]);
/// ```
pub fn try_box_kernel(size: usize) -> Result<Kernel, ImageError> {
    validate_kernel_dimensions(size, size)?;
    let data = vec![1u8; size * size];
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
/// # Errors
///
/// Returns [`ImageError::InvalidKernelShape`] if `size` is zero or even.
///
/// # Example
///
/// ```rust
/// use kornia_imgproc::morphology::try_cross_kernel;
///
/// let kernel = try_cross_kernel(3).unwrap();
/// assert_eq!(kernel.data(), &[0, 1, 0, 1, 1, 1, 0, 1, 0]);
/// ```
pub fn try_cross_kernel(size: usize) -> Result<Kernel, ImageError> {
    validate_kernel_dimensions(size, size)?;
    let mut data = vec![0u8; size * size];
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
/// # Errors
///
/// Returns [`ImageError::InvalidKernelShape`] if either dimension is zero,
/// even-sized, or if the generated ellipse contains no active elements.
///
/// # Example
///
/// ```rust
/// use kornia_imgproc::morphology::try_ellipse_kernel;
///
/// let kernel = try_ellipse_kernel(5, 5).unwrap();
/// assert_eq!(kernel.width(), 5);
/// assert_eq!(kernel.height(), 5);
/// ```
pub fn try_ellipse_kernel(width: usize, height: usize) -> Result<Kernel, ImageError> {
    validate_kernel_dimensions(width, height)?;
    let mut data = vec![0u8; width * height];
    let cx = (width / 2) as f32;
    let cy = (height / 2) as f32;
    let rx = cx.max(1.0);
    let ry = cy.max(1.0);

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

    let kernel = Kernel {
        data,
        width,
        height,
    };
    kernel.validate()?;
    Ok(kernel)
}

fn validate_kernel_dimensions(width: usize, height: usize) -> Result<(), ImageError> {
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidKernelShape(
            "kernel dimensions must be greater than zero".to_string(),
        ));
    }

    if width % 2 == 0 || height % 2 == 0 {
        return Err(ImageError::InvalidKernelShape(
            "kernel dimensions must be odd so the anchor is centered".to_string(),
        ));
    }

    Ok(())
}
