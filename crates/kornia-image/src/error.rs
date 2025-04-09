/// An error typr for the io module.
#[derive(thiserror::Error, Debug)]
pub enum ImageError {
    /// Error when the pixels type are not compatible.
    #[error("Incompatible pixel types")]
    IncompatiblePixelTypes,

    /// Error when the image size is not initialized.
    #[error("Image data is not initialized")]
    ImageDataNotInitialized,

    /// Error when the image data is not contiguous.
    #[error("Image data is not contiguous")]
    ImageDataNotContiguous,

    /// Error when the image shape is not valid.
    #[error(transparent)]
    InvalidImageShape(#[from] kornia_tensor::TensorError),

    /// Error when the image size is not valid.
    #[error("Invalid image size ({0}, {1}) mismatch ({2}, {3})")]
    InvalidImageSize(usize, usize, usize, usize),

    /// Error when channel and shape are not valid.
    #[error("Data length ({0}) does not match the image size ({1})")]
    InvalidChannelShape(usize, usize),

    /// Error when the cast operation fails.
    #[error("Failed to cast image data")]
    CastError,

    /// Error when the channel index is out of bounds.
    #[error("Channel index {0} is out of bounds {1}")]
    ChannelIndexOutOfBounds(usize, usize),

    /// Error when pixel index is out of bounds.
    #[error("Pixel coordinate ({0}, {1}) is out of bounds ({2}, {3})")]
    PixelIndexOutOfBounds(usize, usize, usize, usize),

    /// Error when the number of bins is invalid.
    #[error("Invalid number of bins {0}")]
    InvalidHistogramBins(usize),

    /// Error when the cannot compute the determinant.
    #[error("Cannot compute the determinant: matrix is singular")]
    CannotComputeDeterminant,

    /// Error when the kernel length is invalid.
    #[error("Invalid kernel length {0} and {1}")]
    InvalidKernelLength(usize, usize),
}
