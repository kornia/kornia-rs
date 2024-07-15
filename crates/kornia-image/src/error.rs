/// An error typr for the io module.
#[derive(thiserror::Error, Debug)]
pub enum ImageError {
    /// Error when the image data is not contiguous.
    #[error("Image data is not contiguous")]
    ImageDataNotContiguous,

    /// Error when shape is not valid.
    #[error("Invalid shape")]
    InvalidShape(#[from] ndarray::ShapeError),

    /// Error when channel and shape are not valid.
    #[error("Data length ({0}) does not match the image size ({1})")]
    InvalidChannelShape(usize, usize),
}
