/// An error type for the io module.
#[derive(thiserror::Error, Debug)]
pub enum IoError {
    /// Error when the file does not exist.
    #[error("File does not exist: {0}")]
    FileDoesNotExist(std::path::PathBuf),

    /// Invalid file extension.
    #[error("File is does not have a valid extension: {0}")]
    InvalidFileExtension(std::path::PathBuf),

    /// Error to open the file.
    #[error(transparent)]
    FileError(#[from] std::io::Error),

    /// Error to validate image resolution.
    #[error(
        "The Image Resolution didn't matched. Expected H: {0}, W: {1}, but found H: {2}, W: {3}"
    )]
    DecodeMismatchResolution(usize, usize, usize, usize),

    /// Error when invalid buffer size slice is provided.
    #[error("Expected buffer size to be {1}, but found {0}")]
    InvalidBufferSize(usize, usize),

    /// Error to map the file to memory.
    #[cfg(feature = "turbojpeg")]
    #[error(transparent)]
    JpegTurboError(#[from] crate::jpegturbo::JpegTurboError),

    /// Error to decode the JPEG image.
    #[error(transparent)]
    JpegDecodingError(#[from] zune_jpeg::errors::DecodeErrors),

    /// Error to encode the JPEG image.
    #[error(transparent)]
    JpegEncodingError(#[from] jpeg_encoder::EncodingError),

    /// Error to create the image.
    #[error(transparent)]
    ImageCreationError(#[from] kornia_image::ImageError),

    /// Error to decode the image.
    #[error(transparent)]
    ImageDecodeError(#[from] image::ImageError),

    /// Error to encode the PNG image.
    #[error("Failed to encode the png image. {0}")]
    PngEncodingError(String),

    /// Error to decode the PNG image.
    #[error("Failed to decode the png image. {0}")]
    PngDecodeError(String),

    /// Error to decode the TIFF image.
    #[error(transparent)]
    TiffDecodingError(#[from] tiff::TiffError),
}
