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
    #[error("Failed to manipulate the file")]
    FileError(#[from] std::io::Error),

    /// Error to map the file to memory.
    #[cfg(feature = "turbojpeg")]
    #[error("Error with Jpeg encoding/decoding")]
    JpegTurboError(#[from] crate::jpegturbo::JpegTurboError),

    /// Error to decode the JPEG image.
    #[error("Error with Jpeg decoding")]
    JpegDecodingError(#[from] jpeg_decoder::Error),

    /// Error to encode the JPEG image.
    #[error("Error with Jpeg encoding")]
    JpegEncodingError(#[from] jpeg_encoder::EncodingError),

    /// Error to create the image.
    #[error("Failed to create image")]
    ImageCreationError(#[from] kornia_image::ImageError),

    /// Error to decode the image.
    #[error("Failed to decode the image")]
    ImageDecodeError(#[from] image::ImageError),

    /// Error to encode the PNG image.
    #[error("Failed to encode the png image")]
    PngEncodingError(String),

    /// Error to decode the PNG image.
    #[error("Failed to decode the png image")]
    PngDecodeError(String),
}
