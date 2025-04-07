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
    #[error("Failed to manipulate the file. {0}")]
    FileError(#[from] std::io::Error),

    /// Error to map the file to memory.
    #[cfg(feature = "turbojpeg")]
    #[error("Error with Jpeg encoding/decoding. {0}")]
    JpegTurboError(#[from] crate::jpegturbo::JpegTurboError),

    /// Error to decode the JPEG image.
    #[error("Error with Jpeg decoding. {0}")]
    JpegDecodingError(#[from] zune_jpeg::errors::DecodeErrors),

    /// Error to encode the JPEG image.
    #[error("Error with Jpeg encoding. {0}")]
    JpegEncodingError(#[from] jpeg_encoder::EncodingError),

    /// Error to create the image.
    #[error("Failed to create image. {0}")]
    ImageCreationError(#[from] kornia_image::ImageError),

    /// Error to decode the image.
    #[error("Failed to decode the image. {0}")]
    ImageDecodeError(#[from] image::ImageError),

    /// Error to encode the PNG image.
    #[error("Failed to encode the png image. {0}")]
    PngEncodingError(String),

    /// Error to decode the PNG image.
    #[error("Failed to decode the png image. {0}")]
    PngDecodeError(String),
}
