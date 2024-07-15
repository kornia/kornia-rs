/// An error type for the io module.
#[derive(thiserror::Error, Debug)]
pub enum IoError {
    /// Error when the file does not exist.
    #[error("File does not exist: {0}")]
    FileDoesNotExist(String),

    /// Invalid file extension.
    #[error("File is does not have a valid extension: {0}")]
    InvalidFileExtension(String),

    /// Error to open the file.
    #[error("Failed to manipulate the file")]
    FileError(#[from] std::io::Error),

    /// Error to map the file to memory.
    #[error("Error with Jpeg encoding/decoding")]
    JpegError(#[from] crate::jpeg::JpegError),

    /// Error to create the image.
    #[error("Failed to create image")]
    ImageCreationError(#[from] kornia_image::ImageError),
}
