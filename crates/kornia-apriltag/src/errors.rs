/// Errors that can occur when working with AprilTag detection.
#[derive(Debug, thiserror::Error)]
pub enum AprilTagError {
    /// Error related to image.
    #[error(transparent)]
    ImageError(#[from] kornia_image::error::ImageError),
    /// The length of the TileBuffer was not as expected.
    #[error("Expected the length of TileBuffer to be {0} but found {1}")]
    InvalidTileBufferSize(usize, usize),
}
