/// Errors that can occur when working with AprilTag detection.
#[derive(Debug, thiserror::Error)]
pub enum AprilTagError {
    /// Error related to image.
    #[error(transparent)]
    ImageError(#[from] kornia_image::error::ImageError),

    /// The source image dimensions do not match the [TileMinMax](crate::threshold::TileMinMax) dimensions.
    #[error("The source image dimensions do not match the TileMinMax dimensions")]
    ImageTileSizeMismatch,

    /// The minimum image size should be tile_size x tile_size
    #[error("The minimum image size should be tile_size x tile_size")]
    InvalidImageSize,
}
